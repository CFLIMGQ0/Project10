"""Numerical and integration tests for the Word-aligned PC-CMKA method."""

from __future__ import annotations

from copy import deepcopy
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from datasets.dataset_survival import fit_training_fold_survival_bins
from models.layers.pc_cmka import (
    ChebyshevInverseCalibrator,
    GraphViewAugmenter,
    PCCMKADDKACEncoder,
    ReferenceSpectralOperator,
)
from utils.pc_cmka import (
    _external_prior_graphs,
    load_pc_cmka_config,
)
from scripts.run_pc_cmka_word_ablations import GRAPH_STRUCTURE_PRESETS


CONFIG = "configs/pc_cmka_ddkac_word.json"


def chain(size: int) -> torch.Tensor:
    adjacency = torch.zeros(size, size)
    index = torch.arange(size - 1)
    adjacency[index, index + 1] = 1.0
    adjacency[index + 1, index] = 1.0
    return adjacency


def small_operator(size: int = 6) -> ReferenceSpectralOperator:
    return ReferenceSpectralOperator.from_adjacency(
        chain(size), spectral_bound=4.0, max_log_deformation=0.5
    )


class ReferenceGeometryTests(unittest.TestCase):
    def test_sparse_dense_operator_consistency(self) -> None:
        operator = small_operator()
        signal = torch.randn(3, 6)
        weights = operator.prior_weights * torch.linspace(0.8, 1.2, 5)
        sparse = operator.apply_operator(weights, signal)
        dense = signal @ operator.dense_matrix(weights).t()
        torch.testing.assert_close(sparse, dense, atol=1e-6, rtol=1e-6)

    def test_chebyshev_recurrence_matches_explicit_polynomial(self) -> None:
        operator = small_operator()
        signal = torch.randn(2, 6)
        weights = operator.prior_weights
        response = operator.chebyshev_responses(weights, signal, 2)[2]
        matrix = operator.dense_matrix(weights, scaled=True)
        explicit = signal @ (2.0 * matrix @ matrix - torch.eye(6)).t()
        torch.testing.assert_close(response, explicit, atol=1e-6, rtol=1e-6)

    def test_zero_coefficients_are_exact_prior(self) -> None:
        config = load_pc_cmka_config(CONFIG, "A8_no_identifiability")
        calibrator = ChebyshevInverseCalibrator(6, small_operator(), config)
        coefficients = torch.zeros(2, calibrator.dictionary.rank)
        weights, deformation = calibrator.weights_from_coefficients(coefficients)
        torch.testing.assert_close(
            weights, calibrator.operator.prior_weights.unsqueeze(0).expand_as(weights)
        )
        self.assertEqual(float(deformation.detach().abs().max()), 0.0)

    def test_patient_normalization_is_a_control_only(self) -> None:
        config = load_pc_cmka_config(CONFIG, "C1_patient_degree_edge_gate")
        self.assertEqual(config["spectral"]["normalization"], "patient")
        full = load_pc_cmka_config(CONFIG, "A0_full")
        self.assertEqual(full["spectral"]["normalization"], "reference")


class CalibrationAugmentationTests(unittest.TestCase):
    def _calibration(self, experiment: str = "A8_no_identifiability"):
        config = load_pc_cmka_config(CONFIG, experiment)
        operator = small_operator()
        calibrator = ChebyshevInverseCalibrator(6, operator, config)
        inputs = torch.randn(1, 6)
        probes = torch.nn.functional.normalize(inputs, dim=1)
        return config, operator, calibrator, inputs, probes

    def test_zero_target_offset_returns_prior(self) -> None:
        _, operator, calibrator, inputs, probes = self._calibration()
        torch.nn.init.zeros_(calibrator.target_network.offset.weight)
        torch.nn.init.zeros_(calibrator.target_network.offset.bias)
        result = calibrator(inputs, probes)
        torch.testing.assert_close(
            result.weights,
            operator.prior_weights.unsqueeze(0),
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertLessEqual(float(result.rho.detach().max()), 1e-6)

    def test_antithetic_views_are_centered_positive_and_krylov_safe(self) -> None:
        config, _, calibrator, inputs, probes = self._calibration()
        result = calibrator(inputs, probes)
        augmenter = GraphViewAugmenter(calibrator.operator, calibrator, config)
        augmenter.train()
        views = augmenter(result, probes)
        torch.testing.assert_close(
            0.5 * (views.positive + views.negative),
            result.weights,
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertTrue(bool((views.positive > 0).all()))
        self.assertTrue(bool((views.negative > 0).all()))
        self.assertLessEqual(
            float(views.krylov_after.detach().max()),
            float(config["augmentation"]["krylov_eta"]) + 1e-5,
        )

    def test_exact_diagonal_and_low_rank_hessian_modes(self) -> None:
        for mode in ("exact", "diagonal", "low_rank"):
            with self.subTest(mode=mode):
                config, _, calibrator, inputs, probes = self._calibration()
                config = deepcopy(config)
                config["augmentation"]["hessian_mode"] = mode
                augmenter = GraphViewAugmenter(calibrator.operator, calibrator, config)
                result = calibrator(inputs, probes)
                views = augmenter(result, probes)
                self.assertTrue(bool(torch.isfinite(views.positive).all()))

    def test_eval_does_not_execute_random_augmentation(self) -> None:
        config, _, calibrator, inputs, probes = self._calibration()
        augmenter = GraphViewAugmenter(calibrator.operator, calibrator, config)
        augmenter.eval()
        result = calibrator(inputs, probes)
        first = augmenter(result, probes)
        second = augmenter(result, probes)
        torch.testing.assert_close(first.positive, result.weights)
        torch.testing.assert_close(first.positive, second.positive)


class EncoderAndGradientTests(unittest.TestCase):
    dims = (5, 6, 7, 8, 9, 10)

    def _encoder(self, experiment: str) -> PCCMKADDKACEncoder:
        config = load_pc_cmka_config(CONFIG, experiment)
        return PCCMKADDKACEncoder(
            self.dims, [chain(size) for size in self.dims], config, output_dim=4
        )

    def _inputs(self) -> list[torch.Tensor]:
        return [torch.randn(size) for size in self.dims]

    def test_six_groups_return_six_tokens(self) -> None:
        model = self._encoder("A8_no_identifiability")
        output = model(self._inputs())
        self.assertEqual(tuple(output.shape), (1, 6, 4))
        self.assertEqual(len(model.pathways), 6)

    def test_survival_gradient_reaches_target_solver_and_dictionary(self) -> None:
        model = self._encoder("A8_no_identifiability")
        output = model(self._inputs())
        # Use only a survival-head proxy, not an auxiliary calibration loss,
        # so this verifies the complete prediction -> graph -> unrolled-solver
        # gradient path.
        output.square().mean().backward()
        target_gradient = model.pathways[0].calibrator.target_network.offset.weight.grad
        dictionary_gradient = model.pathways[0].calibrator.dictionary.basis.grad
        self.assertIsNotNone(target_gradient)
        self.assertIsNotNone(dictionary_gradient)
        self.assertGreater(float(target_gradient.abs().sum()), 0.0)
        self.assertGreater(float(dictionary_gradient.abs().sum()), 0.0)

    def test_shared_frequency_routes_are_identical_across_views(self) -> None:
        model = self._encoder("A0_full")
        model.train()
        model(self._inputs())
        diagnostics = model.pathways[0].diagnostics
        torch.testing.assert_close(
            diagnostics["route_positive"], diagnostics["route_negative"]
        )
        torch.testing.assert_close(
            diagnostics["route_positive"], diagnostics["route_base"]
        )

    def test_seed_repeats_and_changes_stochastic_views(self) -> None:
        model = self._encoder("A8_no_identifiability")
        model.train()
        inputs = self._inputs()
        torch.manual_seed(13)
        model(inputs)
        first = model.pathways[0].diagnostics["positive_edge_delta"].clone()
        torch.manual_seed(13)
        model(inputs)
        repeated = model.pathways[0].diagnostics["positive_edge_delta"].clone()
        torch.manual_seed(14)
        model(inputs)
        changed = model.pathways[0].diagnostics["positive_edge_delta"].clone()
        torch.testing.assert_close(first, repeated)
        self.assertFalse(torch.equal(first, changed))

    def test_word_table_five_experiment_semantics(self) -> None:
        expected = {
            "A0_full",
            "A1_direct_coefficients",
            "A2_fixed_rho",
            "A3_random_probe",
            "A4_isotropic_uncertainty",
            "A5_independent_views",
            "A6_no_krylov",
            "A7_separate_routes",
            "A8_no_identifiability",
            "A9_structure_only",
            "A9_fusion_only",
        }
        for name in expected:
            self.assertEqual(load_pc_cmka_config(CONFIG, name)["experiment_name"], name)

    def test_prompt_staged_experiment_semantics(self) -> None:
        expected = [
            "S_A0_original_ddkac",
            *[f"S_A{index}_{suffix}" for index, suffix in (
                (1, "reference_operator"),
                (2, "direct_edge_gate"),
                (3, "inverse_calibration"),
                (4, "random_edge_drop"),
                (5, "independent_views"),
                (6, "hessian_antithetic"),
                (7, "krylov_safe"),
                (8, "shared_route"),
                (9, "structure_consistency"),
                (10, "identifiability"),
            )],
            "S_Full_pc_cmka_ddkac",
        ]
        for name in expected:
            self.assertEqual(load_pc_cmka_config(CONFIG, name)["experiment_name"], name)


class LeakageAndPriorTests(unittest.TestCase):
    def test_survival_bins_ignore_validation_statistics(self) -> None:
        frame = pd.DataFrame(
            {
                "case_id": [f"T{i}" for i in range(8)] + ["V0", "V1"],
                "time": list(range(1, 9)) + [100.0, 200.0],
                "censor": [0] * 10,
            }
        )
        train = [f"T{i}" for i in range(8)]
        first, _ = fit_training_fold_survival_bins(
            frame, train, "time", "censor", 4
        )
        frame.loc[frame.case_id.str.startswith("V"), "time"] = [10000.0, 20000.0]
        second, _ = fit_training_fold_survival_bins(
            frame, train, "time", "censor", 4
        )
        np.testing.assert_array_equal(first, second)

    def test_external_prior_uses_gene_names_not_csv_order(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".csv") as handle:
            handle.write("source,target,weight\nB,D,0.7\nA,C,0.9\n")
            handle.flush()
            graph = _external_prior_graphs(
                [["D", "C", "B", "A"]],
                __import__("pathlib").Path(handle.name),
                0.01,
            )[0]
        self.assertAlmostEqual(float(graph[2, 0]), 0.7)
        self.assertAlmostEqual(float(graph[3, 1]), 0.9)


class GraphStructurePresetTests(unittest.TestCase):
    def test_all_fifteen_model_yaml_choices_resolve(self) -> None:
        self.assertEqual(len(GRAPH_STRUCTURE_PRESETS), 15)
        self.assertEqual(len(set(GRAPH_STRUCTURE_PRESETS.values())), 15)
        for graph_name, experiment_name in GRAPH_STRUCTURE_PRESETS.items():
            with self.subTest(graph_structure=graph_name):
                resolved = load_pc_cmka_config(CONFIG, experiment_name)
                self.assertEqual(resolved["experiment_name"], experiment_name)


if __name__ == "__main__":
    unittest.main()
