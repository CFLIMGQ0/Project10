import unittest
import json
import tempfile
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

from models.layers.pc_cmka_ddkac import (
    CalibrationUncertaintyAugmentor,
    ControlGraphAugmentor,
    DirectPatientEdgeGate,
    ChebyshevMomentResponse,
    DifferentiableMomentSolver,
    EdgeDeformationDictionary,
    IdentifiabilityTangentRegularizer,
    MomentTargetNetwork,
    PCCMKADDKACEncoder,
    SharedRouteDDKACPathway,
    ReferenceSpectralOperator,
    chebyshev_recurrence,
    normalized_patient_probe,
    build_krylov_basis,
    krylov_operator_error,
    negative_free_consistency_loss,
)
from utils.pc_cmka_graph import build_training_correlation_prior
from utils.pc_cmka_diagnostics import PCCMKADiagnosticRecorder
from utils.fold_survival_bins import FoldSurvivalBinner
from utils.pc_cmka_config import encoder_kwargs, load_pc_cmka_config
from models.model_HGNN import MRePath
from utils.core_utils import _init_optim
from types import SimpleNamespace


class ReferenceSpectralOperatorTests(unittest.TestCase):
    def setUp(self):
        self.edges = torch.tensor(
            [[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]], dtype=torch.long
        )
        self.prior = torch.tensor([1.0, 0.8, 1.2, 0.7, 0.5])
        self.operator = ReferenceSpectralOperator(
            self.edges,
            self.prior,
            num_nodes=4,
            max_log_deformation=0.4,
        )

    def test_reference_degree_is_fixed_and_correct(self):
        expected = torch.tensor([2.2, 1.8, 2.5, 1.9])
        self.assertTrue(
            torch.allclose(self.operator.reference_degree, expected)
        )
        patient_weights = self.operator.patient_weights(torch.ones(5))
        self.operator.matvec(torch.randn(4), patient_weights)
        self.assertTrue(
            torch.allclose(self.operator.reference_degree, expected)
        )

    def test_sparse_mvp_matches_dense_for_vectors_and_features(self):
        weights = self.operator.patient_weights(
            torch.tensor([0.2, -0.1, 0.3, -0.2, 0.1])
        )
        dense = self.operator.dense_matrix(weights)
        vector = torch.randn(4)
        features = torch.randn(4, 3)
        self.assertTrue(
            torch.allclose(
                self.operator.matvec(vector, weights), dense @ vector, atol=1e-6
            )
        )
        self.assertTrue(
            torch.allclose(
                self.operator.matvec(features, weights), dense @ features, atol=1e-6
            )
        )

    def test_batched_sparse_mvp_matches_dense(self):
        log_deformation = torch.tensor(
            [[0.1, -0.2, 0.3, 0.0, -0.1], [-0.2, 0.2, 0.0, 0.1, 0.3]]
        )
        weights = self.operator.patient_weights(log_deformation)
        vectors = torch.randn(2, 4)
        features = torch.randn(2, 4, 3)
        dense = self.operator.dense_matrix(weights)
        expected_vectors = torch.einsum("bij,bj->bi", dense, vectors)
        expected_features = torch.einsum("bij,bjk->bik", dense, features)
        self.assertTrue(
            torch.allclose(
                self.operator.matvec(vectors, weights), expected_vectors, atol=1e-6
            )
        )
        self.assertTrue(
            torch.allclose(
                self.operator.matvec(features, weights), expected_features, atol=1e-6
            )
        )

    def test_scaled_operator_matches_formula(self):
        values = torch.randn(4, 2)
        expected = (
            2.0 / self.operator.lambda_max
        ) * self.operator.dense_matrix() @ values - values
        self.assertTrue(
            torch.allclose(self.operator.apply_scaled(values), expected, atol=1e-6)
        )

    def test_patient_weights_are_positive_finite_and_bounded(self):
        log_deformation = torch.tensor(
            [-100.0, -1.0, 0.0, 1.0, 100.0], requires_grad=True
        )
        weights = self.operator.patient_weights(log_deformation)
        ratio = weights / self.prior
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue((weights > 0).all())
        self.assertGreaterEqual(float(ratio.detach().min()), torch.exp(torch.tensor(-0.4)).item() - 1e-6)
        self.assertLessEqual(float(ratio.detach().max()), torch.exp(torch.tensor(0.4)).item() + 1e-6)
        self.operator.matvec(torch.randn(4), weights).sum().backward()
        self.assertIsNotNone(log_deformation.grad)

    def test_reference_radius_respects_normalized_laplacian_bound(self):
        estimate = self.operator.estimate_reference_radius(iterations=80)
        exact = torch.linalg.eigvalsh(self.operator.dense_matrix()).max()
        self.assertTrue(torch.allclose(estimate, exact, atol=1e-4))
        self.assertLessEqual(float(exact), 2.0 + 1e-6)
        self.assertGreaterEqual(
            float(self.operator.lambda_max),
            ReferenceSpectralOperator.theoretical_upper_bound(0.4) - 1e-6,
        )

    def test_invalid_support_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "self-loops"):
            ReferenceSpectralOperator(
                torch.tensor([[0, 1], [0, 2]]), torch.ones(2), 3
            )
        with self.assertRaisesRegex(ValueError, "isolated"):
            ReferenceSpectralOperator(
                torch.tensor([[0], [1]]), torch.ones(1), 3
            )


class FoldPriorGraphTests(unittest.TestCase):
    def test_training_correlation_prior_is_deterministic_and_well_formed(self):
        rng = np.random.default_rng(5)
        values = rng.normal(size=(20, 7)).astype(np.float32)
        genes = [f"gene_{index}" for index in range(7)]
        first = build_training_correlation_prior(values, genes, neighbours=2)
        second = build_training_correlation_prior(values, genes, neighbours=2)
        self.assertTrue(torch.equal(first.edge_index, second.edge_index))
        self.assertTrue(torch.equal(first.prior_weights, second.prior_weights))
        self.assertTrue((first.edge_index[0] < first.edge_index[1]).all())
        self.assertEqual(first.source, "training_correlation")
        degree = torch.bincount(first.edge_index.flatten(), minlength=7)
        self.assertTrue((degree > 0).all())

    def test_gene_order_is_explicit_not_csv_adjacency(self):
        rng = np.random.default_rng(9)
        values = rng.normal(size=(18, 6)).astype(np.float32)
        genes = [f"g{index}" for index in range(6)]
        permutation = np.array([3, 0, 5, 1, 4, 2])
        original = build_training_correlation_prior(values, genes, neighbours=2)
        permuted = build_training_correlation_prior(
            values[:, permutation],
            [genes[index] for index in permutation],
            neighbours=2,
        )
        original_edges = {
            frozenset((genes[int(a)], genes[int(b)]))
            for a, b in original.edge_index.t().tolist()
        }
        permuted_edges = {
            frozenset((permuted.gene_names[int(a)], permuted.gene_names[int(b)]))
            for a, b in permuted.edge_index.t().tolist()
        }
        self.assertEqual(original_edges, permuted_edges)


class FoldSurvivalBinnerTests(unittest.TestCase):
    def test_validation_outcomes_cannot_change_training_boundaries(self):
        training = pd.DataFrame(
            {
                "time": np.arange(1, 17, dtype=float),
                "censor": [0] * 16,
            }
        )
        first = FoldSurvivalBinner.fit(
            training,
            label_col="time",
            censorship_col="censor",
            n_bins=4,
        )
        validation_a = pd.DataFrame(
            {"time": [-100.0, 1000.0], "censor": [0, 1]}
        )
        validation_b = pd.DataFrame(
            {"time": [-1e9, 1e9], "censor": [1, 0]}
        )
        # Neither validation table is passed to fit; transformed extremes are
        # accepted by the fixed infinite end caps.
        first_a = first.transform(validation_a)
        first_b = first.transform(validation_b)
        second = FoldSurvivalBinner.fit(
            training,
            label_col="time",
            censorship_col="censor",
            n_bins=4,
        )
        self.assertTrue(np.array_equal(first.boundaries, second.boundaries))
        self.assertEqual(first_a["disc_label"].tolist(), [0, 3])
        self.assertEqual(first_b["disc_label"].tolist(), [0, 3])

    def test_censored_training_times_do_not_fit_inner_quantiles(self):
        base = pd.DataFrame(
            {"time": np.arange(1, 17, dtype=float), "censor": [0] * 16}
        )
        with_censored_extreme = pd.concat(
            [
                base,
                pd.DataFrame({"time": [1e12], "censor": [1]}),
            ],
            ignore_index=True,
        )
        first = FoldSurvivalBinner.fit(
            base, label_col="time", censorship_col="censor", n_bins=4
        )
        second = FoldSurvivalBinner.fit(
            with_censored_extreme,
            label_col="time",
            censorship_col="censor",
            n_bins=4,
        )
        self.assertTrue(np.array_equal(first.boundaries, second.boundaries))


class ChebyshevMomentResponseTests(unittest.TestCase):
    def setUp(self):
        edges = torch.tensor(
            [[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]], dtype=torch.long
        )
        prior = torch.tensor([1.0, 0.8, 1.2, 0.7, 0.5])
        self.operator = ReferenceSpectralOperator(
            edges, prior, 4, lambda_max=2.5, max_log_deformation=0.5
        )

    def test_probe_matches_definition_and_zero_is_finite(self):
        expression = torch.tensor([3.0, 4.0, 9.0, 2.0])
        mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
        probe = normalized_patient_probe(expression, mask, epsilon=1e-8)
        self.assertTrue(
            torch.allclose(probe, torch.tensor([0.6, 0.8, 0.0, 0.0]))
        )
        zero = normalized_patient_probe(torch.zeros(2, 4))
        self.assertTrue(torch.isfinite(zero).all())
        self.assertTrue(torch.equal(zero, torch.zeros_like(zero)))

    def test_recurrence_matches_explicit_dense_calculation(self):
        probe = normalized_patient_probe(torch.tensor([0.4, -0.2, 0.8, 0.1]))
        scaled = (
            2.0 / self.operator.lambda_max
        ) * self.operator.dense_matrix() - torch.eye(4)
        t0 = probe
        t1 = scaled @ t0
        t2 = 2.0 * scaled @ t1 - t0
        t3 = 2.0 * scaled @ t2 - t1
        expected = torch.stack((t0, t1, t2, t3))
        actual = chebyshev_recurrence(
            self.operator, probe, self.operator.prior_weights, order=3
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_moments_exclude_t0_and_match_quadratic_form(self):
        expression = torch.tensor([0.4, -0.2, 0.8, 0.1])
        module = ChebyshevMomentResponse(self.operator, order=3)
        moments, probe, recurrence = module(
            expression, return_recurrence=True
        )
        expected = torch.stack(
            [probe @ recurrence[index] for index in range(1, 4)]
        )
        self.assertEqual(moments.shape, (3,))
        self.assertTrue(torch.allclose(moments, expected, atol=1e-6))

    def test_batched_patient_weights_receive_moment_gradients(self):
        expression = torch.randn(2, 4, requires_grad=True)
        log_deformation = torch.randn(2, 5, requires_grad=True) * 0.1
        log_deformation.retain_grad()
        weights = self.operator.patient_weights(log_deformation)
        module = ChebyshevMomentResponse(self.operator, order=3)
        moments = module(expression, weights)
        self.assertEqual(moments.shape, (2, 3))
        self.assertTrue(torch.isfinite(moments).all())
        moments.square().mean().backward()
        self.assertIsNotNone(expression.grad)
        self.assertIsNotNone(log_deformation.grad)
        self.assertTrue(torch.isfinite(log_deformation.grad).all())

    def test_prior_response_is_repeatable(self):
        module = ChebyshevMomentResponse(self.operator, order=2)
        expression = torch.randn(3, 4)
        first = module(expression)
        second = module(expression, self.operator.prior_weights)
        self.assertTrue(torch.equal(first, second))


class DifferentiableMomentCalibrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        edges = torch.tensor(
            [[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]], dtype=torch.long
        )
        prior = torch.tensor([1.0, 0.8, 1.2, 0.7, 0.5])
        operator = ReferenceSpectralOperator(
            edges, prior, 4, lambda_max=3.0, max_log_deformation=0.5
        )
        self.response = ChebyshevMomentResponse(operator, order=2)
        self.dictionary = EdgeDeformationDictionary(prior, rank=2)
        self.solver = DifferentiableMomentSolver(
            self.response,
            self.dictionary,
            iterations=4,
            step_size=0.08,
            beta=1e-2,
            lambda_rho=1e-3,
            rho_max=0.5,
        )

    def test_dictionary_starts_weighted_orthonormal(self):
        gram = self.dictionary.weighted_gram()
        self.assertTrue(torch.allclose(gram, torch.eye(2), atol=1e-5))
        self.assertLess(float(self.dictionary.orthogonality_loss().detach()), 1e-10)

    def test_zero_offset_recovers_prior_and_zero_trust_radius(self):
        expression = torch.randn(3, 4)
        result = self.solver(
            expression,
            torch.zeros(3, 2),
            torch.ones(3, 2),
            mode="joint",
        )
        self.assertTrue(torch.allclose(result.coefficients, torch.zeros_like(result.coefficients)))
        self.assertTrue(torch.allclose(result.weights, self.response.operator.prior_weights.expand(3, -1)))
        self.assertTrue(torch.allclose(result.actual_moments, result.prior_moments))
        self.assertTrue(torch.equal(result.rho, torch.zeros_like(result.rho)))
        self.assertTrue(bool(result.finite))

    def test_nonzero_target_is_bounded_and_records_history(self):
        expression = torch.randn(2, 4)
        result = self.solver(
            expression,
            torch.tensor([[0.15, -0.1], [-0.1, 0.12]]),
            torch.ones(2, 2),
            mode="detach_solver",
        )
        self.assertEqual(result.objective_history.shape, (5, 2))
        self.assertTrue((result.rho <= 0.5 + 1e-6).all())
        self.assertTrue((result.weights > 0).all())
        self.assertTrue(torch.isfinite(result.residual).all())
        self.assertGreater(float(result.coefficient_norm.detach().sum()), 0.0)

    def test_fixed_graph_mode_does_not_calibrate(self):
        expression = torch.randn(2, 4)
        result = self.solver(
            expression, torch.full((2, 2), 0.2), torch.ones(2, 2), mode="fixed_graph"
        )
        self.assertTrue(torch.equal(result.coefficients, torch.zeros_like(result.coefficients)))
        self.assertTrue(torch.equal(result.actual_moments, result.prior_moments))

    def test_joint_solver_backpropagates_to_target_and_dictionary(self):
        network = MomentTargetNetwork(4, moment_order=2, hidden_dim=8)
        # Move away from the deliberately conservative all-zero target init.
        with torch.no_grad():
            network.offset_head.bias.copy_(torch.tensor([0.15, -0.12]))
        expression = torch.randn(2, 4, requires_grad=True)
        offset, precision = network(expression)
        result = self.solver(expression, offset, precision, mode="joint")
        surrogate_survival_loss = result.actual_moments.square().mean()
        total = surrogate_survival_loss + result.moment_loss + result.dictionary_loss
        total.backward()
        self.assertIsNotNone(network.offset_head.weight.grad)
        self.assertIsNotNone(network.precision_head.weight.grad)
        self.assertIsNotNone(self.dictionary.matrix.grad)
        self.assertIsNotNone(expression.grad)
        for gradient in (
            network.offset_head.weight.grad,
            network.precision_head.weight.grad,
            self.dictionary.matrix.grad,
            expression.grad,
        ):
            self.assertTrue(torch.isfinite(gradient).all())

    def test_all_solver_modes_run(self):
        expression = torch.randn(1, 4)
        for mode in sorted(DifferentiableMomentSolver.VALID_MODES):
            with self.subTest(mode=mode):
                result = self.solver(
                    expression,
                    torch.tensor([[0.1, -0.1]]),
                    torch.ones(1, 2),
                    mode=mode,
                )
                self.assertEqual(result.weights.shape, (1, 5))
                self.assertTrue(bool(result.finite))

    def test_joint_solver_runs_inside_validation_no_grad(self):
        expression = torch.randn(1, 4)
        with torch.no_grad():
            result = self.solver(
                expression,
                torch.tensor([[0.1, -0.1]]),
                torch.ones(1, 2),
                mode="joint",
            )
        self.assertFalse(result.coefficients.requires_grad)
        self.assertTrue(bool(result.finite))


class CalibrationUncertaintyAugmentationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        edges = torch.tensor(
            [[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]], dtype=torch.long
        )
        prior = torch.tensor([1.0, 0.8, 1.2, 0.7, 0.5])
        operator = ReferenceSpectralOperator(
            edges, prior, 4, lambda_max=3.0, max_log_deformation=0.5
        )
        self.response = ChebyshevMomentResponse(operator, order=2)
        self.dictionary = EdgeDeformationDictionary(prior, rank=2)
        self.solver = DifferentiableMomentSolver(
            self.response,
            self.dictionary,
            iterations=3,
            step_size=0.08,
            rho_max=0.5,
        )
        self.expression = torch.randn(2, 4)
        self.calibration = self.solver(
            self.expression,
            torch.tensor([[0.12, -0.08], [-0.1, 0.1]]),
            torch.ones(2, 2),
            mode="detach_solver",
        )

    def test_krylov_basis_is_orthonormal_and_error_is_zero_for_base(self):
        probe = normalized_patient_probe(self.expression[0])
        basis = build_krylov_basis(
            self.response.operator,
            probe,
            self.calibration.weights[0],
            order=2,
        )
        self.assertTrue(
            torch.allclose(basis.t() @ basis, torch.eye(basis.shape[1]), atol=1e-5)
        )
        error = krylov_operator_error(
            self.response.operator,
            self.calibration.weights[0],
            self.calibration.weights[0],
            basis,
        )
        self.assertEqual(float(error.detach()), 0.0)

    def test_all_hessian_modes_are_positive_centered_and_safe(self):
        for mode in sorted(CalibrationUncertaintyAugmentor.VALID_MODES):
            with self.subTest(mode=mode):
                augmentor = CalibrationUncertaintyAugmentor(
                    self.response,
                    self.dictionary,
                    mode=mode,
                    low_rank=1,
                    xi_max=0.25,
                    krylov_eta=0.04,
                )
                generator = torch.Generator().manual_seed(41)
                result = augmentor(
                    self.expression, self.calibration, generator=generator
                )
                center = 0.5 * (
                    result.positive_weights + result.negative_weights
                )
                self.assertTrue(torch.allclose(center, result.base_weights, atol=1e-7))
                self.assertTrue((result.positive_weights > 0).all())
                self.assertTrue((result.negative_weights > 0).all())
                self.assertLessEqual(
                    float(result.relative_perturbation.abs().max()), 0.25 + 1e-6
                )
                self.assertTrue((result.krylov_error <= 0.04 + 1e-5).all())
                self.assertTrue((result.hessian_eigenvalues > 0).all())
                self.assertTrue(bool(result.finite))

    def test_same_generator_seed_repeats_and_different_seed_changes_views(self):
        augmentor = CalibrationUncertaintyAugmentor(
            self.response, self.dictionary, mode="diagonal"
        )
        first = augmentor(
            self.expression,
            self.calibration,
            generator=torch.Generator().manual_seed(101),
        )
        second = augmentor(
            self.expression,
            self.calibration,
            generator=torch.Generator().manual_seed(101),
        )
        third = augmentor(
            self.expression,
            self.calibration,
            generator=torch.Generator().manual_seed(102),
        )
        self.assertTrue(torch.equal(first.relative_perturbation, second.relative_perturbation))
        self.assertFalse(torch.equal(first.relative_perturbation, third.relative_perturbation))

    def test_evaluation_disables_augmentation(self):
        augmentor = CalibrationUncertaintyAugmentor(
            self.response, self.dictionary
        ).eval()
        result = augmentor(self.expression, self.calibration)
        self.assertTrue(torch.equal(result.positive_weights, result.base_weights))
        self.assertTrue(torch.equal(result.negative_weights, result.base_weights))
        self.assertTrue(torch.equal(result.relative_perturbation, torch.zeros_like(result.relative_perturbation)))

    def test_required_random_and_effective_resistance_controls(self):
        for mode in sorted(ControlGraphAugmentor.VALID_MODES):
            with self.subTest(mode=mode):
                control = ControlGraphAugmentor(
                    self.response.operator,
                    mode=mode,
                    drop_probability=0.5,
                )
                result = control(
                    self.calibration.weights,
                    generator=torch.Generator().manual_seed(47),
                )
                self.assertTrue((result.positive_weights > 0).all())
                self.assertTrue((result.negative_weights > 0).all())
                self.assertTrue(bool(result.finite))
                if mode == "random_drop":
                    self.assertTrue(
                        torch.equal(
                            result.negative_weights, self.calibration.weights
                        )
                    )
                else:
                    self.assertFalse(
                        torch.equal(
                            result.positive_weights, result.negative_weights
                        )
                    )


class SharedRouteDDKACTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(53)
        edges = torch.tensor(
            [[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]], dtype=torch.long
        )
        prior = torch.tensor([1.0, 0.8, 1.2, 0.7, 0.5])
        self.operator = ReferenceSpectralOperator(
            edges, prior, 4, lambda_max=3.0, max_log_deformation=0.5
        )
        self.pathway = SharedRouteDDKACPathway(
            4, 8, self.operator, order=2
        )

    def test_shapes_shared_route_and_linear_residual(self):
        inputs = torch.randn(2, 4, requires_grad=True)
        base = self.operator.prior_weights.expand(2, -1)
        perturbation = torch.tensor(
            [[0.1, -0.1, 0.05, 0.0, -0.05], [-0.05, 0.08, 0.0, -0.1, 0.1]]
        )
        result = self.pathway(
            inputs,
            base,
            base * (1 + perturbation),
            base * (1 - perturbation),
        )
        self.assertEqual(result.base_token.shape, (2, 8))
        self.assertEqual(result.positive_token.shape, (2, 8))
        self.assertEqual(result.negative_token.shape, (2, 8))
        self.assertEqual(result.route.shape, (2, 3))
        self.assertTrue(torch.allclose(result.route.sum(dim=1), torch.ones(2)))
        self.assertTrue(torch.equal(result.route, result.positive_route))
        self.assertTrue(torch.equal(result.route, result.negative_route))
        self.assertTrue((result.gate > 0).all() and (result.gate < 1).all())
        self.assertEqual(result.positive_gate.shape, result.gate.shape)
        self.assertEqual(result.negative_gate.shape, result.gate.shape)
        # A single returned route is the auditable guarantee that base/+/- did
        # not obtain independent router calls.
        loss = (
            result.base_token.mean()
            + result.positive_token.mean()
            + result.negative_token.mean()
        )
        loss.backward()
        self.assertIsNotNone(inputs.grad)
        self.assertIsNotNone(self.pathway.residual_projection.weight.grad)
        self.assertIsNotNone(self.pathway.router[0].weight.grad)

    def test_identical_views_produce_identical_tokens(self):
        inputs = torch.randn(1, 4)
        result = self.pathway(inputs, self.operator.prior_weights)
        self.assertTrue(torch.equal(result.base_token, result.positive_token))
        self.assertTrue(torch.equal(result.base_token, result.negative_token))

    def test_independent_route_ablation_recomputes_view_routes(self):
        inputs = torch.randn(1, 4)
        base = self.operator.prior_weights.unsqueeze(0)
        positive = base * torch.tensor([[1.2, 0.8, 1.1, 0.7, 1.0]])
        negative = base * torch.tensor([[0.8, 1.2, 0.9, 1.3, 1.0]])
        result = self.pathway(
            inputs,
            base,
            positive,
            negative,
            shared_route=False,
        )
        self.assertFalse(torch.equal(result.route, result.positive_route))
        self.assertFalse(torch.equal(result.route, result.negative_route))

    def test_negative_free_loss_is_zero_for_identical_nonzero_views(self):
        tokens = torch.randn(2, 6, 8)
        loss = negative_free_consistency_loss(tokens, tokens)
        self.assertLess(float(loss), 1e-6)
        changed = tokens.clone()
        changed[..., 0] = -changed[..., 0]
        self.assertGreater(
            float(negative_free_consistency_loss(tokens, changed)),
            float(loss),
        )

    def test_vicreg_options_are_finite_for_single_patient_six_tokens(self):
        first = torch.randn(1, 6, 8, requires_grad=True)
        second = torch.randn(1, 6, 8, requires_grad=True)
        loss = negative_free_consistency_loss(
            first,
            second,
            variance_weight=0.1,
            covariance_weight=0.01,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(first.grad)
        self.assertIsNotNone(second.grad)


class IdentifiabilityTangentRegularizerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(67)
        edges = torch.tensor(
            [[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]], dtype=torch.long
        )
        prior = torch.tensor([1.0, 0.8, 1.2, 0.7, 0.5])
        operator = ReferenceSpectralOperator(
            edges, prior, 4, lambda_max=3.0, max_log_deformation=0.5
        )
        self.response = ChebyshevMomentResponse(operator, order=2)
        self.dictionary = EdgeDeformationDictionary(prior, rank=2)
        self.ddkac = SharedRouteDDKACPathway(4, 6, operator, order=2)

    def test_off_mode_is_exact_zero(self):
        module = IdentifiabilityTangentRegularizer(
            self.response, self.dictionary, self.ddkac, mode="off"
        )
        loss = module(
            torch.randn(1, 4),
            torch.randn(1, 2) * 0.1,
            torch.randn(1, 3),
        )
        self.assertEqual(float(loss), 0.0)

    def test_randomized_mode_is_bounded_and_differentiable(self):
        module = IdentifiabilityTangentRegularizer(
            self.response,
            self.dictionary,
            self.ddkac,
            mode="randomized",
            random_probes=2,
        )
        expression = torch.randn(1, 4, requires_grad=True)
        coefficients = (torch.randn(1, 2) * 0.1).requires_grad_()
        route_logits = torch.randn(1, 3, requires_grad=True)
        loss = module(expression, coefficients, route_logits)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss.detach()), 0.0)
        self.assertLessEqual(float(loss.detach()), 1.0 + 1e-6)
        loss.backward()
        self.assertIsNotNone(coefficients.grad)
        self.assertIsNotNone(route_logits.grad)
        self.assertIsNotNone(self.dictionary.matrix.grad)
        self.assertIsNotNone(self.ddkac.structure_projection.weight.grad)

    def test_full_mode_runs_on_small_graph(self):
        module = IdentifiabilityTangentRegularizer(
            self.response, self.dictionary, self.ddkac, mode="full"
        )
        loss = module(
            torch.randn(1, 4),
            torch.randn(1, 2) * 0.1,
            torch.randn(1, 3),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss.detach()), 0.0)
        self.assertLessEqual(float(loss.detach()), 1.0 + 1e-5)


class SixGroupPCCMKADDKACEncoderTests(unittest.TestCase):
    def _make_encoder(self, lambda_ssl=0.1, lambda_id=0.0, **kwargs):
        dimensions = [4, 5, 6, 7, 8, 9]
        priors = []
        rng = np.random.default_rng(81)
        for group, dimension in enumerate(dimensions):
            priors.append(
                build_training_correlation_prior(
                    rng.normal(size=(16, dimension)).astype(np.float32),
                    [f"group{group}_gene{index}" for index in range(dimension)],
                    neighbours=min(2, dimension - 1),
                )
            )
        return PCCMKADDKACEncoder(
            dimensions,
            priors,
            output_dim=8,
            moment_order=2,
            dictionary_rank=2,
            solver_iterations=2,
            solver_step_size=0.05,
            lambda_ssl=lambda_ssl,
            lambda_id=lambda_id,
            **kwargs,
        ), dimensions

    def test_six_tokens_losses_diagnostics_and_gradients(self):
        encoder, dimensions = self._make_encoder(lambda_ssl=0.1, lambda_id=0.0)
        inputs = [
            torch.randn(1, dimension, requires_grad=True)
            for dimension in dimensions
        ]
        tokens = encoder(inputs, generator=torch.Generator().manual_seed(91))
        self.assertEqual(tokens.shape, (1, 6, 8))
        self.assertEqual(encoder.last_positive_tokens.shape, tokens.shape)
        self.assertEqual(encoder.last_negative_tokens.shape, tokens.shape)
        self.assertEqual(encoder.diagnostics["rho"].shape, (1, 6))
        self.assertEqual(encoder.diagnostics["target_moments"].shape, (1, 6, 2))
        self.assertEqual(encoder.diagnostics["routes"].shape, (1, 6, 3))
        self.assertEqual(encoder.diagnostics["gates"].shape, (1, 6))
        self.assertEqual(len(encoder.last_edge_diagnostics), 6)
        self.assertTrue(torch.isfinite(encoder.auxiliary_loss))
        (tokens.square().mean() + encoder.auxiliary_loss).backward()
        self.assertTrue(all(item.grad is not None for item in inputs))
        self.assertIsNotNone(encoder.pathways[0].target_network.offset_head.weight.grad)
        self.assertIsNotNone(encoder.pathways[0].dictionary.matrix.grad)

    def test_eval_has_no_augmentation_and_is_repeatable(self):
        encoder, dimensions = self._make_encoder(lambda_ssl=0.1, lambda_id=0.0)
        encoder.eval()
        inputs = [torch.randn(1, dimension) for dimension in dimensions]
        first = encoder(inputs)
        second = encoder(inputs)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first, encoder.last_positive_tokens))
        self.assertTrue(torch.equal(first, encoder.last_negative_tokens))
        self.assertTrue(torch.equal(encoder.diagnostics["krylov_error"], torch.zeros(1, 6)))

    def test_all_groups_share_one_lambda(self):
        encoder, _ = self._make_encoder()
        lambdas = [
            float(pathway.moment_response.operator.lambda_max)
            for pathway in encoder.pathways
        ]
        self.assertEqual(len(set(lambdas)), 1)
        self.assertAlmostEqual(lambdas[0], encoder.shared_lambda, places=6)

    def test_target_only_mode_freezes_calibration_dictionary(self):
        encoder, _ = self._make_encoder(
            calibration_mode="target_only",
            augmentation_mode="off",
            lambda_ssl=0.0,
            lambda_id=0.0,
        )
        self.assertFalse(encoder.pathways[0].dictionary.matrix.requires_grad)
        self.assertTrue(
            encoder.pathways[0].target_network.offset_head.weight.requires_grad
        )

    def test_direct_edge_gate_control_receives_prediction_gradient(self):
        encoder, dimensions = self._make_encoder(
            calibration_mode="direct_edge_gate",
            augmentation_mode="off",
            lambda_ssl=0.0,
            lambda_id=0.0,
        )
        inputs = [torch.randn(1, dimension) for dimension in dimensions]
        tokens = encoder(inputs)
        tokens.square().mean().backward()
        gate = encoder.pathways[0].direct_edge_gate.network[-1]
        self.assertIsNotNone(gate.weight.grad)
        self.assertTrue(torch.isfinite(gate.weight.grad).all())

    def test_control_augmentation_modes_preserve_six_token_interface(self):
        for mode in sorted(ControlGraphAugmentor.VALID_MODES):
            with self.subTest(mode=mode):
                encoder, dimensions = self._make_encoder(
                    augmentation_mode=mode,
                    lambda_ssl=0.1,
                    lambda_id=0.0,
                )
                inputs = [torch.randn(1, dimension) for dimension in dimensions]
                tokens = encoder(
                    inputs, generator=torch.Generator().manual_seed(99)
                )
                self.assertEqual(tokens.shape, (1, 6, 8))
                self.assertTrue(torch.isfinite(encoder.auxiliary_loss))

    def test_diagnostics_export_all_machine_readable_formats(self):
        encoder, dimensions = self._make_encoder(lambda_ssl=0.0, lambda_id=0.0)
        encoder.eval()
        encoder([torch.randn(1, dimension) for dimension in dimensions])
        with tempfile.TemporaryDirectory() as directory:
            recorder = PCCMKADiagnosticRecorder(directory, fold=2)
            recorder.add(
                "case-demo",
                "val",
                encoder.diagnostics,
                encoder.last_edge_diagnostics,
                epoch=3,
            )
            recorder.flush()
            recorder.save_fold_summary(
                c_index=0.65,
                best_epoch=3,
                elapsed_seconds=12.5,
                peak_gpu_bytes=1024,
            )
            root = Path(directory)
            self.assertTrue((root / "fold_2_patients.jsonl").is_file())
            self.assertTrue((root / "fold_2_patients.csv").is_file())
            self.assertTrue((root / "fold_2_diagnostics.npz").is_file())
            summary = json.loads(
                (root / "fold_2_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["best_epoch"], 3)


class MRePathPCCMKAIntegrationTests(unittest.TestCase):
    def test_resolved_full_config_and_ablation(self):
        full = load_pc_cmka_config(
            "configs/pc_cmka_ddkac.json", "full_pc_cmka_ddkac"
        )
        kwargs = encoder_kwargs(full)
        self.assertEqual(kwargs["augmentation_mode"], "hessian_antithetic")
        self.assertTrue(kwargs["krylov_enabled"])
        self.assertEqual(kwargs["target_hidden_dim"], 64)
        self.assertEqual(kwargs["solver_gradient_clip"], 10.0)
        self.assertEqual(kwargs["hessian_damping"], 1e-5)
        self.assertEqual(kwargs["top_edges"], 10)
        structure = load_pc_cmka_config(
            "configs/pc_cmka_ddkac.json",
            "moment_shared_route_structure_ssl",
        )
        structure_kwargs = encoder_kwargs(structure)
        self.assertEqual(structure_kwargs["ssl_structure_weight"], 1.0)
        self.assertEqual(structure_kwargs["ssl_fusion_weight"], 0.0)
        self.assertEqual(structure_kwargs["lambda_id"], 0.0)

    def test_mrepath_end_to_end_forward_backward_keeps_four_logits(self):
        torch.manual_seed(113)
        dimensions = [4, 5, 6, 7, 8, 9]
        rng = np.random.default_rng(114)
        priors = [
            build_training_correlation_prior(
                rng.normal(size=(18, dimension)).astype(np.float32),
                [f"g{group}_{index}" for index in range(dimension)],
                neighbours=min(2, dimension - 1),
            )
            for group, dimension in enumerate(dimensions)
        ]
        model = MRePath(
            omic_sizes=dimensions,
            n_classes=4,
            graph_type="mlp",
            path_input_dim=16,
            num_patches=5,
            weighting_mode="fixed",
            genomic_encoder="pc_cmka_ddkac",
            pc_cmka_priors=priors,
            pc_cmka_kwargs={
                "moment_order": 2,
                "dictionary_rank": 2,
                "solver_iterations": 2,
                "solver_step_size": 0.05,
                "lambda_ssl": 0.0,
                "lambda_id": 0.0,
            },
        )
        # Parent initialization must not destroy the Delta_mu=0 prior fallback.
        for pathway in model.genomic_encoder.pathways:
            self.assertTrue(
                torch.equal(
                    pathway.target_network.offset_head.weight,
                    torch.zeros_like(pathway.target_network.offset_head.weight),
                )
            )
        graph = Data(
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_latent=torch.empty((2, 0), dtype=torch.long),
        )
        inputs = {
            "x_path": torch.randn(5, 16),
            "graph": graph,
            **{
                f"x_omic{index + 1}": torch.randn(1, dimension)
                for index, dimension in enumerate(dimensions)
            },
        }
        logits = model(**inputs)
        self.assertEqual(logits.shape, (1, 4))
        loss = logits.square().mean() + model.auxiliary_loss
        loss.backward()
        self.assertIsNotNone(
            model.genomic_encoder.pathways[0].target_network.offset_head.weight.grad
        )

    def test_dictionary_parameter_group_has_no_generic_weight_decay(self):
        dimensions = [4, 5, 6, 7, 8, 9]
        rng = np.random.default_rng(121)
        priors = [
            build_training_correlation_prior(
                rng.normal(size=(16, dimension)),
                [f"g{group}_{index}" for index in range(dimension)],
                neighbours=2,
            )
            for group, dimension in enumerate(dimensions)
        ]
        encoder = PCCMKADDKACEncoder(
            dimensions,
            priors,
            output_dim=8,
            dictionary_rank=2,
            solver_iterations=1,
            lambda_ssl=0.0,
            lambda_id=0.0,
        )
        args = SimpleNamespace(
            opt="adam",
            lr=1e-4,
            reg=1e-5,
            mrepath_genomic_encoder="pc_cmka_ddkac",
        )
        optimizer = _init_optim(args, encoder)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 1e-5)
        self.assertEqual(optimizer.param_groups[1]["weight_decay"], 0.0)
        dictionary_ids = {
            id(pathway.dictionary.matrix) for pathway in encoder.pathways
        }
        no_decay_ids = {
            id(parameter) for parameter in optimizer.param_groups[1]["params"]
        }
        self.assertEqual(dictionary_ids, no_decay_ids)


if __name__ == "__main__":
    unittest.main()
