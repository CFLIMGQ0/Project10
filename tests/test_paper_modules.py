import unittest
import json
import re
from pathlib import Path
from types import SimpleNamespace

import torch
import dhg

from models.layers.fusion import AlignFusion
from models.layers.incidence_hypergraph import IncidenceHypergraph
from models.layers.kan import KANLinear
from models.layers.genomic_encoders import ENCODER_NAMES, build_genomic_encoder
from models.layers.reliability_rebalance import (
    QualityConflictWeighting,
    discrete_survival_distribution,
    jensen_shannon_divergence,
)
from models.model_HGNN import DynamicWeighting, GeneGraphAggregator, MRePath
from utils.core_utils import _init_optim
from utils.hypergraph_cache import edges_to_incidence, subset_incidence


class PaperInteractiveAlignmentFusionTests(unittest.TestCase):
    def test_token_shapes_and_gradients(self):
        torch.manual_seed(11)
        module = AlignFusion(embedding_dim=32, num_heads=4, num_pathways=6)
        pathology = torch.randn(2, 13, 32, requires_grad=True)
        genomics = torch.randn(2, 6, 32, requires_grad=True)

        pathology_fused, genomics_fused = module(pathology, genomics)

        self.assertEqual(pathology_fused.shape, pathology.shape)
        self.assertEqual(genomics_fused.shape, genomics.shape)
        self.assertTrue(torch.all(torch.isfinite(pathology_fused)))
        self.assertTrue(torch.all(torch.isfinite(genomics_fused)))
        (pathology_fused.mean() + genomics_fused.mean()).backward()
        self.assertIsNotNone(pathology.grad)
        self.assertIsNotNone(genomics.grad)

    def test_all_paper_fusion_ablation_variants(self):
        torch.manual_seed(13)
        pathology = torch.randn(1, 9, 32)
        genomics = torch.randn(1, 6, 32)
        for variant in ("ifa", "pg_gp", "sa_pg", "sa_gp"):
            with self.subTest(variant=variant):
                module = AlignFusion(
                    embedding_dim=32,
                    num_heads=4,
                    num_pathways=6,
                    variant=variant,
                )
                pathology_fused, genomics_fused = module(pathology, genomics)
                self.assertEqual(pathology_fused.shape, pathology.shape)
                self.assertEqual(genomics_fused.shape, genomics.shape)
                self.assertTrue(torch.isfinite(pathology_fused).all())
                self.assertTrue(torch.isfinite(genomics_fused).all())


class PaperMRePathStructureTests(unittest.TestCase):
    def test_cached_incidence_matches_dhg_mean_message_passing(self):
        edges = torch.tensor(
            [
                [0, 0, 1, 1, 2, 2, 3, 3],
                [1, 2, 0, 3, 0, 3, 1, 2],
            ],
            dtype=torch.long,
        )
        incidence, _ = edges_to_incidence(edges)
        cached = IncidenceHypergraph(4, incidence)
        reference = dhg.Hypergraph(
            num_v=4,
            e_list=[[0, 1, 2], [1, 0, 3], [2, 0, 3], [3, 1, 2]],
        )
        features = torch.randn(4, 7)
        self.assertTrue(
            torch.allclose(
                cached.v2v(features, aggr="mean"),
                reference.v2v(features, aggr="mean"),
                atol=1e-6,
            )
        )

    def test_cached_incidence_preserves_random_patch_order(self):
        edges = torch.tensor(
            [
                [0, 0, 1, 1, 2, 2, 3, 3],
                [1, 2, 0, 3, 0, 3, 1, 2],
            ],
            dtype=torch.long,
        )
        selected = torch.tensor([3, 0, 2])
        full_incidence, centers = edges_to_incidence(edges)
        cached_subset = subset_incidence(
            full_incidence, centers, selected, num_source_nodes=4
        )

        mapping = torch.full((4,), -1, dtype=torch.long)
        mapping[selected] = torch.arange(selected.numel())
        valid = (mapping[edges[0]] >= 0) & (mapping[edges[1]] >= 0)
        remapped = mapping[edges[:, valid]]
        direct_subset, _ = edges_to_incidence(remapped)

        features = torch.randn(3, 5)
        cached_output = IncidenceHypergraph(
            3, cached_subset
        ).v2v(features)
        direct_output = IncidenceHypergraph(
            3, direct_subset
        ).v2v(features)
        self.assertTrue(
            torch.allclose(cached_output, direct_output, atol=1e-6)
        )

    def test_resnet50_input_and_paper_modules(self):
        model = MRePath(
            omic_sizes=[4, 5, 6, 7, 8, 9],
            path_input_dim=1024,
            num_patches=8,
        )

        self.assertEqual(model.pathomics_fc[0].in_features, 1024)
        self.assertEqual(model.dynamic_weighting.path_confidence[0].in_features, 8)
        self.assertEqual(
            model.dynamic_weighting.genomic_confidence[0].in_features, 6 * 256
        )
        self.assertTrue(hasattr(model, "dynamic_weighting"))

    def test_all_pathology_aggregators_and_edge_modes_construct(self):
        for graph_type in ("mlp", "gat", "gcn", "hgnn", "shgnn"):
            with self.subTest(graph_type=graph_type):
                model = MRePath(
                    omic_sizes=[4, 5, 6, 7, 8, 9],
                    path_input_dim=32,
                    num_patches=8,
                    graph_type=graph_type,
                    hyperedge_mode="both",
                    weighting_mode="fixed",
                    fixed_pathology_weight=0.7,
                    fixed_genomic_weight=0.3,
                )
                self.assertEqual(model.graph_type, graph_type)
                self.assertIsNone(model.dynamic_weighting)
                self.assertTrue(
                    torch.allclose(
                        model.fixed_modality_weights,
                        torch.tensor([[0.7, 0.3]]),
                    )
                )

    def test_gene_aggregation_ablation_variants(self):
        genomics = torch.randn(1, 6, 32)
        for method in ("default", "gcn", "gat", "kan"):
            with self.subTest(method=method):
                module = GeneGraphAggregator(embedding_dim=32, method=method)
                input_features = genomics.clone().requires_grad_(True)
                output = module(input_features)
                self.assertEqual(output.shape, genomics.shape)
                self.assertTrue(torch.isfinite(output).all())
                if method == "kan":
                    output.mean().backward()
                    self.assertIsNotNone(input_features.grad)

    def test_kan_spline_layer_shapes_and_gradients(self):
        module = KANLinear(7, 5)
        inputs = torch.randn(2, 3, 7, requires_grad=True)
        output = module(inputs)
        self.assertEqual(output.shape, (2, 3, 5))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertIsNotNone(module.spline_weight.grad)

    def test_five_improved_genomic_encoders(self):
        sizes = [7, 8, 9, 10, 11, 12]
        inputs = [torch.randn(size, requires_grad=True) for size in sizes]
        graphs = [torch.eye(size) for size in sizes]
        for name in ENCODER_NAMES:
            with self.subTest(name=name):
                module = build_genomic_encoder(
                    name,
                    sizes,
                    hidden_dim=32,
                    output_dim=16,
                    gene_graphs=graphs if name == "dd_kac" else None,
                )
                output = module(inputs)
                self.assertEqual(output.shape, (1, 6, 16))
                self.assertTrue(torch.isfinite(output).all())
                (output.square().mean() + module.auxiliary_loss).backward()
                self.assertTrue(
                    any(parameter.grad is not None for parameter in module.parameters())
                )

    def test_dynamic_weights_follow_equations_and_receive_gradients(self):
        torch.manual_seed(7)
        module = DynamicWeighting(
            embedding_dim=8, num_pathways=2, num_patches=5
        )
        pathology = torch.randn(2, 5, 8, requires_grad=True)
        genomics = torch.randn(2, 2, 8, requires_grad=True)

        weights, (path_mono, gene_mono, path_holo, gene_holo) = module(
            pathology, genomics
        )

        log_joint = torch.log(path_mono) + torch.log(gene_mono)
        self.assertTrue(
            torch.allclose(path_holo, torch.log(path_mono) / log_joint)
        )
        self.assertTrue(
            torch.allclose(gene_holo, torch.log(gene_mono) / log_joint)
        )
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(2)))

        weighted_signal = (
            weights[:, 0] * pathology.mean(dim=(1, 2))
            + weights[:, 1] * genomics.mean(dim=(1, 2))
        ).sum()
        weighted_signal.backward()
        self.assertIsNotNone(module.path_confidence[0].weight.grad)
        self.assertIsNotNone(module.genomic_confidence[0].weight.grad)


class QualityConflictRebalanceTests(unittest.TestCase):
    def test_survival_distribution_and_js_divergence(self):
        logits = torch.randn(3, 4)
        distribution = discrete_survival_distribution(logits)
        self.assertEqual(distribution.shape, (3, 5))
        self.assertTrue(
            torch.allclose(
                distribution.sum(dim=1),
                torch.ones(3),
                atol=1e-6,
            )
        )
        divergence = jensen_shannon_divergence(
            distribution, distribution
        )
        self.assertTrue(torch.allclose(divergence, torch.zeros_like(divergence)))

    def test_all_rebalance_variants_are_finite_and_trainable(self):
        pathology = torch.randn(2, 12, 16, requires_grad=True)
        genomics = torch.randn(2, 6, 16, requires_grad=True)
        for variant in ("quality", "conflict", "quality_conflict"):
            with self.subTest(variant=variant):
                module = QualityConflictWeighting(
                    embedding_dim=16,
                    n_classes=4,
                    variant=variant,
                    modality_dropout=0.0,
                    monotonicity_weight=0.1,
                    mismatch_loss_weight=0.1,
                )
                weights, confidence = module(pathology, genomics)
                self.assertEqual(weights.shape, (2, 2))
                self.assertTrue(torch.isfinite(weights).all())
                self.assertTrue(
                    torch.allclose(weights.sum(dim=1), torch.ones(2))
                )
                self.assertEqual(len(confidence), 5)
                (
                    weights.mean()
                    + module.auxiliary_loss
                    + module.last_pathology_logits.mean()
                    + module.last_genomic_logits.mean()
                ).backward(retain_graph=True)

    def test_modality_dropout_never_drops_both_modalities(self):
        module = QualityConflictWeighting(
            embedding_dim=8,
            variant="quality_conflict",
            modality_dropout=0.99,
        )
        module.train()
        pathology = torch.randn(128, 4, 8)
        genomics = torch.randn(128, 6, 8)
        weights, _ = module(pathology, genomics)
        availability = module.last_availability
        self.assertTrue((availability.sum(dim=1) >= 1).all())
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(128)))


class PaperTrainingProfileTests(unittest.TestCase):
    def test_adam_uses_reported_weight_decay(self):
        model = torch.nn.Linear(3, 2)
        args = SimpleNamespace(opt="adam", lr=1e-4, reg=1e-5)

        optimizer = _init_optim(args, model)

        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-4)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 1e-5)

    def test_launcher_matches_machine_readable_contract(self):
        root = Path(__file__).resolve().parents[1]
        contract = json.loads(
            (root / "configs/paper_coadread.json").read_text()
        )
        launcher = (root / "scripts/run_coadread.sh").read_text()

        expected_flags = {
            "--encoding_dim": contract["pathology"]["feature_dimension"],
            "--num_patches": contract["pathology"]["patches_per_case"],
            "--n_classes": contract["cohort"]["survival_bins"],
            "--k": contract["cohort"]["folds"],
            "--lr": contract["training"]["learning_rate"],
            "--reg": contract["training"]["weight_decay"],
            "--max_epochs": contract["training"]["epochs"],
            "--seed": contract["training"]["seed"],
            "--warmup_epochs": contract["training"]["warmup_epochs"],
        }
        for flag, value in expected_flags.items():
            match = re.search(
                rf"{re.escape(flag)}\s+([^\s\\]+)", launcher
            )
            self.assertIsNotNone(match, flag)
            if isinstance(value, float):
                self.assertEqual(float(match.group(1)), value)
            else:
                self.assertEqual(int(match.group(1)), value)
        self.assertIn(
            f'--opt {contract["training"]["optimizer"]}', launcher
        )
        self.assertIn(
            f'--bag_loss {contract["training"]["loss"]}', launcher
        )
        self.assertIn(
            f'--checkpoint_selection '
            f'{contract["training"]["checkpoint_selection"]}',
            launcher,
        )
        self.assertNotIn("--weighted_sample", launcher)


if __name__ == "__main__":
    unittest.main()
