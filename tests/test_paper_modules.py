import unittest
import json
import re
from pathlib import Path
from types import SimpleNamespace

import torch

from models.layers.fusion import AlignFusion
from models.model_HGNN import DynamicWeighting, GeneGraphAggregator, MRePath
from utils.core_utils import _init_optim


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
        for method in ("default", "gcn", "gat"):
            with self.subTest(method=method):
                module = GeneGraphAggregator(embedding_dim=32, method=method)
                output = module(genomics)
                self.assertEqual(output.shape, genomics.shape)
                self.assertTrue(torch.isfinite(output).all())

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
