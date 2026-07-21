import unittest
from types import SimpleNamespace

import torch

from models.layers.fusion import AlignFusion
from models.model_HGNN import MRePath
from utils.core_utils import _init_optim


class ReleasedInteractiveAlignmentFusionTests(unittest.TestCase):
    def test_token_shapes_and_gradients(self):
        torch.manual_seed(11)
        module = AlignFusion(embedding_dim=32, num_heads=4, num_pathways=6)
        token = torch.randn(2, 19, 32, requires_grad=True)

        fused = module(token)

        self.assertEqual(fused.shape, token.shape)
        self.assertTrue(torch.all(torch.isfinite(fused)))
        fused.mean().backward()
        self.assertIsNotNone(token.grad)
        self.assertIn("final_cross_attn", dict(module.named_modules()))


class ReleasedMRePathStructureTests(unittest.TestCase):
    def test_resnet50_input_and_released_modules(self):
        model = MRePath(
            omic_sizes=[4, 5, 6, 7, 8, 9],
            path_input_dim=1024,
            num_patches=8,
        )

        self.assertEqual(model.pathomics_fc[0].in_features, 1024)
        self.assertEqual(model.ConfidNet_p[0].in_features, 8)
        self.assertEqual(model.ConfidNet_g[0].in_features, 6 * 256)
        self.assertTrue(hasattr(model, "feed_forward"))
        self.assertTrue(hasattr(model, "layer_norm"))
        self.assertFalse(hasattr(model, "dynamic_weighting"))


class PaperTrainingProfileTests(unittest.TestCase):
    def test_adam_uses_reported_weight_decay(self):
        model = torch.nn.Linear(3, 2)
        args = SimpleNamespace(opt="adam", lr=1e-4, reg=1e-5)

        optimizer = _init_optim(args, model)

        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-4)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 1e-5)


if __name__ == "__main__":
    unittest.main()
