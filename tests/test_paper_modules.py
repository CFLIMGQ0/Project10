import unittest

import torch

from models.layers.fusion import AlignFusion
from models.model_HGNN import DynamicWeighting


class DynamicWeightingTests(unittest.TestCase):
    def test_equations_and_gradients(self):
        torch.manual_seed(7)
        module = DynamicWeighting(
            embedding_dim=16, num_pathways=6, num_patches=8
        )
        pathology = torch.randn(2, 8, 16, requires_grad=True)
        genomics = torch.randn(2, 6, 16, requires_grad=True)

        weights, confidence = module(pathology, genomics)
        path_mono, gene_mono, path_holo, gene_holo = confidence
        expected = torch.softmax(
            torch.cat(
                (path_mono + path_holo, gene_mono + gene_holo), dim=1
            ),
            dim=1,
        )

        self.assertEqual(weights.shape, (2, 2))
        self.assertTrue(torch.all(torch.isfinite(weights)))
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(2)))
        self.assertTrue(torch.allclose(weights, expected))
        self.assertTrue(
            torch.allclose(path_holo + gene_holo, torch.ones_like(path_holo))
        )

        loss = (2.0 * weights[:, 0] - weights[:, 1]).sum()
        loss.backward()
        for parameter in module.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.all(torch.isfinite(parameter.grad)))
        self.assertIsNotNone(pathology.grad)
        self.assertIsNotNone(genomics.grad)

    def test_variable_patch_fallback_is_finite(self):
        module = DynamicWeighting(
            embedding_dim=16, num_pathways=6, num_patches=8
        )
        weights, _ = module(torch.randn(1, 5, 16), torch.randn(1, 6, 16))
        self.assertTrue(torch.all(torch.isfinite(weights)))
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(1)))


class InteractiveAlignmentFusionTests(unittest.TestCase):
    def test_paper_token_shapes_and_gradients(self):
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
        self.assertNotIn("final_cross_attn", dict(module.named_modules()))


if __name__ == "__main__":
    unittest.main()
