"""Archived tests for the non-canonical ABMIL baseline."""

import unittest

import torch

from archive.experimental_models.model_abmil import ABMIL


class ABMILTests(unittest.TestCase):
    def test_forward_shape_and_gradients(self):
        torch.manual_seed(3)
        model = ABMIL(path_input_dim=16, hidden_dim=8, attention_dim=4)
        features = torch.randn(1, 11, 16, requires_grad=True)
        logits = model(data_WSI=features)
        self.assertEqual(logits.shape, (1, 4))
        logits.sum().backward()
        self.assertIsNotNone(features.grad)


if __name__ == "__main__":
    unittest.main()
