"""Optimization-only losses used by the PC-CMKA genomic branch."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_variance_consistency(
    positive: torch.Tensor,
    negative: torch.Tensor,
    variance_floor: float = 0.5,
) -> torch.Tensor:
    positive = positive.flatten(start_dim=1)
    negative = negative.flatten(start_dim=1)
    cosine = (1.0 - F.cosine_similarity(positive, negative, dim=1)).mean()
    if positive.shape[0] > 1:
        std_positive = positive.std(dim=0, unbiased=False)
        std_negative = negative.std(dim=0, unbiased=False)
        variance = 0.5 * (
            F.relu(variance_floor - std_positive).mean()
            + F.relu(variance_floor - std_negative).mean()
        )
    else:
        # Batch-size one is the formal WSI setting.  Feature-wise dispersion is
        # used rather than fabricating negative patients.
        variance = 0.5 * (
            F.relu(variance_floor - positive.std(dim=1, unbiased=False)).mean()
            + F.relu(variance_floor - negative.std(dim=1, unbiased=False)).mean()
        )
    return cosine + variance


def gate_consistency(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    return (positive - negative).square().mean()


def squared_tangent_cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = first.flatten(start_dim=1)
    second = second.flatten(start_dim=1)
    numerator = (first * second).sum(dim=1).square()
    denominator = first.square().sum(dim=1) * second.square().sum(dim=1)
    return (numerator / denominator.clamp_min(1e-8)).mean()
