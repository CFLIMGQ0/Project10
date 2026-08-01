"""Quality- and conflict-aware modality rebalancing for MRePath.

The module is intentionally separate from the paper's original weighting
equations.  Selecting the ``original`` rebalance variant therefore preserves
the released MRePath execution path exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def discrete_survival_distribution(logits: torch.Tensor) -> torch.Tensor:
    """Convert discrete-time hazard logits to an event/tail distribution."""

    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1.0 - hazards, dim=1)
    survival_before = torch.cat(
        (torch.ones_like(hazards[:, :1]), survival[:, :-1]), dim=1
    )
    event_probability = survival_before * hazards
    return torch.cat((event_probability, survival[:, -1:]), dim=1)


def jensen_shannon_divergence(
    first: torch.Tensor, second: torch.Tensor
) -> torch.Tensor:
    """Per-sample Jensen-Shannon divergence for probability vectors."""

    eps = torch.finfo(first.dtype).eps
    first = first.clamp_min(eps)
    second = second.clamp_min(eps)
    midpoint = 0.5 * (first + second)
    return 0.5 * (
        (first * (first.log() - midpoint.log())).sum(dim=1, keepdim=True)
        + (second * (second.log() - midpoint.log())).sum(
            dim=1, keepdim=True
        )
    )


class QualityEstimator(nn.Module):
    """Estimate one modality's reliability from pooled feature statistics."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        hidden_dim = max(embedding_dim // 2, 32)
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        summary = torch.cat(
            (
                tokens.mean(dim=1),
                tokens.std(dim=1, unbiased=False),
            ),
            dim=1,
        )
        return self.network(summary)


class ConflictEstimator(nn.Module):
    """Produce directional modality scores and a pair-matching logit."""

    def __init__(self, embedding_dim: int, n_classes: int) -> None:
        super().__init__()
        probability_dim = n_classes + 1
        input_dim = embedding_dim * 4 + probability_dim * 3 + 1
        hidden_dim = max(embedding_dim, 64)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.modality_scores = nn.Linear(hidden_dim // 2, 2)
        self.match_logit = nn.Linear(hidden_dim // 2, 1)

    def forward(
        self,
        pathology_summary: torch.Tensor,
        genomic_summary: torch.Tensor,
        pathology_distribution: torch.Tensor,
        genomic_distribution: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        js_divergence = jensen_shannon_divergence(
            pathology_distribution, genomic_distribution
        )
        inputs = torch.cat(
            (
                pathology_summary,
                genomic_summary,
                torch.abs(pathology_summary - genomic_summary),
                pathology_summary * genomic_summary,
                pathology_distribution,
                genomic_distribution,
                torch.abs(
                    pathology_distribution - genomic_distribution
                ),
                js_divergence,
            ),
            dim=1,
        )
        encoded = self.encoder(inputs)
        return (
            self.modality_scores(encoded),
            self.match_logit(encoded),
            js_divergence,
        )


class QualityConflictWeighting(nn.Module):
    """Interpretable quality/conflict gating proposed in the research plan."""

    VALID_VARIANTS = {"quality", "conflict", "quality_conflict"}

    def __init__(
        self,
        embedding_dim: int = 256,
        n_classes: int = 4,
        variant: str = "quality_conflict",
        modality_dropout: float = 0.2,
        monotonicity_weight: float = 0.1,
        monotonicity_margin: float = 0.02,
        mismatch_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"Unknown reliability variant: {variant}")
        if not 0.0 <= modality_dropout < 1.0:
            raise ValueError("modality_dropout must be in [0, 1)")
        if min(
            monotonicity_weight,
            monotonicity_margin,
            mismatch_loss_weight,
        ) < 0:
            raise ValueError("auxiliary loss values must be non-negative")

        self.variant = variant
        self.modality_dropout = modality_dropout
        self.monotonicity_weight = monotonicity_weight
        self.monotonicity_margin = monotonicity_margin
        self.mismatch_loss_weight = mismatch_loss_weight

        self.pathology_quality = QualityEstimator(embedding_dim)
        self.genomic_quality = QualityEstimator(embedding_dim)
        self.pathology_risk = nn.Linear(embedding_dim, n_classes)
        self.genomic_risk = nn.Linear(embedding_dim, n_classes)
        self.conflict = ConflictEstimator(embedding_dim, n_classes)

        self.register_buffer(
            "previous_genomic_summary",
            torch.zeros(1, embedding_dim),
            persistent=False,
        )
        self.register_buffer(
            "memory_ready",
            torch.tensor(False),
            persistent=False,
        )
        self.last_auxiliary_losses: dict[str, torch.Tensor] = {}

    @staticmethod
    def _summaries(
        pathology: torch.Tensor, genomics: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return pathology.mean(dim=1), genomics.mean(dim=1)

    def _components(
        self, pathology: torch.Tensor, genomics: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        pathology_summary, genomic_summary = self._summaries(
            pathology, genomics
        )
        pathology_quality = self.pathology_quality(pathology)
        genomic_quality = self.genomic_quality(genomics)
        quality_scores = torch.cat(
            (pathology_quality, genomic_quality), dim=1
        )

        pathology_logits = self.pathology_risk(pathology_summary)
        genomic_logits = self.genomic_risk(genomic_summary)
        pathology_distribution = discrete_survival_distribution(
            pathology_logits
        )
        genomic_distribution = discrete_survival_distribution(
            genomic_logits
        )
        conflict_scores, match_logit, js_divergence = self.conflict(
            pathology_summary,
            genomic_summary,
            pathology_distribution,
            genomic_distribution,
        )
        return {
            "pathology_summary": pathology_summary,
            "genomic_summary": genomic_summary,
            "quality_scores": quality_scores,
            "pathology_logits": pathology_logits,
            "genomic_logits": genomic_logits,
            "pathology_distribution": pathology_distribution,
            "genomic_distribution": genomic_distribution,
            "conflict_scores": conflict_scores,
            "match_logit": match_logit,
            "js_divergence": js_divergence,
        }

    def _scores(self, components: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.variant == "quality":
            return components["quality_scores"]
        if self.variant == "conflict":
            return components["conflict_scores"]
        return (
            components["quality_scores"] + components["conflict_scores"]
        )

    def _dropout_mask(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        mask = torch.ones(batch_size, 2, device=device)
        if not self.training or self.modality_dropout == 0.0:
            return mask
        draw = torch.rand(batch_size, device=device)
        mask[draw < self.modality_dropout / 2.0, 0] = 0.0
        mask[
            (draw >= self.modality_dropout / 2.0)
            & (draw < self.modality_dropout),
            1,
        ] = 0.0
        return mask

    @staticmethod
    def _apply_mask(
        weights: torch.Tensor, availability: torch.Tensor
    ) -> torch.Tensor:
        weights = weights * availability
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _degrade(tokens: torch.Tensor) -> torch.Tensor:
        keep = (
            torch.rand(
                tokens.shape[:2],
                device=tokens.device,
                dtype=tokens.dtype,
            )
            >= 0.5
        ).unsqueeze(-1)
        noise = torch.randn_like(tokens) * 0.1
        return tokens * keep + noise * keep

    def _monotonicity_loss(
        self,
        pathology: torch.Tensor,
        genomics: torch.Tensor,
        clean_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.monotonicity_weight == 0.0:
            return clean_weights.new_zeros(())
        pathology_degraded = self._components(
            self._degrade(pathology), genomics
        )
        genomic_degraded = self._components(
            pathology, self._degrade(genomics)
        )
        pathology_weights = torch.softmax(
            self._scores(pathology_degraded), dim=1
        )
        genomic_weights = torch.softmax(
            self._scores(genomic_degraded), dim=1
        )
        return 0.5 * (
            F.relu(
                pathology_weights[:, 0]
                - clean_weights[:, 0]
                + self.monotonicity_margin
            ).mean()
            + F.relu(
                genomic_weights[:, 1]
                - clean_weights[:, 1]
                + self.monotonicity_margin
            ).mean()
        )

    def _mismatch_loss(
        self, components: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if (
            self.variant == "quality"
            or self.mismatch_loss_weight == 0.0
        ):
            return components["match_logit"].new_zeros(())

        positive_loss = F.binary_cross_entropy_with_logits(
            components["match_logit"],
            torch.ones_like(components["match_logit"]),
        )
        if not bool(self.memory_ready.item()):
            return positive_loss

        # Clone the memory before using it in the current autograd graph.
        # The buffer is refreshed later in this forward pass; using an
        # expanded view directly would change its version before backward.
        previous_genomics = (
            self.previous_genomic_summary.detach()
            .clone()
            .expand_as(components["pathology_summary"])
        )
        previous_logits = self.genomic_risk(previous_genomics)
        _, mismatch_logit, _ = self.conflict(
            components["pathology_summary"],
            previous_genomics,
            components["pathology_distribution"],
            discrete_survival_distribution(previous_logits),
        )
        negative_loss = F.binary_cross_entropy_with_logits(
            mismatch_logit, torch.zeros_like(mismatch_logit)
        )
        return 0.5 * (positive_loss + negative_loss)

    def forward(
        self, pathology: torch.Tensor, genomics: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if pathology.ndim != 3 or genomics.ndim != 3:
            raise ValueError("pathology and genomics must be [B, tokens, D]")

        components = self._components(pathology, genomics)
        clean_weights = torch.softmax(self._scores(components), dim=1)
        availability = self._dropout_mask(
            clean_weights.shape[0], clean_weights.device
        )
        weights = self._apply_mask(clean_weights, availability)

        monotonicity_loss = weights.new_zeros(())
        mismatch_loss = weights.new_zeros(())
        if self.training:
            monotonicity_loss = self._monotonicity_loss(
                pathology, genomics, clean_weights
            )
            mismatch_loss = self._mismatch_loss(components)
            self.previous_genomic_summary.copy_(
                components["genomic_summary"].detach().mean(
                    dim=0, keepdim=True
                )
            )
            self.memory_ready.fill_(True)

        self.auxiliary_loss = (
            self.monotonicity_weight * monotonicity_loss
            + self.mismatch_loss_weight * mismatch_loss
        )
        self.last_auxiliary_losses = {
            "monotonicity": monotonicity_loss.detach(),
            "mismatch": mismatch_loss.detach(),
        }
        self.last_pathology_logits = components["pathology_logits"]
        self.last_genomic_logits = components["genomic_logits"]
        self.last_availability = availability
        confidence = (
            components["quality_scores"][:, 0:1],
            components["quality_scores"][:, 1:2],
            components["conflict_scores"][:, 0:1],
            components["conflict_scores"][:, 1:2],
            components["js_divergence"],
        )
        return weights, confidence
