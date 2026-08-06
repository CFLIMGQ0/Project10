"""Calibration-uncertainty graph views and Krylov safety scaling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .calibration import CalibrationResult, ChebyshevInverseCalibrator
from .spectral import ReferenceSpectralOperator


@dataclass
class GraphViews:
    positive: torch.Tensor
    negative: torch.Tensor
    positive_delta: torch.Tensor
    negative_delta: torch.Tensor
    hessian_summary: torch.Tensor
    scale: torch.Tensor
    krylov_before: torch.Tensor
    krylov_after: torch.Tensor
    is_antithetic: bool


class GraphViewAugmenter(nn.Module):
    """Generate uncertainty-aligned views in the low-dimensional ``a`` space."""

    def __init__(
        self,
        operator: ReferenceSpectralOperator,
        calibrator: ChebyshevInverseCalibrator,
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.operator = operator
        self.calibrator = calibrator
        self.settings = dict(config["augmentation"])
        self.order = int(config["spectral"]["moment_order"])
        if str(self.settings.get("mode", "hessian_antithetic")) == "effective_resistance":
            probabilities = self._compute_effective_resistance_probabilities()
        else:
            probabilities = torch.empty(0, dtype=operator.prior_weights.dtype)
        self.register_buffer("effective_resistance_probabilities", probabilities)

    def _moment_jacobian(
        self,
        coefficients: torch.Tensor,
        probes: torch.Tensor,
    ) -> torch.Tensor:
        """Return a batched low-dimensional Jacobian [B,R,K]."""

        if not coefficients.requires_grad:
            coefficients = coefficients.detach().requires_grad_(True)
        with torch.enable_grad():
            moments = self.calibrator.moments_from_coefficients(coefficients, probes)
            rows = []
            for index in range(moments.shape[1]):
                gradient = torch.autograd.grad(
                    moments[:, index].sum(),
                    coefficients,
                    create_graph=self.training,
                    retain_graph=True,
                )[0]
                rows.append(gradient)
        return torch.stack(rows, dim=1)

    def _uncertainty_direction(
        self,
        calibration: CalibrationResult,
        probes: torch.Tensor,
        isotropic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, rank = calibration.coefficients.shape
        noise = torch.randn(
            batch, rank, device=probes.device, dtype=probes.dtype
        )
        if isotropic:
            return noise, torch.ones(batch, rank, device=probes.device, dtype=probes.dtype)

        jacobian = self._moment_jacobian(calibration.coefficients, probes)
        weighted = jacobian * calibration.precision.sqrt().unsqueeze(-1)
        identity = torch.eye(rank, device=probes.device, dtype=probes.dtype)
        hessian = weighted.transpose(1, 2) @ weighted
        hessian = hessian + float(self.settings["damping"]) * identity.unsqueeze(0)
        mode = str(self.settings["hessian_mode"])
        if mode == "diagonal":
            diagonal = torch.diagonal(hessian, dim1=-2, dim2=-1).clamp_min(1e-6)
            direction = noise * diagonal.rsqrt()
            summary = diagonal
        elif mode == "exact":
            eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
            inverse_root = eigenvectors @ torch.diag_embed(
                eigenvalues.clamp_min(1e-6).rsqrt()
            ) @ eigenvectors.transpose(1, 2)
            direction = (inverse_root @ noise.unsqueeze(-1)).squeeze(-1)
            summary = eigenvalues
        elif mode == "low_rank":
            eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
            retained = min(int(self.settings["hessian_low_rank"]), rank)
            values = eigenvalues[:, :retained].clamp_min(1e-6)
            vectors = eigenvectors[:, :, :retained]
            projected = (vectors.transpose(1, 2) @ noise.unsqueeze(-1)).squeeze(-1)
            low = (vectors @ (projected * values.rsqrt()).unsqueeze(-1)).squeeze(-1)
            residual = noise - (vectors @ projected.unsqueeze(-1)).squeeze(-1)
            direction = low + residual / float(self.settings["damping"]) ** 0.5
            summary = eigenvalues
        else:
            raise ValueError(f"unknown Hessian mode: {mode}")
        return direction, summary

    def _edge_direction(self, coefficient_direction: torch.Tensor) -> torch.Tensor:
        edge = self.calibrator.dictionary(coefficient_direction)
        maximum = edge.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        return edge * torch.clamp(float(self.settings["xi_max"]) / maximum, max=1.0)

    def _krylov_error(
        self,
        base: torch.Tensor,
        view: torch.Tensor,
        probe: torch.Tensor,
    ) -> torch.Tensor:
        values = []
        for batch_index in range(base.shape[0]):
            basis = self.operator.krylov_basis(
                base[batch_index], probe[batch_index], self.order
            )
            # Each basis vector is treated as one signal in the batch.
            base_action = self.operator.apply_operator(
                base[batch_index], basis.t(), scaled=True
            )
            view_action = self.operator.apply_operator(
                view[batch_index], basis.t(), scaled=True
            )
            values.append((view_action - base_action).norm(p="fro"))
        return torch.stack(values)

    def _krylov_scale(
        self,
        base: torch.Tensor,
        positive_delta: torch.Tensor,
        negative_delta: torch.Tensor,
        probes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        positive = base * (1.0 + positive_delta)
        negative = base * (1.0 + negative_delta)
        before_pos = self._krylov_error(base, positive, probes)
        before_neg = self._krylov_error(base, negative, probes)
        before = torch.maximum(before_pos, before_neg)
        if not bool(self.settings.get("krylov_enabled", True)):
            scale = torch.ones_like(before)
            return positive_delta, negative_delta, scale, before
        eta = float(self.settings["krylov_eta"])
        scale = torch.clamp(eta / before.clamp_min(1e-8), max=1.0)
        positive_delta = positive_delta * scale.unsqueeze(1)
        negative_delta = negative_delta * scale.unsqueeze(1)
        positive = base * (1.0 + positive_delta)
        negative = base * (1.0 + negative_delta)
        after = torch.maximum(
            self._krylov_error(base, positive, probes),
            self._krylov_error(base, negative, probes),
        )
        return positive_delta, negative_delta, scale, after

    def _compute_effective_resistance_probabilities(self) -> torch.Tensor:
        # Exact classical control. It depends only on the reference graph and
        # is cached at construction instead of recomputed for every patient.
        laplacian = self.operator.dense_matrix(self.operator.prior_weights)
        pseudoinverse = torch.linalg.pinv(laplacian)
        source, target = self.operator.edge_index
        resistance = (
            pseudoinverse[source, source]
            + pseudoinverse[target, target]
            - 2.0 * pseudoinverse[source, target]
        ).clamp_min(0.0)
        leverage = self.operator.prior_weights * resistance
        normalized = leverage / leverage.max().clamp_min(1e-8)
        minimum = float(self.settings.get("minimum_probability", 0.25))
        return (minimum + (1.0 - minimum) * normalized).clamp(max=1.0)

    def _effective_resistance_probabilities(self) -> torch.Tensor:
        if self.effective_resistance_probabilities.numel() == 0:
            raise RuntimeError("effective-resistance probabilities were not initialised")
        return self.effective_resistance_probabilities

    def forward(
        self,
        calibration: CalibrationResult,
        probes: torch.Tensor,
    ) -> GraphViews:
        base = calibration.weights
        zeros = torch.zeros_like(base)
        ones = torch.ones(base.shape[0], device=base.device, dtype=base.dtype)
        mode = str(self.settings.get("mode", "hessian_antithetic"))
        if mode == "off" or (bool(self.settings.get("train_only", True)) and not self.training):
            return GraphViews(base, base, zeros, zeros, zeros[:, :1], ones, zeros[:, 0], zeros[:, 0], True)

        if mode in {"hessian_antithetic", "isotropic_antithetic"}:
            direction, hessian = self._uncertainty_direction(
                calibration, probes, isotropic=mode == "isotropic_antithetic"
            )
            edge = self._edge_direction(direction)
            positive_delta, negative_delta = edge, -edge
            antithetic = True
        elif mode == "independent_random":
            first = torch.randn_like(calibration.coefficients)
            second = torch.randn_like(calibration.coefficients)
            positive_delta = self._edge_direction(first)
            negative_delta = self._edge_direction(second)
            hessian = torch.ones_like(calibration.coefficients)
            antithetic = False
        elif mode in {"bernoulli", "effective_resistance"}:
            if mode == "effective_resistance":
                probability = self._effective_resistance_probabilities()
            else:
                keep = float(self.settings.get("bernoulli_keep", 0.8))
                probability = torch.full_like(self.operator.prior_weights, keep)
            probability = probability.unsqueeze(0).expand_as(base)
            positive = base * torch.bernoulli(probability) / probability.clamp_min(1e-6)
            negative = base * torch.bernoulli(probability) / probability.clamp_min(1e-6)
            positive_delta = positive / base.clamp_min(1e-8) - 1.0
            negative_delta = negative / base.clamp_min(1e-8) - 1.0
            hessian = probability
            antithetic = False
        else:
            raise ValueError(f"unknown augmentation mode: {mode}")

        before = torch.maximum(
            self._krylov_error(base, base * (1.0 + positive_delta), probes),
            self._krylov_error(base, base * (1.0 + negative_delta), probes),
        )
        positive_delta, negative_delta, scale, after = self._krylov_scale(
            base, positive_delta, negative_delta, probes
        )
        positive = (base * (1.0 + positive_delta)).clamp_min(1e-8)
        negative = (base * (1.0 + negative_delta)).clamp_min(1e-8)
        return GraphViews(
            positive=positive,
            negative=negative,
            positive_delta=positive_delta,
            negative_delta=negative_delta,
            hessian_summary=hessian,
            scale=scale,
            krylov_before=before,
            krylov_after=after,
            is_antithetic=antithetic,
        )
