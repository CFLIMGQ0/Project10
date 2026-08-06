"""Patient-conditioned Chebyshev-moment inverse graph calibration."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral import ReferenceSpectralOperator


@dataclass
class CalibrationResult:
    weights: torch.Tensor
    coefficients: torch.Tensor
    rho: torch.Tensor
    target_moments: torch.Tensor
    actual_moments: torch.Tensor
    baseline_moments: torch.Tensor
    precision: torch.Tensor
    moment_loss: torch.Tensor
    trust_loss: torch.Tensor
    dictionary_loss: torch.Tensor
    convergence: torch.Tensor
    solve_seconds: float


class PatientMomentTarget(nn.Module):
    """Predict only target response offsets and positive precision values."""

    def __init__(
        self,
        input_dim: int,
        order: int,
        hidden_dim: int,
        max_offset: float,
        min_precision: float,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.offset = nn.Linear(hidden_dim, order)
        self.precision = nn.Linear(hidden_dim, order)
        self.max_offset = float(max_offset)
        self.min_precision = float(min_precision)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(inputs)
        offset = self.max_offset * torch.tanh(self.offset(hidden))
        precision = F.softplus(self.precision(hidden)) + self.min_precision
        return offset, precision, hidden


class EdgeDeformationDictionary(nn.Module):
    """Weighted-orthogonal low-rank log-edge deformation dictionary."""

    def __init__(self, prior_weights: torch.Tensor, rank: int) -> None:
        super().__init__()
        edge_count = int(prior_weights.numel())
        rank = min(int(rank), edge_count)
        if rank <= 0:
            raise ValueError("dictionary rank must be positive")
        with torch.no_grad():
            raw = torch.randn(edge_count, rank)
            weighted = raw * prior_weights.sqrt().unsqueeze(1)
            orthogonal, _ = torch.linalg.qr(weighted, mode="reduced")
            basis = orthogonal / prior_weights.sqrt().clamp_min(1e-6).unsqueeze(1)
        self.basis = nn.Parameter(basis)
        self.register_buffer("prior_weights", prior_weights.detach().clone())

    @property
    def rank(self) -> int:
        return self.basis.shape[1]

    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        return coefficients @ self.basis.t()

    def orthogonality_loss(self) -> torch.Tensor:
        gram = self.basis.t() @ (self.prior_weights.unsqueeze(1) * self.basis)
        identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        return (gram - identity).square().mean()


class ChebyshevInverseCalibrator(nn.Module):
    """Recover patient graph weights from target Chebyshev moments.

    The epigraph variable rho can be eliminated exactly as
    ``rho=max(abs(Pa))``.  Keeping it in this derived form enforces the Word
    document's joint minimum-evidence trust region without a second MLP head.
    """

    def __init__(
        self,
        input_dim: int,
        operator: ReferenceSpectralOperator,
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.operator = operator
        self.order = int(config["spectral"]["moment_order"])
        target = config["target"]
        self.target_network = PatientMomentTarget(
            input_dim,
            self.order,
            int(target["hidden_dim"]),
            float(target["max_offset"]),
            float(target["min_precision"]),
        )
        self.dictionary = EdgeDeformationDictionary(
            operator.prior_weights, int(config["dictionary"]["rank"])
        )
        hidden_dim = int(target["hidden_dim"])
        self.direct_coefficients = nn.Linear(hidden_dim, self.dictionary.rank)
        self.direct_edges = nn.Linear(hidden_dim, operator.prior_weights.numel())
        self.settings = dict(config["solver"])
        self.calibration_mode = str(config.get("calibration_mode", "inverse"))
        self.trust_mode = str(config.get("trust_mode", "minimum"))

    def weights_from_coefficients(
        self,
        coefficients: torch.Tensor,
        limit: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_deformation = self.dictionary(coefficients)
        if limit is None:
            limit = float(self.settings["rho_max"])
        log_deformation = log_deformation.clamp(-limit, limit)
        weights = self.operator.prior_weights.unsqueeze(0) * torch.exp(log_deformation)
        return weights, log_deformation

    def moments_from_coefficients(
        self,
        coefficients: torch.Tensor,
        probes: torch.Tensor,
    ) -> torch.Tensor:
        weights, _ = self.weights_from_coefficients(coefficients)
        return self.operator.moments(weights, probes, self.order)

    def _project_coefficients(self, coefficients: torch.Tensor, limit: float) -> torch.Tensor:
        deformation = self.dictionary(coefficients)
        maximum = deformation.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        scale = torch.clamp(limit / maximum, max=1.0)
        return coefficients * scale

    def _objective(
        self,
        coefficients: torch.Tensor,
        probes: torch.Tensor,
        target: torch.Tensor,
        precision: torch.Tensor,
        include_minimum_rho: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actual = self.moments_from_coefficients(coefficients, probes)
        residual = actual - target
        moment = (precision * residual.square()).mean()
        deformation = self.dictionary(coefficients)
        rho = deformation.abs().amax(dim=1)
        objective = moment + float(self.settings["beta"]) * coefficients.square().mean()
        if include_minimum_rho:
            objective = objective + float(self.settings["lambda_rho"]) * rho.mean()
        return objective, actual, rho

    def _inverse_solve(
        self,
        probes: torch.Tensor,
        target: torch.Tensor,
        precision: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        differentiable = str(self.settings.get("mode", "differentiable")) == "differentiable"
        iterations = int(self.settings["iterations"])
        step_size = float(self.settings["step_size"])
        fixed = self.trust_mode == "fixed"
        limit = float(
            self.settings["fixed_rho"] if fixed else self.settings["rho_max"]
        )
        coefficients = torch.zeros(
            probes.shape[0],
            self.dictionary.rank,
            dtype=probes.dtype,
            device=probes.device,
            requires_grad=True,
        )
        convergence = []
        with torch.enable_grad():
            for _ in range(iterations):
                objective, _, _ = self._objective(
                    coefficients,
                    probes,
                    target,
                    precision,
                    include_minimum_rho=not fixed,
                )
                gradient = torch.autograd.grad(
                    objective,
                    coefficients,
                    create_graph=differentiable and self.training,
                    retain_graph=True,
                )[0]
                gradient = gradient.clamp(
                    -float(self.settings["gradient_clip"]),
                    float(self.settings["gradient_clip"]),
                )
                coefficients = self._project_coefficients(
                    coefficients - step_size * gradient, limit
                )
                convergence.append(objective.detach())
                if not differentiable:
                    coefficients = coefficients.detach().requires_grad_(True)
        if not self.training:
            coefficients = coefficients.detach()
        return coefficients, torch.stack(convergence)

    def forward(self, inputs: torch.Tensor, probes: torch.Tensor) -> CalibrationResult:
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        if probes.ndim == 1:
            probes = probes.unsqueeze(0)
        start = perf_counter()
        offset, precision, hidden = self.target_network(inputs)
        prior = self.operator.prior_weights.unsqueeze(0).expand(inputs.shape[0], -1)
        baseline = self.operator.moments(prior, probes, self.order)
        target = baseline + offset
        convergence = inputs.new_zeros(0)

        if self.calibration_mode == "fixed":
            coefficients = inputs.new_zeros((inputs.shape[0], self.dictionary.rank))
            weights = prior
        elif self.calibration_mode == "direct_coefficients":
            limit = float(self.settings["rho_max"])
            coefficients = limit * torch.tanh(self.direct_coefficients(hidden))
            coefficients = self._project_coefficients(coefficients, limit)
            weights, _ = self.weights_from_coefficients(coefficients, limit)
        elif self.calibration_mode == "direct_edges":
            limit = float(self.settings["rho_max"])
            log_deformation = limit * torch.tanh(self.direct_edges(hidden))
            weights = prior * torch.exp(log_deformation)
            # A projected least-squares coefficient is retained only for
            # diagnostics; the ordinary edge gate is not constrained to P.
            coefficients = torch.linalg.lstsq(
                self.dictionary.basis, log_deformation.t()
            ).solution.t()
        elif self.calibration_mode == "inverse":
            coefficients, convergence = self._inverse_solve(probes, target, precision)
            weights, _ = self.weights_from_coefficients(coefficients)
        else:
            raise ValueError(f"unknown calibration_mode: {self.calibration_mode}")

        actual = self.operator.moments(weights, probes, self.order)
        residual = actual - target
        moment_loss = (precision * residual.square()).mean()
        log_change = torch.log(weights / prior.clamp_min(1e-8))
        rho = log_change.abs().amax(dim=1)
        trust_loss = rho.mean()
        dictionary_loss = self.dictionary.orthogonality_loss()
        return CalibrationResult(
            weights=weights,
            coefficients=coefficients,
            rho=rho,
            target_moments=target,
            actual_moments=actual,
            baseline_moments=baseline,
            precision=precision,
            moment_loss=moment_loss,
            trust_loss=trust_loss,
            dictionary_loss=dictionary_loss,
            convergence=convergence,
            solve_seconds=perf_counter() - start,
        )
