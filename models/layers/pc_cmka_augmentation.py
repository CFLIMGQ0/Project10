"""Calibration-curvature augmentation and Krylov safety controls."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.layers.pc_cmka_calibration import (
    CalibrationResult,
    EdgeDeformationDictionary,
)
from models.layers.pc_cmka_spectral import (
    ChebyshevMomentResponse,
    ReferenceSpectralOperator,
    normalized_patient_probe,
)

def build_krylov_basis(
    operator: ReferenceSpectralOperator,
    probe: torch.Tensor,
    weights: torch.Tensor,
    order: int,
    *,
    tolerance: float = 1e-7,
) -> torch.Tensor:
    """Build an orthonormal basis of ``span(u, S_hat u, ..., S_hat^R u)``."""

    if probe.ndim != 1 or probe.shape[0] != operator.num_nodes:
        raise ValueError("probe must be one unbatched node vector")
    if order < 0 or tolerance <= 0:
        raise ValueError("invalid Krylov settings")
    columns = [probe]
    current = probe
    for _ in range(order):
        current = operator.apply_scaled(current, weights)
        columns.append(current)
    matrix = torch.stack(columns, dim=1)
    # SVD handles exactly/near dependent Krylov vectors without returning
    # arbitrary extra QR columns for a rank-deficient matrix.
    left, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
    if singular_values.numel() == 0:
        return matrix[:, :0]
    threshold = tolerance * singular_values.max().clamp_min(tolerance)
    rank = int((singular_values > threshold).sum().item())
    return left[:, :rank]


def krylov_operator_error(
    operator: ReferenceSpectralOperator,
    base_weights: torch.Tensor,
    augmented_weights: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Frobenius upper bound of the operator difference on a Krylov basis."""

    if basis.ndim != 2 or basis.shape[0] != operator.num_nodes:
        raise ValueError("basis must be [nodes, krylov_rank]")
    if basis.shape[1] == 0:
        return base_weights.new_zeros(())
    difference = operator.apply_scaled(basis, augmented_weights) - operator.apply_scaled(
        basis, base_weights
    )
    return torch.linalg.matrix_norm(difference, ord="fro")


@dataclass
class AntitheticAugmentationResult:
    base_weights: torch.Tensor
    positive_weights: torch.Tensor
    negative_weights: torch.Tensor
    relative_perturbation: torch.Tensor
    hessian_diagonal: torch.Tensor
    hessian_eigenvalues: torch.Tensor
    infinity_scale: torch.Tensor
    krylov_scale: torch.Tensor
    krylov_error: torch.Tensor
    finite: torch.Tensor


class CalibrationUncertaintyAugmentor(nn.Module):
    """Antithetic graph views sampled from low-dimensional calibration curvature."""

    VALID_MODES = {"exact", "diagonal", "low_rank"}

    def __init__(
        self,
        moment_response: ChebyshevMomentResponse,
        dictionary: EdgeDeformationDictionary,
        *,
        beta: float = 1e-2,
        damping: float = 1e-5,
        mode: str = "diagonal",
        low_rank: int = 2,
        xi_max: float = 0.2,
        krylov_order: int | None = None,
        krylov_eta: float = 0.1,
        use_krylov_safety: bool = True,
        detach_direction: bool = True,
    ) -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"unknown Hessian mode: {mode}")
        if beta < 0 or damping <= 0:
            raise ValueError("beta must be non-negative and damping positive")
        if low_rank < 1:
            raise ValueError("low_rank must be positive")
        if not 0 < xi_max < 1:
            raise ValueError("xi_max must be strictly between zero and one")
        if krylov_eta <= 0:
            raise ValueError("krylov_eta must be positive")
        if dictionary.num_edges != moment_response.operator.num_edges:
            raise ValueError("dictionary and graph edge dimensions differ")
        object.__setattr__(self, "moment_response", moment_response)
        object.__setattr__(self, "dictionary", dictionary)
        self.beta = float(beta)
        self.damping = float(damping)
        self.mode = mode
        self.low_rank = int(low_rank)
        self.xi_max = float(xi_max)
        self.krylov_order = (
            moment_response.order if krylov_order is None else int(krylov_order)
        )
        self.krylov_eta = float(krylov_eta)
        self.use_krylov_safety = bool(use_krylov_safety)
        self.detach_direction = bool(detach_direction)

    def _moment_jacobian(
        self, expression: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        dictionary_matrix = self.dictionary.matrix.detach()
        with torch.enable_grad():
            local = coefficients.detach().requires_grad_(True)

            def response(local_coefficients: torch.Tensor) -> torch.Tensor:
                log_deformation = local_coefficients @ dictionary_matrix.t()
                weights = self.moment_response.operator.patient_weights(
                    log_deformation
                )
                return self.moment_response(expression.detach(), weights)

            jacobian = torch.autograd.functional.jacobian(
                response, local, create_graph=False, vectorize=True
            )
        return jacobian.detach()

    def _inverse_sqrt_direction(
        self,
        hessian: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
        eigenvalues = eigenvalues.clamp_min(self.damping)
        if self.mode == "diagonal":
            direction = noise / hessian.diagonal().clamp_min(self.damping).sqrt()
        elif self.mode == "exact":
            direction = eigenvectors @ (
                (eigenvectors.t() @ noise) / eigenvalues.sqrt()
            )
        else:
            rank = min(self.low_rank, hessian.shape[0])
            selected_values = eigenvalues[-rank:]
            selected_vectors = eigenvectors[:, -rank:]
            residual_value = hessian.new_tensor(self.beta + self.damping)
            base = noise / residual_value.sqrt()
            projection = selected_vectors.t() @ noise
            correction = selected_vectors @ (
                projection
                * (selected_values.rsqrt() - residual_value.rsqrt())
            )
            direction = base + correction
        return direction, eigenvalues

    def forward(
        self,
        expression: torch.Tensor,
        calibration: CalibrationResult,
        *,
        generator: torch.Generator | None = None,
    ) -> AntitheticAugmentationResult:
        if expression.ndim == 1:
            expression = expression.unsqueeze(0)
        batch_size = expression.shape[0]
        base_weights = calibration.weights
        if base_weights.shape != (batch_size, self.dictionary.num_edges):
            raise ValueError("calibration and expression batch sizes differ")

        if not self.training:
            zeros_edges = torch.zeros_like(base_weights)
            zeros_batch = base_weights.new_zeros(batch_size)
            zeros_hessian = base_weights.new_zeros(batch_size, self.dictionary.rank)
            return AntitheticAugmentationResult(
                base_weights=base_weights,
                positive_weights=base_weights,
                negative_weights=base_weights,
                relative_perturbation=zeros_edges,
                hessian_diagonal=zeros_hessian,
                hessian_eigenvalues=zeros_hessian,
                infinity_scale=torch.ones_like(zeros_batch),
                krylov_scale=torch.ones_like(zeros_batch),
                krylov_error=zeros_batch,
                finite=torch.isfinite(base_weights).all(),
            )

        perturbations = []
        hessian_diagonals = []
        hessian_eigenvalues = []
        infinity_scales = []
        krylov_scales = []
        krylov_errors = []
        dictionary_matrix = (
            self.dictionary.matrix.detach()
            if self.detach_direction
            else self.dictionary.matrix
        )
        for patient in range(batch_size):
            jacobian = self._moment_jacobian(
                expression[patient], calibration.coefficients[patient]
            )
            precision = calibration.precision[patient].detach()
            hessian = (
                jacobian.t() @ (precision[:, None] * jacobian)
                + (self.beta + self.damping)
                * torch.eye(
                    self.dictionary.rank,
                    device=jacobian.device,
                    dtype=jacobian.dtype,
                )
            )
            noise = torch.randn(
                self.dictionary.rank,
                device=hessian.device,
                dtype=hessian.dtype,
                generator=generator,
            )
            direction, eigenvalues = self._inverse_sqrt_direction(hessian, noise)
            perturbation = dictionary_matrix @ direction
            maximum = perturbation.abs().max().clamp_min(1e-12)
            infinity_scale = torch.clamp(self.xi_max / maximum, max=1.0)
            perturbation = perturbation * infinity_scale

            if self.use_krylov_safety:
                probe = normalized_patient_probe(expression[patient].detach())
                basis = build_krylov_basis(
                    self.moment_response.operator,
                    probe,
                    base_weights[patient].detach(),
                    self.krylov_order,
                )
                candidate = base_weights[patient].detach() * (
                    1.0 + perturbation.detach()
                )
                error = krylov_operator_error(
                    self.moment_response.operator,
                    base_weights[patient].detach(),
                    candidate,
                    basis,
                )
                krylov_scale = torch.clamp(
                    self.krylov_eta / error.clamp_min(1e-12), max=1.0
                )
                perturbation = perturbation * krylov_scale
                final_candidate = base_weights[patient].detach() * (
                    1.0 + perturbation.detach()
                )
                final_error = krylov_operator_error(
                    self.moment_response.operator,
                    base_weights[patient].detach(),
                    final_candidate,
                    basis,
                )
            else:
                krylov_scale = perturbation.new_ones(())
                final_error = perturbation.new_zeros(())

            perturbations.append(perturbation)
            hessian_diagonals.append(hessian.diagonal())
            hessian_eigenvalues.append(eigenvalues)
            infinity_scales.append(infinity_scale)
            krylov_scales.append(krylov_scale)
            krylov_errors.append(final_error)

        relative_perturbation = torch.stack(perturbations)
        positive = base_weights * (1.0 + relative_perturbation)
        negative = base_weights * (1.0 - relative_perturbation)
        finite = torch.stack(
            [
                torch.isfinite(relative_perturbation).all(),
                torch.isfinite(positive).all(),
                torch.isfinite(negative).all(),
            ]
        ).all()
        return AntitheticAugmentationResult(
            base_weights=base_weights,
            positive_weights=positive,
            negative_weights=negative,
            relative_perturbation=relative_perturbation,
            hessian_diagonal=torch.stack(hessian_diagonals),
            hessian_eigenvalues=torch.stack(hessian_eigenvalues),
            infinity_scale=torch.stack(infinity_scales),
            krylov_scale=torch.stack(krylov_scales),
            krylov_error=torch.stack(krylov_errors),
            finite=finite,
        )
class ControlGraphAugmentor(nn.Module):
    """Random-drop/independent/effective-resistance ablation controls.

    These views are intentionally not claimed to satisfy the antithetic-center
    invariant.  A small positive factor replaces a hard zero so all graph
    operators remain finite and positive during fair numerical comparisons.
    """

    VALID_MODES = {"random_drop", "independent_random", "effective_resistance"}

    def __init__(
        self,
        operator: ReferenceSpectralOperator,
        *,
        mode: str,
        drop_probability: float = 0.1,
        minimum_factor: float = 0.05,
    ) -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"unknown control augmentation: {mode}")
        if not 0 <= drop_probability < 1 or not 0 < minimum_factor <= 1:
            raise ValueError("invalid control augmentation settings")
        object.__setattr__(self, "operator", operator)
        self.mode = mode
        self.drop_probability = float(drop_probability)
        self.minimum_factor = float(minimum_factor)
        if mode == "effective_resistance":
            with torch.no_grad():
                laplacian = operator.dense_matrix()
                pseudoinverse = torch.linalg.pinv(laplacian, hermitian=True)
                source, target = operator.edge_index
                resistance = (
                    pseudoinverse[source, source]
                    + pseudoinverse[target, target]
                    - 2.0 * pseudoinverse[source, target]
                ).clamp_min(0.0)
                leverage = operator.prior_weights * resistance
                leverage = leverage / leverage.mean().clamp_min(1e-8)
            self.register_buffer("resistance_score", leverage)
        else:
            self.register_buffer(
                "resistance_score", torch.ones_like(operator.prior_weights)
            )

    def _view(
        self,
        base_weights: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        probability = self.drop_probability * self.resistance_score
        probability = probability.clamp(0.0, 0.95).expand_as(base_weights)
        random = torch.rand(
            base_weights.shape,
            device=base_weights.device,
            dtype=base_weights.dtype,
            generator=generator,
        )
        factor = torch.where(
            random < probability,
            base_weights.new_tensor(self.minimum_factor),
            base_weights.new_tensor(1.0),
        )
        return base_weights * factor

    def forward(
        self,
        base_weights: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> AntitheticAugmentationResult:
        batch_size, _ = base_weights.shape
        if not self.training:
            positive = negative = base_weights
        elif self.mode == "random_drop":
            positive = self._view(base_weights, generator)
            negative = base_weights
        else:
            positive = self._view(base_weights, generator)
            negative = self._view(base_weights, generator)
        zeros_rank = base_weights.new_zeros(batch_size, 1)
        zeros_batch = base_weights.new_zeros(batch_size)
        relative = positive / base_weights.clamp_min(1e-12) - 1.0
        return AntitheticAugmentationResult(
            base_weights=base_weights,
            positive_weights=positive,
            negative_weights=negative,
            relative_perturbation=relative,
            hessian_diagonal=zeros_rank,
            hessian_eigenvalues=zeros_rank,
            infinity_scale=torch.ones_like(zeros_batch),
            krylov_scale=torch.ones_like(zeros_batch),
            krylov_error=zeros_batch,
            finite=torch.stack(
                [
                    torch.isfinite(positive).all(),
                    torch.isfinite(negative).all(),
                    (positive > 0).all(),
                    (negative > 0).all(),
                ]
            ).all(),
        )
