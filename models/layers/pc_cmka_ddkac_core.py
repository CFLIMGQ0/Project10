"""Six-group PC-CMKA-DDKAC encoder and DD-KAC shared-route integration."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.pc_cmka_augmentation import (
    AntitheticAugmentationResult,
    CalibrationUncertaintyAugmentor,
    ControlGraphAugmentor,
)
from models.layers.pc_cmka_calibration import (
    CalibrationResult,
    DifferentiableMomentSolver,
    DirectPatientEdgeGate,
    EdgeDeformationDictionary,
    MomentTargetNetwork,
)
from models.layers.pc_cmka_spectral import (
    ChebyshevMomentResponse,
    ReferenceSpectralOperator,
    chebyshev_recurrence,
)

@dataclass
class SharedRoutePathwayResult:
    base_token: torch.Tensor
    positive_token: torch.Tensor
    negative_token: torch.Tensor
    route_logits: torch.Tensor
    route: torch.Tensor
    positive_route: torch.Tensor
    negative_route: torch.Tensor
    gate: torch.Tensor
    positive_gate: torch.Tensor
    negative_gate: torch.Tensor
    base_structure: torch.Tensor
    positive_structure: torch.Tensor
    negative_structure: torch.Tensor


class SharedRouteDDKACPathway(nn.Module):
    """DD-KAC value/structure encoder with one base-graph route for all views."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        operator: ReferenceSpectralOperator,
        *,
        order: int = 2,
        value_centers: int = 8,
    ) -> None:
        super().__init__()
        if input_dim != operator.num_nodes:
            raise ValueError("input_dim and reference graph node count differ")
        if min(output_dim, order, value_centers) < 1:
            raise ValueError("invalid DD-KAC dimensions")
        object.__setattr__(self, "operator", operator)
        self.order = int(order)
        self.value_linear = nn.Parameter(torch.tensor(1.0))
        self.value_coefficients = nn.Parameter(torch.zeros(value_centers))
        self.register_buffer(
            "value_centers", torch.linspace(-1.0, 1.0, value_centers)
        )
        self.value_projection = nn.Linear(input_dim, output_dim)
        self.structure_projection = nn.Linear(input_dim, output_dim)
        self.residual_projection = nn.Linear(input_dim, output_dim)
        self.router = nn.Sequential(
            nn.Linear(5, 32), nn.GELU(), nn.Linear(32, order + 1)
        )
        self.gate = nn.Linear(5, 1)
        self.norm = nn.LayerNorm(output_dim)

    @staticmethod
    def _as_batch(inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 1:
            return inputs.unsqueeze(0)
        if inputs.ndim == 2:
            return inputs
        raise ValueError("pathway inputs must be [nodes] or [batch, nodes]")

    def _stats(
        self, inputs: torch.Tensor, base_weights: torch.Tensor
    ) -> torch.Tensor:
        centered = inputs - inputs.mean(dim=1, keepdim=True)
        filtered = self.operator.matvec(centered, base_weights)
        energy = (centered * filtered).sum(dim=1) / centered.square().sum(
            dim=1
        ).clamp_min(1e-6)
        sparsity = (inputs.abs() < 1e-6).float().mean(dim=1)
        burden = (inputs.abs() > 0.5).float().mean(dim=1)
        return torch.stack(
            (
                inputs.mean(dim=1),
                inputs.std(dim=1, unbiased=False),
                sparsity,
                energy,
                burden,
            ),
            dim=1,
        )

    def _value_features(self, inputs: torch.Tensor) -> torch.Tensor:
        widths = 2.0 / max(self.value_centers.numel() - 1, 1)
        basis = torch.exp(
            -0.5 * ((inputs.unsqueeze(-1) - self.value_centers) / widths).square()
        )
        transformed = self.value_linear * inputs + torch.einsum(
            "bik,k->bi", basis, self.value_coefficients
        )
        return self.value_projection(transformed)

    def _structure_features(
        self,
        centered: torch.Tensor,
        weights: torch.Tensor,
        route: torch.Tensor,
    ) -> torch.Tensor:
        recurrence = chebyshev_recurrence(
            self.operator, centered, weights, self.order
        )
        filtered = torch.einsum("br,brn->bn", route, recurrence)
        return self.structure_projection(filtered)

    def _combine(
        self,
        value: torch.Tensor,
        structure: torch.Tensor,
        residual: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        return self.norm(gate * value + (1.0 - gate) * structure + residual)

    def forward(
        self,
        inputs: torch.Tensor,
        base_weights: torch.Tensor,
        positive_weights: torch.Tensor | None = None,
        negative_weights: torch.Tensor | None = None,
        *,
        shared_route: bool = True,
    ) -> SharedRoutePathwayResult:
        inputs = self._as_batch(inputs)
        if base_weights.ndim == 1:
            base_weights = base_weights.unsqueeze(0).expand(inputs.shape[0], -1)
        if positive_weights is None:
            positive_weights = base_weights
        if negative_weights is None:
            negative_weights = base_weights
        stats = self._stats(inputs, base_weights)
        # This is the only route computation.  Both augmented structure views
        # consume this exact tensor rather than rerunning the router.
        route_logits = self.router(stats)
        route = torch.softmax(route_logits, dim=1)
        gate = torch.sigmoid(self.gate(stats))
        positive_stats = self._stats(inputs, positive_weights)
        negative_stats = self._stats(inputs, negative_weights)
        positive_gate = torch.sigmoid(self.gate(positive_stats))
        negative_gate = torch.sigmoid(self.gate(negative_stats))
        if shared_route:
            positive_route = negative_route = route
        else:
            positive_route = torch.softmax(self.router(positive_stats), dim=1)
            negative_route = torch.softmax(self.router(negative_stats), dim=1)
        value = self._value_features(inputs)
        residual = self.residual_projection(inputs)
        centered = inputs - inputs.mean(dim=1, keepdim=True)
        base_structure = self._structure_features(
            centered, base_weights, route
        )
        positive_structure = self._structure_features(
            centered, positive_weights, positive_route
        )
        negative_structure = self._structure_features(
            centered, negative_weights, negative_route
        )
        return SharedRoutePathwayResult(
            base_token=self._combine(value, base_structure, residual, gate),
            positive_token=self._combine(
                value, positive_structure, residual, positive_gate
            ),
            negative_token=self._combine(
                value, negative_structure, residual, negative_gate
            ),
            route_logits=route_logits,
            route=route,
            positive_route=positive_route,
            negative_route=negative_route,
            gate=gate,
            positive_gate=positive_gate,
            negative_gate=negative_gate,
            base_structure=base_structure,
            positive_structure=positive_structure,
            negative_structure=negative_structure,
        )


def negative_free_consistency_loss(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    variance_weight: float = 0.0,
    covariance_weight: float = 0.0,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Cosine invariance with optional VICReg variance/covariance terms."""

    if positive.shape != negative.shape or positive.ndim < 2:
        raise ValueError("positive and negative views must have the same shape")
    if min(variance_weight, covariance_weight) < 0 or epsilon <= 0:
        raise ValueError("invalid consistency loss settings")
    flat_positive = positive.reshape(-1, positive.shape[-1])
    flat_negative = negative.reshape(-1, negative.shape[-1])
    invariance = (
        1.0 - F.cosine_similarity(flat_positive, flat_negative, dim=1)
    ).mean()
    loss = invariance
    if variance_weight > 0:
        std_positive = torch.sqrt(
            flat_positive.var(dim=0, unbiased=False) + epsilon
        )
        std_negative = torch.sqrt(
            flat_negative.var(dim=0, unbiased=False) + epsilon
        )
        variance = 0.5 * (
            F.relu(1.0 - std_positive).mean()
            + F.relu(1.0 - std_negative).mean()
        )
        loss = loss + variance_weight * variance
    if covariance_weight > 0 and flat_positive.shape[0] > 1:
        def covariance_penalty(values: torch.Tensor) -> torch.Tensor:
            centered = values - values.mean(dim=0, keepdim=True)
            covariance = centered.t() @ centered / (values.shape[0] - 1)
            off_diagonal = covariance - torch.diag(covariance.diagonal())
            return off_diagonal.square().sum() / covariance.shape[0]

        loss = loss + covariance_weight * 0.5 * (
            covariance_penalty(flat_positive)
            + covariance_penalty(flat_negative)
        )
    return loss


class IdentifiabilityTangentRegularizer(nn.Module):
    """Penalize alignment of graph-calibration and frequency-route tangents.

    ``randomized`` uses random Jacobian-vector products and is the default
    scalable estimator.  ``full`` explicitly builds both small output-space
    Jacobians and is intended only for diagnostics/ablations.
    """

    VALID_MODES = {"off", "randomized", "full"}

    def __init__(
        self,
        moment_response: ChebyshevMomentResponse,
        dictionary: EdgeDeformationDictionary,
        ddkac: SharedRouteDDKACPathway,
        *,
        mode: str = "randomized",
        random_probes: int = 1,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"unknown identifiability mode: {mode}")
        if random_probes < 1 or epsilon <= 0:
            raise ValueError("invalid identifiability settings")
        object.__setattr__(self, "moment_response", moment_response)
        object.__setattr__(self, "dictionary", dictionary)
        object.__setattr__(self, "ddkac", ddkac)
        self.mode = mode
        self.random_probes = int(random_probes)
        self.epsilon = float(epsilon)

    def _structure_function(
        self,
        expression: torch.Tensor,
        coefficients: torch.Tensor,
        route_logits: torch.Tensor,
    ) -> torch.Tensor:
        log_deformation = self.dictionary(coefficients)
        weights = self.moment_response.operator.patient_weights(log_deformation)
        centered = expression - expression.mean(dim=1, keepdim=True)
        route = torch.softmax(route_logits, dim=1)
        return self.ddkac._structure_features(
            centered, weights, route
        )
        return offset

    def _randomized(
        self,
        expression: torch.Tensor,
        coefficients: torch.Tensor,
        route_logits: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        for _ in range(self.random_probes):
            coefficient_direction = torch.randn_like(coefficients)
            route_direction = torch.randn_like(route_logits)
            _, graph_tangent = torch.autograd.functional.jvp(
                lambda value: self._structure_function(
                    expression, value, route_logits
                ),
                coefficients,
                coefficient_direction,
                create_graph=True,
            )
            _, route_tangent = torch.autograd.functional.jvp(
                lambda value: self._structure_function(
                    expression, coefficients, value
                ),
                route_logits,
                route_direction,
                create_graph=True,
            )
            graph_flat = graph_tangent.flatten()
            route_flat = route_tangent.flatten()
            numerator = torch.dot(graph_flat, route_flat).square()
            denominator = (
                graph_flat.square().sum()
                * route_flat.square().sum()
            ).clamp_min(self.epsilon)
            losses.append(numerator / denominator)
        return torch.stack(losses).mean()

    def _full(
        self,
        expression: torch.Tensor,
        coefficients: torch.Tensor,
        route_logits: torch.Tensor,
    ) -> torch.Tensor:
        graph_jacobian = torch.autograd.functional.jacobian(
            lambda value: self._structure_function(
                expression, value, route_logits
            ),
            coefficients,
            create_graph=True,
            vectorize=True,
        )
        graph_matrix = graph_jacobian.reshape(
            expression.shape[0] * self.ddkac.structure_projection.out_features,
            -1,
        )
        route_jacobian = torch.autograd.functional.jacobian(
            lambda value: self._structure_function(
                expression, coefficients, value
            ),
            route_logits,
            create_graph=True,
            vectorize=True,
        )
        route_matrix = route_jacobian.reshape(
            expression.shape[0] * self.ddkac.structure_projection.out_features,
            -1,
        )

        def tangent_basis(matrix: torch.Tensor) -> torch.Tensor:
            left, singular_values, _ = torch.linalg.svd(
                matrix, full_matrices=False
            )
            if singular_values.numel() == 0:
                return left[:, :0]
            threshold = self.epsilon * singular_values.max().clamp_min(
                self.epsilon
            )
            rank = int((singular_values > threshold).sum().item())
            return left[:, :rank]

        graph_basis = tangent_basis(graph_matrix)
        route_basis = tangent_basis(route_matrix)
        if graph_basis.shape[1] == 0 or route_basis.shape[1] == 0:
            return coefficients.new_zeros(())
        overlap = graph_basis.t() @ route_basis
        normalizer = min(
            graph_basis.shape[1], route_basis.shape[1]
        )
        return overlap.square().sum() / max(normalizer, 1)

    def forward(
        self,
        expression: torch.Tensor,
        coefficients: torch.Tensor,
        route_logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "off":
            return expression.new_zeros(())
        if expression.ndim == 1:
            expression = expression.unsqueeze(0)
        if coefficients.ndim == 1:
            coefficients = coefficients.unsqueeze(0)
        if route_logits.ndim == 1:
            route_logits = route_logits.unsqueeze(0)
        if self.mode == "randomized":
            return self._randomized(expression, coefficients, route_logits)
        return self._full(expression, coefficients, route_logits)
