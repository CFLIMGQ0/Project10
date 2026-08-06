"""PC-CMKA-DDKAC genomic modules.

This file is deliberately independent from the released DD-KAC path while the
new method is validated phase by phase.  In particular, the reference spectral
operator below uses an oriented edge-node incidence convention and never
normalizes with patient-specific degrees.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReferenceSpectralOperator(nn.Module):
    """Fixed-reference preconditioned weighted graph Laplacian.

    For an oriented edge-node incidence matrix ``B`` this module represents

    ``S(w) = D0^-1/2 B.T diag(w) B D0^-1/2``.

    ``D0`` is computed once from ``prior_weights`` and is shared by every
    patient.  Patient weights therefore change only the middle diagonal term;
    they never trigger patient-specific degree normalization.  The normal
    execution path is an edge-list matrix-vector product.  ``dense_matrix`` is
    provided only for tests and small-graph diagnostics.

    Node features are accepted as ``[nodes]``, ``[nodes, channels]``,
    ``[batch, nodes]`` (when batched weights are supplied), or
    ``[batch, nodes, channels]``.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        prior_weights: torch.Tensor,
        num_nodes: int,
        *,
        lambda_max: float | torch.Tensor | None = None,
        max_log_deformation: float = 0.5,
        degree_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        prior_weights = torch.as_tensor(prior_weights).float().flatten()
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_index.shape[1] != prior_weights.numel():
            raise ValueError("edge_index and prior_weights disagree on edge count")
        if num_nodes < 1:
            raise ValueError("num_nodes must be positive")
        if edge_index.numel() and (
            int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes
        ):
            raise ValueError("edge_index contains an out-of-range node")
        if edge_index.numel() and bool((edge_index[0] == edge_index[1]).any()):
            raise ValueError("oriented incidence support cannot contain self-loops")
        if not bool(torch.isfinite(prior_weights).all()):
            raise ValueError("prior_weights must be finite")
        if bool((prior_weights <= 0).any()):
            raise ValueError("prior_weights must be strictly positive")
        if max_log_deformation < 0:
            raise ValueError("max_log_deformation must be non-negative")
        if degree_epsilon <= 0:
            raise ValueError("degree_epsilon must be positive")

        degree = prior_weights.new_zeros(num_nodes)
        degree.index_add_(0, edge_index[0], prior_weights)
        degree.index_add_(0, edge_index[1], prior_weights)
        if bool((degree <= degree_epsilon).any()):
            isolated = torch.nonzero(degree <= degree_epsilon).flatten().tolist()
            raise ValueError(f"reference graph has isolated nodes: {isolated}")

        if lambda_max is None:
            # Since w/w0 <= exp(rho_max), S(w) is Loewner-bounded by
            # exp(rho_max) S(w0).  A normalized reference Laplacian has
            # lambda_max <= 2, hence this is a patient-independent bound.
            lambda_tensor = prior_weights.new_tensor(
                self.theoretical_upper_bound(max_log_deformation)
            )
            lambda_source = "theoretical"
        else:
            lambda_tensor = torch.as_tensor(
                lambda_max, dtype=prior_weights.dtype
            ).reshape(())
            lambda_source = "provided"
        if not bool(torch.isfinite(lambda_tensor)) or float(lambda_tensor) <= 0:
            raise ValueError("lambda_max must be finite and positive")

        self.num_nodes = int(num_nodes)
        self.max_log_deformation = float(max_log_deformation)
        self.degree_epsilon = float(degree_epsilon)
        self.lambda_source = lambda_source
        self.register_buffer("edge_index", edge_index.contiguous())
        self.register_buffer("prior_weights", prior_weights.contiguous())
        self.register_buffer("reference_degree", degree)
        self.register_buffer("reference_degree_inv_sqrt", degree.rsqrt())
        self.register_buffer("lambda_max", lambda_tensor)

    @staticmethod
    def theoretical_upper_bound(max_log_deformation: float) -> float:
        """Return a shared safe bound when ``|log(w/w0)| <= rho_max``."""

        if max_log_deformation < 0:
            raise ValueError("max_log_deformation must be non-negative")
        return 2.0 * math.exp(max_log_deformation)

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def patient_weights(self, log_deformation: torch.Tensor) -> torch.Tensor:
        """Map bounded log-deformations to finite positive edge weights."""

        log_deformation = torch.as_tensor(
            log_deformation,
            device=self.prior_weights.device,
            dtype=self.prior_weights.dtype,
        )
        if log_deformation.shape[-1] != self.num_edges:
            raise ValueError("log_deformation has the wrong edge dimension")
        bounded = log_deformation.clamp(
            -self.max_log_deformation, self.max_log_deformation
        )
        return self.prior_weights * torch.exp(bounded)

    def _canonicalize(
        self, node_values: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, Literal["vector", "features", "batch_vector", "batch_features"]]:
        values = torch.as_tensor(node_values)
        if values.ndim == 1:
            if values.shape[0] != self.num_nodes:
                raise ValueError("node_values has the wrong node dimension")
            return values[None, :, None], "vector"
        if values.ndim == 2:
            if weights.ndim == 2:
                if values.shape[1] != self.num_nodes:
                    raise ValueError("batched node_values must be [batch, nodes]")
                return values[:, :, None], "batch_vector"
            if values.shape[0] != self.num_nodes:
                raise ValueError("node features must be [nodes, channels]")
            return values[None, :, :], "features"
        if values.ndim == 3:
            if values.shape[1] != self.num_nodes:
                raise ValueError("batched node features must be [batch, nodes, channels]")
            return values, "batch_features"
        raise ValueError("node_values must have one, two, or three dimensions")

    @staticmethod
    def _restore(node_values: torch.Tensor, kind: str) -> torch.Tensor:
        if kind == "vector":
            return node_values[0, :, 0]
        if kind == "features":
            return node_values[0]
        if kind == "batch_vector":
            return node_values[:, :, 0]
        return node_values

    def matvec(
        self,
        node_values: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply ``S(w)`` using edge-list matrix-vector products."""

        if weights is None:
            weights = self.prior_weights
        weights = torch.as_tensor(
            weights,
            device=node_values.device,
            dtype=node_values.dtype,
        )
        if weights.ndim not in (1, 2) or weights.shape[-1] != self.num_edges:
            raise ValueError("weights must be [edges] or [batch, edges]")
        values, kind = self._canonicalize(node_values, weights)
        batch_size = values.shape[0]
        if weights.ndim == 1:
            weights = weights.unsqueeze(0).expand(batch_size, -1)
        elif weights.shape[0] != batch_size:
            raise ValueError("weights and node_values have different batch sizes")

        inv_sqrt = self.reference_degree_inv_sqrt.to(
            device=values.device, dtype=values.dtype
        )
        normalized = values * inv_sqrt[None, :, None]
        source, target = self.edge_index.to(values.device)
        edge_difference = normalized[:, source] - normalized[:, target]
        flux = edge_difference * weights[:, :, None]
        output = torch.zeros_like(normalized)
        output.index_add_(1, source, flux)
        output.index_add_(1, target, -flux)
        output = output * inv_sqrt[None, :, None]
        return self._restore(output, kind)

    def apply_scaled(
        self,
        node_values: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply ``S_hat(w) = 2 S(w) / Lambda - I``."""

        scale = self.lambda_max.to(
            device=node_values.device, dtype=node_values.dtype
        )
        return (2.0 / scale) * self.matvec(node_values, weights) - node_values

    def dense_matrix(self, weights: torch.Tensor | None = None) -> torch.Tensor:
        """Materialize ``S(w)`` for small-graph tests and diagnostics only."""

        if weights is None:
            weights = self.prior_weights
        weights = torch.as_tensor(
            weights,
            device=self.prior_weights.device,
            dtype=self.prior_weights.dtype,
        )
        if weights.ndim not in (1, 2) or weights.shape[-1] != self.num_edges:
            raise ValueError("weights must be [edges] or [batch, edges]")
        unbatched = weights.ndim == 1
        if unbatched:
            weights = weights.unsqueeze(0)
        incidence = weights.new_zeros(self.num_edges, self.num_nodes)
        source, target = self.edge_index
        edge_ids = torch.arange(self.num_edges, device=weights.device)
        incidence[edge_ids, source] = 1.0
        incidence[edge_ids, target] = -1.0
        normalized_incidence = (
            incidence * self.reference_degree_inv_sqrt[None, :]
        )
        matrix = torch.einsum(
            "en,be,em->bnm", normalized_incidence, weights, normalized_incidence
        )
        return matrix[0] if unbatched else matrix

    @torch.no_grad()
    def estimate_reference_radius(
        self, iterations: int = 30, tolerance: float = 1e-7
    ) -> torch.Tensor:
        """Estimate the prior operator radius by cached-safe power iteration."""

        if iterations < 1:
            raise ValueError("iterations must be positive")
        vector = torch.ones_like(self.reference_degree)
        vector = vector / vector.norm().clamp_min(tolerance)
        estimate = vector.new_zeros(())
        for _ in range(iterations):
            product = self.matvec(vector)
            norm = product.norm()
            if float(norm) <= tolerance:
                return estimate
            vector = product / norm
            estimate = torch.dot(vector, self.matvec(vector))
        return estimate

    def extra_repr(self) -> str:
        return (
            f"num_nodes={self.num_nodes}, num_edges={self.num_edges}, "
            f"lambda_max={float(self.lambda_max):.6g}, "
            f"lambda_source={self.lambda_source}"
        )


def normalized_patient_probe(
    expression: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Construct ``u = (m * x) / (||m * x||_2 + epsilon)``.

    MRePath already passes each functional group as a local vector, so its
    normal mask is all ones.  The explicit mask argument is retained for tests
    and for a future full-gene input adapter; it must never encode CSV order as
    a graph relation.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if expression.ndim not in (1, 2):
        raise ValueError("expression must be [nodes] or [batch, nodes]")
    masked = expression
    if mask is not None:
        mask = torch.as_tensor(
            mask, device=expression.device, dtype=expression.dtype
        )
        try:
            masked = expression * mask
        except RuntimeError as error:
            raise ValueError("mask is not broadcastable to expression") from error
    norm = torch.linalg.vector_norm(masked, dim=-1, keepdim=True)
    return masked / (norm + epsilon)


def chebyshev_recurrence(
    operator: ReferenceSpectralOperator,
    probe: torch.Tensor,
    weights: torch.Tensor | None,
    order: int,
) -> torch.Tensor:
    """Return ``T_0(S_hat)u, ..., T_order(S_hat)u``.

    The returned shape is ``[order + 1, nodes]`` for one patient and
    ``[batch, order + 1, nodes]`` for a batch.  Only spectral MVPs are used.
    """

    if order < 0:
        raise ValueError("order must be non-negative")
    if probe.ndim not in (1, 2) or probe.shape[-1] != operator.num_nodes:
        raise ValueError("probe must be [nodes] or [batch, nodes]")
    unbatched = probe.ndim == 1
    batched_probe = probe.unsqueeze(0) if unbatched else probe
    if weights is None:
        weights = operator.prior_weights
    weights = torch.as_tensor(
        weights, device=probe.device, dtype=probe.dtype
    )
    if weights.ndim == 1:
        weights = weights.unsqueeze(0).expand(batched_probe.shape[0], -1)
    if weights.shape != (batched_probe.shape[0], operator.num_edges):
        raise ValueError("weights must match probe batch and edge dimensions")

    responses = [batched_probe]
    if order >= 1:
        responses.append(operator.apply_scaled(batched_probe, weights))
    for _ in range(2, order + 1):
        responses.append(
            2.0 * operator.apply_scaled(responses[-1], weights)
            - responses[-2]
        )
    stacked = torch.stack(responses, dim=1)
    return stacked[0] if unbatched else stacked


class ChebyshevMomentResponse(nn.Module):
    """Patient-conditioned Chebyshev response moments on a reference graph."""

    def __init__(
        self,
        operator: ReferenceSpectralOperator,
        order: int = 2,
        probe_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if order < 1:
            raise ValueError("moment order must be at least one")
        if probe_epsilon <= 0:
            raise ValueError("probe_epsilon must be positive")
        self.operator = operator
        self.order = int(order)
        self.probe_epsilon = float(probe_epsilon)

    def forward(
        self,
        expression: torch.Tensor,
        weights: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        *,
        return_recurrence: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probe = normalized_patient_probe(
            expression, mask, epsilon=self.probe_epsilon
        )
        recurrence = chebyshev_recurrence(
            self.operator, probe, weights, self.order
        )
        if probe.ndim == 1:
            moments = torch.einsum("rn,n->r", recurrence[1:], probe)
        else:
            moments = torch.einsum("brn,bn->br", recurrence[:, 1:], probe)
        if return_recurrence:
            return moments, probe, recurrence
        return moments
