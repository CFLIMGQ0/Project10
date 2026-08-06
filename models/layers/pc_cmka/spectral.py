"""Fixed-reference spectral geometry and Chebyshev response moments."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class ReferenceSpectralOperator(nn.Module):
    """Apply ``D0^-1/2 B^T diag(w) B D0^-1/2`` without dense ``B``.

    Edges are stored once as an undirected upper-triangular support.  Patient
    weights may change, but the degree preconditioner and spectral bound are
    always those derived from the fixed prior support.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        prior_weights: torch.Tensor,
        num_nodes: int,
        spectral_bound: float | None = None,
        max_log_deformation: float = 0.5,
        normalization: str = "reference",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        prior_weights = torch.as_tensor(prior_weights, dtype=torch.float32)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edges]")
        if edge_index.shape[1] != prior_weights.numel():
            raise ValueError("one prior weight is required per edge")
        if prior_weights.numel() == 0:
            raise ValueError("the reference graph must contain at least one edge")
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes:
            raise ValueError("edge_index contains an out-of-range node")
        if torch.any(edge_index[0] == edge_index[1]):
            raise ValueError("self loops are not part of the incidence support")
        if torch.any(prior_weights <= 0):
            raise ValueError("reference edge weights must be strictly positive")

        degree = torch.zeros(num_nodes, dtype=prior_weights.dtype)
        degree.scatter_add_(0, edge_index[0], prior_weights)
        degree.scatter_add_(0, edge_index[1], prior_weights)
        inverse_sqrt_degree = degree.clamp_min(eps).rsqrt()

        # For a weighted normalized Laplacian, lambda_max <= 2.  With fixed
        # D0 and |log(w/w0)| <= delta, the patient operator is bounded by
        # 2*exp(delta).  This bound is shared by every patient and cached.
        if spectral_bound is None:
            spectral_bound = 2.0 * float(torch.exp(torch.tensor(max_log_deformation)))
        if spectral_bound <= 0:
            raise ValueError("spectral_bound must be positive")
        if normalization not in {"reference", "patient"}:
            raise ValueError("normalization must be reference or patient")

        self.num_nodes = int(num_nodes)
        self.eps = float(eps)
        self.normalization = normalization
        self.register_buffer("edge_index", edge_index.contiguous())
        self.register_buffer("prior_weights", prior_weights.contiguous())
        self.register_buffer("reference_degree", degree)
        self.register_buffer("inverse_sqrt_degree", inverse_sqrt_degree)
        self.register_buffer(
            "spectral_bound", torch.tensor(float(spectral_bound), dtype=torch.float32)
        )

    @classmethod
    def from_adjacency(
        cls,
        adjacency: torch.Tensor,
        **kwargs,
    ) -> "ReferenceSpectralOperator":
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be square")
        adjacency = torch.maximum(adjacency, adjacency.t())
        support = torch.triu(adjacency, diagonal=1).nonzero(as_tuple=False)
        if support.numel() == 0:
            # A deterministic chain is a conservative fallback for a group
            # whose supplied prior contains no usable edge.
            nodes = torch.arange(adjacency.shape[0] - 1)
            support = torch.stack((nodes, nodes + 1), dim=1)
            weights = torch.ones(support.shape[0], dtype=adjacency.dtype)
        else:
            weights = adjacency[support[:, 0], support[:, 1]].clamp_min(1e-6)
        return cls(support.t(), weights, adjacency.shape[0], **kwargs)

    def _batch(self, values: torch.Tensor, width: int, name: str) -> tuple[torch.Tensor, bool]:
        values = torch.as_tensor(values)
        squeezed = values.ndim == 1
        if squeezed:
            values = values.unsqueeze(0)
        if values.ndim != 2 or values.shape[1] != width:
            raise ValueError(f"{name} must have shape [batch, {width}]")
        return values, squeezed

    def apply_operator(
        self,
        weights: torch.Tensor,
        signals: torch.Tensor,
        *,
        scaled: bool = False,
    ) -> torch.Tensor:
        """Apply the reference-preconditioned operator to batched signals."""

        signals, squeeze_signal = self._batch(signals, self.num_nodes, "signals")
        weights, _ = self._batch(weights, self.edge_index.shape[1], "weights")
        if weights.shape[0] == 1 and signals.shape[0] > 1:
            weights = weights.expand(signals.shape[0], -1)
        if signals.shape[0] == 1 and weights.shape[0] > 1:
            signals = signals.expand(weights.shape[0], -1)
            squeeze_signal = False
        if weights.shape[0] != signals.shape[0]:
            raise ValueError("weights and signals have incompatible batch sizes")

        source, target = self.edge_index
        if self.normalization == "reference":
            inverse_degree = self.inverse_sqrt_degree.unsqueeze(0)
        else:
            degree = torch.zeros_like(signals)
            degree.scatter_add_(1, source.unsqueeze(0).expand_as(weights), weights)
            degree.scatter_add_(1, target.unsqueeze(0).expand_as(weights), weights)
            inverse_degree = degree.clamp_min(self.eps).rsqrt()
        normalized = signals * inverse_degree
        difference = normalized[:, source] - normalized[:, target]
        messages = weights * difference
        output = torch.zeros_like(signals)
        output.scatter_add_(1, source.unsqueeze(0).expand_as(messages), messages)
        output.scatter_add_(1, target.unsqueeze(0).expand_as(messages), -messages)
        output = output * inverse_degree
        if scaled:
            output = 2.0 * output / self.spectral_bound - signals
        return output.squeeze(0) if squeeze_signal else output

    def dense_matrix(
        self,
        weights: torch.Tensor | None = None,
        *,
        scaled: bool = False,
    ) -> torch.Tensor:
        """Materialize a small dense matrix for tests and diagnostics only."""

        if weights is None:
            weights = self.prior_weights
        weights = torch.as_tensor(
            weights, device=self.prior_weights.device, dtype=self.prior_weights.dtype
        )
        if weights.ndim != 1:
            raise ValueError("dense_matrix accepts one edge-weight vector")
        source, target = self.edge_index
        incidence = torch.zeros(
            (weights.numel(), self.num_nodes),
            dtype=weights.dtype,
            device=weights.device,
        )
        row = torch.arange(weights.numel(), device=weights.device)
        incidence[row, source] = 1.0
        incidence[row, target] = -1.0
        if self.normalization == "reference":
            inverse_degree = self.inverse_sqrt_degree
        else:
            degree = torch.zeros(
                self.num_nodes, device=weights.device, dtype=weights.dtype
            )
            degree.scatter_add_(0, source, weights)
            degree.scatter_add_(0, target, weights)
            inverse_degree = degree.clamp_min(self.eps).rsqrt()
        scaled_incidence = incidence * inverse_degree.unsqueeze(0)
        matrix = scaled_incidence.t() @ (weights.unsqueeze(1) * scaled_incidence)
        if scaled:
            matrix = 2.0 * matrix / self.spectral_bound - torch.eye(
                self.num_nodes, device=matrix.device, dtype=matrix.dtype
            )
        return matrix

    def chebyshev_responses(
        self,
        weights: torch.Tensor,
        signals: torch.Tensor,
        order: int,
    ) -> list[torch.Tensor]:
        if order < 0:
            raise ValueError("Chebyshev order must be non-negative")
        signals, squeezed = self._batch(signals, self.num_nodes, "signals")
        responses = [signals]
        if order >= 1:
            responses.append(self.apply_operator(weights, signals, scaled=True))
        for _ in range(2, order + 1):
            responses.append(
                2.0 * self.apply_operator(weights, responses[-1], scaled=True)
                - responses[-2]
            )
        if squeezed:
            return [response.squeeze(0) for response in responses]
        return responses

    def moments(
        self,
        weights: torch.Tensor,
        probes: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        """Return moments for orders 1..R (order zero is intentionally omitted)."""

        probes, squeezed = self._batch(probes, self.num_nodes, "probes")
        responses = self.chebyshev_responses(weights, probes, order)
        values = torch.stack(
            [(probes * responses[index]).sum(dim=1) for index in range(1, order + 1)],
            dim=1,
        )
        return values.squeeze(0) if squeezed else values

    def krylov_basis(
        self,
        weights: torch.Tensor,
        probes: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        """Build an orthonormal basis for the accessible patient subspace."""

        if probes.ndim == 1:
            probes = probes.unsqueeze(0)
        if probes.ndim != 2 or probes.shape[1] != self.num_nodes:
            raise ValueError("probes must be [count, nodes]")
        columns: list[torch.Tensor] = []
        current = probes
        for _ in range(order + 1):
            columns.extend(row for row in current)
            current = self.apply_operator(weights, current, scaled=True)
        matrix = torch.stack(columns, dim=1)
        basis, triangular = torch.linalg.qr(matrix, mode="reduced")
        diagonal = torch.diagonal(triangular).abs()
        rank = max(int((diagonal > self.eps).sum().item()), 1)
        return basis[:, :rank]
