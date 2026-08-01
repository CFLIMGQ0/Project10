"""Minimal GPU incidence hypergraph used by ordinary HGNN convolutions."""

from __future__ import annotations

import torch
from torch_scatter import scatter_mean


class IncidenceHypergraph:
    """DHG-compatible mean vertex-to-edge-to-vertex message passing."""

    def __init__(
        self,
        num_vertices: int,
        incidence: torch.Tensor,
    ) -> None:
        if incidence.ndim != 2 or incidence.shape[0] != 2:
            raise ValueError("incidence must be [2, num_incidence_entries]")
        self.num_v = num_vertices
        self.incidence = incidence.long()
        self.num_e = (
            int(self.incidence[1].max().item()) + 1
            if self.incidence.numel()
            else 0
        )

    def v2v(
        self,
        features: torch.Tensor,
        aggr: str = "mean",
        **_,
    ) -> torch.Tensor:
        if aggr != "mean":
            raise ValueError("IncidenceHypergraph currently supports mean only")
        if self.incidence.device != features.device:
            self.incidence = self.incidence.to(features.device)
        nodes, hyperedges = self.incidence
        edge_features = scatter_mean(
            features[nodes], hyperedges, dim=0, dim_size=self.num_e
        )
        return scatter_mean(
            edge_features[hyperedges],
            nodes,
            dim=0,
            dim_size=self.num_v,
        )
