"""Utilities for persistent, seed-independent WSI hypergraph caches."""

from __future__ import annotations

import os
from pathlib import Path

import torch


CACHE_VERSION = 1
EDGE_TYPES = {
    "topology": "edge_index",
    "feature": "edge_latent",
}


def edges_to_incidence(
    edges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert directed neighbourhood edges to a deduplicated incidence COO."""
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edges must be [2, num_edges]")
    edges = edges.detach().cpu().long()
    if edges.shape[1] == 0:
        empty = torch.empty((2, 0), dtype=torch.long)
        return empty, torch.empty(0, dtype=torch.long)

    unique_pairs = torch.unique(edges.t(), dim=0).t().contiguous()
    centers, inverse = torch.unique(
        unique_pairs[0], sorted=True, return_inverse=True
    )
    hyperedge_ids = torch.arange(centers.numel(), dtype=torch.long)
    incidence = torch.stack(
        (
            torch.cat((centers, unique_pairs[1])),
            torch.cat((hyperedge_ids, inverse)),
        )
    )
    incidence = torch.unique(incidence.t(), dim=0).t().contiguous()
    return incidence, centers


def subset_incidence(
    incidence: torch.Tensor,
    centers: torch.Tensor,
    selected_nodes: torch.Tensor,
    num_source_nodes: int,
) -> torch.Tensor:
    """Create the induced incidence matrix in the selected-node order."""
    selected_nodes = selected_nodes.detach().cpu().long()
    if incidence.numel() == 0 or selected_nodes.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)

    node_mapping = torch.full(
        (num_source_nodes,), -1, dtype=torch.long
    )
    node_mapping[selected_nodes] = torch.arange(
        selected_nodes.numel(), dtype=torch.long
    )

    incidence = incidence.detach().cpu().long()
    centers = centers.detach().cpu().long()
    selected_centers = node_mapping[centers] >= 0
    selected_entries = (
        (node_mapping[incidence[0]] >= 0)
        & selected_centers[incidence[1]]
    )
    if not selected_entries.any():
        return torch.empty((2, 0), dtype=torch.long)

    counts = torch.bincount(
        incidence[1, selected_entries], minlength=centers.numel()
    )
    # The released conversion creates a hyperedge only when the selected
    # centre retains at least one selected neighbour. The centre itself is
    # the second incidence, hence a minimum cardinality of two.
    kept_hyperedges = counts >= 2
    selected_entries &= kept_hyperedges[incidence[1]]
    if not selected_entries.any():
        return torch.empty((2, 0), dtype=torch.long)

    hyperedge_mapping = torch.full(
        (centers.numel(),), -1, dtype=torch.long
    )
    kept_ids = torch.nonzero(kept_hyperedges, as_tuple=False).flatten()
    hyperedge_mapping[kept_ids] = torch.arange(
        kept_ids.numel(), dtype=torch.long
    )
    return torch.stack(
        (
            node_mapping[incidence[0, selected_entries]],
            hyperedge_mapping[incidence[1, selected_entries]],
        )
    ).contiguous()


def graph_fingerprint(path: str | Path) -> dict[str, int]:
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def build_cache_record(graph, source_path: str | Path) -> dict:
    record = {
        "version": CACHE_VERSION,
        "source": graph_fingerprint(source_path),
        "num_nodes": int(graph.x.shape[0]),
        "hyperedge_size": int(getattr(graph, "hyperedge_size", 0)),
    }
    for cache_name, graph_attribute in EDGE_TYPES.items():
        incidence, centers = edges_to_incidence(
            getattr(graph, graph_attribute)
        )
        record[cache_name] = {
            "incidence": incidence,
            "centers": centers,
        }
    return record


def cache_matches(record: dict, source_path: str | Path) -> bool:
    return (
        record.get("version") == CACHE_VERSION
        and record.get("source") == graph_fingerprint(source_path)
        and all(name in record for name in EDGE_TYPES)
    )


def save_cache_record(record: dict, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(record, temporary)
    os.replace(temporary, destination)


def load_cache_record(
    source_path: str | Path,
    cache_dir: str | Path,
) -> dict:
    source_path = Path(source_path)
    cache_path = Path(cache_dir) / f"{source_path.stem}.pt"
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Missing hypergraph cache for {source_path.name}: {cache_path}"
        )
    record = torch.load(
        cache_path, map_location="cpu", weights_only=False, mmap=True
    )
    if not cache_matches(record, source_path):
        raise RuntimeError(
            f"Stale hypergraph cache for {source_path.name}: {cache_path}"
        )
    return record
