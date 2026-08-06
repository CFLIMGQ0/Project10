"""Configuration, prior construction, and JSON serialization helpers."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_pc_cmka_config(path: str | Path, experiment: str) -> dict[str, Any]:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    experiments = {item["name"]: item for item in raw["experiments"]}
    staged = {item["name"]: item for item in raw.get("staged_experiments", [])}
    controls = {item["name"]: item for item in raw.get("controls", [])}
    available = {**controls, **staged, **experiments}
    if experiment not in available:
        raise ValueError(
            f"unknown PC-CMKA experiment {experiment!r}; "
            f"available={sorted(available)}"
        )
    selected = available[experiment]
    config = {
        key: value
        for key, value in raw.items()
        if key not in {"experiments", "staged_experiments", "controls"}
    }
    config = deep_update(config, selected.get("overrides", {}))
    config["experiment_name"] = experiment
    config["source_label"] = selected.get("source_label", experiment)
    config["question"] = selected.get("question", "")
    config["encoder"] = selected.get("encoder", "pc_cmka_ddkac")
    return config


def _correlation_graphs(train_split, neighbours: int, minimum: float) -> list[torch.Tensor]:
    rna = train_split.omics_data_dict["rna"]
    graphs = []
    for genes in train_split.omic_names:
        values = rna[list(genes)].to_numpy(dtype=np.float32, copy=True)
        count = values.shape[1]
        if count == 1:
            graphs.append(torch.ones((1, 1), dtype=torch.float32))
            continue
        correlation = np.corrcoef(values, rowvar=False)
        correlation = np.nan_to_num(
            np.abs(correlation), nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
        np.fill_diagonal(correlation, 0.0)
        keep = min(int(neighbours), count - 1)
        indices = np.argpartition(correlation, -keep, axis=1)[:, -keep:]
        adjacency = np.zeros_like(correlation)
        rows = np.arange(count)[:, None]
        adjacency[rows, indices] = np.maximum(correlation[rows, indices], minimum)
        adjacency = np.maximum(adjacency, adjacency.T)
        graphs.append(torch.from_numpy(adjacency))
    return graphs


def _external_prior_graphs(
    gene_groups: Sequence[Sequence[str]],
    edge_list: Path,
    minimum: float,
) -> list[torch.Tensor]:
    frame = pd.read_csv(edge_list, sep=None, engine="python")
    aliases = {
        "source": ("source", "gene_a", "gene1", "from"),
        "target": ("target", "gene_b", "gene2", "to"),
        "weight": ("weight", "score", "confidence"),
    }
    columns = {}
    lowered = {str(column).lower(): column for column in frame.columns}
    for name, candidates in aliases.items():
        for candidate in candidates:
            if candidate in lowered:
                columns[name] = lowered[candidate]
                break
    if "source" not in columns or "target" not in columns:
        raise ValueError("prior edge list requires source and target gene columns")
    graphs = []
    for genes in gene_groups:
        index = {str(gene): position for position, gene in enumerate(genes)}
        adjacency = torch.zeros((len(genes), len(genes)), dtype=torch.float32)
        for row in frame.itertuples(index=False):
            source = str(getattr(row, columns["source"]))
            target = str(getattr(row, columns["target"]))
            if source not in index or target not in index or source == target:
                continue
            weight = (
                float(getattr(row, columns["weight"]))
                if "weight" in columns
                else 1.0
            )
            i, j = index[source], index[target]
            adjacency[i, j] = adjacency[j, i] = max(weight, minimum)
        # Keep every group operable even if the external network has isolated
        # genes: connect isolated nodes to the next gene with minimum weight.
        if len(genes) > 1:
            isolated = adjacency.sum(dim=1) == 0
            for node in isolated.nonzero(as_tuple=False).flatten().tolist():
                other = (node + 1) % len(genes)
                adjacency[node, other] = adjacency[other, node] = minimum
        graphs.append(adjacency)
    return graphs


def build_fold_pc_cmka_priors(train_split, config: dict[str, Any]) -> tuple[list[torch.Tensor], dict[str, Any]]:
    prior = config["prior"]
    minimum = float(prior.get("minimum_weight", 1e-3))
    requested = str(prior.get("source", "train_fold_correlation_fallback"))
    edge_path = prior.get("edge_list")
    if edge_path:
        path = Path(edge_path)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file():
            graphs = _external_prior_graphs(train_split.omic_names, path, minimum)
            provenance = {
                "source": "external_biological_edge_list",
                "path": str(path.resolve()),
                "training_fold_only": True,
                "warning": None,
            }
            return graphs, provenance
    graphs = _correlation_graphs(
        train_split,
        int(prior.get("neighbours", 8)),
        minimum,
    )
    provenance = {
        "source": "train_fold_correlation_fallback",
        "requested_source": requested,
        "training_fold_only": True,
        "warning": (
            "No external PPI/Reactome/KEGG/GO edge list was supplied. "
            "This run must not be described as using a biological prior."
        ),
    }
    return graphs, provenance


def to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
