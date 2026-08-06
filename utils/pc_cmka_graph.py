"""Fold-local prior graph construction for PC-CMKA-DDKAC.

The current repository does not ship an explicit PPI/Reactome edge list.  The
only immediately reproducible fallback is therefore the released DD-KAC's
training-fold absolute-correlation support.  Results produced with this source
must be labelled ``training_correlation`` and must not be described as using an
external biological interaction graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class PathwayGraphPrior:
    edge_index: torch.Tensor
    prior_weights: torch.Tensor
    num_nodes: int
    gene_names: tuple[str, ...]
    source: str

    def validate(self) -> None:
        if self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must be [2, edges]")
        if self.edge_index.shape[1] != self.prior_weights.numel():
            raise ValueError("edge and weight counts differ")
        if len(self.gene_names) != self.num_nodes:
            raise ValueError("gene_names and num_nodes differ")
        if self.edge_index.numel() and (
            int(self.edge_index.min()) < 0
            or int(self.edge_index.max()) >= self.num_nodes
        ):
            raise ValueError("edge_index contains an invalid node")
        if bool((self.edge_index[0] >= self.edge_index[1]).any()):
            raise ValueError("edges must be unique undirected pairs with source < target")
        if bool((self.prior_weights <= 0).any()) or not bool(
            torch.isfinite(self.prior_weights).all()
        ):
            raise ValueError("prior weights must be finite and positive")


def build_training_correlation_prior(
    values: np.ndarray,
    gene_names: Sequence[str],
    *,
    neighbours: int = 8,
    minimum_weight: float = 1e-4,
) -> PathwayGraphPrior:
    """Build one fixed support from training-fold samples only."""

    values = np.asarray(values, dtype=np.float32)
    gene_names = tuple(str(name) for name in gene_names)
    if values.ndim != 2 or values.shape[1] != len(gene_names):
        raise ValueError("values must be [training patients, pathway genes]")
    if values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("a correlation prior needs at least two samples and genes")
    if neighbours < 1 or minimum_weight <= 0:
        raise ValueError("invalid prior graph settings")
    feature_count = values.shape[1]
    keep = min(int(neighbours), feature_count - 1)
    # Constant training-fold genes have undefined Pearson correlation.  They
    # are converted to zero below and receive only the minimum positive prior
    # weight when selected, without emitting misleading runtime warnings.
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(values, rowvar=False)
    correlation = np.nan_to_num(
        np.abs(correlation), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)
    np.fill_diagonal(correlation, 0.0)
    indices = np.argpartition(correlation, -keep, axis=1)[:, -keep:]
    selected = np.zeros_like(correlation, dtype=bool)
    selected[np.arange(feature_count)[:, None], indices] = True
    selected = np.logical_or(selected, selected.T)
    source, target = np.nonzero(np.triu(selected, k=1))
    weights = np.maximum(correlation[source, target], minimum_weight)
    prior = PathwayGraphPrior(
        edge_index=torch.from_numpy(
            np.stack((source, target), axis=0).astype(np.int64)
        ),
        prior_weights=torch.from_numpy(weights.astype(np.float32)),
        num_nodes=feature_count,
        gene_names=gene_names,
        source="training_correlation",
    )
    prior.validate()
    degree_count = torch.bincount(
        prior.edge_index.flatten(), minlength=feature_count
    )
    if bool((degree_count == 0).any()):
        raise RuntimeError("correlation prior unexpectedly contains isolated genes")
    return prior


def build_fold_pathway_priors(
    train_split,
    *,
    neighbours: int = 8,
    minimum_weight: float = 1e-4,
) -> list[PathwayGraphPrior]:
    """Build all six priors without reading validation/test objects."""

    if len(train_split.omic_names) != 6:
        raise ValueError("PC-CMKA-DDKAC requires exactly six functional groups")
    rna = train_split.omics_data_dict["rna"]
    priors = []
    for genes in train_split.omic_names:
        # The explicit gene-name selection makes CSV column adjacency irrelevant.
        values = rna[list(genes)].to_numpy(dtype=np.float32, copy=True)
        priors.append(
            build_training_correlation_prior(
                values,
                genes,
                neighbours=neighbours,
                minimum_weight=minimum_weight,
            )
        )
    return priors
