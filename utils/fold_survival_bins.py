"""Leakage-safe discrete survival bins fitted within one training fold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FoldSurvivalBinner:
    boundaries: np.ndarray
    label_col: str
    censorship_col: str

    @classmethod
    def fit(
        cls,
        training_metadata: pd.DataFrame,
        *,
        label_col: str,
        censorship_col: str,
        n_bins: int,
    ) -> "FoldSurvivalBinner":
        if n_bins < 2:
            raise ValueError("n_bins must be at least two")
        uncensored = training_metadata.loc[
            training_metadata[censorship_col] < 1, label_col
        ].dropna()
        if len(uncensored) < n_bins:
            raise ValueError("not enough uncensored training patients for bins")
        _, boundaries = pd.qcut(
            uncensored,
            q=n_bins,
            retbins=True,
            labels=False,
            duplicates="raise",
        )
        boundaries = np.asarray(boundaries, dtype=np.float64)
        # Validation outcomes may lie outside the training range.  Infinite end
        # caps assign them without allowing those outcomes to change inner cuts.
        boundaries[0] = -np.inf
        boundaries[-1] = np.inf
        return cls(boundaries, label_col, censorship_col)

    def transform(self, metadata: pd.DataFrame) -> pd.DataFrame:
        transformed = metadata.copy()
        labels = pd.cut(
            transformed[self.label_col],
            bins=self.boundaries,
            labels=False,
            right=False,
            include_lowest=True,
        )
        if labels.isna().any():
            raise ValueError("survival binning produced missing labels")
        transformed["disc_label"] = labels.astype(int)
        transformed["label"] = (
            transformed["disc_label"] * 2
            + transformed[self.censorship_col].astype(int)
        )
        return transformed


def apply_training_fold_survival_bins(
    train_split,
    val_split,
    *,
    label_col: str,
    censorship_col: str,
    n_bins: int,
) -> FoldSurvivalBinner:
    """Mutate only split-local metadata using training-derived boundaries."""

    binner = FoldSurvivalBinner.fit(
        train_split.metadata,
        label_col=label_col,
        censorship_col=censorship_col,
        n_bins=n_bins,
    )
    train_split.metadata = binner.transform(train_split.metadata)
    val_split.metadata = binner.transform(val_split.metadata)
    # Weighted samplers cache class-index lists, so refresh after relabelling.
    train_split.slide_cls_id_prep()
    val_split.slide_cls_id_prep()
    return binner
