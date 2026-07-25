#!/usr/bin/env python3
"""Combine independently trained SNN and CLAM-style WSI risks by averaging."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snn-root", type=Path, required=True)
    parser.add_argument("--clam-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def locate(root: Path, fold: int) -> Path:
    matches = list(root.rglob(f"split_latest_val_{fold}_results.pkl"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one fold-{fold} result below {root}, found {matches}"
        )
    return matches[0]


def load(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def scalar(value: object) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_scores = []

    if args.folds < 1:
        raise ValueError("--folds must be at least 1")

    for fold in range(args.folds):
        snn = load(locate(args.snn_root, fold))
        clam = load(locate(args.clam_root, fold))
        case_ids = sorted(set(snn) & set(clam))
        if len(case_ids) < 50:
            raise RuntimeError(
                f"Fold {fold} has only {len(case_ids)} aligned validation cases."
            )

        rows = []
        for case_id in case_ids:
            left = snn[case_id]
            right = clam[case_id]
            if (
                scalar(left["survival"]) != scalar(right["survival"])
                or scalar(left["censorship"]) != scalar(right["censorship"])
            ):
                raise RuntimeError(f"Endpoint mismatch for {case_id}")
            rows.append(
                {
                    "case_id": case_id,
                    "risk_snn": scalar(left["risk"]),
                    "risk_clam": scalar(right["risk"]),
                    "risk_ensemble": (
                        scalar(left["risk"]) + scalar(right["risk"])
                    )
                    / 2.0,
                    "survival_months_dss": scalar(left["survival"]),
                    "censorship_dss": scalar(left["censorship"]),
                }
            )

        predictions = pd.DataFrame(rows)
        score = concordance_index_censored(
            (1 - predictions["censorship_dss"].to_numpy()).astype(bool),
            predictions["survival_months_dss"].to_numpy(),
            predictions["risk_ensemble"].to_numpy(),
            tied_tol=1e-8,
        )[0]
        fold_scores.append(float(score))
        predictions.to_csv(
            args.output_dir / f"fold_{fold}_predictions.csv", index=False
        )

    summary = pd.DataFrame(
        {"folds": range(args.folds), "val_cindex": fold_scores}
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(
        {
            "mean_cindex": [np.mean(fold_scores)],
            "std_cindex": [np.std(fold_scores)],
        }
    ).to_csv(args.output_dir / "aggregate.csv", index=False)
    print(
        "SNN+CLAM C-index: "
        f"{np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}"
    )


if __name__ == "__main__":
    main()
