#!/usr/bin/env python3
"""Audit the released CO-READ artifacts against the paper contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data/tcga_coadread/clam_20x_resnet50_paper_k9",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=ROOT / "data/tcga_coadread/clam_20x_resnet50",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a failure status when released artifacts differ from the paper",
    )
    return parser.parse_args()


def report(label: str, status: str, detail: str) -> None:
    print(f"[{status}] {label}: {detail}")


def main() -> int:
    args = parse_args()
    contract = json.loads((ROOT / "configs/paper_coadread.json").read_text())
    expected_cases = int(contract["cohort"]["paper_cases"])
    expected_dim = int(contract["pathology"]["feature_dimension"])
    expected_patch_size = int(contract["pathology"]["patch_size"])

    metadata = pd.read_csv(
        ROOT / "datasets_csv/metadata/tcga_coadread.csv"
    )
    metadata_cases = set(metadata["case_id"].astype(str))
    metadata_slides = set(
        metadata["slide_id"].astype(str).map(lambda value: Path(value).stem)
    )
    blockers: list[str] = []

    if len(metadata_cases) == expected_cases:
        report("metadata cohort", "PASS", f"{len(metadata_cases)} cases")
    else:
        blockers.append("metadata cohort size")
        report(
            "metadata cohort",
            "FAIL",
            f"{len(metadata_cases)} cases; paper reports {expected_cases}",
        )

    split_universe: set[str] = set()
    split_sizes = []
    for fold in range(int(contract["cohort"]["folds"])):
        split = pd.read_csv(
            ROOT / f"splits/5folds/tcga_coadread/splits_{fold}.csv"
        )
        train = set(split["train"].dropna().astype(str))
        val = set(split["val"].dropna().astype(str))
        if train & val:
            blockers.append(f"fold {fold} leakage")
            report(
                f"fold {fold}",
                "FAIL",
                f"{len(train & val)} cases occur in train and validation",
            )
        split_universe.update(train | val)
        split_sizes.append((len(train), len(val)))

    if split_universe == metadata_cases:
        report("split coverage", "PASS", f"{len(split_universe)} cases")
    else:
        blockers.append("released split coverage")
        report(
            "split coverage",
            "LIMIT",
            f"{len(split_universe)}/{len(metadata_cases)} metadata cases; "
            f"missing={sorted(metadata_cases - split_universe)}; "
            f"sizes={split_sizes}",
        )

    rna_path = (
        ROOT
        / "datasets_csv/raw_rna_data/combine/coadread/rna_clean.csv"
    )
    rna_cases = set(pd.read_csv(rna_path, usecols=[0]).iloc[:, 0].astype(str))
    missing_rna = sorted(split_universe - rna_cases)
    effective_cases = split_universe & rna_cases
    if not missing_rna:
        report("released omics coverage", "PASS", f"{len(effective_cases)} cases")
    else:
        blockers.append("released omics coverage")
        report(
            "released omics coverage",
            "LIMIT",
            f"{len(effective_cases)} effective cases; missing RNA={missing_rna}",
        )

    graph_dir = args.data_root.resolve() / "graph_files"
    graph_files = {
        path.stem for path in graph_dir.glob("*.pt") if path.is_file()
    }
    missing_graphs = sorted(metadata_slides - graph_files)
    if not missing_graphs and len(graph_files) == len(metadata_slides):
        report("paper graph inventory", "PASS", f"{len(graph_files)} slides")
    else:
        blockers.append("paper graph inventory")
        report(
            "paper graph inventory",
            "FAIL",
            f"{len(graph_files)}/{len(metadata_slides)} slides; "
            f"missing={missing_graphs[:5]}",
        )

    h5_dir = args.feature_root.resolve() / "h5_files"
    h5_files = sorted(h5_dir.glob("*.h5"))
    feature_failures = []
    for path in h5_files:
        try:
            with h5py.File(path, "r") as handle:
                features = handle["features"]
                coords = handle["coords"]
                if (
                    features.ndim != 2
                    or features.shape[1] != expected_dim
                    or coords.shape != (features.shape[0], 2)
                ):
                    feature_failures.append(
                        f"{path.name}:{features.shape}/{coords.shape}"
                    )
        except (OSError, KeyError) as exc:
            feature_failures.append(f"{path.name}:{exc}")
    if len(h5_files) == len(metadata_slides) and not feature_failures:
        report(
            "ResNet50 feature inventory",
            "PASS",
            f"{len(h5_files)} slides, {expected_dim}-D",
        )
    else:
        blockers.append("ResNet50 feature inventory")
        report(
            "ResNet50 feature inventory",
            "FAIL",
            f"{len(h5_files)}/{len(metadata_slides)} slides; "
            f"invalid={feature_failures[:5]}",
        )

    coord_files = sorted(
        (args.feature_root.resolve() / "patches").glob("*.h5")
    )
    patch_attrs = set()
    for path in coord_files:
        with h5py.File(path, "r") as handle:
            coords = handle["coords"]
            patch_attrs.add(
                (
                    int(coords.attrs.get("patch_level", -1)),
                    int(coords.attrs.get("patch_size", -1)),
                )
            )
    # A 512-pixel level-0 field from a native 40x WSI represents a
    # 256-pixel field at the paper's 20x magnification.
    accepted_attrs = {(0, expected_patch_size), (0, expected_patch_size * 2)}
    if len(coord_files) == len(metadata_slides) and patch_attrs <= accepted_attrs:
        report(
            "patch coordinates",
            "PASS",
            f"{len(coord_files)} slides, attrs={sorted(patch_attrs)}",
        )
    else:
        blockers.append("patch coordinates")
        report(
            "patch coordinates",
            "FAIL",
            f"{len(coord_files)}/{len(metadata_slides)} slides, "
            f"attrs={sorted(patch_attrs)}",
        )

    report(
        "paper genomics modalities",
        "LIMIT",
        "paper states RNA-seq+CNV+SNV; released repository contains RNA only",
    )
    blockers.append("CNV/SNV are not present in the released matrix")

    print(
        "[SUMMARY] "
        f"metadata={len(metadata_cases)} split={len(split_universe)} "
        f"effective_omics={len(effective_cases)} slides={len(metadata_slides)} "
        f"blockers={len(blockers)}"
    )
    if args.strict and blockers:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
