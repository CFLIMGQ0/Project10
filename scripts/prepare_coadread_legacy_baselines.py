#!/usr/bin/env python3
"""Build a DSS/RNA-only table for legacy multimodal baseline loaders.

CMTA and PORPOISE expect one wide CSV with clinical columns followed by
``*_rnaseq`` features.  MRePath stores the same information in separate Xena
RNA and endpoint files.  This converter keeps the official model loaders while
making the cohort, endpoint, and five folds match the MRePath experiment.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    cache = Path("/home/administrator/.cache/mrepath/third_party")
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=project)
    parser.add_argument("--survpath-repo", type=Path, default=cache / "SurvPath")
    parser.add_argument("--cmta-repo", type=Path, default=cache / "CMTA")
    parser.add_argument("--porpoise-repo", type=Path, default=cache / "PORPOISE")
    return parser.parse_args()


def build_table(project_dir: Path) -> pd.DataFrame:
    metadata_path = (
        project_dir / "datasets_csv/metadata/tcga_coadread.csv"
    )
    rna_path = (
        project_dir
        / "datasets_csv/raw_rna_data/combine/coadread/rna_clean.csv"
    )
    metadata = pd.read_csv(metadata_path)
    rna = pd.read_csv(rna_path).rename(columns={"Unnamed: 0": "case_id"})
    # Match MRePath's loader, which selects the first RNA row when Xena
    # contains more than one aliquot for the same TCGA case.
    rna = rna.drop_duplicates("case_id", keep="first")

    metadata = metadata[
        [
            "Unnamed: 0",
            "case_id",
            "slide_id",
            "age",
            "site",
            "survival_months_dss",
            "censorship_dss",
            "is_female",
            "oncotree_code",
            "train",
        ]
    ].rename(
        columns={
            "survival_months_dss": "survival_months",
            "censorship_dss": "censorship",
        }
    )
    rna = rna.rename(
        columns={
            column: f"{column}_rnaseq"
            for column in rna.columns
            if column != "case_id"
        }
    )
    table = metadata.merge(rna, on="case_id", how="inner", validate="many_to_one")
    if table["case_id"].nunique() < 290:
        raise RuntimeError(
            "Too few paired COREAD cases after joining metadata and RNA: "
            f"{table['case_id'].nunique()}"
        )
    return table


def install_for_repo(
    table: pd.DataFrame,
    project_dir: Path,
    repo: Path,
    csv_dir_name: str | None,
) -> None:
    split_dir = repo / "splits/5foldcv/tcga_coadread"
    split_dir.mkdir(parents=True, exist_ok=True)
    if csv_dir_name is not None:
        csv_dir = repo / csv_dir_name
        csv_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(csv_dir / "tcga_coadread_all_clean.csv", index=False)

    source_splits = project_dir / "splits/5folds/tcga_coadread"
    for fold in range(5):
        shutil.copyfile(
            source_splits / f"splits_{fold}.csv",
            split_dir / f"splits_{fold}.csv",
        )


def main() -> None:
    args = parse_args()
    table = build_table(args.project_dir)
    # SurvPath's released COREAD folds contain one additional paired case
    # (TCGA-F5-6861). Replace them so every model uses MRePath's exact folds.
    install_for_repo(table, args.project_dir, args.survpath_repo, None)
    install_for_repo(table, args.project_dir, args.cmta_repo, "csv")
    install_for_repo(table, args.project_dir, args.porpoise_repo, "datasets_csv")
    print(
        "Prepared legacy baseline inputs: "
        f"{table['case_id'].nunique()} cases, "
        f"{len(table)} slide rows, "
        f"{len(table.columns) - 10} RNA features."
    )


if __name__ == "__main__":
    main()
