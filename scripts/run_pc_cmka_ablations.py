#!/usr/bin/env python3
"""Sequential, resumable PC-CMKA-DDKAC fold/ablation runner."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/administrator/miniconda3/envs/mrepath-train/bin/python")
DATASETS = {
    "coadread": {
        "study": "tcga_coadread",
        "data": ROOT / "data/tcga_coadread/clam_20x_resnet50_paper_k9",
        "labels": ROOT / "datasets_csv/metadata/tcga_coadread.csv",
        "omics": ROOT / "datasets_csv/raw_rna_data/combine/coadread",
    },
    "stad": {
        "study": "tcga_stad",
        "data": ROOT / "data/tcga_stad/clam_20x_resnet50_paper_k9",
        "labels": ROOT / "datasets_csv/metadata/tcga_stad.csv",
        "omics": ROOT / "datasets_csv/raw_rna_data/combine/stad",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument(
        "--config", default=str(ROOT / "configs/pc_cmka_ddkac.json")
    )
    parser.add_argument("--ablations", nargs="*", default=None)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-patches", type=int, default=4096)
    parser.add_argument("--results-root", default="results_pc_cmka_ablations")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_ablations(config_path: Path, names: list[str] | None) -> list[dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    available = config["ablations"]
    if names is None:
        return available
    by_name = {item["name"]: item for item in available}
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise ValueError(f"unknown ablations: {missing}")
    return [by_name[name] for name in names]


def fold_complete(results_dir: Path, fold: int) -> bool:
    for summary in results_dir.glob("**/summary.csv"):
        frame = pd.read_csv(summary)
        if "fold" in frame and fold in frame["fold"].astype(int).tolist():
            return True
    return False


def command(
    dataset: dict,
    config_path: Path,
    ablation: dict,
    fold: int,
    args: argparse.Namespace,
    results_dir: Path,
) -> list[str]:
    encoder = (
        "dd_kac"
        if ablation.get("encoder") == "dd_kac"
        else "pc_cmka_ddkac"
    )
    values = [
        str(PYTHON), "-u", "main.py",
        "--study", dataset["study"],
        "--task", "survival",
        "--which_splits", "5folds",
        "--type_of_path", "combine",
        "--modality", "hgnn",
        "--data_root_dir", str(dataset["data"]),
        "--label_file", str(dataset["labels"]),
        "--omics_dir", str(dataset["omics"]),
        "--results_dir", str(results_dir),
        "--batch_size", "1",
        "--num_workers", str(args.num_workers),
        "--lr", "0.0001",
        "--opt", "adam",
        "--reg", "0.00001",
        "--seed", "1",
        "--alpha_surv", "0.0",
        "--max_epochs", str(args.max_epochs),
        "--encoding_dim", "1024",
        "--label_col", "survival_months_dss",
        "--k", "5",
        "--k_start", str(fold),
        "--k_end", str(fold + 1),
        "--bag_loss", "nll_surv",
        "--n_classes", "4",
        "--num_patches", str(args.num_patches),
        "--wsi_projection_dim", "256",
        "--fusion", "concat",
        "--lr_scheduler", "constant",
        "--warmup_epochs", "0",
        "--checkpoint_selection", "best",
        "--mrepath_graph_type", "shgnn",
        "--mrepath_hyperedges", "both",
        "--mrepath_weighting", "dynamic",
        "--mrepath_fusion", "ifa",
        "--mrepath_gene_aggregation", "default",
        "--mrepath_genomic_encoder", encoder,
        "--mrepath_rebalance_variant", "original",
        "--mrepath_hypergraph_cache_dir", str(dataset["data"] / "hypergraph_cache"),
        "--fold_survival_bins",
    ]
    if encoder == "pc_cmka_ddkac":
        values.extend(
            [
                "--pc_cmka_config", str(config_path),
                "--pc_cmka_ablation", ablation["name"],
                "--pc_cmka_diagnostics_dir", str(results_dir / "diagnostics"),
            ]
        )
    return values


def main() -> int:
    args = parse_args()
    if any(fold not in range(5) for fold in args.folds):
        raise ValueError("folds must be in [0, 4]")
    dataset = DATASETS[args.dataset]
    config_path = Path(args.config).resolve()
    ablations = selected_ablations(config_path, args.ablations)
    for ablation in ablations:
        results_dir = ROOT / args.results_root / args.dataset / ablation["name"]
        results_dir.mkdir(parents=True, exist_ok=True)
        for fold in args.folds:
            if fold_complete(results_dir, fold):
                print(f"skip completed {ablation['name']} fold {fold}")
                continue
            values = command(
                dataset, config_path, ablation, fold, args, results_dir
            )
            print(" ".join(values))
            if not args.dry_run:
                subprocess.run(values, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
