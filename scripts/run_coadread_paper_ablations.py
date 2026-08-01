#!/usr/bin/env python3
"""Run the paper's unique COREAD MRePath ablations sequentially and resumably."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "configs" / "coadread_paper_ablation_matrix.json"
DEFAULT_PYTHON = Path(
    os.environ.get(
        "MREPATH_PYTHON",
        "/home/administrator/miniconda3/envs/mrepath-train/bin/python",
    )
)
DEFAULT_RESNET_ROOT = (
    ROOT / "data" / "tcga_coadread" / "clam_20x_resnet50_paper_k9"
)
ENCODER_ROOTS = {
    "resnet50": DEFAULT_RESNET_ROOT,
    "ctranspath": Path(
        "/home/administrator/.cache/mrepath/tcga_coadread/"
        "clam_20x_ctranspath_samecoords"
    ),
    "uni": Path(
        "/home/administrator/.cache/mrepath/tcga_coadread/"
        "clam_20x_uni_samecoords"
    ),
    "conch": Path(
        "/home/administrator/.cache/mrepath/tcga_coadread/"
        "clam_20x_conch_samecoords"
    ),
    "phikon2": Path(
        "/home/administrator/.cache/mrepath/tcga_coadread/"
        "clam_20x_phikon2_samecoords"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "formal"], required=True)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=MATRIX_PATH,
        help="Ablation or combined-candidate JSON matrix to run.",
    )
    parser.add_argument(
        "--results-root",
        default="results_coadread_paper_ablations_20260726",
    )
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument(
        "--disable-hypergraph-cache",
        action="store_true",
        help="Use the original online DHG hypergraph construction path.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def load_configs(matrix_path: Path = MATRIX_PATH) -> list[dict]:
    matrix = json.loads(matrix_path.read_text())
    reference = matrix["shared_reference"]
    configs = []
    for entry in matrix["configs"]:
        merged = {
            "graph_type": reference["graph_type"],
            "hyperedges": reference["hyperedges"],
            "graph_k": reference["graph_k"],
            "weighting": reference["weighting"],
            "path_weight": 0.5,
            "gene_weight": 0.5,
            "fusion": reference["fusion"],
            "gene_aggregation": reference["gene_aggregation"],
            "encoder": reference["encoder"],
            "encoding_dim": 1024,
            "rebalance_variant": reference.get(
                "rebalance_variant", "original"
            ),
            "modality_dropout": reference.get("modality_dropout", 0.0),
            "monotonicity_weight": reference.get(
                "monotonicity_weight", 0.0
            ),
            "monotonicity_margin": reference.get(
                "monotonicity_margin", 0.02
            ),
            "unimodal_loss_weight": reference.get(
                "unimodal_loss_weight", 0.0
            ),
            "mismatch_loss_weight": reference.get(
                "mismatch_loss_weight", 0.0
            ),
        }
        merged.update(entry)
        configs.append(merged)
    return configs


def data_root(config: dict) -> Path:
    if config["encoder"] != "resnet50":
        return ENCODER_ROOTS[config["encoder"]]
    if config["graph_k"] == 9:
        return DEFAULT_RESNET_ROOT
    return (
        ROOT
        / "data"
        / "tcga_coadread"
        / f"clam_20x_resnet50_paper_k{config['graph_k']}"
    )


def graph_inventory_ready(path: Path) -> bool:
    graph_dir = path / "graph_files"
    return graph_dir.is_dir() and sum(1 for _ in graph_dir.glob("*.pt")) >= 300


def locate_summary(results_root: str, config_name: str) -> Path | None:
    config_root = ROOT / "results" / results_root / config_name
    summaries = sorted(config_root.glob("*/summary.csv"))
    if len(summaries) > 1:
        raise RuntimeError(f"Multiple summaries found for {config_name}: {summaries}")
    return summaries[0] if summaries else None


def completed_folds(summary: Path | None) -> set[int]:
    if summary is None or not summary.is_file():
        return set()
    table = pd.read_csv(summary)
    if "fold" not in table:
        return set()
    return {int(value) for value in table["fold"].dropna()}


def command_for(
    config: dict,
    results_root: str,
    fold: int,
    mode: str,
    use_hypergraph_cache: bool = True,
) -> list[str]:
    smoke = mode == "smoke"
    command = [
        str(DEFAULT_PYTHON),
        "main.py",
        "--study",
        "tcga_coadread",
        "--task",
        "survival",
        "--which_splits",
        "5folds",
        "--type_of_path",
        "combine",
        "--modality",
        "hgnn",
        "--data_root_dir",
        str(data_root(config)),
        "--label_file",
        str(ROOT / "datasets_csv" / "metadata" / "tcga_coadread.csv"),
        "--omics_dir",
        str(ROOT / "datasets_csv" / "raw_rna_data" / "combine" / "coadread"),
        "--results_dir",
        f"{results_root}/{config['name']}",
        "--batch_size",
        "1",
        "--num_workers",
        "0",
        "--lr",
        "0.0001",
        "--opt",
        "adam",
        "--reg",
        "0.00001",
        "--seed",
        "1",
        "--alpha_surv",
        "0.0",
        "--max_epochs",
        "1" if smoke else "30",
        "--encoding_dim",
        str(config["encoding_dim"]),
        "--label_col",
        "survival_months_dss",
        "--k",
        "5",
        "--k_start",
        str(fold),
        "--k_end",
        str(fold + 1),
        "--bag_loss",
        "nll_surv",
        "--n_classes",
        "4",
        "--num_patches",
        "64" if smoke else "4096",
        "--wsi_projection_dim",
        "256",
        "--fusion",
        "concat",
        "--lr_scheduler",
        "constant",
        "--warmup_epochs",
        "0",
        "--checkpoint_selection",
        "best",
        "--mrepath_graph_type",
        config["graph_type"],
        "--mrepath_hyperedges",
        config["hyperedges"],
        "--mrepath_weighting",
        config["weighting"],
        "--mrepath_path_weight",
        str(config["path_weight"]),
        "--mrepath_gene_weight",
        str(config["gene_weight"]),
        "--mrepath_fusion",
        config["fusion"],
        "--mrepath_gene_aggregation",
        config["gene_aggregation"],
        "--mrepath_encoder",
        config["encoder"],
        "--mrepath_rebalance_variant",
        config["rebalance_variant"],
        "--mrepath_modality_dropout",
        str(config["modality_dropout"]),
        "--mrepath_monotonicity_weight",
        str(config["monotonicity_weight"]),
        "--mrepath_monotonicity_margin",
        str(config["monotonicity_margin"]),
        "--mrepath_unimodal_loss_weight",
        str(config["unimodal_loss_weight"]),
        "--mrepath_mismatch_loss_weight",
        str(config["mismatch_loss_weight"]),
    ]
    if use_hypergraph_cache and config["graph_type"] in {"hgnn", "shgnn"}:
        command.extend(
            [
                "--mrepath_hypergraph_cache_dir",
                str(data_root(config) / "hypergraph_cache"),
            ]
        )
    return command


def main() -> int:
    args = parse_args()
    results_root = (
        args.results_root + "_smoke"
        if args.mode == "smoke" and not args.results_root.endswith("_smoke")
        else args.results_root
    )
    configs = load_configs(args.matrix)
    if args.only:
        requested = set(args.only)
        configs = [config for config in configs if config["name"] in requested]
        missing = requested - {config["name"] for config in configs}
        if missing:
            raise ValueError(f"Unknown config names: {sorted(missing)}")

    failures = 0
    deferred = []
    for index, config in enumerate(configs, start=1):
        name = config["name"]
        if name == "full_reference" and not (
            args.include_reference or args.mode == "smoke"
        ):
            print("[reuse] full_reference formal five-fold result already exists", flush=True)
            continue
        root = data_root(config)
        if not graph_inventory_ready(root):
            deferred.append(name)
            print(f"[defer] {name}: graph inventory is not ready at {root}", flush=True)
            continue

        folds = [0] if args.mode == "smoke" else list(range(5))
        done = completed_folds(locate_summary(results_root, name))
        print(
            f"[config] {index}/{len(configs)} {name} "
            f"mode={args.mode} completed={sorted(done)}",
            flush=True,
        )
        for fold in folds:
            if fold in done:
                print(f"[skip] {name} fold={fold}", flush=True)
                continue
            command = command_for(
                config,
                results_root,
                fold,
                args.mode,
                use_hypergraph_cache=not args.disable_hypergraph_cache,
            )
            print("[run] " + " ".join(command), flush=True)
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode:
                failures += 1
                print(
                    f"[failed] {name} fold={fold} exit={result.returncode}",
                    flush=True,
                )
                if args.fail_fast:
                    return result.returncode
                break
            print(f"[complete] {name} fold={fold}", flush=True)

    print(
        f"[suite] mode={args.mode} failures={failures} "
        f"deferred={deferred}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
