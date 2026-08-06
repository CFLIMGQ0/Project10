#!/usr/bin/env python3
"""Run cached MRePath configurations in parallel without summary-file races."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "configs" / "cached_coadread_stad_experiments.json"
DEFAULT_PYTHON = Path(
    os.environ.get(
        "MREPATH_PYTHON",
        "/home/administrator/miniconda3/envs/mrepath-train/bin/python",
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["coadread", "stad"], required=True)
    parser.add_argument(
        "--group",
        choices=[
            "base_six",
            "kan_variants",
            "improved_kan",
            "improved_kan_quality_conflict",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--folds", nargs="*", type=int, default=list(range(5)))
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def load_plan(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    matrix = json.loads(args.matrix.read_text())
    dataset = matrix["datasets"][args.dataset]
    defaults = matrix["defaults"]
    experiments = []
    for entry in matrix["experiments"]:
        if args.dataset not in entry["datasets"]:
            continue
        if args.group != "all" and entry["group"] != args.group:
            continue
        config = dict(defaults)
        config.update(entry)
        experiments.append(config)
    if args.only:
        requested = set(args.only)
        experiments = [
            config for config in experiments if config["name"] in requested
        ]
        missing = requested - {config["name"] for config in experiments}
        if missing:
            raise ValueError(f"Unknown or unavailable configs: {sorted(missing)}")
    if not experiments:
        raise ValueError("No experiments matched the requested dataset/group")
    return dataset, experiments


def completed_folds(results_root: str, name: str) -> set[int]:
    root = ROOT / "results" / results_root / name
    summaries = list(root.glob("*/summary.csv"))
    if len(summaries) > 1:
        raise RuntimeError(f"Multiple summary files found for {name}: {summaries}")
    if not summaries:
        return set()
    table = pd.read_csv(summaries[0])
    if "fold" not in table:
        return set()
    return {int(value) for value in table["fold"].dropna()}


def verify_dataset(dataset: dict) -> tuple[Path, Path]:
    data_root = ROOT / dataset["data_root"]
    graph_dir = data_root / "graph_files"
    cache_dir = data_root / "hypergraph_cache"
    graph_count = len(list(graph_dir.glob("*.pt")))
    cache_count = len(list(cache_dir.glob("*.pt")))
    if graph_count == 0:
        raise FileNotFoundError(f"No graph files in {graph_dir}")
    if cache_count != graph_count:
        raise RuntimeError(
            f"Incomplete hypergraph cache: graphs={graph_count}, "
            f"cache={cache_count}, directory={cache_dir}"
        )
    return data_root, cache_dir


def command_for(
    dataset: dict,
    config: dict,
    results_root: str,
    fold: int,
    max_epochs: int,
    data_root: Path,
    cache_dir: Path,
    batch_size: int,
    num_workers: int,
) -> list[str]:
    return [
        str(DEFAULT_PYTHON),
        "-u",
        "main.py",
        "--study",
        dataset["study"],
        "--task",
        "survival",
        "--which_splits",
        dataset["split_name"],
        "--type_of_path",
        "combine",
        "--modality",
        "hgnn",
        "--data_root_dir",
        str(data_root),
        "--label_file",
        str(ROOT / dataset["label_file"]),
        "--omics_dir",
        str(ROOT / dataset["omics_dir"]),
        "--results_dir",
        f"{results_root}/{config['name']}",
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
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
        str(max_epochs),
        "--encoding_dim",
        "1024",
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
        "4096",
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
        "both",
        "--mrepath_weighting",
        "dynamic",
        "--mrepath_path_weight",
        "0.5",
        "--mrepath_gene_weight",
        "0.5",
        "--mrepath_fusion",
        "ifa",
        "--mrepath_gene_aggregation",
        config["gene_aggregation"],
        "--mrepath_genomic_encoder",
        config.get("genomic_encoder", "original"),
        "--mrepath_encoder",
        "resnet50",
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
        "--mrepath_hypergraph_cache_dir",
        str(cache_dir),
    ]


def main() -> int:
    args = parse_args()
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be positive")
    if not args.folds or any(fold not in range(5) for fold in args.folds):
        raise ValueError("--folds must contain values from 0 through 4")
    dataset, configs = load_plan(args)
    data_root, cache_dir = verify_dataset(dataset)
    log_root = ROOT / "logs" / args.results_root
    log_root.mkdir(parents=True, exist_ok=True)

    queues: dict[str, list[int]] = {}
    by_name = {config["name"]: config for config in configs}
    for config in configs:
        done = completed_folds(args.results_root, config["name"])
        queues[config["name"]] = [
            fold for fold in args.folds if fold not in done
        ]
        print(
            f"[queue] {config['name']} completed={sorted(done)} "
            f"pending={queues[config['name']]}",
            flush=True,
        )

    active: dict[str, tuple[subprocess.Popen, object, int]] = {}
    failures = 0
    while any(queues.values()) or active:
        for name in sorted(queues):
            if len(active) >= args.max_parallel:
                break
            if name in active or not queues[name]:
                continue
            fold = queues[name].pop(0)
            command = command_for(
                dataset,
                by_name[name],
                args.results_root,
                fold,
                args.max_epochs,
                data_root,
                cache_dir,
                args.batch_size,
                args.num_workers,
            )
            log_path = log_root / f"{name}_fold{fold}.log"
            handle = log_path.open("a", buffering=1)
            print(
                f"[launch] {name} fold={fold} log={log_path}",
                flush=True,
            )
            handle.write("[command] " + " ".join(command) + "\n")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            active[name] = (process, handle, fold)

        if not active:
            break
        time.sleep(5)
        for name, (process, handle, fold) in list(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            del active[name]
            if returncode == 0:
                print(f"[complete] {name} fold={fold}", flush=True)
            else:
                failures += 1
                queues[name].clear()
                print(
                    f"[failed] {name} fold={fold} exit={returncode}",
                    flush=True,
                )
                if args.fail_fast:
                    for child, child_handle, _ in active.values():
                        child.terminate()
                        child_handle.close()
                    return returncode

    print(f"[suite] failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
