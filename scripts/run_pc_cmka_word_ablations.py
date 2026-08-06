#!/usr/bin/env python3
"""Sequential, resumable runner for the PC-CMKA experimental suites."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

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

# User-facing graph-structure names from model.yaml. Numerical details remain
# in the method configuration, while this map guarantees that every displayed
# choice resolves to a distinct executable preset.
GRAPH_STRUCTURE_PRESETS = {
    "fixed_fold_graph": "C0_original_ddkac",
    "reference_operator": "S_A1_reference_operator",
    "patient_degree_edge_gate": "C1_patient_degree_edge_gate",
    "reference_degree_edge_gate": "C2_reference_degree_edge_gate",
    "direct_edge_gate": "S_A2_direct_edge_gate",
    "direct_low_rank": "A1_direct_coefficients",
    "inverse_calibration": "S_A3_inverse_calibration",
    "inverse_fixed_rho": "A2_fixed_rho",
    "inverse_random_probe": "A3_random_probe",
    "bernoulli_views": "C4_bernoulli_views",
    "effective_resistance_views": "C3_effective_resistance",
    "independent_views": "S_A5_independent_views",
    "isotropic_antithetic": "A4_isotropic_uncertainty",
    "hessian_antithetic": "S_A6_hessian_antithetic",
    "pc_cmka_full_graph": "A0_full",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument(
        "--config", default=str(ROOT / "configs/pc_cmka_ddkac_word.json")
    )
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument(
        "--graph-structures",
        nargs="*",
        choices=tuple(GRAPH_STRUCTURE_PRESETS),
        default=None,
        help=(
            "Run graph_structure choices from model.yaml. Supplying the flag "
            "without names runs all 15 choices."
        ),
    )
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-patches", type=int, default=4096)
    parser.add_argument("--results-root", default="results_pc_cmka_word_5fold")
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument(
        "--suite",
        choices=("word", "staged", "controls", "all"),
        default="word",
        help="Word Table 5, prompt-staged, classical controls, or every suite.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def available(config_path: Path, suite: str, include_controls: bool) -> list[dict]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    entries = []
    if suite in {"word", "all"}:
        entries.extend(raw["experiments"])
    if suite in {"staged", "all"}:
        entries.extend(raw.get("staged_experiments", []))
    if suite in {"controls", "all"} or include_controls:
        entries = list(raw.get("controls", [])) + entries
    return entries


def selected(
    config_path: Path,
    names: list[str] | None,
    suite: str,
    include_controls: bool,
) -> list[dict]:
    lookup_suite = "all" if names is not None else suite
    entries = available(config_path, lookup_suite, include_controls)
    if names is None:
        return entries
    mapping = {item["name"]: item for item in entries}
    missing = sorted(set(names) - set(mapping))
    if missing:
        raise ValueError(f"unknown experiments: {missing}")
    return [mapping[name] for name in names]


def selected_graph_structures(
    config_path: Path, names: list[str] | None
) -> list[dict]:
    requested = names or list(GRAPH_STRUCTURE_PRESETS)
    target_names = [GRAPH_STRUCTURE_PRESETS[name] for name in requested]
    entries = selected(config_path, target_names, "all", True)
    by_name = {entry["name"]: entry for entry in entries}
    resolved = []
    for graph_name, target_name in zip(requested, target_names):
        entry = dict(by_name[target_name])
        entry["result_name"] = graph_name
        entry["graph_structure"] = graph_name
        resolved.append(entry)
    return resolved


def fold_complete(results_dir: Path, fold: int) -> bool:
    if any(results_dir.glob(f"**/split_{fold}_results.pkl")):
        return True
    aggregate = results_dir / "fold_metrics.csv"
    if aggregate.is_file():
        try:
            with aggregate.open(newline="", encoding="utf-8") as handle:
                if any(
                    int(row.get("fold", -1)) == fold
                    for row in csv.DictReader(handle)
                ):
                    return True
        except Exception:
            pass
    for summary in results_dir.glob("**/summary.csv"):
        try:
            with summary.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except Exception:
            continue
        if any(int(row.get("fold", -1)) == fold for row in rows):
            return True
    return False


def record_fold_summary(results_dir: Path, fold: int) -> None:
    """Preserve all folds because each one-fold main.py call rewrites summary.csv."""

    matching = []
    for summary in results_dir.glob("**/summary.csv"):
        try:
            with summary.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except Exception:
            continue
        matching.extend(row for row in rows if int(row.get("fold", -1)) == fold)
    if not matching:
        raise RuntimeError(f"completed fold {fold} has no readable summary.csv")
    aggregate = results_dir / "fold_metrics.csv"
    existing = []
    if aggregate.is_file():
        with aggregate.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    rows_by_fold = {int(row["fold"]): row for row in existing}
    rows_by_fold[fold] = matching[-1]
    rows = [rows_by_fold[index] for index in sorted(rows_by_fold)]
    fieldnames = list(rows[0])
    aggregate.parent.mkdir(parents=True, exist_ok=True)
    with aggregate.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def command(
    dataset: dict,
    config_path: Path,
    experiment: dict,
    fold: int,
    args: argparse.Namespace,
    results_dir: Path,
) -> list[str]:
    encoder = experiment.get("encoder", "pc_cmka_ddkac")
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
                "--pc_cmka_experiment", experiment["name"],
            ]
        )
    return values


def main() -> int:
    args = parse_args()
    if args.graph_structures is not None and args.experiments is not None:
        raise ValueError("use either --graph-structures or --experiments, not both")
    if any(fold not in range(5) for fold in args.folds):
        raise ValueError("folds must be in [0, 4]")
    config_path = Path(args.config).resolve()
    if args.graph_structures is not None:
        experiments = selected_graph_structures(config_path, args.graph_structures)
    else:
        experiments = selected(
            config_path, args.experiments, args.suite, args.include_controls
        )
    dataset = DATASETS[args.dataset]
    for experiment in experiments:
        result_name = experiment.get("result_name", experiment["name"])
        results_dir = Path(args.results_root) / args.dataset / result_name
        completed_root = ROOT / "results" / results_dir
        for fold in args.folds:
            if fold_complete(completed_root, fold):
                print(f"skip completed {result_name} fold {fold}")
                continue
            values = command(
                dataset, config_path, experiment, fold, args, results_dir
            )
            print(" ".join(values), flush=True)
            if not args.dry_run:
                subprocess.run(values, cwd=ROOT, check=True)
                record_fold_summary(completed_root, fold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
