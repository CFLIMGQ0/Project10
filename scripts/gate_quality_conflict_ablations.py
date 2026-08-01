#!/usr/bin/env python3
"""Gate remaining ablations on five-fold Quality + Conflict performance."""

from __future__ import annotations

import csv
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/administrator/miniconda3/envs/mrepath-train/bin/python"
RESULTS = ROOT / "results/results_coadread_kan_variants_cached_20260731"
STATUS_FILE = ROOT / "configs/coadread_quality_conflict_gate.txt"
WORKERS_FILE = ROOT / "configs/selected_num_workers.txt"
RUNNING_UNIT = (
    "mrepath-coadread-kan-sequential-auto-workers-cached-20260801.service"
)
FOLLOWUP_UNIT = "mrepath-coadread-ablations-after-gate-20260801"
STAD_UNIT = "mrepath-stad-cached-experiments-after-ready-20260801.service"
THRESHOLD = 0.74


def completed_scores() -> list[float]:
    summaries = list((RESULTS / "kan_quality_conflict").glob("*/summary.csv"))
    if len(summaries) != 1:
        return []
    with summaries[0].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    folds = {int(row["fold"]): float(row["val_cindex"]) for row in rows}
    if set(folds) != set(range(5)):
        return []
    return [folds[index] for index in range(5)]


def main() -> int:
    scores: list[float] = []
    while not scores:
        scores = completed_scores()
        if not scores:
            print("[wait] Quality + Conflict five-fold is incomplete", flush=True)
            time.sleep(2)

    subprocess.run(
        ["systemctl", "--user", "stop", RUNNING_UNIT],
        check=False,
    )
    mean_cindex = sum(scores) / len(scores)
    passed = mean_cindex >= THRESHOLD
    STATUS_FILE.write_text(
        f"{'pass' if passed else 'fail'},{mean_cindex:.8f}\n"
    )
    print(
        f"[gate] scores={scores} mean={mean_cindex:.6f} "
        f"threshold={THRESHOLD:.4f} passed={passed}",
        flush=True,
    )

    if passed:
        workers = int(WORKERS_FILE.read_text().strip())
        command = [
            "systemd-run",
            "--user",
            f"--unit={FOLLOWUP_UNIT}",
            "--description=Run COREAD ablations after Quality Conflict gate",
            "--property=WorkingDirectory=/mnt/e/MRePath",
            "--property=StandardOutput=append:/mnt/e/MRePath/logs/"
            "coadread_ablations_after_gate_20260801.log",
            "--property=StandardError=append:/mnt/e/MRePath/logs/"
            "coadread_ablations_after_gate_20260801.log",
            PYTHON,
            "-u",
            "scripts/run_cached_multicohort_experiments.py",
            "--dataset",
            "coadread",
            "--group",
            "kan_variants",
            "--results-root",
            "results_coadread_kan_variants_cached_20260731",
            "--max-parallel",
            "1",
            "--max-epochs",
            "30",
            "--num-workers",
            str(workers),
            "--fail-fast",
            "--only",
            "kan_quality_only",
            "kan_conflict_only",
            "kan_quality_conflict_no_dropout",
            "kan_quality_conflict_no_monotonicity",
        ]
        subprocess.run(command, cwd=ROOT, check=True)

    subprocess.run(
        ["systemctl", "--user", "restart", STAD_UNIT],
        check=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
