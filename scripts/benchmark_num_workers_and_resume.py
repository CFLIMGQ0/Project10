#!/usr/bin/env python3
"""Benchmark COREAD DataLoader worker counts, select the fastest, and resume."""

from __future__ import annotations

import csv
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/administrator/miniconda3/envs/mrepath-train/bin/python")
PRODUCTION_RESULTS = (
    ROOT / "results/results_coadread_kan_variants_cached_20260731"
)
BENCHMARK_LOG_DIR = ROOT / "logs/num_workers_benchmark_20260801"
BENCHMARK_RESULTS_PREFIX = "benchmark_num_workers_20260801"
SELECTED_FILE = ROOT / "configs/selected_num_workers.txt"
CSV_FILE = ROOT / "logs/num_workers_benchmark_20260801.csv"
OLD_UNIT = "mrepath-coadread-kan-sequential-cached-20260801.service"
NEW_UNIT = (
    "mrepath-coadread-kan-sequential-auto-workers-cached-20260801"
)
STAD_UNIT = "mrepath-stad-cached-experiments-after-ready-20260801.service"


def conflict_fold_four_complete() -> bool:
    summaries = list(
        (PRODUCTION_RESULTS / "kan_conflict_only").glob("*/summary.csv")
    )
    if len(summaries) != 1:
        return False
    with summaries[0].open(newline="") as handle:
        return any(row.get("fold") == "4" for row in csv.DictReader(handle))


def benchmark(round_index: int, workers: int) -> tuple[float, int]:
    result_root = (
        f"{BENCHMARK_RESULTS_PREFIX}/round{round_index}_workers{workers}"
    )
    log_path = BENCHMARK_LOG_DIR / f"round{round_index}_workers{workers}.log"
    command = [
        str(PYTHON),
        "-u",
        "scripts/run_cached_multicohort_experiments.py",
        "--dataset",
        "coadread",
        "--group",
        "kan_variants",
        "--results-root",
        result_root,
        "--max-parallel",
        "1",
        "--max-epochs",
        "1",
        "--num-workers",
        str(workers),
        "--folds",
        "3",
        "--only",
        "kan_quality_conflict",
    ]
    started = time.monotonic()
    with log_path.open("w") as handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return time.monotonic() - started, completed.returncode


def main() -> int:
    BENCHMARK_LOG_DIR.mkdir(parents=True, exist_ok=True)
    while not conflict_fold_four_complete():
        print("[wait] current production fold is still running", flush=True)
        time.sleep(20)

    subprocess.run(
        ["systemctl", "--user", "stop", OLD_UNIT],
        check=False,
    )

    rows: list[dict[str, int | float]] = []
    orders = ((2, 4, 6, 8), (8, 6, 4, 2))
    for round_index, order in enumerate(orders, start=1):
        for workers in order:
            print(
                f"[benchmark] round={round_index} num_workers={workers}",
                flush=True,
            )
            seconds, returncode = benchmark(round_index, workers)
            rows.append(
                {
                    "round": round_index,
                    "num_workers": workers,
                    "seconds": seconds,
                    "returncode": returncode,
                }
            )
            print(
                f"[benchmark] workers={workers} seconds={seconds:.2f} "
                f"exit={returncode}",
                flush=True,
            )

    with CSV_FILE.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    means: dict[int, float] = {}
    for workers in (2, 4, 6, 8):
        successful = [
            float(row["seconds"])
            for row in rows
            if row["num_workers"] == workers and row["returncode"] == 0
        ]
        if len(successful) == 2:
            means[workers] = sum(successful) / len(successful)
    if not means:
        raise RuntimeError("Every num_workers benchmark failed")

    selected = min(means, key=means.get)
    SELECTED_FILE.write_text(f"{selected}\n")
    print(f"[selected] num_workers={selected} means={means}", flush=True)

    launch = [
        "systemd-run",
        "--user",
        f"--unit={NEW_UNIT}",
        "--description=Continue cached COREAD variants with selected workers",
        "--property=WorkingDirectory=/mnt/e/MRePath",
        "--property=StandardOutput=append:/mnt/e/MRePath/logs/"
        "coadread_kan_sequential_auto_workers_cached_20260801.log",
        "--property=StandardError=append:/mnt/e/MRePath/logs/"
        "coadread_kan_sequential_auto_workers_cached_20260801.log",
        str(PYTHON),
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
        str(selected),
        "--fail-fast",
        "--only",
        "kan_quality_conflict",
    ]
    subprocess.run(launch, cwd=ROOT, check=True)
    subprocess.run(
        ["systemctl", "--user", "restart", STAD_UNIT],
        check=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
