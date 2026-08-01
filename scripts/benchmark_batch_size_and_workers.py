#!/usr/bin/env python3
"""After Quality+Conflict finishes, benchmark batch size and workers."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/administrator/miniconda3/envs/mrepath-train/bin/python"
PRODUCTION_RESULTS = (
    ROOT / "results/results_coadread_kan_variants_cached_20260731"
)
RUNNING_UNIT = (
    "mrepath-coadread-kan-sequential-auto-workers-cached-20260801.service"
)
BENCHMARK_ROOT = "benchmark_batch_workers_20260801"
LOG_DIR = ROOT / "logs/benchmark_batch_workers_20260801"
BATCH_CSV = ROOT / "logs/batch_size_benchmark_20260801.csv"
WORKER_CSV = ROOT / "logs/num_workers_0_16_benchmark_20260801.csv"
SELECTION_FILE = ROOT / "configs/selected_batch_workers.txt"


def quality_conflict_complete() -> bool:
    summaries = list(
        (PRODUCTION_RESULTS / "kan_quality_conflict").glob("*/summary.csv")
    )
    if len(summaries) != 1:
        return False
    with summaries[0].open(newline="") as handle:
        folds = {int(row["fold"]) for row in csv.DictReader(handle)}
    return folds == set(range(5))


def run_benchmark(
    phase: str,
    batch_size: int,
    num_workers: int,
) -> dict[str, int | float | str]:
    name = f"{phase}_batch{batch_size}_workers{num_workers}"
    result_root = f"{BENCHMARK_ROOT}/{name}"
    log_path = LOG_DIR / f"{name}.log"
    command = [
        PYTHON,
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
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
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
    seconds = time.monotonic() - started
    return {
        "phase": phase,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "seconds": seconds,
        "returncode": completed.returncode,
        "log": str(log_path.relative_to(ROOT)),
    }


def write_csv(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Skip the Quality+Conflict completion wait and benchmark now.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    while not args.run_now and not quality_conflict_complete():
        print("[wait] Quality + Conflict five-fold is incomplete", flush=True)
        time.sleep(2)

    # The production runner has more ablations queued in memory. Stop it as
    # soon as the requested full model has all five recorded folds.
    subprocess.run(
        ["systemctl", "--user", "stop", RUNNING_UNIT],
        check=False,
    )

    batch_rows: list[dict[str, int | float | str]] = []
    for batch_size in (1, 2, 4, 8):
        print(f"[batch] testing batch_size={batch_size}", flush=True)
        row = run_benchmark("batch", batch_size, 6)
        batch_rows.append(row)
        print(
            f"[batch] batch_size={batch_size} seconds={row['seconds']:.2f} "
            f"exit={row['returncode']}",
            flush=True,
        )
    write_csv(BATCH_CSV, batch_rows)

    successful_batches = [
        row for row in batch_rows if int(row["returncode"]) == 0
    ]
    if not successful_batches:
        raise RuntimeError("Every batch-size benchmark failed")
    selected_batch = int(
        min(successful_batches, key=lambda row: float(row["seconds"]))[
            "batch_size"
        ]
    )

    # Interleave low/high worker counts so filesystem page-cache warming does
    # not monotonically favour larger values.
    worker_order = (0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15, 16)
    worker_rows: list[dict[str, int | float | str]] = []
    for num_workers in worker_order:
        print(
            f"[workers] testing batch_size={selected_batch} "
            f"num_workers={num_workers}",
            flush=True,
        )
        row = run_benchmark("workers", selected_batch, num_workers)
        worker_rows.append(row)
        print(
            f"[workers] num_workers={num_workers} "
            f"seconds={row['seconds']:.2f} exit={row['returncode']}",
            flush=True,
        )
    write_csv(WORKER_CSV, sorted(worker_rows, key=lambda row: int(row["num_workers"])))

    successful_workers = [
        row for row in worker_rows if int(row["returncode"]) == 0
    ]
    if not successful_workers:
        raise RuntimeError("Every num-workers benchmark failed")
    selected_worker = int(
        min(successful_workers, key=lambda row: float(row["seconds"]))[
            "num_workers"
        ]
    )
    selected_seconds = min(
        float(row["seconds"]) for row in successful_workers
    )
    SELECTION_FILE.write_text(
        f"batch_size={selected_batch}\n"
        f"num_workers={selected_worker}\n"
        f"seconds_per_epoch={selected_seconds:.6f}\n"
    )
    print(
        f"[selected] batch_size={selected_batch} "
        f"num_workers={selected_worker} seconds={selected_seconds:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
