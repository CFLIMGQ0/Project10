#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${MREPATH_PYTHON:-/home/administrator/miniconda3/envs/mrepath-train/bin/python}"
RESULTS_ROOT="${MREPATH_PARALLEL_RESULTS:-${PROJECT_DIR}/results_coadread_level0_256_repo_parallel}"
LOG_DIR="${MREPATH_PARALLEL_LOG_DIR:-${PROJECT_DIR}/logs/coadread_level0_repo_parallel}"

mkdir -p "${RESULTS_ROOT}" "${LOG_DIR}"
cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

pids=()

stop_workers() {
  local pid
  for pid in "${pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
}
trap stop_workers TERM INT

launch_worker() {
  local worker="$1"
  local start="$2"
  local end="$3"
  local worker_results="${RESULTS_ROOT}/${worker}"
  local worker_log="${LOG_DIR}/${worker}.log"

  echo "[parallel] launching ${worker}: folds ${start}..$((end - 1))"
  OMP_NUM_THREADS="${MREPATH_TRAIN_THREADS_PER_FOLD:-2}" \
    MKL_NUM_THREADS="${MREPATH_TRAIN_THREADS_PER_FOLD:-2}" \
    OPENBLAS_NUM_THREADS="${MREPATH_TRAIN_THREADS_PER_FOLD:-2}" \
    MREPATH_LEVEL0_RESULTS="${worker_results}" \
    bash scripts/run_coadread_level0_repo.sh \
      --k_start "${start}" \
      --k_end "${end}" \
      >"${worker_log}" 2>&1 &
  pids+=("$!")
}

launch_worker worker0 0 1
sleep "${MREPATH_FOLD_LAUNCH_DELAY:-20}"
launch_worker worker1 1 2
sleep "${MREPATH_FOLD_LAUNCH_DELAY:-20}"
launch_worker worker2 2 3
sleep "${MREPATH_FOLD_LAUNCH_DELAY:-20}"
launch_worker worker3 3 4
sleep "${MREPATH_FOLD_LAUNCH_DELAY:-20}"
launch_worker worker4 4 5

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done

if (( failed != 0 )); then
  echo "[parallel] at least one fold worker failed; inspect ${LOG_DIR}"
  exit 1
fi

"${PYTHON_BIN}" - <<PY
from pathlib import Path
import os
import pandas as pd

root = Path(r"${RESULTS_ROOT}")
summary_files = sorted(root.glob("worker*/**/summary.csv"))
if len(summary_files) != 5:
    raise RuntimeError(f"Expected 5 worker summaries, found {len(summary_files)}")

summary = pd.concat(
    [pd.read_csv(path) for path in summary_files],
    ignore_index=True,
)
if sorted(summary["fold"].astype(int).tolist()) != list(range(5)):
    raise RuntimeError(
        f"Expected folds 0..4 exactly once, got {summary['fold'].tolist()}"
    )

summary = summary.sort_values("fold").reset_index(drop=True)
temporary = root / "summary.csv.tmp"
summary.to_csv(temporary, index=False)
os.replace(temporary, root / "summary.csv")

print(summary.to_string(index=False))
print(f"[parallel] mean val_cindex={summary['val_cindex'].mean():.6f}")
print(f"[parallel] std val_cindex={summary['val_cindex'].std():.6f}")
PY

echo "[parallel] completed at $(date --iso-8601=seconds)"
