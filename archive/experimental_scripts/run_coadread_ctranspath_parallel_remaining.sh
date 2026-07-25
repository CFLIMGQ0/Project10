#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${MREPATH_PYTHON:-/home/administrator/miniconda3/envs/mrepath-train/bin/python}"
BASE_RESULTS="${PROJECT_DIR}/results_coadread_mrepath_ctranspath_samecoords_paper_strict"
RESULTS_ROOT="${MREPATH_CTRANSPATH_PARALLEL_RESULTS:-${PROJECT_DIR}/results_coadread_mrepath_ctranspath_parallel}"
LOG_DIR="${MREPATH_CTRANSPATH_PARALLEL_LOG_DIR:-${PROJECT_DIR}/logs/coadread_mrepath_ctranspath_parallel}"
THREADS="${MREPATH_TRAIN_THREADS_PER_FOLD:-3}"

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
  local fold="$1"
  local worker="worker${fold}"
  local worker_results="${RESULTS_ROOT}/${worker}"
  local worker_log="${LOG_DIR}/${worker}.log"

  echo "[parallel] launching ${worker}: fold ${fold}"
  OMP_NUM_THREADS="${THREADS}" \
    MKL_NUM_THREADS="${THREADS}" \
    OPENBLAS_NUM_THREADS="${THREADS}" \
    MREPATH_CTRANSPATH_RESULTS="${worker_results}" \
    bash scripts/run_coadread_ctranspath.sh \
      --k_start "${fold}" \
      --k_end "$((fold + 1))" \
      >"${worker_log}" 2>&1 &
  LAST_PID=$!
  pids+=("${LAST_PID}")
}

# Three simultaneous MRePath models stay within the 16 GiB GPU budget.
launch_worker 1
worker1_pid="${LAST_PID}"
sleep "${MREPATH_FOLD_LAUNCH_DELAY:-15}"
launch_worker 2
worker2_pid="${LAST_PID}"
sleep "${MREPATH_FOLD_LAUNCH_DELAY:-15}"
launch_worker 3
worker3_pid="${LAST_PID}"

# Keep at most three models resident. Fold 4 starts when Fold 1 frees a slot.
failed=0
wait "${worker1_pid}" || failed=1
if (( failed == 0 )); then
  launch_worker 4
  worker4_pid="${LAST_PID}"
else
  worker4_pid=""
fi

wait "${worker2_pid}" || failed=1
wait "${worker3_pid}" || failed=1
if [[ -n "${worker4_pid}" ]]; then
  wait "${worker4_pid}" || failed=1
fi

if (( failed != 0 )); then
  echo "[parallel] at least one fold worker failed; inspect ${LOG_DIR}"
  exit 1
fi

"${PYTHON_BIN}" - <<PY
from pathlib import Path
import os
import pandas as pd

base = Path(r"${BASE_RESULTS}")
root = Path(r"${RESULTS_ROOT}")
base_summaries = list(base.glob("**/summary.csv"))
worker_summaries = sorted(root.glob("worker*/**/summary.csv"))
if len(base_summaries) != 1:
    raise RuntimeError(f"Expected one base Fold-0 summary, found {len(base_summaries)}")
if len(worker_summaries) != 4:
    raise RuntimeError(f"Expected four worker summaries, found {len(worker_summaries)}")

fold0 = pd.read_csv(base_summaries[0])
fold0 = fold0[fold0["fold"].astype(int) == 0]
summary = pd.concat(
    [fold0] + [pd.read_csv(path) for path in worker_summaries],
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
