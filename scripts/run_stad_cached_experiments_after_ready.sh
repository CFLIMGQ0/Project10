#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${MREPATH_PYTHON:-/home/administrator/miniconda3/envs/mrepath-train/bin/python}"
PREPARE_UNIT="mrepath-stad-cached-resnet50-prepare-v2-20260731.service"
COADREAD_UNITS=(
  "mrepath-coadread-numworkers-benchmark-20260801.service"
  "mrepath-coadread-kan-sequential-auto-workers-cached-20260801.service"
  "mrepath-coadread-quality-conflict-gate-20260801.service"
  "mrepath-coadread-ablations-after-gate-20260801.service"
)
NUM_WORKERS_FILE="${PROJECT_ROOT}/configs/selected_num_workers.txt"
GATE_STATUS_FILE="${PROJECT_ROOT}/configs/coadread_quality_conflict_gate.txt"
STAD_ROOT="${PROJECT_ROOT}/data/tcga_stad/clam_20x_resnet50_paper_k9"
COADREAD_RESULTS="${PROJECT_ROOT}/results/results_coadread_kan_variants_cached_20260731"

cd "${PROJECT_ROOT}"

coadread_active() {
  local unit
  for unit in "${COADREAD_UNITS[@]}"; do
    if systemctl --user is-active --quiet "${unit}"; then
      return 0
    fi
  done
  return 1
}

while systemctl --user is-active --quiet "${PREPARE_UNIT}" \
  || coadread_active; do
  echo "[wait] STAD cache preparation or COREAD KAN variants still active"
  sleep 60
done

graph_count="$(find "${STAD_ROOT}/graph_files" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l)"
cache_count="$(find "${STAD_ROOT}/hypergraph_cache" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l)"
if [[ "${graph_count}" -lt 300 ]] || [[ "${cache_count}" -ne "${graph_count}" ]]; then
  echo "[error] STAD graph/cache inventory is incomplete: graphs=${graph_count} cache=${cache_count}"
  exit 1
fi

if [[ -f "${GATE_STATUS_FILE}" ]] \
  && grep -q '^fail,' "${GATE_STATUS_FILE}"; then
  echo "[gate] Quality + Conflict did not reach 0.7400 on COREAD; " \
       "STAD module ablations remain paused"
  exit 0
fi

complete_coadread=0
while IFS= read -r summary; do
  folds="$(tail -n +2 "${summary}" | cut -d, -f1 | sort -u | wc -l)"
  if [[ "${folds}" -eq 5 ]]; then
    complete_coadread=$((complete_coadread + 1))
  fi
done < <(find "${COADREAD_RESULTS}" -type f -name summary.csv 2>/dev/null)
if [[ "${complete_coadread}" -ne 5 ]]; then
  echo "[error] expected five complete COREAD KAN variant summaries, found ${complete_coadread}"
  exit 1
fi

num_workers="2"
if [[ -f "${NUM_WORKERS_FILE}" ]]; then
  read -r num_workers < "${NUM_WORKERS_FILE}"
fi
if [[ ! "${num_workers}" =~ ^(2|4|6|8)$ ]]; then
  echo "[error] invalid selected num_workers: ${num_workers}"
  exit 1
fi
echo "[config] using selected num_workers=${num_workers}"

"${PYTHON_BIN}" -u scripts/run_cached_multicohort_experiments.py \
  --dataset stad \
  --group all \
  --results-root results_stad_cached_all_smoke_20260731 \
  --max-parallel 5 \
  --max-epochs 1 \
  --num-workers "${num_workers}" \
  --folds 0 \
  --fail-fast

"${PYTHON_BIN}" -u scripts/run_cached_multicohort_experiments.py \
  --dataset stad \
  --group all \
  --results-root results_stad_cached_all_20260731 \
  --max-parallel 5 \
  --max-epochs 30 \
  --num-workers "${num_workers}" \
  --fail-fast
