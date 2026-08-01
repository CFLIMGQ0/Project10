#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${MREPATH_PYTHON:-/home/administrator/miniconda3/envs/mrepath-train/bin/python}"
QUALITY_UNIT="mrepath-coadread-quality-conflict-only-b1-w8-20260801.service"
STAD_PREP_UNIT="mrepath-stad-cached-resnet50-prepare-v2-20260731.service"
QUALITY_ROOT="${PROJECT_ROOT}/results/results_coadread_kan_variants_cached_20260731"
STAD_ROOT="${PROJECT_ROOT}/data/tcga_stad/clam_20x_resnet50_paper_k9"
SELECTION_FILE="${PROJECT_ROOT}/configs/selected_batch_workers.txt"

cd "${PROJECT_ROOT}"

batch_size="$(sed -n 's/^batch_size=//p' "${SELECTION_FILE}")"
num_workers="$(sed -n 's/^num_workers=//p' "${SELECTION_FILE}")"
if [[ "${batch_size}" != "1" ]] || [[ ! "${num_workers}" =~ ^[0-9]+$ ]]; then
  echo "[error] invalid selected batch/worker configuration"
  exit 1
fi

while systemctl --user is-active --quiet "${QUALITY_UNIT}"; do
  echo "[wait] Quality + Conflict is still active"
  sleep 60
done

quality_summary="$(find "${QUALITY_ROOT}/kan_quality_conflict" -type f -name summary.csv -print -quit 2>/dev/null || true)"
quality_folds="0"
if [[ -n "${quality_summary}" ]]; then
  quality_folds="$(tail -n +2 "${quality_summary}" | cut -d, -f1 | sort -u | wc -l)"
fi
if [[ "${quality_folds}" -ne 5 ]]; then
  echo "[error] Quality + Conflict did not finish five folds: ${quality_folds}"
  exit 1
fi

echo "[stage] COREAD five improved KAN-like encoder smoke tests"
"${PYTHON_BIN}" -u scripts/run_cached_multicohort_experiments.py \
  --dataset coadread \
  --group improved_kan \
  --results-root results_coadread_improved_kan_smoke_20260801 \
  --max-parallel 1 \
  --max-epochs 1 \
  --batch-size "${batch_size}" \
  --num-workers "${num_workers}" \
  --folds 0 \
  --fail-fast

echo "[stage] COREAD five improved KAN-like encoder formal five-fold runs"
"${PYTHON_BIN}" -u scripts/run_cached_multicohort_experiments.py \
  --dataset coadread \
  --group improved_kan \
  --results-root results_coadread_improved_kan_5fold_20260801 \
  --max-parallel 1 \
  --max-epochs 30 \
  --batch-size "${batch_size}" \
  --num-workers "${num_workers}" \
  --fail-fast

while systemctl --user is-active --quiet "${STAD_PREP_UNIT}"; do
  echo "[wait] STAD graph/cache preparation is still active"
  sleep 60
done

graph_count="$(find "${STAD_ROOT}/graph_files" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l || true)"
cache_count="$(find "${STAD_ROOT}/hypergraph_cache" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l || true)"
if [[ "${graph_count}" -lt 300 ]] || [[ "${cache_count}" -ne "${graph_count}" ]]; then
  echo "[error] STAD graph/cache incomplete: graphs=${graph_count} cache=${cache_count}"
  exit 1
fi

echo "[stage] STAD five improved KAN-like encoder smoke tests"
"${PYTHON_BIN}" -u scripts/run_cached_multicohort_experiments.py \
  --dataset stad \
  --group improved_kan \
  --results-root results_stad_improved_kan_smoke_20260801 \
  --max-parallel 1 \
  --max-epochs 1 \
  --batch-size "${batch_size}" \
  --num-workers "${num_workers}" \
  --folds 0 \
  --fail-fast

echo "[stage] STAD five improved KAN-like encoder formal five-fold runs"
"${PYTHON_BIN}" -u scripts/run_cached_multicohort_experiments.py \
  --dataset stad \
  --group improved_kan \
  --results-root results_stad_improved_kan_5fold_20260801 \
  --max-parallel 1 \
  --max-epochs 30 \
  --batch-size "${batch_size}" \
  --num-workers "${num_workers}" \
  --fail-fast

echo "[complete] COREAD and STAD improved KAN-like encoder suites finished"
