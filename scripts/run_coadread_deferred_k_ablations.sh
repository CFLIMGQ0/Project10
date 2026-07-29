#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${MREPATH_PYTHON:-/home/administrator/miniconda3/envs/mrepath-train/bin/python}"
RESULTS_ROOT="${MREPATH_ABLATION_RESULTS:-results_coadread_paper_ablations_20260726}"
MAIN_UNIT="${MREPATH_ABLATION_UNIT:-mrepath-coadread-paper-ablations-v2.service}"
GRAPH_UNIT="${MREPATH_GRAPH_UNIT:-mrepath-coadread-k-graph-prep.service}"
CONFIGS=(graph_k5 graph_k25 graph_k49)

cd "${PROJECT_DIR}"
export PYTHONDONTWRITEBYTECODE=1

while systemctl --user is-active --quiet "${MAIN_UNIT}"; do
  sleep 30
done

if systemctl --user is-failed --quiet "${MAIN_UNIT}"; then
  echo "[deferred-k] refusing to continue because ${MAIN_UNIT} failed"
  exit 1
fi

while systemctl --user is-active --quiet "${GRAPH_UNIT}"; do
  sleep 30
done

if systemctl --user is-failed --quiet "${GRAPH_UNIT}"; then
  echo "[deferred-k] refusing to continue because ${GRAPH_UNIT} failed"
  exit 1
fi

echo "[deferred-k] smoke tests started at $(date --iso-8601=seconds)"
"${PYTHON_BIN}" scripts/run_coadread_paper_ablations.py \
  --mode smoke \
  --results-root "${RESULTS_ROOT}" \
  --only "${CONFIGS[@]}" \
  --fail-fast

echo "[deferred-k] formal five-fold runs started at $(date --iso-8601=seconds)"
"${PYTHON_BIN}" scripts/run_coadread_paper_ablations.py \
  --mode formal \
  --results-root "${RESULTS_ROOT}" \
  --only "${CONFIGS[@]}" \
  --fail-fast

echo "[deferred-k] completed at $(date --iso-8601=seconds)"
