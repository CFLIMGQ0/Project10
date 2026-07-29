#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${MREPATH_PYTHON:-/home/administrator/miniconda3/envs/mrepath-train/bin/python}"
RESULTS_ROOT="${MREPATH_ABLATION_RESULTS:-results_coadread_paper_ablations_20260726}"

cd "${PROJECT_DIR}"
export PYTHONDONTWRITEBYTECODE=1

echo "[pipeline] smoke tests started at $(date --iso-8601=seconds)"
"${PYTHON_BIN}" scripts/run_coadread_paper_ablations.py \
  --mode smoke \
  --results-root "${RESULTS_ROOT}" \
  --include-reference \
  --fail-fast

echo "[pipeline] formal sequential five-fold runs started at $(date --iso-8601=seconds)"
"${PYTHON_BIN}" scripts/run_coadread_paper_ablations.py \
  --mode formal \
  --results-root "${RESULTS_ROOT}" \
  --fail-fast

echo "[pipeline] available configurations completed at $(date --iso-8601=seconds)"
