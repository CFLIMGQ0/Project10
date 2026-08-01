#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${MREPATH_PYTHON:-/home/administrator/miniconda3/envs/mrepath-train/bin/python}"
MATRIX="${PROJECT_ROOT}/configs/coadread_quality_conflict_ablations.json"
RESULTS_ROOT="results_coadread_quality_conflict_ablations_20260731"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m unittest \
  tests.test_paper_modules.QualityConflictRebalanceTests

"${PYTHON_BIN}" scripts/run_coadread_paper_ablations.py \
  --mode smoke \
  --matrix "${MATRIX}" \
  --results-root "${RESULTS_ROOT}" \
  --disable-hypergraph-cache \
  --fail-fast

"${PYTHON_BIN}" scripts/run_coadread_paper_ablations.py \
  --mode formal \
  --matrix "${MATRIX}" \
  --results-root "${RESULTS_ROOT}" \
  --disable-hypergraph-cache \
  --fail-fast
