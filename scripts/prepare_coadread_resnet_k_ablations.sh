#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${MREPATH_PREPROCESS_PYTHON:-/home/administrator/miniconda3/envs/mrepath-preprocess/bin/python}"
SOURCE_ROOT="${PROJECT_DIR}/data/tcga_coadread/clam_20x_resnet50"
METADATA="${PROJECT_DIR}/datasets_csv/metadata/tcga_coadread.csv"

cd "${PROJECT_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${MREPATH_GRAPH_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${MREPATH_GRAPH_THREADS:-2}"
export MKL_NUM_THREADS="${MREPATH_GRAPH_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${MREPATH_GRAPH_THREADS:-2}"

for radius in 5 25 49; do
  output="${PROJECT_DIR}/data/tcga_coadread/clam_20x_resnet50_paper_k${radius}/graph_files"
  echo "[graphs] radius=${radius} started at $(date --iso-8601=seconds)"
  "${PYTHON_BIN}" scripts/build_wsi_graphs.py \
    --h5-dir "${SOURCE_ROOT}/h5_files" \
    --output-dir "${output}" \
    --metadata-csv "${METADATA}" \
    --radius "${radius}" \
    --feature-dim 1024 \
    --spatial-space l2 \
    --feature-space cosinesimil \
    --fail-fast
  echo "[graphs] radius=${radius} completed at $(date --iso-8601=seconds)"
done
