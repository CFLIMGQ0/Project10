#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${MREPATH_PYTHON:-/home/administrator/miniconda3/envs/mrepath-train/bin/python}"
PIBD_REPO="${MREPATH_PIBD_REPO:-/home/administrator/.cache/mrepath/third_party/PIBD}"
DATA_ROOT="${MREPATH_PIBD_DATA_ROOT:-/home/administrator/.cache/mrepath/tcga_coadread/pibd_resnet50}"
RESULTS_ROOT="${MREPATH_PIBD_RESULTS:-${PROJECT_DIR}/results_coadread_pibd_resnet}"
BATCH_SIZE="${MREPATH_PIBD_BATCH_SIZE:-32}"

mkdir -p "${RESULTS_ROOT}"
cd "${PIBD_REPO}"
export PYTHONDONTWRITEBYTECODE=1

exec "${PYTHON_BIN}" main.py \
  --study tcga_coadread \
  --task survival \
  --which_splits 5foldcv \
  --type_of_path combine \
  --mode resnet50 \
  --data_root_dir "${DATA_ROOT}" \
  --label_file "${PROJECT_DIR}/datasets_csv/metadata/tcga_coadread.csv" \
  --omics_dir "${PROJECT_DIR}/datasets_csv/raw_rna_data/combine/coadread" \
  --results_dir "${RESULTS_ROOT}" \
  --batch_size "${BATCH_SIZE}" \
  --lr 0.0005 \
  --opt adam \
  --reg 0.001 \
  --seed 1 \
  --alpha_surv 0.5 \
  --max_epochs 30 \
  --encoding_dim 1024 \
  --label_col survival_months_dss \
  --k 5 \
  --bag_loss nll_surv \
  --n_classes 4 \
  --num_patches 4096 \
  --wsi_projection_dim 256 \
  --omics_format pathways \
  "$@"
