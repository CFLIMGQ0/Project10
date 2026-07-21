#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${MREPATH_PYTHON:-/home/administrator/miniconda3/envs/mrepath-train/bin/python}"

cd "${PROJECT_DIR}"
export PYTHONDONTWRITEBYTECODE=1

exec "${PYTHON_BIN}" main.py \
  --study tcga_coadread \
  --task survival \
  --which_splits 5folds \
  --type_of_path combine \
  --modality hgnn \
  --data_root_dir "${PROJECT_DIR}/data/tcga_coadread/clam_20x_resnet50" \
  --label_file "${PROJECT_DIR}/datasets_csv/metadata/tcga_coadread.csv" \
  --omics_dir "${PROJECT_DIR}/datasets_csv/raw_rna_data/combine/coadread" \
  --results_dir "${PROJECT_DIR}/results_coadread_released" \
  --batch_size 1 \
  --num_workers 0 \
  --lr 0.0001 \
  --opt adam \
  --reg 0.00001 \
  --seed 1 \
  --alpha_surv 0.5 \
  --weighted_sample \
  --max_epochs 30 \
  --encoding_dim 1024 \
  --label_col survival_months_dss \
  --k 5 \
  --bag_loss nll_surv \
  --n_classes 4 \
  --num_patches 4096 \
  --wsi_projection_dim 256 \
  --fusion concat \
  --lr_scheduler cosine \
  --warmup_epochs 1 \
  "$@"
