#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${MREPATH_PYTHON:-/home/administrator/miniconda3/envs/mrepath-train/bin/python}"
PIBD_REPO="${MREPATH_PIBD_REPO:-/home/administrator/.cache/mrepath/third_party/PIBD}"
PIBD_COMMIT="bd5bd94e6f8d48e7679c6e68209a3a65c9e56a78"
DATA_ROOT="${MREPATH_PIBD_OFFICIAL_DATA_ROOT:-/home/administrator/.cache/mrepath/tcga_coadread/pibd_official/extracted/coadread/tiles-l1-s224/feats-l1-s224-CTransPath-sampler}"
RESULTS_ROOT="${MREPATH_PIBD_OFFICIAL_RESULTS:-${PROJECT_DIR}/results_coadread_pibd_official_ctranspath}"

actual_commit="$(git -C "${PIBD_REPO}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${PIBD_COMMIT}" ]]; then
  echo "[pibd-official] expected commit ${PIBD_COMMIT}, found ${actual_commit}"
  exit 1
fi

if [[ ! -d "${DATA_ROOT}/pt_files" ]]; then
  echo "[pibd-official] missing official CTransPath features: ${DATA_ROOT}/pt_files"
  exit 1
fi

mkdir -p "${RESULTS_ROOT}"
cd "${PIBD_REPO}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

exec "${PYTHON_BIN}" main.py \
  --study tcga_coadread \
  --task survival \
  --which_splits 5foldcv \
  --type_of_path combine \
  --seed 1 \
  --data_root_dir "${DATA_ROOT}" \
  --label_file "${PIBD_REPO}/datasets_csv/metadata/tcga_coadread.csv" \
  --omics_dir "${PIBD_REPO}/datasets_csv/raw_rna_data/combine/coadread" \
  --results_dir "${RESULTS_ROOT}"
