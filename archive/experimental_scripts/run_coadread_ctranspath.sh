#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${MREPATH_PYTHON:-/home/administrator/miniconda3/envs/mrepath-train/bin/python}"
DATA_ROOT="${MREPATH_CTRANSPATH_DATA_ROOT:-/home/administrator/.cache/mrepath/tcga_coadread/clam_20x_ctranspath_samecoords}"
RESULTS_ROOT="${MREPATH_CTRANSPATH_RESULTS:-${PROJECT_DIR}/results_coadread_mrepath_ctranspath_samecoords_paper_strict}"

cd "${PROJECT_DIR}"
export PYTHONDONTWRITEBYTECODE=1

# Serialize duplicate launches of the same single fold. This lets an
# externally-started fold finish while a scheduler waits, then makes the
# scheduler skip it once split_<fold>_results.pkl is complete.
fold_start=""
fold_end=""
expect_value=""
for argument in "$@"; do
  if [[ "${expect_value}" == "start" ]]; then
    fold_start="${argument}"
    expect_value=""
  elif [[ "${expect_value}" == "end" ]]; then
    fold_end="${argument}"
    expect_value=""
  elif [[ "${argument}" == "--k_start" ]]; then
    expect_value="start"
  elif [[ "${argument}" == "--k_end" ]]; then
    expect_value="end"
  fi
done

if [[ "${fold_start}" =~ ^[0-9]+$ ]] \
  && [[ "${fold_end}" =~ ^[0-9]+$ ]] \
  && (( fold_end == fold_start + 1 )); then
  mkdir -p "${RESULTS_ROOT}"
  exec 8>"${RESULTS_ROOT}/.fold_${fold_start}.lock"
  flock 8
  existing_result="$(
    find "${RESULTS_ROOT}" -type f \
      -name "split_${fold_start}_results.pkl" -print -quit
  )"
  if [[ -n "${existing_result}" ]]; then
    echo "[train] fold ${fold_start} already completed: ${existing_result}"
    exit 0
  fi
fi

# This intentionally matches scripts/run_coadread.sh.  Only the pathology
# feature root and encoder input dimension change for the backbone ablation.
exec "${PYTHON_BIN}" main.py \
  --study tcga_coadread \
  --task survival \
  --which_splits 5folds \
  --type_of_path combine \
  --modality hgnn \
  --data_root_dir "${DATA_ROOT}" \
  --label_file "${PROJECT_DIR}/datasets_csv/metadata/tcga_coadread.csv" \
  --omics_dir "${PROJECT_DIR}/datasets_csv/raw_rna_data/combine/coadread" \
  --results_dir "${RESULTS_ROOT}" \
  --batch_size 1 \
  --num_workers 0 \
  --lr 0.0001 \
  --opt adam \
  --reg 0.00001 \
  --seed 1 \
  --alpha_surv 0.0 \
  --max_epochs 30 \
  --encoding_dim 768 \
  --label_col survival_months_dss \
  --k 5 \
  --bag_loss nll_surv \
  --n_classes 4 \
  --num_patches 4096 \
  --wsi_projection_dim 256 \
  --fusion concat \
  --lr_scheduler constant \
  --warmup_epochs 0 \
  --checkpoint_selection final \
  "$@"
