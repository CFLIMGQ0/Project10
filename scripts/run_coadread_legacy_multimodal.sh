#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${MREPATH_PYTHON:-/home/administrator/miniconda3/envs/mrepath-train/bin/python}"
CACHE_ROOT="${MREPATH_THIRD_PARTY_ROOT:-/home/administrator/.cache/mrepath/third_party}"
CMTA_REPO="${MREPATH_CMTA_REPO:-${CACHE_ROOT}/CMTA}"
PORPOISE_REPO="${MREPATH_PORPOISE_REPO:-${CACHE_ROOT}/PORPOISE}"
FEATURE_ROOT="${MREPATH_PIBD_DATA_ROOT:-/home/administrator/.cache/mrepath/tcga_coadread/pibd_resnet50}"
RESULTS_ROOT="${MREPATH_BASELINE_RESULTS:-${PROJECT_DIR}/results_coadread_multimodal_20260725}"
EPOCHS="${MREPATH_BASELINE_EPOCHS:-30}"
SEED="${MREPATH_BASELINE_SEED:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore

mkdir -p "${RESULTS_ROOT}"

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/prepare_coadread_legacy_baselines.py"

echo "[baseline] starting cmta at $(date --iso-8601=seconds)"
(
    cd "${CMTA_REPO}"
    "${PYTHON_BIN}" main.py \
        --dataset tcga_coadread \
        --data_root_dir "${FEATURE_ROOT}" \
        --results_dir "${RESULTS_ROOT}/cmta" \
        --which_splits 5foldcv \
        --modal coattn \
        --model cmta \
        --num_epoch "${EPOCHS}" \
        --batch_size 1 \
        --loss nll_surv_l1 \
        --lr 0.0001 \
        --weight_decay 0.00001 \
        --optimizer Adam \
        --scheduler None \
        --alpha 1.0 \
        --OOM 4096 \
        --seed "${SEED}"
) 2>&1 | tee -a "${RESULTS_ROOT}/cmta.log"
echo "[baseline] finished cmta at $(date --iso-8601=seconds)"

run_porpoise_model() {
    local name="$1"
    local mode="$2"
    local model_type="$3"
    local fusion="$4"

    mkdir -p "${RESULTS_ROOT}/${name}"
    echo "[baseline] starting ${name} at $(date --iso-8601=seconds)"
    (
        cd "${PORPOISE_REPO}"
        "${PYTHON_BIN}" main.py \
            --data_root_dir "${FEATURE_ROOT}" \
            --seed "${SEED}" \
            --k 5 \
            --results_dir "${RESULTS_ROOT}/${name}" \
            --which_splits 5foldcv \
            --split_dir tcga_coadread \
            --model_type "${model_type}" \
            --mode "${mode}" \
            --fusion "${fusion}" \
            --max_epochs "${EPOCHS}" \
            --lr 0.0001 \
            --reg 0.00001 \
            --alpha_surv 0.0 \
            --bag_loss nll_surv \
            --gc 32 \
            --overwrite
    ) 2>&1 | tee -a "${RESULTS_ROOT}/${name}.log"
    echo "[baseline] finished ${name} at $(date --iso-8601=seconds)"
}

run_porpoise_model porpoise pathomic porpoise_mmf bilinear
run_porpoise_model snn_component omic snn None
run_porpoise_model clam_component path porpoise_amil None

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/summarize_snn_clam_ensemble.py" \
    --snn-root "${RESULTS_ROOT}/snn_component" \
    --clam-root "${RESULTS_ROOT}/clam_component" \
    --output-dir "${RESULTS_ROOT}/snn_clam"
