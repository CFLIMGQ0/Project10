#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${MREPATH_PYTHON:-/home/administrator/miniconda3/envs/mrepath-train/bin/python}"
THIRD_PARTY_ROOT="${MREPATH_THIRD_PARTY_ROOT:-/home/administrator/.cache/mrepath/third_party}"
FEATURE_ROOT="${MREPATH_PIBD_DATA_ROOT:-/home/administrator/.cache/mrepath/tcga_coadread/pibd_resnet50}"
RESULTS_ROOT="${MREPATH_SMOKE_RESULTS:-${PROJECT_DIR}/results_coadread_multimodal_smoke_20260725}"
SUMMARY="${RESULTS_ROOT}/status.tsv"
SELECTED_MODELS="${MREPATH_SMOKE_MODELS:-cmta porpoise snn_component clam_component pibd mrepath}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore

mkdir -p "${RESULTS_ROOT}"
printf "model\tstatus\tlog\n" > "${SUMMARY}"

run_check() {
    local name="$1"
    shift
    if [[ " ${SELECTED_MODELS} " != *" ${name} "* ]]; then
        return 0
    fi
    local log="${RESULTS_ROOT}/${name}.log"
    echo "[smoke] ${name} started at $(date --iso-8601=seconds)"
    if "$@" >"${log}" 2>&1; then
        printf "%s\tPASS\t%s\n" "${name}" "${log}" >> "${SUMMARY}"
        echo "[smoke] ${name}: PASS"
    else
        local code=$?
        printf "%s\tFAIL(%s)\t%s\n" "${name}" "${code}" "${log}" >> "${SUMMARY}"
        echo "[smoke] ${name}: FAIL(${code})"
        tail -n 30 "${log}"
    fi
}

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/prepare_coadread_legacy_baselines.py"

run_check cmta bash -lc "
    cd '${THIRD_PARTY_ROOT}/CMTA' &&
    '${PYTHON_BIN}' main.py \
        --dataset tcga_coadread \
        --data_root_dir '${FEATURE_ROOT}' \
        --results_dir '${RESULTS_ROOT}/cmta' \
        --which_splits 5foldcv \
        --modal coattn \
        --model cmta \
        --num_epoch 1 \
        --batch_size 1 \
        --loss nll_surv_l1 \
        --lr 0.0001 \
        --weight_decay 0.00001 \
        --optimizer Adam \
        --scheduler None \
        --alpha 1.0 \
        --OOM 4096 \
        --seed 1
"

run_porpoise_smoke() {
    local name="$1"
    local mode="$2"
    local model_type="$3"
    local fusion="$4"
    run_check "${name}" bash -lc "
        cd '${THIRD_PARTY_ROOT}/PORPOISE' &&
        '${PYTHON_BIN}' main.py \
            --data_root_dir '${FEATURE_ROOT}' \
            --seed 1 \
            --k 5 \
            --k_start 0 \
            --k_end 1 \
            --results_dir '${RESULTS_ROOT}/${name}' \
            --which_splits 5foldcv \
            --split_dir tcga_coadread \
            --model_type '${model_type}' \
            --mode '${mode}' \
            --fusion '${fusion}' \
            --max_epochs 1 \
            --lr 0.0001 \
            --reg 0.00001 \
            --alpha_surv 0.0 \
            --bag_loss nll_surv \
            --gc 32 \
            --overwrite
    "
}

run_porpoise_smoke porpoise pathomic porpoise_mmf bilinear
run_porpoise_smoke snn_component omic snn None
run_porpoise_smoke clam_component path porpoise_amil None

run_check pibd bash -lc "
    cd '${THIRD_PARTY_ROOT}/PIBD' &&
    '${PYTHON_BIN}' main.py \
        --study tcga_coadread \
        --task survival \
        --which_splits 5foldcv \
        --type_of_path combine \
        --mode resnet50 \
        --data_root_dir '${FEATURE_ROOT}' \
        --label_file '${PROJECT_DIR}/datasets_csv/metadata/tcga_coadread.csv' \
        --omics_dir '${PROJECT_DIR}/datasets_csv/raw_rna_data/combine/coadread' \
        --results_dir '${RESULTS_ROOT}/pibd' \
        --batch_size 32 \
        --lr 0.0001 \
        --opt adam \
        --reg 0.00001 \
        --seed 1 \
        --alpha_surv 0.0 \
        --max_epochs 1 \
        --encoding_dim 1024 \
        --label_col survival_months_dss \
        --k 5 \
        --k_start 0 \
        --k_end 1 \
        --bag_loss nll_surv \
        --n_classes 4 \
        --num_patches 4096 \
        --wsi_projection_dim 256 \
        --omics_format pathways
"

run_check mrepath bash -lc "
    cd '${PROJECT_DIR}' &&
    MREPATH_PAPER_RESULTS='${RESULTS_ROOT}/mrepath' \
        bash scripts/run_coadread.sh \
            --k_start 0 \
            --k_end 1 \
            --max_epochs 1
"

echo
cat "${SUMMARY}"
if rg -q $'\tFAIL\\(' "${SUMMARY}"; then
    exit 1
fi
