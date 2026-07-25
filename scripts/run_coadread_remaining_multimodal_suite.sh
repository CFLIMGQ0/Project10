#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_ROOT="${MREPATH_BASELINE_RESULTS:-${PROJECT_DIR}/results_coadread_multimodal_20260725}"

mkdir -p "${RESULTS_ROOT}"

bash "${SCRIPT_DIR}/run_coadread_legacy_multimodal.sh"
bash "${SCRIPT_DIR}/run_coadread_pibd_resnet.sh" \
    2>&1 | tee -a "${RESULTS_ROOT}/pibd.log"

MREPATH_PAPER_RESULTS="${RESULTS_ROOT}/mrepath" \
    bash "${SCRIPT_DIR}/run_coadread.sh" \
    2>&1 | tee -a "${RESULTS_ROOT}/mrepath.log"
