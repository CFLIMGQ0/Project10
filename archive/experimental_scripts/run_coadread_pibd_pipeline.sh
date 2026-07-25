#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${MREPATH_PYTHON:-/home/administrator/miniconda3/envs/mrepath-train/bin/python}"
PIBD_REPO="${MREPATH_PIBD_REPO:-/home/administrator/.cache/mrepath/third_party/PIBD}"
PIBD_COMMIT="bd5bd94e6f8d48e7679c6e68209a3a65c9e56a78"
H5_DIR="${MREPATH_ABMIL_CACHE:-/home/administrator/.cache/mrepath/tcga_coadread/clam_20x_resnet50}/h5_files"
DATA_ROOT="${MREPATH_PIBD_DATA_ROOT:-/home/administrator/.cache/mrepath/tcga_coadread/pibd_resnet50}"
LOG_DIR="${PROJECT_DIR}/logs"
LOCK_FILE="${LOG_DIR}/coadread_pibd_5fold.lock"
PID_FILE="${LOG_DIR}/coadread_pibd_5fold.pid"
MIN_GPU_FREE_MB="${MREPATH_PIBD_MIN_GPU_FREE_MB:-8500}"
MIN_RAM_MB="${MREPATH_PIBD_MIN_RAM_MB:-4096}"

mkdir -p "${LOG_DIR}" "$(dirname "${PIBD_REPO}")" "${DATA_ROOT}/pt_files"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[pibd] another COREAD PIBD pipeline already holds ${LOCK_FILE}"
  exit 1
fi
echo "$$" > "${PID_FILE}"
trap 'rm -f "${PID_FILE}"' EXIT

if [[ ! -d "${PIBD_REPO}/.git" ]]; then
  echo "[pibd] cloning official PIBD repository"
  git clone https://github.com/zylbuaa/PIBD.git "${PIBD_REPO}"
fi
actual_commit="$(git -C "${PIBD_REPO}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${PIBD_COMMIT}" ]]; then
  echo "[pibd] expected commit ${PIBD_COMMIT}, found ${actual_commit}"
  exit 1
fi

echo "[pibd] preparing tensor feature cache"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/cache_pibd_resnet_features.py" \
  --h5-dir "${H5_DIR}" \
  --output-dir "${DATA_ROOT}/pt_files" \
  --metadata "${PROJECT_DIR}/datasets_csv/metadata/tcga_coadread.csv"

while [[ -f "${LOG_DIR}/coadread_abmil_5fold.pid" ]]; do
  abmil_pid="$(cat "${LOG_DIR}/coadread_abmil_5fold.pid")"
  if ! kill -0 "${abmil_pid}" 2>/dev/null; then
    break
  fi
  echo "[pibd] waiting for ABMIL pipeline PID ${abmil_pid}"
  sleep 60
done

wait_for_resources() {
  while true; do
    available_ram_mb="$(awk '/MemAvailable:/ {printf "%d", $2 / 1024}' /proc/meminfo)"
    available_gpu_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR == 1 {gsub(/ /, "", $1); print $1}')"
    if (( available_ram_mb >= MIN_RAM_MB && available_gpu_mb >= MIN_GPU_FREE_MB )); then
      echo "[pibd] resources ready: RAM=${available_ram_mb}MiB GPU=${available_gpu_mb}MiB"
      break
    fi
    echo "[pibd] waiting for resources: RAM=${available_ram_mb}/${MIN_RAM_MB}MiB GPU=${available_gpu_mb}/${MIN_GPU_FREE_MB}MiB"
    sleep 60
  done
}

wait_for_resources

SMOKE_RESULTS="${PROJECT_DIR}/results_smoke_pibd_resnet"
SMOKE_SUMMARY="${SMOKE_RESULTS}/tcga_coadread_b32_survival_months_dss_wsiDim_256_epochs_1_omics_pathways_pathT_combine_s1/summary_partial_0_1.csv"
if [[ -s "${SMOKE_SUMMARY}" ]]; then
  echo "[pibd] reusing completed smoke test: ${SMOKE_SUMMARY}"
else
  echo "[pibd] running one-fold one-epoch smoke test"
  MREPATH_PIBD_RESULTS="${SMOKE_RESULTS}" \
    bash "${PROJECT_DIR}/scripts/run_coadread_pibd.sh" \
    --k_start 0 --k_end 1 --max_epochs 1
fi

echo "[pibd] smoke test passed; checking resources again"
wait_for_resources

echo "[pibd] launching official five-fold run"
bash "${PROJECT_DIR}/scripts/run_coadread_pibd.sh"
echo "[pibd] pipeline completed at $(date --iso-8601=seconds)"
