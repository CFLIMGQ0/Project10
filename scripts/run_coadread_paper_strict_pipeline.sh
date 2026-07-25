#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREPROCESS_PYTHON="${MREPATH_PREPROCESS_PYTHON:-/home/administrator/miniconda3/envs/mrepath-preprocess/bin/python}"
GRAPH_ROOT="${PROJECT_DIR}/data/tcga_coadread/clam_20x_resnet50_paper_k9"
GRAPH_DIR="${GRAPH_ROOT}/graph_files"
STATUS_FILE="${GRAPH_DIR}/graph_status.csv"
LOCK_FILE="${PROJECT_DIR}/logs/coadread_paper_strict_pipeline.lock"
PID_FILE="${PROJECT_DIR}/logs/coadread_paper_strict_pipeline.pid"
MIN_RAM_MB="${MREPATH_MIN_RAM_MB:-6144}"
MIN_GPU_FREE_MB="${MREPATH_MIN_GPU_FREE_MB:-12000}"

mkdir -p "${PROJECT_DIR}/logs" "${GRAPH_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[pipeline] another strict COREAD pipeline already holds ${LOCK_FILE}"
  exit 1
fi
echo "$$" > "${PID_FILE}"
trap 'rm -f "${PID_FILE}"' EXIT

available_ram_mb() {
  awk '/MemAvailable:/ {printf "%d", $2 / 1024}' /proc/meminfo
}

gpu_free_mb() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    | awk 'NR == 1 {gsub(/ /, "", $1); print $1}'
}

wait_for_ram() {
  while true; do
    local available
    available="$(available_ram_mb)"
    if (( available >= MIN_RAM_MB )); then
      echo "[resource] RAM ready: ${available} MiB available"
      return
    fi
    echo "[resource] waiting for RAM: ${available}/${MIN_RAM_MB} MiB available"
    sleep 60
  done
}

wait_for_gpu() {
  while true; do
    local available
    available="$(gpu_free_mb)"
    if [[ "${available}" =~ ^[0-9]+$ ]] && (( available >= MIN_GPU_FREE_MB )); then
      echo "[resource] GPU ready: ${available} MiB free"
      return
    fi
    echo "[resource] waiting for GPU: ${available:-unknown}/${MIN_GPU_FREE_MB} MiB free"
    sleep 60
  done
}

cd "${PROJECT_DIR}"
echo "[pipeline] paper-strict COREAD run started at $(date --iso-8601=seconds)"
echo "[pipeline] contract: configs/paper_coadread.json"

"${PREPROCESS_PYTHON}" scripts/audit_paper_coadread.py \
  --data-root "${GRAPH_ROOT}"

wait_for_ram
echo "[pipeline] building k=9 graphs (spatial=L2, feature=cosine)"
"${PREPROCESS_PYTHON}" scripts/build_wsi_graphs.py \
  --output-dir "${GRAPH_DIR}" \
  --radius 9 \
  --verify-existing \
  --fail-fast

completed="$(${PREPROCESS_PYTHON} - <<PY
import pandas as pd
d = pd.read_csv(r'${STATUS_FILE}')
print(int(d['status'].isin(['completed', 'already_completed']).sum()))
PY
)"
if [[ "${completed}" != "301" ]]; then
  echo "[pipeline] expected 301 completed graphs, found ${completed}"
  exit 1
fi

wait_for_ram
wait_for_gpu
echo "[pipeline] launching five folds at $(date --iso-8601=seconds)"
bash scripts/run_coadread.sh
echo "[pipeline] completed at $(date --iso-8601=seconds)"
