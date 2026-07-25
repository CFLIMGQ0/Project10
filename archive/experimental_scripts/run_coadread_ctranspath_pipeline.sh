#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREPROCESS_PYTHON="${MREPATH_PREPROCESS_PYTHON:-/home/administrator/miniconda3/envs/mrepath-preprocess/bin/python}"
DATA_ROOT="${MREPATH_CTRANSPATH_DATA_ROOT:-/home/administrator/.cache/mrepath/tcga_coadread/clam_20x_ctranspath_samecoords}"
LOG_DIR="${PROJECT_DIR}/logs"
LOCK_FILE="${LOG_DIR}/coadread_ctranspath_pipeline.lock"
PID_FILE="${LOG_DIR}/coadread_ctranspath_pipeline.pid"
MIN_RAM_MB="${MREPATH_MIN_RAM_MB:-8192}"
MIN_GPU_FREE_MB="${MREPATH_MIN_GPU_FREE_MB:-12000}"

mkdir -p "${LOG_DIR}" "${DATA_ROOT}/h5_files" "${DATA_ROOT}/graph_files"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[pipeline] another CTransPath pipeline holds ${LOCK_FILE}"
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

wait_for_resources() {
  while true; do
    local ram gpu
    ram="$(available_ram_mb)"
    gpu="$(gpu_free_mb)"
    if (( ram >= MIN_RAM_MB && gpu >= MIN_GPU_FREE_MB )); then
      echo "[resource] ready RAM=${ram}MiB GPU=${gpu}MiB"
      return
    fi
    echo "[resource] waiting RAM=${ram}/${MIN_RAM_MB}MiB GPU=${gpu}/${MIN_GPU_FREE_MB}MiB"
    sleep 60
  done
}

cd "${PROJECT_DIR}"
echo "[pipeline] CTransPath MRePath ablation started at $(date --iso-8601=seconds)"
echo "[pipeline] same 301 WSI coordinates, official CTransPath checkpoint, 768-D features"
wait_for_resources

echo "[pipeline] extracting CTransPath features in two resumable shards"
"${PREPROCESS_PYTHON}" scripts/extract_ctranspath_features.py \
  --output "${DATA_ROOT}" \
  --num-shards 2 \
  --shard-index 0 \
  --status-file ctranspath_status_shard0.csv \
  --batch-size "${MREPATH_CTRANSPATH_BATCH_SIZE:-256}" \
  --workers "${MREPATH_CTRANSPATH_WORKERS_PER_SHARD:-3}" \
  --fail-fast &
shard0_pid=$!

"${PREPROCESS_PYTHON}" scripts/extract_ctranspath_features.py \
  --output "${DATA_ROOT}" \
  --num-shards 2 \
  --shard-index 1 \
  --status-file ctranspath_status_shard1.csv \
  --batch-size "${MREPATH_CTRANSPATH_BATCH_SIZE:-256}" \
  --workers "${MREPATH_CTRANSPATH_WORKERS_PER_SHARD:-3}" \
  --fail-fast &
shard1_pid=$!

shard_failure=0
wait "${shard0_pid}" || shard_failure=1
wait "${shard1_pid}" || shard_failure=1
if (( shard_failure != 0 )); then
  echo "[pipeline] at least one CTransPath extraction shard failed"
  exit 1
fi

"${PREPROCESS_PYTHON}" - <<PY
from pathlib import Path
import h5py

root = Path(r'${DATA_ROOT}') / 'h5_files'
files = sorted(root.glob('*.h5'))
if len(files) != 301:
    raise RuntimeError(f'Expected 301 CTransPath HDF5 files, found {len(files)}')
for path in files:
    with h5py.File(path, 'r') as handle:
        features = handle['features']
        coords = handle['coords']
        if features.ndim != 2 or features.shape[1] != 768:
            raise RuntimeError(f'Invalid features in {path}: {features.shape}')
        if coords.shape != (features.shape[0], 2):
            raise RuntimeError(f'Invalid coordinates in {path}: {coords.shape}')
print(f'[pipeline] validated {len(files)} CTransPath feature files')
PY

echo "[pipeline] building k=9 spatial-L2 and feature-cosine graphs"
"${PREPROCESS_PYTHON}" scripts/build_wsi_graphs.py \
  --h5-dir "${DATA_ROOT}/h5_files" \
  --output-dir "${DATA_ROOT}/graph_files" \
  --feature-dim 768 \
  --radius 9 \
  --spatial-space l2 \
  --feature-space cosinesimil \
  --fail-fast

completed="$("${PREPROCESS_PYTHON}" - <<PY
import pandas as pd
d = pd.read_csv(r'${DATA_ROOT}/graph_files/graph_status.csv')
print(int(d['status'].isin(['completed', 'already_completed']).sum()))
PY
)"
if [[ "${completed}" != "301" ]]; then
  echo "[pipeline] expected 301 completed graphs, found ${completed}"
  exit 1
fi

wait_for_resources
echo "[pipeline] launching matched five-fold MRePath training at $(date --iso-8601=seconds)"
bash scripts/run_coadread_ctranspath.sh
echo "[pipeline] completed at $(date --iso-8601=seconds)"
