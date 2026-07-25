#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREPROCESS_PYTHON="${MREPATH_PREPROCESS_PYTHON:-/home/administrator/miniconda3/envs/mrepath-preprocess/bin/python}"
DATA_ROOT="${MREPATH_LEVEL0_DATA_ROOT:-${PROJECT_DIR}/data/tcga_coadread/clam_level0_256_resnet50_repo_k9}"
LOG_DIR="${PROJECT_DIR}/logs"
LOCK_FILE="${LOG_DIR}/coadread_level0_repo_pipeline.lock"
PID_FILE="${LOG_DIR}/coadread_level0_repo_pipeline.pid"
MIN_RAM_MB="${MREPATH_MIN_RAM_MB:-6144}"
MIN_GPU_FREE_MB="${MREPATH_MIN_GPU_FREE_MB:-12000}"

mkdir -p "${LOG_DIR}" "${DATA_ROOT}/graph_files"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[pipeline] another level0 CO-READ pipeline holds ${LOCK_FILE}"
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
echo "[pipeline] level0-256 repository-compatible run started at $(date --iso-8601=seconds)"
wait_for_resources

echo "[pipeline] refreshing WSI inventory"
"${PREPROCESS_PYTHON}" scripts/preprocess_wsi_clam.py \
  --output "${DATA_ROOT}" \
  --patch-mode level0-256 \
  --inventory-only \
  --device cpu

echo "[pipeline] extracting level-0 256x256 ResNet50 features in two shards"
"${PREPROCESS_PYTHON}" scripts/preprocess_wsi_clam.py \
  --output "${DATA_ROOT}" \
  --patch-mode level0-256 \
  --num-shards 2 \
  --shard-index 0 \
  --status-file preprocess_status_shard0.csv \
  --skip-inventory \
  --batch-size "${MREPATH_FEATURE_BATCH_SIZE:-256}" \
  --workers "${MREPATH_FEATURE_WORKERS_PER_SHARD:-2}" \
  --fail-fast &
shard0_pid=$!

"${PREPROCESS_PYTHON}" scripts/preprocess_wsi_clam.py \
  --output "${DATA_ROOT}" \
  --patch-mode level0-256 \
  --num-shards 2 \
  --shard-index 1 \
  --status-file preprocess_status_shard1.csv \
  --skip-inventory \
  --batch-size "${MREPATH_FEATURE_BATCH_SIZE:-256}" \
  --workers "${MREPATH_FEATURE_WORKERS_PER_SHARD:-2}" \
  --fail-fast &
shard1_pid=$!

shard_failure=0
wait "${shard0_pid}" || shard_failure=1
wait "${shard1_pid}" || shard_failure=1
if (( shard_failure != 0 )); then
  echo "[pipeline] at least one preprocessing shard failed"
  exit 1
fi

"${PREPROCESS_PYTHON}" - <<PY
from pathlib import Path
import os
import pandas as pd

root = Path(r'${DATA_ROOT}')
parts = [pd.read_csv(root / f'preprocess_status_shard{i}.csv') for i in range(2)]
merged = pd.concat(parts, ignore_index=True)
if len(merged) != 301 or merged['slide_id'].nunique() != 301:
    raise RuntimeError(
        f'Expected 301 unique shard rows, got {len(merged)} rows and '
        f'{merged["slide_id"].nunique()} unique IDs'
    )
if not (merged['status'] == 'completed').all():
    raise RuntimeError('At least one preprocessing row is incomplete')
temporary = root / 'preprocess_status.csv.tmp'
merged.sort_values('slide_id').to_csv(temporary, index=False)
os.replace(temporary, root / 'preprocess_status.csv')
print(f'[pipeline] merged {len(merged)} completed preprocessing rows')
PY

echo "[pipeline] building repository k=9 graphs (L2 spatial, L2 feature)"
"${PREPROCESS_PYTHON}" scripts/build_wsi_graphs.py \
  --h5-dir "${DATA_ROOT}/h5_files" \
  --output-dir "${DATA_ROOT}/graph_files" \
  --radius 9 \
  --spatial-space l2 \
  --feature-space l2 \
  --fail-fast

completed="$(${PREPROCESS_PYTHON} - <<PY
import pandas as pd
d = pd.read_csv(r'${DATA_ROOT}/graph_files/graph_status.csv')
print(int(d['status'].isin(['completed', 'already_completed']).sum()))
PY
)"
if [[ "${completed}" != "301" ]]; then
  echo "[pipeline] expected 301 graphs, found ${completed}"
  exit 1
fi

wait_for_resources
echo "[pipeline] launching repository-strategy five folds at $(date --iso-8601=seconds)"
bash scripts/run_coadread_level0_repo.sh
echo "[pipeline] completed at $(date --iso-8601=seconds)"
