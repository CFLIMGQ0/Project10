#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_ROOT="${PROJECT_DIR}/data/tcga_coadread/clam_20x_resnet50"
CACHE_ROOT="${MREPATH_ABMIL_CACHE:-/home/administrator/.cache/mrepath/tcga_coadread/clam_20x_resnet50}"
LOG_DIR="${PROJECT_DIR}/logs"
LOCK_FILE="${LOG_DIR}/coadread_abmil_5fold.lock"
PID_FILE="${LOG_DIR}/coadread_abmil_5fold.pid"

mkdir -p "${LOG_DIR}" "${CACHE_ROOT}/h5_files"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[abmil] another COREAD ABMIL run already holds ${LOCK_FILE}"
  exit 1
fi
echo "$$" > "${PID_FILE}"
trap 'rm -f "${PID_FILE}"' EXIT

cd "${PROJECT_DIR}"
echo "[abmil] pipeline started at $(date --iso-8601=seconds)"
echo "[abmil] caching ResNet50 HDF5 features on the WSL filesystem"
rsync -a --ignore-existing "${SOURCE_ROOT}/h5_files/" "${CACHE_ROOT}/h5_files/"

source_count="$(find "${SOURCE_ROOT}/h5_files" -maxdepth 1 -type f -name '*.h5' | wc -l)"
cache_count="$(find "${CACHE_ROOT}/h5_files" -maxdepth 1 -type f -name '*.h5' | wc -l)"
if [[ "${source_count}" != "${cache_count}" ]]; then
  echo "[abmil] cache validation failed: ${cache_count}/${source_count} HDF5 files"
  exit 1
fi

echo "[abmil] launching five folds with ${cache_count} cached HDF5 files"
MREPATH_DATA_ROOT="${CACHE_ROOT}" bash scripts/run_coadread_abmil.sh
echo "[abmil] pipeline completed at $(date --iso-8601=seconds)"
