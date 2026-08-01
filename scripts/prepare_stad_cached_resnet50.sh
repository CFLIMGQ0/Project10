#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${MREPATH_PYTHON:-/home/administrator/miniconda3/envs/mrepath-train/bin/python}"
PREPROCESS_PYTHON_BIN="${MREPATH_PREPROCESS_PYTHON:-/home/administrator/miniconda3/envs/mrepath-preprocess/bin/python}"
SOURCE_ROOT="${PROJECT_ROOT}/data/tcga_stad/raw_svs"
OUTPUT_ROOT="${PROJECT_ROOT}/data/tcga_stad/clam_20x_resnet50_paper_k9"
MANIFEST="${PROJECT_ROOT}/data/tcga_stad/gdc_manifest_tcga_stad_mrepath.tsv"
DOWNLOAD_UNIT="mrepath-stad-download-complete-20260731.service"

cd "${PROJECT_ROOT}"

while systemctl --user is-active --quiet "${DOWNLOAD_UNIT}"; do
  echo "[wait] STAD download is still active"
  sleep 30
done

incomplete=0
while IFS=$'\t' read -r file_id filename md5 size state; do
  target="${SOURCE_ROOT}/${file_id}/${filename}"
  if [[ ! -f "${target}" ]] || [[ "$(stat -c '%s' "${target}" 2>/dev/null || echo 0)" -ne "${size}" ]]; then
    echo "[missing] ${file_id} ${filename}"
    incomplete=$((incomplete + 1))
  fi
done < <(tail -n +2 "${MANIFEST}")
if [[ "${incomplete}" -ne 0 ]]; then
  echo "[error] ${incomplete} manifest files remain incomplete"
  exit 1
fi

"${PREPROCESS_PYTHON_BIN}" -u scripts/preprocess_wsi_clam.py \
  --source "${SOURCE_ROOT}" \
  --output "${OUTPUT_ROOT}" \
  --patch-mode true-20x \
  --inventory-only

pids=()
for shard in 0 1; do
  "${PREPROCESS_PYTHON_BIN}" -u scripts/preprocess_wsi_clam.py \
    --source "${SOURCE_ROOT}" \
    --output "${OUTPUT_ROOT}" \
    --patch-mode true-20x \
    --stages segment,features \
    --skip-inventory \
    --num-shards 2 \
    --shard-index "${shard}" \
    --status-file "preprocess_status_shard${shard}.csv" \
    --batch-size 256 \
    --workers 2 \
    --device cuda &
  pids+=("$!")
done

preprocess_failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    preprocess_failed=1
  fi
done
if [[ "${preprocess_failed}" -ne 0 ]]; then
  echo "[error] at least one STAD preprocessing shard failed"
  exit 1
fi

"${PYTHON_BIN}" -u scripts/build_wsi_graphs.py \
  --h5-dir "${OUTPUT_ROOT}/h5_files" \
  --output-dir "${OUTPUT_ROOT}/graph_files" \
  --metadata-csv "${PROJECT_ROOT}/datasets_csv/metadata/tcga_stad.csv" \
  --radius 9 \
  --feature-dim 1024 \
  --spatial-space l2 \
  --feature-space cosinesimil \
  --verify-existing

"${PYTHON_BIN}" -u scripts/build_hypergraph_cache.py \
  --graph-dir "${OUTPUT_ROOT}/graph_files" \
  --cache-dir "${OUTPUT_ROOT}/hypergraph_cache" \
  --workers 4

echo "[complete] STAD ResNet50 graphs and offline hypergraph cache are ready"
