#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${PROJECT_DIR}/data/tcga_stad/gdc_manifest_tcga_stad_mrepath.tsv"
OUTPUT_DIR="${PROJECT_DIR}/data/tcga_stad/raw_svs"
GDC_CLIENT="${PROJECT_DIR}/tools/gdc-client-2.3.0/gdc-client"
# Download directly from GDC. Do not consume traffic from a desktop/VPN proxy.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
unset ALL_PROXY all_proxy NO_PROXY no_proxy

mapfile -t incomplete_ids < <(
  while IFS=$'\t' read -r file_id filename md5 size state; do
    target="${OUTPUT_DIR}/${file_id}/${filename}"
    if [[ ! -f "${target}" ]] || [[ "$(stat -c '%s' "${target}")" -ne "${size}" ]]; then
      printf '%s\n' "${file_id}"
    fi
  done < <(tail -n +2 "${MANIFEST}")
)

if [[ "${#incomplete_ids[@]}" -eq 0 ]]; then
  echo "[gdc-direct] all manifest files are already complete"
  exit 0
fi

echo "[gdc-direct] queued incomplete files: ${#incomplete_ids[@]}"
exec "${GDC_CLIENT}" download \
  --dir "${OUTPUT_DIR}" \
  --n-processes 4 \
  --http-chunk-size 8388608 \
  --retry-amount 20 \
  --wait-time 30 \
  --no-related-files \
  --no-annotations \
  --log-file "${PROJECT_DIR}/data/tcga_stad/gdc_download_direct_ipv4.log" \
  --color_off \
  "${incomplete_ids[@]}"
