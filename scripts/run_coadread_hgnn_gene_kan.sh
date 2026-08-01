#!/usr/bin/env bash
set -euo pipefail

repo_root=/mnt/e/MRePath
train_python=/home/administrator/miniconda3/envs/mrepath-train/bin/python
dependency_unit=mrepath-coadread-hgnn-gene-gcn-20260730.service

cd "$repo_root"

if [[ "${MREPATH_SKIP_WAIT:-0}" != "1" ]]; then
    echo "[queue] Waiting for $dependency_unit to finish."
    while systemctl --user is-active --quiet "$dependency_unit"; do
        sleep 30
    done
else
    echo "[queue] Parallel launch requested; not waiting for $dependency_unit."
fi

echo "[smoke] Starting HGNN pathology + KAN genomic aggregation."
"$train_python" scripts/run_coadread_paper_ablations.py \
    --mode smoke \
    --matrix configs/coadread_hgnn_gene_kan.json \
    --results-root results_coadread_hgnn_gene_kan_20260730 \
    --fail-fast

echo "[formal] Smoke test passed; starting sequential five-fold training."
"$train_python" scripts/run_coadread_paper_ablations.py \
    --mode formal \
    --matrix configs/coadread_hgnn_gene_kan.json \
    --results-root results_coadread_hgnn_gene_kan_20260730 \
    --fail-fast
