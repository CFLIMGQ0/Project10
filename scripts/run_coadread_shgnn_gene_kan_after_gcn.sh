#!/usr/bin/env bash
set -euo pipefail

repo_root=/mnt/e/MRePath
train_python=/home/administrator/miniconda3/envs/mrepath-train/bin/python
dependency_unit=mrepath-coadread-shgnn-gene-gcn-repeat-20260730.service

cd "$repo_root"

echo "[queue] Waiting for $dependency_unit to finish."
while systemctl --user is-active --quiet "$dependency_unit"; do
    sleep 30
done

echo "[smoke] Starting cached SheafHGNN + KAN smoke test."
"$train_python" scripts/run_coadread_paper_ablations.py \
    --mode smoke \
    --matrix configs/coadread_shgnn_gene_kan.json \
    --results-root results_coadread_shgnn_gene_kan_20260730 \
    --fail-fast

echo "[formal] Smoke passed; starting sequential five-fold training."
"$train_python" scripts/run_coadread_paper_ablations.py \
    --mode formal \
    --matrix configs/coadread_shgnn_gene_kan.json \
    --results-root results_coadread_shgnn_gene_kan_20260730 \
    --fail-fast
