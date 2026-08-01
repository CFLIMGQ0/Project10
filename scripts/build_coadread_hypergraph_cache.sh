#!/usr/bin/env bash
set -euo pipefail

repo_root=/mnt/e/MRePath
train_python=/home/administrator/miniconda3/envs/mrepath-train/bin/python
graph_root="$repo_root/data/tcga_coadread/clam_20x_resnet50_paper_k9"
cache_dir="$graph_root/hypergraph_cache"
training_units=(
    mrepath-coadread-hgnn-gene-gcn-20260730.service
    mrepath-coadread-hgnn-gene-kan-parallel-20260730.service
)

cd "$repo_root"

for training_unit in "${training_units[@]}"; do
    echo "[queue] Waiting for $training_unit to finish."
    while systemctl --user is-active --quiet "$training_unit"; do
        sleep 30
    done
done

echo "[cache] Building seed-independent COREAD hypergraph cache."
"$train_python" scripts/build_hypergraph_cache.py \
    --graph-dir "$graph_root/graph_files" \
    --cache-dir "$cache_dir" \
    --workers 2

echo "[validate] Running one-epoch cached HGNN+KAN smoke test."
"$train_python" scripts/run_coadread_paper_ablations.py \
    --mode smoke \
    --matrix configs/coadread_hgnn_gene_kan.json \
    --results-root results_coadread_hypergraph_cache_validation_20260730 \
    --fail-fast

echo "[resume] Starting cached HGNN+GCN from its first incomplete fold."
systemd-run --user \
    --unit=mrepath-coadread-hgnn-gene-gcn-cached-resume-20260730 \
    --description="Resume COREAD HGNN+GCN with hypergraph cache" \
    --working-directory="$repo_root" \
    --property=StandardOutput=append:"$repo_root/logs/coadread_hgnn_gene_gcn_5fold_20260730.log" \
    --property=StandardError=append:"$repo_root/logs/coadread_hgnn_gene_gcn_5fold_20260730.log" \
    "$train_python" scripts/run_coadread_paper_ablations.py \
        --mode formal \
        --matrix configs/coadread_combined_candidates.json \
        --results-root results_coadread_hgnn_gene_gcn_20260730 \
        --fail-fast

echo "[resume] Starting cached HGNN+KAN from its first incomplete fold."
systemd-run --user \
    --unit=mrepath-coadread-hgnn-gene-kan-cached-resume-20260730 \
    --description="Resume COREAD HGNN+KAN with hypergraph cache" \
    --working-directory="$repo_root" \
    --property=StandardOutput=append:"$repo_root/logs/coadread_hgnn_gene_kan_5fold_20260730.log" \
    --property=StandardError=append:"$repo_root/logs/coadread_hgnn_gene_kan_5fold_20260730.log" \
    "$train_python" scripts/run_coadread_paper_ablations.py \
        --mode formal \
        --matrix configs/coadread_hgnn_gene_kan.json \
        --results-root results_coadread_hgnn_gene_kan_20260730 \
        --fail-fast
