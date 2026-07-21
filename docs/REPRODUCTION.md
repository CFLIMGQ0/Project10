# MRePath reproduction profile

This project uses the model and interactive-alignment module layout released by
`MCPathology/MRePath` at commit `e3c133b`. WSI preprocessing and resumable graph
construction are local wrappers and do not alter the released graph algorithm
(HNSW spatial/feature neighbours with radius 9).

## Paper experiment profile

The default CO-READ launcher follows the settings reported in the IJCAI 2025
paper:

- truncated ImageNet-pretrained ResNet50 patch features (1024 dimensions);
- 4096 sampled patches per case;
- five predefined patient-level folds, with a 4:1 train/validation split;
- Adam, learning rate `1e-4`, weight decay `1e-5`;
- cosine schedule with one warm-up epoch;
- NLL survival loss with `alpha_surv=0.5`;
- 30 epochs and seed 1.

Run all five CO-READ folds with:

```bash
bash scripts/run_coadread.sh
```

Results are written below `results_coadread_released/`. The paper reports
CO-READ C-index `0.808 +/- 0.058`; comparison must use the mean and sample
standard deviation of all five folds, not the best individual fold.

## Necessary compatibility corrections

The released repository contains a few contradictions with its own README and
paper. This reproduction keeps the released model topology while applying only
the corrections required for an executable and auditable experiment:

- the pathology projection accepts the 1024-dimensional ResNet50 features used
  by the released training command (the model file originally hard-coded 768);
- Adam receives the paper's weight decay;
- genomic rows are matched by TCGA case ID rather than relying on dataframe row
  order;
- graph loading is compatible with PyTorch 2.6 and graph node IDs are remapped
  to the sampled feature order;
- slides whose sampled nodes contain no surviving hyperedge fall back to their
  projected pathology tokens instead of crashing;
- every fold writes a stable best checkpoint and `summary.csv`.

These compatibility changes leave the normal three-layer sheaf-hypergraph path,
released confidence modules, released interactive-alignment fusion, classifier,
loss, and five-fold protocol intact.
