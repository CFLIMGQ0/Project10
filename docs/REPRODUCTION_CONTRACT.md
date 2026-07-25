# MRePath CO-READ reproduction contract

This project treats the paper and its released repository as two distinct
sources of evidence:

- Paper: arXiv 2505.11997, *Multimodal Cancer Survival Analysis via
  Hypergraph Learning with Cross-Modality Rebalance*.
- Released code: `MCPathology/MRePath`, commit
  `e3c133b0beb5ded80426afbe49ab41f1003a4ed5`.

The canonical executable settings are recorded in
`configs/paper_coadread.json`. The primary experiment uses non-overlapping
256×256 fields at 20×, truncated ResNet50 1024-D features, spatial and feature
hyperedges with `k=9`, the intended sheaf-hypergraph/rebalance/fusion model,
five sequential folds, Adam at `1e-4`, weight decay `1e-5`, NLL survival loss,
and 30 epochs. Because the paper does not state a checkpoint-selection rule,
the executable profile follows the released training code and reports the
best validation C-index checkpoint from each fold.

## Resolved release-code contradictions

The released `run.sh` uses RAdam, weight decay `1e-4`, `alpha_surv=0.5`, and
weighted sampling. The paper explicitly specifies Adam, learning rate `1e-4`,
weight decay `1e-5`, and 30 epochs; the paper profile therefore takes
precedence.

The released model computes modality weights but does not apply them to the
modalities. Its holo-confidence numerators are also reversed relative to
Equations 7–9, and its random node subgraph remapping can connect edges to the
wrong feature rows. The local implementation follows the equations, applies
the weights, preserves gradients through the confidence MLPs, and remaps graph
edges to the exact sampled feature order.

The paper's main CO-READ result uses the ResNet50 encoder. CTransPath is an
encoder ablation and PIBD is a separate baseline; neither is the canonical
MRePath reproduction.

## Unresolved public-data limitations

The paper reports 298 CO-READ cases and states that genomic input includes
RNA-seq, CNV, and SNV. The released artifacts contain:

- 298 metadata cases;
- 297 cases in every released five-fold partition (TCGA-F5-6861 is absent);
- 295 split cases with released RNA (TCGA-EI-6883 and TCGA-G4-6323 are absent);
- an RNA-only matrix, without the paper-described CNV and SNV channels.

Consequently, public artifacts cannot constitute a bit-for-bit reproduction of
the private paper cohort. `scripts/audit_paper_coadread.py --strict` fails
deliberately while these differences remain. Normal audit mode records the
limitations but permits a clearly labelled released-data reproduction.
