# PC-CMKA-DDKAC repository audit

This report was written before the Word-aligned implementation.  The design
source of truth is `PC-CMKA-DDKAC_最终创新方案.docx` (2026-08-04 document
metadata), not the superseded experiment labels used by the stopped
`mrepath-pc-cmka-13x2-20260805.service` queue.

## 1. Genomics data flow

1. `SurvivalDatasetFactory` reads `rna_clean.csv` into a case-by-gene data
   frame.
2. `signatures.csv` defines six columns.  Each column is intersected with the
   RNA columns, sorted, and stored as one functional group in `omic_names`.
3. For every fold, the training split fits the RNA min/max scaler.  The same
   scaler is applied to validation data.  Zeros are restored after scaling.
4. `SurvivalDataset.__getitem__` selects the case row and emits six tensors in
   the exact `omic_names` order.
5. `_collate_HGNN` carries those six tensors to `_unpack_data`, which moves
   them to the selected device.
6. `MRePath.forward` collects `x_omic1` through `x_omic6`.  The default path
   uses six independent SNN networks; an experimental encoder is built by
   `build_genomic_encoder` and must return `[batch, 6, embedding_dim]`.
7. The optional six-token graph aggregator runs after the pathway encoder.
   Pathology and genomic tokens then enter unchanged dynamic modality
   weighting, IFA, and the survival head.

## 2. Existing six-group interface

The six groups are constructed exclusively from
`datasets_csv/metadata/signatures.csv`.  They are independent gene lists;
their CSV order is never treated as graph adjacency.  Fold-specific gene
graphs are passed as a list with one graph per group and the same node order as
the corresponding `omic_names` entry.

## 3. Existing DD-KAC

`DDKACPathway` accepts one normalized patient vector `[batch, genes]` and one
dense pathway adjacency matrix.  It contains:

- a value-domain radial basis transform and linear projection;
- a structure branch using a dense normalized Laplacian;
- a Chebyshev recurrence of configurable order (currently constructed with
  the default order two);
- a five-statistic patient router producing softmax frequency weights;
- a sigmoid gate between value and structure projections;
- a linear residual, layer normalization, and value/structure cosine
  consistency loss.

The current `scaled_laplacian = laplacian - I` implicitly assumes a normalized
Laplacian bound of two.  It does not implement the Word document's fixed
reference degree, low-dimensional inverse calibration, trust-region epigraph,
Hessian antithetic views, Krylov constraint, shared-view routing, or tangent
identifiability objective.

## 4. Existing gene graph provenance

`_build_fold_gene_graphs` computes absolute Pearson correlation from normalized
RNA in the current training fold, retains a symmetric top-k graph, and adds a
self-loop.  Therefore it is fold-local and patient-independent, but it is not
a PPI/KEGG/Reactome/GO/regulatory prior.  The Word-aligned implementation must
support an external edge list.  Until such a prior is supplied, training-fold
correlation is permitted only as an explicitly logged fallback and no
biological-prior claim may be made for that run.

## 5. Pathology and multimodal fusion

The pathology branch projects cached WSI features and applies the selected
SHGNN/HGNN/GCN/GAT/MLP aggregator.  Dynamic modality weighting is computed
before IFA.  IFA refines six genomic tokens and pathology tokens using the
existing attention sequence, after which the survival classifier returns four
discrete-time logits.  The first PC-CMKA implementation changes none of these
modules.

## 6. Leakage, shape, normalization, and reproducibility audit

- **RNA normalization:** fold-local and validation-safe.
- **Correlation fallback graph:** fold-local and validation-safe.
- **Survival bins:** the current factory computes quantiles before folds are
  created, using all cases.  This is a leakage risk.  PC-CMKA runs must enable
  strict training-fold-only survival bins.
- **Graph dimensions:** the current builder follows each group's gene order;
  explicit checks are still required at the new encoder boundary.
- **Patient batch statistics:** existing router statistics are computed per
  row, not across patients.  The new code preserves this property.
- **Graph normalization:** current DD-KAC recomputes degree from its fixed
  adjacency.  The Word method instead requires fixed prior `D0` for every
  patient edge vector and one cached spectral bound per group.
- **Randomness:** Python, NumPy, Torch, and CUDA are seeded and deterministic
  cuDNN is enabled.  DataLoader workers do not have an explicit worker seed
  function.  PC-CMKA stochastic tests use explicit Torch generators/seeds.
- **Sparse safety:** existing gene graphs and Laplacians are dense.  At the
  current group sizes this is finite but unnecessary.  PC-CMKA stores only
  upper-triangular prior edges and applies the incidence operator by scatter;
  dense matrices are restricted to tests and low-dimensional Hessians.

## 7. Files and implementation plan

New modules:

- `models/layers/pc_cmka/spectral.py`: reference-degree sparse operator and
  Chebyshev moments.
- `models/layers/pc_cmka/calibration.py`: target network, deformation
  dictionary, inverse solver, and direct-coefficient control.
- `models/layers/pc_cmka/augmentation.py`: Hessian/isotropic views and Krylov
  safety scaling.
- `models/layers/pc_cmka/losses.py`: structure/fusion consistency and tangent
  identifiability approximations.
- `models/layers/pc_cmka/encoder.py`: DD-KAC-compatible six-group encoder.
- `utils/pc_cmka.py`: configuration, prior construction, and diagnostics.
- `configs/pc_cmka_ddkac_word.json`: Word Table 5 A0-A9, separately namespaced
  prompt-staged A0-A10/Full, and classical control configurations.
- `scripts/run_pc_cmka_word_ablations.py`: resumable formal runner.
- `tests/test_pc_cmka_word.py`: numerical, gradient, stochastic, and leakage
  tests.

Minimal integration edits:

- `models/layers/genomic_encoders.py` registers a new encoder without changing
  existing encoders.
- `models/model_HGNN.py` forwards configuration and preserves `[B,6,d]`.
- `utils/process_args.py`, `utils/general_utils.py`, and `utils/core_utils.py`
  add guarded configuration, fold-local graph construction, component loss
  logging, and fold-local survival bins.
- `datasets/dataset_survival.py` adds the strict fold-bin path.

Implementation order follows the Word MVP: reference operator, moments,
inverse calibration, antithetic augmentation, Krylov scaling, shared routing,
identifiability, integration, then tests and smoke training.
