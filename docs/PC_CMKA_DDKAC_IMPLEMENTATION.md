# Word-aligned PC-CMKA-DDKAC implementation

## Scope

The implementation follows `PC-CMKA-DDKAC_最终创新方案.docx`. Word Table 5 is
the primary suite: A0 is the complete model and A1-A9 remove or replace one
mechanism. A9 has two executable subvariants, structure-only and fusion-only.
The differently numbered cumulative suite requested in the implementation
prompt is also available under an unambiguous `S_` namespace.

The pathology encoder, pathology graph, dynamic modality weighting, IFA,
survival head, six functional groups, and public five-fold split files are
unchanged.

## Formula-to-code correspondence

| Word object | Implementation |
|---|---|
| `S(w)=D0^-1/2 B^T diag(w) B D0^-1/2` | `ReferenceSpectralOperator.apply_operator` uses edge scatter and cached `D0` |
| shared `Lambda` and `S_hat` | cached theoretical bound and `scaled=True` operator application |
| patient functional probe | `PCCMKADDKACPathway._probe` |
| Chebyshev response moments | `ReferenceSpectralOperator.moments` |
| target `Delta_mu`, precision `pi` | `PatientMomentTarget` |
| weighted-orthogonal dictionary `P` | `EdgeDeformationDictionary` |
| `w=w0*exp(Pa)` | `weights_from_coefficients` |
| inverse calibration and minimum `rho` | fixed-step differentiable solve; exact epigraph `rho=max(abs(Pa))` |
| local Gauss-Newton `H` | `GraphViewAugmenter._moment_jacobian` and `_uncertainty_direction` |
| exact/diagonal/low-rank inverse root | augmentation `hessian_mode` |
| antithetic views | `positive_delta=xi`, `negative_delta=-xi` |
| patient Krylov subspace | `ReferenceSpectralOperator.krylov_basis` |
| safe action constraint | Frobenius upper-bound scaling in `_krylov_scale` |
| shared DD-KAC route | one `base_logits` reused by both views |
| structure/fusion/gate consistency | `losses.py` and encoder SSL modes |
| tangent identifiability | randomized JVP squared-cosine approximation |
| total auxiliary loss | separately configurable components in JSON |

## Word Table 5 configurations

- `A0_full`: complete PC-CMKA-DDKAC.
- `A1_direct_coefficients`: MLP directly predicts low-dimensional `a_p`.
- `A2_fixed_rho`: fixed trust radius and no minimum-radius penalty.
- `A3_random_probe`: patient functional probe replaced by a seeded random
  probe.
- `A4_isotropic_uncertainty`: Hessian direction replaced by isotropic noise.
- `A5_independent_views`: centered antithetic pair replaced by independent
  views.
- `A6_no_krylov`: no Krylov safety scaling.
- `A7_separate_routes`: each view recomputes frequency routing.
- `A8_no_identifiability`: no tangent-space loss.
- `A9_structure_only` and `A9_fusion_only`: the two requested anti-bypass
  consistency controls.

The five main controls are `C0_original_ddkac`,
`C1_patient_degree_edge_gate`, `C2_reference_degree_edge_gate`,
`C3_effective_resistance`, and `C4_bernoulli_views`.  Together with the 11
Table 5 jobs this produces 16 executable configurations; conceptually A9 is
one row with two subvariants.

## Prompt-staged configurations

The implementation prompt and Word Table 5 reuse the A0 labels with different
meanings. To avoid silently mixing results, the cumulative prompt suite uses an
`S_` prefix: `S_A0_original_ddkac` through `S_A10_identifiability`, plus
`S_Full_pc_cmka_ddkac`. Run it with `--suite staged`. Word Table 5 remains the
default `--suite word`.

## Configuration and leakage policy

`configs/pc_cmka_ddkac_word.json` contains every numerical setting.  An
external biological prior can be supplied through `prior.edge_list`.  The
current public repository has no mapped PPI/Reactome edge file, so the default
is an explicitly warned training-fold correlation fallback.  It is not valid
to claim that fallback as a biological prior.

All PC-CMKA commands pass `--fold_survival_bins`.  RNA scaling, graph fallback,
and survival bins are fitted from the training fold only.  Validation uses the
unaugmented patient graph.

## Outputs

Each fold writes the normal checkpoints and predictions plus:

- `s_F_pc_cmka_config.json`: resolved configuration and prior provenance;
- `s_F_pc_cmka_epoch_losses.csv`: separate moment, trust, SSL,
  identifiability, and dictionary losses;
- `s_F_pc_cmka_patient_diagnostics.json`: rho, target/actual moments,
  residuals, edge changes, routes, gates, and validation-safe diagnostics;
- `s_F_pc_cmka_resources.json`: runtime, peak allocated GPU memory, and
  C-index;
- `s_F_genomic_encoder_diagnostics.json`: final structured encoder record.

## Commands

One-fold smoke-style formal-path check:

```bash
python scripts/run_pc_cmka_word_ablations.py \
  --dataset coadread --experiments A0_full --folds 0 \
  --max-epochs 1 --num-patches 16 --num-workers 0
```

One formal fold:

```bash
python scripts/run_pc_cmka_word_ablations.py \
  --dataset coadread --experiments A0_full --folds 0 \
  --max-epochs 30 --num-workers 8
```

Five-fold full model:

```bash
python scripts/run_pc_cmka_word_ablations.py \
  --dataset coadread --experiments A0_full \
  --max-epochs 30 --num-workers 8
```

All Word Table 5 ablations (controls excluded by default):

```bash
python scripts/run_pc_cmka_word_ablations.py \
  --dataset coadread --max-epochs 30 --num-workers 8
```

All main controls and ablations:

```bash
python scripts/run_pc_cmka_word_ablations.py \
  --dataset coadread --include-controls \
  --max-epochs 30 --num-workers 8
```

All prompt-staged ablations:

```bash
python scripts/run_pc_cmka_word_ablations.py \
  --dataset coadread --suite staged \
  --max-epochs 30 --num-workers 8
```

All 15 user-facing `graph_structure` choices from `model.yaml`:

```bash
python scripts/run_pc_cmka_word_ablations.py \
  --dataset coadread --graph-structures \
  --max-epochs 30 --num-workers 8
```

## Known limitations

1. No external PPI/Reactome edge list is bundled, so the default prior is only
   a fold-local correlation fallback.
2. The first identifiability implementation is the Word-requested randomized
   JVP approximation, not the full normalized Jacobian expression.
3. The default Hessian mode is diagonal.  Exact and low-rank modes are
   available for controlled comparisons.
4. At WSI batch size one, variance regularization uses feature-wise dispersion;
   no validation patient is used as a negative or batch statistic.
5. Validation never samples graph views.  Consequently stochastic Hessian-view
   diagnostics are training diagnostics, while validation JSON focuses on
   calibration, routing, and gate quantities.

## Verified smoke test

On 2026-08-07, `A0_full` completed COREAD fold 0 for one epoch with 16 WSI
patches per case. It trained 237 and validated 58 cases, selected validation
C-index 0.727642, used 309,293,568 peak allocated GPU bytes, and finished in
about 134 seconds. This is an integration check only, not a formal result.
