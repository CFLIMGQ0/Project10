# Legacy genomic encoder combination experiments

The experiments in `configs/cached_coadread_stad_experiments.json` are now
classified as **genomic encoder combination experiments**.  They compare
pathology graph aggregators, six-token gene aggregators, five pathway-wise
genomic encoders, and quality/conflict fusion controls.

They are not the PC-CMKA-DDKAC innovation ablations from Word Table 5.  The
previous PC-CMKA queue used superseded A0-A10 labels and was stopped and
disabled on 2026-08-07.  Its existing result files are retained as historical
encoder/control evidence and must not be reported as the final Word-aligned
PC-CMKA ablation study.

The file currently contains 16 executable entries: the 15 previously called
"ideas" plus `dd_kac` as the reference configuration.  Thus the user's 15
ideas are now collectively named **genomic encoder combination experiments**;
`dd_kac` is their comparison baseline, not a sixteenth innovation idea.

This experiment family is intentionally not scheduled while the Word-aligned
method is implemented and validated.
