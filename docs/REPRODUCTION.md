# CO-READ reproduction quick start

The canonical profile is defined in:

- `configs/paper_coadread.json` — machine-readable settings;
- `docs/REPRODUCTION_CONTRACT.md` — paper/repository decisions and known
  public-data limitations;
- `scripts/audit_paper_coadread.py` — preflight validation;
- `scripts/run_coadread.sh` — the only active CO-READ training entry point.

Run the released-data reproduction sequentially with:

```bash
bash scripts/run_coadread.sh
```

This invokes the paper profile: ResNet50 1024-D pathology features, 4096
patches per case, spatial L2 and feature cosine hyperedges at `k=9`, three
general-sheaf layers, dynamic modality weighting, interactive alignment
fusion, Adam at `1e-4`, weight decay `1e-5`, NLL survival loss, and 30 epochs.

The paper reports CO-READ C-index `0.808 ± 0.058`. The public repository does
not release all inputs needed for an exact reproduction: its splits cover 297
of the paper's 298 cases, RNA covers 295 split cases, and CNV/SNV are absent.
The launcher prints this audit before training; setting
`MREPATH_REQUIRE_PRIVATE_PAPER_DATA=1` makes these limitations fatal.
