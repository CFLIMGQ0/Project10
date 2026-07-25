#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# There is one canonical entry point for the current reproduction target. The
# released generic script contradicted the paper's optimizer, weight decay,
# survival-loss weighting, and sampling settings.
exec bash "${SCRIPT_DIR}/run_coadread.sh" "$@"
