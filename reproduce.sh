#!/usr/bin/env bash
set -euo pipefail

# Reproduce TNFlow results from a trained checkpoint (default runs/v2/checkpoint.pt).
# Requires: the checkpoint, data/{spectra,manifest}.parquet, data/splits/v2.parquet,
#           data/JWST_DiSCO-TNOs/
CHECKPOINT="${1:-runs/v2/checkpoint.pt}"

echo "=== Evaluating on synthetic test + OOD splits (Table 1) ==="
python3.10 -m src.pipeline.test --checkpoint "$CHECKPOINT"

echo ""
echo "=== JWST transfer experiment ==="
python3.10 -m src.pipeline.apply_jwst --checkpoint "$CHECKPOINT"
