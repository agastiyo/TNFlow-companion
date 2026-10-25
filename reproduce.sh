#!/usr/bin/env bash
set -euo pipefail

# Reproduce TNFlow results. The checkpoint and model size are set at the top of
# src/test.py and src/pipeline/apply_jwst.py.
# Requires: the checkpoint, data/spectra.parquet, data/JWST_DiSCO-TNOs/
export OMP_NUM_THREADS=1

echo "=== Test + OOD splits (Table 1) ==="
python -m src.test

echo ""
echo "=== JWST transfer experiment ==="
python -m src.pipeline.apply_jwst
