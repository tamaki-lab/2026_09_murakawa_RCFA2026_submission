#!/usr/bin/env bash
# Reproduce CodaBench submission 980314 (config bal_ete_b8d) from the released embeddings.
# Usage: bash reproduce.sh DATA_DIR OUT_DIR [WORKERS]
#   DATA_DIR  folder with track1/ and track2/ (gallery.npz, probe_seq_embeddings/*.npz)
#   OUT_DIR   new output folder (the builder refuses to overwrite an existing submission.zip)
#   WORKERS   worker processes, 1-6 (default 6); the result does not depend on it
# Set PYTHON to choose the interpreter (default: python3).
set -euo pipefail
usage="usage: bash reproduce.sh DATA_DIR OUT_DIR [WORKERS]"
DATA_DIR=$(cd "${1:?$usage}" && pwd)
OUT_DIR=${2:?$usage}
WORKERS=${3:-6}
PY=${PYTHON:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT_DIR"
OUT_DIR=$(cd "$OUT_DIR" && pwd)

# One BLAS thread per process (the builder also sets these); results are bit-reproducible on Linux x86-64.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$HERE"
"$PY" -c 'import platform, sys, numpy; print("python", sys.version.split()[0], "| numpy", numpy.__version__, "|", platform.platform())'
date
RCFA_DATA="$DATA_DIR" "$PY" -B scripts/ogata/build_candidate_par.py bal_ete_b8d "$OUT_DIR" "$WORKERS"
"$PY" -B scripts/validate_submission.py "$OUT_DIR/submission.zip" --data-root "$DATA_DIR"
"$PY" -B scripts/compare_submission.py "$OUT_DIR/submission.zip"
date
