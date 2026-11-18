#!/bin/bash
set -u
ROOT='/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64'
ADOPTED="${1:?separate adopted disposition path required}"
ADOPTED_SHA="${2:?exact adopted disposition SHA256 required}"
test -d "$ROOT/runs" && test ! -L "$ROOT/runs" || exit 2
test ! -e "$ROOT/runs/fixed16-001" || exit 2
test ! -e "$ROOT/runs/fixed16-001.shell.log" || exit 2
test ! -e "$ROOT/runs/fixed16-001.shell.exit" || exit 2
set -o noclobber
env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH -u LD_PRELOAD PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=GPU-ca0a8f46-9b8e-e098-a97b-e0420b412e90 '/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/LIDC3D_reviews_and_zips_oct07_09/LIDC3D-offline-cpu-20261007T181841Z-IIGwzD/env/bin/python' -B "$ROOT/D-plan/fixed_batch.py" --execute --binding "$ROOT/D-plan/fixed16-binding.json" --binding-sha256 '7293e9f5e967799ab5393d30ec0bc081e27f44883b2ce2317e5aa3b84b045ea1' --disposition "$ADOPTED" --disposition-sha256 "$ADOPTED_SHA" --out "$ROOT/runs/fixed16-001" --acknowledge-shared-gpu-permission --acknowledge-sampled-risk > "$ROOT/runs/fixed16-001.shell.log" 2>&1
status=$?
printf '%s\n' "$status" > "$ROOT/runs/fixed16-001.shell.exit"
exit "$status"
