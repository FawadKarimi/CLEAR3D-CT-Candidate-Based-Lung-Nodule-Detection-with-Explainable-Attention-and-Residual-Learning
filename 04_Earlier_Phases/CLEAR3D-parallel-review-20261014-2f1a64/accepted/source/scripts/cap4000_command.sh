#!/bin/sh
set -u
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
PY=/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/LIDC3D_reviews_and_zips_oct07_09/LIDC3D-offline-cpu-20261007T181841Z-IIGwzD/env/bin/python
set +e
env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" -B "$ROOT/work/scripts/cap4000_review.py" "$@"
rc=$?
printf '%s\n' "$rc" > "$CAP4000_EVIDENCE/worker.exit"
exit "$rc"
