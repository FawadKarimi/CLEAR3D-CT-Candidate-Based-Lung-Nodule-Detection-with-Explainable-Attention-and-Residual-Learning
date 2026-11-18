#!/bin/bash
set -u
set -o noclobber
ADOPTED_SHA256="$1"
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/LIDC3D_reviews_and_zips_oct07_09/LIDC3D-offline-cpu-20261007T181841Z-IIGwzD/env/bin/python -B - <<'GATE'
from pathlib import Path
p=Path('/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab/runs/diagnostic-epochs18-001')
assert not any(x.is_symlink() for x in [p,*p.parents])
assert p.parent.is_dir() and not p.exists()
assert not Path(str(p)+'.shell.log').exists() and not Path(str(p)+'.shell.exit').exists()
GATE
gate_status=$?
[ "$gate_status" -eq 0 ] || exit "$gate_status"
/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/LIDC3D_reviews_and_zips_oct07_09/LIDC3D-offline-cpu-20261007T181841Z-IIGwzD/env/bin/python -B /experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab/work/scripts/diagnostic_training_watchdog.py --execute --mode research --depth 18 --binding /experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab/training-binding.json --binding-sha256 adb44927810502300f8931421ab867c27054b4c081d0f2061b55127349047310 --out /experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab/runs/diagnostic-epochs18-001 --disposition /experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab/ADOPTED-training-disposition.json --disposition-sha256 "$ADOPTED_SHA256" --acknowledge-shared-gpu-permission --acknowledge-sampled-risk > /experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab/runs/diagnostic-epochs18-001.shell.log 2>&1
worker_status=$?
printf "%s\n" "$worker_status" > /experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-diagnostic-training-review-20261013-6e91ab/runs/diagnostic-epochs18-001.shell.exit
exit "$worker_status"
