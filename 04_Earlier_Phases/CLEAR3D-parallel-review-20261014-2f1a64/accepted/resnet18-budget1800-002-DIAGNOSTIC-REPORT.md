# CLEAR3D-CT diagnostic report

Date: 2026-10-25
Verified: 2026-10-25 against the worker result, the diagnostics log, and fixed16-001.
Decision: expert review c993f865-511b-41e4-90be-00870e664b0d closed training. This file is the diagnostic report. It is not a detection paper, a benchmark, or a deployment record.

## Allowed statement

A ResNet18 patch-classifier engineering diagnostic finished 180 updates without leaving chance-level validation loss.

## What was run

The run is resnet18-budget1800-002 under /experiment-storage/chaidir/fawad-dsai/CLEAR3D-parallel-review-20261014-2f1a64/runs/. It is an additive ResNet18 diagnostic with an 1800-second envelope. It is not a resume of diagnostic-epochs18-001. The reviewed parent source was not edited. Reference release remains HOLD. cap4000 remains NOT_ADOPTED.

The worker result status is PASS. Endpoint epoch 2. Three epochs. 180 optimizer updates. 0 skipped updates. Device cuda. Peak allocated memory 763214336 bytes. Worker wall time 900.861 seconds.

Validation loss by epoch, from the worker result: 0.696184, 0.701457, 0.699039. Each epoch has 384 validation rows: 369 class 0 and 15 class 1. Updates per epoch: 60, 60, 60. Skips per epoch: 0, 0, 0.

Training weighted means from the class_losses records, which total 720 class 0 and 240 class 1 per epoch: epoch 0 class 0 = 0.649509 and class 1 = 2.228578; epoch 1 class 0 = 0.659179 and class 1 = 2.231206; epoch 2 class 0 = 0.670258 and class 1 = 2.183460.

The fixed16-001 worker result status is PASS. 24 updates, 0 skips, wall 88.642 seconds. Train weighted mean 0.719052 to 0.664510. memorization_indicator is false.

## Evidence check

Diagnostics log SHA256 is 75e99cc5b473494748846908773d0fc7a0cfba69bb37223c88944bfe1c3d9d52. Size 3787245 bytes. Recomputed evidence matches the worker diagnostics after a JSON round trip. Direct object equality is false because validation pairs are tuples in memory and lists in JSON. Counts: class_losses 1440, optimizer_boundary 180, complete_validation_diagnostics 3.

acceptance.json status is FAIL. Error text: ValueError('Stage failed; later stages NOT_RUN'). Watchdog reason: ValueError: Diagnostic evidence mismatch. Stages: research FAIL. Acceptance wall 904.204 seconds. The acceptance file was not rewritten, and the run was not repeated. diagnostic-epochs18-001 is still present.

## What this report does not say

It does not state sensitivity, FROC, CPM, AUC, or clinical performance. The checkpoint is not a detector and is not deployed. ResNet10 was not run. The six publication holds stay open, including reference origin and prior test access. A detection claim would need a later authorization after those holds are resolved, a candidate policy is adopted, and a patient-separated protocol is frozen before any test scoring.
