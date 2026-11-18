# Decisions

| Decision | State | Effect |
| --- | --- | --- |
| Reference release | ADOPTED: HOLD | No reference release is performed. |
| cap4000 scientific policy | ADOPTED: NOT_ADOPTED | cap4000 is not a scientific result. |
| Modify historical evidence or the live LIDC3D checkout | ADOPTED: forbidden | This package is a new directory. |
| Use the 83 other prepared validation patients | DRAFT: unsettled | No training and no confirmation use. |
| Retrieve series absent from the fold0 preparation | DRAFT: unsettled | Do not locate or download them. |
| Treat redesign-001 named test as untouched | REJECTED | It was read once and contains four earlier stop-pool patients. |
| Treat the 384-row validation slice as untouched | REJECTED | Multiple finished runs have scored it. |
| Remove the brightness component filter from the benchmark | DRAFT policy for this package | eligibility.py does not implement that filter. |
| Shared Wang/Zuo/CLEAR3D manifest | DRAFT | Not built. Execution waits on this. |
| GPU budget of 4 trials and 8 hours | DRAFT | Gate closed. Zero hours used by this reconciliation. |
| Pretraining | DRAFT: off | No weight file is bound for the next candidate. |
| Publication or a detection-superiority claim | ADOPTED: not authorized | No such claim is made. |

The launch gate requires the evaluation population and the GPU budget to be ADOPTED. Both are DRAFT, so no training command is valid.
## execution_adoption_20261025
evaluation_population: ADOPTED
gpu_budget: ADOPTED
Development uses prepared train patients, seed 17, 80/20 fit/dev. Confirmation is prepared val except the six reviewed patients, scored once after the threshold is frozen. It is not used for training, epoch choice, or threshold choice. The redesign named test is not used. Absent series are not retrieved. The brightness filter stays off. reference_release remains HOLD. cap4000 remains NOT_ADOPTED.
GPU budget: one smoke, then four trials in one process on GPU-ca0a8f46-9b8e-e098-a97b-e0420b412e90 when at least 10000 MiB are free. No new model or epoch after 7 hours. No automatic extension. One confirmation read. No architecture search.

## execution_result_20261025
The four-trial process finished in one confirmation read. Do not score that confirmation set again for training, epoch choice, or threshold choice. Track 1 is recorded in RESULTS.md. Track 2 was not run. CLEAR3D confirmation positive F1 is 0.170, below the Zuo reproduction at 0.488 and the Wang reproduction at 0.278. The paired intervals for those two gaps exclude zero. The ablation gap includes zero.

## track2_adoption_20261025
Track 2 is one added process. It does not change Track 1 thresholds and it does not open the six reviewed patients.
Stage 1 is a fixed 96-voxel U-Net heatmap, stride 64, trained 4 epochs on fit patients only. No epoch is chosen from a score. At most 200 peaks are kept per scan after 8 mm suppression. A block is scored only when a coarse subsample reaches 0.20 on the stored -1000 to 400 normalization.
Stage 2 rescores those peaks with the four frozen Track 1 checkpoints. Confirmation is scored once for the detection curve. Development scans are used only by the smoke run, which does not open confirmation and does not change the detector.
Wall cap is 3 hours. There is no second confirmation pass. This stage 1 is not Wang cross-layer detector.

## track2_result_20261025
The detection pass finished in one confirmation read. Do not score that set again for detector or threshold changes. CLEAR3D CPM is 0.161. The Zuo-style rescore is 0.397 and the Wang-style rescore is 0.299. Both paired gaps stay below zero. Stage 1 missed 57 of 121 nodules. Details are in TRACK2_RESULTS.md.

completion_report_20261025: metrics, hashes, duplicate signatures, and boundary counts were computed from frozen files. Confirmation was not rescored. No threshold was changed.

development_review_20261027: fit-only development runs finished for wang_aniso, zuo_stride, clear_aug, clear_focal, and clear_lr. Confirmation stayed closed. Wall 1159 s on GPU-ca0a8f46.

phases_complete_20261028: A-F closed under adopted limits. Wang-like stage-1 scored on development only. Confirmation stayed closed. Vessel mining impossible without labels.
