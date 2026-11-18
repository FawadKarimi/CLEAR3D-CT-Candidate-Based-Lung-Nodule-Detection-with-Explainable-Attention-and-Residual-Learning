# Completion report, 2026-10-25

These numbers are computed from the frozen score files. No threshold, epoch, or checkpoint was changed. Confirmation was not scored again.

## Track 1, added metrics

PR-AUC is average precision. Calibration uses ten equal-width bins of the predicted positive probability. ECE is the count-weighted absolute gap between each bin's mean score and its observed positive rate. Brier is the mean squared error of that probability. The threshold is the one already stored for each model.

### development

Candidates 122689, patients 142, prevalence 0.002160, reference nodules 192, positive candidate rows 265, of which 265 match a reference within its radius.

| model | PR-AUC | ROC-AUC | ECE | Brier | threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| zuo | 0.4406 | 0.9780 | 0.1641 | 0.066970 | 0.9633 |
| wang | 0.2003 | 0.8483 | 0.1082 | 0.020171 | 0.3955 |
| clear | 0.2232 | 0.7979 | 0.3683 | 0.159217 | 0.8378 |
| ablation | 0.2163 | 0.8038 | 0.3492 | 0.139225 | 0.7073 |

Positive-candidate recall at that same threshold, by reference diameter. A row with no reference inside the radius is unmatched. This is not a new endpoint and it is not a subgroup superiority test.

| model | under 6 mm | 6 to 10 mm | over 10 mm | unmatched |
| --- | ---: | ---: | ---: | ---: |
| zuo | 0.777 (73/94) | 0.759 (41/54) | 0.598 (70/117) | n=0 |
| wang | 0.213 (20/94) | 0.444 (24/54) | 0.487 (57/117) | n=0 |
| clear | 0.000 (0/94) | 0.000 (0/54) | 0.641 (75/117) | n=0 |
| ablation | 0.000 (0/94) | 0.019 (1/54) | 0.650 (76/117) | n=0 |

Reference nodules on these scans: under 6 mm 97, 6 to 10 mm 53, over 10 mm 42.

### confirmation

Candidates 66135, patients 83, prevalence 0.002434, reference nodules 121, positive candidate rows 161, of which 161 match a reference within its radius.

| model | PR-AUC | ROC-AUC | ECE | Brier | threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| zuo | 0.5211 | 0.9751 | 0.1543 | 0.060796 | 0.9633 |
| wang | 0.2920 | 0.8655 | 0.1048 | 0.019161 | 0.3955 |
| clear | 0.1772 | 0.8143 | 0.3618 | 0.153946 | 0.8378 |
| ablation | 0.1924 | 0.8116 | 0.3452 | 0.136660 | 0.7073 |

Positive-candidate recall at that same threshold, by reference diameter. A row with no reference inside the radius is unmatched. This is not a new endpoint and it is not a subgroup superiority test.

| model | under 6 mm | 6 to 10 mm | over 10 mm | unmatched |
| --- | ---: | ---: | ---: | ---: |
| zuo | 0.692 (27/39) | 0.940 (47/50) | 0.500 (36/72) | n=0 |
| wang | 0.154 (6/39) | 0.520 (26/50) | 0.375 (27/72) | n=0 |
| clear | 0.000 (0/39) | 0.040 (2/50) | 0.486 (35/72) | n=0 |
| ablation | 0.000 (0/39) | 0.100 (5/50) | 0.431 (31/72) | n=0 |

Reference nodules on these scans: under 6 mm 40, 6 to 10 mm 50, over 10 mm 31.

## Sampling and boundaries

Fit bank rows 7872, positive 984, negative 6888, unique patients 565.
The fit bank stores each drawn candidate once. Epochs then draw those rows again with replacement. Repeated draws are not extra nodules.

| role | candidate rows | unique coordinates | repeated coordinates | reference nodules | interior crops | padded crops | outside centers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fit | 482139 | 482139 | 0 | 754 | 481404 | 735 | 0 |
| dev | 122689 | 122689 | 0 | 192 | 122408 | 281 | 0 |
| confirm | 66135 | 66135 | 0 | 121 | 65865 | 270 | 0 |

Series overlap is fit/dev 0, fit/confirm 0, dev/confirm 0, excluded-with-scored 0.
Slice-signature groups that contain two folders: 1. Cross-role slice matches: 0.
A slice signature is the shape, the affine, and three axial slices. Full-file hashes were computed only when two folders shared a signature.

## Track 2, FROC from the saved marks

Recomputed from marks.npz and the annotations. Scans 83, nodules 121, marks 16600. Wall time stored with the run: 2915.9941585063934 seconds. Peak VRAM was not sampled during that run, so no memory figure is added here.

| model | CPM | matches stored | coverage | missed | sensitivities at 0.125, 0.25, 0.5, 1, 2, 4, 8 |
| --- | ---: | --- | ---: | ---: | --- |
| spot | 0.0885 | yes | 0.5289 | 57 | 0.0413, 0.0496, 0.0661, 0.0744, 0.1074, 0.1322, 0.1488 |
| zuo | 0.3967 | yes | 0.5289 | 57 | 0.3388, 0.3554, 0.3884, 0.4050, 0.4050, 0.4132, 0.4711 |
| wang | 0.2987 | yes | 0.5289 | 57 | 0.1488, 0.2231, 0.2727, 0.3058, 0.3554, 0.3884, 0.3967 |
| clear | 0.1606 | yes | 0.5289 | 57 | 0.0909, 0.0992, 0.1157, 0.1488, 0.1653, 0.2149, 0.2893 |
| ablation | 0.1771 | yes | 0.5289 | 57 | 0.1405, 0.1488, 0.1488, 0.1570, 0.1653, 0.2149, 0.2645 |

Sensitivity inside each diameter bin uses the same global false-positive count. Bins are not separate operating points.

### spot

| bin | nodules | missed before any mark | sensitivity at the seven rates |
| --- | ---: | ---: | --- |
| under 6 mm | 40 | 35 | 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 |
| 6 to 10 mm | 50 | 19 | 0.000, 0.000, 0.000, 0.000, 0.000, 0.020, 0.020 |
| over 10 mm | 31 | 3 | 0.161, 0.194, 0.258, 0.290, 0.419, 0.484, 0.548 |

### zuo

| bin | nodules | missed before any mark | sensitivity at the seven rates |
| --- | ---: | ---: | --- |
| under 6 mm | 40 | 35 | 0.100, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125 |
| 6 to 10 mm | 50 | 19 | 0.460, 0.480, 0.540, 0.560, 0.560, 0.560, 0.600 |
| over 10 mm | 31 | 3 | 0.452, 0.452, 0.484, 0.516, 0.516, 0.548, 0.710 |

### wang

| bin | nodules | missed before any mark | sensitivity at the seven rates |
| --- | ---: | ---: | --- |
| under 6 mm | 40 | 35 | 0.025, 0.075, 0.100, 0.100, 0.125, 0.125, 0.125 |
| 6 to 10 mm | 50 | 19 | 0.180, 0.300, 0.360, 0.420, 0.460, 0.520, 0.520 |
| over 10 mm | 31 | 3 | 0.258, 0.290, 0.355, 0.387, 0.484, 0.516, 0.548 |

### clear

| bin | nodules | missed before any mark | sensitivity at the seven rates |
| --- | ---: | ---: | --- |
| under 6 mm | 40 | 35 | 0.000, 0.000, 0.000, 0.000, 0.000, 0.025, 0.050 |
| 6 to 10 mm | 50 | 19 | 0.000, 0.000, 0.020, 0.040, 0.060, 0.100, 0.200 |
| over 10 mm | 31 | 3 | 0.355, 0.387, 0.419, 0.516, 0.548, 0.645, 0.742 |

### ablation

| bin | nodules | missed before any mark | sensitivity at the seven rates |
| --- | ---: | ---: | --- |
| under 6 mm | 40 | 35 | 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.050 |
| 6 to 10 mm | 50 | 19 | 0.080, 0.080, 0.080, 0.100, 0.120, 0.180, 0.240 |
| over 10 mm | 31 | 3 | 0.419, 0.452, 0.452, 0.452, 0.452, 0.548, 0.581 |

## Hashes

SHA-256 of the large October banks and of every checkpoint at least 100 MB under the October run directory. The files were not deserialized and were not copied.

- `73d4ce4922b583c0cf033d8c1c754bc4b9c68b19c47e6daee6c3f6bcffc29ef8`
  /experiment-storage/chaidir/fawad-dsai/CLEAR3D-parallel-review-20261014-2f1a64/runs/fixed16-001/state/engineering-checkpoints/4b588d3680eb45f9a5efd36d950052d6/state.pt (136357265 bytes)
- `9bf4c9929d03d8179d6a49f773303819cfce68263e0371c61dc4443e7e7bfc27`
  /experiment-storage/chaidir/fawad-dsai/CLEAR3D-parallel-review-20261014-2f1a64/runs/fixed16-001/state/engineering-checkpoints/d482c45cc3c04e8bbfe41f14da47c4d6/state.pt (408945855 bytes)
- `c391ff9a81fa630a4a37592255a2df4541a4153e0d0ae9bd15fdb744d46f8d78`
  /experiment-storage/chaidir/fawad-dsai/CLEAR3D-parallel-review-20261014-2f1a64/runs/frozen-clean10-001/best-state.pt (136325031 bytes)
- `12135d8af0d2343e585dded0e82fb0089f816a156ee1224d119240a86ba2d105`
  /experiment-storage/chaidir/fawad-dsai/CLEAR3D-parallel-review-20261014-2f1a64/runs/luna-expand-001/banks.npz (801307426 bytes)
- `5760482a47763aa10608bd07ec0eaa0efadf6d608ee413842b333e8012909684`
  /experiment-storage/chaidir/fawad-dsai/CLEAR3D-parallel-review-20261014-2f1a64/runs/med48-001/banks.npz (1667711458 bytes)
- `b8cc70c43ee791d0c063c92a67d51adaa5d7d64b6d57b54fb5c13416d899fd92`
  /experiment-storage/chaidir/fawad-dsai/CLEAR3D-parallel-review-20261014-2f1a64/runs/med48-001/best-state.pt (136325031 bytes)
- `d83ff40c806fec15ca8345b4caf0663c790801fd0326365f1be457d6e935abdd`
  /experiment-storage/chaidir/fawad-dsai/CLEAR3D-parallel-review-20261014-2f1a64/runs/pretrain-depth18-001/best-state.pt (136325031 bytes)
- `c96b3c7173d5cbd8743949d93cf518bcd5d4248d1a8ef59eb4da6a2a492633dd`
  /experiment-storage/chaidir/fawad-dsai/CLEAR3D-parallel-review-20261014-2f1a64/runs/resnet18-budget1800-002/research/state/epoch-checkpoints/1b218786fe1341429e2254131ffad6bf/state.pt (136373517 bytes)
- `f67b51e9682574e263d6f685b95341fbf7ce045fda16511c1efc525b23843c69`
  /experiment-storage/chaidir/fawad-dsai/CLEAR3D-parallel-review-20261014-2f1a64/runs/resnet18-budget1800-002/research/state/epoch-checkpoints/5130091f10684e06b131ee1c07c2480a/state.pt (408966267 bytes)
- `1a21253597d6617bd02edba664a73d78d5ab9e139d7434eddbb24c2cef1ad1a1`
  /experiment-storage/chaidir/fawad-dsai/CLEAR3D-parallel-review-20261014-2f1a64/runs/resnet18-budget1800-002/research/state/epoch-checkpoints/6a27ad08fcd747e6b6fcfd95863b8ad5/state.pt (408966267 bytes)
- `5456358991efc7a6b4c8f8dc29c00f5b83af68d39748b460d35e5e84b6446b6d`
  /experiment-storage/chaidir/fawad-dsai/CLEAR3D-parallel-review-20261014-2f1a64/runs/resnet18-budget1800-002/research/state/epoch-checkpoints/77ccc110c07247caa92da99612e61dd2/state.pt (408966267 bytes)
- `853d4054157c3969cd77ea22da295588aeffda98be2f26f5ffd25518732821e7`
  /experiment-storage/chaidir/fawad-dsai/CLEAR3D-parallel-review-20261014-2f1a64/runs/resnet18-constlr-001/research/state/epoch-checkpoints/386a27e54ccd436d8ae3f775c5b9e835/state.pt (136373517 bytes)
- `fadefcf8efcd618caba0321f7e78b3fe0169abbd6002120c8eb629e34bf373c9`
  /experiment-storage/chaidir/fawad-dsai/CLEAR3D-parallel-review-20261014-2f1a64/runs/resnet18-constlr-001/research/state/epoch-checkpoints/3b7b1cbc789c4d07856f675d2b526903/state.pt (408966267 bytes)
- `8fd75fd75540c807866046fdf554ace3a26b86e4374016b1ac077561c66e502c`
  /experiment-storage/chaidir/fawad-dsai/CLEAR3D-parallel-review-20261014-2f1a64/runs/resnet18-constlr-001/research/state/epoch-checkpoints/9d91019330fe46b49600a41a20faad84/state.pt (408966267 bytes)
- `bc74a744c55c51c0aaceea8beb9eec433f19edccb602f16b9e305e11a38c45b5`
  /experiment-storage/chaidir/fawad-dsai/CLEAR3D-parallel-review-20261014-2f1a64/runs/resnet18-constlr-001/research/state/epoch-checkpoints/e591f5f2aaca4b2488906e1c49da9edc/state.pt (408966267 bytes)
- `518172a3a79862aff6354ae7cfa40a4f19b4148c316c06da99c0b6c47d319cec`
  /experiment-storage/chaidir/fawad-dsai/CLEAR3D-parallel-review-20261014-2f1a64/runs/resnet18-lr10x-002/research/state/epoch-checkpoints/44d4201160554b80863af300dc399c38/state.pt (136373517 bytes)
- `5971bb1c0dd0d74bd3afba1ea2dd9b298c6c74be9d8207e3c25db9d9c606ce0c`
  /experiment-storage/chaidir/fawad-dsai/CLEAR3D-parallel-review-20261014-2f1a64/runs/resnet18-lr10x-002/research/state/epoch-checkpoints/99ffee7550df4ad1a8b43d695edeaadd/state.pt (408966267 bytes)
- `8892987a2d9c11b1d72d1eb30d13b09fa3da5960e8cd8ed431dd71fc935c6ba9`
  /experiment-storage/chaidir/fawad-dsai/CLEAR3D-parallel-review-20261014-2f1a64/runs/resnet18-lr10x-002/research/state/epoch-checkpoints/bf01c22059ea4c75ae659dcb8becec7f/state.pt (408966267 bytes)
- `cddaa01ccb3e084812634501122fe374a3c048e7d5bb8a222385a5c97ea7449a`
  /experiment-storage/chaidir/fawad-dsai/CLEAR3D-parallel-review-20261014-2f1a64/runs/resnet18-lr10x-002/research/state/epoch-checkpoints/f08bca5d4f8c4aeeb7f9affde0af110f/state.pt (408966267 bytes)

## What this pass does not change

- The published Wang stage-1 detector and the paper's anisotropic 32x48x48, 16x24x24, and 8x12x12 stage-2 inputs were not trained. Building that network after the confirmation read would be a new method, and the confirmation set is closed.
- No vessel or airway label exists on these volumes. None was invented. Hard-negative weights in the finished run remain the training-bank losses after epoch 2.
- The finished runs stored a state dict and a threshold. Their optimizer and RNG state were not kept, so those files are not resume sources. New training was not started, so a replacement checkpoint was not written over them.
- Learning rate stayed at the values fixed before the first development score. Focal loss was not added. Either change would select a new configuration after confirmation.
- Intensity augmentation beyond the training-only flips was not applied to the frozen models.
- CLEAR3D remains below both reproductions on the frozen confirmation endpoint. This report does not replace that result.

The same-role full-file match is fit patients LIDC-IDRI-0138 and LIDC-IDRI-0029. Both image files share one sha256. They are not in development or confirmation.
