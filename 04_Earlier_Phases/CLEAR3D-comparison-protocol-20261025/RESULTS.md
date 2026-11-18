# Phase C-F result, 2026-10-25

Track 1 is false-positive reduction on supplied LUNA candidates. The primary number is positive-class F1 at a threshold chosen on the development candidates and then frozen. Confirmation was scored once.

Track 2, a scan-to-mark detector and its CPM, was not run. These figures are a candidate-list comparison. They are not a detection CPM, and they are not a comparison against the published Wang or Zuo tables. Those published figures come from other populations and other training recipes.

## Population

Prepared train patients were shuffled with seed 17 and cut 80/20 into fit and development. Confirmation is the prepared validation patients except the six already scored. Those six are in neither set. Absent LUNA series were not retrieved. The brightness filter was off.

- Fit patients: 567
- Development patients: 142
- Confirmation patients: 83
- Excluded reviewed patients: LIDC-IDRI-0020, LIDC-IDRI-0428, LIDC-IDRI-0634, LIDC-IDRI-0722, LIDC-IDRI-0865, LIDC-IDRI-0895
- Overlap of those six with fit, development, and confirmation: empty
- Fit candidates: 481155 negative, 984 positive
- Development candidates: 122424 negative, 265 positive
- Confirmation candidates: 65974 negative, 161 positive
- Absent-series candidate rows left unused: 79135
- Excluded reviewed candidate rows: 4877
- Fit bank: 7872 crops, 984 positives, 0 centers outside the array
- Wall time: 1491 seconds on GPU-ca0a8f46-9b8e-e098-a97b-e0420b412e90
- One confirmation read. The set is closed for training, epoch choice, and threshold choice.

## Confirmation at the frozen threshold

| model | F1 | recall | precision | specificity | accuracy | AUC | TP | FP | FN | TN | threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zuo | 0.4878 | 0.6832 | 0.3793 | 0.9973 | 0.9965 | 0.9751 | 110 | 180 | 51 | 65794 | 0.9633 |
| wang | 0.2776 | 0.3665 | 0.2235 | 0.9969 | 0.9954 | 0.8655 | 59 | 205 | 102 | 65769 | 0.3955 |
| clear | 0.1697 | 0.2298 | 0.1345 | 0.9964 | 0.9945 | 0.8143 | 37 | 238 | 124 | 65736 | 0.8378 |
| ablation | 0.1841 | 0.2236 | 0.1565 | 0.9971 | 0.9952 | 0.8116 | 36 | 194 | 125 | 65780 | 0.7073 |

Confirmation has 161 positives and 65974 negatives, so accuracy stays high for every model. Positive F1 is the prespecified endpoint.

## Paired patient bootstrap

200 patient resamples, seed 17, of confirmation positive F1 for CLEAR3D minus each other model. An interval below zero means CLEAR3D is lower on this set.

- clear minus zuo: mean -0.3174, interval -0.3938 to -0.2398
- clear minus wang: mean -0.1088, interval -0.1643 to -0.0524
- clear minus ablation: mean -0.0122, interval -0.0540 to 0.0261

CLEAR3D minus Zuo and CLEAR3D minus Wang stay below zero. CLEAR3D minus the ablation includes zero.

## Development scores

These scores chose the threshold. They are not a second confirmation.

- zuo: F1 0.4187, recall 0.6943, precision 0.2997, AUC 0.9780, threshold 0.9633, TP 184 FP 430 FN 81 TN 121994
- wang: F1 0.2298, recall 0.3811, precision 0.1645, AUC 0.8483, threshold 0.3955, TP 101 FP 513 FN 164 TN 121911
- clear: F1 0.1706, recall 0.2830, precision 0.1221, AUC 0.7979, threshold 0.8378, TP 75 FP 539 FN 190 TN 121885
- ablation* F1 0.1752, recall 0.2906, precision 0.1254, AUC 0.8038, threshold 0.7073, TP 77 FP 537 FN 188 TN 121887

## Departures from the papers

- Shared cubic 32-voxel crops and per-crop z-score.
- Zuo view is the center 18 voxels. Wang views are cubic 32, 16, and 8, not 32x48x48, 16x24x24, and 8x12x12.
- Negatives are a seeded draw of 7 per positive from the training patients, not every LUNA candidate. Wang uses a seeded 3-per-positive subset of that draw.
- No brightness filter and no lung-mask exclusion. A candidate is ignored only when its center is outside the array.
- Epoch count is fixed at 8 before evaluation. Threshold is chosen once on the development candidates.
- Hard-negative weights after epoch 2 use training-bank losses only. There is no vessel mask.

Epoch count was fixed before any development score. Zuo loss fell from 0.613 to 0.248. Wang stayed near 0.44 to 0.51. CLEAR3D stayed near 0.59 to 0.64. The ablation stayed near 0.60 to 0.64. No epoch or threshold was changed after confirmation.

## Reading

On this confirmation set the Zuo-style net has positive F1 0.488, the Wang-style net 0.278, CLEAR3D 0.170, and the ablation 0.184. CLEAR3D is below both reproductions. The paired intervals for those two gaps exclude zero. The fusion branch does not separate from the ablation. None of these runs reproduces the published Wang or Zuo classification tables, and none is a detection CPM.

Saved scores: runs/phase-cf/*-dev-scores.npz and runs/phase-cf/*-confirm-scores.npz. The smoke run used 40 fit and 10 development patients and is stored under runs/phase-cf/smoke-keep. It is not this split.
