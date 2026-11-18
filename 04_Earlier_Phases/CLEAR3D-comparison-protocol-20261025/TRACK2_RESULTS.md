# Track 2 result, 2026-10-25

This is one scan-to-mark pass on the same confirmation patients used for Track 1. The primary number is CPM: the mean of sensitivity at 0.125, 0.25, 0.5, 1, 2, 4, and 8 false positives per scan.

Stage 1 is a fixed 96-voxel U-Net heatmap trained for 4 epochs on the 567 fit patients. It is not Wang's detector. Peaks are local maxima, suppressed at 8 mm, and capped at 200 per scan. Stage 2 rescores those peaks with the frozen Track 1 checkpoints. The classification threshold from Track 1 is not reused. Confirmation was scored once. The six reviewed patients were not opened.

The run used 83 confirmation scans, 121 annotated nodules, and 16,600 retained marks. Every scan hit the 200-mark cap. Wall time was 2,916 seconds. Of the 121 nodules, 64 (0.529) had at least one retained mark inside the nodule radius. The other 57 are candidate-generation misses and stay in the sensitivity denominator for every model.

## CPM

| model | CPM | 0.125 | 0.25 | 0.5 | 1 | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stage-1 heatmap | 0.0885 | 0.0413 | 0.0496 | 0.0661 | 0.0744 | 0.1074 | 0.1322 | 0.1488 |
| Zuo-style rescore | 0.3967 | 0.3388 | 0.3554 | 0.3884 | 0.4050 | 0.4050 | 0.4132 | 0.4711 |
| Wang-style rescore | 0.2987 | 0.1488 | 0.2231 | 0.2727 | 0.3058 | 0.3554 | 0.3884 | 0.3967 |
| CLEAR3D rescore | 0.1606 | 0.0909 | 0.0992 | 0.1157 | 0.1488 | 0.1653 | 0.2149 | 0.2893 |
| ablation rescore | 0.1771 | 0.1405 | 0.1488 | 0.1488 | 0.1570 | 0.1653 | 0.2149 | 0.2645 |

Sensitivity at each rate is the best operating point that does not exceed that false-positive rate. A mark matches a nodule when the world distance is within the nodule radius.

## Paired patient bootstrap

200 patient resamples, seed 17, of confirmation CPM for the CLEAR3D rescore minus each comparison.

- clear minus Zuo: mean -0.2358, interval -0.3218 to -0.1625
- clear minus Wang: mean -0.1409, interval -0.2064 to -0.0889
- clear minus stage-1: mean 0.0694, interval 0.0278 to 0.1167

CLEAR3D is below both frozen stage-2 reproductions, and those two intervals stay below zero. CLEAR3D is above the stage-1 heatmap score, and that interval stays above zero. Three comparisons were computed.

## Reading

On this confirmation set the Zuo-style rescore has CPM 0.397, the Wang-style rescore 0.299, CLEAR3D 0.161, and the ablation 0.177. The stage-1 list itself misses 57 of 121 nodules, so none of the stage-2 scores can recover those nodules. These figures are not the published Wang CPM of 0.878 or the published Zuo CPM of 0.830. Those numbers come from other detectors, other populations, and other candidate lists.

Saved marks: runs/phase-track2/marks.npz. Summary: runs/phase-track2/summary.json.
