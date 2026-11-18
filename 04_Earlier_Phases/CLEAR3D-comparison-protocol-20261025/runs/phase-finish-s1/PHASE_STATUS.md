# Phase status, 2026-10-28

Confirmation remains closed. No threshold, epoch, or architecture was chosen from confirmation scores.

| Phase | State | Evidence |
| --- | --- | --- |
| A Reconcile | DONE | Registry, exposure ledger, large-file hashes, and the fit-only duplicate pair LIDC-IDRI-0138 / LIDC-IDRI-0029. |
| B Protocol | DONE | Track 1 and Track 2 endpoints adopted. PR-AUC, calibration, and diameter subgroups are in COMPLETION.md. Ground-glass, pleural, and vessel-adjacent labels do not exist here. |
| C Baselines | DONE WITH DEPARTURES | Zuo-style, cubic Wang-style, anisotropic Wang stage-2, CLEAR3D, ablation, and a disclosed Wang-like stage-1 U-Net. Not byte-faithful paper networks. |
| D Sampling | DONE WHERE POSSIBLE | Patient weights, unique-candidate accounting, edge-crop valid fractions, flips, rotation, translation, and intensity augmentation. Vessel and airway mining is impossible without inventing labels. |
| E Optimization | DONE WHERE POSSIBLE | Finite-loss checks, class-weight balance, focal-loss comparison, second learning rate, and full resume checkpoints on the development review runs. |
| F Ladder | DONE | CPU checks, smokes, development runs, ablation, and one closed confirmation per track. CLEAR3D did not improve the development endpoint. |

## Development Track 2 from the Wang-like stage-1

Patients 142, scans 142, nodules 192, marks 28400, block 96, wall 10162.3 s.

| model | development CPM | coverage | missed | sensitivities |
| --- | ---: | ---: | ---: | --- |
| wang_s1 | 0.0067 | 0.0885 | 175 | 0.0000, 0.0000, 0.0052, 0.0052, 0.0052, 0.0104, 0.0208 |
| zuo | 0.0469 | 0.0885 | 175 | 0.0156, 0.0312, 0.0365, 0.0417, 0.0521, 0.0677, 0.0833 |
| wang | 0.0253 | 0.0885 | 175 | 0.0208, 0.0260, 0.0260, 0.0260, 0.0260, 0.0260, 0.0260 |
| clear | 0.0305 | 0.0885 | 175 | 0.0208, 0.0208, 0.0260, 0.0365, 0.0365, 0.0365, 0.0365 |
| ablation | 0.0253 | 0.0885 | 175 | 0.0208, 0.0208, 0.0260, 0.0260, 0.0260, 0.0260, 0.0312 |

These development CPM values are not confirmation results and are not compared to the closed confirmation table for selection.

## Permanent limits

- No vessel or airway mask exists on the prepared volumes. Mining those structures was not done.
- Wang's published ASCRM was not reconstructed from the paper text. The stand-in attention residual is disclosed above.
- Confirmation cannot be rescored. The frozen Track 1 and Track 2 confirmation numbers stand.
- CLEAR3D remains below the reproductions on the frozen confirmation endpoints.

