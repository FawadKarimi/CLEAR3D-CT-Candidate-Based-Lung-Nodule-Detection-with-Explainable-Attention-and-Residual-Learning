# CLEAR3D-CT current state

Inspection date: 25 October 2026. Read-only reconciliation. No result is adopted, no reference is released, and no training run is authorized.

## Standing constraints

- Reference release: HOLD.
- cap4000: NOT_ADOPTED as a scientific policy.
- Historical evidence was not modified. Checkpoints were not deserialized.
- The six reviewed validation patients and the other prepared validation scans were not opened for a new evaluation.
- GPU training was not started.

## Prepared data

`luna16-prepare-development-fold0-v1` has 710 train scans from 709 patients and 89 val scans from 89 patients. It has no scan labeled test. The reviewed 384-row slice is six of those val patients: LIDC-IDRI-0020, 0428, 0634, 0722, 0865, 0895. The other 83 val patients are unsettled and were not used by the October trainers. About 89 LUNA16 series are absent from this preparation and were not sought.

## redesign-001

Finished 2026-10-25T10:02:54Z. Source SHA-256 d0477b881b200d9ed39349af6ae233ae36548f98e097454b9bfe5bc938a54dc6. Random multibranch 18 cubed CNN. Reference-centred positives that pass a brightness filter. Seven class-0 candidates per positive. A read-only recompute of lines 1-97 reproduced train 251/431, validation 54/95, named test 54/86, with empty patient and series overlap. The six reviewed patients are in none of those groups. Epoch 11 and threshold 0.9985646686159578 were chosen on the script validation split. The named test was then read once: TP 70, FN 16, FP 6, TN 596, positive F1 0.8641975308641974. Per-candidate scores were not saved. `zuo_beat` and `wang_beat` are 0. This is development evidence, not an untouched confirmation set, and not a reproduction of either paper.

## Cross-run exposure

All 30 luna-expand stop patients fall inside the redesign pool: 18 train, 8 validation, 4 named test. The four named-test patients are LIDC-IDRI-0042, 0054, 0080, and 0090. LIDC-IDRI-0694 is in redesign training and was the hold patient for earlier checkpoint selection.

## Not established

No untouched test set is claimed. No Wang or Zuo reproduction has been trained. No superiority claim is made.
