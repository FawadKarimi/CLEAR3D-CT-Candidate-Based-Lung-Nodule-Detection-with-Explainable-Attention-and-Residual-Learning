# Comparison protocol

Status: DRAFT. This file does not authorize a GPU run.

Track 1 is false-positive reduction. A future Wang stage-2 reproduction, a future Zuo reproduction, and one CLEAR3D classifier must use the same patients, the same candidate coordinates, the same matching rule, and the same exclusion list. Model-specific input size is allowed and must be recorded. This track scores discrimination among supplied candidates.

Track 2 is complete detection, from scan to final marks. Candidate-generation misses count. Scans with no retained candidate stay in the denominator. Report sensitivity at 0.125, 0.25, 0.5, 1, 2, 4, and 8 false positives per scan, CPM, the full FROC curve, and a paired patient-level interval. Reference-centred patch accuracy is not Track 2.

Primary endpoints: Track 1 positive-class F1 at a threshold chosen only on development data; Track 2 CPM. Also report recall, precision, specificity, accuracy, ROC-AUC, PR-AUC, confusion counts, prevalence, calibration, and prespecified subgroups.

The redesign-001 named test and the 384-row slice have already been read. Neither is a confirmation set. No epoch, threshold, or architecture is chosen from confirmation errors.

The only planned models are a faithful Zuo reproduction, a Wang stage-2 reproduction, and one CLEAR3D candidate with a fine branch, a context branch, residual blocks, and light fusion, plus the same candidate with fusion removed. No transformer, segmentation head, or ensemble is in this ladder. Pretraining stays off until a weight file and its population overlap are adopted.

A point estimate above a published number is not the same as an improvement over our own reproduction on shared patients. `isolated/clear3d_compare/gate.py` stays closed while the population and GPU budget are DRAFT.
