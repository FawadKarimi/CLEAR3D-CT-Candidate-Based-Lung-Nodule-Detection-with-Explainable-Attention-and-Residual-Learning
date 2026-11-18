# Phase 2 experiment design

Status: prospective protocol; PENDING_VERIFIED_EXPERIMENT. This project is a new 3D implementation. It is independent of ConsenDxFormer and LungDxFormer. Only the authorised raw LIDC-IDRI CT/XML archive may be shared, read-only.

## Cohort and reference

Build a header inventory before any cohort count is reported. Use one reconstructable annotation-linked CT series per examination, retaining all alternatives and reason-coded exclusions. Use a 70/15/15 patient split with seed 42; repeated examinations remain in one partition. Test records remain withheld until all development decisions are frozen.

Primary positive references require at least three distinct readers and mean maximum axial diameter 3–30 mm. Use mean physical centroid and radius half the mean diameter. Build strict-majority contour masks for localization. Lower-agreement, size-ineligible and non-nodule marks are ignored regions. Point-only ignored marks use 3 mm radius, with 1.5 and 5 mm sensitivity replays.

The XML adapter must pass a real-data audit for linkage, contour coverage, reader identities, and competing associations. Unresolved associations block reference release. Any processing failure on an otherwise eligible frozen scan remains in evaluation as a zero-prediction scan.

## Reference configuration

Reconstruct HU, resample to 1 mm LPS, clip [-1000, 400] HU, normalize [0,1], and use 48 cubed one-channel patches. Proposals use CT-only lung masking, 5 mm mask dilation, LoG sigmas [1, 1.5, 2, 3] mm, 3 mm NMS and cap 400. Search thresholds [0.005, 0.01, 0.02, 0.04, 0.08] on validation once, maximizing candidate recall, then minimizing candidate count.

The classifier is width-16 3D ResNet-18 with GroupNorm, three independent 128-dimensional Transformer branches, two layers, four heads, feed-forward width 256, dropout 0.1 and hierarchical learned feature fusion. It outputs non-nodule/nodule logits. No pretraining, hard-negative mining, coordinate regression, malignancy label, calibration or training-time explanation alignment is enabled in the reference arm.

Train all non-ignored candidates once per epoch with weighted CE [1,3], AdamW, micro-batch 2, accumulation 8, five warm-up epochs, cosine decay to 1e-6, maximum 100 epochs, and validation CPM checkpointing. Run seeds 17, 42 and 2026.

## Matrix

| Arm | Change | Role |
| --- | --- | --- |
| L0 | LoG score only | deterministic proposal-score reference |
| C16 | Plain width-16 CNN | first confirmatory control |
| H16 | Full hybrid | primary model |
| C32 | Plain width-32 CNN | approximate compute control |
| S1 | One Transformer scale | exploratory scale ablation |
| P0 | No positional encoding | exploratory |
| FC / FG | Concatenation / gated fusion | exploratory fusion controls |
| PS32 / PS64 | Patch edge 32 / 64 | context sensitivity |
| UW | CE weights [1,1] | loss sensitivity |
| W200 | Window [-1000,200] | complete-system window sensitivity |
| MD0 | No lung-mask dilation | complete-system search-region sensitivity |
| CAP200 / CAP800 | Candidate caps | proposal-cap sensitivity |

Every neural arm uses the three seeds. H16−C16 and H16−C32 are the two confirmatory contrasts. All other contrasts are exploratory. The included code does not provide a learned proposal network, published multi-view reproduction, contemporary external comparator, pure Transformer or attention-alignment experiment. These require separately reviewed source and protocol changes.

## Evaluation

At each distinct score threshold, use maximum-cardinality one-to-one matches inside each lesion radius, then minimum normalized distance. Matched predictions are true positives. Unmatched eligible predictions are duplicate false positives. Unmatched noneligible predictions inside ignored regions are ignored; remaining predictions are false positives.

Primary FROC has lesion sensitivity versus FP per eligible scan. Custom CPM is the mean achieved sensitivity at 0.125, 0.25, 0.5, 1, 2, 4 and 8 FP/scan using an empirical staircase, not interpolation. This is not official LUNA16 evaluation. Select an illustrative ≤1 FP/scan operating threshold on validation only.

Candidate AUROC, sensitivity, specificity, precision, F1 and Brier score are secondary candidate-population measures, never cancer diagnosis results. Bootstrap patients 2,000 times with seed 2026 and report percentile intervals plus undefined replicate rate. The two confirmatory contrasts may use 10,000 synchronized patient-block permutations with Holm adjustment. Do not use independent-candidate DeLong or McNemar inference.

## XAI

Use the raw nodule-minus-non-nodule logit margin. Save Grad-CAM at the final pre-fusion CNN feature, IG from air and blurred references with completeness checks, and received-attention maps for all scales. Compare individual maps, heuristic [0.40, 0.35, 0.25] fusion, equal fusion, center Gaussian and seeded random maps. Use top 10% valid voxels with 5%/20% sensitivity checks, plus perturbation, stability and parameter-randomization reports. Spatial agreement is not proof of faithfulness.

## Gates

Before training: source inventory, XML/geometry audit, patient partition and access-history record, proposal diagnostics, frozen validation threshold, and GPU preflight. Before test: all development choices, run records and checkpoint hashes are frozen. No results/discussion prose is written until predictions, scan status, references, selection histories and metric recomputation are reviewed.
