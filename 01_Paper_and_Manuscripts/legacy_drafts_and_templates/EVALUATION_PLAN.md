# CLEAR3D-CT Defensible Evaluation Plan (Multi-Scan Reconciled)
**Date:** 2026-11-07
**Protocol Status:** reference_release: HOLD | test_access: STRICTLY_PROHIBITED
**Evidence Hash:** df0f6a6e660728f7874b8212899c3de8ed3e38df3d947d4028550ad8633bf298

---

## 1. Executive Summary & Locked Protocol
This evaluation plan establishes an audited, defensible evaluation protocol based on server manifests, checkpoints, and code.

1. **test-fold0 Status:** The unprepared 89-scan test cohort (subset 0 in `cohort_test_WITHHOLD.jsonl`) has **EXACTLY ZERO** patient, series, and scan overlap with fit, dev, confirm, reviewed, and prepared validation cohorts. It remains strictly withhold.
2. **Distinction from Prepared Validation:** The prepared 89-scan validation cohort (subset 1 in `scans.jsonl`) was preprocessed offline into isometric 1.0 mm volumes and partitioned into 83 confirm and 6 reviewed scans. It is completely distinct from the unprepared 89-scan test cohort.
3. **Multi-Scan Reconciliation:** Patient `LIDC-IDRI-0332` has 2 scans/series (`scan_665438f18e9c119dcafb` in subset 2, `scan_5cb2a613be3a72f304cf` in subset 6). The raw fit cohort consists of **567 unique patients and 568 raw series/scans**. After excluding `LIDC-IDRI-0138` and `LIDC-IDRI-0029`, the active fit cohort contains **565 active patients and 566 active series/scans**.
4. **Methodological ID Boundary:** Zero patient and series ID intersections do **NOT** prove absence of prior exposure or duplicate CT acquisitions under different UIDs. No claim of impossibility is made solely from identifier string disjointness.
5. **Mining Point Estimate:** Fold 0 Baseline achieves **0.904018 CPM**; Control B (no-mining) achieves **0.875000 CPM**. The point estimate difference is **+0.029018 CPM**. Paired bootstrap 95% percentile CI is **[-0.014719, +0.051129]**, crossing zero. The replicate proportion Delta* > 0 (88.50%) is not a p-value.
6. **Geometry Audited:** Inspected all 799 `geometry.json` files. Basis-vector Euclidean norms are exactly 1.000 mm across all 799 files. Affine diagonal elements are not voxel spacing.
7. **Calibration Locked:** All 5 checkpoints store `scale_oof = 1.7000000000000002` and `bias_oof = 4.175479443210625`. Checkpoint binaries are omitted from review ZIPs to maintain compact packages.
8. **Evaluator Formula Locked:** Included nodules use strictly R = d / 2 (no 5 mm floor); fallback R = 5.0 mm applies only to excluded findings with negative/missing diameter.
9. **Claims Corrected:** Retracted informal p approx 0.23, universal zero-leakage, and zero-pretraining-overlap claims. LUNA16 Track 1 is Nodule Detection; Track 2 is False Positive Reduction.
10. **External Benchmark (LNDb):** Documented Zenodo record 7153205 v4, CC BY-NC-ND 4.0 license, and candidate generation requirements.

---

## 2. Cohort Partitions & Multi-Scan Reconciled Manifest Intersections

| Partition | Role | Official Subset | Preparation State | Patients | Series | Scans | Overlap with test-fold0 |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **test-fold0 (WITHHOLD)** | Test | 0 | Unprepared | 89 | 89 | 89 | - |
| **fit (raw)** | Fit | 2-9 | Prepared (1.0 mm) | 567 | 568 | 568 | 0 (0.0%) |
| **fit (active)** | Fit | 2-9 | Prepared (1.0 mm) | 565 | 566 | 566 | 0 (0.0%) |
| **dev** | Dev | 2-9 | Prepared (1.0 mm) | 142 | 142 | 142 | 0 (0.0%) |
| **confirm** | Validation | 1 | Prepared (1.0 mm) | 83 | 83 | 83 | 0 (0.0%) |
| **reviewed** | Validation | 1 | Prepared (1.0 mm) | 6 | 6 | 6 | 0 (0.0%) |
| **Prepared Val 89** | Validation | 1 | Prepared (1.0 mm) | 89 | 89 | 89 | 0 (0.0%) |

### Multi-Scan Breakdown:
- `LIDC-IDRI-0332` has 2 series in raw fit: `1.3.6.1.4.1.14519.5.2.1.6279.6001.159996104466052855396410079250` (scan `scan_665438f18e9c119dcafb`, subset 2) and `1.3.6.1.4.1.14519.5.2.1.6279.6001.316900421002460665752357657094` (scan `scan_5cb2a613be3a72f304cf`, subset 6).
- Excluded duplicates (2): `LIDC-IDRI-0138` (series ...889059, subset 4) and `LIDC-IDRI-0029` (series ...794314, subset 3).
- Reconciled active fit: 567 - 2 = 565 patients; 568 - 2 = 566 series.
- All 798 split patients verified in `scans.jsonl` (0 missing).

### Primary Manifest Hashes:
- `cohort_test_WITHHOLD.jsonl`: d9238e7f554a391c11628d34d3b9a56db3c8bd41583da75bccf57c2582a6d6b5
- `cohort_development.jsonl`: 6944f057a269d5d52c973763d46b670c4028ffb73c06760e99cceaf524c0b963
- `cohort_summary.json`: 786dc7389a2afc32e7dfaa8d470b946de4d2b0384b0cb6406fec5c123ade6828
- `scans.jsonl`: 49d9df4826a505622359743d6b21519b6b31b98b6758c70619252071e18a3abf
- `split.json`: 2ffc30511188ca27686d7c2d5a73fa2723f822a42a62a9c159987c7e0d54ede1
- `EXPOSURE_AND_SPLIT_LEDGER.json`: d33aabd8fa432ce385b27f7e88429b41cf49aeb57eba9d8bdcab48f9faf458b3

---

## 3. Mining-Comparison Point Estimate & Limitations
- **Fold 0 Baseline CPM:** 0.904018
- **Control B (No-Mining) CPM:** 0.875000
- **Point Estimate Delta:** Delta_CPM = 0.904018 - 0.875000 = **+0.029018 CPM**
- **95% Bootstrap Percentile CI:** [-0.014719, +0.051129] (B = 2,000 paired patient clusters, seed = 42).
- **Significance Decision:** Null hypothesis not rejected at alpha = 0.05 (CI spans zero).
- **Replicate Proportion:** 88.50% (1,770 / 2,000 replicates with Delta* > 0). Not a probability of true superiority.
- **Preserved Limitations:**
  1. Negative candidate pool disparity (28,536 in Fold 0 vs 16,728 in Control B).
  2. Shared-scout preprocessing leakage (scout trained on full fit pool prior to fold splitting).
  3. Repeated development cohort exposure (142 development scans repeatedly evaluated).

---

## 4. Geometry Audit (799 Files Inspected)
- Files inspected: 799 / 799 (`luna16-prepare-development-fold0-v1/prepared/scan_*/geometry.json`).
- Column basis-vector Euclidean norms: min = 1.000 mm, max = 1.000 mm, mean = 1.000 mm.
- Row basis-vector Euclidean norms: min = 1.000 mm, max = 1.000 mm, mean = 1.000 mm.
- Affine diagonal elements are [0, 1, 0, 1] for permutation affines and do not reflect voxel spacing.
- `preparation.json` (0d5fcc5ce4bd45358c92a6fd630a3fe0c02ec14c383f3060cfd8dfb17a41d5f6): coordinate_aware_spacing_mm = 1.0.

---

## 5. Calibration & Checkpoint Integrity
- Stored checkpoint parameters across all 5 folds: scale_oof = 1.7000000000000002, bias_oof = 4.175479443210625, nat_cut_oof = 0.9999467134475708.
- Scoring code: `score_clear3d_msf_official_dev_20261106_004.py` (ca70a02552c67864199b8ae1645b7e809a60487c4711700cb7e85f0275ac1463) line 247 applies `lo_sum + (scale_oof * lo_m + bias_oof)`.
- Checkpoint binaries (.pt files, ~81 MB) are omitted from the ZIP archive to maintain lightweight packaging.

---

## 6. Corrected Challenge Protocols & External Data
1. **LUNA16 Track Names:** Track 1 is Complete Nodule Detection (NDET); Track 2 is False-Positive Reduction (FPR).
2. **Evaluator Matching Rule:** Included nodules use strictly R = d / 2 (no 5 mm floor); excluded findings fallback to R = 5.0 mm only when recorded diameter < 0.0.
3. **LNDb Specifications:** Zenodo record 7153205 v4, CC BY-NC-ND 4.0 license, 294 thoracic CTs (~30.3 GB). Candidate classification requires generating candidates with an open-source detector (baseline U-Net or LUNA16 scout).
