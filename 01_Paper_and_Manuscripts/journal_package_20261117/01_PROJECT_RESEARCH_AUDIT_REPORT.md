# Deliverable 1: CLEAR3D-CT Comprehensive Project Research Audit Report

**Prepared by:** Agent 1 (Principal Research Scientist), Agent 2 (Medical Imaging & 3D CT Specialist), Agent 3 (Deep Learning & AI Specialist), Agent 4 (Scientific Literature Review Specialist), and Agent 5 (Statistical & Experimental Validation Specialist)
**Audit Date:** 2026-11-17
**Workspace Audited:** `/experiment-storage/chaidir/fawad-dsai/CLEAR3D-broader-loader-20261028/v2` and preceding project directories (`CLEAR3D-parallel-review-20261014-2f1a64`, `CLEAR3D-comparison-protocol-20261025`)

---

## 1. Executive Audit Verdict & Epistemic Separation

A forensic audit across source code, training scripts, cohort manifests, checkpoint hashes, evaluation logs, and raw prediction artifacts reveals **four distinct codebases** and **five chronological experimental phases**. Conflating these phases—specifically attributing an unexecuted MedicalNet-initialized voxel-wise 3D Transformer proposal to experiments that actually trained scratch or randomly initialized networks—would constitute a fatal scientific integrity error.

| Epistemic Category | Verified Project Reality |
|---|---|
| **1. Originally Proposed** | Phase 1 design doc (`A-model/MODEL-PATH-AUDIT.md`) envisioned pretrained MedicalNet (`resnet_18`/`resnet_10`) on `48^3` CT patches projected to per-scale voxel-token `ScaleTransformer` modules (`1728 / 216 / 27` tokens, 2 pre-LN blocks, 4 heads, FF=256). |
| **2. Actually Implemented** | **(a) Phase 1 (`lidc3d/models.py`):** Scratch `Encoder(width=16)` + `ScaleTransformer`. **(b) Phase 2 (`isolated/phase_cf.py`):** `ClearNet`, a scratch-initialized two-branch 3D CNN (`16^3` fine + `32^3` context, 16 ch, zero MedicalNet weights, zero SE, zero Transformer), alongside adapted `ZuoNet` and `WangNet`. **(c) Phase 3–5 (`clear3d_msf_model_20261106_002.py`):** `CLEAR3DMSFv2(in_channels=3, dim=160, dropout=0.25)`—a **randomly initialized** (zero pretrained weights) three-branch 3D `SEResBlock3D` + `GroupNorm` network on `21^3` focal, `33^3 @ 1mm` fine, and `25^3 @ 2mm` context patches fused by **4-head `nn.MultiheadAttention` over 3 scale tokens** (token-level, not voxel-wise). **(d) Standalone Comparator:** Loaded pretrained `MedicalNet ResNet-18` (`resnet_18.pth`, SHA-256 `38b3a174...`, `48^3` valid-mask normalization, 33.16M params). |
| **3. Experimentally Evaluated** | (i) Phase 2 83-scan Track 1/Track 2 comparison; (ii) Historical Run 004 5-model ensemble (142-scan dev, 89-scan Subset 0, 58-scan LUNGx); (iii) Archived 797-scan 10-fold CV (12 epochs, 4k negatives/fold); (iv) 142-scan development comparators (Run 004, HighRes, Corrected MedicalNet, Fold 0 Fine-Only, No-Mining, Hard-Prior, Curriculum); (v) Controlled Prespecified `dev708` 10-fold CV (`708` scans, `603,652` cands, strictly excluding Subset 1 duplicates and HOLD Subset 0); (vi) Controlled Split-RNG Fold 0 ablations (Arms A, C, D, E) with matched Inner Fold 1 nested Platt calibration and frozen one-shot test evaluation on untouched `HOLD Subset 0` (`89` scans, `79,135` candidates, zero dev patient overlap). |
| **4. Empirically Demonstrated** | **(a)** Controlled prespecified mining (`16k` rand + `12k` OOF Scout hard negatives) + `22`-epoch cosine training + 4v TTA improved 10-fold patient-disjoint `dev708` pooled CPM from `0.6821` to `0.8680` (`+0.1859`) and fold mean from `0.7224 +/- 0.1246` to `0.8676 +/- 0.0359` (`10/10` folds improved; PR-AUC `0.4439 -> 0.7240`). **(b)** On Fold 0, Round-1 focal loss (`gamma=2.0`, Arm C) reached `0.9063` CPM vs. `0.8937` CE (Arm A); naive cross-objective threshold transfer failed completely (`0 TP, 0 FP, 0.0% F1+`). **(c)** Single-model Arm E (mixed `6k R1 + 6k R2` hard negatives, softened focal `gamma=1.5`, `ls=0.005`, `26` epochs, 8v TTA) reached Fold 0 `0.9016` CPM / `0.9990` ROC-AUC / `0.7779` PR-AUC / `70.35%` calibrated-0.5 F1+, and generalized frozen to untouched `HOLD Subset 0` with `0.8750` CPM (`95% CI [0.8036, 0.9340]`, `100%` sensitivity at `8 FPs/scan`), `0.9993` ROC-AUC, `0.8226` PR-AUC, and `75.54%` natural-distribution positive F1 (`+8.11` pp F1+ and `+19.34` pp precision over the archived 5-model ensemble on the same 89 scans). **(d)** External LUNGx generalization is bottlenecked by LoG Stage-1 proposal recall (`47/70 = 67.14%`), not Stage-2 classification (`47/47 = 100%` covered recall at cutoff `0.5`). |
| **5. Remaining Future Work** | Scaling the Arm E mixed-mining + softened-focal recipe across all 10 `dev708` folds; one-factor-at-a-time (OFAT) decoupling of mixed mining vs. `gamma=1.5` vs. `ls=0.005` vs. 26 epochs; replacing handcrafted LoG with a frozen learned 3D Stage-1 detector; prospective multi-center external clinical validation with complete exhaustive annotations; institutional IRB exemption confirmation; resolving archived Subset 1 / `confirm_83` administrative provenance. |

---

## 2. Verification of Provisional Source-of-Truth Metrics & Root Causes

All provisional source-of-truth values specified in the audit mandate were verified byte-for-byte against original logs and JSON outputs in `CLEAR3D-comparison-protocol-20261025`:

1. **Zuo-style baseline vs. ClearNet (Phase 2 Track 1 Confirmation, 83 scans, 66,135 cands = 161 pos + 65,974 neg, 121 nodules):**
   - `ZuoNet`: **Positive F1 = `0.4878`**, **ROC-AUC = `0.9751`**, PR-AUC = `0.5211`, Recall = `0.6832` (`110/161`), Precision = `0.3793` (`110/290`), Specificity = `0.9973`, Accuracy = `0.9965`, ECE = `0.1543`, Brier = `0.060796`, `th=0.9633` (`TP=110, FP=180, FN=51, TN=65,794`). *Code departure (`isolated/phase_cf.py`):* Uses three depth branches with global average pooling instead of Zuo et al. (2020) pool strides, with conv bias initialized to `0.2`.
   - `WangNet`: Positive F1 = `0.2776`, ROC-AUC = `0.8655` (cubic `32/16/8` views instead of anisotropic `32x48x48`).
   - `ClearNet(fuse=True)`: **Positive F1 = `0.1697`**, **ROC-AUC = `0.8143`**, PR-AUC = `0.1772`, Recall = `0.2298` (`37/161`), Precision = `0.1345` (`37/275`), Specificity = `0.9964`, Accuracy = `0.9945`, ECE = `0.3618`, Brier = `0.153946`, `th=0.8378` (`TP=37, FP=238, FN=124, TN=65,736`). Size recall: `0/39` (<6 mm), `2/50` (6–10 mm), `35/72` (>10 mm). Scratch two-branch CNN lacking MedicalNet weights, SE blocks, focal focus channels, and cross-scale attention.

2. **Track 2 Confirmation & Candidate Coverage Bottleneck (`isolated/track2.py`, 83 scans, 121 nodules, 16,600 SpotNet marks capped at 200/scan):**
   - **Stage-1 Coverage: `64 of 121 nodules (52.9% / 0.5289)`**, missing `57` nodules (`35/40 <6 mm`, `19/50 6–10 mm`, `3/31 >10 mm`).
   - **CPMs:** `Zuo-style CPM = 0.3967` (`0.396694`), `Wang-style CPM = 0.2987`, **`CLEAR3D rescore CPM = 0.1606`** (`0.160567`), `ablation = 0.1771`, `spot = 0.0885`.

3. **Why Previous Development Optimization Did Not Improve F1 (`DEV_REVIEW.md`):**
   - Evaluated on `142` dev scans (`122,689` cands) on weak scratch `ClearNet` under `NEG_CAP=7`: stored `clear` F1 (`0.1706`) exceeded naive `clear_aug` (`0.1456`, paired diff `-0.0235`, 95% CI `[-0.0466, +0.0012]`), `clear_focal` (`0.1274`), and `clear_lr` (`0.1342`). Root causes: extreme negative under-sampling, absence of dual-resolution 1mm/2mm local-contrast channels, zero hard-negative mining, and short `8`-epoch runs.

---

## 3. Inter-Agent Disagreements & Evidence-Based Reconciliations

- **Disagreement A (MedicalNet Comparator Values — Agent 3 vs. Agent 5):** `metrics_summary.json` had `AP = 0.00284, ROC-AUC = 0.5861` (buggy mismatched `raw32` crops), whereas corrected `evaluate_and_audit_all_three_models_20261109.py` / `metrics_summary_audit.json` (`48^3` valid-mask normalization) yields MedicalNet ResNet-18 **`CPM = 0.0975` (`[0.0631, 0.1404]`), `ROC-AUC = 0.7765`, `average_precision = 0.1648` (`PR-AUC_trap = 0.1678`)** vs. Run 004 **`CPM = 0.9129`, `average_precision = 0.7974` (`PR-AUC_trap = 0.7995`), `ROC-AUC = 0.9968`**. The earlier uncontrolled comparison claiming MedicalNet `0.0424 -> 0.9043` is formally **retracted** (6 confounded factors).
- **Disagreement B (Fit-OOF Leakage vs. Nested Inner Calibration — Agent 5 vs. Agent 1):** `reconcile_audit_and_exposure_20261107.json` flagged Archived Run 004 (`scale=1.70, bias=4.175479, nat_cut=0.9999467`) as `FAILED` on strict fold independence due to global fit-pool Scout mining. The Nov 14–16 controlled studies fix this via strictly per-fold OOF Scouts (`dev708`) and Inner Fold 1 nested Platt fits (`exp_fold0_*`), never leaking evaluation or `HOLD Subset 0` data.
- **Disagreement C (`797`-Scan CV vs. Controlled `dev708` 10-Fold CV vs. Archived `89`-Scan Ensemble — Agent 2 vs. Agent 5):** Archived `797`-scan CV (`0.6768` pooled / `0.7235` fold mean, `12` epochs, `4k` negatives, `100`-mark evaluator cap) is superseded on methodology by **prespecified `dev708` 10-fold CV** (`708` scans, `707` pts, `603,652` cands, zero Subset 1 or HOLD Subset 0 overlap): Baseline `0.6821` (fold mean `0.7224 +/- 0.1246`) -> Improved **`0.8680` (fold mean `0.8676 +/- 0.0359`, `10/10` folds improved, `+0.1859` pooled CPM)**.
- **Disagreement D (Arm E Multi-Factor Attribution — Agent 3 vs. Agent 8):** Arms A (`0.8937`), C (`0.9063`), and D (`0.8905`) provide clean single-factor split-RNG ablations, and `4v vs. 8v` TTA is isolated directly per arm; Arm E (`6k R1 + 6k R2`, `gamma=1.5`, `ls=0.005`, `26` ep, 8v TTA) is accurately reported as a composite frozen configuration rather than single-factor causality.
- **Disagreement E (Natural `candidates_V2` Prevalence Ceiling — Agent 5):** Raw `candidates_V2` leaves `annotations_excluded.csv` positives labeled `0`; no label filtering or class balancing is used in primary metrics. Natural `75.54%` Pos-F1 and `0.8226` PR-AUC on `HOLD Subset 0` (`89` scans, `79,135` cands) are contrasted transparently against literature `1:1`/`1:3` balanced numbers.

---

## 4. Audited Quantitative Summary Across Experiments

- **`dev708` 10-Fold Patient-Disjoint CV (`708` scans, `603,652` cands, `1,249` pos, `946` ref nodules):** Pooled CPM `0.6821` -> **`0.8680`** (`[0.8388, 0.8967]`); Fold mean `0.7224 +/- 0.1246` -> **`0.8676 +/- 0.0359`** (`[0.8471, 0.8896]`); Paired fold delta `+0.1452` (`[+0.0845, +0.2122]`, `10/10` folds improved); ROC-AUC `0.9958 -> 0.9979`; PR-AUC `0.4439 -> 0.7240`; `F1+@0.5` `21.69% -> 36.67%`.
- **Fold 0 Split-RNG Single-Model Ablation (`70` scans, `62,942` cands, `110` pos) & Inner Fold 1 (`75` scans, `64,386` cands, `118` pos):** Arm A CE `0.8937` (`tau*` F1+ `66.09%`); Arm C R1 Focal (`gamma=2.0`, 4v) **`0.9063`** (`tau*` F1+ `67.57%`; uniform xfer `0.0%`; 8v `0.9048`, PR-AUC `0.7460`); Arm D R2 Focal (`4v`) `0.8905` (`tau*` Prec `71.43%`, F1+ `69.77%`; 8v `0.8857`, PR-AUC `0.7387`); Arm E Mixed R1+R2 (`gamma=1.5, ls=0.005, 26ep`, 8v) **`0.9016`** (`[0.8070, 0.9725]`), **ROC-AUC `0.9990`**, **PR-AUC `0.7779`**, **`cal@0.5` F1+ `70.35%`** (`T_0=0.619432, b_0=2.757011, tau*=0.917052`).
- **Frozen Untouched `HOLD Subset 0` (`89` scans, `89` pts, `79,135` cands = `138` pos + `78,997` neg, `0` dev overlap):** Frozen Arm E 8v TTA: **`CPM 0.8750`** (`[0.8036, 0.9340]`, `100.0%` at 8 FPs/scan), **ROC-AUC `0.9993`**, **PR-AUC `0.8226`**, frozen `tau*`: `105 TP / 35 FP / 78,962 TN / 33 FN` (**Sens `76.09%`, Prec `75.00%`, Spec `99.96%`, Acc `99.91%`, Pos-F1 `75.54%`, Macro-F1 `87.75%`**); vs. Archived 5-model ensemble on the same 89 scans (`CPM 0.9043`, `118/94/78903/20`, Prec `55.66%`, Rec `85.51%`, Pos-F1 `67.43%`).
- **Exploratory External SPIE-AAPM LUNGx (`58` unexposed scans, `70` targets, LoG cap `4000`):** Stage-1 coverage `47/70 (67.14%)`, Rule A closest CPM `0.4279` (`[0.3469, 0.5193]`), Rule B highest-score CPM `0.4340`; `100%` covered recall (`47/47`) at threshold `0.5` under Rule B.

---

## 5. Transparent Disclosure of Missing Items / Unsettled Records
1. `confirm_83` / LUNA16 Subset 1 authorization status remains `UNSETTLED`; strictly excluded from all controlled experiments (`dev708` and Fold 0–HOLD evaluations).
2. While `HOLD Subset 0` has verified zero patient overlap with the controlled `708`-scan development corpus, earlier exploratory project archives touched all LUNA16 subsets overall (disclosed candidly as internal held-out subset testing rather than unexposed multi-center external clinical proof).
3. `dev708` positive extraction reads pre-built `positives_cache.npz` (`1,249` dev positive patches filtered by `dev_scan_map`).
4. Secondary computational analysis IRB exemption confirmation and clean post-publication code release commit remain administrative pre-submission items.