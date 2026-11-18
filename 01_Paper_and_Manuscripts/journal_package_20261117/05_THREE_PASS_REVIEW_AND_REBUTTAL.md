# Deliverable 5 & Three-Pass Independent Peer Review Report + Rebuttal Matrix

**Review Team:** Agent 5 (Statistical & Experimental Validation Specialist), Agent 7 (Senior Academic Editor), Agent 8 (Independent Peer Reviewer), Reconciled by Agent 1 (Principal Research Scientist)
**Review Date:** 2026-11-17

---

## 1. Summary of Completed Three-Pass Quality Audit

| Pass | Reviewer Focus | Defects / Risks Identified | Corrective Action Executed in Final Files | Verification Status |
|---|---|---|---|---|
| **Pass 1** | Scientific, Epistemic & Statistical Audit | (a) Legacy metrics_summary.json MedicalNet raw32 bug (0.00284 AP / 0.0030 CPM) vs. corrected 48^3 valid-mask (0.1648 AP / 0.0975 CPM); (b) Retracted uncontrolled 0.0424 -> 0.9043; (c) Clarified randomly initialized CLEAR3D-MSFv2 (3 scale tokens) vs. unexecuted MedicalNet-Transformer concept; (d) arm_e_mixed_focal scaffold cloning caveat & Arm E composite multi-factor attribution; (e) Caught uncited refs [13], [21], [22] & corrected Nat. Med. 26(9) vs 26(10). | Documented exact Table II forensic reconciliations; added citations [13], [21], [22]; enforced bijective [1]..[42] verification in manuscript_data.py and verify_package.py; framed Arm E accurately as composite recipe. | **PASS** (42/42 refs synced; all 13 benchmark strings verified in both .docx). |
| **Pass 2** | Editorial, Double-Blind Anonymity & Template Compliance | (a) Initial abstracts exceeded bounds (282 JIIM, 252 JOIV); (b) riza substring false-positive vs. genuine author-leak verification; (c) Unused sample template media (.wmf/.bin/image2/3/5) bloated JOIV zip to 2.36 MB; (d) fig1 superscripts & fig4 legend clearance over Fold 6–9 bars. | Tightened abstracts to 234 words (JIIM) & 235 words (JOIV); added word-boundary anonymity guard (0 leaks in JIIM); stripped unused sample assets while retaining CC-BY-SA badge; fixed fig1 Unicode superscripts/arrows & raised fig4 ylim(0, 1.22) with clean upper legend; upgraded JOIV equations with Unicode/subscripts (no images). | **PASS** (0 JIIM leaks; 0 JOIV hdrs/ftrs/pgnums/hyperlinks; JOIV 235w abs, 6,075 body words). |
| **Pass 3** | Hostile Reviewer Stress-Test & Pre-Submission Defense | Identified 7 high-probability Major Revision objections covering 10-fold Arm E scaling, OFAT ablation, internal-vs-external HOLD semantics, LUNGx LoG ceiling, MedicalNet low CPM, balanced vs. natural F1, and Subset 1 quarantine. | Embedded preemptive disclosures directly into Discussion Sections 5.1–5.2 (III-F–G in JOIV) + built complete Point-by-Point Rebuttal Strategy below. | **READY FOR SUBMISSION** (pending 4 administrative author checks listed in Section 3). |

---

## 2. Simulated Peer-Review Objection Matrix & Point-by-Point Rebuttal Plan

### Critique 1 (Major): "Your improved 10-fold CV on dev708 (0.8680 CPM) uses the 22-epoch Round-1 recipe with 4-view TTA, whereas your headline HOLD Subset 0 result (0.8750 CPM, 75.54% F1+) uses a different composite Arm E recipe (mixed R1+R2, gamma=1.5, ls=0.005, 26 epochs, 8v TTA) trained only on Fold 0. Why didn't you evaluate Arm E across all 10 folds?"
- **Vulnerability Rating:** High (likely asked by Associate Editor / Reviewer #2).
- **Preemptive Manuscript Fix:** Explicitly stated as Limitation #1 and Future Work #1; Fold 0 Arm E vs. Fold 0 22-ep Round-1 Control and Arm C/D comparisons are cleanly separated in Tables III and IV rather than blurred.
- **Rebuttal Strategy:** Acknowledge transparently that full 10-fold scaling of Arm E (~21,000s/run per variant on shared RTX 2080 Ti hardware) was constrained by compute budget after completing the 708-scan 10-fold Scout-mining study. Emphasize that (i) prespecified 10-fold dev708 rigorously proves the core OOF hard-negative mining mechanism across all 708 scans (10/10 folds improved, +0.1859 pooled CPM), and (ii) Fold 0 -> Inner Fold 1 -> frozen HOLD Subset 0 tests Arm E under an even stricter single frozen model + fixed external-to-training threshold transfer protocol than 10-fold OOF averaging. Commit to releasing 10-fold Arm E + OFAT scripts or executing during Major Revision if requested.

### Critique 2 (Major): "Arm E changes four factors simultaneously relative to Arm C (6k+6k mixed mining, softened focal gamma=1.5, label smoothing 0.005, 26 epochs) plus 8-view TTA, so you cannot isolate which factor drove the HOLD Subset 0 precision gain."
- **Vulnerability Rating:** High.
- **Preemptive Manuscript Fix:** Discussion Section 5.2 explicitly discloses that Arm E was evaluated as a frozen composite configuration, while isolated causal factors already exist in Table IV: Loss (Arm A CE 0.8937 vs. Arm C Focal 0.9063), Mining Round (Arm C R1 0.9063 / F1 67.57% vs. Arm D R2 0.8905 / F1 69.77%), and TTA (4v vs. 8v isolated across Arms C, D, and E showing consistent PR-AUC gains +0.0084..+0.0165).
- **Rebuttal Strategy:** Present the clean two-tier ablation hierarchy and offer to run one-factor-at-a-time (OFAT) sweeps on Fold 0 if reviewers require individual coefficient isolation during revision.

### Critique 3 (Major): "Early runs in your workspace touched LUNA16 Subsets 0–9 overall. Doesn't that compromise the claim that HOLD Subset 0 is an external or strictly unseen test set?"
- **Vulnerability Rating:** Critical for epistemic honesty.
- **Preemptive Manuscript Fix:** Never claim HOLD Subset 0 is external multi-center validation. Explicitly define HOLD Subset 0 as an untouched internal held-out cohort (89 scans, 89 patients, 0 dev patient overlap, zero post-freeze tuning of Arm E weights/calibration/cutoffs), and explicitly report archival exposure history alongside exploratory LUNGx external evaluation.
- **Rebuttal Strategy:** Point reviewers to Sections 3.1, 5.2, and Table I, noting compliance with CLAIM 2024 [33] transparency standards and clarifying that genuine multi-center proof requires prospective external testing with learned Stage-1 detection (e.g., LNDb).

### Critique 4 (Moderate): "Why is SPIE-AAPM LUNGx CPM only 0.4279–0.4340? Doesn't that prove your classifier fails to generalize externally?"
- **Vulnerability Rating:** Moderate-High.
- **Preemptive Manuscript Fix:** Table VI and Section 4.5 (III-E in JOIV) dissect Stage-1 LoG coverage (47/70 = 67.14%) separately from conditional Stage-2 classification (47/47 = 100% covered recall at cutoff 0.5, 39/47 = 83.0% at strict nat_cut).
- **Rebuttal Strategy:** Highlight that identical root cause was independently observed in Phase 2 Track 2 (64/121 = 52.9% SpotNet coverage). No second-stage classifier can score missed Stage-1 coordinates; hence Stage-2 generalization is intact while Stage-1 LoG must be replaced by a learned 3D proposal generator in future work.

### Critique 5 (Moderate): "MedicalNet ResNet-18 achieves only 0.0975 CPM (0.1648 AP), which seems unusually low—was the comparator handicapped?"
- **Vulnerability Rating:** Moderate.
- **Preemptive Manuscript Fix:** Section 4.1 & Table II transparently distinguish the earlier buggy raw32 evaluation (0.00284 AP) from the audited 48^3 valid-mask z-score comparator (0.0975 CPM), and explicitly retract the uncontrolled 0.0424 -> 0.9043 narrative claim.
- **Rebuttal Strategy:** Explain that generic multi-organ segmentation ResNet-18 weights pre-trained across coarse 8-dataset segmentation tasks lack 1 mm dual-resolution center-focus inductive bias and struggle under ~463:1 unfiltered full-candidate ranking unless re-engineered with multi-scale candidate pooling and OOF hard-negative mining.

### Critique 6 (Minor-Moderate): "Your headline positive F1 (75.54%) is lower than 94–97% F1 reported by Almahasneh et al. (2025) and Wang et al. (2026)."
- **Vulnerability Rating:** Moderate.
- **Preemptive Manuscript Fix:** Table VII & Section 5.1 contrast prevalence directly: Almahasneh (1:3) and Wang (1:1) compute candidate F1 on artificially balanced subsets, whereas HOLD Subset 0 computes Pos-F1 over all 79,135 unfiltered candidates (572:1). Indeed, on dev-142 1:3 balanced subset (dev_1060), CLEAR3D-CT reaches 97.95% Pos-F1.
- **Rebuttal Strategy:** Cite Saito & Rehmsmeier [12] and Mosquera et al. [32] demonstrating that natural-prevalence PR-AUC (0.8226) and natural Pos-F1 (75.54%) measure true clinical screening precision without sampling distortion.

### Critique 7 (Minor): "What is the administrative provenance of quarantined Subset 1 / confirm_83, and why are annotations_excluded candidates left in natural classification denominators?"
- **Vulnerability Rating:** Minor.
- **Preemptive Manuscript Fix:** Sections 3.1, 5.2, and Table I clearly disclose confirm_83 unsettled quarantine (entire Subset 1 excluded from all controlled post-audit training/testing), while showing in Table II that excluded-filtering lifts Control A AP (0.5882 -> 0.8363) and Control B AP (0.6055 -> 0.8609). Leaving raw candidates_V2 unfiltered preserves conservative lower-bound precision estimates.

---

## 3. Remaining Human-Author Pre-Submission Action Items

Before uploading to **Springer Nature Editorial Manager (JIIM)** or submitting for coursework (**JOIV**):
1. Fill [INSERT ORCID iD] in JIIM_Title_Page_and_Declarations.docx.
2. Confirm final CRediT author role assignments and corresponding author preference (Mohammad Fawad Karimi vs. Kahlil Muchtar) across Title Page / Cover Letter / JOIV header.
3. Attach the archived institutional ethics exemption confirmation reference/date for public retrospective TCIA/LUNA16/LUNGx computational analysis.
4. Keep CLEAR3D_CT_JIIM_DoubleBlind_Manuscript.docx and JIIM_Title_Page_and_Declarations.docx strictly separate during JIIM double-blind upload.
