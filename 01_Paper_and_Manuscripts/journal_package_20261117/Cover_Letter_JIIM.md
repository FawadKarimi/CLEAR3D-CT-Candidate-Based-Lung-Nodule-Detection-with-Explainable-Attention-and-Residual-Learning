# Cover Letter — Journal of Imaging Informatics in Medicine (JIIM)

**Date:** November 17, 2026
**To:** Editor-in-Chief, *Journal of Imaging Informatics in Medicine* (Springer Nature / Society for Imaging Informatics in Medicine)
**Manuscript Title:** CLEAR3D-CT: Multi-Scale Volumetric Candidate Classification with Hard-Negative Mining and Nested Platt Calibration for Pulmonary Nodule False-Positive Reduction
**Article Type:** Original Research

Dear Editor-in-Chief and Editorial Board,

We respectfully submit our original research manuscript entitled *"CLEAR3D-CT: Multi-Scale Volumetric Candidate Classification with Hard-Negative Mining and Nested Platt Calibration for Pulmonary Nodule False-Positive Reduction"* for consideration for publication in the *Journal of Imaging Informatics in Medicine (JIIM)* under the subscription-based publishing model (zero article processing charge).

Second-stage false-positive reduction (FPR) in thoracic CT computer-aided detection remains hindered by extreme candidate-level imbalance (~571 non-nodules per true positive in official LUNA16 `candidates_V2`), cross-fold mining leakage, and catastrophic probability-threshold shift when transferring models across training objectives. Addressing JIIM's core focus on reproducible clinical imaging informatics and transparent validation, our manuscript delivers three contributions grounded in forensic artifact auditing:

1. **Leak-Free Patient-Disjoint 10-Fold Validation (`dev708`):** Quarantining duplicate-volume LUNA16 Subset 1 and withholding Subset 0 completely, fold-isolated out-of-fold hard-negative mining improved **10 of 10 folds**, elevating pooled CPM from `0.6821` to `0.8680` (`+0.1859`) and natural-prevalence PR-AUC from `0.4439` to `0.7240` across 708 scans (603,652 candidates).
2. **Quantifying & Solving Cross-Objective Calibration Shift:** Controlled split-RNG Fold 0 ablations prove that transferring uniform-negative thresholds naively onto focal-trained models causes total operational collapse (`0.0%` positive F1), whereas nested Inner Fold 1 Platt calibration recovers high natural precision (`70.35%` positive F1 at calibrated `0.5`, PR-AUC `0.7779`).
3. **Frozen Single-Model Generalization & Stage-1 Bottleneck Proof:** Evaluated frozen once on untouched `HOLD Subset 0` (89 scans, 79,135 candidates), single-model Arm E (8-view TTA) attained **0.9993 ROC-AUC, 0.8226 PR-AUC, 0.8750 CPM (100% sensitivity at 8 FPs/scan)**, and **75.54% natural-prevalence positive F1**—surpassing an archived five-model ensemble by `+8.11` points in positive F1 and `+19.34` points in precision while using one-fifth of the inference checkpoints. External exploratory testing on unexposed SPIE-AAPM LUNGx (58 scans, 70 targets) further establishes that cross-cohort recall ceilings (`67.14%`) arise in Stage-1 Laplacian-of-Gaussian proposal coverage rather than Stage-2 classification (`47/47` covered recall).

In full compliance with JIIM's **double-blind peer-review policy**, all author names, institutional affiliations (Universitas Syiah Kuala), scholarship acknowledgments (KNB), and repository usernames have been removed from the manuscript file and placed in the separate Title Page & Declarations document. We confirm this manuscript is original, has not been published previously, and is not under consideration elsewhere. We also acknowledge recent relevant JIIM scholarship (e.g., Ma et al., TiCNet, JIIM 2024) within our related work and benchmark comparison.

Thank you for your time and editorial consideration.

Sincerely,

**Mohammad Fawad Karimi** *(Corresponding Author)*
Department of Informatics, Faculty of Mathematics and Natural Sciences
Universitas Syiah Kuala, Banda Aceh 23111, Indonesia
Email: mfkarimi@mhs.usk.ac.id
On behalf of co-authors **Kahlil Muchtar** and **Hammam Riza**
