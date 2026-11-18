# CLEAR3D-CT Development Ablation Verification Package (Package 004)
**Package Version:** `paper-ablation-verify-package-20261107-004`
**Date:** 2026-11-07
**Scope:** Compact development-only single-fold (Fold 0) ablations on 142 LUNA16 scans (122,689 candidates, 192 true nodules).
**Governance:** Reference release HOLD maintained. Test-fold0 completely unaccessed. Candidate cap `cap4000 = NOT_ADOPTED`.

---

## 1. Directory Structure
```
paper-ablation-verify-package-20261107-004/
|-- README.md                      # This verification documentation
|-- manifest.json                  # Cryptographic SHA-256 manifest of all package files
|-- verify_package.py              # Standalone 4-step package verification runner
|-- predictions/                   # 100% aligned development prediction CSVs (122,689 rows each)
|   |-- fold0_baseline_scores.csv       # Full Fold 0 Baseline (CPM 0.904018)
|   |-- control_b_nomining_scores.csv   # Control B No-Mining (CPM 0.875000)
|   +-- control_a_fineonly_scores.csv   # Control A Fine-Only (CPM 0.865327)
|-- reference/                     # Ground truth annotations and patient mapping
|   |-- annotations.csv                 # 1,186 official LUNA16 included nodules (192 in dev)
|   |-- annotations_excluded.csv        # Excluded nodules (neither TP nor FP)
|   |-- seriesuids.csv                  # 142 development CT series UIDs
|   +-- patient_series_mapping.json     # 1:1 mapping between 142 patients and 142 series
|-- evaluator/                     # Pure Python 3 adapted LUNA16 evaluator
|   |-- noduleCADEvaluationLUNA16.py    # Official FROC CAD evaluator adapted for Python 3
|   |-- NoduleFinding.py                # Finding data structures
|   |-- tools/csvTools.py               # CSV utility helpers
|   +-- regenerate_froc_curves.py       # Standalone curve & CPM regeneration script
|-- bootstrap/                     # Patient-level paired bootstrap resampling
|   |-- run_paired_bootstrap.py         # Standalone bootstrap runner (B=2000, seed=42)
|   |-- precomputed_scan_matches.npz    # Precomputed candidate-to-nodule matches
|   +-- bootstrap_results.json          # Complete bootstrap statistics, quantiles, and CIs
|-- froc_outputs/                  # Official CADAnalysis.txt and froc_curve.txt outputs
|   |-- fold0_baseline/
|   |-- control_b_nomining/
|   +-- control_a_fineonly/
+-- provenance/                    # Provenance audit, hashes, environment, and logs
    |-- checkpoint_hashes.json          # Checkpoint byte sizes, paths, and SHA-256 digests
    |-- environment.json                # Python, PyTorch, CUDA, and hardware configuration
    |-- provenance_audit.md             # Forensic audit and uncertainty preservation
    +-- configs_and_logs/               # Training and evaluation logs for each model
```

---

## 2. Completed Ablations & Development Point Estimates
Evaluated on all 142 development scans across all 122,689 natural candidates using 4-view TTA:

| Model / Control | Parameters | VRAM Peak | Train Time | LUNA16 CPM | Sensitivities [0.125, 0.25, 0.5, 1, 2, 4, 8 FP/scan] |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fold 0 Baseline** | 4.05M (100%) | 1.34 GB | 1,598 s | **0.904018** | [0.781250, 0.864583, 0.890625, 0.911458, 0.953125, 0.963542, 0.963542] |
| **Control B (No-Mining)** | 4.05M (100%) | 1.37 GB | 1,598 s | **0.875000** | [0.677083, 0.796875, 0.869792, 0.911458, 0.932292, 0.968750, 0.968750] |
| **Control A (Fine-Only)** | 2.69M (-33.6%) | 1.02 GB | 808 s | **0.865327** | [0.671875, 0.776042, 0.864583, 0.890625, 0.932292, 0.947917, 0.973958] |

### Point Estimate Differences
- **Hard-Negative Mining Contribution** (Fold 0 vs. Control B):
  Delta CPM_1 = 0.904018 - 0.875000 = +0.029018 CPM
- **Multi-Scale Context Fusion Contribution** (Control B vs. Control A):
  Delta CPM_2 = 0.875000 - 0.865327 = +0.009673 CPM

---

## 3. Patient-Level Paired Bootstrap Resampling Analysis
- **Resampling Unit:** Patient ID (142 unique LIDC-IDRI patients, mapped 1:1 to development scans).
- **Paired Sampling:** In each replicate, the identical resampled patient cohort is evaluated across all three models simultaneously.
- **Replicates & Seed:** B = 2,000 replicates, random seed = 42.
- **Curve Interpolation:** `np.interp` piecewise linear interpolation over sorted empirical FROC operating points at [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0] FP/scan grid.

### Bootstrap Statistics & Confidence Intervals

| Comparison | Cohort Point Est. | Bootstrap Mean | Std Error | 95% Percentile CI | 90% Percentile CI | 99% Percentile CI | P(Delta > 0) | Statistically Sig. (alpha=0.05) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Fold 0 vs. No-Mining** | +0.029018 | +0.021081 | 0.016996 | **[-0.014719, +0.051129]** | [-0.009902, +0.047032] | [-0.024813, +0.060505] | 88.50% | **NO (crosses 0)** |
| **2. No-Mining vs. Fine-Only** | +0.009673 | +0.011948 | 0.010907 | **[-0.009086, +0.033859]** | [-0.005862, +0.030028] | [-0.015121, +0.041447] | 86.35% | **NO (crosses 0)** |

### Critical Scientific Limitations
1. **Exploratory Cohort:** Evaluated on a single development fold (Fold 0) across a repeatedly explored development split (142 scans).
2. **Confidence Intervals Cross Zero:** While point estimates favor the proposed mechanisms, both 95% percentile confidence intervals cross zero due to inter-patient nodule heterogeneity. Neither difference is statistically significant at alpha = 0.05.
3. **No Generalization Claim:** These results are reported honestly as exploratory development trends, NOT as proven isolated generalizable effects.

---

## 4. Verification and Execution Instructions

### A. Run Full Verification Suite
```bash
python3 verify_package.py
```
Verifies all 4 dimensions:
1. Manifest SHA-256 integrity for all package files.
2. 100% candidate key alignment across all 122,689 rows.
3. Standalone FROC regeneration directly from predictions.
4. Patient-level paired bootstrap confidence interval reproduction.

### B. Regenerate FROC Curves & CPM Directly
```bash
python3 evaluator/regenerate_froc_curves.py
```
Executes the official LUNA16 CAD evaluation logic on each prediction CSV against ground truth, reproducing exact CPMs (0.904018, 0.875000, 0.865327).

### C. Re-run Patient-Level Paired Bootstrap
```bash
# Fast re-run from precomputed matches (~10-15s)
python3 bootstrap/run_paired_bootstrap.py --b 2000 --seed 42

# Or recompute matches from scratch from raw CSVs (~80s)
python3 bootstrap/run_paired_bootstrap.py --b 2000 --seed 42 --recompute
```

---

## 5. Provenance & Preserved Uncertainty
- **Cryptographic Pre-flight Checks:** Every scoring script cryptographically verified `hash_file(checkpoint) == expected_sha` before instantiation:
  - Fold 0 Baseline Checkpoint SHA-256: 739d1d7883d66922f568be4cdc66f491bdabe8eb9e1c2c22ecb8c5bf7b8f5494
  - Control B No-Mining Checkpoint SHA-256: 941aaa0f357cf339e937d4b728a503fc60364aab543e419dfa487370a260d7ba
  - Control A Fine-Only Checkpoint SHA-256: 0907483e7e1aac661cb20063eb08cb8c543b11369b74bbcdb741d153a5f241af
- **Preserved Uncertainty:** PyTorch .pt binary archives store state dict tensors and calibration scalars, but do not internally store git commit hashes or code digests. Binding between checkpoint files and training scripts relies on external execution records (invocation.json, worker.log, summary.json).
- **External Baseline Status (MedicalNet):** MedicalNet ResNet-10/18 weights are verified and pass 823 CPU tests, but have NEVER been trained or evaluated on GPU in the CLEAR3D v2 pipeline. It is strictly a planned comparator.
- **Sensitivity Vector Reconciliation:** The authentic 5-fold ensemble sensitivity vector is Vector A ([0.760417, 0.859375, 0.916667, 0.937500, 0.963542, 0.973958, 0.979167], CPM 0.912946). Vector B from turn 2863 was an unverified text drafting error and is permanently retracted.
