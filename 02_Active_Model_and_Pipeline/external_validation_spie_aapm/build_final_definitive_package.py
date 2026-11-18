import hashlib, json, os, sys, zipfile
from pathlib import Path

BASE = Path("/experiment-storage/chaidir/fawad-dsai")
ROOT = BASE / "CLEAR3D_CT project/02_Active_Model_and_Pipeline"
EXT_DIR = ROOT / "external_validation_spie_aapm"
ZIP_OUT = ROOT / "clear3d_definitive_evidence_package_20261109.zip"
MANIFEST_OUT = ROOT / "clear3d_definitive_evidence_manifest_20261109.json"

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()

files_to_pack = [
    # System Binding & Model Architecture
    (ROOT / "primary_reference_system_binding.json", "primary_reference_system_binding.json"),
    (ROOT / "clear3d_msf_model_20261106_002.py", "clear3d_msf_model_20261106_002.py"),
    (ROOT / "train_clear3d_msf_ensemble_20261106_004.py", "train_clear3d_msf_ensemble_20261106_004.py"),
    
    # Cohorts, Exposure Reconciliation & Benchmarks
    (EXT_DIR / "cohort_exposure_reconciliation.json", "cohort_exposure_reconciliation.json"),
    (EXT_DIR / "published_benchmarks_comparison.json", "published_benchmarks_comparison.json"),
    (ROOT / "reconcile_audit_and_exposure_20261107.json", "reconcile_audit_and_exposure_20261107.json"),
    
    # Internal Locked Test Verification
    (ROOT / "runs/clear3d-msf-official-test-fold0-001/final_test_report.json", "internal_test_report_fold0.json"),
    (ROOT / "runs/clear3d-msf-official-test-fold0-001/patient_clustered_bootstrap_ci.json", "internal_patient_clustered_bootstrap_ci.json"),
    (ROOT / "runs/clear3d-msf-official-test-fold0-001/reproduce_metrics.py", "internal_reproduce_metrics.py"),
    
    # External Evaluation Protocol & Portable Geometry
    (EXT_DIR / "frozen_evaluation_protocol_cap4000.json", "frozen_evaluation_protocol_cap4000.json"),
    (EXT_DIR / "test_ground_truth.json", "test_ground_truth.json"),
    (EXT_DIR / "portable_target_geometry.csv", "portable_target_geometry.csv"),
    (EXT_DIR / "portable_target_geometry.json", "portable_target_geometry.json"),
    
    # External Results & Summaries
    (EXT_DIR / "test_cohort_cap4000_corrected_evaluation_summary.json", "test_cohort_cap4000_corrected_evaluation_summary.json"),
    (EXT_DIR / "test_cohort_cap4000_evaluation_summary.json", "test_cohort_cap4000_evaluation_summary_original_preserved.json"),
    (EXT_DIR / "froc_test_cohort_cap4000_corrected.png", "froc_test_cohort_cap4000_corrected.png"),
    (EXT_DIR / "froc_test_cohort_cap4000.png", "froc_test_cohort_cap4000_original.png"),
    (EXT_DIR / "environment_details.json", "environment_details.json"),
    (EXT_DIR / "eval_cap4000.log", "eval_cap4000.log"),
    
    # Audit, Test & Reproduction Code
    (EXT_DIR / "reproduce_entire_evidence_package.py", "reproduce_entire_evidence_package.py"),
    (EXT_DIR / "test_patient_clustered_bootstrap_fix.py", "test_patient_clustered_bootstrap_fix.py"),
    (EXT_DIR / "test_froc_unsubsampled_validation.py", "test_froc_unsubsampled_validation.py"),
    (EXT_DIR / "audit_dicom_geometry.py", "audit_dicom_geometry.py"),
    (EXT_DIR / "audit_slice_ordering.py", "audit_slice_ordering.py"),
    (EXT_DIR / "compare_matching_rules.py", "compare_matching_rules.py"),
    (EXT_DIR / "recompute_and_compare_all_metrics.py", "recompute_and_compare_all_metrics.py"),
    (EXT_DIR / "export_deidentified_geometry_table.py", "export_deidentified_geometry_table.py"),
    (EXT_DIR / "generate_corrected_froc_plots.py", "generate_corrected_froc_plots.py")
]

manifest = {
    "package_name": "clear3d_definitive_evidence_package_20261109.zip",
    "status": "DEFINITIVE_MANUSCRIPT_EVIDENCE_PACKAGE",
    "timestamp": "2026-11-09T12:26:00Z",
    "reproduction_command": "python3 reproduce_entire_evidence_package.py",
    "governance": {
        "reference_release": "HOLD",
        "confirmation_cohort_83pts": "RESTRICTED_UNTOUCHED",
        "internal_test_fold0": "FROZEN_NO_TUNING",
        "medicalnet_baseline_cpm_claim": "RETRACTED_UNCONTROLLED_MULTIFACTOR",
        "manuscript_writing": "DEFERRED"
    },
    "key_results_summary": {
        "internal_test_fold0": {
            "cohort": "LUNA16 Subset 0 (89 scans, 138 nodules, 79,135 candidates)",
            "froc_cpm": 0.9043,
            "froc_cpm_95ci": [0.8452, 0.9533],
            "sens_nat_cut": 0.8551,
            "precision_nat_cut": 0.5566,
            "specificity_nat_cut": 0.9988
        },
        "external_independent_test": {
            "cohort": "SPIE-AAPM LUNGx (58 unexposed scans, 70 nodules, 232,000 candidates)",
            "candidate_cap": 4000,
            "matching_rule": "Fixed 5.0 mm Euclidean radius",
            "proposal_coverage": 0.6714,
            "proposal_coverage_95ci": [0.5538, 0.7838],
            "rule_A_closest_cpm": 0.4279,
            "rule_A_closest_cpm_95ci": [0.3469, 0.5193],
            "rule_B_highest_scoring_cpm": 0.4340,
            "rule_B_highest_scoring_cpm_95ci": [0.3531, 0.5216],
            "sens_nat_cut": 0.5571,
            "fp_per_scan_nat_cut": 2.655,
            "sens_half_cut_ruleB": 0.6714,
            "covered_nodule_retrieval_at_half_cut": 1.000,
            "fp_per_scan_half_cut": 17.224
        }
    },
    "files": {}
}

print(f"Creating definitive evidence package {ZIP_OUT}...")
with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for src_path, arcname in files_to_pack:
        if src_path.exists():
            h = sha256_file(src_path)
            sz = src_path.stat().st_size
            manifest["files"][arcname] = {"sha256": h, "size_bytes": sz}
            zf.write(src_path, arcname=arcname)
            print(f"  Added {arcname} ({sz/1024:.1f} KB, sha256: {h[:12]}...)")
        else:
            print(f"  ERROR: {src_path} NOT FOUND!")

zip_hash = sha256_file(ZIP_OUT)
manifest["package_sha256"] = zip_hash
manifest["package_size_bytes"] = ZIP_OUT.stat().st_size
manifest["total_files"] = len(manifest["files"])

with open(MANIFEST_OUT, "w") as f:
    json.dump(manifest, f, indent=2)

# Also copy manifest into EXT_DIR
with open(EXT_DIR / "clear3d_definitive_evidence_manifest_20261109.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nPackage created successfully: {ZIP_OUT}")
print(f"SHA-256: {zip_hash}")
print(f"Total files: {manifest['total_files']} (Size: {manifest['package_size_bytes']/1048576:.2f} MB)")
print(f"Manifest written to: {MANIFEST_OUT}")
