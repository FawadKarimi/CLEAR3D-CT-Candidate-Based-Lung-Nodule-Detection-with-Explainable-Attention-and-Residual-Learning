import hashlib, json, os, sys, zipfile
from pathlib import Path

EXT_DIR = Path("/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/02_Active_Model_and_Pipeline/external_validation_spie_aapm")
ZIP_OUT = EXT_DIR / "clear3d-spie-aapm-test-cohort-cap4000-corrected-review-package-20261109.zip"
MANIFEST_OUT = EXT_DIR / "corrected_package_manifest_20261109.json"

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()

files_to_pack = [
    # Protocols and Ground Truth
    "frozen_evaluation_protocol_cap4000.json",
    "test_ground_truth.json",
    "tcia_series_list.json",
    "download_status.json",
    # Evaluation Scripts & Predictions
    "batch_download_test_scans.py",
    "evaluate_test_cohort_cap4000.py",
    "scores_cap4000.csv",
    "eval_cap4000.log",
    "compute_evaluation_metrics_fast.py",
    "recompute_and_compare_all_metrics.py",
    # Original (Preserved) & Corrected Summaries
    "test_cohort_cap4000_evaluation_summary.json",
    "test_cohort_cap4000_corrected_evaluation_summary.json",
    # Audit & Verification Scripts
    "audit_dicom_geometry.py",
    "audit_slice_ordering.py",
    "test_patient_clustered_bootstrap_fix.py",
    "test_froc_implementations.py",
    "compare_matching_rules.py",
    # Visualizations & Plots
    "generate_test_cohort_froc_and_overlays.py",
    "generate_corrected_froc_plots.py",
    "froc_test_cohort_cap4000.png",
    "froc_test_cohort_cap4000_corrected.png",
    # Environment
    "environment_details.json",
    # Original Preserved Package
    "clear3d-spie-aapm-test-cohort-cap4000-package-20261109.zip"
]

overlays_dir = EXT_DIR / "test_cohort_overlays"
overlay_files = sorted(list(overlays_dir.glob("*.png"))) if overlays_dir.exists() else []

manifest = {
    "package_name": "clear3d-spie-aapm-test-cohort-cap4000-corrected-review-package-20261109.zip",
    "status": "CORRECTED_REVIEW_PACKAGE",
    "original_preserved_package_sha256": "bd08e186e668d8069e68d34f65e343b5124049d0ea204ad2b627dae5f3260d30",
    "superseded_results_record": {
        "superseded_artifact": "test_cohort_cap4000_evaluation_summary.json",
        "reason_superseded": "Patient-clustered bootstrap did not assign distinct identities to duplicate patient copies, artificially deflating bootstrap FROC CPM CI from [0.3469, 0.5193] to [0.2143, 0.3317]. Point estimates for coverage, sensitivity, and CPM were unaltered.",
        "corrected_artifact": "test_cohort_cap4000_corrected_evaluation_summary.json"
    },
    "matching_rules_compared": {
        "rule_A": "Original Closest-Proposal Rule",
        "rule_B": "Highest-Scoring-Proposal-Per-Target Rule"
    },
    "governance": {
        "reference_release": "HOLD",
        "confirmation_cohort": "83 patients strictly restricted / untouched",
        "internal_test_fold0": "Untouched / zero retraining",
        "manuscript_writing": "DEFERRED"
    },
    "files": {}
}

print(f"Creating consolidated corrected review archive {ZIP_OUT}...")
with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for fn in files_to_pack:
        fp = EXT_DIR / fn
        if fp.exists():
            h = sha256_file(fp)
            manifest["files"][fn] = {"sha256": h, "size_bytes": fp.stat().st_size}
            zf.write(fp, arcname=fn)
            print(f"  Added {fn} ({h[:12]}...)")
        else:
            print(f"  WARNING: {fn} not found!")

    for of in overlay_files:
        rel_name = f"test_cohort_overlays/{of.name}"
        h = sha256_file(of)
        manifest["files"][rel_name] = {"sha256": h, "size_bytes": of.stat().st_size}
        zf.write(of, arcname=rel_name)
        print(f"  Added {rel_name}")

zip_hash = sha256_file(ZIP_OUT)
manifest["package_sha256"] = zip_hash
manifest["total_files"] = len(manifest["files"])
manifest["package_size_bytes"] = ZIP_OUT.stat().st_size

with open(MANIFEST_OUT, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Package created: {ZIP_OUT}")
print(f"SHA-256: {zip_hash}")
print(f"Total files: {manifest['total_files']} (Size: {manifest['package_size_bytes'] / 1048576:.2f} MB)")
print(f"Manifest saved to {MANIFEST_OUT}")
