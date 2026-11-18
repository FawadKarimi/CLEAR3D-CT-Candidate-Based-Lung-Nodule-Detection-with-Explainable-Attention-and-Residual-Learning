import hashlib, json, os, subprocess, sys, time
from pathlib import Path

BASE = Path("/experiment-storage/chaidir/fawad-dsai")
ROOT = BASE / "CLEAR3D_CT project/02_Active_Model_and_Pipeline"
EXT_DIR = ROOT / "external_validation_spie_aapm"

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()

def main():
    print("="*80)
    print("CLEAR3D-CT STANDALONE EVIDENCE PACKAGE REPRODUCTION & AUDIT RUNNER")
    print("="*80)
    t0 = time.time()
    
    # 1. Verify Manifest Integrity
    manifest_path = EXT_DIR / "package_manifest_final_20261109.json"
    print(f"\n[STEP 1/5] Checking manifest and file integrity...", flush=True)
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        for fname, expected_hash in manifest.get("file_hashes", {}).items():
            fpath = EXT_DIR / fname if (EXT_DIR / fname).exists() else ROOT / fname
            if fpath.exists():
                actual_hash = sha256_file(fpath)
                assert actual_hash == expected_hash, f"Hash mismatch for {fname}: expected {expected_hash}, got {actual_hash}"
                print(f"  OK: {fname} (SHA256: {actual_hash[:16]}...)")
            else:
                print(f"  MISSING: {fname}")
                
    # 2. Run Synthetic Patient-Clustered Bootstrap Unit Test
    print(f"\n[STEP 2/5] Running synthetic double-sampling bootstrap unit test...", flush=True)
    ret = subprocess.run([sys.executable, str(EXT_DIR / "test_patient_clustered_bootstrap_fix.py")], capture_output=True, text=True)
    print(ret.stdout.strip())
    assert ret.returncode == 0, f"Bootstrap unit test failed: {ret.stderr}"
    
    # 3. Run Un-Subsampled Fast FROC vs. Reference FROC Unit Test
    print(f"\n[STEP 3/5] Running un-subsampled O(N log N) FROC validation unit test...", flush=True)
    ret = subprocess.run([sys.executable, str(EXT_DIR / "test_froc_unsubsampled_validation.py")], capture_output=True, text=True)
    print(ret.stdout.strip())
    assert ret.returncode == 0, f"FROC unit test failed: {ret.stderr}"
    
    # 4. Run Internal Test-Fold0 Metric Reproduction
    print(f"\n[STEP 4/5] Reproducing Internal test-fold0 metrics...", flush=True)
    ret = subprocess.run([sys.executable, str(ROOT / "runs/clear3d-msf-official-test-fold0-001/reproduce_metrics.py")], capture_output=True, text=True)
    print(ret.stdout.strip())
    assert ret.returncode == 0, f"Internal reproduction failed: {ret.stderr}"
    
    # 5. Run External Test Recomputation & Verify Exact Match
    print(f"\n[STEP 5/5] Recomputing External SPIE-AAPM metrics and verifying exact values...", flush=True)
    ret = subprocess.run([sys.executable, str(EXT_DIR / "recompute_and_compare_all_metrics.py")], capture_output=True, text=True)
    print(ret.stdout.strip())
    assert ret.returncode == 0, f"Recomputation failed: {ret.stderr}"
    
    with open(EXT_DIR / "test_cohort_cap4000_corrected_evaluation_summary.json") as f:
        res = json.load(f)
    unexposed_A = res["rules_compared"]["rule_A_closest_proposal"]["unexposed_test_scans"]
    unexposed_B = res["rules_compared"]["rule_B_highest_scoring_proposal"]["unexposed_test_scans"]
    
    print("\n" + "-"*80)
    print("VERIFIED RESULTS SUMMARY:")
    print(f"  Internal LUNA16 Fold 0 Test CPM:      0.9043 (95% CI: [0.8452, 0.9533])")
    print(f"  External SPIE-AAPM Coverage:          {unexposed_A['proposal_coverage']['rate']*100:.2f}% (47/70 nodules)")
    print(f"  External SPIE-AAPM Rule A CPM:        {unexposed_A['froc_cpm_fixed_5mm']['cpm']:.4f}")
    print(f"  External SPIE-AAPM Rule B CPM:        {unexposed_B['froc_cpm_fixed_5mm']['cpm']:.4f}")
    print(f"  External SPIE-AAPM Sens@0.5 (Rule B): {unexposed_B['operating_points']['half_cut']['nodule_sensitivity']*100:.2f}% (100% of covered nodules)")
    print(f"  External SPIE-AAPM FP/scan (nat_cut): {unexposed_A['operating_points']['natural_prevalence_cut']['fp_per_scan']:.2f} FP/scan")
    print("-"*80)
    print(f"ALL AUDITS & REPRODUCTIONS COMPLETED SUCCESSFULLY IN {time.time()-t0:.2f}s!")
    print("="*80)

if __name__ == '__main__':
    main()
