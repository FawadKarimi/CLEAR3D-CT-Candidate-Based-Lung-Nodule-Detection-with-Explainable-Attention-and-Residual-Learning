#!/usr/bin/env python3
"""
CLEAR3D-CT Development Ablation Package Verification Runner.
Executes all self-contained audits:
1. Manifest SHA-256 integrity check.
2. Candidate key alignment across prediction CSVs.
3. Standalone FROC regeneration directly from predictions.
4. Patient-level paired bootstrap resampling verification.
"""
from __future__ import annotations
import csv, hashlib, json, os, subprocess, sys, time
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest() -> bool:
    print("")
    print("[1/4] Verifying Package Manifest Integrity...")
    manifest_path = PKG_ROOT / 'manifest.json'
    if not manifest_path.exists():
        print("  ERROR: manifest.json not found!")
        return False
    manifest = json.load(manifest_path.open())
    failures = 0
    total = len(manifest['files'])
    for rel_path, meta in manifest['files'].items():
        p = PKG_ROOT / rel_path
        if not p.exists():
            print(f"  MISSING: {rel_path}")
            failures += 1
            continue
        actual_sha = hash_file(p)
        if actual_sha != meta['sha256']:
            print(f"  HASH MISMATCH: {rel_path} (expected {meta['sha256'][:8]}, got {actual_sha[:8]})")
            failures += 1
    if failures == 0:
        print(f"  PASSED: All {total} package files match manifest SHA-256 checksums.")
        return True
    else:
        print(f"  FAILED: {failures}/{total} files failed integrity check.")
        return False


def verify_candidate_alignment() -> bool:
    print("")
    print("[2/4] Verifying Prediction Candidate Key Alignment...")
    preds = [
        PKG_ROOT / 'predictions/fold0_baseline_scores.csv',
        PKG_ROOT / 'predictions/control_b_nomining_scores.csv',
        PKG_ROOT / 'predictions/control_a_fineonly_scores.csv'
    ]
    keys_list = []
    for p in preds:
        with p.open(newline='') as f:
            r = csv.DictReader(f)
            keys = [(row['seriesuid'], row['candidate_id'], row['coordX'], row['coordY'], row['coordZ'], row['label']) for row in r]
            keys_list.append(keys)
            
    n = len(keys_list[0])
    match_b = (keys_list[0] == keys_list[1])
    match_a = (keys_list[0] == keys_list[2])
    
    if n == 122689 and match_b and match_a:
        print(f"  PASSED: Exactly 122,689 rows with 100% key match across all 3 models.")
        return True
    else:
        print(f"  FAILED: Candidate row mismatch (n={n}, match_b={match_b}, match_a={match_a})")
        return False


def verify_froc_regeneration() -> bool:
    print("")
    print("[3/4] Verifying FROC Curve & CPM Regeneration...")
    cmd = [sys.executable, str(PKG_ROOT / 'evaluator/regenerate_froc_curves.py'), '--pkg-root', str(PKG_ROOT)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        print("  PASSED: FROC regeneration reproduced all 3 CPM metrics perfectly.")
        for line in proc.stdout.strip().splitlines():
            if 'Regenerated CPM' in line or 'MATCH' in line:
                print(f"    {line.strip()}")
        return True
    else:
        print("  FAILED: FROC regeneration script returned non-zero exit code.")
        print(proc.stderr)
        return False


def verify_bootstrap() -> bool:
    print("")
    print("[4/4] Verifying Patient-Level Paired Bootstrap Resampling...")
    cmd = [sys.executable, str(PKG_ROOT / 'bootstrap/run_paired_bootstrap.py'), '--pkg-root', str(PKG_ROOT), '--b', '2000', '--seed', '42']
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        print("  PASSED: Paired bootstrap analysis completed successfully.")
        out_json = PKG_ROOT / 'bootstrap/bootstrap_results.json'
        data = json.load(out_json.open())
        d1 = data['paired_differences']['diff1_fold0_vs_nomining']
        d2 = data['paired_differences']['diff2_nomining_vs_fineonly']
        print(f"    Diff 1 (Fold0 - NoMining):    Point={d1['point_estimate']:+.6f}, Mean={d1['bootstrap_mean']:+.6f}, 95% CI=[{d1['ci_95_percentile'][0]:+.6f}, {d1['ci_95_percentile'][1]:+.6f}]")
        print(f"    Diff 2 (NoMining - FineOnly): Point={d2['point_estimate']:+.6f}, Mean={d2['bootstrap_mean']:+.6f}, 95% CI=[{d2['ci_95_percentile'][0]:+.6f}, {d2['ci_95_percentile'][1]:+.6f}]")
        return True
    else:
        print("  FAILED: Paired bootstrap script returned non-zero exit code.")
        print(proc.stderr)
        return False


def main():
    print("=" * 80)
    print("CLEAR3D-CT DEVELOPMENT ABLATION VERIFICATION SUITE")
    print(f"Package Root: {PKG_ROOT}")
    print("=" * 80)
    t0 = time.time()
    
    ok1 = verify_manifest()
    ok2 = verify_candidate_alignment()
    ok3 = verify_froc_regeneration()
    ok4 = verify_bootstrap()
    
    elapsed = time.time() - t0
    print("")
    print("=" * 80)
    all_ok = ok1 and ok2 and ok3 and ok4
    if all_ok:
        print(f"VERIFICATION STATUS: ALL AUDITS PASSED (Total time: {elapsed:.2f} s)")
    else:
        print(f"VERIFICATION STATUS: FAILURES DETECTED")
    print("=" * 80)
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
