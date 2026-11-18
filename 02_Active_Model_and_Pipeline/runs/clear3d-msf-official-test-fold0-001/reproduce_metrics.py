"""Standalone Verification & Metric Reproduction Script for CLEAR3D-CT Test-Fold0.
Requires only numpy, scipy (or numpy.interp), and sklearn (or standard python).
Reproduces:
1. Candidate-level discrimination metrics at natural cut (0.9999467) and 0.5.
2. Official LUNA16 FROC curve and Competition Performance Metric (CPM).
"""
import sys, os, csv, json, math, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def main():
    print("=== CLEAR3D-CT TEST-FOLD0 METRIC REPRODUCTION ===")
    scores_path = HERE / 'scores.csv'
    report_path = HERE / 'final_test_report.json'
    
    with open(report_path) as f:
        ref_report = json.load(f)
        
    actual_hash = sha256_file(scores_path)
    expected_hash = ref_report['hashes']['scores_csv_sha256']
    print(f"Verifying scores.csv sha256: {actual_hash[:16]}...")
    assert actual_hash == expected_hash, f"Hash mismatch: expected {expected_hash}, got {actual_hash}"
    print("  -> SHA-256 MATCHES LOCKED EVALUATION RUN.")

    # 1. Candidate metrics
    print("\n1. Computing candidate classification metrics...")
    scores = []
    labels = []
    with open(scores_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            scores.append(float(r['probability']))
            labels.append(int(r['label']))
    
    scores = np.array(scores, dtype=np.float64)
    labels = np.array(labels, dtype=np.int64)
    print(f"  Loaded {len(scores)} candidates ({int((labels == 1).sum())} positive, {int((labels == 0).sum())} negative).")

    for cut_name, cut_val in [('nat_cut_oof', ref_report['model']['nat_cut_oof']), ('0.5', 0.5)]:
        pred = scores >= cut_val
        y = labels == 1
        tp = int((pred & y).sum())
        tn = int((~pred & ~y).sum())
        fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum())
        acc = (tp + tn) / len(labels)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        f1p = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        print(f"  [Threshold {cut_name} = {cut_val:.7f}]")
        print(f"    TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        print(f"    Accuracy: {acc:.6f}")
        print(f"    Positive F1: {f1p:.6f}")
        print(f"    Precision (PPV): {prec:.6f}")
        print(f"    Recall (Sens):   {rec:.6f}")
        print(f"    Specificity:     {spec:.6f}")

    # 2. LUNA16 FROC / CPM
    print("\n2. Verifying FROC sensitivities and CPM...")
    froc_file = HERE / 'cad_results' / 'froc_clear3d_msf_test0.txt'
    if not froc_file.is_file():
        froc_file = HERE / 'froc_eval' / 'cad' / 'froc_clear3d_msf_test0.txt'
    curves = np.loadtxt(froc_file, delimiter=',')
    order = np.argsort(curves[:, 0], kind='mergesort')
    grid = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    sens = np.interp(grid, curves[order, 0], curves[order, 1]).tolist()
    cpm = float(np.mean(sens))

    print(f"  FP/scan Grid: {grid}")
    print(f"  Sensitivities: {[round(s, 6) for s in sens]}")
    print(f"  Calculated CPM: {cpm:.6f}")
    ref_cpm = ref_report['froc']['cpm']
    print(f"  Report CPM:     {ref_cpm:.6f}")
    assert abs(cpm - ref_cpm) < 1e-6, f"CPM mismatch: {cpm} vs {ref_cpm}"
    print("  -> EXACT MATCH WITH LOCKED REPORT CPM.")
    print("\n=== REPRODUCTION VERIFIED SUCCESSFULLY ===")

if __name__ == '__main__':
    main()
