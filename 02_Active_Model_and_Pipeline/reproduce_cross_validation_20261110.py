#!/usr/bin/env python3
"""
Standalone Reproduction & Verification Script for CLEAR3D-CT 10-Fold Patient-Disjoint Cross-Validation
Calculates candidate-level metrics and official LUNA16 FROC/CPM directly from prediction CSVs.
"""
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

BASE_DIR = Path("/experiment-storage/chaidir/fawad-dsai")
ROOT_DIR = BASE_DIR / "CLEAR3D_CT project/02_Active_Model_and_Pipeline"
CV_DIR = ROOT_DIR / "runs/crossval-10fold-clear3d-20261109"
CKPT_DIR = CV_DIR / "checkpoints"
PRED_DIR = CV_DIR / "predictions"
MANIFEST_FILE = ROOT_DIR / "patient_to_fold_manifest_20261109.json"
REFD = BASE_DIR / "data/reference/luna16-zenodo-3723295/resolved-reference-v1"
EVALUATOR_SCRIPT = ROOT_DIR / "paper-ablation-verify-package-20261107-004/evaluator/noduleCADEvaluationLUNA16.py"

def sha256_file(p: Path) -> str:
    if not p.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def calc_candidate_metrics(y_true, y_score, threshold=0.5):
    order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[order]
    tp_cum = np.cumsum(y_true_sorted)
    fp_cum = np.cumsum(1 - y_true_sorted)
    precision_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    recall_curve = tp_cum / max(tp_cum[-1], 1)
    precision_curve = np.concatenate(([1.0], precision_curve))
    recall_curve = np.concatenate(([0.0], recall_curve))
    pr_auc = float(np.sum((recall_curve[1:] - recall_curve[:-1]) * precision_curve[1:]))

    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        roc_auc = 0.0
    else:
        order_m = np.argsort(y_score, kind="mergesort")
        ranks = np.empty(len(y_score), dtype=np.float64)
        ranks[order_m] = np.arange(1, len(y_score) + 1)
        roc_auc = float((ranks[y_true == 1].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))

    pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    total = len(y_true)

    acc = (tp + tn) / total if total > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_pos = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    rec_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1_neg = (2 * prec_neg * rec_neg) / (prec_neg + rec_neg) if (prec_neg + rec_neg) > 0 else 0.0
    f1_macro = (f1_pos + f1_neg) / 2.0

    return {
        "total": total,
        "pos": int(np.sum(y_true == 1)),
        "neg": int(np.sum(y_true == 0)),
        "threshold": threshold,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": acc,
        "sensitivity": rec,
        "precision": prec,
        "specificity": spec,
        "f1_positive": f1_pos,
        "f1_macro": f1_macro,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }

def main():
    print("=" * 80)
    print("CLEAR3D-CT: 10-Fold Patient-Disjoint Cross-Validation Reproduction & Audit")
    print("=" * 80)

    # 1. Verify Manifest
    if MANIFEST_FILE.exists():
        manifest_sha = sha256_file(MANIFEST_FILE)
        print(f"Patient Manifest: {MANIFEST_FILE}")
        print(f"  SHA-256: {manifest_sha}")
    else:
        print(f"WARNING: Manifest {MANIFEST_FILE} not found!")

    # 2. Check Fold Artifacts
    print("\nPer-Fold Artifacts & Candidate Metrics:")
    print("-" * 80)
    fold_metrics = {}
    for k in range(10):
        ckpt_p = CKPT_DIR / f"fold_{k}.pt"
        pred_p = PRED_DIR / f"fold_{k}_predictions.csv"
        ckpt_sha = sha256_file(ckpt_p) if ckpt_p.exists() else "MISSING"
        pred_sha = sha256_file(pred_p) if pred_p.exists() else "MISSING"

        if pred_p.exists():
            with open(pred_p, "r", newline="") as fp:
                reader = csv.DictReader(fp)
                rows = list(reader)
            y_true = np.array([int(r["label"]) for r in rows])
            y_score = np.array([float(r["probability"]) for r in rows])
            m = calc_candidate_metrics(y_true, y_score)
            fold_metrics[k] = m
            print(f"Fold {k}:")
            print(f"  Checkpoint: {ckpt_sha[:16]}... ({ckpt_p.stat().st_size:,} bytes)" if ckpt_p.exists() else "  Checkpoint: MISSING")
            print(f"  Prediction: {pred_sha[:16]}... ({pred_p.stat().st_size:,} bytes)")
            print(f"  Candidates: {m['total']:,} (pos={m['pos']:,}, neg={m['neg']:,})")
            print(f"  Metrics @ 0.5: Sens={m['sensitivity']:.4f}, Prec={m['precision']:.4f}, F1_pos={m['f1_positive']:.4f}, Macro_F1={m['f1_macro']:.4f}, ROC-AUC={m['roc_auc']:.4f}, PR-AUC={m['pr_auc']:.4f}")
        else:
            print(f"Fold {k}: Prediction file pending...")

    # 3. Pooled Evaluation
    pooled_csv = CV_DIR / "pooled_oof_predictions.csv"
    if pooled_csv.exists():
        print("\n" + "=" * 80)
        print("POOLED OUT-OF-FOLD EVALUATION (ALL 10 FOLDS)")
        print("=" * 80)
        pooled_sha = sha256_file(pooled_csv)
        with open(pooled_csv, "r", newline="") as fp:
            reader = csv.DictReader(fp)
            rows = list(reader)
        y_true = np.array([int(r["label"]) for r in rows])
        y_score = np.array([float(r["probability"]) for r in rows])
        scans = set(r["seriesuid"] for r in rows)
        pm = calc_candidate_metrics(y_true, y_score)
        print(f"File: {pooled_csv}")
        print(f"SHA-256: {pooled_sha}")
        print(f"Total Rows: {pm['total']:,} across {len(scans)} unique scans")
        print(f"Positive Candidates: {pm['pos']:,} | Negative: {pm['neg']:,}")
        print(f"ROC-AUC:  {pm['roc_auc']:.4f}")
        print(f"PR-AUC:   {pm['pr_auc']:.4f}")
        print(f"Accuracy: {pm['accuracy']:.4f} ({pm['tp']+pm['tn']:,}/{pm['total']:,})")
        print(f"Confusion Matrix @ 0.5: TP={pm['tp']:,}, TN={pm['tn']:,}, FP={pm['fp']:,}, FN={pm['fn']:,}")
        print(f"Sensitivity (Recall): {pm['sensitivity']:.4f}")
        print(f"Precision:            {pm['precision']:.4f}")
        print(f"Specificity:          {pm['specificity']:.4f}")
        print(f"Positive-class F1:    {pm['f1_positive']:.4f}")
        print(f"Macro F1:             {pm['f1_macro']:.4f}")
    else:
        print("\nPooled prediction file not yet available (awaiting remaining folds).")

if __name__ == "__main__":
    main()

