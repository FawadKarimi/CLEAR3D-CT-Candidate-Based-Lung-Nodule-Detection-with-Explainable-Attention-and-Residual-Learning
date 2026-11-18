import os, sys, time, json, csv, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn

BASE = Path("/experiment-storage/chaidir/fawad-dsai")
ROOT = BASE / "CLEAR3D_CT project/02_Active_Model_and_Pipeline"
WORK = BASE / "CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64/source"
MEDNET_DIR = BASE / "CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-MedicalNet-review-20261013-6b802e"
PREP = BASE / "CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-development-fold0-v1/prepared"
SPLIT_PATH = BASE / "CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-comparison-protocol-20261025/runs/phase-cf/split.json"
DEV_JSONL = BASE / "CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-comparison-protocol-20261025/runs/phase-cf/dev.jsonl"
SELECT = ROOT / "selection-001.json"
RUN004_CSV = ROOT / "runs/clear3d-msf-official-dev-20261106-004/scores.csv"

OUT_DIR = ROOT / "runs/evaluation-full-dev-20261109"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path[:0] = [str(ROOT), str(WORK / "src"), str(WORK / "scripts"), str(MEDNET_DIR)]
from lidc3d.geometry import crop
import loader
from train_clear3d_highres_improvement import (
    CLEAR3DHighRes, make_inputs, build_focus_grids,
    extract_pair_u16, unpack_u16
)
from train_medicalnet_resnet18_comparator import MedicalNetClassifier

print("=== STARTING FULL-COHORT SCORING OF ALL 142 DEV SCANS ===", flush=True)

# 1. Load models
dev_hr = torch.device('cuda:1')
model_hr = CLEAR3DHighRes(in_channels=3, dim=160, dropout=0.25).to(dev_hr)
ckpt_hr = torch.load(ROOT / 'runs/improvement-clear3d-highres-20261109/best_checkpoint.pt', map_location=dev_hr)
model_hr.load_state_dict(ckpt_hr['model_state_dict'])
model_hr.eval()
f_fine, f_ctx = build_focus_grids(dev_hr)
print("HighRes model loaded on cuda:1", flush=True)

dev_mn = torch.device('cuda:3')
model_mn = MedicalNetClassifier(num_classes=2, dropout=0.2).to(dev_mn)
ckpt_mn = torch.load(ROOT / 'runs/comparator-medicalnet-resnet18-20261109/best_checkpoint.pt', map_location=dev_mn)
model_mn.load_state_dict(ckpt_mn['model_state_dict'])
model_mn.eval()
print("MedicalNet model loaded on cuda:3", flush=True)

# 2. Load reference Run 004 candidates
print("Loading reference Run 004 candidates from scores.csv...", flush=True)
import pandas as pd
df_ref = pd.read_csv(RUN004_CSV)
n_cands = len(df_ref)
print(f"Total reference candidates: {n_cands}", flush=True)
assert n_cands == 122689

# Map series to scan
series_to_scan = {}
with open(DEV_JSONL) as f:
    for line in f:
        d = json.loads(line)
        series_to_scan[d['series']] = d['scan']

# Group rows by scan maintaining order
scan_cands = defaultdict(list)
for idx, r in df_ref.iterrows():
    scan_id = series_to_scan[r['seriesuid']]
    scan_cands[scan_id].append((idx, r))

total_scans = len(scan_cands)
print(f"Total scans to process: {total_scans}", flush=True)
assert total_scans == 142

highres_probs = np.zeros(n_cands, dtype=np.float32)
medicalnet_probs = np.zeros(n_cands, dtype=np.float32)

t_start = time.time()
views = [(False, False), (True, False), (False, True), (True, True)]

for s_idx, (sid, items) in enumerate(scan_cands.items(), 1):
    t_s0 = time.time()
    d = PREP / sid
    geo = json.loads((d / "geometry.json").read_text())
    aff = np.asarray(geo["affine"], dtype=np.float64)
    inv = np.linalg.inv(aff)
    img = np.load(d / "image.npy", mmap_mode="r")
    val = np.load(d / "valid.npy", mmap_mode="r")
    
    n_sc = len(items)
    
    # 1. HighRes inputs
    f_im = np.empty((n_sc, 33, 33, 33), dtype=np.float32)
    f_va = np.empty((n_sc, 33, 33, 33), dtype=np.float32)
    c_im = np.empty((n_sc, 25, 25, 25), dtype=np.float32)
    c_va = np.empty((n_sc, 25, 25, 25), dtype=np.float32)
    
    # 2. MedicalNet inputs
    mn_crops = np.empty((n_sc, 1, 32, 32, 32), dtype=np.float32)
    
    for k, (orig_idx, r) in enumerate(items):
        zyx = (inv @ np.array([r['coordX'], r['coordY'], r['coordZ'], 1.0]))[:3]
        
        # HighRes patch extraction
        f37, c27 = extract_pair_u16(img, val, zyx)
        f_im[k], f_va[k] = unpack_u16(f37[2:35, 2:35, 2:35])
        c_im[k], c_va[k] = unpack_u16(c27[1:26, 1:26, 1:26])
        
        # MedicalNet patch extraction
        c32, _ = crop(img, zyx, 32, 0.0)
        m = np.mean(c32)
        s = np.std(c32)
        if s > 1e-4:
            c32 = (c32 - m) / s
        else:
            c32 = c32 - m
        mn_crops[k, 0] = c32
        
    # HighRes 4V inference
    with torch.no_grad():
        for b_idx in range(0, n_sc, 64):
            end_idx = min(b_idx + 64, n_sc)
            sub_len = end_idx - b_idx
            vf, vc = make_inputs(
                torch.from_numpy(f_im[b_idx:end_idx]).to(dev_hr),
                torch.from_numpy(f_va[b_idx:end_idx]).to(dev_hr),
                torch.from_numpy(c_im[b_idx:end_idx]).to(dev_hr),
                torch.from_numpy(c_va[b_idx:end_idx]).to(dev_hr),
                f_fine, f_ctx
            )
            acc = torch.zeros(sub_len, device=dev_hr, dtype=torch.float32)
            for fz, fx in views:
                xf = vf; xc = vc
                if fz: xf = torch.flip(xf, [2]); xc = torch.flip(xc, [2])
                if fx: xf = torch.flip(xf, [4]); xc = torch.flip(xc, [4])
                logits = model_hr(xf, xc).float()
                acc = acc + (logits[:, 1] - logits[:, 0])
            p_hr = 1.0 / (1.0 + np.exp(-(acc * 0.25).cpu().numpy()))
            for k_sub, p_val in enumerate(p_hr):
                orig_idx = items[b_idx + k_sub][0]
                highres_probs[orig_idx] = p_val
                
    # MedicalNet inference
    with torch.no_grad():
        for b_idx in range(0, n_sc, 64):
            end_idx = min(b_idx + 64, n_sc)
            b_mn = torch.from_numpy(mn_crops[b_idx:end_idx]).to(dev_mn)
            logits_mn = model_mn(b_mn)
            p_mn = torch.softmax(logits_mn, dim=-1)[:, 1].cpu().numpy()
            for k_sub, p_val in enumerate(p_mn):
                orig_idx = items[b_idx + k_sub][0]
                medicalnet_probs[orig_idx] = p_val
                
    if s_idx % 10 == 0 or s_idx == total_scans:
        elapsed = time.time() - t_start
        print(f"[{s_idx:3d}/{total_scans}] Scored scan {sid} ({n_sc:4d} cands) | Total elapsed: {elapsed:.1f}s | Est. remaining: {elapsed/s_idx*(total_scans-s_idx):.1f}s", flush=True)

print("Scoring completed in {:.1f}s!".format(time.time() - t_start), flush=True)

# 3. Write output CSVs with exact alignment
print("Writing aligned prediction CSVs...", flush=True)

# Copy reference Run 004
df_ref.to_csv(OUT_DIR / "full_dev_predictions_run004.csv", index=False)

# Write HighRes
df_hr = df_ref.copy()
df_hr['probability'] = highres_probs
# Recompute rank per scan
df_hr['rank'] = df_hr.groupby('seriesuid')['probability'].rank(ascending=False, method='first').astype(int)
df_hr.to_csv(OUT_DIR / "full_dev_predictions_highres.csv", index=False)

# Write MedicalNet
df_mn = df_ref.copy()
df_mn['probability'] = medicalnet_probs
df_mn['rank'] = df_mn.groupby('seriesuid')['probability'].rank(ascending=False, method='first').astype(int)
df_mn.to_csv(OUT_DIR / "full_dev_predictions_medicalnet.csv", index=False)

print("Aligned CSVs written successfully!", flush=True)

# 4. Compute metrics
def calc_metrics(y_true, y_score, cut=0.5):
    pred = (y_score >= cut).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    prec = float(tp / (tp + fp)) if (tp + fp) else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) else 0.0
    f1p = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    f1n = float(2 * tn / (2 * tn + fp + fn)) if (2 * tn + fp + fn) else 0.0
    acc = float((tp + tn) / len(y_true))
    return {
        'threshold': float(cut), 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'accuracy': acc, 'precision': prec, 'recall': rec, 'specificity': spec,
        'f1_positive': f1p, 'f1_negative': f1n, 'macro_f1': 0.5 * (f1p + f1n),
        'fp_per_scan': float(fp / 142.0)
    }

def pr_auc(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(tp[-1], 1)
    prec = np.concatenate(([1.0], prec))
    rec = np.concatenate(([0.0], rec))
    return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))

def roc_auc(y_true, y_score):
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    order = np.argsort(y_score, kind='mergesort')
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    return float((ranks[y_true == 1].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))

y_true = df_ref['label'].values

metrics_summary = {}
for name, probs in [('Run 004 Ensemble', df_ref['probability'].values), ('CLEAR3D-HighRes', highres_probs), ('MedicalNet ResNet-18', medicalnet_probs)]:
    prauc = pr_auc(y_true, probs)
    rocauc = roc_auc(y_true, probs)
    m_half = calc_metrics(y_true, probs, 0.5)
    
    # Sweep best Pos-F1
    best_f1 = -1
    best_th = 0.5
    for th in np.linspace(0.1, 0.99, 90):
        m = calc_metrics(y_true, probs, th)
        if m['f1_positive'] > best_f1:
            best_f1 = m['f1_positive']
            best_th = th
    m_best = calc_metrics(y_true, probs, best_th)
    
    metrics_summary[name] = {
        'natural_population_122689': {
            'pr_auc': prauc,
            'roc_auc': rocauc,
            'threshold_0_5': m_half,
            'best_threshold': m_best
        }
    }

# Also evaluate on 1060 dev subset
sel = json.load(open(SELECT))
dev_sel_ids = set(c['id'] for c in sel['dev']['selected'])
sub_mask = df_ref['candidate_id'].isin(dev_sel_ids).values
y_true_1060 = y_true[sub_mask]

for name, probs in [('Run 004 Ensemble', df_ref['probability'].values[sub_mask]), ('CLEAR3D-HighRes', highres_probs[sub_mask]), ('MedicalNet ResNet-18', medicalnet_probs[sub_mask])]:
    prauc_1060 = pr_auc(y_true_1060, probs)
    rocauc_1060 = roc_auc(y_true_1060, probs)
    m_half_1060 = calc_metrics(y_true_1060, probs, 0.5)
    metrics_summary[name]['dev_1060_subset'] = {
        'pr_auc': prauc_1060,
        'roc_auc': rocauc_1060,
        'threshold_0_5': m_half_1060
    }

with open(OUT_DIR / "metrics_summary.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)

print("Metrics summary saved to:", OUT_DIR / "metrics_summary.json", flush=True)
print("=== COMPLETED SUCCESSFULLY ===", flush=True)
