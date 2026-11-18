#!/usr/bin/env python3
"""
Corrected MedicalNet Full-Development Cohort Inference (142 Scans, 122,689 Candidates)
Using exact 48x48x48 valid-voxel-mask normalization matching original training.
"""
import os, sys, json, csv, time, shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path("/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/02_Active_Model_and_Pipeline")
sys.path.insert(0, str(ROOT))
from train_medicalnet_resnet18_comparator import MedicalNetClassifier
from lidc3d.geometry import crop

EV_DIR = ROOT / "runs/evaluation-full-dev-20261109"
PREP = Path("/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-development-fold0-v1/prepared")
DEV_JSONL = Path("/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-comparison-protocol-20261025/runs/phase-cf/dev.jsonl")

# 1. Backup old mismatched file if not already backed up
old_mn_csv = EV_DIR / "full_dev_predictions_medicalnet.csv"
backup_mn_csv = EV_DIR / "full_dev_predictions_medicalnet_mismatched_raw32_backup.csv"
if old_mn_csv.exists() and not backup_mn_csv.exists():
    shutil.copy2(old_mn_csv, backup_mn_csv)
    print(f"Backed up old MedicalNet CSV to {backup_mn_csv}", flush=True)

# 2. Load series to scan mapping
s2s = {json.loads(l)['series']: json.loads(l)['scan'] for l in open(DEV_JSONL)}

# 3. Load reference dataframe (122,689 candidates)
df_ref = pd.read_csv(EV_DIR / "full_dev_predictions_run004.csv")
n_cands = len(df_ref)
print(f"Total candidates to score: {n_cands}", flush=True)

# 4. Load MedicalNet model on cuda:1
device = torch.device("cuda:1")
model = MedicalNetClassifier(num_classes=2, dropout=0.2).to(device)
ckpt_path = ROOT / "runs/comparator-medicalnet-resnet18-20261109/best_checkpoint.pt"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded MedicalNet model from {ckpt_path} on {device}", flush=True)

# 5. Group candidates by scan
cands_by_scan = defaultdict(list)
for idx in range(n_cands):
    suid = df_ref["seriesuid"].iloc[idx]
    scan_id = s2s[suid]
    cands_by_scan[scan_id].append(idx)

scans_list = list(cands_by_scan.keys())
n_scans = len(scans_list)
print(f"Total scans to process: {n_scans}", flush=True)

# 6. Run inference scan by scan
medicalnet_probs = np.zeros(n_cands, dtype=np.float32)
t_start = time.time()

with torch.no_grad():
    for s_i, scan_id in enumerate(scans_list, 1):
        cand_indices = cands_by_scan[scan_id]
        n_sc = len(cand_indices)
        
        d = PREP / scan_id
        geo = json.loads((d / "geometry.json").read_text())
        aff = np.asarray(geo["affine"], dtype=np.float64)
        inv = np.linalg.inv(aff)
        img = np.load(d / "image.npy", mmap_mode="r")
        val = np.load(d / "valid.npy", mmap_mode="r")
        
        c_batch = np.empty((n_sc, 1, 48, 48, 48), dtype=np.float32)
        for k, orig_idx in enumerate(cand_indices):
            cx = df_ref["coordX"].iloc[orig_idx]
            cy = df_ref["coordY"].iloc[orig_idx]
            cz = df_ref["coordZ"].iloc[orig_idx]
            zyx = (inv @ np.array([cx, cy, cz, 1.0]))[:3]
            
            p, _ = crop(img, zyx, 48, 0.0)
            v, _ = crop(val, zyx, 48, False)
            v_mask = v > 0.5
            if v_mask.any():
                vals = p[v_mask]
                m = float(vals.mean())
                s = max(float(vals.std()), 1e-6)
                c_batch[k, 0] = np.where(v_mask, (p - m) / s, 0.0)
            else:
                c_batch[k, 0] = 0.0
                
        # Batched model inference
        for b_idx in range(0, n_sc, 64):
            end_idx = min(b_idx + 64, n_sc)
            bx = torch.from_numpy(c_batch[b_idx:end_idx]).to(device)
            logits = model(bx)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            for k_sub, p_val in enumerate(probs):
                orig_idx = cand_indices[b_idx + k_sub]
                medicalnet_probs[orig_idx] = p_val
                
        if s_i % 10 == 0 or s_i == n_scans:
            el = time.time() - t_start
            rem = (el / s_i) * (n_scans - s_i)
            print(f"[{s_i:3d}/{n_scans}] Scored scan {scan_id} ({n_sc:4d} cands) | Elapsed: {el:.1f}s | Est. remaining: {rem:.1f}s", flush=True)

t_total = time.time() - t_start
print(f"Full inference completed in {t_total:.1f}s ({(t_total/n_cands)*1000:.2f} ms/cand)", flush=True)

# 7. Verification against saved dev_predictions_1060.csv
saved_1060 = {}
with open(ROOT / "runs/comparator-medicalnet-resnet18-20261109/dev_predictions_1060.csv") as f:
    for row in csv.DictReader(f):
        saved_1060[row["id"]] = float(row["probability"])

cid_to_prob = dict(zip(df_ref["candidate_id"], medicalnet_probs))
diffs = [abs(cid_to_prob[cid] - saved_1060[cid]) for cid in saved_1060]
max_diff = max(diffs)
mean_diff = sum(diffs) / len(diffs)
print(chr(10) + "=" * 60, flush=True)
print(f"VERIFICATION ON 1,060 SAVED DEV PREDICTIONS:", flush=True)
print(f"  Max absolute difference: {max_diff:.8e}", flush=True)
print(f"  Mean absolute difference: {mean_diff:.8e}", flush=True)
print(f"  Count within 1e-5 tolerance: {sum(1 for d in diffs if d < 1e-5)} / 1060", flush=True)
assert max_diff < 1e-4, f"Verification failed! Max difference {max_diff} exceeds 1e-4"
print("  VERIFICATION PASSED! Exact match within floating-point tolerance.", flush=True)
print("=" * 60 + chr(10), flush=True)

# 8. Save corrected CSV
df_out = df_ref[["seriesuid", "coordX", "coordY", "coordZ"]].copy()
df_out["probability"] = medicalnet_probs
df_out["candidate_id"] = df_ref["candidate_id"]
df_out["rank"] = df_ref["rank"]
df_out["label"] = df_ref["label"]

out_csv = EV_DIR / "full_dev_predictions_medicalnet.csv"
df_out.to_csv(out_csv, index=False)
print(f"Saved corrected MedicalNet predictions to {out_csv} ({os.path.getsize(out_csv):,} bytes)", flush=True)

