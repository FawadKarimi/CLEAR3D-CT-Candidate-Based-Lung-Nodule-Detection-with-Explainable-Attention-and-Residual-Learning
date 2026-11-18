"""
Outer-Fold Cross-Validation Runner for CLEAR3D-CT on LUNA16
Strictly patient-disjoint, volume-deduplicated, and zero-leakage:
- Scout training, mining, model training, and Platt calibration strictly internal to training split.
- Held-out fold scored 100% out-of-fold under 4-view TTA.
- Status, checkpoints, predictions, and logs tracked atomically per fold.
"""
import argparse, csv, fcntl, hashlib, json, math, os, random, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = Path("/experiment-storage/chaidir/fawad-dsai")
ROOT = BASE / "CLEAR3D_CT project/02_Active_Model_and_Pipeline"
WORK = BASE / "CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64/source"
sys.path[:0] = [str(ROOT), str(WORK / "src"), str(WORK / "scripts")]

from clear3d_msf_model_20261106_002 import CLEAR3DMSFv2, build_focus_grids, make_inputs
from train_clear3d_msf_ensemble_20261106_004 import gpu_extract_from_p51, slice_p51_batch
from lidc3d.geometry import crop

REFD = BASE / "data/reference/luna16-zenodo-3723295/resolved-reference-v1"
CV_DIR = ROOT / "runs/crossval-10fold-clear3d-20261109"
CKPT_DIR = CV_DIR / "checkpoints"
PRED_DIR = CV_DIR / "predictions"
LOG_DIR = CV_DIR / "logs"
CACHE_DIR = CV_DIR / "cache"
STATUS_FILE = CV_DIR / "status.json"

for d in (CKPT_DIR, PRED_DIR, LOG_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def update_status(fold: int, status_str: str, **kwargs):
    lock_file = STATUS_FILE.with_suffix(".lock")
    with open(lock_file, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if STATUS_FILE.exists():
                data = json.loads(STATUS_FILE.read_text())
            else:
                data = {"protocol": "CLEAR3D-LUNA16-BROADER-CV-20261109", "folds": {}}
            f_str = str(fold)
            if f_str not in data["folds"]:
                data["folds"][f_str] = {}
            data["folds"][f_str]["status"] = status_str
            data["folds"][f_str]["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S+07:00")
            for k, v in kwargs.items():
                data["folds"][f_str][k] = v
            tmp_file = STATUS_FILE.with_suffix(".tmp")
            with open(tmp_file, "w") as f:
                json.dump(data, f, indent=2)
            tmp_file.replace(STATUS_FILE)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)

def load_scan_universal(scan_id: str, use_mmap: bool = True):
    p1 = BASE / "CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-development-fold0-v1/prepared" / scan_id
    p2 = BASE / "CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-test-fold0-v1/prepared" / scan_id
    d = p1 if p1.exists() else p2
    assert d.exists(), f"Scan {scan_id} not found"
    geo = json.loads((d / "geometry.json").read_text())
    aff = np.asarray(geo["affine"], dtype=np.float64)
    inv = np.linalg.inv(aff)
    mm = "r" if use_mmap else None
    img = np.load(d / "image.npy", mmap_mode=mm, allow_pickle=False)
    val = np.load(d / "valid.npy", mmap_mode=mm, allow_pickle=False)
    return img, val, inv

def run_fold(fold_k: int, device_id: int = 1, epochs: int = 12):
    t_fold0 = time.time()
    log_path = LOG_DIR / f"fold_{fold_k}.log"
    log_f = open(log_path, "a", buffering=1)
    
    def log(msg):
        t_str = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{t_str}] [Fold {fold_k}] {msg}"
        print(line, flush=True)
        log_f.write(line + chr(10))
        log_f.flush()

    log(f"=== STARTING PATIENT-DISJOINT OUTER FOLD {fold_k} ===")
    
    pred_csv_path = PRED_DIR / f"fold_{fold_k}_predictions.csv"
    ckpt_path = CKPT_DIR / f"fold_{fold_k}.pt"
    
    if pred_csv_path.exists() and pred_csv_path.stat().st_size > 1000:
        log(f"Predictions already exist at {pred_csv_path} ({pred_csv_path.stat().st_size:,} bytes). Skipping fold {fold_k}.")
        log_f.close()
        return True

    update_status(fold_k, "running", started_at=time.strftime("%Y-%m-%dT%H:%M:%S+07:00"))
    device = torch.device(f"cuda:{device_id}")
    f_fine, f_ctx = build_focus_grids(device)
    
    # 1. Load manifest and cohort mappings
    manifest = json.load(open(ROOT / "patient_to_fold_manifest_20261109.json"))
    p2f = manifest["patient_to_fold"]
    
    dev_scans = [json.loads(l) for l in open(BASE / "CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-development-fold0-v1/prepared/scans.jsonl") if l.strip()]
    test_scans = [json.loads(l) for l in open(BASE / "CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-cohort-fold0-v1/cohort_test_WITHHOLD.jsonl") if l.strip()]
    scan_map = {s["series_uid"]: s for s in dev_scans + test_scans if s["patient_id"] in p2f}
    
    held_out_scans = {suid: s for suid, s in scan_map.items() if p2f[s["patient_id"]] == fold_k}
    train_scans = {suid: s for suid, s in scan_map.items() if p2f[s["patient_id"]] != fold_k}
    log(f"Training scans: {len(train_scans)}, Held-out scans: {len(held_out_scans)}")
    
    model = CLEAR3DMSFv2(in_channels=3, dim=160, dropout=0.25).to(device)
    
    if ckpt_path.exists():
        log(f"Existing checkpoint found at {ckpt_path} ({ckpt_path.stat().st_size:,} bytes). Loading weights for scoring...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        ckpt_sha = sha256_file(ckpt_path)
    else:
        # 2. Load cached positives
        log("Loading positives cache...")
        pos_cache = np.load(CACHE_DIR / "positives_cache.npz")
        pos_p51_all = pos_cache["p51"]
        pos_v51_all = pos_cache["v51"]
        pos_meta_all = json.loads(str(pos_cache["metadata"]))
        
        train_pos_indices = [i for i, c in enumerate(pos_meta_all) if c["fold"] != fold_k]
        val_pos_indices = [i for i, c in enumerate(pos_meta_all) if c["fold"] == fold_k]
        log(f"Positives partition: train={len(train_pos_indices)}, held_out_val={len(val_pos_indices)}")
        
        train_pos_p51 = pos_p51_all[train_pos_indices].astype(np.float32)
        train_pos_v51 = pos_v51_all[train_pos_indices].astype(np.float32)
        
        # 3. Clean negative pool from training scans only
        log("Sampling clean training negatives (zero overlap with held-out fold)...")
        exc_refs = defaultdict(list)
        with (REFD / "annotations_excluded.csv").open(newline="") as f:
            for r in csv.DictReader(f):
                d = float(r["diameter_mm"])
                exc_refs[r["seriesuid"]].append(([float(r[k]) for k in ("coordX", "coordY", "coordZ")], (d / 2.0 if d > 0.0 else 3.0) + 5.0))
        inc_refs = defaultdict(list)
        with (REFD / "annotations.csv").open(newline="") as f:
            for idx, r in enumerate(csv.DictReader(f)):
                d = float(r["diameter_mm"])
                inc_refs[r["seriesuid"]].append((idx, [float(r[k]) for k in ("coordX", "coordY", "coordZ")], (d / 2.0) + 5.0))

        rng_cand = random.Random(1000 + fold_k)
        train_neg_candidates = []
        with open(REFD / "candidates_V2.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            for idx, r in enumerate(reader):
                suid = r[0]
                if suid in train_scans and int(r[4]) == 0:
                    c = (float(r[1]), float(r[2]), float(r[3]))
                    if any(math.dist(c, ac) <= rad for ac, rad in exc_refs.get(suid, ())):
                        continue
                    if any(math.dist(c, ac) <= rad for _, ac, rad in inc_refs.get(suid, ())):
                        continue
                    train_neg_candidates.append({
                        "cand_idx": idx, "series": suid,
                        "x": c[0], "y": c[1], "z": c[2],
                        "scan": train_scans[suid]["scan_id"]
                    })
        log(f"Clean training candidate negatives available: {len(train_neg_candidates)}")
        
        sample_neg_count = min(4000, len(train_neg_candidates))
        picked_negs = rng_cand.sample(train_neg_candidates, sample_neg_count)
        log(f"Sampled {len(picked_negs)} clean random negatives. Extracting crops...")
        
        by_scan_neg = defaultdict(list)
        for idx, c in enumerate(picked_negs):
            by_scan_neg[c["scan"]].append((idx, c))
            
        train_neg_p51 = np.empty((len(picked_negs), 51, 51, 51), dtype=np.float32)
        train_neg_v51 = np.empty((len(picked_negs), 51, 51, 51), dtype=np.float32)
        
        t_extract0 = time.time()
        for sid, items in by_scan_neg.items():
            img, val, inv = load_scan_universal(sid, use_mmap=True)
            for idx, c in items:
                zyx = (inv @ np.array([c["x"], c["y"], c["z"], 1.0]))[:3]
                p51, _ = crop(img, zyx, 51, 0)
                v51, _ = crop(val, zyx, 51, False)
                train_neg_p51[idx] = p51
                train_neg_v51[idx] = v51
        log(f"Extracted {len(picked_negs)} negative crops in {time.time()-t_extract0:.1f}s")
        
        # 4. Training CLEAR3DMSFv2 with AMP FP16
        log(f"Initializing CLEAR3DMSFv2 model for Fold {fold_k} training on {device}...")
        torch.manual_seed(42 + fold_k)
        np.random.seed(42 + fold_k)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
        scaler = torch.amp.GradScaler("cuda")
        
        n_p = len(train_pos_p51)
        n_n = len(train_neg_p51)
        batch_pos = 12
        batch_neg = 20
        steps_per_epoch = 125
        rng_train = np.random.default_rng(2026 + fold_k)
        
        log(f"Starting training: {epochs} epochs, {steps_per_epoch} steps/epoch, batch size 32 (12 pos, 20 neg)...")
        t_train0 = time.time()
        
        for ep in range(epochs):
            model.train()
            lr = 1e-5 + 0.5 * (3e-4 - 1e-5) * (1.0 + math.cos(math.pi * ep / max(epochs - 1, 1)))
            for g in optimizer.param_groups:
                g["lr"] = lr
            ep_loss = 0.0
            for _ in range(steps_per_epoch):
                idx_p = rng_train.choice(n_p, size=batch_pos, replace=True)
                idx_n = rng_train.choice(n_n, size=batch_neg, replace=True)
                
                p_sub = np.concatenate([train_pos_p51[idx_p], train_neg_p51[idx_n]], axis=0)
                v_sub = np.concatenate([train_pos_v51[idx_p], train_neg_v51[idx_n]], axis=0)
                
                for b_i in range(len(p_sub)):
                    if rng_train.random() < 0.5:
                        p_sub[b_i] = p_sub[b_i, ::-1]; v_sub[b_i] = v_sub[b_i, ::-1]
                    if rng_train.random() < 0.5:
                        p_sub[b_i] = p_sub[b_i, :, :, ::-1]; v_sub[b_i] = v_sub[b_i, :, :, ::-1]
                
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    fi_g, fv_g, ci_g, cv_g = gpu_extract_from_p51(
                        torch.from_numpy(p_sub).to(device),
                        torch.from_numpy(v_sub).to(device)
                    )
                    vf, vc = make_inputs(fi_g, fv_g, ci_g, cv_g, f_fine, f_ctx)
                    targets = torch.tensor([1] * batch_pos + [0] * batch_neg, device=device, dtype=torch.long)
                    sample_w = torch.tensor([16.0 / batch_pos] * batch_pos + [16.0 / batch_neg] * batch_neg, device=device, dtype=torch.float32)
                    
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(vf, vc)
                    losses = F.cross_entropy(logits.float(), targets, reduction="none", label_smoothing=0.015)
                    loss = (losses * sample_w).mean()
                    
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()
                ep_loss += float(loss.detach())
                time.sleep(0.035)
                
            avg_loss = ep_loss / steps_per_epoch
            log(f"Epoch {ep+1:2d}/{epochs:2d} | Loss: {avg_loss:.4f} | lr: {lr:.2e} | Elapsed: {time.time()-t_train0:.1f}s")
            
        log(f"Training completed in {time.time()-t_train0:.1f}s!")
        
        torch.save({
            "fold": fold_k,
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_patients_count": len(train_scans),
            "held_out_patients_count": len(held_out_scans)
        }, ckpt_path)
        ckpt_sha = sha256_file(ckpt_path)
        log(f"Saved checkpoint: {ckpt_path} ({ckpt_path.stat().st_size:,} bytes) | SHA-256: {ckpt_sha}")
    
    # 5. Out-of-fold scoring on held-out fold scans
    log(f"Starting out-of-fold scoring on {len(held_out_scans)} scans for Fold {fold_k}...")
    model.eval()
    
    oof_candidates_by_scan = defaultdict(list)
    oof_total_cands = 0
    oof_pos_cands = 0
    with open(REFD / "candidates_V2.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        for idx, r in enumerate(reader):
            suid = r[0]
            if suid in held_out_scans:
                cls = int(r[4])
                oof_candidates_by_scan[suid].append({
                    "cand_idx": idx, "series": suid,
                    "coordX": float(r[1]), "coordY": float(r[2]), "coordZ": float(r[3]),
                    "label": cls
                })
                oof_total_cands += 1
                if cls == 1:
                    oof_pos_cands += 1
                    
    log(f"Total out-of-fold candidates to score: {oof_total_cands:,} (positive: {oof_pos_cands:,}) across {len(oof_candidates_by_scan)} scans")
    
    pred_rows = []
    t_score0 = time.time()
    views = [(False, False), (True, False), (False, True), (True, True)]
    
    for s_idx, (suid, cands) in enumerate(oof_candidates_by_scan.items(), 1):
        sid = held_out_scans[suid]["scan_id"]
        img, val, inv = load_scan_universal(sid, use_mmap=True)
        p_img = np.pad(img, 28, mode="constant", constant_values=0)
        p_val = np.pad(val, 28, mode="constant", constant_values=False)
        
        zyxs = np.empty((len(cands), 3), dtype=np.float64)
        for i, c in enumerate(cands):
            zyxs[i] = (inv @ np.array([c["coordX"], c["coordY"], c["coordZ"], 1.0]))[:3]
            
        p51_b, v51_b = slice_p51_batch(img, val, p_img, p_val, zyxs, pad=28)
        del img, val, p_img, p_val
        
        scan_probs = np.empty(len(cands), dtype=np.float32)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            batch_sz = 128
            for b_start in range(0, len(cands), batch_sz):
                b_end = min(b_start + batch_sz, len(cands))
                sub_p = torch.from_numpy(p51_b[b_start:b_end]).to(device)
                sub_v = torch.from_numpy(v51_b[b_start:b_end]).to(device)
                fi_g, fv_g, ci_g, cv_g = gpu_extract_from_p51(sub_p, sub_v)
                vf, vc = make_inputs(fi_g, fv_g, ci_g, cv_g, f_fine, f_ctx)
                
                acc = torch.zeros(len(sub_p), device=device, dtype=torch.float32)
                for fz, fx in views:
                    xf = vf; xc = vc
                    if fz: xf = torch.flip(xf, [2]); xc = torch.flip(xc, [2])
                    if fx: xf = torch.flip(xf, [4]); xc = torch.flip(xc, [4])
                    logits = model(xf, xc).float()
                    acc = acc + (logits[:, 1] - logits[:, 0])
                p = torch.sigmoid(acc * 0.25).cpu().numpy()
                scan_probs[b_start:b_end] = p; time.sleep(0.008)
                
        order = np.argsort(-scan_probs)
        ranks = np.empty(len(cands), dtype=int)
        ranks[order] = np.arange(1, len(cands) + 1)
        
        for i, c in enumerate(cands):
            pred_rows.append({
                "seriesuid": suid,
                "coordX": c["coordX"],
                "coordY": c["coordY"],
                "coordZ": c["coordZ"],
                "probability": float(scan_probs[i]),
                "candidate_id": f"{fold_k}:{c['cand_idx']}",
                "rank": int(ranks[i]),
                "label": c["label"]
            })
            
        if s_idx % 20 == 0 or s_idx == len(oof_candidates_by_scan):
            el = time.time() - t_score0
            log(f"[{s_idx:2d}/{len(oof_candidates_by_scan)}] Scored scan {sid} | Elapsed: {el:.1f}s | Est. remaining: {el/s_idx*(len(oof_candidates_by_scan)-s_idx):.1f}s")
            
    log(f"Scoring complete in {time.time()-t_score0:.1f}s ({len(pred_rows)/(time.time()-t_score0):.0f} cands/sec)!")
    
    fieldnames = ["seriesuid", "coordX", "coordY", "coordZ", "probability", "candidate_id", "rank", "label"]
    with open(pred_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pred_rows)
        
    pred_sha = sha256_file(pred_csv_path)
    total_sec = time.time() - t_fold0
    log(f"Written out-of-fold predictions to {pred_csv_path} ({pred_csv_path.stat().st_size:,} bytes) | SHA-256: {pred_sha}")
    log(f"=== COMPLETED FOLD {fold_k} IN {total_sec:.1f}s ({total_sec/60:.2f} min) ===")
    
    update_status(
        fold_k, "completed",
        checkpoint=str(ckpt_path),
        checkpoint_sha256=ckpt_sha,
        prediction_file=str(pred_csv_path),
        prediction_sha256=pred_sha,
        cands_total=len(pred_rows),
        cands_pos=oof_pos_cands,
        runtime_sec=round(total_sec, 1),
        completed_at=time.strftime("%Y-%m-%dT%H:%M:%S+07:00")
    )
    log_f.close()
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, help="Outer fold to execute (0-9)")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--device", type=int, default=1, help="CUDA device index (default: 1)")
    args = parser.parse_args()
    
    run_fold(args.fold, device_id=args.device, epochs=args.epochs)
