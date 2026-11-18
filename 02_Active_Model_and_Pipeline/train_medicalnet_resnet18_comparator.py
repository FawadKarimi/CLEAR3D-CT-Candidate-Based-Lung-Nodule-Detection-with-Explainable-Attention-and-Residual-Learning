"""Fine-tune Pretrained MedicalNet 3D ResNet-18 Comparator on Permitted CLEAR3D Fit Split
Evaluated on Identical Development Candidates (1,060 sampled dev & 20-scan dev set).
Strict guards:
- Fit split only: 3,936 candidates (984 pos, 2,952 neg; 1:3 ratio, 567 patients)
- Pretrained weights: MedicalNet ResNet-18 (resnet_18.pth, SHA256: 38b3a17444c5c18a795355a868e6a2d9ce9318054067f5373d337f55c0f61da3)
- Evaluated on dev split only: 1,060 candidates (265 pos, 795 neg; 142 patients) & 20 dev scans
- 0 access to test-fold0 or confirmation cohort (83 patients restricted, reference_release HOLD).
"""
import argparse, csv, hashlib, json, math, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

BASE = Path("/experiment-storage/chaidir/fawad-dsai")
ROOT = BASE / "CLEAR3D_CT project/02_Active_Model_and_Pipeline"
WORK = BASE / "CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64/source"
MEDNET_DIR = BASE / "CLEAR3D_CT project/05_Historical_Archives_and_Reviews/CLEAR3D_reviews_and_zips_oct13_14/CLEAR3D-MedicalNet-review-20261013-6b802e"
PREP = BASE / "CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-development-fold0-v1/prepared"
SPLIT_PATH = BASE / "CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-comparison-protocol-20261025/runs/phase-cf/split.json"
DEV_JSONL = BASE / "CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-comparison-protocol-20261025/runs/phase-cf/dev.jsonl"
SELECT = ROOT / "selection-001.json"
OUT_DIR = ROOT / "runs/comparator-medicalnet-resnet18-20261109"
CACHE_DIR = ROOT / "runs/cached_candidate_crops_20261109"

sys.path[:0] = [str(ROOT), str(WORK / "src"), str(WORK / "scripts"), str(MEDNET_DIR)]
from lidc3d.geometry import crop
import upstream.resnet as resnet
import loader

def require(ok: bool, msg: str):
    if not ok:
        raise ValueError(msg)

def pr_auc_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[order]
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(tp[-1], 1)
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))

def roc_auc_score_np(y_true, y_score):
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    return float((ranks[y_true == 1].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))

def calc_confusion_metrics(y_true, y_score, threshold=0.5):
    pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    total = len(y_true)
    
    acc = float((tp + tn) / total) if total else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) else 0.0
    f1p = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    f1n = float(2 * tn / (2 * tn + fp + fn)) if (2 * tn + fp + fn) else 0.0
    macro_f1 = float(0.5 * (f1p + f1n))
    
    return {
        "threshold": float(threshold),
        "total": total, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": acc, "recall": rec, "precision": prec,
        "specificity": spec, "f1_positive": f1p, "f1_negative": f1n,
        "macro_f1": macro_f1
    }

class MedicalNetClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.2):
        super().__init__()
        base = resnet.resnet18(sample_input_D=48, sample_input_H=48, sample_input_W=48, num_seg_classes=2)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.maxpool(h)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        feat = self.gap(h).flatten(1)
        feat = self.drop(feat)
        logits = self.fc(feat)
        return logits

def load_pretrained_medicalnet(model, weight_path):
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    stripped = {k[7:]: v for k, v in state_dict.items() if k.startswith("module.")}
    model_keys = set(model.state_dict().keys())
    matched = {}
    for k, v in stripped.items():
        if k in model_keys and v.shape == model.state_dict()[k].shape:
            matched[k] = v
    msg = model.load_state_dict(matched, strict=False)
    print(f"Loaded {len(matched)} pretrained encoder weights from {weight_path.name}. Missing in model: {len(msg.missing_keys)} (expected head/GAP)", flush=True)
    return len(matched)

def get_or_extract_crops(rows, desc="data", crop_size=48):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    crop_cache = CACHE_DIR / f"{desc}_crops_{len(rows)}.npy"
    label_cache = CACHE_DIR / f"{desc}_labels_{len(rows)}.npy"
    id_cache = CACHE_DIR / f"{desc}_ids_{len(rows)}.json"
    
    if crop_cache.exists() and label_cache.exists() and id_cache.exists():
        print(f"Loading cached {desc} crops from {crop_cache}...", flush=True)
        crops = np.load(crop_cache)
        labels = np.load(label_cache)
        ids = json.loads(id_cache.read_text())
        print(f"Loaded {len(crops)} cached crops ({np.sum(labels==1)} pos, {np.sum(labels==0)} neg) in 0.1s", flush=True)
        return crops, labels, ids
        
    groups = defaultdict(list)
    for r in rows:
        groups[r["scan"]].append(r)
    
    crops = np.empty((len(rows), 1, crop_size, crop_size, crop_size), dtype=np.float32)
    labels = np.empty(len(rows), dtype=np.int64)
    row_ids = [r["id"] for r in rows]
    idx_map = {r["id"]: i for i, r in enumerate(rows)}
    
    t0 = time.time()
    total_groups = len(groups)
    for sid_idx, (sid, cands) in enumerate(groups.items(), 1):
        if sid_idx % 100 == 0 or sid_idx == total_groups:
            print(f"[{desc}] Extraction progress: {sid_idx}/{total_groups} scans elapsed={time.time()-t0:.1f}s", flush=True)
        d = PREP / sid
        geo = json.loads((d / "geometry.json").read_text())
        aff = np.asarray(geo["affine"], dtype=np.float64)
        inv = np.linalg.inv(aff)
        img = np.load(d / "image.npy", mmap_mode="r")
        val = np.load(d / "valid.npy", mmap_mode="r")
        
        for c in cands:
            zyx = (inv @ np.array([c["x"], c["y"], c["z"], 1.0]))[:3]
            p, _ = crop(img, zyx, crop_size, 0.0)
            v, _ = crop(val, zyx, crop_size, False)
            
            v_mask = v > 0.5
            if v_mask.any():
                vals = p[v_mask]
                m = float(vals.mean())
                s = max(float(vals.std()), 1e-6)
                norm_p = np.where(v_mask, (p - m) / s, 0.0).astype(np.float32)
            else:
                norm_p = np.zeros_like(p, dtype=np.float32)
            
            target_idx = idx_map[c["id"]]
            crops[target_idx, 0] = norm_p
            labels[target_idx] = int(c["label"])
            
    print(f"Extracted {len(rows)} crops ({np.sum(labels==1)} pos, {np.sum(labels==0)} neg) from {len(groups)} scans in {time.time()-t0:.2f}s", flush=True)
    np.save(crop_cache, crops)
    np.save(label_cache, labels)
    id_cache.write_text(json.dumps(row_ids))
    return crops, labels, row_ids

class CandidatePatchDataset(Dataset):
    def __init__(self, crops, labels, augment=False, seed=42):
        self.crops = crops
        self.labels = labels
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.crops[idx]
        y = self.labels[idx]
        
        if self.augment and y == 1:
            k = int(self.rng.integers(0, 4))
            if k > 0:
                x = np.rot90(x, k, axes=(2, 3)).copy()
            if self.rng.random() < 0.5:
                x = x[:, ::-1, :, :].copy()
            if self.rng.random() < 0.5:
                x = x[:, :, :, ::-1].copy()
                
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

def main():
    print("=" * 80, flush=True)
    print("TRAINING COMPARATOR: MEDICALNET 3D RESNET-18 ON PERMITTED CLEAR3D FIT SPLIT", flush=True)
    print("=" * 80, flush=True)
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(device)})", flush=True)
    
    sel = json.loads(SELECT.read_text())
    require(sel["reference_release"] == "HOLD", "reference_release must be HOLD")
    fit_rows = sel["fit"]["selected"]
    dev_rows = sel["dev"]["selected"]
    require(len(fit_rows) == 3936, f"Expected 3,936 fit rows, got {len(fit_rows)}")
    require(len(dev_rows) == 1060, f"Expected 1,060 dev rows, got {len(dev_rows)}")
    
    mednet_weight = MEDNET_DIR / "weights/resnet_18.pth"
    require(mednet_weight.exists(), "MedicalNet weight file not found")
    w_hash = hashlib.sha256(mednet_weight.read_bytes()).hexdigest()
    expected_w_hash = "38b3a17444c5c18a795355a868e6a2d9ce9318054067f5373d337f55c0f61da3"
    require(w_hash == expected_w_hash, f"Weight hash mismatch: {w_hash}")
    print(f"Verified MedicalNet ResNet-18 weights: SHA256={w_hash}", flush=True)
    
    print("Retrieving or extracting fit candidate crops...", flush=True)
    fit_crops, fit_labels, fit_ids = get_or_extract_crops(fit_rows, desc="fit")
    print("Retrieving or extracting dev candidate crops...", flush=True)
    dev_crops, dev_labels, dev_ids = get_or_extract_crops(dev_rows, desc="dev")
    
    model = MedicalNetClassifier(num_classes=2, dropout=0.2).to(device)
    matched_count = load_pretrained_medicalnet(model, mednet_weight)
    require(matched_count == 102, f"Expected 102 matched weights, got {matched_count}")
    
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    head_params = [p for n, p in model.named_parameters() if n.startswith("fc.")]
    
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": 1e-4, "weight_decay": 1e-4},
        {"params": head_params, "lr": 5e-4, "weight_decay": 1e-4}
    ])
    
    epochs = 6
    batch_size = 32
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0], device=device))
    
    train_ds = CandidatePatchDataset(fit_crops, fit_labels, augment=True, seed=42)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    
    dev_ds = CandidatePatchDataset(dev_crops, dev_labels, augment=False)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUT_DIR / "train.log"
    log_fp = open(log_file, "w")
    
    def log(msg):
        print(msg, flush=True)
        log_fp.write(msg + chr(10))
        log_fp.flush()
        
    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    log(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    log(f"Starting fine-tuning: {epochs} epochs, batch_size={batch_size}, fit_samples={len(fit_crops)}")
    
    t_start = time.time()
    best_dev_prauc = -1.0
    best_epoch = -1
    
    for epoch in range(1, epochs + 1):
        t_ep_start = time.time()
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += float(loss.item())
            train_steps += 1
            
        scheduler.step()
        avg_train_loss = train_loss / max(train_steps, 1)
        
        model.eval()
        dev_scores = []
        with torch.no_grad():
            for bx, by in dev_loader:
                bx = bx.to(device)
                logits = model(bx)
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                dev_scores.extend(probs)
                
        dev_scores = np.array(dev_scores, dtype=np.float32)
        dev_prauc = pr_auc_score(dev_labels, dev_scores)
        dev_rocauc = roc_auc_score_np(dev_labels, dev_scores)
        c_metrics = calc_confusion_metrics(dev_labels, dev_scores, threshold=0.5)
        
        ep_time = time.time() - t_ep_start
        log(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_train_loss:.4f} | Dev PR-AUC: {dev_prauc:.4f} | Dev ROC-AUC: {dev_rocauc:.4f} | Dev F1-pos: {c_metrics['f1_positive']:.4f} | Time: {ep_time:.1f}s")
            
        if dev_prauc > best_dev_prauc:
            best_dev_prauc = dev_prauc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dev_prauc": dev_prauc,
                "dev_rocauc": dev_rocauc,
                "dev_metrics": c_metrics,
                "dev_scores": dev_scores.tolist(),
                "dev_ids": dev_ids,
                "dev_labels": dev_labels.tolist()
            }, OUT_DIR / "best_checkpoint.pt")
            
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "final_dev_scores": dev_scores.tolist(),
        "dev_ids": dev_ids,
        "dev_labels": dev_labels.tolist()
    }, OUT_DIR / "checkpoint.pt")
    
    total_train_time = time.time() - t_start
    log(f"Training completed in {total_train_time:.1f}s. Best epoch: {best_epoch} (PR-AUC: {best_dev_prauc:.4f})")
    
    best_ck = torch.load(OUT_DIR / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ck["model_state_dict"])
    model.eval()
    
    t_lat_start = time.time()
    all_scores = []
    with torch.no_grad():
        for bx, by in dev_loader:
            bx = bx.to(device)
            logits = model(bx)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_scores.extend(probs)
    lat_total = time.time() - t_lat_start
    ms_per_cand = (lat_total / len(dev_labels)) * 1000.0
    all_scores = np.array(all_scores, dtype=np.float32)
    
    prauc_1060 = pr_auc_score(dev_labels, all_scores)
    rocauc_1060 = roc_auc_score_np(dev_labels, all_scores)
    m_05 = calc_confusion_metrics(dev_labels, all_scores, threshold=0.5)
    
    best_th = 0.5
    best_f1 = -1.0
    for th in np.linspace(0.1, 0.9, 81):
        f1_cur = calc_confusion_metrics(dev_labels, all_scores, threshold=th)["f1_positive"]
        if f1_cur > best_f1:
            best_f1 = f1_cur
            best_th = float(th)
    m_best = calc_confusion_metrics(dev_labels, all_scores, threshold=best_th)
    
    log(chr(10) + "=" * 50)
    log("FINAL MEDICALNET RESNET-18 DEV RESULTS (1,060 Candidates):")
    log(f"  PR-AUC:        {prauc_1060:.4f}")
    log(f"  ROC-AUC:       {rocauc_1060:.4f}")
    log(f"  Threshold 0.5: Precision={m_05['precision']:.4f}, Recall={m_05['recall']:.4f}, Specificity={m_05['specificity']:.4f}, Pos-F1={m_05['f1_positive']:.4f}, Macro-F1={m_05['macro_f1']:.4f}")
    log(f"  Best Threshold ({best_th:.2f}): Precision={m_best['precision']:.4f}, Recall={m_best['recall']:.4f}, Specificity={m_best['specificity']:.4f}, Pos-F1={m_best['f1_positive']:.4f}, Macro-F1={m_best['macro_f1']:.4f}")
    log(f"  Inference Latency: {ms_per_cand:.2f} ms/cand ({len(dev_labels)/lat_total:.1f} cand/s)")
    log(f"  Parameters:    {sum(p.numel() for p in model.parameters()):,}")
    
    with open(OUT_DIR / "dev_predictions_1060.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "probability"])
        for cid, lbl, prob in zip(dev_ids, dev_labels, all_scores):
            w.writerow([cid, lbl, f"{prob:.6f}"])
            
    print(chr(10) + "Running broader evaluation on 20 patient-separated dev scans (18,678 candidates)...", flush=True)
    split = loader.get_split(SPLIT_PATH)
    rows_all = loader.candidate_rows(DEV_JSONL, "dev", split["dev"])
    scan_groups = defaultdict(list)
    for r in rows_all:
        scan_groups[r["scan"]].append(r)
    sorted_scans = sorted(scan_groups.items(), key=lambda x: -sum(1 for r in x[1] if r["label"] == 1))[:20]
    
    scan_20_scores = []
    scan_20_labels = []
    scan_20_ids = []
    t_20_start = time.time()
    
    for s_i, (sid, rr) in enumerate(sorted_scans, 1):
        if s_i % 5 == 0 or s_i == len(sorted_scans):
            print(f"[20 scans] Evaluated {s_i}/{len(sorted_scans)} scans elapsed={time.time()-t_20_start:.1f}s", flush=True)
        d = PREP / sid
        geo = json.loads((d / "geometry.json").read_text())
        aff = np.asarray(geo["affine"], dtype=np.float64)
        inv = np.linalg.inv(aff)
        img = np.load(d / "image.npy", mmap_mode="r")
        val = np.load(d / "valid.npy", mmap_mode="r")
        
        c_batch = np.empty((len(rr), 1, 48, 48, 48), dtype=np.float32)
        for k_i, c in enumerate(rr):
            zyx = (inv @ np.array([c["x"], c["y"], c["z"], 1.0]))[:3]
            p, _ = crop(img, zyx, 48, 0.0)
            v, _ = crop(val, zyx, 48, False)
            v_mask = v > 0.5
            if v_mask.any():
                vals = p[v_mask]
                m = float(vals.mean())
                s = max(float(vals.std()), 1e-6)
                c_batch[k_i, 0] = np.where(v_mask, (p - m) / s, 0.0)
            else:
                c_batch[k_i, 0] = 0.0
                
        with torch.no_grad():
            for b_idx in range(0, len(rr), 64):
                bx = torch.from_numpy(c_batch[b_idx:b_idx+64]).to(device)
                logits = model(bx)
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                scan_20_scores.extend(probs)
                scan_20_labels.extend([int(r["label"]) for r in rr[b_idx:b_idx+64]])
                scan_20_ids.extend([r["id"] for r in rr[b_idx:b_idx+64]])
                
    t_20_total = time.time() - t_20_start
    scan_20_scores = np.array(scan_20_scores, dtype=np.float32)
    scan_20_labels = np.array(scan_20_labels, dtype=np.int64)
    prauc_20 = pr_auc_score(scan_20_labels, scan_20_scores)
    rocauc_20 = roc_auc_score_np(scan_20_labels, scan_20_scores)
    m_20_05 = calc_confusion_metrics(scan_20_labels, scan_20_scores, threshold=0.5)
    
    log(chr(10) + "=" * 50)
    log(f"MEDICALNET RESNET-18 BROADER EVALUATION (20 Scans, {len(scan_20_labels)} Candidates, {sum(scan_20_labels==1)} pos):")
    log(f"  PR-AUC:        {prauc_20:.4f}")
    log(f"  ROC-AUC:       {rocauc_20:.4f}")
    log(f"  Threshold 0.5: Precision={m_20_05['precision']:.4f}, Recall={m_20_05['recall']:.4f}, Specificity={m_20_05['specificity']:.4f}, Pos-F1={m_20_05['f1_positive']:.4f}, Macro-F1={m_20_05['macro_f1']:.4f}")
    log(f"  Evaluation Time: {t_20_total:.1f}s ({(t_20_total/len(scan_20_labels))*1000:.2f} ms/cand)")
    
    summary_results = {
        "model_name": "MedicalNet-ResNet18-FineTuned",
        "checkpoint_sha256": hashlib.sha256((OUT_DIR / "checkpoint.pt").read_bytes()).hexdigest(),
        "best_checkpoint_sha256": hashlib.sha256((OUT_DIR / "best_checkpoint.pt").read_bytes()).hexdigest(),
        "pretrained_weights_sha256": w_hash,
        "training_epochs": epochs,
        "batch_size": batch_size,
        "training_time_seconds": total_train_time,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "latency_ms_per_cand": ms_per_cand,
        "dev_1060": {
            "total": len(dev_labels),
            "positives": int((dev_labels == 1).sum()),
            "negatives": int((dev_labels == 0).sum()),
            "pr_auc": prauc_1060,
            "roc_auc": rocauc_1060,
            "threshold_0_5": m_05,
            "best_threshold": {"threshold": best_th, **m_best}
        },
        "dev_20scans": {
            "total": len(scan_20_labels),
            "positives": int((scan_20_labels == 1).sum()),
            "negatives": int((scan_20_labels == 0).sum()),
            "pr_auc": prauc_20,
            "roc_auc": rocauc_20,
            "threshold_0_5": m_20_05
        }
    }
    with open(OUT_DIR / "medicalnet_resnet18_evaluation_summary.json", "w") as f:
        json.dump(summary_results, f, indent=2)
        
    log_fp.close()
    print(chr(10) + "MedicalNet comparator training and evaluation finished successfully!", flush=True)

if __name__ == "__main__":
    main()
