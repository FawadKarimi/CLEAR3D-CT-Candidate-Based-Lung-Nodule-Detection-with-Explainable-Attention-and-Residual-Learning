"""Train and Evaluate CLEAR3D-HighRes (Reduced Early Downsampling in Scale 1)
Hypothesis: Sub-6mm nodules have diameter 3-6mm (3-6 voxels). Premature 2x maxpooling
decimates them to 1-3 voxels in Scale 1, losing subtle edge, texture and spiculation cues.
CLEAR3D-HighRes replaces early maxpool with a stride-1 dilated SE-ResBlock3D (dilation=2),
preserving full 20^3 spatial resolution before pooling down to 10^3 and 5^3.
Evaluated under matched 4-view TTA on identical dev candidates (1,060 dev & 20 dev scans).
"""
import argparse, csv, hashlib, json, math, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BASE = Path("/experiment-storage/chaidir/fawad-dsai")
ROOT = BASE / "CLEAR3D_CT project/02_Active_Model_and_Pipeline"
WORK = BASE / "CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64/source"
PREP = BASE / "CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-development-fold0-v1/prepared"
SPLIT_PATH = BASE / "CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-comparison-protocol-20261025/runs/phase-cf/split.json"
DEV_JSONL = BASE / "CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-comparison-protocol-20261025/runs/phase-cf/dev.jsonl"
SELECT = ROOT / "selection-001.json"
OUT_DIR = ROOT / "runs/improvement-clear3d-highres-20261109"
CACHE_DIR = ROOT / "runs/cached_clear3d_banks_20261109"

sys.path[:0] = [str(ROOT), str(WORK / "src"), str(WORK / "scripts")]
from lidc3d.geometry import crop
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

def maxpool2(x: torch.Tensor) -> torch.Tensor:
    b, c, d, h, w = x.shape
    return x.reshape(b, c, d // 2, 2, h // 2, 2, w // 2, 2).amax((3, 5, 7))

class SE3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean((2, 3, 4))
        w = torch.sigmoid(self.fc2(F.silu(self.fc1(s))))[:, :, None, None, None]
        return x * w

class SEResBlock3D(nn.Module):
    def __init__(self, cin: int, cout: int, dilation: int = 1):
        super().__init__()
        g = 8 if cout % 8 == 0 else 4
        p = dilation
        self.conv1 = nn.Conv3d(cin, cout, 3, padding=p, dilation=dilation, bias=False)
        self.gn1 = nn.GroupNorm(g, cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=p, dilation=dilation, bias=False)
        self.gn2 = nn.GroupNorm(g, cout)
        self.se = SE3D(cout)
        self.short = nn.Identity() if cin == cout else nn.Sequential(
            nn.Conv3d(cin, cout, 1, bias=False), nn.GroupNorm(g, cout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.short(x)
        h = F.silu(self.gn1(self.conv1(x)))
        h = self.se(self.gn2(self.conv2(h)))
        return F.silu(h + r)

class CLEAR3DHighRes(nn.Module):
    def __init__(self, in_channels: int = 3, dim: int = 160, dropout: float = 0.25):
        super().__init__()
        # Scale 1 (High-Res Focal Core):
        # 21^3 -> Conv(k=2) -> 20^3 -> SEResBlock3D(dil=2, keeps 20^3!) -> maxpool2(10^3) -> maxpool2(5^3)
        self.f1_stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, 2, padding=0, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            SEResBlock3D(32, 32),
        )
        self.f1_highres = SEResBlock3D(32, 32, dilation=2)  # High-res feature preservation at 20^3
        self.f1_s2 = SEResBlock3D(32, 64)                  # At 10^3
        self.f1_s3 = SEResBlock3D(64, 128)                 # At 5^3
        self.f1_proj = nn.Sequential(nn.Linear(32 + 32 + 64 + 128 + 128, dim), nn.LayerNorm(dim), nn.SiLU())

        # Scale 2 (Medium 33^3): Conv(k=2) -> 32^3 -> 16^3 -> 8^3 -> 4^3
        self.f2_stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, 2, padding=0, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.f2_s1 = SEResBlock3D(32, 64)   # 16^3
        self.f2_s2 = SEResBlock3D(64, 96)   # 8^3
        self.f2_s3 = SEResBlock3D(96, 128)  # 4^3
        self.f2_proj = nn.Sequential(nn.Linear(64 + 96 + 128 + 128, dim), nn.LayerNorm(dim), nn.SiLU())

        # Scale 3 (Context 25^3 @ 2mm = 50mm FOV): Conv(k=2) -> 24^3 -> 12^3 -> 6^3 -> 3^3
        self.f3_stem = nn.Sequential(
            nn.Conv3d(in_channels, 24, 2, padding=0, bias=False),
            nn.GroupNorm(8, 24),
            nn.SiLU(),
        )
        self.f3_s1 = SEResBlock3D(24, 48)   # 12^3
        self.f3_s2 = SEResBlock3D(48, 96)   # 6^3
        self.f3_s3 = SEResBlock3D(96, 128)  # 3^3
        self.f3_proj = nn.Sequential(nn.Linear(96 + 128 + 128, dim), nn.LayerNorm(dim), nn.SiLU())

        self.scale_emb = nn.Parameter(torch.zeros(1, 3, dim))
        nn.init.normal_(self.scale_emb, std=0.02)
        self.ln_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=4, dropout=0.1, batch_first=True)
        self.ln_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Dropout(0.1), nn.Linear(dim * 2, dim))

        self.head = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 2),
        )

    def forward(self, v_fine: torch.Tensor, v_ctx: torch.Tensor) -> torch.Tensor:
        # Scale 1: central 21^3 around index 16 -> [6:27]
        x1 = v_fine[:, :, 6:27, 6:27, 6:27]
        h1_1 = self.f1_stem(x1)                     # (B, 32, 20, 20, 20)
        h1_hr = self.f1_highres(h1_1)               # (B, 32, 20, 20, 20) high-res preserved!
        h1_2 = self.f1_s2(maxpool2(h1_hr))          # (B, 64, 10, 10, 10)
        h1_3 = self.f1_s3(maxpool2(h1_2))           # (B, 128, 5, 5, 5)
        t1 = self.f1_proj(torch.cat([
            h1_hr[:, :, 6:14, 6:14, 6:14].mean((2, 3, 4)),  # inner 8mm high-res core
            h1_hr.mean((2, 3, 4)),
            h1_2.mean((2, 3, 4)),
            h1_3.mean((2, 3, 4)),
            h1_3.amax((2, 3, 4)),
        ], dim=1))

        # Scale 2: 33^3 -> 32^3 -> 16^3 -> 8^3 -> 4^3
        h2_0 = self.f2_stem(v_fine)
        h2_1 = self.f2_s1(maxpool2(h2_0))
        h2_2 = self.f2_s2(maxpool2(h2_1))
        h2_3 = self.f2_s3(maxpool2(h2_2))
        t2 = self.f2_proj(torch.cat([
            h2_1[:, :, 4:12, 4:12, 4:12].mean((2, 3, 4)),
            h2_2.mean((2, 3, 4)),
            h2_3.mean((2, 3, 4)),
            h2_3.amax((2, 3, 4)),
        ], dim=1))

        # Scale 3: 25^3 -> 24^3 -> 12^3 -> 6^3 -> 3^3
        h3_0 = self.f3_stem(v_ctx)
        h3_1 = self.f3_s1(maxpool2(h3_0))
        h3_2 = self.f3_s2(maxpool2(h3_1))
        h3_3 = self.f3_s3(maxpool2(h3_2))
        t3 = self.f3_proj(torch.cat([
            h3_2.mean((2, 3, 4)),
            h3_3.mean((2, 3, 4)),
            h3_3.amax((2, 3, 4)),
        ], dim=1))

        tokens = torch.stack([t1, t2, t3], dim=1) + self.scale_emb
        normed = self.ln_attn(tokens)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.ff(self.ln_ff(tokens))
        return self.head(tokens.flatten(1))

def build_focus_grids(device: torch.device):
    g33 = torch.arange(33, device=device, dtype=torch.float32) - 16.0
    zz, yy, xx = torch.meshgrid(g33, g33, g33, indexing="ij")
    f_fine = torch.exp(-(zz * zz + yy * yy + xx * xx) / (2.0 * 4.5 * 4.5))[None, None]

    g25 = (torch.arange(25, device=device, dtype=torch.float32) - 12.0) * 2.0
    zz2, yy2, xx2 = torch.meshgrid(g25, g25, g25, indexing="ij")
    f_ctx = torch.exp(-(zz2 * zz2 + yy2 * yy2 + xx2 * xx2) / (2.0 * 10.0 * 10.0))[None, None]
    return f_fine, f_ctx

def make_inputs(fine_img: torch.Tensor, fine_val: torch.Tensor,
                ctx_img: torch.Tensor, ctx_val: torch.Tensor,
                f_fine: torch.Tensor, f_ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b = fine_img.shape[0]
    fval = fine_val.bool()
    cval = ctx_val.bool()

    cnt = fval.flatten(1).sum(1, keepdim=True).clamp_min(1).float()
    mean = (fine_img * fval).flatten(1).sum(1, keepdim=True) / cnt
    var = (((fine_img - mean[:, :, None, None]) ** 2) * fval).flatten(1).sum(1, keepdim=True) / cnt
    std = var.sqrt().clamp_min(1e-3)
    mean4 = mean[:, :, None, None]
    std4 = std[:, :, None, None]

    f_hu = torch.where(fval, fine_img * 2.0 - 1.0, torch.full_like(fine_img, -1.0))
    f_z = torch.where(fval, ((fine_img - mean4) / std4).clamp(-5.0, 5.0), torch.zeros_like(fine_img))
    v_fine = torch.cat([f_hu[:, None], f_z[:, None], f_fine.expand(b, -1, -1, -1, -1)], dim=1)

    c_hu = torch.where(cval, ctx_img * 2.0 - 1.0, torch.full_like(ctx_img, -1.0))
    c_z = torch.where(cval, ((ctx_img - mean4) / std4).clamp(-5.0, 5.0), torch.zeros_like(ctx_img))
    v_ctx = torch.cat([c_hu[:, None], c_z[:, None], f_ctx.expand(b, -1, -1, -1, -1)], dim=1)
    return v_fine, v_ctx

def pack_u16(img: np.ndarray, val: np.ndarray) -> np.ndarray:
    q = np.clip(np.rint(img * 32767.0), 0.0, 32767.0).astype(np.uint16)
    return q | (val.astype(np.uint16) << np.uint16(15))

def unpack_u16(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    val = (u >> np.uint16(15)).astype(np.float32)
    img = (u & np.uint16(0x7FFF)).astype(np.float32) * (1.0 / 32767.0)
    return img, val

def downsample_55_to_27(img55: np.ndarray, val55: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = img55 * val55.astype(np.float32)
    m = val55.astype(np.float32)
    for ax in (0, 1, 2):
        s0 = [slice(None)] * 3; s0[ax] = slice(0, -2, 2)
        s1 = [slice(None)] * 3; s1[ax] = slice(1, -1, 2)
        s2 = [slice(None)] * 3; s2[ax] = slice(2, None, 2)
        w = 0.25 * w[tuple(s0)] + 0.5 * w[tuple(s1)] + 0.25 * w[tuple(s2)]
        m = 0.25 * m[tuple(s0)] + 0.5 * m[tuple(s1)] + 0.25 * m[tuple(s2)]
    cval = m >= 0.5
    cimg = np.where(cval, w / np.maximum(m, 1e-6), 0.0)
    return np.clip(cimg, 0.0, 1.0), cval

def extract_pair_u16(img_vol: np.ndarray, val_vol: np.ndarray, zyx) -> tuple[np.ndarray, np.ndarray]:
    p55, _ = crop(img_vol, zyx, 55, 0)
    v55, _ = crop(val_vol, zyx, 55, False)
    f37 = pack_u16(p55[9:46, 9:46, 9:46], v55[9:46, 9:46, 9:46])
    cimg27, cval27 = downsample_55_to_27(p55, v55)
    c27 = pack_u16(cimg27, cval27)
    return f37, c27

def get_or_extract_clear3d_banks(rows, desc="fit"):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f37_cache = CACHE_DIR / f"{desc}_f37_{len(rows)}.npy"
    c27_cache = CACHE_DIR / f"{desc}_c27_{len(rows)}.npy"
    lbl_cache = CACHE_DIR / f"{desc}_lbl_{len(rows)}.npy"
    id_cache = CACHE_DIR / f"{desc}_ids_{len(rows)}.json"
    
    if f37_cache.exists() and c27_cache.exists() and lbl_cache.exists() and id_cache.exists():
        print(f"Loading cached CLEAR3D {desc} banks...", flush=True)
        f37 = np.load(f37_cache)
        c27 = np.load(c27_cache)
        lbl = np.load(lbl_cache)
        ids = json.loads(id_cache.read_text())
        print(f"Loaded {len(f37)} cached candidate pairs in 0.2s", flush=True)
        return f37, c27, lbl, ids
        
    groups = defaultdict(list)
    for r in rows:
        groups[r["scan"]].append(r)
        
    f37_bank = np.empty((len(rows), 37, 37, 37), dtype=np.uint16)
    c27_bank = np.empty((len(rows), 27, 27, 27), dtype=np.uint16)
    labels = np.empty(len(rows), dtype=np.int64)
    ids = [r["id"] for r in rows]
    idx_map = {r["id"]: i for i, r in enumerate(rows)}
    
    t0 = time.time()
    for sid_idx, (sid, cands) in enumerate(groups.items(), 1):
        if sid_idx % 100 == 0 or sid_idx == len(groups):
            print(f"[{desc}] Extraction: {sid_idx}/{len(groups)} scans elapsed={time.time()-t0:.1f}s", flush=True)
        d = PREP / sid
        geo = json.loads((d / "geometry.json").read_text())
        aff = np.asarray(geo["affine"], dtype=np.float64)
        inv = np.linalg.inv(aff)
        img = np.load(d / "image.npy", mmap_mode="r")
        val = np.load(d / "valid.npy", mmap_mode="r")
        for c in cands:
            zyx = (inv @ np.array([c["x"], c["y"], c["z"], 1.0]))[:3]
            f37, c27 = extract_pair_u16(img, val, zyx)
            t_idx = idx_map[c["id"]]
            f37_bank[t_idx] = f37
            c27_bank[t_idx] = c27
            labels[t_idx] = int(c["label"])
            
    print(f"Extracted {len(rows)} candidate pairs in {time.time()-t0:.2f}s", flush=True)
    np.save(f37_cache, f37_bank)
    np.save(c27_cache, c27_bank)
    np.save(lbl_cache, labels)
    id_cache.write_text(json.dumps(ids))
    return f37_bank, c27_bank, labels, ids

def sample_augmented_batch(f37_bank: np.ndarray, c27_bank: np.ndarray, indices: np.ndarray, rng: np.random.Generator):
    b = len(indices)
    fi = np.empty((b, 33, 33, 33), dtype=np.float32)
    fv = np.empty((b, 33, 33, 33), dtype=np.float32)
    ci = np.empty((b, 25, 25, 25), dtype=np.float32)
    cv = np.empty((b, 25, 25, 25), dtype=np.float32)

    do_shift = rng.random(b) < 0.50
    dzs = np.where(do_shift, rng.integers(-1, 2, size=b), 0)
    dys = np.where(do_shift, rng.integers(-1, 2, size=b), 0)
    dxs = np.where(do_shift, rng.integers(-1, 2, size=b), 0)
    ks = rng.integers(0, 4, size=b)
    flip_z = rng.random(b) < 0.5
    flip_x = rng.random(b) < 0.5

    for j, idx in enumerate(indices):
        dz, dy, dx = int(dzs[j]), int(dys[j]), int(dxs[j])
        f_im, f_va = unpack_u16(f37_bank[idx, 2 + dz:35 + dz, 2 + dy:35 + dy, 2 + dx:35 + dx])
        c_im, c_va = unpack_u16(c27_bank[idx, 1:26, 1:26, 1:26])

        k = int(ks[j])
        if k:
            f_im = np.rot90(f_im, k, (1, 2)); f_va = np.rot90(f_va, k, (1, 2))
            c_im = np.rot90(c_im, k, (1, 2)); c_va = np.rot90(c_va, k, (1, 2))
        if flip_z[j]:
            f_im = f_im[::-1]; f_va = f_va[::-1]
            c_im = c_im[::-1]; c_va = c_va[::-1]
        if flip_x[j]:
            f_im = f_im[:, :, ::-1]; f_va = f_va[:, :, ::-1]
            c_im = c_im[:, :, ::-1]; c_va = c_va[:, :, ::-1]

        fi[j] = f_im; fv[j] = f_va
        ci[j] = c_im; cv[j] = c_va
    return fi, fv, ci, cv

def eval_tta4(model, f37_bank, c27_bank, indices, device, f_fine, f_ctx):
    model.eval()
    scores = np.empty(len(indices), dtype=np.float32)
    views = [(False, False), (True, False), (False, True), (True, True)]
    
    with torch.no_grad():
        for s in range(0, len(indices), 64):
            sub = indices[s:s + 64]
            b_cnt = len(sub)
            f_im = np.empty((b_cnt, 33, 33, 33), dtype=np.float32)
            f_va = np.empty((b_cnt, 33, 33, 33), dtype=np.float32)
            c_im = np.empty((b_cnt, 25, 25, 25), dtype=np.float32)
            c_va = np.empty((b_cnt, 25, 25, 25), dtype=np.float32)
            for j, idx in enumerate(sub):
                f_im[j], f_va[j] = unpack_u16(f37_bank[idx, 2:35, 2:35, 2:35])
                c_im[j], c_va[j] = unpack_u16(c27_bank[idx, 1:26, 1:26, 1:26])
                
            vf, vc = make_inputs(
                torch.from_numpy(f_im).to(device), torch.from_numpy(f_va).to(device),
                torch.from_numpy(c_im).to(device), torch.from_numpy(c_va).to(device),
                f_fine, f_ctx
            )
            
            # 4-view average
            acc = torch.zeros(b_cnt, device=device, dtype=torch.float32)
            for fz, fx in views:
                x_f = vf
                x_c = vc
                if fz:
                    x_f = torch.flip(x_f, [2])
                    x_c = torch.flip(x_c, [2])
                if fx:
                    x_f = torch.flip(x_f, [4])
                    x_c = torch.flip(x_c, [4])
                logits = model(x_f, x_c).float()
                acc = acc + (logits[:, 1] - logits[:, 0])
                
            avg_log_odds = (acc * 0.25).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-avg_log_odds))
            scores[s:s + 64] = probs
            
    return scores

def main():
    print("=" * 80, flush=True)
    print("TRAINING CLEAR3D-HIGHRES IMPROVEMENT (REDUCED EARLY DOWNSAMPLING IN SCALE 1)", flush=True)
    print("=" * 80, flush=True)
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(device)})", flush=True)
    f_fine, f_ctx = build_focus_grids(device)
    
    sel = json.loads(SELECT.read_text())
    require(sel["reference_release"] == "HOLD", "reference_release must be HOLD")
    fit_rows = sel["fit"]["selected"]
    dev_rows = sel["dev"]["selected"]
    require(len(fit_rows) == 3936, f"Expected 3,936 fit rows, got {len(fit_rows)}")
    require(len(dev_rows) == 1060, f"Expected 1,060 dev rows, got {len(dev_rows)}")
    
    print("Retrieving or extracting fit CLEAR3D banks...", flush=True)
    fit_f37, fit_c27, fit_lbl, fit_ids = get_or_extract_clear3d_banks(fit_rows, desc="fit")
    print("Retrieving or extracting dev CLEAR3D banks...", flush=True)
    dev_f37, dev_c27, dev_lbl, dev_ids = get_or_extract_clear3d_banks(dev_rows, desc="dev")
    
    model = CLEAR3DHighRes(in_channels=3, dim=160, dropout=0.25).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    epochs = 6
    batch_size = 32
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0], device=device))
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUT_DIR / "train.log"
    log_fp = open(log_file, "w")
    
    def log(msg):
        print(msg, flush=True)
        log_fp.write(msg + chr(10))
        log_fp.flush()
        
    log(f"CLEAR3D-HighRes parameters: {sum(p.numel() for p in model.parameters()):,}")
    log(f"Starting training: {epochs} epochs, batch_size={batch_size}, fit_samples={len(fit_lbl)}")
    
    rng = np.random.default_rng(42)
    pos_idx = np.where(fit_lbl == 1)[0]
    neg_idx = np.where(fit_lbl == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    
    t_start = time.time()
    best_dev_prauc = -1.0
    best_epoch = -1
    
    dev_all_idx = np.arange(len(dev_lbl))
    steps_per_epoch = len(fit_lbl) // batch_size
    
    for epoch in range(1, epochs + 1):
        t_ep_start = time.time()
        model.train()
        train_loss = 0.0
        
        for step in range(steps_per_epoch):
            # Balanced mini-batch: 8 positives, 24 negatives (1:3 ratio)
            b_pos = pos_idx[rng.choice(n_pos, size=8, replace=True)]
            b_neg = neg_idx[rng.choice(n_neg, size=24, replace=False)]
            b_indices = np.concatenate([b_pos, b_neg])
            
            fi, fv, ci, cv = sample_augmented_batch(fit_f37, fit_c27, b_indices, rng)
            vf, vc = make_inputs(
                torch.from_numpy(fi).to(device), torch.from_numpy(fv).to(device),
                torch.from_numpy(ci).to(device), torch.from_numpy(cv).to(device),
                f_fine, f_ctx
            )
            targets = torch.tensor([1] * 8 + [0] * 24, device=device, dtype=torch.long)
            
            optimizer.zero_grad()
            logits = model(vf, vc)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += float(loss.item())
            time.sleep(0.025)  # Thermal pacing sleep for GPU 1
            
        scheduler.step()
        avg_train_loss = train_loss / max(steps_per_epoch, 1)
        
        # Evaluate on dev under 4-view TTA
        dev_scores = eval_tta4(model, dev_f37, dev_c27, dev_all_idx, device, f_fine, f_ctx)
        dev_prauc = pr_auc_score(dev_lbl, dev_scores)
        dev_rocauc = roc_auc_score_np(dev_lbl, dev_scores)
        c_metrics = calc_confusion_metrics(dev_lbl, dev_scores, threshold=0.5)
        
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
                "dev_labels": dev_lbl.tolist()
            }, OUT_DIR / "best_checkpoint.pt")
            
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "final_dev_scores": dev_scores.tolist(),
        "dev_ids": dev_ids,
        "dev_labels": dev_lbl.tolist()
    }, OUT_DIR / "checkpoint.pt")
    
    total_train_time = time.time() - t_start
    log(f"Training completed in {total_train_time:.1f}s. Best epoch: {best_epoch} (PR-AUC: {best_dev_prauc:.4f})")
    
    # 4. Final Evaluation on 1,060 Dev Set (Using Best Checkpoint)
    best_ck = torch.load(OUT_DIR / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ck["model_state_dict"])
    model.eval()
    
    t_lat_start = time.time()
    final_dev_scores = eval_tta4(model, dev_f37, dev_c27, dev_all_idx, device, f_fine, f_ctx)
    lat_total = time.time() - t_lat_start
    ms_per_cand = (lat_total / len(dev_lbl)) * 1000.0
    
    prauc_1060 = pr_auc_score(dev_lbl, final_dev_scores)
    rocauc_1060 = roc_auc_score_np(dev_lbl, final_dev_scores)
    m_05 = calc_confusion_metrics(dev_lbl, final_dev_scores, threshold=0.5)
    
    best_th = 0.5
    best_f1 = -1.0
    for th in np.linspace(0.1, 0.9, 81):
        f1_cur = calc_confusion_metrics(dev_lbl, final_dev_scores, threshold=th)["f1_positive"]
        if f1_cur > best_f1:
            best_f1 = f1_cur
            best_th = float(th)
    m_best = calc_confusion_metrics(dev_lbl, final_dev_scores, threshold=best_th)
    
    log(chr(10) + "=" * 50)
    log("FINAL CLEAR3D-HIGHRES DEV RESULTS (1,060 Candidates, 4-view TTA):")
    log(f"  PR-AUC:        {prauc_1060:.4f}")
    log(f"  ROC-AUC:       {rocauc_1060:.4f}")
    log(f"  Threshold 0.5: Precision={m_05['precision']:.4f}, Recall={m_05['recall']:.4f}, Specificity={m_05['specificity']:.4f}, Pos-F1={m_05['f1_positive']:.4f}, Macro-F1={m_05['macro_f1']:.4f}")
    log(f"  Best Threshold ({best_th:.2f}): Precision={m_best['precision']:.4f}, Recall={m_best['recall']:.4f}, Specificity={m_best['specificity']:.4f}, Pos-F1={m_best['f1_positive']:.4f}, Macro-F1={m_best['macro_f1']:.4f}")
    log(f"  Inference Latency: {ms_per_cand:.2f} ms/cand ({len(dev_lbl)/lat_total:.1f} cand/s)")
    log(f"  Parameters:    {sum(p.numel() for p in model.parameters()):,}")
    
    with open(OUT_DIR / "dev_predictions_1060.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "probability"])
        for cid, lbl, prob in zip(dev_ids, dev_lbl, final_dev_scores):
            w.writerow([cid, lbl, f"{prob:.6f}"])
            
    # 5. Broader Evaluation: 20 Patient-Separated Dev Scans (18,678 candidates)
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
    views = [(False, False), (True, False), (False, True), (True, True)]
    
    for s_i, (sid, rr) in enumerate(sorted_scans, 1):
        if s_i % 5 == 0 or s_i == len(sorted_scans):
            print(f"[20 scans] Evaluated {s_i}/{len(sorted_scans)} scans elapsed={time.time()-t_20_start:.1f}s", flush=True)
        d = PREP / sid
        geo = json.loads((d / "geometry.json").read_text())
        aff = np.asarray(geo["affine"], dtype=np.float64)
        inv = np.linalg.inv(aff)
        img = np.load(d / "image.npy", mmap_mode="r")
        val = np.load(d / "valid.npy", mmap_mode="r")
        
        # Extract pairs
        n_rr = len(rr)
        f_im = np.empty((n_rr, 33, 33, 33), dtype=np.float32)
        f_va = np.empty((n_rr, 33, 33, 33), dtype=np.float32)
        c_im = np.empty((n_rr, 25, 25, 25), dtype=np.float32)
        c_va = np.empty((n_rr, 25, 25, 25), dtype=np.float32)
        
        for k_i, c in enumerate(rr):
            zyx = (inv @ np.array([c["x"], c["y"], c["z"], 1.0]))[:3]
            f37, c27 = extract_pair_u16(img, val, zyx)
            f_im[k_i], f_va[k_i] = unpack_u16(f37[2:35, 2:35, 2:35])
            c_im[k_i], c_va[k_i] = unpack_u16(c27[1:26, 1:26, 1:26])
            
        with torch.no_grad():
            for b_idx in range(0, n_rr, 64):
                end_idx = min(b_idx + 64, n_rr)
                sub_len = end_idx - b_idx
                vf, vc = make_inputs(
                    torch.from_numpy(f_im[b_idx:end_idx]).to(device),
                    torch.from_numpy(f_va[b_idx:end_idx]).to(device),
                    torch.from_numpy(c_im[b_idx:end_idx]).to(device),
                    torch.from_numpy(c_va[b_idx:end_idx]).to(device),
                    f_fine, f_ctx
                )
                acc = torch.zeros(sub_len, device=device, dtype=torch.float32)
                for fz, fx in views:
                    xf = vf; xc = vc
                    if fz: xf = torch.flip(xf, [2]); xc = torch.flip(xc, [2])
                    if fx: xf = torch.flip(xf, [4]); xc = torch.flip(xc, [4])
                    logits = model(xf, xc).float()
                    acc = acc + (logits[:, 1] - logits[:, 0])
                probs = 1.0 / (1.0 + np.exp(-(acc * 0.25).cpu().numpy()))
                scan_20_scores.extend(probs)
                scan_20_labels.extend([int(r["label"]) for r in rr[b_idx:end_idx]])
                scan_20_ids.extend([r["id"] for r in rr[b_idx:end_idx]])
                
    t_20_total = time.time() - t_20_start
    scan_20_scores = np.array(scan_20_scores, dtype=np.float32)
    scan_20_labels = np.array(scan_20_labels, dtype=np.int64)
    prauc_20 = pr_auc_score(scan_20_labels, scan_20_scores)
    rocauc_20 = roc_auc_score_np(scan_20_labels, scan_20_scores)
    m_20_05 = calc_confusion_metrics(scan_20_labels, scan_20_scores, threshold=0.5)
    
    log(chr(10) + "=" * 50)
    log(f"CLEAR3D-HIGHRES BROADER EVALUATION (20 Scans, {len(scan_20_labels)} Candidates, {sum(scan_20_labels==1)} pos):")
    log(f"  PR-AUC:        {prauc_20:.4f}")
    log(f"  ROC-AUC:       {rocauc_20:.4f}")
    log(f"  Threshold 0.5: Precision={m_20_05['precision']:.4f}, Recall={m_20_05['recall']:.4f}, Specificity={m_20_05['specificity']:.4f}, Pos-F1={m_20_05['f1_positive']:.4f}, Macro-F1={m_20_05['macro_f1']:.4f}")
    log(f"  Evaluation Time: {t_20_total:.1f}s ({(t_20_total/len(scan_20_labels))*1000:.2f} ms/cand)")
    
    summary_results = {
        "model_name": "CLEAR3D-HighRes-Intervention",
        "checkpoint_sha256": hashlib.sha256((OUT_DIR / "checkpoint.pt").read_bytes()).hexdigest(),
        "best_checkpoint_sha256": hashlib.sha256((OUT_DIR / "best_checkpoint.pt").read_bytes()).hexdigest(),
        "training_epochs": epochs,
        "batch_size": batch_size,
        "training_time_seconds": total_train_time,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "latency_ms_per_cand": ms_per_cand,
        "dev_1060": {
            "total": len(dev_lbl),
            "positives": int((dev_lbl == 1).sum()),
            "negatives": int((dev_lbl == 0).sum()),
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
    with open(OUT_DIR / "clear3d_highres_evaluation_summary.json", "w") as f:
        json.dump(summary_results, f, indent=2)
        
    log_fp.close()
    print(chr(10) + "CLEAR3D-HighRes training and evaluation finished successfully!", flush=True)

if __name__ == "__main__":
    main()
