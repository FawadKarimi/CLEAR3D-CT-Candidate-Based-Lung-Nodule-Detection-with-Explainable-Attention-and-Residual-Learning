#!/usr/bin/env python3
"""
CLEAR3D-CT Fold 0 Single-Factor Focal-Loss Intervention & Matched-Recipe Nested Calibration
===========================================================================================
Strictly Exploratory Development Specification on dev708 (708 scans, 707 patients).
Preserves "reference_release": "HOLD" and excludes withheld Subset 0 (89 scans) and
quarantined Subset 1 / duplicate-volume scans (91 patients).

Compares under identical 4-stream RNG (rng_pos, rng_rand, rng_sec, rng_aug),
identical Scout-mined top-12,000 hard-negative bank & weights (0.5 + 0.5*s),
identical initialization, optimizer, LR schedule, and 22x125 update budget:
  - Arm A (Control): CrossEntropy(label_smoothing=0.015) * sw
  - Arm C (Focal Intervention): (1 - p_t)^2.0 * CrossEntropy(label_smoothing=0.0) * sw
And fits matched-recipe Inner Fold 1 calibration & thresholds using M_inner_control and
M_inner_focal trained strictly on Folds 2..9 (with their own Folds 2..9 Scout).
"""
import argparse
import concurrent.futures
import csv
import gc
import hashlib
import inspect
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn.functional as F

BASE = Path("/experiment-storage/chaidir/fawad-dsai")
ROOT = BASE / "CLEAR3D_CT project/02_Active_Model_and_Pipeline"
WORK = BASE / "CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64/source"
REFD = BASE / "data/reference/luna16-zenodo-3723295/resolved-reference-v1"
BASE_CV_DIR = ROOT / "runs/crossval-10fold-clear3d-20261109"
EXP2 = ROOT / "runs/exp_fold0_matched_negsampling_20261115_v2"
EXP_DIR = ROOT / "runs/exp_fold0_focal_and_matched_inner_20261115"
SEL_PATH = ROOT / "selection-001.json"
EVAL_PY = ROOT / "paper-ablation-verify-package-20261107-004/evaluator/noduleCADEvaluationLUNA16.py"

sys.path[:0] = [str(ROOT), str(WORK / "src"), str(WORK / "scripts"), str(EXP_DIR)]
from clear3d_msf_model_20261106_002 import CLEAR3DMSFv2, build_focus_grids, make_inputs
from train_clear3d_msf_ensemble_20261106_004 import gpu_extract_from_p51

N_FOLDS = 10
RAND_PER_FOLD = 1778
RAND_NEG_COUNT = 16000
SEC_NEG_COUNT = 12000
SCOUT_EPOCHS = 12
SCOUT_STEPS = 125
MAIN_EPOCHS = 22
MAIN_STEPS = 125
LABEL_SMOOTHING = 0.015
FOCAL_GAMMA = 2.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg: str, log_path: Optional[Path] = None) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(line + chr(10))


def write_status_atomic(stage: str, extra: Optional[Dict[str, Any]] = None) -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "stage": stage,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pid": os.getpid(),
    }
    if extra:
        payload.update(extra)
    tmp = EXP_DIR / "status.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, EXP_DIR / "status.json")


def load_checkpoint_cpu(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def verify_governance_hold() -> Dict[str, Any]:
    with open(SEL_PATH, "r") as f:
        sel = json.load(f)
    if sel.get("reference_release") != "HOLD":
        raise RuntimeError(f"Governance violation: reference_release={sel.get('reference_release')} != HOLD")
    return sel


def validate_dev_cohort(
    dev_scans: Sequence[Dict[str, Any]],
    quarantined_patients: Set[str],
    quarantined_scans: Set[str],
    withheld_suids: Set[str],
    withheld_pids: Set[str],
    expected_scans: int = 708,
    expected_patients: int = 707,
) -> None:
    assert len(dev_scans) == expected_scans
    dev_pids = {s["patient_id"] for s in dev_scans}
    dev_suids = {s["series_uid"] for s in dev_scans}
    dev_sids = {s["scan_id"] for s in dev_scans}
    assert len(dev_pids) == expected_patients
    assert len(dev_suids) == expected_scans and len(dev_sids) == expected_scans
    assert not (dev_pids & quarantined_patients), "Quarantined patient leak!"
    assert not (dev_sids & quarantined_scans), "Quarantined duplicate scan leak!"
    assert not (dev_suids & withheld_suids) and not (dev_pids & withheld_pids), "Withheld test scan/patient leak!"
    fold_patients: Dict[int, Set[str]] = defaultdict(set)
    for s in dev_scans:
        fold_patients[int(s["fold"])].add(s["patient_id"])
    for i in range(N_FOLDS):
        for j in range(i + 1, N_FOLDS):
            assert not (fold_patients[i] & fold_patients[j]), f"Patient overlap between fold {i} and {j}!"


def make_training_rng_streams(seed_base: int = 5000) -> Dict[str, np.random.Generator]:
    return {
        "rng_pos": np.random.default_rng(seed_base + 100),
        "rng_rand": np.random.default_rng(seed_base + 200),
        "rng_sec": np.random.default_rng(seed_base + 300),
        "rng_aug": np.random.default_rng(seed_base + 400),
    }


def compute_epoch_lr_and_sw(ep: int, main_epochs: int = MAIN_EPOCHS) -> Tuple[float, np.ndarray]:
    if ep == 0:
        lr = 4e-4
    else:
        prog = (ep - 1) / max(main_epochs - 2, 1)
        lr = 1e-5 + 0.5 * (8e-4 - 1e-5) * (1.0 + math.cos(math.pi * prog))
    cut_ep = int(0.70 * main_epochs)
    if ep < cut_ep:
        hard_share = 0.25
    else:
        hard_share = 0.08 + 0.17 * 0.5 * (1.0 + math.cos(math.pi * (ep - cut_ep) / max(main_epochs - 1 - cut_ep, 1)))
    w_pos = 16.0 / 10.0
    w_rand = (16.0 * (1.0 - hard_share)) / 12.0
    w_hard = (16.0 * hard_share) / 10.0
    sw = np.array([w_pos] * 10 + [w_rand] * 12 + [w_hard] * 10, dtype=np.float32)
    return float(lr), sw


def compute_batch_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sw: torch.Tensor,
    loss_mode: str = "ce",
    focal_gamma: float = FOCAL_GAMMA,
) -> torch.Tensor:
    logits_f = logits.float()
    if loss_mode == "ce":
        ce = F.cross_entropy(logits_f, targets, reduction="none", label_smoothing=LABEL_SMOOTHING)
        return (ce * sw).mean()
    elif loss_mode == "focal":
        ce = F.cross_entropy(logits_f, targets, reduction="none", label_smoothing=0.0)
        pt = torch.exp(-ce)
        focal_weight = (1.0 - pt) ** focal_gamma
        return (focal_weight * ce * sw).mean()
    else:
        raise ValueError(f"Unknown loss_mode={loss_mode}")


def sample_aug_params_dev708(rng_aug: np.random.Generator, B: int = 32) -> Dict[str, np.ndarray]:
    do_shift = rng_aug.random(B) < 0.5
    shifts = np.where(do_shift[:, None], rng_aug.integers(-1, 2, size=(B, 3), dtype=np.int64), np.zeros((B, 3), dtype=np.int64))
    ks = rng_aug.integers(0, 4, size=B, dtype=np.int64)
    flip_z = rng_aug.random(B) < 0.5
    flip_x = rng_aug.random(B) < 0.5
    do_int = rng_aug.random(B) < 0.5
    scale = np.where(do_int, rng_aug.uniform(0.95, 1.05, size=B), np.ones(B)).astype(np.float32)
    shift = np.where(do_int, rng_aug.uniform(-0.02, 0.02, size=B), np.zeros(B)).astype(np.float32)
    return {"shifts": shifts, "ks": ks, "flip_z": flip_z, "flip_x": flip_x, "scale": scale, "shift": shift}


def aug_gpu_from_params(p_sub_f16: np.ndarray, v_sub_bool: np.ndarray, params: Dict[str, np.ndarray], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    out_p = torch.from_numpy(p_sub_f16).to(device=device, dtype=torch.float32).clone()
    out_v = torch.from_numpy(v_sub_bool).to(device=device, dtype=torch.float32).clone()
    shifts, ks, flip_z, flip_x, scale, shift = params["shifts"], params["ks"], params["flip_z"], params["flip_x"], params["scale"], params["shift"]
    for i in range(len(p_sub_f16)):
        dz, dy, dx = int(shifts[i, 0]), int(shifts[i, 1]), int(shifts[i, 2])
        if dz or dy or dx:
            out_p[i] = torch.roll(out_p[i], shifts=(dz, dy, dx), dims=(0, 1, 2))
            out_v[i] = torch.roll(out_v[i], shifts=(dz, dy, dx), dims=(0, 1, 2))
        k = int(ks[i])
        if k:
            out_p[i] = torch.rot90(out_p[i], k, dims=(1, 2))
            out_v[i] = torch.rot90(out_v[i], k, dims=(1, 2))
        if flip_z[i]:
            out_p[i] = torch.flip(out_p[i], dims=(0,))
            out_v[i] = torch.flip(out_v[i], dims=(0,))
        if flip_x[i]:
            out_p[i] = torch.flip(out_p[i], dims=(2,))
            out_v[i] = torch.flip(out_v[i], dims=(2,))
        sc, sh = float(scale[i]), float(shift[i])
        if sc != 1.0 or sh != 0.0:
            out_p[i] = torch.clamp(out_p[i] * sc + sh, 0.0, 1.0)
    return out_p, out_v


def load_scan_dev(scan_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = BASE / "CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-development-fold0-v1/prepared" / scan_id
    assert d.exists() and not d.is_symlink(), f"Invalid or non-dev scan path: {d}"
    geo = json.loads((d / "geometry.json").read_text())
    inv = np.linalg.inv(np.asarray(geo["affine"], dtype=np.float64))
    img = np.load(d / "image.npy", mmap_mode=None, allow_pickle=False)
    val = np.load(d / "valid.npy", mmap_mode=None, allow_pickle=False)
    return img, val, inv


def iter_prefetched_scans(scan_ids: Sequence[str], max_prefetch: int = 4) -> Iterable[Tuple[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    if not scan_ids:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_prefetch) as ex:
        futures: List[Tuple[str, concurrent.futures.Future]] = []
        idx = 0
        while idx < len(scan_ids) and len(futures) < max_prefetch:
            sid = scan_ids[idx]
            futures.append((sid, ex.submit(load_scan_dev, sid)))
            idx += 1
        while futures:
            sid, fut = futures.pop(0)
            if idx < len(scan_ids):
                nsid = scan_ids[idx]
                futures.append((nsid, ex.submit(load_scan_dev, nsid)))
                idx += 1
            yield sid, fut.result()


def gpu_pad_and_slice_setup(img: np.ndarray, val: np.ndarray, zyxs: np.ndarray, device: torch.device):
    centers = np.floor(zyxs + 0.5).astype(np.int64)
    shape = np.array(img.shape, dtype=np.int64)
    pad = int(max(28, 26 - int(centers.min()), int((centers + 27 - shape).max())))
    img_t = F.pad(torch.from_numpy(np.asarray(img, dtype=np.float32)).to(device), (pad,) * 6, value=0.0)
    val_t = F.pad(torch.from_numpy(np.asarray(val, dtype=np.bool_)).to(device), (pad,) * 6, value=False)
    c_pad = torch.from_numpy(centers + pad).to(device=device, dtype=torch.long)
    off = torch.arange(-25, 26, device=device, dtype=torch.long)
    return img_t, val_t, c_pad, off


def gpu_slice_chunk(img_t: torch.Tensor, val_t: torch.Tensor, c_chunk: torch.Tensor, off: torch.Tensor):
    iz = c_chunk[:, 0, None, None, None] + off[None, :, None, None]
    iy = c_chunk[:, 1, None, None, None] + off[None, None, :, None]
    ix = c_chunk[:, 2, None, None, None] + off[None, None, None, :]
    return img_t[iz, iy, ix], val_t[iz, iy, ix]


def validate_sec_weights(sec_weights: np.ndarray, min_count: int = 10) -> np.ndarray:
    sec_p = np.asarray(sec_weights, dtype=np.float64)
    assert sec_p.ndim == 1 and len(sec_p) >= min_count
    assert np.all(np.isfinite(sec_p)) and np.all(sec_p >= 0.0)
    assert np.count_nonzero(sec_p) >= min_count
    s = float(sec_p.sum())
    assert s > 0.0
    return sec_p / s


def train_scout_fold(
    device_str: str,
    pos_p51: np.ndarray,
    pos_v51: np.ndarray,
    rand_p51: np.ndarray,
    rand_v51: np.ndarray,
    seed_init: int = 42,
    seed_rng: int = 3000,
    epochs: int = SCOUT_EPOCHS,
    steps_per_epoch: int = SCOUT_STEPS,
) -> Tuple[Dict[str, torch.Tensor], float, float]:
    t0 = time.time()
    device = torch.device(device_str)
    random.seed(seed_init)
    np.random.seed(seed_init)
    torch.manual_seed(seed_init)
    torch.cuda.manual_seed_all(seed_init)

    scout = CLEAR3DMSFv2(in_channels=3, dim=160, dropout=0.25).to(device)
    scout_opt = torch.optim.AdamW(scout.parameters(), lr=8e-4, weight_decay=1e-2)
    scout_scaler = torch.amp.GradScaler("cuda")
    f_fine, f_ctx = build_focus_grids(device)
    targets = torch.tensor([1] * 12 + [0] * 20, device=device, dtype=torch.long)
    sw = torch.tensor([16.0 / 12] * 12 + [16.0 / 20] * 20, device=device, dtype=torch.float32)
    rng_scout = np.random.default_rng(seed_rng)
    last_loss = 0.0

    for ep in range(epochs):
        scout.train()
        lr = 1e-5 + 0.5 * (8e-4 - 1e-5) * (1.0 + math.cos(math.pi * ep / max(epochs - 1, 1)))
        for g in scout_opt.param_groups:
            g["lr"] = lr
        ep_loss = 0.0
        for _ in range(steps_per_epoch):
            ip = rng_scout.choice(len(pos_p51), size=12, replace=True)
            ir = rng_scout.choice(len(rand_p51), size=20, replace=True)
            p_cat = np.concatenate([pos_p51[ip], rand_p51[ir]], axis=0)
            v_cat = np.concatenate([pos_v51[ip], rand_v51[ir]], axis=0)
            aug_p = sample_aug_params_dev708(rng_scout, B=32)
            p_t, v_t = aug_gpu_from_params(p_cat, v_cat, aug_p, device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                fi_g, fv_g, ci_g, cv_g = gpu_extract_from_p51(p_t, v_t)
                vf, vc = make_inputs(fi_g, fv_g, ci_g, cv_g, f_fine, f_ctx)
                scout_opt.zero_grad(set_to_none=True)
                logits = scout(vf, vc)
                loss = (F.cross_entropy(logits.float(), targets, reduction="none", label_smoothing=LABEL_SMOOTHING) * sw).mean()
            scout_scaler.scale(loss).backward()
            scout_scaler.unscale_(scout_opt)
            torch.nn.utils.clip_grad_norm_(scout.parameters(), 2.0)
            scout_scaler.step(scout_opt)
            scout_scaler.update()
            ep_loss += float(loss.detach())
        last_loss = ep_loss / steps_per_epoch

    torch.cuda.synchronize(device)
    sd = {k: v.detach().cpu() for k, v in scout.state_dict().items()}
    del scout, scout_opt, scout_scaler
    torch.cuda.empty_cache()
    return sd, last_loss, float(time.time() - t0)


def train_main_model_split_rng(
    device_str: str,
    pos_p51: np.ndarray,
    pos_v51: np.ndarray,
    rand_p51: np.ndarray,
    rand_v51: np.ndarray,
    sec_p51: np.ndarray,
    sec_v51: np.ndarray,
    sec_weights: np.ndarray,
    loss_mode: str = "ce",
    seed_init: int = 4200,
    seed_stream_base: int = 5000,
    epochs: int = MAIN_EPOCHS,
    steps_per_epoch: int = MAIN_STEPS,
) -> Tuple[Dict[str, torch.Tensor], float, Dict[str, str], float]:
    t0 = time.time()
    sec_p = validate_sec_weights(sec_weights, min_count=10)
    device = torch.device(device_str)
    torch.manual_seed(seed_init)
    torch.cuda.manual_seed_all(seed_init)
    model = CLEAR3DMSFv2(in_channels=3, dim=160, dropout=0.25).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda")
    f_fine, f_ctx = build_focus_grids(device)
    targets = torch.tensor([1] * 10 + [0] * 22, device=device, dtype=torch.long)

    streams = make_training_rng_streams(seed_stream_base)
    rng_pos, rng_rand, rng_sec, rng_aug = streams["rng_pos"], streams["rng_rand"], streams["rng_sec"], streams["rng_aug"]
    ip_trace, ir_trace, ih_trace = [], [], []
    last_loss = 0.0

    for ep in range(epochs):
        lr, sw_np = compute_epoch_lr_and_sw(ep, epochs)
        for g in opt.param_groups:
            g["lr"] = lr
        sw = torch.from_numpy(sw_np).to(device=device, dtype=torch.float32)
        model.train()
        ep_loss = 0.0
        for _ in range(steps_per_epoch):
            ip = rng_pos.choice(len(pos_p51), size=10, replace=True)
            ir = rng_rand.choice(len(rand_p51), size=12, replace=False)
            ih = rng_sec.choice(len(sec_p51), size=10, replace=False, p=sec_p)
            aug_p = sample_aug_params_dev708(rng_aug, B=32)
            ip_trace.append(ip); ir_trace.append(ir); ih_trace.append(ih)

            p_cat = np.concatenate([pos_p51[ip], rand_p51[ir], sec_p51[ih]], axis=0)
            v_cat = np.concatenate([pos_v51[ip], rand_v51[ir], sec_v51[ih]], axis=0)
            p_t, v_t = aug_gpu_from_params(p_cat, v_cat, aug_p, device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                fi_g, fv_g, ci_g, cv_g = gpu_extract_from_p51(p_t, v_t)
                vf, vc = make_inputs(fi_g, fv_g, ci_g, cv_g, f_fine, f_ctx)
                opt.zero_grad(set_to_none=True)
                logits = model(vf, vc)
                loss = compute_batch_loss(logits, targets, sw, loss_mode=loss_mode)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            ep_loss += float(loss.detach())
        last_loss = ep_loss / steps_per_epoch

    torch.cuda.synchronize(device)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    traces = {
        "ip_sha256": hashlib.sha256(np.stack(ip_trace).tobytes()).hexdigest(),
        "ir_sha256": hashlib.sha256(np.stack(ir_trace).tobytes()).hexdigest(),
        "ih_sha256": hashlib.sha256(np.stack(ih_trace).tobytes()).hexdigest(),
    }
    del model, opt, scaler
    torch.cuda.empty_cache()
    return sd, last_loss, traces, float(time.time() - t0)


def sigmoid_np(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def logit_np(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p_clip = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clip / (1.0 - p_clip))


def compute_natural_candidate_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    y_pred = (scores >= threshold).astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    n_total = tp + fp + tn + fn
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = (tp + tn) / n_total if n_total > 0 else 0.0
    pos_f1 = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
    neg_f1 = (2.0 * tn) / (2.0 * tn + fp + fn) if (2.0 * tn + fp + fn) > 0 else 0.0
    return {
        "threshold": float(threshold),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "N": int(n_total), "positives": int(tp + fn), "negatives": int(tn + fp),
        "sensitivity": float(sens), "precision": float(prec), "specificity": float(spec),
        "accuracy": float(acc), "positive_f1": float(pos_f1), "negative_f1": float(neg_f1),
        "macro_f1": float(0.5 * (pos_f1 + neg_f1)),
        "targets_98pct_achieved": {
            "sensitivity_ge_0.98": bool(sens >= 0.98),
            "precision_ge_0.98": bool(prec >= 0.98),
            "specificity_ge_0.98": bool(spec >= 0.98),
            "accuracy_ge_0.98": bool(acc >= 0.98),
            "positive_f1_ge_0.98": bool(pos_f1 >= 0.98),
        },
    }


def compute_roc_and_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores)[::-1]
    yt = y_true[order]
    tp_c = np.cumsum(yt)
    fp_c = np.cumsum(1 - yt)
    prec_c = np.concatenate(([1.0], tp_c / np.maximum(tp_c + fp_c, 1)))
    rec_c = np.concatenate(([0.0], tp_c / max(int(tp_c[-1]), 1)))
    pr_auc = float(np.sum((rec_c[1:] - rec_c[:-1]) * prec_c[1:]))
    n_pos, n_neg = int(np.sum(y_true == 1)), int(np.sum(y_true == 0))
    om = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[om] = np.arange(1, len(scores) + 1)
    roc_auc = float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1))
    return roc_auc, pr_auc


def select_max_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    assert len(scores) > 0 and len(scores) == len(y_true)
    order = np.argsort(-scores, kind="mergesort")
    s_sorted, y_sorted = scores[order], y_true[order]
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    total_pos = int(np.sum(y_true == 1))
    distinct_mask = np.r_[s_sorted[:-1] != s_sorted[1:], True]
    tp_d, fp_d, s_d = tp_cum[distinct_mask], fp_cum[distinct_mask], s_sorted[distinct_mask]
    fn_d = total_pos - tp_d
    denom = 2.0 * tp_d + fp_d + fn_d
    f1_d = np.where(denom > 0, (2.0 * tp_d) / denom, 0.0)
    tau_star = float(s_d[int(np.argmax(f1_d))])
    return tau_star, compute_natural_candidate_metrics(y_true, scores, tau_star)


def fit_nested_platt_and_thresholds(p_raw_inner: np.ndarray, y_inner: np.ndarray) -> Dict[str, Any]:
    y_inner = np.asarray(y_inner, dtype=np.int64)
    p_raw_inner = np.asarray(p_raw_inner, dtype=np.float64)
    assert len(y_inner) > 0 and 0 < int(y_inner.sum()) < len(y_inner)
    z_inner = logit_np(p_raw_inner)

    def nll_obj(params: np.ndarray) -> float:
        log_t, b = float(params[0]), float(params[1])
        t = np.exp(log_t)
        u = np.clip((z_inner - b) / t, -40.0, 40.0)
        return float(np.mean(np.maximum(u, 0.0) - u * y_inner + np.log1p(np.exp(-np.abs(u)))) + 1e-5 * (log_t ** 2 + b ** 2))

    res = minimize(nll_obj, x0=np.array([0.0, 0.0], dtype=np.float64), method="BFGS")
    T_0, b_0 = float(np.exp(res.x[0])), float(res.x[1])
    p_cal_inner = sigmoid_np((z_inner - b_0) / T_0)
    tau_raw_star, inner_raw_opt = select_max_f1_threshold(y_inner, p_raw_inner)
    tau_cal_star, inner_cal_opt = select_max_f1_threshold(y_inner, p_cal_inner)
    return {
        "optimizer_success": bool(res.success),
        "optimizer_message": str(res.message),
        "T_0": T_0, "b_0": b_0,
        "tau_raw_0_star": tau_raw_star, "tau_cal_0_star": tau_cal_star,
        "inner_fold1_metrics": {
            "raw_at_0.5": compute_natural_candidate_metrics(y_inner, p_raw_inner, 0.5),
            "raw_at_tau_raw_star": inner_raw_opt,
            "cal_at_0.5": compute_natural_candidate_metrics(y_inner, p_cal_inner, 0.5),
            "cal_at_tau_cal_star": inner_cal_opt,
        },
    }


def apply_nested_calibration_to_outer(p_raw_outer: np.ndarray, y_outer: np.ndarray, cal_dict: Dict[str, Any]) -> Dict[str, Any]:
    p_raw_outer = np.asarray(p_raw_outer, dtype=np.float64)
    y_outer = np.asarray(y_outer, dtype=np.int64)
    T_0, b_0 = float(cal_dict["T_0"]), float(cal_dict["b_0"])
    tau_raw_star, tau_cal_star = float(cal_dict["tau_raw_0_star"]), float(cal_dict["tau_cal_0_star"])
    p_cal_outer = sigmoid_np((logit_np(p_raw_outer) - b_0) / T_0)
    order_raw = np.argsort(-p_raw_outer, kind="mergesort")
    rank_preserved = bool(np.all(np.diff(p_cal_outer[order_raw]) <= 1e-12))
    roc_auc, pr_auc = compute_roc_and_pr_auc(y_outer, p_raw_outer)
    return {
        "T_0": T_0, "b_0": b_0,
        "tau_raw_0_star": tau_raw_star, "tau_cal_0_star": tau_cal_star,
        "rank_order_preserved": rank_preserved,
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "p_cal_outer": p_cal_outer,
        "metrics": {
            "raw_at_0.5": compute_natural_candidate_metrics(y_outer, p_raw_outer, 0.5),
            "raw_at_tau_raw_star": compute_natural_candidate_metrics(y_outer, p_raw_outer, tau_raw_star),
            "cal_at_0.5": compute_natural_candidate_metrics(y_outer, p_cal_outer, 0.5),
            "cal_at_tau_cal_star": compute_natural_candidate_metrics(y_outer, p_cal_outer, tau_cal_star),
        },
    }


def run_official_luna16_eval(pred_csv: Path, eval_out_dir: Path) -> Dict[str, Any]:
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    with open(pred_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    scans = sorted({r["seriesuid"] for r in rows})
    suid_p = eval_out_dir / "seriesuids.csv"
    with open(suid_p, "w", newline="") as f:
        w = csv.writer(f)
        for s in scans:
            w.writerow([s])
    cmd = [sys.executable, str(EVAL_PY), str(REFD / "annotations.csv"), str(REFD / "annotations_excluded.csv"), str(suid_p), str(pred_csv), str(eval_out_dir)]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    froc_files = [p for p in eval_out_dir.glob("froc_*.txt") if "bootstrapping" not in p.name]
    fps_grid = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    sens_at_fps: Dict[str, float] = {}
    if froc_files:
        pts = []
        for line in froc_files[0].read_text().splitlines():
            parts = line.strip().split(",")
            if len(parts) == 3:
                try:
                    pts.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
        pts.sort(key=lambda x: x[0])
        f_arr, s_arr = np.array([p[0] for p in pts], dtype=np.float64), np.array([p[1] for p in pts], dtype=np.float64)
        for g in fps_grid:
            sens_at_fps[str(g)] = float(np.interp(g, f_arr, s_arr))
    cpm = float(np.mean(list(sens_at_fps.values()))) if sens_at_fps else 0.0

    boot_ci = None
    boot_files = list(eval_out_dir.glob("froc_*_bootstrapping.csv"))
    if boot_files:
        b_lines = boot_files[0].read_text().splitlines()
        if len(b_lines) > 1:
            h_arr = np.array([float(x) for x in b_lines[0].strip().split(",")[1:]], dtype=np.float64)
            boot_cpms = []
            for bl in b_lines[1:]:
                if not bl.strip():
                    continue
                v_arr = np.array([float(x) for x in bl.strip().split(",")[1:]], dtype=np.float64)
                if not np.any(np.isnan(v_arr)):
                    boot_cpms.append(float(np.mean([np.interp(g, h_arr, v_arr) for g in fps_grid])))
            if boot_cpms:
                bc = np.array(boot_cpms, dtype=np.float64)
                boot_ci = {"mean": float(np.mean(bc)), "std": float(np.std(bc)), "ci95_low": float(np.percentile(bc, 2.5)), "ci95_high": float(np.percentile(bc, 97.5)), "n_boot": int(len(bc))}
    return {"scans": len(scans), "cpm": cpm, "froc": sens_at_fps, "official_bootstrap": boot_ci}


def score_scans_4view_tta(
    models_by_name: Dict[str, Dict[str, torch.Tensor]],
    scans_list: Sequence[Dict[str, Any]],
    eval_cands_by_scan: Dict[str, List[Tuple[int, float, float, float, int]]],
    fold_label: int,
    device: torch.device,
) -> Tuple[Dict[str, List[Dict[str, Any]]], float]:
    t0 = time.time()
    f_fine, f_ctx = build_focus_grids(device)
    views = [(False, False), (True, False), (False, True), (True, True)]
    loaded_models = {}
    for name, sd in models_by_name.items():
        m = CLEAR3DMSFv2(in_channels=3, dim=160, dropout=0.25).to(device)
        m.load_state_dict(sd)
        m.eval()
        loaded_models[name] = m

    rows_by_name: Dict[str, List[Dict[str, Any]]] = {name: [] for name in loaded_models}
    sid_to_suid = {s["scan_id"]: s["series_uid"] for s in scans_list}
    with torch.no_grad():
        for sid, (img, val, inv) in iter_prefetched_scans(sorted(sid_to_suid.keys()), max_prefetch=4):
            suid = sid_to_suid[sid]
            cands = eval_cands_by_scan[suid]
            zyxs = np.array([(inv @ np.array([x, y, z, 1.0]))[:3] for _, x, y, z, _ in cands], dtype=np.float64)
            img_t, val_t, c_pad, off = gpu_pad_and_slice_setup(img, val, zyxs, device)
            probs_by_name = {name: np.empty(len(cands), dtype=np.float32) for name in loaded_models}
            for st in range(0, len(cands), 128):
                en = min(st + 128, len(cands))
                p_t, v_t = gpu_slice_chunk(img_t, val_t, c_pad[st:en], off)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    fi_g, fv_g, ci_g, cv_g = gpu_extract_from_p51(p_t, v_t.float())
                    vf, vc = make_inputs(fi_g, fv_g, ci_g, cv_g, f_fine, f_ctx)
                    for name, m in loaded_models.items():
                        acc = torch.zeros(en - st, device=device, dtype=torch.float32)
                        for fz, fx in views:
                            xf, xc = vf, vc
                            if fz:
                                xf = torch.flip(xf, [2]); xc = torch.flip(xc, [2])
                            if fx:
                                xf = torch.flip(xf, [4]); xc = torch.flip(xc, [4])
                            lg = m(xf, xc).float()
                            acc = acc + (lg[:, 1] - lg[:, 0])
                        probs_by_name[name][st:en] = torch.sigmoid(acc * 0.25).cpu().numpy()

            for name, scan_probs in probs_by_name.items():
                order = np.argsort(-scan_probs)
                ranks = np.empty(len(cands), dtype=int)
                ranks[order] = np.arange(1, len(cands) + 1)
                for i, (c_idx, x, y, z, cls) in enumerate(cands):
                    rows_by_name[name].append({
                        "seriesuid": suid, "coordX": x, "coordY": y, "coordZ": z,
                        "probability": f"{float(scan_probs[i]):.8f}",
                        "candidate_id": f"{fold_label}:{c_idx}", "rank": int(ranks[i]), "label": int(cls),
                    })

    del loaded_models
    torch.cuda.empty_cache()
    return rows_by_name, float(time.time() - t0)


def write_pred_csv(rows: Sequence[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["seriesuid", "coordX", "coordY", "coordZ", "probability", "candidate_id", "rank", "label"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def run_experiment() -> None:
    t_all = time.time()
    log_path = EXP_DIR / "experiment_run.log"
    ckpt_dir = EXP_DIR / "checkpoints"
    pred_dir = EXP_DIR / "predictions"
    eval_dir = EXP_DIR / "official_cad_eval"
    for d in (EXP_DIR, ckpt_dir, pred_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(0.80, 0)
    device = torch.device("cuda:0")

    try:
        write_status_atomic("preflight")
        log("=== STEP 0: Pre-Flight Governance, Cohort & Provenance Verification ===", log_path)
        sel = verify_governance_hold()
        provenance = {
            "run_fold0_focal_experiment.py": sha256_file(EXP_DIR / "run_fold0_focal_experiment.py"),
            "test_focal_experiment_fixture.py": sha256_file(EXP_DIR / "test_focal_experiment_fixture.py"),
            "clear3d_msf_model_20261106_002.py": sha256_file(Path(inspect.getfile(CLEAR3DMSFv2))),
            "train_clear3d_msf_ensemble_20261106_004.py": sha256_file(Path(inspect.getfile(gpu_extract_from_p51))),
            "selection-001.json": sha256_file(SEL_PATH),
        }
        log(f"Provenance hashes: {json.dumps(provenance)}", log_path)

        spec = json.loads((ROOT / "crossval_protocol_specification_20261109.json").read_text())
        q_info = spec["cohort_inventory"]
        manifest = json.loads((ROOT / "patient_to_fold_manifest_20261109.json").read_text())
        p2f = manifest["patient_to_fold"]
        coh_dev_rows = [json.loads(l) for l in open(BASE / "CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-cohort-fold0-v1/cohort_development.jsonl") if l.strip()]
        quarantined_scans = {x["scan"] for x in q_info["excluded_duplicate_volume_details"]}
        quarantined_patients = {r["patient_id"] for r in coh_dev_rows if r["patient_id"] not in p2f}
        withheld_rows = [json.loads(l) for l in open(BASE / "CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-cohort-fold0-v1/cohort_test_WITHHOLD.jsonl") if l.strip()]
        withheld_suids = {r["series_uid"] for r in withheld_rows}
        withheld_pids = {r["patient_id"] for r in withheld_rows}

        dev_rows_raw = [json.loads(l) for l in open(BASE / "CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-development-fold0-v1/prepared/scans.jsonl") if l.strip()]
        dev_scans = [s for s in dev_rows_raw if s["patient_id"] in p2f]
        for s in dev_scans:
            s["fold"] = int(p2f[s["patient_id"]])

        validate_dev_cohort(dev_scans, quarantined_patients, quarantined_scans, withheld_suids, withheld_pids)
        dev_scan_map = {s["series_uid"]: s for s in dev_scans}
        fold0_scans = [s for s in dev_scans if s["fold"] == 0]
        fold1_scans = [s for s in dev_scans if s["fold"] == 1]

        pos_cache = np.load(BASE_CV_DIR / "cache/positives_cache.npz")
        pos_meta_all = json.loads(str(pos_cache["metadata"]))
        dev_pos_keep = [i for i, c in enumerate(pos_meta_all) if c["series"] in dev_scan_map]
        assert len(dev_pos_keep) == 1249
        pos_meta = [pos_meta_all[i] for i in dev_pos_keep]
        pos_p51_all = pos_cache["p51"][dev_pos_keep].astype(np.float16)
        pos_v51_all = pos_cache["v51"][dev_pos_keep] > 0.5
        pos_folds = np.array([int(c["fold"]) for c in pos_meta], dtype=np.int64)
        del pos_cache
        gc.collect()

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

        clean_negs_by_fold = defaultdict(list)
        eval_cands_by_scan = defaultdict(list)
        with open(REFD / "candidates_V2.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            for idx, r in enumerate(reader):
                suid = r[0]
                if suid not in dev_scan_map:
                    continue
                x, y, z, cls = float(r[1]), float(r[2]), float(r[3]), int(r[4])
                s_info = dev_scan_map[suid]
                f_k = s_info["fold"]
                eval_cands_by_scan[suid].append((idx, x, y, z, cls))
                if cls == 0:
                    c = (x, y, z)
                    if any(math.dist(c, ac) <= rad for ac, rad in exc_refs.get(suid, ())):
                        continue
                    if any(math.dist(c, ac) <= rad for _, ac, rad in inc_refs.get(suid, ())):
                        continue
                    clean_negs_by_fold[f_k].append({"cand_idx": idx, "series": suid, "scan": s_info["scan_id"], "fold": f_k, "x": x, "y": y, "z": z})

        sampled_rand_list, sampled_rand_cand_ids, fold_rand_ranges = [], set(), {}
        for f_k in range(N_FOLDS):
            picked_f = random.Random(1000 + f_k).sample(clean_negs_by_fold[f_k], RAND_PER_FOLD)
            st_idx = len(sampled_rand_list)
            sampled_rand_list.extend(picked_f)
            fold_rand_ranges[f_k] = (st_idx, len(sampled_rand_list))
            for item in picked_f:
                sampled_rand_cand_ids.add(item["cand_idx"])

        pool_idx_f0 = [i for f_k in range(1, N_FOLDS) for i in range(*fold_rand_ranges[f_k])]
        random.Random(1500).shuffle(pool_idx_f0)
        rand_bank_f0_idx = np.array(pool_idx_f0[:RAND_NEG_COUNT], dtype=np.int64)
        inner_rand_idx = np.array([i for i, item in enumerate(sampled_rand_list) if item["fold"] not in (0, 1)], dtype=np.int64)

        rem_negs = [item for f_k in range(N_FOLDS) for item in clean_negs_by_fold[f_k] if item["cand_idx"] not in sampled_rand_cand_ids]
        rem_folds = np.array([item["fold"] for item in rem_negs], dtype=np.int64)
        f0_rem_global_idx = np.flatnonzero(rem_folds != 0)
        inner_rem_global_idx = np.flatnonzero((rem_folds != 0) & (rem_folds != 1))

        # === STAGE 1A: Extract 17,780 random negative crops ===
        write_status_atomic("stage1a_extract_rand_bank")
        t_s1a = time.time()
        rand_p51 = np.empty((len(sampled_rand_list), 51, 51, 51), dtype=np.float16)
        rand_v51 = np.empty((len(sampled_rand_list), 51, 51, 51), dtype=np.bool_)
        rand_by_scan = defaultdict(list)
        for dst, item in enumerate(sampled_rand_list):
            rand_by_scan[item["scan"]].append((dst, item["x"], item["y"], item["z"]))
        with torch.no_grad():
            for sid, (img, val, inv) in iter_prefetched_scans(sorted(rand_by_scan.keys()), max_prefetch=4):
                jobs = rand_by_scan[sid]
                zyxs = np.array([(inv @ np.array([x, y, z, 1.0]))[:3] for _, x, y, z in jobs], dtype=np.float64)
                img_t, val_t, c_pad, off = gpu_pad_and_slice_setup(img, val, zyxs, device)
                p_t, v_t = gpu_slice_chunk(img_t, val_t, c_pad, off)
                p_np, v_np = p_t.to(torch.float16).cpu().numpy(), v_t.cpu().numpy()
                for j, (dst, _, _, _) in enumerate(jobs):
                    rand_p51[dst] = p_np[j]; rand_v51[dst] = v_np[j]
        torch.cuda.synchronize(device)
        s1a_sec = float(time.time() - t_s1a)
        log(f"Stage 1A extracted {len(sampled_rand_list)} random neg crops in {s1a_sec:.1f}s", log_path)

        pos_idx_f0 = np.flatnonzero(pos_folds != 0)
        f0_pos_p51, f0_pos_v51 = pos_p51_all[pos_idx_f0], pos_v51_all[pos_idx_f0]
        f0_rand_p51, f0_rand_v51 = rand_p51[rand_bank_f0_idx], rand_v51[rand_bank_f0_idx]

        inner_pos_idx = np.flatnonzero((pos_folds != 0) & (pos_folds != 1))
        in_pos_p51, in_pos_v51 = pos_p51_all[inner_pos_idx], pos_v51_all[inner_pos_idx]
        in_rand_p51, in_rand_v51 = rand_p51[inner_rand_idx], rand_v51[inner_rand_idx]

        # === STAGE 1B: Train Scout Fold 0 (Folds 1..9) & Scout Inner (Folds 2..9) ===
        write_status_atomic("stage1b_train_scouts")
        scout_f0_sd, scout_f0_loss, scout_f0_sec = train_scout_fold("cuda:0", f0_pos_p51, f0_pos_v51, f0_rand_p51, f0_rand_v51, seed_init=42, seed_rng=3000)
        torch.save({"model_state_dict": scout_f0_sd, "fold": 0, "epochs": SCOUT_EPOCHS}, ckpt_dir / "scout_fold0.pt")
        scout_in_sd, scout_in_loss, scout_in_sec = train_scout_fold("cuda:0", in_pos_p51, in_pos_v51, in_rand_p51, in_rand_v51, seed_init=43, seed_rng=3001)
        torch.save({"model_state_dict": scout_in_sd, "folds": "2..9", "epochs": SCOUT_EPOCHS}, ckpt_dir / "scout_inner_folds2to9.pt")
        log(f"Stage 1B Scouts trained: Fold0={scout_f0_sec:.1f}s (loss={scout_f0_loss:.4f}), Inner={scout_in_sec:.1f}s (loss={scout_in_loss:.4f})", log_path)

        # === STAGE 2A: Single-pass Mining for Scout Fold 0 & Scout Inner ===
        write_status_atomic("stage2a_mine_both_scouts")
        t_mine = time.time()
        m_sf0 = CLEAR3DMSFv2(in_channels=3, dim=160, dropout=0.25).to(device); m_sf0.load_state_dict(scout_f0_sd); m_sf0.eval()
        m_sin = CLEAR3DMSFv2(in_channels=3, dim=160, dropout=0.25).to(device); m_sin.load_state_dict(scout_in_sd); m_sin.eval()
        f_fine, f_ctx = build_focus_grids(device)

        global_to_f0_local = {int(g): li for li, g in enumerate(f0_rem_global_idx)}
        global_to_in_local = {int(g): li for li, g in enumerate(inner_rem_global_idx)}
        rem_by_scan = defaultdict(list)
        for g_idx in f0_rem_global_idx:
            item = rem_negs[int(g_idx)]
            rem_by_scan[item["scan"]].append((int(g_idx), item["fold"], item["x"], item["y"], item["z"]))

        sf0_scores = np.empty(len(f0_rem_global_idx), dtype=np.float32)
        sin_scores = np.empty(len(inner_rem_global_idx), dtype=np.float32)
        mine_sids = sorted(rem_by_scan.keys())
        with torch.no_grad():
            for s_i, (sid, (img, val, inv)) in enumerate(iter_prefetched_scans(mine_sids, max_prefetch=4), 1):
                items = rem_by_scan[sid]
                f_scan = items[0][1]
                zyxs = np.array([(inv @ np.array([x, y, z, 1.0]))[:3] for _, _, x, y, z in items], dtype=np.float64)
                img_t, val_t, c_pad, off = gpu_pad_and_slice_setup(img, val, zyxs, device)
                f0_l = np.array([global_to_f0_local[g] for g, _, _, _, _ in items], dtype=np.int64)
                in_l = np.array([global_to_in_local[g] for g, _, _, _, _ in items], dtype=np.int64) if f_scan != 1 else None
                for st in range(0, len(items), 128):
                    en = min(st + 128, len(items))
                    p_t, v_t = gpu_slice_chunk(img_t, val_t, c_pad[st:en], off)
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        fi_g, fv_g, ci_g, cv_g = gpu_extract_from_p51(p_t, v_t.float())
                        vf, vc = make_inputs(fi_g, fv_g, ci_g, cv_g, f_fine, f_ctx)
                        sf0_scores[f0_l[st:en]] = torch.softmax(m_sf0(vf, vc).float(), dim=1)[:, 1].cpu().numpy()
                        if in_l is not None:
                            sin_scores[in_l[st:en]] = torch.softmax(m_sin(vf, vc).float(), dim=1)[:, 1].cpu().numpy()
                if s_i % 150 == 0 or s_i == len(mine_sids):
                    log(f"  Mined {s_i}/{len(mine_sids)} scans in {time.time()-t_mine:.1f}s", log_path)
        del m_sf0, m_sin, scout_f0_sd, scout_in_sd
        torch.cuda.empty_cache()
        mine_sec = float(time.time() - t_mine)

        top_f0 = np.argsort(-sf0_scores)[:SEC_NEG_COUNT]
        arma_global_idx = f0_rem_global_idx[top_f0]
        arma_sec_weights = validate_sec_weights(0.5 + 0.5 * sf0_scores[top_f0].astype(np.float64))

        top_in = np.argsort(-sin_scores)[:SEC_NEG_COUNT]
        in_hard_global_idx = inner_rem_global_idx[top_in]
        in_sec_weights = validate_sec_weights(0.5 + 0.5 * sin_scores[top_in].astype(np.float64))
        log(f"Stage 2A mining complete in {mine_sec:.1f}s | F0 top12000={sf0_scores[top_f0[-1]]:.4f} | Inner top12000={sin_scores[top_in[-1]]:.4f}", log_path)

        # === STAGE 2B: Single-pass extraction of both 12,000 hard-negative banks ===
        write_status_atomic("stage2b_extract_hard_banks")
        t_hb = time.time()
        f0_hard_p51 = np.empty((SEC_NEG_COUNT, 51, 51, 51), dtype=np.float16)
        f0_hard_v51 = np.empty((SEC_NEG_COUNT, 51, 51, 51), dtype=np.bool_)
        in_hard_p51 = np.empty((SEC_NEG_COUNT, 51, 51, 51), dtype=np.float16)
        in_hard_v51 = np.empty((SEC_NEG_COUNT, 51, 51, 51), dtype=np.bool_)
        hb_jobs = defaultdict(list)
        for dst, g_idx in enumerate(arma_global_idx):
            item = rem_negs[int(g_idx)]
            hb_jobs[item["scan"]].append(("f0", dst, item["x"], item["y"], item["z"]))
        for dst, g_idx in enumerate(in_hard_global_idx):
            item = rem_negs[int(g_idx)]
            hb_jobs[item["scan"]].append(("in", dst, item["x"], item["y"], item["z"]))
        with torch.no_grad():
            for sid, (img, val, inv) in iter_prefetched_scans(sorted(hb_jobs.keys()), max_prefetch=4):
                jobs = hb_jobs[sid]
                zyxs = np.array([(inv @ np.array([x, y, z, 1.0]))[:3] for _, _, x, y, z in jobs], dtype=np.float64)
                img_t, val_t, c_pad, off = gpu_pad_and_slice_setup(img, val, zyxs, device)
                p_t, v_t = gpu_slice_chunk(img_t, val_t, c_pad, off)
                p_np, v_np = p_t.to(torch.float16).cpu().numpy(), v_t.cpu().numpy()
                for j, (kind, dst, _, _, _) in enumerate(jobs):
                    if kind == "f0":
                        f0_hard_p51[dst] = p_np[j]; f0_hard_v51[dst] = v_np[j]
                    else:
                        in_hard_p51[dst] = p_np[j]; in_hard_v51[dst] = v_np[j]
        torch.cuda.synchronize(device)
        hb_sec = float(time.time() - t_hb)
        log(f"Stage 2B extracted both 12,000 hard banks in {hb_sec:.1f}s", log_path)

        # === STAGE 3: Train Arm A (CE Control), Arm C (Focal Intervention), and Matched Inner Models ===
        write_status_atomic("stage3a_train_arm_a_ce")
        arma_sd, arma_loss, arma_traces, arma_sec = train_main_model_split_rng(
            "cuda:0", f0_pos_p51, f0_pos_v51, f0_rand_p51, f0_rand_v51, f0_hard_p51, f0_hard_v51, arma_sec_weights,
            loss_mode="ce", seed_init=4200, seed_stream_base=5000,
        )
        torch.save({"model_state_dict": arma_sd, "arm": "arm_a_ce_control", "epochs": MAIN_EPOCHS}, ckpt_dir / "arm_a_ce_control.pt")
        log(f"Stage 3A Arm A (CE Control) trained in {arma_sec:.1f}s | final_loss={arma_loss:.4f}", log_path)

        write_status_atomic("stage3b_train_arm_c_focal")
        armc_sd, armc_loss, armc_traces, armc_sec = train_main_model_split_rng(
            "cuda:0", f0_pos_p51, f0_pos_v51, f0_rand_p51, f0_rand_v51, f0_hard_p51, f0_hard_v51, arma_sec_weights,
            loss_mode="focal", seed_init=4200, seed_stream_base=5000,
        )
        torch.save({"model_state_dict": armc_sd, "arm": "arm_c_focal_gamma2", "epochs": MAIN_EPOCHS}, ckpt_dir / "arm_c_focal_gamma2.pt")
        log(f"Stage 3B Arm C (Focal gamma=2) trained in {armc_sec:.1f}s | final_loss={armc_loss:.4f}", log_path)

        assert arma_traces == armc_traces, "Arm A vs Arm C RNG stream traces must be 100% identical!"

        write_status_atomic("stage3c_train_inner_models")
        in_ce_sd, in_ce_loss, _, in_ce_sec = train_main_model_split_rng(
            "cuda:0", in_pos_p51, in_pos_v51, in_rand_p51, in_rand_v51, in_hard_p51, in_hard_v51, in_sec_weights,
            loss_mode="ce", seed_init=4201, seed_stream_base=5100,
        )
        torch.save({"model_state_dict": in_ce_sd, "arm": "inner_ce_control", "epochs": MAIN_EPOCHS}, ckpt_dir / "inner_ce_control.pt")
        in_foc_sd, in_foc_loss, _, in_foc_sec = train_main_model_split_rng(
            "cuda:0", in_pos_p51, in_pos_v51, in_rand_p51, in_rand_v51, in_hard_p51, in_hard_v51, in_sec_weights,
            loss_mode="focal", seed_init=4201, seed_stream_base=5100,
        )
        torch.save({"model_state_dict": in_foc_sd, "arm": "inner_focal_gamma2", "epochs": MAIN_EPOCHS}, ckpt_dir / "inner_focal_gamma2.pt")
        log(f"Stage 3C Inner models trained: CE={in_ce_sec:.1f}s (loss={in_ce_loss:.4f}), Focal={in_foc_sec:.1f}s (loss={in_foc_loss:.4f})", log_path)

        del rand_p51, rand_v51, f0_hard_p51, f0_hard_v51, in_hard_p51, in_hard_v51, pos_p51_all, pos_v51_all
        gc.collect()

        # === STAGE 4: Score Outer Fold 0 (Arm A & Arm C) ===
        write_status_atomic("stage4_score_outer_fold0")
        f0_rows, f0_score_sec = score_scans_4view_tta({"arm_a": arma_sd, "arm_c": armc_sd}, fold0_scans, eval_cands_by_scan, fold_label=0, device=device)
        arma_csv = pred_dir / "arm_a_fold0_predictions.csv"
        armc_csv = pred_dir / "arm_c_fold0_predictions.csv"
        write_pred_csv(f0_rows["arm_a"], arma_csv)
        write_pred_csv(f0_rows["arm_c"], armc_csv)
        log(f"Stage 4 scored Outer Fold 0 (2 arms) in {f0_score_sec:.1f}s", log_path)

        # === STAGE 5: Score Inner Fold 1 (Inner CE & Inner Focal) ===
        write_status_atomic("stage5_score_inner_fold1")
        f1_rows, f1_score_sec = score_scans_4view_tta({"in_ce": in_ce_sd, "in_foc": in_foc_sd}, fold1_scans, eval_cands_by_scan, fold_label=1, device=device)
        write_pred_csv(f1_rows["in_ce"], pred_dir / "inner_ce_fold1_predictions.csv")
        write_pred_csv(f1_rows["in_foc"], pred_dir / "inner_focal_fold1_predictions.csv")
        log(f"Stage 5 scored Inner Fold 1 (2 inner models) in {f1_score_sec:.1f}s", log_path)

        # === STAGE 6: Matched-Recipe & Cross-Recipe Nested Calibration + Official FROC ===
        write_status_atomic("stage6_eval_and_calibration")
        cal_in_ce = fit_nested_platt_and_thresholds(
            np.array([float(r["probability"]) for r in f1_rows["in_ce"]], dtype=np.float64),
            np.array([int(r["label"]) for r in f1_rows["in_ce"]], dtype=np.int64),
        )
        cal_in_foc = fit_nested_platt_and_thresholds(
            np.array([float(r["probability"]) for r in f1_rows["in_foc"]], dtype=np.float64),
            np.array([int(r["label"]) for r in f1_rows["in_foc"]], dtype=np.int64),
        )
        with open(EXP2 / "predictions/inner_fold1_predictions.csv", newline="") as f:
            unif_rows = list(csv.DictReader(f))
        cal_in_unif = fit_nested_platt_and_thresholds(
            np.array([float(r["probability"]) for r in unif_rows], dtype=np.float64),
            np.array([int(r["label"]) for r in unif_rows], dtype=np.int64),
        )

        def eval_arm(name: str, rows: List[Dict[str, Any]], raw_csv: Path, cal_matched: Dict[str, Any]) -> Dict[str, Any]:
            p_raw = np.array([float(r["probability"]) for r in rows], dtype=np.float64)
            y_out = np.array([int(r["label"]) for r in rows], dtype=np.int64)
            outer_matched = apply_nested_calibration_to_outer(p_raw, y_out, cal_matched)
            outer_unif_transfer = apply_nested_calibration_to_outer(p_raw, y_out, cal_in_unif)
            cal_rows = []
            for r, pc in zip(rows, outer_matched["p_cal_outer"]):
                rc = dict(r)
                rc["probability"] = f"{float(pc):.8f}"
                cal_rows.append(rc)
            cal_csv = pred_dir / f"{name}_fold0_calibrated_predictions.csv"
            write_pred_csv(cal_rows, cal_csv)
            froc_raw = run_official_luna16_eval(raw_csv, eval_dir / f"{name}_raw")
            froc_cal = run_official_luna16_eval(cal_csv, eval_dir / f"{name}_cal")
            return {
                "roc_auc": outer_matched["roc_auc"],
                "pr_auc": outer_matched["pr_auc"],
                "rank_order_preserved": outer_matched["rank_order_preserved"],
                "metrics_matched_inner": outer_matched["metrics"],
                "metrics_uniform_inner_transfer_check": outer_unif_transfer["metrics"],
                "froc_raw": froc_raw,
                "froc_cal": froc_cal,
                "froc_cpm_abs_diff": float(abs(froc_raw["cpm"] - froc_cal["cpm"])),
            }

        res_arma = eval_arm("arm_a", f0_rows["arm_a"], arma_csv, cal_in_ce)
        res_armc = eval_arm("arm_c", f0_rows["arm_c"], armc_csv, cal_in_foc)

        summary = {
            "governance": {"reference_release": sel.get("reference_release"), "withheld_subset0_excluded": True, "quarantined_subset1_excluded": True},
            "provenance_sha256": provenance,
            "compute_costs_seconds": {
                "stage1a_rand_extraction": s1a_sec, "scout_f0_12ep": scout_f0_sec, "scout_inner_12ep": scout_in_sec,
                "stage2a_mining_both": mine_sec, "stage2b_hard_extraction": hb_sec,
                "arm_a_ce_22ep": arma_sec, "arm_c_focal_22ep": armc_sec,
                "inner_ce_22ep": in_ce_sec, "inner_focal_22ep": in_foc_sec,
                "outer_fold0_tta_2arms": f0_score_sec, "inner_fold1_tta_2models": f1_score_sec,
                "total_wall_clock": float(time.time() - t_all),
            },
            "rng_stream_traces": {"arm_a": arma_traces, "arm_c": armc_traces, "exact_stream_match": arma_traces == armc_traces},
            "calibration_fits_inner_fold1": {"matched_ce": cal_in_ce, "matched_focal": cal_in_foc, "uniform_run2": cal_in_unif},
            "results": {"arm_a_ce_control": res_arma, "arm_c_focal_intervention": res_armc},
        }
        (EXP_DIR / "results_summary.json").write_text(json.dumps(summary, indent=2))
        write_status_atomic("completed", {"total_wall_clock_sec": summary["compute_costs_seconds"]["total_wall_clock"]})
        log(f"EXPERIMENT COMPLETE in {summary['compute_costs_seconds']['total_wall_clock']:.1f}s | Arm A CPM={res_arma['froc_raw']['cpm']:.4f} | Arm C CPM={res_armc['froc_raw']['cpm']:.4f}", log_path)
    except Exception as e:
        write_status_atomic("failed", {"error": str(e)})
        log(f"EXPERIMENT FAILED: {e}{chr(10)}{traceback.format_exc()}", log_path)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="CLEAR3D-CT Fold 0 Focal-Loss & Matched Inner Calibration Runner")
    parser.add_argument("--mode", choices=["verify-fixtures", "run-experiment"], default="verify-fixtures")
    args = parser.parse_args()
    verify_governance_hold()
    if args.mode == "verify-fixtures":
        subprocess.run([sys.executable, str(EXP_DIR / "test_focal_experiment_fixture.py")], check=True)
    else:
        run_experiment()


if __name__ == "__main__":
    main()
