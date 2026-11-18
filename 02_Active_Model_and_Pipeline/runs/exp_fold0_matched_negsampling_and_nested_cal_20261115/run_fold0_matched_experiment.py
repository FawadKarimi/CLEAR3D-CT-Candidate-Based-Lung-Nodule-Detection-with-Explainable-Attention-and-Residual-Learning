#!/usr/bin/env python3
"""
CLEAR3D-CT Fold 0 Single-Factor Negative-Sampling & Nested Calibration Experiment
=================================================================================
Strictly Exploratory Development Specification on dev708 (708 scans, 707 patients).
Grounded in:
  - run_10fold_dev708.py (SHA-256: fadbc9dfa0f0eb346687847bbc4c4c2962f16298db224fb15c2ff356d46411cb)
  - clear3d_msf_model_20261106_002.py (SHA-256: e83899960b095eab9f92fe104f13e3380c9c5bc48e6ebc82b173003a520914e7)
  - train_clear3d_msf_ensemble_20261106_004.py (SHA-256: b479ea29341791b8457b39f4241832bf7fe2cfd5320d513e27a69202a05769e8)

Key Properties:
  - Preserves "reference_release": "HOLD" in selection-001.json.
  - Never accesses withheld Subset 0 (89 scans) or quarantined Subset 1 / duplicate-volume scans (91 patients).
  - Splits Main Classifier RNG into FOUR independent PCG64 streams (rng_pos, rng_rand, rng_sec, rng_aug)
    while preserving 100% of the exact run_10fold_dev708.py architecture (154 tensors, 4,046,822 params,
    0 buffers), CrossEntropy(label_smoothing=0.015) + dynamic sw, 22-epoch warmup+cosine LR schedule,
    51x51x51 -> (33^3, 25^3) multi-scale inputs, and exact aug_gpu distributions.
  - Compares:
      * Saved_Control_SingleRNG: untouched historical checkpoints/fold_0.pt
      * Arm A (Split-RNG Control): Scout-mined top-12,000 hard-negative bank with weights 0.5 + 0.5*s
      * Arm B (Split-RNG Intervention): Uniform-random 12,000 clean-negative bank with uniform weights 1/12,000
  - Fits nested Platt calibration (T_0, b_0) and Positive-F1 thresholds (tau_raw_0*, tau_cal_0*) on
    Inner Fold 1 (75 scans) scored by M_inner_fold0 (trained strictly on Folds 2..9), and transfers
    them to Outer Fold 0 without touching Fold 0 labels during calibration or thresholding.
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
PREV_RUN = ROOT / "runs/cv_improved_10fold_dev708_20261114"
EXP_DIR = ROOT / "runs/exp_fold0_matched_negsampling_and_nested_cal_20261115"
SEL_PATH = ROOT / "selection-001.json"
EVAL_PY = ROOT / "paper-ablation-verify-package-20261107-004/evaluator/noduleCADEvaluationLUNA16.py"

sys.path[:0] = [str(ROOT), str(WORK / "src"), str(WORK / "scripts"), str(PREV_RUN)]
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
            f.write(line + "\n")


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
    assert len(dev_scans) == expected_scans, f"Expected {expected_scans} permitted dev scans, got {len(dev_scans)}"
    dev_pids = {s["patient_id"] for s in dev_scans}
    dev_suids = {s["series_uid"] for s in dev_scans}
    dev_sids = {s["scan_id"] for s in dev_scans}
    assert len(dev_pids) == expected_patients, f"Expected {expected_patients} unique dev patients, got {len(dev_pids)}"
    assert len(dev_suids) == expected_scans and len(dev_sids) == expected_scans, "Duplicate series_uid or scan_id in dev_scans!"
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
    """Exact LR and dynamic sample-weight schedule from run_10fold_dev708.py lines 617-629."""
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


def sample_aug_params_dev708(rng_aug: np.random.Generator, B: int = 32) -> Dict[str, np.ndarray]:
    """Exact per-patch random draws matching aug_gpu() in run_10fold_dev708.py lines 103-126."""
    do_shift = rng_aug.random(B) < 0.5
    shifts = np.where(
        do_shift[:, None],
        rng_aug.integers(-1, 2, size=(B, 3), dtype=np.int64),
        np.zeros((B, 3), dtype=np.int64),
    )
    ks = rng_aug.integers(0, 4, size=B, dtype=np.int64)
    flip_z = rng_aug.random(B) < 0.5
    flip_x = rng_aug.random(B) < 0.5
    do_int = rng_aug.random(B) < 0.5
    scale = np.where(do_int, rng_aug.uniform(0.95, 1.05, size=B), np.ones(B)).astype(np.float32)
    shift = np.where(do_int, rng_aug.uniform(-0.02, 0.02, size=B), np.zeros(B)).astype(np.float32)
    return {
        "shifts": shifts,
        "ks": ks,
        "flip_z": flip_z,
        "flip_x": flip_x,
        "scale": scale,
        "shift": shift,
    }


def aug_gpu_from_params(
    p_sub_f16: np.ndarray,
    v_sub_bool: np.ndarray,
    params: Dict[str, np.ndarray],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply pre-sampled aug_gpu parameters to (B, 51, 51, 51) image & valid mask tensors."""
    out_p = torch.from_numpy(p_sub_f16).to(device=device, dtype=torch.float32).clone()
    out_v = torch.from_numpy(v_sub_bool).to(device=device, dtype=torch.float32).clone()
    shifts = params["shifts"]
    ks = params["ks"]
    flip_z = params["flip_z"]
    flip_x = params["flip_x"]
    scale = params["scale"]
    shift = params["shift"]
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
        sc = float(scale[i])
        sh = float(shift[i])
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
    assert sec_p.ndim == 1 and len(sec_p) >= min_count, f"Need >= {min_count} secondary negatives, got {sec_p.shape}"
    assert np.all(np.isfinite(sec_p)), "sec_weights contains non-finite values"
    assert np.all(sec_p >= 0.0), "sec_weights contains negative values"
    assert np.count_nonzero(sec_p) >= min_count, f"Need >= {min_count} positive secondary weights"
    s = float(sec_p.sum())
    assert s > 0.0, "sec_weights sum must be > 0"
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
    n_pos = len(pos_p51)
    n_rand = len(rand_p51)
    last_loss = 0.0

    for ep in range(epochs):
        scout.train()
        lr = 1e-5 + 0.5 * (8e-4 - 1e-5) * (1.0 + math.cos(math.pi * ep / max(epochs - 1, 1)))
        for g in scout_opt.param_groups:
            g["lr"] = lr
        ep_loss = 0.0
        for _ in range(steps_per_epoch):
            ip = rng_scout.choice(n_pos, size=12, replace=True)
            ir = rng_scout.choice(n_rand, size=20, replace=True)
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
    rng_pos = streams["rng_pos"]
    rng_rand = streams["rng_rand"]
    rng_sec = streams["rng_sec"]
    rng_aug = streams["rng_aug"]

    n_pos = len(pos_p51)
    n_rand = len(rand_p51)
    n_sec = len(sec_p51)

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
            ip = rng_pos.choice(n_pos, size=10, replace=True)
            ir = rng_rand.choice(n_rand, size=12, replace=False)
            ih = rng_sec.choice(n_sec, size=10, replace=False, p=sec_p)
            aug_p = sample_aug_params_dev708(rng_aug, B=32)

            ip_trace.append(ip)
            ir_trace.append(ir)
            ih_trace.append(ih)

            p_cat = np.concatenate([pos_p51[ip], rand_p51[ir], sec_p51[ih]], axis=0)
            v_cat = np.concatenate([pos_v51[ip], rand_v51[ir], sec_v51[ih]], axis=0)
            p_t, v_t = aug_gpu_from_params(p_cat, v_cat, aug_p, device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                fi_g, fv_g, ci_g, cv_g = gpu_extract_from_p51(p_t, v_t)
                vf, vc = make_inputs(fi_g, fv_g, ci_g, cv_g, f_fine, f_ctx)
                opt.zero_grad(set_to_none=True)
                logits = model(vf, vc)
                loss = (F.cross_entropy(logits.float(), targets, reduction="none", label_smoothing=LABEL_SMOOTHING) * sw).mean()

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
    z_clip = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z_clip))


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
    macro_f1 = 0.5 * (pos_f1 + neg_f1)

    return {
        "threshold": float(threshold),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "N": int(n_total),
        "positives": int(tp + fn),
        "negatives": int(tn + fp),
        "sensitivity": float(sens),
        "precision": float(prec),
        "specificity": float(spec),
        "accuracy": float(acc),
        "positive_f1": float(pos_f1),
        "negative_f1": float(neg_f1),
        "macro_f1": float(macro_f1),
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

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    om = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[om] = np.arange(1, len(scores) + 1)
    roc_auc = float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1))
    return roc_auc, pr_auc


def select_max_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    assert len(scores) > 0 and len(scores) == len(y_true), "Non-empty matched y_true/scores required"
    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = y_true[order]

    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    total_pos = int(np.sum(y_true == 1))

    distinct_mask = np.r_[s_sorted[:-1] != s_sorted[1:], True]
    tp_d = tp_cum[distinct_mask]
    fp_d = fp_cum[distinct_mask]
    fn_d = total_pos - tp_d
    s_d = s_sorted[distinct_mask]

    denom = 2.0 * tp_d + fp_d + fn_d
    f1_d = np.where(denom > 0, (2.0 * tp_d) / denom, 0.0)
    # argmax returns first index in descending score order -> highest threshold on exact F1 ties
    best_idx = int(np.argmax(f1_d))
    tau_star = float(s_d[best_idx])
    return tau_star, compute_natural_candidate_metrics(y_true, scores, tau_star)


def fit_nested_platt_and_thresholds(p_raw_inner: np.ndarray, y_inner: np.ndarray) -> Dict[str, Any]:
    y_inner = np.asarray(y_inner, dtype=np.int64)
    p_raw_inner = np.asarray(p_raw_inner, dtype=np.float64)
    assert len(y_inner) > 0 and 0 < int(y_inner.sum()) < len(y_inner), "Both classes must be present in inner fold"
    z_inner = logit_np(p_raw_inner)

    def nll_obj(params: np.ndarray) -> float:
        log_t, b = float(params[0]), float(params[1])
        t = np.exp(log_t)
        u = np.clip((z_inner - b) / t, -40.0, 40.0)
        loss = np.mean(np.maximum(u, 0.0) - u * y_inner + np.log1p(np.exp(-np.abs(u))))
        return float(loss + 1e-5 * (log_t ** 2 + b ** 2))

    res = minimize(nll_obj, x0=np.array([0.0, 0.0], dtype=np.float64), method="BFGS")
    T_0 = float(np.exp(res.x[0]))
    b_0 = float(res.x[1])

    p_cal_inner = sigmoid_np((z_inner - b_0) / T_0)
    tau_raw_star, inner_raw_opt = select_max_f1_threshold(y_inner, p_raw_inner)
    tau_cal_star, inner_cal_opt = select_max_f1_threshold(y_inner, p_cal_inner)
    tau_cal_from_formula = float(sigmoid_np((logit_np(np.array([tau_raw_star]))[0] - b_0) / T_0))

    return {
        "optimizer_success": bool(res.success),
        "optimizer_message": str(res.message),
        "T_0": T_0,
        "b_0": b_0,
        "tau_raw_0_star": tau_raw_star,
        "tau_cal_0_star": tau_cal_star,
        "tau_cal_0_from_analytic_formula": tau_cal_from_formula,
        "tau_cal_analytic_abs_diff": float(abs(tau_cal_star - tau_cal_from_formula)),
        "inner_fold1_metrics": {
            "raw_at_0.5": compute_natural_candidate_metrics(y_inner, p_raw_inner, 0.5),
            "raw_at_tau_raw_star": inner_raw_opt,
            "cal_at_0.5": compute_natural_candidate_metrics(y_inner, p_cal_inner, 0.5),
            "cal_at_tau_cal_star": inner_cal_opt,
        },
    }


def apply_nested_calibration_to_outer(
    p_raw_outer: np.ndarray,
    y_outer: np.ndarray,
    cal_dict: Dict[str, Any],
) -> Dict[str, Any]:
    p_raw_outer = np.asarray(p_raw_outer, dtype=np.float64)
    y_outer = np.asarray(y_outer, dtype=np.int64)
    T_0 = float(cal_dict["T_0"])
    b_0 = float(cal_dict["b_0"])
    tau_raw_star = float(cal_dict["tau_raw_0_star"])
    tau_cal_star = float(cal_dict["tau_cal_0_star"])

    z_outer = logit_np(p_raw_outer)
    p_cal_outer = sigmoid_np((z_outer - b_0) / T_0)

    order_raw = np.argsort(-p_raw_outer, kind="mergesort")
    diffs_cal_sorted = np.diff(p_cal_outer[order_raw])
    rank_preserved = bool(np.all(diffs_cal_sorted <= 1e-12))
    raw_equiv_of_cal_05 = float(sigmoid_np(np.array([b_0]))[0])
    roc_auc, pr_auc = compute_roc_and_pr_auc(y_outer, p_raw_outer)

    return {
        "T_0": T_0,
        "b_0": b_0,
        "tau_raw_0_star": tau_raw_star,
        "tau_cal_0_star": tau_cal_star,
        "raw_equivalent_of_cal_0.5": raw_equiv_of_cal_05,
        "rank_order_preserved": rank_preserved,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
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
    cmd = [
        sys.executable,
        str(EVAL_PY),
        str(REFD / "annotations.csv"),
        str(REFD / "annotations_excluded.csv"),
        str(suid_p),
        str(pred_csv),
        str(eval_out_dir),
    ]
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
        f_arr = np.array([p[0] for p in pts], dtype=np.float64)
        s_arr = np.array([p[1] for p in pts], dtype=np.float64)
        for g in fps_grid:
            sens_at_fps[str(g)] = float(np.interp(g, f_arr, s_arr))
    cpm = float(np.mean(list(sens_at_fps.values()))) if sens_at_fps else 0.0
    return {"scans": len(scans), "cpm": cpm, "froc": sens_at_fps}


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
    sids = sorted(sid_to_suid.keys())

    with torch.no_grad():
        for sid, (img, val, inv) in iter_prefetched_scans(sids, max_prefetch=4):
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
                        "seriesuid": suid,
                        "coordX": x,
                        "coordY": y,
                        "coordZ": z,
                        "probability": f"{float(scan_probs[i]):.8f}",
                        "candidate_id": f"{fold_label}:{c_idx}",
                        "rank": int(ranks[i]),
                        "label": int(cls),
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
    assert torch.cuda.is_available(), "CUDA device required for --mode run-experiment"
    torch.cuda.set_per_process_memory_fraction(0.80, 0)
    device = torch.device("cuda:0")

    try:
        write_status_atomic("preflight")
        log("=== STEP 0: Pre-Flight Governance, Cohort & Provenance Verification ===", log_path)
        sel = verify_governance_hold()
        provenance = {
            "run_fold0_matched_experiment.py": sha256_file(EXP_DIR / "run_fold0_matched_experiment.py"),
            "test_matched_streams_fixture.py": sha256_file(EXP_DIR / "test_matched_streams_fixture.py"),
            "clear3d_msf_model_20261106_002.py": sha256_file(Path(inspect.getfile(CLEAR3DMSFv2))),
            "train_clear3d_msf_ensemble_20261106_004.py": sha256_file(Path(inspect.getfile(gpu_extract_from_p51))),
            "selection-001.json": sha256_file(SEL_PATH),
            "historical_fold0_ckpt": sha256_file(PREV_RUN / "checkpoints/fold_0.pt"),
        }
        log(f"Provenance hashes: {json.dumps(provenance)}", log_path)

        spec = json.loads((ROOT / "crossval_protocol_specification_20261109.json").read_text())
        q_info = spec["cohort_inventory"]
        manifest = json.loads((ROOT / "patient_to_fold_manifest_20261109.json").read_text())
        p2f = manifest["patient_to_fold"]
        coh_dev_rows = [json.loads(l) for l in open(BASE / "CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-cohort-fold0-v1/cohort_development.jsonl") if l.strip()]
        quarantined_scans = {x["scan"] for x in q_info["excluded_duplicate_volume_details"]}
        quarantined_patients = {r["patient_id"] for r in coh_dev_rows if r["patient_id"] not in p2f}
        assert len(quarantined_patients) == 91, f"Expected 91 quarantined patients, got {len(quarantined_patients)}"

        withheld_rows = [json.loads(l) for l in open(BASE / "CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-cohort-fold0-v1/cohort_test_WITHHOLD.jsonl") if l.strip()]
        withheld_suids = {r["series_uid"] for r in withheld_rows}
        withheld_pids = {r["patient_id"] for r in withheld_rows}
        assert len(withheld_suids) == 89, f"Expected 89 withheld test scans, got {len(withheld_suids)}"

        dev_rows_raw = [json.loads(l) for l in open(BASE / "CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-development-fold0-v1/prepared/scans.jsonl") if l.strip()]
        dev_scans = [s for s in dev_rows_raw if s["patient_id"] in p2f]
        for s in dev_scans:
            s["fold"] = int(p2f[s["patient_id"]])

        validate_dev_cohort(dev_scans, quarantined_patients, quarantined_scans, withheld_suids, withheld_pids)
        dev_scan_map = {s["series_uid"]: s for s in dev_scans}
        dev_sid_map = {s["scan_id"]: s for s in dev_scans}
        fold0_scans = [s for s in dev_scans if s["fold"] == 0]
        fold1_scans = [s for s in dev_scans if s["fold"] == 1]
        log(f"Cohort verified: 708 dev scans (Fold 0={len(fold0_scans)}, Fold 1={len(fold1_scans)}), 0 withheld/quarantined overlap.", log_path)

        pos_cache = np.load(BASE_CV_DIR / "cache/positives_cache.npz")
        pos_meta_all = json.loads(str(pos_cache["metadata"]))
        dev_pos_keep = [i for i, c in enumerate(pos_meta_all) if c["series"] in dev_scan_map]
        assert len(dev_pos_keep) == 1249, f"Expected 1249 dev positive candidates, got {len(dev_pos_keep)}"
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
                x, y, z = float(r[1]), float(r[2]), float(r[3])
                cls = int(r[4])
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

        sampled_rand_list = []
        sampled_rand_cand_ids = set()
        fold_rand_ranges = {}
        for f_k in range(N_FOLDS):
            rng_c = random.Random(1000 + f_k)
            picked_f = rng_c.sample(clean_negs_by_fold[f_k], RAND_PER_FOLD)
            st_idx = len(sampled_rand_list)
            sampled_rand_list.extend(picked_f)
            fold_rand_ranges[f_k] = (st_idx, len(sampled_rand_list))
            for item in picked_f:
                sampled_rand_cand_ids.add(item["cand_idx"])

        pool_idx_f0 = []
        for f_k in range(1, N_FOLDS):
            s_i, e_i = fold_rand_ranges[f_k]
            pool_idx_f0.extend(range(s_i, e_i))
        random.Random(1500).shuffle(pool_idx_f0)
        rand_bank_f0_idx = np.array(pool_idx_f0[:RAND_NEG_COUNT], dtype=np.int64)
        assert all(sampled_rand_list[i]["fold"] != 0 for i in rand_bank_f0_idx)

        inner_rand_idx = np.array([i for i, item in enumerate(sampled_rand_list) if item["fold"] not in (0, 1)], dtype=np.int64)

        rem_negs = []
        for f_k in range(N_FOLDS):
            for item in clean_negs_by_fold[f_k]:
                if item["cand_idx"] not in sampled_rand_cand_ids:
                    rem_negs.append(item)
        rem_folds = np.array([item["fold"] for item in rem_negs], dtype=np.int64)

        f0_rem_mask = rem_folds != 0
        f0_rem_global_idx = np.flatnonzero(f0_rem_mask)
        assert len(f0_rem_global_idx) == 514882, f"Expected 514,882 rem negs for Fold 0, got {len(f0_rem_global_idx)}"

        rng_bank = np.random.default_rng(9000)
        interv_local_idx = np.sort(rng_bank.choice(len(f0_rem_global_idx), size=SEC_NEG_COUNT, replace=False))
        interv_global_idx = f0_rem_global_idx[interv_local_idx]

        inner_rem_global_idx = np.flatnonzero((rem_folds != 0) & (rem_folds != 1))
        rng_inner_bank = np.random.default_rng(9100)
        inner_sec_global_idx = inner_rem_global_idx[np.sort(rng_inner_bank.choice(len(inner_rem_global_idx), size=SEC_NEG_COUNT, replace=False))]

        # === STAGE 1A: Single-pass extraction of random, uniform-ArmB, and inner-uniform banks ===
        write_status_atomic("stage1a_extract_banks")
        t_s1a = time.time()
        rand_p51 = np.empty((len(sampled_rand_list), 51, 51, 51), dtype=np.float16)
        rand_v51 = np.empty((len(sampled_rand_list), 51, 51, 51), dtype=np.bool_)
        armb_sec_p51 = np.empty((SEC_NEG_COUNT, 51, 51, 51), dtype=np.float16)
        armb_sec_v51 = np.empty((SEC_NEG_COUNT, 51, 51, 51), dtype=np.bool_)
        inner_sec_p51 = np.empty((SEC_NEG_COUNT, 51, 51, 51), dtype=np.float16)
        inner_sec_v51 = np.empty((SEC_NEG_COUNT, 51, 51, 51), dtype=np.bool_)

        extract_jobs_by_scan = defaultdict(list)
        for dst, item in enumerate(sampled_rand_list):
            extract_jobs_by_scan[item["scan"]].append(("rand", dst, item["x"], item["y"], item["z"]))
        for dst, g_idx in enumerate(interv_global_idx):
            item = rem_negs[int(g_idx)]
            extract_jobs_by_scan[item["scan"]].append(("armb", dst, item["x"], item["y"], item["z"]))
        for dst, g_idx in enumerate(inner_sec_global_idx):
            item = rem_negs[int(g_idx)]
            extract_jobs_by_scan[item["scan"]].append(("inner", dst, item["x"], item["y"], item["z"]))

        with torch.no_grad():
            for sid, (img, val, inv) in iter_prefetched_scans(sorted(extract_jobs_by_scan.keys()), max_prefetch=4):
                jobs = extract_jobs_by_scan[sid]
                zyxs = np.array([(inv @ np.array([x, y, z, 1.0]))[:3] for _, _, x, y, z in jobs], dtype=np.float64)
                img_t, val_t, c_pad, off = gpu_pad_and_slice_setup(img, val, zyxs, device)
                p_t, v_t = gpu_slice_chunk(img_t, val_t, c_pad, off)
                p_np = p_t.to(torch.float16).cpu().numpy()
                v_np = v_t.cpu().numpy()
                for j, (kind, dst, _, _, _) in enumerate(jobs):
                    if kind == "rand":
                        rand_p51[dst] = p_np[j]
                        rand_v51[dst] = v_np[j]
                    elif kind == "armb":
                        armb_sec_p51[dst] = p_np[j]
                        armb_sec_v51[dst] = v_np[j]
                    else:
                        inner_sec_p51[dst] = p_np[j]
                        inner_sec_v51[dst] = v_np[j]
        torch.cuda.synchronize(device)
        s1a_sec = float(time.time() - t_s1a)
        log(f"Stage 1A complete in {s1a_sec:.1f}s (rand={len(sampled_rand_list)}, armb_sec={SEC_NEG_COUNT}, inner_sec={SEC_NEG_COUNT}).", log_path)

        pos_idx_f0 = np.flatnonzero(pos_folds != 0)
        f0_pos_p51, f0_pos_v51 = pos_p51_all[pos_idx_f0], pos_v51_all[pos_idx_f0]
        f0_rand_p51, f0_rand_v51 = rand_p51[rand_bank_f0_idx], rand_v51[rand_bank_f0_idx]

        # === STAGE 1B: Train Scout Fold 0 ===
        write_status_atomic("stage1b_train_scout_fold0")
        scout_sd, scout_loss, scout_train_sec = train_scout_fold("cuda:0", f0_pos_p51, f0_pos_v51, f0_rand_p51, f0_rand_v51)
        torch.save({"model_state_dict": scout_sd, "fold": 0, "epochs": SCOUT_EPOCHS}, ckpt_dir / "scout_fold0.pt")
        log(f"Stage 1B Scout Fold 0 trained (12 ep) in {scout_train_sec:.1f}s | final_loss={scout_loss:.4f}", log_path)

        # === STAGE 2: Scout Mining on Folds 1..9 (514,882 clean negatives) ===
        write_status_atomic("stage2_mine_fold0")
        t_mine = time.time()
        scout_m = CLEAR3DMSFv2(in_channels=3, dim=160, dropout=0.25).to(device)
        scout_m.load_state_dict(scout_sd)
        scout_m.eval()
        f_fine, f_ctx = build_focus_grids(device)
        f0_rem_by_scan = defaultdict(list)
        for local_i, g_idx in enumerate(f0_rem_global_idx):
            item = rem_negs[int(g_idx)]
            f0_rem_by_scan[item["scan"]].append((local_i, item["x"], item["y"], item["z"]))

        scout_scores = np.empty(len(f0_rem_global_idx), dtype=np.float32)
        mine_sids = sorted(f0_rem_by_scan.keys())
        with torch.no_grad():
            for s_i, (sid, (img, val, inv)) in enumerate(iter_prefetched_scans(mine_sids, max_prefetch=4), 1):
                items = f0_rem_by_scan[sid]
                zyxs = np.array([(inv @ np.array([x, y, z, 1.0]))[:3] for _, x, y, z in items], dtype=np.float64)
                img_t, val_t, c_pad, off = gpu_pad_and_slice_setup(img, val, zyxs, device)
                l_idx = np.array([li for li, _, _, _ in items], dtype=np.int64)
                for st in range(0, len(items), 512):
                    en = min(st + 512, len(items))
                    p_t, v_t = gpu_slice_chunk(img_t, val_t, c_pad[st:en], off)
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        fi_g, fv_g, ci_g, cv_g = gpu_extract_from_p51(p_t, v_t.float())
                        vf, vc = make_inputs(fi_g, fv_g, ci_g, cv_g, f_fine, f_ctx)
                        lg = scout_m(vf, vc).float()
                        scout_scores[l_idx[st:en]] = torch.softmax(lg, dim=1)[:, 1].cpu().numpy()
                if s_i % 150 == 0 or s_i == len(mine_sids):
                    log(f"  Mined {s_i}/{len(mine_sids)} scans in {time.time()-t_mine:.1f}s", log_path)
        del scout_m, scout_sd
        torch.cuda.empty_cache()

        top_local = np.argsort(-scout_scores)[:SEC_NEG_COUNT]
        top_scores = scout_scores[top_local].astype(np.float64)
        arma_global_idx = f0_rem_global_idx[top_local]
        arma_sec_weights = (0.5 + 0.5 * top_scores) / np.sum(0.5 + 0.5 * top_scores)
        mine_score_sec = float(time.time() - t_mine)
        log(f"Stage 2A mining complete in {mine_score_sec:.1f}s | top1={top_scores[0]:.4f}, top12000={top_scores[-1]:.4f}", log_path)

        # === STAGE 2B: Extract Top-12,000 Hard Negatives for Arm A ===
        write_status_atomic("stage2b_extract_hard_bank")
        t_hb = time.time()
        arma_sec_p51 = np.empty((SEC_NEG_COUNT, 51, 51, 51), dtype=np.float16)
        arma_sec_v51 = np.empty((SEC_NEG_COUNT, 51, 51, 51), dtype=np.bool_)
        arma_by_scan = defaultdict(list)
        for dst, g_idx in enumerate(arma_global_idx):
            item = rem_negs[int(g_idx)]
            arma_by_scan[item["scan"]].append((dst, item["x"], item["y"], item["z"]))
        with torch.no_grad():
            for sid, (img, val, inv) in iter_prefetched_scans(sorted(arma_by_scan.keys()), max_prefetch=4):
                items = arma_by_scan[sid]
                zyxs = np.array([(inv @ np.array([x, y, z, 1.0]))[:3] for _, x, y, z in items], dtype=np.float64)
                img_t, val_t, c_pad, off = gpu_pad_and_slice_setup(img, val, zyxs, device)
                p_t, v_t = gpu_slice_chunk(img_t, val_t, c_pad, off)
                p_np = p_t.to(torch.float16).cpu().numpy()
                v_np = v_t.cpu().numpy()
                for j, (dst, _, _, _) in enumerate(items):
                    arma_sec_p51[dst] = p_np[j]
                    arma_sec_v51[dst] = v_np[j]
        torch.cuda.synchronize(device)
        hb_extract_sec = float(time.time() - t_hb)
        log(f"Stage 2B extracted 12,000 hard negs in {hb_extract_sec:.1f}s", log_path)

        # === STAGE 3A: Arm A (Split-RNG Control) ===
        write_status_atomic("stage3a_train_arm_a")
        arma_sd, arma_loss, arma_traces, arma_train_sec = train_main_model_split_rng(
            "cuda:0", f0_pos_p51, f0_pos_v51, f0_rand_p51, f0_rand_v51, arma_sec_p51, arma_sec_v51, arma_sec_weights,
            seed_init=4200, seed_stream_base=5000,
        )
        torch.save({"model_state_dict": arma_sd, "arm": "arm_a_split_rng_control", "epochs": MAIN_EPOCHS}, ckpt_dir / "arm_a_split_rng_control.pt")
        log(f"Stage 3A Arm A trained in {arma_train_sec:.1f}s | final_loss={arma_loss:.4f}", log_path)

        # === STAGE 3B: Arm B (Split-RNG Intervention) ===
        write_status_atomic("stage3b_train_arm_b")
        armb_weights = np.full(SEC_NEG_COUNT, 1.0 / SEC_NEG_COUNT, dtype=np.float64)
        armb_sd, armb_loss, armb_traces, armb_train_sec = train_main_model_split_rng(
            "cuda:0", f0_pos_p51, f0_pos_v51, f0_rand_p51, f0_rand_v51, armb_sec_p51, armb_sec_v51, armb_weights,
            seed_init=4200, seed_stream_base=5000,
        )
        torch.save({"model_state_dict": armb_sd, "arm": "arm_b_split_rng_intervention", "epochs": MAIN_EPOCHS}, ckpt_dir / "arm_b_split_rng_intervention.pt")
        log(f"Stage 3B Arm B trained in {armb_train_sec:.1f}s | final_loss={armb_loss:.4f}", log_path)

        assert arma_traces["ip_sha256"] == armb_traces["ip_sha256"], "ip stream mismatch!"
        assert arma_traces["ir_sha256"] == armb_traces["ir_sha256"], "ir stream mismatch!"
        assert arma_traces["ih_sha256"] != armb_traces["ih_sha256"], "ih stream should differ!"

        # === STAGE 3C: Inner Fold 0 Calibration Model (Folds 2..9 only; zero Fold 0 or Fold 1 exposure) ===
        write_status_atomic("stage3c_train_inner_fold0")
        inner_pos_idx = np.flatnonzero((pos_folds != 0) & (pos_folds != 1))
        inner_weights = np.full(SEC_NEG_COUNT, 1.0 / SEC_NEG_COUNT, dtype=np.float64)
        inner_sd, inner_loss, _, inner_train_sec = train_main_model_split_rng(
            "cuda:0", pos_p51_all[inner_pos_idx], pos_v51_all[inner_pos_idx],
            rand_p51[inner_rand_idx], rand_v51[inner_rand_idx],
            inner_sec_p51, inner_sec_v51, inner_weights,
            seed_init=4201, seed_stream_base=5100,
        )
        torch.save({"model_state_dict": inner_sd, "arm": "m_inner_fold0_folds2to9", "epochs": MAIN_EPOCHS}, ckpt_dir / "inner_fold0_model.pt")
        log(f"Stage 3C M_inner_fold0 (Folds 2..9) trained in {inner_train_sec:.1f}s | final_loss={inner_loss:.4f}", log_path)

        del rand_p51, rand_v51, arma_sec_p51, arma_sec_v51, armb_sec_p51, armb_sec_v51, inner_sec_p51, inner_sec_v51, pos_p51_all, pos_v51_all
        gc.collect()

        # === STAGE 4: Score Outer Fold 0 (70 scans) with Arm A & Arm B in one pass ===
        write_status_atomic("stage4_score_outer_fold0")
        f0_rows, f0_score_sec = score_scans_4view_tta(
            {"arm_a": arma_sd, "arm_b": armb_sd}, fold0_scans, eval_cands_by_scan, fold_label=0, device=device,
        )
        arma_csv = pred_dir / "arm_a_fold0_predictions.csv"
        armb_csv = pred_dir / "arm_b_fold0_predictions.csv"
        write_pred_csv(f0_rows["arm_a"], arma_csv)
        write_pred_csv(f0_rows["arm_b"], armb_csv)
        log(f"Stage 4 scored Outer Fold 0 (70 scans, 2 arms) in {f0_score_sec:.1f}s", log_path)

        # === STAGE 5: Score Inner Fold 1 (75 scans) with M_inner_fold0 ===
        write_status_atomic("stage5_score_inner_fold1")
        f1_rows, f1_score_sec = score_scans_4view_tta(
            {"inner": inner_sd}, fold1_scans, eval_cands_by_scan, fold_label=1, device=device,
        )
        inner_csv = pred_dir / "inner_fold1_predictions.csv"
        write_pred_csv(f1_rows["inner"], inner_csv)
        log(f"Stage 5 scored Inner Fold 1 (75 scans) in {f1_score_sec:.1f}s", log_path)

        # === STAGE 6: Nested Calibration + Natural-Distribution Metrics + Official LUNA16 FROC ===
        write_status_atomic("stage6_eval_and_calibration")
        p_in = np.array([float(r["probability"]) for r in f1_rows["inner"]], dtype=np.float64)
        y_in = np.array([int(r["label"]) for r in f1_rows["inner"]], dtype=np.int64)
        cal_fit = fit_nested_platt_and_thresholds(p_in, y_in)

        def eval_arm(name: str, rows: List[Dict[str, Any]], raw_csv: Path) -> Dict[str, Any]:
            p_raw = np.array([float(r["probability"]) for r in rows], dtype=np.float64)
            y_out = np.array([int(r["label"]) for r in rows], dtype=np.int64)
            outer_eval = apply_nested_calibration_to_outer(p_raw, y_out, cal_fit)
            cal_rows = []
            for r, pc in zip(rows, outer_eval["p_cal_outer"]):
                rc = dict(r)
                rc["probability"] = f"{float(pc):.8f}"
                cal_rows.append(rc)
            cal_csv = pred_dir / f"{name}_fold0_calibrated_predictions.csv"
            write_pred_csv(cal_rows, cal_csv)
            froc_raw = run_official_luna16_eval(raw_csv, eval_dir / f"{name}_raw")
            froc_cal = run_official_luna16_eval(cal_csv, eval_dir / f"{name}_cal")
            return {
                "roc_auc": outer_eval["roc_auc"],
                "pr_auc": outer_eval["pr_auc"],
                "rank_order_preserved": outer_eval["rank_order_preserved"],
                "metrics": outer_eval["metrics"],
                "froc_raw": froc_raw,
                "froc_cal": froc_cal,
                "froc_cpm_abs_diff": float(abs(froc_raw["cpm"] - froc_cal["cpm"])),
            }

        res_arma = eval_arm("arm_a", f0_rows["arm_a"], arma_csv)
        res_armb = eval_arm("arm_b", f0_rows["arm_b"], armb_csv)

        # Historical single-RNG reference scored under the same Inner Fold 1 calibration
        with open(PREV_RUN / "predictions/fold_0_predictions.csv", newline="") as f:
            hist_rows = list(csv.DictReader(f))
        res_hist = eval_arm("historical_single_rng", hist_rows, PREV_RUN / "predictions/fold_0_predictions.csv")

        summary = {
            "governance": {
                "reference_release": sel.get("reference_release"),
                "withheld_subset0_excluded": True,
                "quarantined_subset1_excluded": True,
            },
            "provenance_sha256": provenance,
            "compute_costs_seconds": {
                "stage1a_bank_extraction": s1a_sec,
                "scout_training_12ep": scout_train_sec,
                "scout_mining_514882_cands": mine_score_sec,
                "hard_bank_extraction_12000": hb_extract_sec,
                "arm_a_training_22ep": arma_train_sec,
                "arm_b_training_22ep": armb_train_sec,
                "inner_fold0_training_22ep": inner_train_sec,
                "outer_fold0_tta_scoring_2arms": f0_score_sec,
                "inner_fold1_tta_scoring": f1_score_sec,
                "total_wall_clock": float(time.time() - t_all),
            },
            "rng_stream_traces": {"arm_a": arma_traces, "arm_b": armb_traces},
            "nested_calibration_fit_on_inner_fold1": cal_fit,
            "results": {
                "arm_a_split_rng_control": res_arma,
                "arm_b_split_rng_intervention": res_armb,
                "historical_single_rng_reference": res_hist,
            },
        }
        (EXP_DIR / "results_summary.json").write_text(json.dumps(summary, indent=2))
        write_status_atomic("completed", {"total_wall_clock_sec": summary["compute_costs_seconds"]["total_wall_clock"]})
        log(f"EXPERIMENT COMPLETE in {summary['compute_costs_seconds']['total_wall_clock']:.1f}s | Arm A CPM={res_arma['froc_raw']['cpm']:.4f} | Arm B CPM={res_armb['froc_raw']['cpm']:.4f}", log_path)
    except Exception as e:
        write_status_atomic("failed", {"error": str(e)})
        log(f"EXPERIMENT FAILED: {e}\n{traceback.format_exc()}", log_path)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="CLEAR3D-CT Revised Fold 0 Experiment Runner")
    parser.add_argument("--mode", choices=["verify-fixtures", "run-experiment"], default="verify-fixtures")
    args = parser.parse_args()
    verify_governance_hold()
    if args.mode == "verify-fixtures":
        subprocess.run([sys.executable, str(EXP_DIR / "test_matched_streams_fixture.py")], check=True)
    else:
        run_experiment()


if __name__ == "__main__":
    main()
