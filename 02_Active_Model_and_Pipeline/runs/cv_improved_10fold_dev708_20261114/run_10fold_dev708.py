#!/usr/bin/env python3
"""
Prespecified 10-Fold Patient-Disjoint Cross-Validation (Development-Only 708 Scans, Subsets 2-9).
Strictly enforces:
- 0 overlap with Quarantined Confirmation Cohort (91 patients/scans: Subset 1 + 6 reviewed + 2 duplicate-volume scans).
- 0 overlap with Withheld Single-Split Test Cohort (89 patients/scans: cohort_test_WITHHOLD.jsonl, Subset 0).
- Reference release HOLD.
- Identical candidates, scan identities, matching rules, and official LUNA16 FROC evaluator settings.
- Prespecified recipe (Scout 12ep -> Full-Pool Out-of-Fold-Isolated Hard-Negative Mining -> Main 22ep -> 4-View TTA)
  with zero post-hoc threshold or ensemble tuning on evaluation outputs.
"""
import concurrent.futures
import csv
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

BASE = Path("/experiment-storage/chaidir/fawad-dsai")
ROOT = BASE / "CLEAR3D_CT project/02_Active_Model_and_Pipeline"
WORK = BASE / "CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64/source"
sys.path[:0] = [str(ROOT), str(WORK / "src"), str(WORK / "scripts")]

from clear3d_msf_model_20261106_002 import CLEAR3DMSFv2, build_focus_grids, make_inputs
from train_clear3d_msf_ensemble_20261106_004 import gpu_extract_from_p51

REFD = BASE / "data/reference/luna16-zenodo-3723295/resolved-reference-v1"
BASE_CV_DIR = ROOT / "runs/crossval-10fold-clear3d-20261109"
OUT_DIR = ROOT / "runs/cv_improved_10fold_dev708_20261114"
CKPT_DIR = OUT_DIR / "checkpoints"
PRED_DIR = OUT_DIR / "predictions"
BASE_DEV_PRED_DIR = OUT_DIR / "baseline_dev708_predictions"
EVAL_DIR = OUT_DIR / "official_cad_eval"
EVAL_PY = ROOT / "paper-ablation-verify-package-20261107-004/evaluator/noduleCADEvaluationLUNA16.py"
PYTHON_BIN = "/experiment-storage/chaidir/env/env-310-fawad/bin/python3"

N_FOLDS = 10
RAND_PER_FOLD = 1778       # 9 * 1778 = 16,002 -> [:16000] strictly from fold != k
RAND_NEG_COUNT = 16000
HARD_NEG_COUNT = 12000
SCOUT_EPOCHS = 12
SCOUT_STEPS = 125
MAIN_EPOCHS = 22
MAIN_STEPS = 125

def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT_DIR / "run.log", "a") as f:
        f.write(line + "\n")

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def update_status(stage: str, extra: dict = None):
    st = {"stage": stage, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if extra:
        st.update(extra)
    (OUT_DIR / "status.json").write_text(json.dumps(st, indent=2))

def load_scan_dev(scan_id: str):
    d = BASE / "CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-development-fold0-v1/prepared" / scan_id
    assert d.exists() and not d.is_symlink(), f"Invalid or non-dev scan path: {d}"
    geo = json.loads((d / "geometry.json").read_text())
    inv = np.linalg.inv(np.asarray(geo["affine"], dtype=np.float64))
    img = np.load(d / "image.npy", mmap_mode=None, allow_pickle=False)
    val = np.load(d / "valid.npy", mmap_mode=None, allow_pickle=False)
    return img, val, inv

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

def aug_gpu(p_sub_f16: np.ndarray, v_sub_bool: np.ndarray, rng: np.random.Generator, device: torch.device):
    out_p = torch.from_numpy(p_sub_f16).to(device=device, dtype=torch.float32).clone()
    out_v = torch.from_numpy(v_sub_bool).to(device=device, dtype=torch.float32).clone()
    for i in range(len(p_sub_f16)):
        if rng.random() < 0.5:
            dz, dy, dx = rng.integers(-1, 2, size=3)
            if dz or dy or dx:
                out_p[i] = torch.roll(out_p[i], shifts=(int(dz), int(dy), int(dx)), dims=(0, 1, 2))
                out_v[i] = torch.roll(out_v[i], shifts=(int(dz), int(dy), int(dx)), dims=(0, 1, 2))
        k = int(rng.integers(0, 4))
        if k:
            out_p[i] = torch.rot90(out_p[i], k, dims=(1, 2))
            out_v[i] = torch.rot90(out_v[i], k, dims=(1, 2))
        if rng.random() < 0.5:
            out_p[i] = torch.flip(out_p[i], dims=(0,))
            out_v[i] = torch.flip(out_v[i], dims=(0,))
        if rng.random() < 0.5:
            out_p[i] = torch.flip(out_p[i], dims=(2,))
            out_v[i] = torch.flip(out_v[i], dims=(2,))
        if rng.random() < 0.5:
            sc = float(rng.uniform(0.95, 1.05))
            sh = float(rng.uniform(-0.02, 0.02))
            out_p[i] = torch.clamp(out_p[i] * sc + sh, 0.0, 1.0)
    return out_p, out_v

def calc_metrics_and_froc(pred_csv: Path, eval_out_dir: Path, label_name: str, keep_eval_dir: bool = False):
    rows = list(csv.DictReader(open(pred_csv, newline="")))
    y_true = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    y_score = np.array([float(r["probability"]) for r in rows], dtype=np.float64)
    order = np.argsort(y_score)[::-1]
    yt = y_true[order]
    tp_c = np.cumsum(yt)
    fp_c = np.cumsum(1 - yt)
    prec_c = np.concatenate(([1.0], tp_c / np.maximum(tp_c + fp_c, 1)))
    rec_c = np.concatenate(([0.0], tp_c / max(tp_c[-1], 1)))
    pr_auc = float(np.sum((rec_c[1:] - rec_c[:-1]) * prec_c[1:]))
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    om = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[om] = np.arange(1, len(y_score) + 1)
    roc_auc = float((ranks[y_true == 1].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))
    pred = (y_score >= 0.5).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    sens = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1_pos = 2 * tp / max(2 * tp + fp + fn, 1)

    scans = sorted({r["seriesuid"] for r in rows})
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    suid_p = eval_out_dir / "seriesuids.csv"
    with open(suid_p, "w", newline="") as f:
        w = csv.writer(f)
        for s in scans:
            w.writerow([s])
    cmd = [
        PYTHON_BIN, str(EVAL_PY),
        str(REFD / "annotations.csv"), str(REFD / "annotations_excluded.csv"),
        str(suid_p), str(pred_csv), str(eval_out_dir)
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    froc_files = [p for p in eval_out_dir.glob("froc_*.txt") if "bootstrapping" not in p.name]
    fps_grid = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    sens_at_fps = {}
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
        f_arr = np.array([p[0] for p in pts])
        s_arr = np.array([p[1] for p in pts])
        for g in fps_grid:
            sens_at_fps[str(g)] = float(np.interp(g, f_arr, s_arr))
    cpm = float(np.mean(list(sens_at_fps.values()))) if sens_at_fps else 0.0

    # Parse bootstrap CI if available
    boot_ci = None
    boot_files = list(eval_out_dir.glob("froc_*_bootstrapping.csv"))
    if boot_files:
        try:
            b_lines = boot_files[0].read_text().splitlines()
            if len(b_lines) > 1:
                hdr = [float(x) for x in b_lines[0].strip().split(",")[1:]]
                h_arr = np.array(hdr)
                boot_cpms = []
                for bl in b_lines[1:]:
                    if not bl.strip():
                        continue
                    vals = [float(x) for x in bl.strip().split(",")[1:]]
                    v_arr = np.array(vals)
                    if not np.any(np.isnan(v_arr)):
                        c_b = float(np.mean([np.interp(g, h_arr, v_arr) for g in fps_grid]))
                        boot_cpms.append(c_b)
                if boot_cpms:
                    boot_cpms = np.array(boot_cpms)
                    boot_ci = {
                        "mean": float(np.mean(boot_cpms)),
                        "std": float(np.std(boot_cpms)),
                        "ci95_low": float(np.percentile(boot_cpms, 2.5)),
                        "ci95_high": float(np.percentile(boot_cpms, 97.5)),
                        "n_boot": int(len(boot_cpms))
                    }
        except Exception:
            pass

    log(f"[{label_name}] scans={len(scans)} cands={len(rows):,} ROC-AUC={roc_auc:.4f} PR-AUC={pr_auc:.4f} @0.5(TP={tp},FP={fp},FN={fn},Sens={sens:.4f},Prec={prec:.4f},F1={f1_pos:.4f}) | CPM={cpm:.4f}")
    return {
        "scans": len(scans),
        "candidates": len(rows),
        "pos_candidates": int(np.sum(y_true == 1)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp_05": tp,
        "fp_05": fp,
        "fn_05": fn,
        "sens_05": sens,
        "prec_05": prec,
        "f1_05": f1_pos,
        "cpm": cpm,
        "froc": sens_at_fps,
        "official_bootstrap": boot_ci,
    }

def main():
    t_start = time.time()
    for d in (OUT_DIR, CKPT_DIR, PRED_DIR, BASE_DEV_PRED_DIR, EVAL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    log("=== STEP 0: Pre-Flight Audit & Boundary Verification ===")
    sel = json.loads((ROOT / "selection-001.json").read_text())
    assert sel["reference_release"] == "HOLD", "reference_release must be HOLD!"

    spec = json.loads((ROOT / "crossval_protocol_specification_20261109.json").read_text())
    q_info = spec["cohort_inventory"]
    manifest_tmp = json.loads((ROOT / "patient_to_fold_manifest_20261109.json").read_text())["patient_to_fold"]
    coh_dev_rows = [json.loads(l) for l in open(BASE / "CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-cohort-fold0-v1/cohort_development.jsonl") if l.strip()]
    quarantined_scans = {x["scan"] for x in q_info["excluded_duplicate_volume_details"]}
    quarantined_patients = {r["patient_id"] for r in coh_dev_rows if r["patient_id"] not in manifest_tmp}
    assert len(quarantined_patients) == 91, f"Expected 91 quarantined patients, got {len(quarantined_patients)}"

    withheld_rows = [json.loads(l) for l in open(BASE / "CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-cohort-fold0-v1/cohort_test_WITHHOLD.jsonl") if l.strip()]
    withheld_suids = {r["series_uid"] for r in withheld_rows}
    withheld_pids = {r["patient_id"] for r in withheld_rows}
    assert len(withheld_suids) == 89, f"Expected 89 withheld test scans, got {len(withheld_suids)}"

    manifest = json.loads((ROOT / "patient_to_fold_manifest_20261109.json").read_text())
    p2f = manifest["patient_to_fold"]

    dev_rows_raw = [json.loads(l) for l in open(BASE / "CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-development-fold0-v1/prepared/scans.jsonl") if l.strip()]
    dev_scans = [s for s in dev_rows_raw if s["patient_id"] in p2f]
    for s in dev_scans:
        s["fold"] = int(p2f[s["patient_id"]])

    assert len(dev_scans) == 708, f"Expected 708 permitted dev scans, got {len(dev_scans)}"
    dev_pids = {s["patient_id"] for s in dev_scans}
    dev_suids = {s["series_uid"] for s in dev_scans}
    dev_sids = {s["scan_id"] for s in dev_scans}
    assert len(dev_pids) == 707, f"Expected 707 unique dev patients, got {len(dev_pids)}"
    assert len(dev_suids) == 708 and len(dev_sids) == 708, "Duplicate series_uid or scan_id in dev_scans!"
    assert not (dev_pids & quarantined_patients), "Quarantined patient leak!"
    assert not (dev_sids & quarantined_scans), "Quarantined duplicate scan leak!"
    assert not (dev_suids & withheld_suids) and not (dev_pids & withheld_pids), "Withheld test scan/patient leak!"

    # Verify pairwise patient disjointness across all 10 folds
    fold_patients = defaultdict(set)
    for s in dev_scans:
        fold_patients[s["fold"]].add(s["patient_id"])
    for i in range(N_FOLDS):
        for j in range(i + 1, N_FOLDS):
            assert not (fold_patients[i] & fold_patients[j]), f"Patient overlap between fold {i} and {j}!"

    dev_scan_map = {s["series_uid"]: s for s in dev_scans}
    dev_sid_map = {s["scan_id"]: s for s in dev_scans}

    # Write prespecified protocol manifest before training begins
    prespec = {
        "protocol_id": "CLEAR3D-10FOLD-DEV708-PRESPECIFIED-20261114",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reference_release": sel["reference_release"],
        "cohort_boundaries": {
            "permitted_dev_scans": len(dev_scans),
            "permitted_dev_patients": len(dev_pids),
            "quarantined_patients_excluded": len(quarantined_patients),
            "withheld_test_scans_excluded": len(withheld_suids),
            "pairwise_patient_overlap_across_folds": 0,
            "multi_scan_patient_LIDC_IDRI_0332_fold": p2f["LIDC-IDRI-0332"],
        },
        "prespecified_training_recipe": {
            "model_architecture": "CLEAR3DMSFv2(in_channels=3, dim=160, dropout=0.25)",
            "scout_epochs": SCOUT_EPOCHS,
            "scout_steps_per_epoch": SCOUT_STEPS,
            "scout_batch_composition": "12 pos + 20 rand_neg",
            "random_neg_bank_per_fold": RAND_NEG_COUNT,
            "hard_neg_bank_per_fold": HARD_NEG_COUNT,
            "main_epochs": MAIN_EPOCHS,
            "main_steps_per_epoch": MAIN_STEPS,
            "main_batch_composition": "10 pos + 12 rand_neg + 10 hard_neg",
            "tta_views": [[False, False], [True, False], [False, True], [True, True]],
            "threshold_or_ensemble_tuning_on_eval": False,
        },
    }
    (OUT_DIR / "prespecified_protocol_dev708.json").write_text(json.dumps(prespec, indent=2))
    log("Pre-flight verification passed and prespecified_protocol_dev708.json saved.")

    # Load positives cache & filter strictly to the 708 dev_scans
    pos_cache = np.load(BASE_CV_DIR / "cache/positives_cache.npz")
    pos_meta_all = json.loads(str(pos_cache["metadata"]))
    dev_pos_keep = [i for i, c in enumerate(pos_meta_all) if c["series"] in dev_scan_map]
    assert len(dev_pos_keep) == 1249, f"Expected 1249 dev positive candidates, got {len(dev_pos_keep)}"
    pos_meta = [pos_meta_all[i] for i in dev_pos_keep]
    for c in pos_meta:
        assert int(c["fold"]) == dev_scan_map[c["series"]]["fold"], "Positive cache fold mismatch!"
    pos_p51_all = pos_cache["p51"][dev_pos_keep].astype(np.float16)
    pos_v51_all = (pos_cache["v51"][dev_pos_keep] > 0.5)
    pos_folds = np.array([int(c["fold"]) for c in pos_meta], dtype=np.int64)
    del pos_cache
    gc.collect()
    log(f"Loaded {len(pos_p51_all):,} dev-only positive crops across {len({c['series'] for c in pos_meta})} scans.")

    # Build clean negative pool and full evaluation candidate pool across the 708 dev_scans
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
                clean_negs_by_fold[f_k].append({
                    "cand_idx": idx,
                    "series": suid,
                    "scan": s_info["scan_id"],
                    "fold": f_k,
                    "x": x, "y": y, "z": z
                })

    total_clean_negs = sum(len(v) for v in clean_negs_by_fold.values())
    total_eval_cands = sum(len(v) for v in eval_cands_by_scan.values())
    assert total_eval_cands == 603652, f"Expected 603,652 dev eval candidates, got {total_eval_cands}"
    log(f"Indexed {total_eval_cands:,} total dev candidates and {total_clean_negs:,} clean negative candidates across 10 folds.")

    # Sample RAND_PER_FOLD (1,778) random clean negatives per fold f (17,780 total)
    sampled_rand_list = []
    sampled_rand_cand_ids = set()
    fold_rand_ranges = {}
    for f_k in range(N_FOLDS):
        rng_c = random.Random(1000 + f_k)
        pool_f = clean_negs_by_fold[f_k]
        picked_f = rng_c.sample(pool_f, RAND_PER_FOLD)
        st_idx = len(sampled_rand_list)
        sampled_rand_list.extend(picked_f)
        fold_rand_ranges[f_k] = (st_idx, len(sampled_rand_list))
        for item in picked_f:
            sampled_rand_cand_ids.add(item["cand_idx"])

    # For each target fold k, build its 16,000 random negative indices strictly from f != k
    rand_bank_idx = {}
    for k in range(N_FOLDS):
        pool_idx = []
        for f_k in range(N_FOLDS):
            if f_k != k:
                s_i, e_i = fold_rand_ranges[f_k]
                pool_idx.extend(range(s_i, e_i))
        rng_shuf = random.Random(1500 + k)
        rng_shuf.shuffle(pool_idx)
        rand_bank_idx[k] = np.array(pool_idx[:RAND_NEG_COUNT], dtype=np.int64)
        # Verify 0 leakage from fold k
        assert all(sampled_rand_list[i]["fold"] != k for i in rand_bank_idx[k]), f"Random bank leak in fold {k}!"

    # Build remaining non-random clean negatives grouped by scan for Stage 2 mining
    rem_negs = []
    rem_by_scan = defaultdict(list)
    for f_k in range(N_FOLDS):
        for item in clean_negs_by_fold[f_k]:
            if item["cand_idx"] not in sampled_rand_cand_ids:
                r_idx = len(rem_negs)
                rem_negs.append(item)
                rem_by_scan[item["scan"]].append((r_idx, item["x"], item["y"], item["z"]))
    rem_folds = np.array([item["fold"] for item in rem_negs], dtype=np.int64)
    log(f"Random bank pool: {len(sampled_rand_list):,} crops | Remaining mining pool: {len(rem_negs):,} clean negatives.")

    # === STAGE 1A: Extract 17,780 random negative crops in parallel across cuda:1 and cuda:3 ===
    update_status("stage1_extract_random_negatives")
    t_s1 = time.time()
    rand_p51 = np.empty((len(sampled_rand_list), 51, 51, 51), dtype=np.float16)
    rand_v51 = np.empty((len(sampled_rand_list), 51, 51, 51), dtype=np.bool_)
    rand_by_scan = defaultdict(list)
    for idx_r, item in enumerate(sampled_rand_list):
        rand_by_scan[item["scan"]].append((idx_r, item["x"], item["y"], item["z"]))
    rand_scan_items = sorted(rand_by_scan.items())

    def extract_subset_to_arrays(scan_items_sub, out_p, out_v, dev_id):
        device = torch.device(f"cuda:{dev_id}")
        with torch.no_grad():
            for sid, items in scan_items_sub:
                img, val, inv = load_scan_dev(sid)
                zyxs = np.array([(inv @ np.array([x, y, z, 1.0]))[:3] for _, x, y, z in items], dtype=np.float64)
                img_t, val_t, c_pad, off = gpu_pad_and_slice_setup(img, val, zyxs, device)
                p_t, v_t = gpu_slice_chunk(img_t, val_t, c_pad, off)
                p_np = p_t.to(torch.float16).cpu().numpy()
                v_np = v_t.cpu().numpy()
                for j, (dst_idx, _, _, _) in enumerate(items):
                    out_p[dst_idx] = p_np[j]
                    out_v[dst_idx] = v_np[j]
        torch.cuda.synchronize(device)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(extract_subset_to_arrays, rand_scan_items[0::2], rand_p51, rand_v51, 1)
        f3 = ex.submit(extract_subset_to_arrays, rand_scan_items[1::2], rand_p51, rand_v51, 3)
        f1.result(); f3.result()
    log(f"Stage 1A complete: Extracted {len(sampled_rand_list):,} random negative crops in {time.time()-t_s1:.1f}s.")

    # === STAGE 1B: Train all 10 Scout models in parallel on cuda:1 (even folds) and cuda:3 (odd folds) ===
    update_status("stage1_train_scouts")
    t_s1b = time.time()
    scouts = {}
    for k in range(N_FOLDS):
        random.seed(42 + k)
        np.random.seed(42 + k)
        torch.manual_seed(42 + k)
        torch.cuda.manual_seed_all(42 + k)
        dev_k = torch.device("cuda:1" if k % 2 == 0 else "cuda:3")
        scouts[k] = CLEAR3DMSFv2(in_channels=3, dim=160, dropout=0.25).to(dev_k)

    def train_scout_worker(folds_list, dev_id):
        device = torch.device(f"cuda:{dev_id}")
        f_fine, f_ctx = build_focus_grids(device)
        targets = torch.tensor([1] * 12 + [0] * 20, device=device, dtype=torch.long)
        sw = torch.tensor([16.0 / 12] * 12 + [16.0 / 20] * 20, device=device, dtype=torch.float32)
        for k in folds_list:
            t_k = time.time()
            scout = scouts[k]
            scout_opt = torch.optim.AdamW(scout.parameters(), lr=8e-4, weight_decay=1e-2)
            scout_scaler = torch.amp.GradScaler("cuda")
            rng_scout = np.random.default_rng(3000 + k)
            pos_idx_k = np.flatnonzero(pos_folds != k)
            rand_idx_k = rand_bank_idx[k]
            for ep in range(SCOUT_EPOCHS):
                scout.train()
                lr = 1e-5 + 0.5 * (8e-4 - 1e-5) * (1.0 + math.cos(math.pi * ep / max(SCOUT_EPOCHS - 1, 1)))
                for g in scout_opt.param_groups:
                    g["lr"] = lr
                ep_loss = 0.0
                for _ in range(SCOUT_STEPS):
                    ip = pos_idx_k[rng_scout.choice(len(pos_idx_k), size=12, replace=True)]
                    ir = rand_idx_k[rng_scout.choice(len(rand_idx_k), size=20, replace=True)]
                    p_cat = np.concatenate([pos_p51_all[ip], rand_p51[ir]], axis=0)
                    v_cat = np.concatenate([pos_v51_all[ip], rand_v51[ir]], axis=0)
                    p_t, v_t = aug_gpu(p_cat, v_cat, rng_scout, device)
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        fi_g, fv_g, ci_g, cv_g = gpu_extract_from_p51(p_t, v_t)
                        vf, vc = make_inputs(fi_g, fv_g, ci_g, cv_g, f_fine, f_ctx)
                        scout_opt.zero_grad(set_to_none=True)
                        logits = scout(vf, vc)
                        loss = (F.cross_entropy(logits.float(), targets, reduction="none", label_smoothing=0.015) * sw).mean()
                    scout_scaler.scale(loss).backward()
                    scout_scaler.unscale_(scout_opt)
                    torch.nn.utils.clip_grad_norm_(scout.parameters(), 2.0)
                    scout_scaler.step(scout_opt)
                    scout_scaler.update()
                    ep_loss += float(loss.detach())
            torch.cuda.synchronize(device)
            log(f"  [GPU {dev_id}] Scout Fold {k} trained ({SCOUT_EPOCHS} ep) in {time.time()-t_k:.1f}s | final_loss={ep_loss/SCOUT_STEPS:.4f}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(train_scout_worker, [0, 2, 4, 6, 8], 1)
        f3 = ex.submit(train_scout_worker, [1, 3, 5, 7, 9], 3)
        f1.result(); f3.result()
    log(f"Stage 1B complete: All 10 Scouts trained in {time.time()-t_s1b:.1f}s.")

    # === STAGE 2: Multi-Scout Full-Pool Out-of-Fold-Isolated Hard-Negative Mining ===
    update_status("stage2_mine_hard_negatives")
    t_s2 = time.time()
    scout_state_dicts = {k: { key: val.cpu() for key, val in scouts[k].state_dict().items() } for k in range(N_FOLDS)}
    del scouts
    torch.cuda.empty_cache()

    scout_scores = np.full((N_FOLDS, len(rem_negs)), -1.0, dtype=np.float32)
    mine_scan_items = sorted(rem_by_scan.items())

    def mine_worker(scan_items_sub, dev_id):
        device = torch.device(f"cuda:{dev_id}")
        f_fine, f_ctx = build_focus_grids(device)
        local_scouts = {}
        for k in range(N_FOLDS):
            m = CLEAR3DMSFv2(in_channels=3, dim=160, dropout=0.25).to(device)
            m.load_state_dict(scout_state_dicts[k])
            m.eval()
            local_scouts[k] = m
        t_w0 = time.time()
        with torch.no_grad():
            for s_i, (sid, items) in enumerate(scan_items_sub, 1):
                f_s = dev_sid_map[sid]["fold"]
                active_k = [k for k in range(N_FOLDS) if k != f_s]
                img, val, inv = load_scan_dev(sid)
                zyxs = np.array([(inv @ np.array([x, y, z, 1.0]))[:3] for _, x, y, z in items], dtype=np.float64)
                img_t, val_t, c_pad, off = gpu_pad_and_slice_setup(img, val, zyxs, device)
                r_indices = np.array([r_idx for r_idx, _, _, _ in items], dtype=np.int64)
                for st in range(0, len(items), 256):
                    en = min(st + 256, len(items))
                    p_t, v_t = gpu_slice_chunk(img_t, val_t, c_pad[st:en], off)
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        fi_g, fv_g, ci_g, cv_g = gpu_extract_from_p51(p_t, v_t.float())
                        vf, vc = make_inputs(fi_g, fv_g, ci_g, cv_g, f_fine, f_ctx)
                        r_sub = r_indices[st:en]
                        for k in active_k:
                            lg = local_scouts[k](vf, vc).float()
                            pr = torch.softmax(lg, dim=1)[:, 1].cpu().numpy()
                            scout_scores[k, r_sub] = pr
                if s_i % 50 == 0 or s_i == len(scan_items_sub):
                    log(f"  [GPU {dev_id}] Mined {s_i}/{len(scan_items_sub)} scans in {time.time()-t_w0:.1f}s")
        torch.cuda.synchronize(device)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(mine_worker, mine_scan_items[0::2], 1)
        f3 = ex.submit(mine_worker, mine_scan_items[1::2], 3)
        f1.result(); f3.result()

    top_rem_idx = {}
    top_hard_scores = {}
    for k in range(N_FOLDS):
        valid_mask = (rem_folds != k)
        valid_idx = np.flatnonzero(valid_mask)
        sc_k = scout_scores[k, valid_idx]
        assert np.all(sc_k >= 0.0), f"Unscored negative found for fold {k}!"
        order_k = np.argsort(-sc_k, kind="mergesort")[:HARD_NEG_COUNT]
        top_rem_idx[k] = valid_idx[order_k]
        top_hard_scores[k] = sc_k[order_k].astype(np.float64)
        assert np.all(rem_folds[top_rem_idx[k]] != k), f"Hard negative fold leak in fold {k}!"
        log(f"  Fold {k}: mined {len(valid_idx):,} negs -> top1={top_hard_scores[k][0]:.4f}, top12000={top_hard_scores[k][-1]:.4f}")

    del scout_scores, scout_state_dicts
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Stage 2 complete: Full-pool hard-negative mining finished in {time.time()-t_s2:.1f}s.")

    # === STAGE 3A: Extract deduplicated union of top-12,000 hard negatives across all 10 folds ===
    update_status("stage3_extract_hard_negatives")
    t_s3a = time.time()
    unique_rem = np.unique(np.concatenate([top_rem_idx[k] for k in range(N_FOLDS)]))
    log(f"Extracting {len(unique_rem):,} unique hard-negative crops across all 10 folds (union of 10 x 12,000)...")
    hard_union_p51 = np.empty((len(unique_rem), 51, 51, 51), dtype=np.float16)
    hard_union_v51 = np.empty((len(unique_rem), 51, 51, 51), dtype=np.bool_)
    hard_by_scan = defaultdict(list)
    for u_i, r_idx in enumerate(unique_rem):
        item = rem_negs[int(r_idx)]
        hard_by_scan[item["scan"]].append((u_i, item["x"], item["y"], item["z"]))
    hard_scan_items = sorted(hard_by_scan.items())

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(extract_subset_to_arrays, hard_scan_items[0::2], hard_union_p51, hard_union_v51, 1)
        f3 = ex.submit(extract_subset_to_arrays, hard_scan_items[1::2], hard_union_p51, hard_union_v51, 3)
        f1.result(); f3.result()

    hard_local_idx = {k: np.searchsorted(unique_rem, top_rem_idx[k]) for k in range(N_FOLDS)}
    log(f"Stage 3A complete: Extracted {len(unique_rem):,} unique hard negative crops in {time.time()-t_s3a:.1f}s.")

    # === STAGE 3B: Train all 10 Main CLEAR3DMSFv2 models in parallel on cuda:1 and cuda:3 ===
    update_status("stage3_train_main_models")
    t_s3b = time.time()
    models = {}
    for k in range(N_FOLDS):
        torch.manual_seed(4200 + k)
        torch.cuda.manual_seed_all(4200 + k)
        dev_k = torch.device("cuda:1" if k % 2 == 0 else "cuda:3")
        models[k] = CLEAR3DMSFv2(in_channels=3, dim=160, dropout=0.25).to(dev_k)

    def train_main_worker(folds_list, dev_id):
        device = torch.device(f"cuda:{dev_id}")
        f_fine, f_ctx = build_focus_grids(device)
        targets = torch.tensor([1] * 10 + [0] * 22, device=device, dtype=torch.long)
        for k in folds_list:
            t_k = time.time()
            model = models[k]
            opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-2)
            scaler = torch.amp.GradScaler("cuda")
            rng_f = np.random.default_rng(5000 + k)
            pos_idx_k = np.flatnonzero(pos_folds != k)
            rand_idx_k = rand_bank_idx[k]
            hard_idx_k = hard_local_idx[k]
            tr_hw = 0.5 + 0.5 * top_hard_scores[k]
            tr_hw = tr_hw / tr_hw.sum()

            for ep in range(MAIN_EPOCHS):
                if ep == 0:
                    lr = 4e-4
                else:
                    prog = (ep - 1) / max(MAIN_EPOCHS - 2, 1)
                    lr = 1e-5 + 0.5 * (8e-4 - 1e-5) * (1.0 + math.cos(math.pi * prog))
                for g in opt.param_groups:
                    g["lr"] = lr
                cut_ep = int(0.70 * MAIN_EPOCHS)
                hard_share = 0.25 if ep < cut_ep else (0.08 + 0.17 * 0.5 * (1.0 + math.cos(math.pi * (ep - cut_ep) / max(MAIN_EPOCHS - 1 - cut_ep, 1))))
                w_pos = 16.0 / 10.0
                w_rand = (16.0 * (1.0 - hard_share)) / 12.0
                w_hard = (16.0 * hard_share) / 10.0
                sw = torch.tensor([w_pos] * 10 + [w_rand] * 12 + [w_hard] * 10, device=device, dtype=torch.float32)

                model.train()
                ep_loss = 0.0
                for _ in range(MAIN_STEPS):
                    ip = pos_idx_k[rng_f.choice(len(pos_idx_k), size=10, replace=True)]
                    ir = rand_idx_k[rng_f.choice(len(rand_idx_k), size=12, replace=False)]
                    ih = hard_idx_k[rng_f.choice(len(hard_idx_k), size=10, replace=False, p=tr_hw)]
                    p_cat = np.concatenate([pos_p51_all[ip], rand_p51[ir], hard_union_p51[ih]], axis=0)
                    v_cat = np.concatenate([pos_v51_all[ip], rand_v51[ir], hard_union_v51[ih]], axis=0)
                    p_t, v_t = aug_gpu(p_cat, v_cat, rng_f, device)
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        fi_g, fv_g, ci_g, cv_g = gpu_extract_from_p51(p_t, v_t)
                        vf, vc = make_inputs(fi_g, fv_g, ci_g, cv_g, f_fine, f_ctx)
                        opt.zero_grad(set_to_none=True)
                        logits = model(vf, vc)
                        loss = (F.cross_entropy(logits.float(), targets, reduction="none", label_smoothing=0.015) * sw).mean()
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    scaler.step(opt)
                    scaler.update()
                    ep_loss += float(loss.detach())
            torch.cuda.synchronize(device)
            ckpt_p = CKPT_DIR / f"fold_{k}.pt"
            torch.save({"model_state_dict": model.state_dict(), "fold": k, "epochs": MAIN_EPOCHS}, ckpt_p)
            log(f"  [GPU {dev_id}] Main Fold {k} trained ({MAIN_EPOCHS} ep) in {time.time()-t_k:.1f}s | final_loss={ep_loss/MAIN_STEPS:.4f} | ckpt={ckpt_p.name}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(train_main_worker, [0, 2, 4, 6, 8], 1)
        f3 = ex.submit(train_main_worker, [1, 3, 5, 7, 9], 3)
        f1.result(); f3.result()
    log(f"Stage 3B complete: All 10 Main CLEAR3DMSFv2 models trained and saved in {time.time()-t_s3b:.1f}s.")

    # Free training banks before Stage 4 OOF TTA evaluation
    del rand_p51, rand_v51, hard_union_p51, hard_union_v51, pos_p51_all, pos_v51_all
    gc.collect()
    torch.cuda.empty_cache()

    # === STAGE 4: Out-of-Fold 4-View TTA Scoring & Official LUNA16 Evaluation ===
    update_status("stage4_oof_tta_scoring")
    t_s4 = time.time()
    views = [(False, False), (True, False), (False, True), (True, True)]

    def score_folds_worker(folds_list, dev_id):
        device = torch.device(f"cuda:{dev_id}")
        f_fine, f_ctx = build_focus_grids(device)
        fieldnames = ["seriesuid", "coordX", "coordY", "coordZ", "probability", "candidate_id", "rank", "label"]
        with torch.no_grad():
            for k in folds_list:
                t_fk = time.time()
                model = models[k]
                model.eval()
                scans_k = sorted([s["series_uid"] for s in dev_scans if s["fold"] == k])
                pred_rows = []
                for suid in scans_k:
                    sid = dev_scan_map[suid]["scan_id"]
                    cands = eval_cands_by_scan[suid]
                    img, val, inv = load_scan_dev(sid)
                    zyxs = np.array([(inv @ np.array([x, y, z, 1.0]))[:3] for _, x, y, z, _ in cands], dtype=np.float64)
                    img_t, val_t, c_pad, off = gpu_pad_and_slice_setup(img, val, zyxs, device)
                    scan_probs = np.empty(len(cands), dtype=np.float32)
                    for st in range(0, len(cands), 128):
                        en = min(st + 128, len(cands))
                        p_t, v_t = gpu_slice_chunk(img_t, val_t, c_pad[st:en], off)
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            fi_g, fv_g, ci_g, cv_g = gpu_extract_from_p51(p_t, v_t.float())
                            vf, vc = make_inputs(fi_g, fv_g, ci_g, cv_g, f_fine, f_ctx)
                            acc = torch.zeros(en - st, device=device, dtype=torch.float32)
                            for fz, fx in views:
                                xf = vf; xc = vc
                                if fz:
                                    xf = torch.flip(xf, [2]); xc = torch.flip(xc, [2])
                                if fx:
                                    xf = torch.flip(xf, [4]); xc = torch.flip(xc, [4])
                                lg = model(xf, xc).float()
                                acc = acc + (lg[:, 1] - lg[:, 0])
                            scan_probs[st:en] = torch.sigmoid(acc * 0.25).cpu().numpy()
                    order = np.argsort(-scan_probs)
                    ranks = np.empty(len(cands), dtype=int)
                    ranks[order] = np.arange(1, len(cands) + 1)
                    for i, (c_idx, x, y, z, cls) in enumerate(cands):
                        pred_rows.append({
                            "seriesuid": suid,
                            "coordX": x,
                            "coordY": y,
                            "coordZ": z,
                            "probability": f"{float(scan_probs[i]):.8f}",
                            "candidate_id": f"{k}:{c_idx}",
                            "rank": int(ranks[i]),
                            "label": int(cls)
                        })
                pred_csv = PRED_DIR / f"fold_{k}_predictions.csv"
                with open(pred_csv, "w", newline="") as f_out:
                    w = csv.DictWriter(f_out, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerows(pred_rows)
                log(f"  [GPU {dev_id}] Scored Fold {k} ({len(scans_k)} scans, {len(pred_rows):,} cands) in {time.time()-t_fk:.1f}s")
        torch.cuda.synchronize(device)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(score_folds_worker, [0, 2, 4, 6, 8], 1)
        f3 = ex.submit(score_folds_worker, [1, 3, 5, 7, 9], 3)
        f1.result(); f3.result()
    log(f"Stage 4A complete: All 10 folds scored under 4-view TTA in {time.time()-t_s4:.1f}s.")

    # Pool predictions & build restricted-baseline dev708 predictions for exact paired comparison
    update_status("stage4_official_evaluation")
    fieldnames = ["seriesuid", "coordX", "coordY", "coordZ", "probability", "candidate_id", "rank", "label"]
    pooled_imp_csv = OUT_DIR / "pooled_oof_predictions.csv"
    pooled_base_csv = OUT_DIR / "baseline_dev708_pooled_oof_predictions.csv"

    with open(pooled_imp_csv, "w", newline="") as f_imp, open(pooled_base_csv, "w", newline="") as f_base:
        w_imp = csv.DictWriter(f_imp, fieldnames=fieldnames)
        w_base = csv.DictWriter(f_base, fieldnames=fieldnames)
        w_imp.writeheader()
        w_base.writeheader()
        for k in range(N_FOLDS):
            imp_k_csv = PRED_DIR / f"fold_{k}_predictions.csv"
            base_orig_csv = BASE_CV_DIR / f"predictions/fold_{k}_predictions.csv"
            base_dev_k_csv = BASE_DEV_PRED_DIR / f"fold_{k}_predictions.csv"

            imp_rows = list(csv.DictReader(open(imp_k_csv, newline="")))
            base_rows = [r for r in csv.DictReader(open(base_orig_csv, newline="")) if r["seriesuid"] in dev_scan_map]
            assert len(imp_rows) == len(base_rows), f"Row count mismatch on Fold {k}: {len(imp_rows)} vs {len(base_rows)}"
            with open(base_dev_k_csv, "w", newline="") as fb_k:
                wb_k = csv.DictWriter(fb_k, fieldnames=fieldnames)
                wb_k.writeheader()
                wb_k.writerows(base_rows)
            w_imp.writerows(imp_rows)
            w_base.writerows(base_rows)

    # Evaluate per-fold in parallel across CPU workers
    def eval_pair_fold(k):
        imp_res = calc_metrics_and_froc(PRED_DIR / f"fold_{k}_predictions.csv", EVAL_DIR / f"improved_fold_{k}", f"IMPROVED_FOLD_{k}")
        base_res = calc_metrics_and_froc(BASE_DEV_PRED_DIR / f"fold_{k}_predictions.csv", EVAL_DIR / f"baseline_fold_{k}", f"BASELINE_FOLD_{k}")
        return k, base_res, imp_res

    per_fold_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        for k, b_res, i_res in ex.map(eval_pair_fold, range(N_FOLDS)):
            per_fold_results[str(k)] = {
                "baseline_dev708": b_res,
                "improved_dev708": i_res,
                "delta_cpm": float(i_res["cpm"] - b_res["cpm"]),
            }

    log("Running Pooled OOF Official LUNA16 Evaluation (603,652 candidates across 708 dev scans)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fb = ex.submit(calc_metrics_and_froc, pooled_base_csv, EVAL_DIR / "baseline_pooled_dev708", "BASELINE_POOLED_DEV708")
        fi = ex.submit(calc_metrics_and_froc, pooled_imp_csv, EVAL_DIR / "improved_pooled_dev708", "IMPROVED_POOLED_DEV708")
        pooled_base_res = fb.result()
        pooled_imp_res = fi.result()

    base_cpms = np.array([per_fold_results[str(k)]["baseline_dev708"]["cpm"] for k in range(N_FOLDS)])
    imp_cpms = np.array([per_fold_results[str(k)]["improved_dev708"]["cpm"] for k in range(N_FOLDS)])
    diffs = imp_cpms - base_cpms

    # 10,000-iteration fold-cluster bootstrap CI for mean CPM and paired delta
    rng_b = np.random.default_rng(20261114)
    idx_boot = rng_b.integers(0, N_FOLDS, size=(10000, N_FOLDS))
    boot_imp = imp_cpms[idx_boot].mean(axis=1)
    boot_base = base_cpms[idx_boot].mean(axis=1)
    boot_diff = diffs[idx_boot].mean(axis=1)

    summary = {
        "protocol_id": "CLEAR3D-10FOLD-DEV708-PRESPECIFIED-20261114",
        "elapsed_seconds": round(time.time() - t_start, 2),
        "cohort": {
            "permitted_dev_scans": 708,
            "permitted_dev_patients": 707,
            "total_candidates": 603652,
            "total_positive_candidates": 1249,
            "total_reference_nodules": 946,
            "quarantined_patients_excluded": 91,
            "withheld_test_scans_excluded": 89,
            "reference_release": "HOLD"
        },
        "pooled_oof": {
            "baseline_dev708": pooled_base_res,
            "improved_dev708": pooled_imp_res,
            "delta_pooled_cpm": float(pooled_imp_res["cpm"] - pooled_base_res["cpm"]),
        },
        "fold_distribution": {
            "baseline_mean_cpm": float(np.mean(base_cpms)),
            "baseline_std_cpm": float(np.std(base_cpms, ddof=1)),
            "baseline_ci95_fold_boot": [float(np.percentile(boot_base, 2.5)), float(np.percentile(boot_base, 97.5))],
            "improved_mean_cpm": float(np.mean(imp_cpms)),
            "improved_std_cpm": float(np.std(imp_cpms, ddof=1)),
            "improved_ci95_fold_boot": [float(np.percentile(boot_imp, 2.5)), float(np.percentile(boot_imp, 97.5))],
            "mean_paired_delta_cpm": float(np.mean(diffs)),
            "std_paired_delta_cpm": float(np.std(diffs, ddof=1)),
            "paired_delta_ci95_fold_boot": [float(np.percentile(boot_diff, 2.5)), float(np.percentile(boot_diff, 97.5))],
            "folds_improved_count": int(np.sum(diffs > 0)),
        },
        "per_fold": per_fold_results,
    }
    (OUT_DIR / "results_summary.json").write_text(json.dumps(summary, indent=2))
    update_status("completed", {
        "elapsed_seconds": round(time.time() - t_start, 2),
        "pooled_baseline_cpm": pooled_base_res["cpm"],
        "pooled_improved_cpm": pooled_imp_res["cpm"],
        "mean_baseline_cpm": float(np.mean(base_cpms)),
        "mean_improved_cpm": float(np.mean(imp_cpms)),
    })
    log(f"=== ALL 10 FOLDS & POOLED EVALUATION COMPLETED in {time.time()-t_start:.1f}s! ===")
    log(f"Pooled OOF CPM: Baseline={pooled_base_res['cpm']:.4f} -> Improved={pooled_imp_res['cpm']:.4f} (Delta={pooled_imp_res['cpm']-pooled_base_res['cpm']:+.4f})")
    log(f"10-Fold Mean CPM: Baseline={np.mean(base_cpms):.4f}±{np.std(base_cpms,ddof=1):.4f} -> Improved={np.mean(imp_cpms):.4f}±{np.std(imp_cpms,ddof=1):.4f} (Delta={np.mean(diffs):+.4f})")

if __name__ == "__main__":
    main()
