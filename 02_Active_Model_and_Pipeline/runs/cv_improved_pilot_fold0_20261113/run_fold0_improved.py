#!/usr/bin/env python3
"""
Controlled Fold 0 Experiment: Baseline 10-Fold CV vs. Hard-Negative Mining + 3D Aug + 22-Epoch Budget.
Strictly zero leakage with Fold 0; preserves frozen Run 004 and baseline CV artifacts.
"""
import csv, hashlib, json, math, os, random, subprocess, sys, tempfile, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

BASE = Path("/experiment-storage/chaidir/fawad-dsai")
ROOT = BASE / "CLEAR3D_CT project/02_Active_Model_and_Pipeline"
WORK = BASE / "CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64/source"
sys.path[:0] = [str(ROOT), str(WORK / "src"), str(WORK / "scripts")]

from clear3d_msf_model_20261106_002 import CLEAR3DMSFv2, build_focus_grids, make_inputs
from train_clear3d_msf_ensemble_20261106_004 import gpu_extract_from_p51, slice_p51_batch
from lidc3d.geometry import crop

REFD = BASE / "data/reference/luna16-zenodo-3723295/resolved-reference-v1"
BASE_CV_DIR = ROOT / "runs/crossval-10fold-clear3d-20261109"
OUT_DIR = ROOT / "runs/cv_improved_pilot_fold0_20261113"
EVAL_PY = ROOT / "paper-ablation-verify-package-20261107-004/evaluator/noduleCADEvaluationLUNA16.py"

FOLD_K = 0
DEVICE_ID = 1
RAND_NEG_COUNT = 16000
HARD_NEG_COUNT = 12000
SCOUT_EPOCHS = 12
SCOUT_STEPS = 125
MAIN_EPOCHS = 22
MAIN_STEPS = 125

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def load_scan_universal(scan_id: str):
    p1 = BASE / "CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-development-fold0-v1/prepared" / scan_id
    p2 = BASE / "CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-test-fold0-v1/prepared" / scan_id
    d = p1 if p1.exists() else p2
    geo = json.loads((d / "geometry.json").read_text())
    inv = np.linalg.inv(np.asarray(geo["affine"], dtype=np.float64))
    img = np.load(d / "image.npy", mmap_mode="r", allow_pickle=False)
    val = np.load(d / "valid.npy", mmap_mode="r", allow_pickle=False)
    return img, val, inv

def augment_p51_batch(p_sub: np.ndarray, v_sub: np.ndarray, rng: np.random.Generator):
    b = len(p_sub)
    out_p = p_sub.copy()
    out_v = v_sub.copy()
    for i in range(b):
        # Random +/-1 voxel 3D shift (zero-padded edge)
        if rng.random() < 0.5:
            dz, dy, dx = rng.integers(-1, 2, size=3)
            if dz or dy or dx:
                out_p[i] = np.roll(out_p[i], shift=(int(dz), int(dy), int(dx)), axis=(0, 1, 2))
                out_v[i] = np.roll(out_v[i], shift=(int(dz), int(dy), int(dx)), axis=(0, 1, 2))
        # Random 90-deg axial rotation
        k = int(rng.integers(0, 4))
        if k:
            out_p[i] = np.rot90(out_p[i], k, axes=(1, 2))
            out_v[i] = np.rot90(out_v[i], k, axes=(1, 2))
        # Random flips
        if rng.random() < 0.5:
            out_p[i] = out_p[i, ::-1]; out_v[i] = out_v[i, ::-1]
        if rng.random() < 0.5:
            out_p[i] = out_p[i, :, :, ::-1]; out_v[i] = out_v[i, :, :, ::-1]
        # Mild intensity jitter
        if rng.random() < 0.5:
            sc = float(rng.uniform(0.95, 1.05))
            sh = float(rng.uniform(-0.02, 0.02))
            out_p[i] = np.clip(out_p[i] * sc + sh, 0.0, 1.0)
    return out_p, out_v

def calc_metrics_and_froc(pred_csv: Path, label_name: str):
    rows = list(csv.DictReader(open(pred_csv, newline="")))
    y_true = np.array([int(r["label"]) for r in rows])
    y_score = np.array([float(r["probability"]) for r in rows])
    order = np.argsort(y_score)[::-1]
    yt = y_true[order]
    tp_c = np.cumsum(yt)
    fp_c = np.cumsum(1 - yt)
    prec_c = np.concatenate(([1.0], tp_c / np.maximum(tp_c + fp_c, 1)))
    rec_c = np.concatenate(([0.0], tp_c / max(tp_c[-1], 1)))
    pr_auc = float(np.sum((rec_c[1:] - rec_c[:-1]) * prec_c[1:]))
    pos = y_score[y_true == 1]; neg = y_score[y_true == 0]
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
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        suid_p = td_p / "seriesuids.csv"
        out_p = td_p / "eval_out"
        out_p.mkdir()
        with open(suid_p, "w", newline="") as f:
            w = csv.writer(f)
            for s in scans: w.writerow([s])
        cmd = [
            "/experiment-storage/chaidir/env/env-310-fawad/bin/python3", str(EVAL_PY),
            str(REFD / "annotations.csv"), str(REFD / "annotations_excluded.csv"),
            str(suid_p), str(pred_csv), str(out_p)
        ]
        subprocess.run(cmd, capture_output=True, text=True)
        froc_files = list(out_p.glob("froc_*.txt"))
        fps_grid = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        sens_at_fps = {}
        if froc_files:
            pts = []
            for line in froc_files[0].read_text().splitlines():
                parts = line.strip().split(",")
                if len(parts) == 3:
                    try: pts.append((float(parts[0]), float(parts[1])))
                    except ValueError: pass
            pts.sort(key=lambda x: x[0])
            f_arr = np.array([p[0] for p in pts])
            s_arr = np.array([p[1] for p in pts])
            for g in fps_grid:
                sens_at_fps[g] = float(np.interp(g, f_arr, s_arr))
        cpm = float(np.mean(list(sens_at_fps.values()))) if sens_at_fps else 0.0
    log(f"[{label_name}] ROC-AUC={roc_auc:.4f} PR-AUC={pr_auc:.4f} @0.5(TP={tp},FP={fp},FN={fn},Sens={sens:.4f},Prec={prec:.4f},F1={f1_pos:.4f}) | CPM={cpm:.4f}")
    if sens_at_fps:
        log(f"[{label_name}] FROC sensitivities: " + ", ".join(f"{g}:{v:.4f}" for g, v in sens_at_fps.items()))
    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "tp": tp, "fp": fp, "fn": fn, "sens": sens, "prec": prec, "f1_pos": f1_pos, "cpm": cpm, "froc": sens_at_fps}

def main():
    t0 = time.time()
    # Fix RNG BEFORE any model instantiation
    random.seed(42 + FOLD_K)
    np.random.seed(42 + FOLD_K)
    torch.manual_seed(42 + FOLD_K)
    torch.cuda.manual_seed_all(42 + FOLD_K)

    device = torch.device(f"cuda:{DEVICE_ID}")
    f_fine, f_ctx = build_focus_grids(device)

    manifest = json.load(open(ROOT / "patient_to_fold_manifest_20261109.json"))
    p2f = manifest["patient_to_fold"]
    dev_scans = [json.loads(l) for l in open(BASE / "CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-development-fold0-v1/prepared/scans.jsonl") if l.strip()]
    test_scans = [json.loads(l) for l in open(BASE / "CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-cohort-fold0-v1/cohort_test_WITHHOLD.jsonl") if l.strip()]
    scan_map = {s["series_uid"]: s for s in dev_scans + test_scans if s["patient_id"] in p2f}
    held_out_scans = {suid: s for suid, s in scan_map.items() if p2f[s["patient_id"]] == FOLD_K}
    train_scans = {suid: s for suid, s in scan_map.items() if p2f[s["patient_id"]] != FOLD_K}
    log(f"Fold {FOLD_K} | train_scans={len(train_scans)}, held_out_scans={len(held_out_scans)}")

    # Load positives cache
    pos_cache = np.load(BASE_CV_DIR / "cache/positives_cache.npz")
    pos_meta_all = json.loads(str(pos_cache["metadata"]))
    train_pos_idx = [i for i, c in enumerate(pos_meta_all) if c["fold"] != FOLD_K]
    train_pos_p51 = pos_cache["p51"][train_pos_idx].astype(np.float32)
    train_pos_v51 = pos_cache["v51"][train_pos_idx].astype(np.float32)
    log(f"Loaded {len(train_pos_p51)} training positives")

    # Build clean negative pool from Fold 1-9 only
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

    train_neg_candidates = []
    with open(REFD / "candidates_V2.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        for idx, r in enumerate(reader):
            suid = r[0]
            if suid in train_scans and int(r[4]) == 0:
                c = (float(r[1]), float(r[2]), float(r[3]))
                if any(math.dist(c, ac) <= rad for ac, rad in exc_refs.get(suid, ())): continue
                if any(math.dist(c, ac) <= rad for _, ac, rad in inc_refs.get(suid, ())): continue
                train_neg_candidates.append({"series": suid, "x": c[0], "y": c[1], "z": c[2], "scan": train_scans[suid]["scan_id"]})
    log(f"Clean training negatives available: {len(train_neg_candidates):,}")

    # Sample 16,000 random negatives for Scout & main bank
    rng_cand = random.Random(1000 + FOLD_K)
    rand_indices = rng_cand.sample(range(len(train_neg_candidates)), min(RAND_NEG_COUNT, len(train_neg_candidates)))
    rand_set = set(rand_indices)
    picked_rand = [train_neg_candidates[i] for i in rand_indices]

    def extract_crops(items_list):
        by_scan = defaultdict(list)
        for idx, c in enumerate(items_list):
            by_scan[c["scan"]].append((idx, c))
        p_arr = np.empty((len(items_list), 51, 51, 51), dtype=np.float32)
        v_arr = np.empty((len(items_list), 51, 51, 51), dtype=np.float32)
        for sid, items in by_scan.items():
            img, val, inv = load_scan_universal(sid)
            for idx, c in items:
                zyx = (inv @ np.array([c["x"], c["y"], c["z"], 1.0]))[:3]
                p51, _ = crop(img, zyx, 51, 0)
                v51, _ = crop(val, zyx, 51, False)
                p_arr[idx] = p51; v_arr[idx] = v51
        return p_arr, v_arr

    log(f"Extracting {len(picked_rand):,} random negative crops...")
    rand_neg_p51, rand_neg_v51 = extract_crops(picked_rand)

    # 1. Train Scout Model on training positives + 16k random negatives
    log(f"Training Scout CLEAR3DMSFv2 ({SCOUT_EPOCHS} epochs x {SCOUT_STEPS} steps)...")
    scout = CLEAR3DMSFv2(in_channels=3, dim=160, dropout=0.25).to(device)
    scout_opt = torch.optim.AdamW(scout.parameters(), lr=8e-4, weight_decay=1e-2)
    scout_scaler = torch.amp.GradScaler("cuda")
    rng_scout = np.random.default_rng(3000 + FOLD_K)

    for ep in range(SCOUT_EPOCHS):
        scout.train()
        lr = 1e-5 + 0.5 * (8e-4 - 1e-5) * (1.0 + math.cos(math.pi * ep / max(SCOUT_EPOCHS - 1, 1)))
        for g in scout_opt.param_groups: g["lr"] = lr
        ep_loss = 0.0
        for _ in range(SCOUT_STEPS):
            ip = rng_scout.choice(len(train_pos_p51), size=12, replace=True)
            ir = rng_scout.choice(len(rand_neg_p51), size=20, replace=True)
            p_sub, v_sub = augment_p51_batch(
                np.concatenate([train_pos_p51[ip], rand_neg_p51[ir]], axis=0),
                np.concatenate([train_pos_v51[ip], rand_neg_v51[ir]], axis=0),
                rng_scout
            )
            with torch.amp.autocast("cuda", dtype=torch.float16):
                fi_g, fv_g, ci_g, cv_g = gpu_extract_from_p51(torch.from_numpy(p_sub).to(device), torch.from_numpy(v_sub).to(device))
                vf, vc = make_inputs(fi_g, fv_g, ci_g, cv_g, f_fine, f_ctx)
                targets = torch.tensor([1]*12 + [0]*20, device=device, dtype=torch.long)
                sw = torch.tensor([16.0/12]*12 + [16.0/20]*20, device=device, dtype=torch.float32)
                scout_opt.zero_grad(set_to_none=True)
                logits = scout(vf, vc)
                loss = (F.cross_entropy(logits.float(), targets, reduction="none", label_smoothing=0.015) * sw).mean()
            scout_scaler.scale(loss).backward()
            scout_scaler.unscale_(scout_opt)
            torch.nn.utils.clip_grad_norm_(scout.parameters(), 2.0)
            scout_scaler.step(scout_opt); scout_scaler.update()
            ep_loss += float(loss.detach())
        if (ep + 1) % 4 == 0 or ep == SCOUT_EPOCHS - 1:
            log(f"  Scout Epoch {ep+1}/{SCOUT_EPOCHS} | loss={ep_loss/SCOUT_STEPS:.4f}")

    # 2. Mine hardest negatives across all remaining clean Fold 1-9 negatives
    log("Mining top hard negatives across remaining Fold 1-9 negative pool...")
    scout.eval()
    by_scan_mine = defaultdict(list)
    for idx, c in enumerate(train_neg_candidates):
        if idx not in rand_set:
            by_scan_mine[c["scan"]].append((idx, c))

    mined_scores = []
    t_m0 = time.time()
    with torch.no_grad():
        for s_idx, (sid, items) in enumerate(by_scan_mine.items()):
            img, val, inv = load_scan_universal(sid)
            p_buf = np.empty((len(items), 51, 51, 51), dtype=np.float32)
            v_buf = np.empty((len(items), 51, 51, 51), dtype=np.float32)
            for j, (_, c) in enumerate(items):
                zyx = (inv @ np.array([c["x"], c["y"], c["z"], 1.0]))[:3]
                p51, _ = crop(img, zyx, 51, 0)
                v51, _ = crop(val, zyx, 51, False)
                p_buf[j] = p51; v_buf[j] = v51
            for st in range(0, len(items), 128):
                en = min(st + 128, len(items))
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    fi_g, fv_g, ci_g, cv_g = gpu_extract_from_p51(torch.from_numpy(p_buf[st:en]).to(device), torch.from_numpy(v_buf[st:en]).to(device))
                    vf, vc = make_inputs(fi_g, fv_g, ci_g, cv_g, f_fine, f_ctx)
                    lg = scout(vf, vc).float()
                    probs = torch.softmax(lg, dim=1)[:, 1].cpu().numpy()
                for k_off, pr in enumerate(probs):
                    mined_scores.append((float(pr), items[st + k_off][0]))
            if (s_idx + 1) % 200 == 0:
                log(f"  Mined {s_idx+1}/{len(by_scan_mine)} scans ({len(mined_scores):,} negs) in {time.time()-t_m0:.1f}s")

    mined_scores.sort(key=lambda x: x[0], reverse=True)
    top_hard = mined_scores[:HARD_NEG_COUNT]
    hard_neg_rows = [train_neg_candidates[idx] for _, idx in top_hard]
    hard_neg_scores = np.array([sc for sc, _ in top_hard], dtype=np.float64)
    log(f"Mining complete: scored={len(mined_scores):,}, top1={hard_neg_scores[0]:.4f}, top_end={hard_neg_scores[-1]:.4f}")
    del scout, scout_opt, scout_scaler, mined_scores
    torch.cuda.empty_cache()

    log(f"Extracting {len(hard_neg_rows):,} hard negative crops...")
    hard_neg_p51, hard_neg_v51 = extract_crops(hard_neg_rows)

    # 3. Train Improved CLEAR3DMSFv2 (seeded BEFORE model instantiation!)
    torch.manual_seed(4200 + FOLD_K)
    torch.cuda.manual_seed_all(4200 + FOLD_K)
    model = CLEAR3DMSFv2(in_channels=3, dim=160, dropout=0.25).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda")
    rng_f = np.random.default_rng(5000 + FOLD_K)
    tr_hw = (0.5 + 0.5 * hard_neg_scores)
    tr_hw = tr_hw / tr_hw.sum()

    log(f"Training Improved CLEAR3DMSFv2 ({MAIN_EPOCHS} epochs x {MAIN_STEPS} steps, 10 pos + 12 rand + 10 hard)...")
    for ep in range(MAIN_EPOCHS):
        if ep == 0:
            lr = 4e-4
        else:
            prog = (ep - 1) / max(MAIN_EPOCHS - 2, 1)
            lr = 1e-5 + 0.5 * (8e-4 - 1e-5) * (1.0 + math.cos(math.pi * prog))
        for g in opt.param_groups: g["lr"] = lr
        cut_ep = int(0.70 * MAIN_EPOCHS)
        hard_share = 0.25 if ep < cut_ep else (0.08 + 0.17 * 0.5 * (1.0 + math.cos(math.pi * (ep - cut_ep) / max(MAIN_EPOCHS - 1 - cut_ep, 1))))
        w_pos = 16.0 / 10.0
        w_rand = (16.0 * (1.0 - hard_share)) / 12.0
        w_hard = (16.0 * hard_share) / 10.0
        sw = torch.tensor([w_pos]*10 + [w_rand]*12 + [w_hard]*10, device=device, dtype=torch.float32)
        targets = torch.tensor([1]*10 + [0]*22, device=device, dtype=torch.long)

        model.train()
        ep_loss = 0.0
        for _ in range(MAIN_STEPS):
            ip = rng_f.choice(len(train_pos_p51), size=10, replace=True)
            ir = rng_f.choice(len(rand_neg_p51), size=12, replace=False)
            ih = rng_f.choice(len(hard_neg_p51), size=10, replace=False, p=tr_hw)
            p_sub, v_sub = augment_p51_batch(
                np.concatenate([train_pos_p51[ip], rand_neg_p51[ir], hard_neg_p51[ih]], axis=0),
                np.concatenate([train_pos_v51[ip], rand_neg_v51[ir], hard_neg_v51[ih]], axis=0),
                rng_f
            )
            with torch.amp.autocast("cuda", dtype=torch.float16):
                fi_g, fv_g, ci_g, cv_g = gpu_extract_from_p51(torch.from_numpy(p_sub).to(device), torch.from_numpy(v_sub).to(device))
                vf, vc = make_inputs(fi_g, fv_g, ci_g, cv_g, f_fine, f_ctx)
                opt.zero_grad(set_to_none=True)
                logits = model(vf, vc)
                loss = (F.cross_entropy(logits.float(), targets, reduction="none", label_smoothing=0.015) * sw).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update()
            ep_loss += float(loss.detach())
        if (ep + 1) % 4 == 0 or ep == MAIN_EPOCHS - 1:
            log(f"  Main Epoch {ep+1:2d}/{MAIN_EPOCHS} | loss={ep_loss/MAIN_STEPS:.4f} | hard_share={hard_share:.3f}")

    ckpt_p = OUT_DIR / "fold_0_improved.pt"
    torch.save({"model_state_dict": model.state_dict(), "fold": FOLD_K, "epochs": MAIN_EPOCHS}, ckpt_p)

    # 4. Score Held-Out Fold 0 under 4-view TTA
    log("Scoring Held-Out Fold 0 under 4-view TTA...")
    model.eval()
    ho_cands_by_scan = defaultdict(list)
    with open(REFD / "candidates_V2.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        for idx, r in enumerate(reader):
            suid = r[0]
            if suid in held_out_scans:
                c = (float(r[1]), float(r[2]), float(r[3]))
                if any(math.dist(c, ac) <= rad for ac, rad in exc_refs.get(suid, ())): continue
                lbl = 1 if any(math.dist(c, ac) <= rad for _, ac, rad in inc_refs.get(suid, ())) else 0
                ho_cands_by_scan[suid].append((idx, c[0], c[1], c[2], lbl))

    pred_csv = OUT_DIR / "fold_0_improved_predictions.csv"
    with open(pred_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seriesuid", "coordX", "coordY", "coordZ", "probability", "candidate_id", "label"])
        with torch.no_grad():
            for suid, items in sorted(ho_cands_by_scan.items()):
                sid = held_out_scans[suid]["scan_id"]
                img, val, inv = load_scan_universal(sid)
                p_buf = np.empty((len(items), 51, 51, 51), dtype=np.float32)
                v_buf = np.empty((len(items), 51, 51, 51), dtype=np.float32)
                for j, (_, x, y, z, _) in enumerate(items):
                    zyx = (inv @ np.array([x, y, z, 1.0]))[:3]
                    p51, _ = crop(img, zyx, 51, 0)
                    v51, _ = crop(val, zyx, 51, False)
                    p_buf[j] = p51; v_buf[j] = v51
                for st in range(0, len(items), 64):
                    en = min(st + 64, len(items))
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        fi_g, fv_g, ci_g, cv_g = gpu_extract_from_p51(torch.from_numpy(p_buf[st:en]).to(device), torch.from_numpy(v_buf[st:en]).to(device))
                        vf, vc = make_inputs(fi_g, fv_g, ci_g, cv_g, f_fine, f_ctx)
                        lo_sum = torch.zeros(en - st, device=device, dtype=torch.float32)
                        for flips in ((), (4,), (2, 3), (2, 3, 4)):
                            vf_v = torch.flip(vf, flips) if flips else vf
                            vc_v = torch.flip(vc, flips) if flips else vc
                            lg = model(vf_v, vc_v).float()
                            lo_sum += (lg[:, 1] - lg[:, 0]) * 0.25
                        probs = torch.sigmoid(lo_sum).cpu().numpy()
                    for k_off, pr in enumerate(probs):
                        c_idx, x, y, z, lbl = items[st + k_off]
                        w.writerow([suid, x, y, z, float(pr), f"0:{c_idx}", lbl])

    log("Running Official LUNA16 Comparison (Baseline Fold 0 vs. Improved Fold 0)...")
    base_res = calc_metrics_and_froc(BASE_CV_DIR / "predictions/fold_0_predictions.csv", "BASELINE_FOLD_0")
    imp_res = calc_metrics_and_froc(pred_csv, "IMPROVED_FOLD_0")
    summary = {"baseline_fold_0": base_res, "improved_fold_0": imp_res, "elapsed_sec": time.time() - t0}
    (OUT_DIR / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"DONE in {time.time()-t0:.1f}s!")

if __name__ == "__main__":
    main()
