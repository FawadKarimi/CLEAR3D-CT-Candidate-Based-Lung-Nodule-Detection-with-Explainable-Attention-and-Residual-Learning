import sys, json, time, csv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch

BASE = Path("/experiment-storage/chaidir/fawad-dsai")
ROOT = BASE / "CLEAR3D_CT project/02_Active_Model_and_Pipeline"
REFD = BASE / "data/reference/luna16-zenodo-3723295/resolved-reference-v1"
PREP_TEST = BASE / "CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-test-fold0-v1/prepared"
sys.path.insert(0, str(ROOT / "runs/exp_fold0_arm_e_mixed_focal_20261116"))
import run_arm_e_experiment as ae

EXP_DIR = ROOT / "runs/exp_hold_test_arm_e_8view_20261116"
ae.EXP_DIR = EXP_DIR
pred_dir = EXP_DIR / "predictions"
eval_dir = EXP_DIR / "official_cad_eval"
pred_dir.mkdir(parents=True, exist_ok=True)
eval_dir.mkdir(parents=True, exist_ok=True)
log_path = EXP_DIR / "run.log"

def load_scan_test(scan_id: str):
    d = PREP_TEST / scan_id
    assert d.exists() and not d.is_symlink(), f"Missing test scan: {d}"
    geo = json.loads((d / "geometry.json").read_text())
    inv = np.linalg.inv(np.asarray(geo["affine"], dtype=np.float64))
    img = np.load(d / "image.npy", mmap_mode=None, allow_pickle=False)
    val = np.load(d / "valid.npy", mmap_mode=None, allow_pickle=False)
    return img, val, inv

def iter_prefetched_test(scan_ids, max_prefetch=4):
    sids = list(scan_ids)
    with ThreadPoolExecutor(max_workers=max_prefetch) as ex:
        futs = [(sid, ex.submit(load_scan_test, sid)) for sid in sids[:max_prefetch]]
        nxt = max_prefetch
        while futs:
            sid, fut = futs.pop(0)
            data = fut.result()
            if nxt < len(sids):
                futs.append((sids[nxt], ex.submit(load_scan_test, sids[nxt])))
                nxt += 1
            yield sid, data

withheld_rows = [json.loads(l) for l in open(BASE / "CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-cohort-fold0-v1/cohort_test_WITHHOLD.jsonl") if l.strip()]
hold_suids = {r["series_uid"] for r in withheld_rows}
hold_pids = {r["patient_id"] for r in withheld_rows}

dev_scans = [json.loads(l) for l in open(BASE / "CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-development-fold0-v1/prepared/scans.jsonl") if l.strip()]
dev_pids = {r["patient_id"] for r in dev_scans}
assert len(hold_pids & dev_pids) == 0, "Patient overlap between HOLD and dev!"

cands_by_scan = defaultdict(list)
with open(REFD / "candidates_V2.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)
    for idx, r in enumerate(reader):
        suid = r[0]
        if suid in hold_suids:
            cands_by_scan[suid].append((idx, float(r[1]), float(r[2]), float(r[3]), int(r[4])))

arme_sd = ae.load_checkpoint_cpu(ROOT / "runs/exp_fold0_arm_e_mixed_focal_20261116/checkpoints/arm_d_round2_focal_gamma2.pt")["model_state_dict"]
cal_e = json.loads((ROOT / "runs/exp_fold0_arm_e_8view_tta_20261116/results_summary.json").read_text())["inner_fit"]

ae.log(f"Scoring HOLD Subset 0 ({len(withheld_rows)} scans, {len(hold_pids)} patients) with frozen Arm E 8-view TTA...", log_path)
t0 = time.time()
device = torch.device("cuda:0")
f_fine, f_ctx = ae.build_focus_grids(device)
views = [(fz, fy, fx) for fz in (False, True) for fy in (False, True) for fx in (False, True)]
m = ae.CLEAR3DMSFv2(in_channels=3, dim=160, dropout=0.25).to(device)
m.load_state_dict(arme_sd)
m.eval()

rows = []
sid_to_suid = {s["scan_id"]: s["series_uid"] for s in withheld_rows}
with torch.no_grad():
    for i_s, (sid, (img, val, inv)) in enumerate(iter_prefetched_test(sorted(sid_to_suid.keys()), max_prefetch=4), 1):
        suid = sid_to_suid[sid]
        cands = cands_by_scan[suid]
        zyxs = np.array([(inv @ np.array([x, y, z, 1.0]))[:3] for _, x, y, z, _ in cands], dtype=np.float64)
        img_t, val_t, c_pad, off = ae.gpu_pad_and_slice_setup(img, val, zyxs, device)
        probs = np.empty(len(cands), dtype=np.float32)
        for st in range(0, len(cands), 128):
            en = min(st + 128, len(cands))
            p_t, v_t = ae.gpu_slice_chunk(img_t, val_t, c_pad[st:en], off)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                fi_g, fv_g, ci_g, cv_g = ae.gpu_extract_from_p51(p_t, v_t.float())
                vf, vc = ae.make_inputs(fi_g, fv_g, ci_g, cv_g, f_fine, f_ctx)
                acc = torch.zeros(en - st, device=device, dtype=torch.float32)
                for fz, fy, fx in views:
                    xf, xc = vf, vc
                    if fz: xf = torch.flip(xf, [2]); xc = torch.flip(xc, [2])
                    if fy: xf = torch.flip(xf, [3]); xc = torch.flip(xc, [3])
                    if fx: xf = torch.flip(xf, [4]); xc = torch.flip(xc, [4])
                    lg = m(xf, xc).float()
                    acc = acc + (lg[:, 1] - lg[:, 0])
                probs[st:en] = torch.sigmoid(acc * 0.125).cpu().numpy()
        order = np.argsort(-probs)
        ranks = np.empty(len(cands), dtype=int)
        ranks[order] = np.arange(1, len(cands) + 1)
        for i, (c_idx, x, y, z, cls) in enumerate(cands):
            rows.append({
                "seriesuid": suid, "coordX": x, "coordY": y, "coordZ": z,
                "probability": f"{float(probs[i]):.8f}",
                "candidate_id": f"hold:{c_idx}", "rank": int(ranks[i]), "label": int(cls),
            })
        if i_s % 30 == 0 or i_s == len(sid_to_suid):
            ae.log(f"  Scored {i_s}/{len(sid_to_suid)} HOLD scans in {time.time()-t0:.1f}s", log_path)

raw_csv = pred_dir / "arm_e_hold_8view_raw.csv"
ae.write_pred_csv(rows, raw_csv)
p_raw = np.array([float(r["probability"]) for r in rows], dtype=np.float64)
y_out = np.array([int(r["label"]) for r in rows], dtype=np.int64)
out = ae.apply_nested_calibration_to_outer(p_raw, y_out, cal_e)
froc = ae.run_official_luna16_eval(raw_csv, eval_dir / "arm_e_hold_8view")

summary = {
    "cohort": {"name": "HOLD_Subset0", "scans": len(withheld_rows), "patients": len(hold_pids), "candidates": len(rows), "positives": int(y_out.sum()), "negatives": int((y_out == 0).sum()), "patient_overlap_with_dev": 0},
    "frozen_inner_fit": cal_e,
    "arm_e_hold_8view": {"roc_auc": out["roc_auc"], "pr_auc": out["pr_auc"], "metrics": out["metrics"], "froc": froc},
    "elapsed_sec": float(time.time() - t0),
}
(EXP_DIR / "results_summary.json").write_text(json.dumps(summary, indent=2))
ae.log("DONE HOLD Subset 0 eval!", log_path)
