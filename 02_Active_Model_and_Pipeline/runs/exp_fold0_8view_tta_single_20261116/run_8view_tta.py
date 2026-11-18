import sys, json, time, csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

BASE = Path("/experiment-storage/chaidir/fawad-dsai")
ROOT = BASE / "CLEAR3D_CT project/02_Active_Model_and_Pipeline"
REFD = BASE / "data/reference/luna16-zenodo-3723295/resolved-reference-v1"
sys.path.insert(0, str(ROOT / "runs/exp_fold0_round2_focal_mining_20261115_v3"))
import run_fold0_round2_focal_experiment as r2

EXP_DIR = ROOT / "runs/exp_fold0_8view_tta_single_20261116"
r2.EXP_DIR = EXP_DIR
pred_dir = EXP_DIR / "predictions"
eval_dir = EXP_DIR / "official_cad_eval"
pred_dir.mkdir(parents=True, exist_ok=True)
eval_dir.mkdir(parents=True, exist_ok=True)
log_path = EXP_DIR / "run.log"

def score_scans_8view_tta(device_str, scans_list, eval_cands_by_scan, models_by_name, fold_label):
    t0 = time.time()
    device = torch.device(device_str)
    f_fine, f_ctx = r2.build_focus_grids(device)
    views = [(fz, fy, fx) for fz in (False, True) for fy in (False, True) for fx in (False, True)]
    loaded = {}
    for name, sd in models_by_name.items():
        m = r2.CLEAR3DMSFv2(in_channels=3, dim=160, dropout=0.25).to(device)
        m.load_state_dict(sd)
        m.eval()
        loaded[name] = m
    rows_by_name = {name: [] for name in loaded}
    sid_to_suid = {s["scan_id"]: s["series_uid"] for s in scans_list}
    with torch.no_grad():
        for sid, (img, val, inv) in r2.iter_prefetched_scans(sorted(sid_to_suid.keys()), max_prefetch=4):
            suid = sid_to_suid[sid]
            cands = eval_cands_by_scan[suid]
            zyxs = np.array([(inv @ np.array([x, y, z, 1.0]))[:3] for _, x, y, z, _ in cands], dtype=np.float64)
            img_t, val_t, c_pad, off = r2.gpu_pad_and_slice_setup(img, val, zyxs, device)
            probs_by_name = {name: np.empty(len(cands), dtype=np.float32) for name in loaded}
            for st in range(0, len(cands), 128):
                en = min(st + 128, len(cands))
                p_t, v_t = r2.gpu_slice_chunk(img_t, val_t, c_pad[st:en], off)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    fi_g, fv_g, ci_g, cv_g = r2.gpu_extract_from_p51(p_t, v_t.float())
                    vf, vc = r2.make_inputs(fi_g, fv_g, ci_g, cv_g, f_fine, f_ctx)
                    for name, m in loaded.items():
                        acc = torch.zeros(en - st, device=device, dtype=torch.float32)
                        for fz, fy, fx in views:
                            xf, xc = vf, vc
                            if fz: xf = torch.flip(xf, [2]); xc = torch.flip(xc, [2])
                            if fy: xf = torch.flip(xf, [3]); xc = torch.flip(xc, [3])
                            if fx: xf = torch.flip(xf, [4]); xc = torch.flip(xc, [4])
                            lg = m(xf, xc).float()
                            acc = acc + (lg[:, 1] - lg[:, 0])
                        probs_by_name[name][st:en] = torch.sigmoid(acc * 0.125).cpu().numpy()
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
    del loaded
    torch.cuda.empty_cache()
    return rows_by_name, float(time.time() - t0)

gov = r2.verify_governance_hold()
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
r2.validate_dev_cohort(dev_scans, quarantined_patients, quarantined_scans, withheld_suids, withheld_pids)
dev_scan_map = {s["series_uid"]: s for s in dev_scans}
f0_scans = [s for s in dev_scans if s["fold"] == 0]
f1_scans = [s for s in dev_scans if s["fold"] == 1]

eval_cands_by_scan = defaultdict(list)
with open(REFD / "candidates_V2.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)
    for idx, r in enumerate(reader):
        suid = r[0]
        if suid not in dev_scan_map:
            continue
        eval_cands_by_scan[suid].append((idx, float(r[1]), float(r[2]), float(r[3]), int(r[4])))

armc_sd = r2.load_checkpoint_cpu(ROOT / "runs/exp_fold0_focal_and_matched_inner_20261115/checkpoints/arm_c_focal_gamma2.pt")["model_state_dict"]
inc_sd  = r2.load_checkpoint_cpu(ROOT / "runs/exp_fold0_focal_and_matched_inner_20261115/checkpoints/inner_focal_gamma2.pt")["model_state_dict"]
armd_sd = r2.load_checkpoint_cpu(ROOT / "runs/exp_fold0_round2_focal_mining_20261115_v3/checkpoints/arm_d_round2_focal_gamma2.pt")["model_state_dict"]
ind_sd  = r2.load_checkpoint_cpu(ROOT / "runs/exp_fold0_round2_focal_mining_20261115_v3/checkpoints/inner_d_round2_focal_gamma2.pt")["model_state_dict"]

r2.log("Scoring Inner Fold 1 with 8-view TTA...", log_path)
f1_rows, f1_sec = score_scans_8view_tta("cuda:0", f1_scans, eval_cands_by_scan, {"inner_c": inc_sd, "inner_d": ind_sd}, "f1")
r2.log(f"Inner Fold 1 scored in {f1_sec:.1f}s", log_path)

r2.log("Scoring Outer Fold 0 with 8-view TTA...", log_path)
f0_rows, f0_sec = score_scans_8view_tta("cuda:0", f0_scans, eval_cands_by_scan, {"arm_c": armc_sd, "arm_d": armd_sd}, "f0")
r2.log(f"Outer Fold 0 scored in {f0_sec:.1f}s", log_path)

cal_c = r2.fit_nested_platt_and_thresholds(np.array([float(r["probability"]) for r in f1_rows["inner_c"]], dtype=np.float64), np.array([int(r["label"]) for r in f1_rows["inner_c"]], dtype=np.int64))
cal_d = r2.fit_nested_platt_and_thresholds(np.array([float(r["probability"]) for r in f1_rows["inner_d"]], dtype=np.float64), np.array([int(r["label"]) for r in f1_rows["inner_d"]], dtype=np.int64))

def eval_one(name, rows, cal):
    raw_csv = pred_dir / f"{name}_fold0_8view_raw.csv"
    r2.write_pred_csv(rows, raw_csv)
    p_raw = np.array([float(r["probability"]) for r in rows], dtype=np.float64)
    y_out = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    out = r2.apply_nested_calibration_to_outer(p_raw, y_out, cal)
    froc = r2.run_official_luna16_eval(raw_csv, eval_dir / name)
    return {"roc_auc": out["roc_auc"], "pr_auc": out["pr_auc"], "metrics": out["metrics"], "froc": froc}

summary = {
    "governance": gov,
    "arm_c_8view": eval_one("arm_c_8view", f0_rows["arm_c"], cal_c),
    "arm_d_8view": eval_one("arm_d_8view", f0_rows["arm_d"], cal_d),
    "timing_sec": {"inner_f1": f1_sec, "outer_f0": f0_sec},
}
(EXP_DIR / "results_summary.json").write_text(json.dumps(summary, indent=2))
r2.log("DONE 8-view TTA eval!", log_path)
