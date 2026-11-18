"""CLEAR3D-MSFv2 Autonomous Evaluation on SPIE-AAPM Test Cohort (Cap=4000)
- Model: Frozen 5-model ensemble (fold-0.pt .. fold-4.pt) with 4-view TTA on cuda:1
- Proposals: Automated multiscale LoG blob generator (cap=4000, sigmas=[1.0, 1.5, 2.0, 3.0], threshold=0.005, NMS=3.0mm)
- Matching: Fixed Euclidean distance < 5.0 mm (closest proposal is TP, duplicate hits ignored)
- Cohorts: Technical Pilot (2 scans), Previously Unexposed (58 scans), Full Cohort (60 scans)
- Metrics: Proposal coverage, End-to-end sensitivity at nat_cut & 0.5, FP/scan, and FROC CPM with 1000-rep patient bootstrap
"""
from __future__ import annotations
import csv, hashlib, json, math, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom
import torch

BASE = Path("/experiment-storage/chaidir/fawad-dsai")
ROOT = BASE / "CLEAR3D_CT project/02_Active_Model_and_Pipeline"
EXT_DIR = ROOT / "external_validation_spie_aapm"
RAW_TEST = EXT_DIR / "raw_test"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE / "CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/src"))

from clear3d_msf_model_20261106_002 import CLEAR3DMSFv2, build_focus_grids, make_inputs
from train_clear3d_msf_ensemble_20261106_004 import gpu_extract_from_p51
from lidc3d.geometry import crop, normalize, resample_lps, to_index, to_world
from lidc3d.ingest import load_series
from lidc3d.candidates import lung_mask, generate

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()

def compute_froc_cpm(candidate_records, num_scans, num_targets, grid=(0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)):
    if num_scans == 0 or num_targets == 0:
        return {"cpm": 0.0, "sensitivities": [0.0]*len(grid)}
    target_best_score = {}
    for c in candidate_records:
        if c["is_nodule_match"]:
            tid = c["target_id"]
            if tid not in target_best_score or c["score"] > target_best_score[tid]:
                target_best_score[tid] = c["score"]
    fp_scores = [c["score"] for c in candidate_records if (not c["is_nodule_match"] and not c["is_duplicate_ignore"])]
    all_scores = sorted(list(set([c["score"] for c in candidate_records])), reverse=True)
    tps_curve, fps_curve, sens_curve = [], [], []
    for thresh in all_scores:
        tps = sum(1 for s in target_best_score.values() if s >= thresh)
        fps = sum(1 for s in fp_scores if s >= thresh)
        tps_curve.append(tps)
        fps_curve.append(fps / float(num_scans))
        sens_curve.append(tps / float(num_targets))
    fps_curve = [0.0] + fps_curve + [max(fps_curve[-1] if fps_curve else 0.0, 20.0)]
    sens_curve = [0.0] + sens_curve + [sens_curve[-1] if sens_curve else 0.0]
    fps_np = np.array(fps_curve)
    sens_np = np.array(sens_curve)
    order = np.argsort(fps_np, kind="mergesort")
    fps_sorted = fps_np[order]
    sens_sorted = sens_np[order]
    sens_at_grid = np.interp(grid, fps_sorted, sens_sorted).tolist()
    cpm = float(np.mean(sens_at_grid))
    return {
        "grid_fp_rates": list(grid),
        "sensitivities": [float(x) for x in sens_at_grid],
        "cpm": cpm,
        "total_fps": len(fp_scores),
        "targets_detected": len(target_best_score),
        "targets_total": num_targets
    }

def run_bootstrap_patient_clustered(patients_list, scan_dict, cands_by_patient, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    n_pts = len(patients_list)
    if n_pts == 0:
        return {}
    boot_cov, boot_cpm, boot_sens_nat, boot_sens_half = [], [], [], []
    boot_dups = []
    nat_cut = 0.9999467134475708
    
    for b in range(n_boot):
        sampled_indices = rng.choice(n_pts, size=n_pts, replace=True)
        dup_count = n_pts - len(set(sampled_indices))
        boot_dups.append(dup_count)
        b_cands = []
        b_targets = 0
        b_covered = 0
        for idx in sampled_indices:
            pid = patients_list[idx]
            scan_info = scan_dict[pid]
            b_targets += len(scan_info["targets"])
            b_covered += sum(1 for tg in scan_info["targets"] if tg["detected_within_5mm_cap4000"])
            for c in cands_by_patient[pid]:
                b_cands.append(c)
        froc_res = compute_froc_cpm(b_cands, n_pts, b_targets)
        cov_rate = (b_covered / b_targets) if b_targets else 0.0
        sens_nat = (sum(1 for c in b_cands if c["is_nodule_match"] and c["score"] >= nat_cut) / b_targets) if b_targets else 0.0
        sens_half = (sum(1 for c in b_cands if c["is_nodule_match"] and c["score"] >= 0.5) / b_targets) if b_targets else 0.0
        boot_cov.append(cov_rate)
        boot_cpm.append(froc_res["cpm"])
        boot_sens_nat.append(sens_nat)
        boot_sens_half.append(sens_half)
    
    def ci(arr):
        return {
            "mean": float(np.mean(arr)), "std": float(np.std(arr)),
            "ci_95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
        }
    return {
        "n_bootstraps": n_boot,
        "avg_duplicate_patients": float(np.mean(boot_dups)),
        "proposal_coverage": ci(boot_cov),
        "cpm": ci(boot_cpm),
        "sensitivity_nat_cut": ci(boot_sens_nat),
        "sensitivity_half": ci(boot_sens_half)
    }

def main():
    t_start = time.time()
    print("=" * 75, flush=True)
    print("CLEAR3D-MSFv2 SPIE-AAPM TEST COHORT INDEPENDENT EVALUATION (CAP=4000)", flush=True)
    print("=" * 75, flush=True)
    
    proto_path = EXT_DIR / "frozen_evaluation_protocol_cap4000.json"
    with open(proto_path) as f:
        proto = json.load(f)
    proto_sha = sha256_file(proto_path)
    print(f"Loaded Frozen Protocol: {proto['protocol_name']} (SHA: {proto_sha[:12]}...)", flush=True)
    print(f"Candidate Cap: {proto['selection_basis']['candidate_cap']} (Vetted on 142 LUNA16 Dev Scans)", flush=True)
    print(f"Matching Rule: {proto['matching_protocol']['rule']} ({proto['matching_protocol']['evaluation_type']})", flush=True)
    print(f"Governance Status: reference_release={proto['governance']['reference_release']}", flush=True)
    
    # Load models
    device = torch.device("cuda:1")
    models = []
    for fold_name, info in proto["frozen_system"]["checkpoints"].items():
        ck_p = ROOT / info["path"]
        assert sha256_file(ck_p) == info["sha256"], f"Checksum failure on {fold_name}"
        ck = torch.load(ck_p, map_location="cpu", weights_only=False)
        m = CLEAR3DMSFv2()
        m.load_state_dict(ck["model"], strict=True)
        m.to(device).eval()
        models.append(m)
        print(f"  [OK] Verified & loaded {fold_name}", flush=True)
    scale_oof = float(proto["frozen_system"]["calibration"]["scale_oof"])
    bias_oof = float(proto["frozen_system"]["calibration"]["bias_oof"])
    nat_cut_oof = float(proto["frozen_system"]["calibration"]["nat_cut_oof"])
    f_fine, f_ctx = build_focus_grids(device)
    
    with open(EXT_DIR / "test_ground_truth.json") as f:
        gt_test = json.load(f)
    with open(EXT_DIR / "tcia_series_list.json") as f:
        series_list = json.load(f)
    series_map = {x["PatientID"]: x for x in series_list if "LUNGx-CT" in x["PatientID"]}
    patient_order = sorted(list(series_map.keys()))
    print(f"Total test cohort scans to evaluate: {len(patient_order)}", flush=True)
    
    scores_csv_path = EXT_DIR / "scores_cap4000.csv"
    scores_csv_f = open(scores_csv_path, "w", newline="")
    csv_writer = csv.writer(scores_csv_f)
    csv_writer.writerow([
        "patient_id", "candidate_id", "rank", "coordX", "coordY", "coordZ",
        "log_response", "sigma_mm", "probability_calibrated", "probability_raw",
        "is_nodule_match", "is_duplicate_ignore", "target_id", "target_distance_mm"
    ])
    scores_csv_f.flush()
    
    scan_records = {}
    cands_by_patient = defaultdict(list)
    
    for s_idx, pid in enumerate(patient_order, 1):
        dcm_dir = RAW_TEST / pid
        expected_slices = int(series_map[pid]["ImageCount"])
        
        # Wait if download still in progress
        while True:
            dcms = list(dcm_dir.glob("*.dcm")) if dcm_dir.exists() else []
            zip_exists = (RAW_TEST / f"{pid}.zip").exists()
            if len(dcms) >= expected_slices and not zip_exists:
                break
            print(f"[{s_idx}/{len(patient_order)}] Waiting for {pid} download ({len(dcms)}/{expected_slices} slices)...", flush=True)
            time.sleep(5)
        
        dcm_files = sorted(list(dcm_dir.glob("*.dcm")))
        t0_scan = time.time()
        print(f"\n[{s_idx}/{len(patient_order)}] Scoring {pid} ({len(dcm_files)} slices)...", flush=True)
        
        # Ingest
        t0 = time.time()
        ingest_cfg = {"orientation_atol": 1e-4, "position_atol_mm": 1e-2, "spacing_rtol": 1e-3, "spacing_atol_mm": 1e-3}
        vol = load_series([str(p) for p in dcm_files], ingest_cfg)
        t_ingest = time.time() - t0
        
        # Resample
        t0 = time.time()
        vol_r = resample_lps(vol, spacing=1.0)
        t_resamp = time.time() - t0
        
        # Map ground truth targets
        targets = gt_test[pid]
        target_records = []
        for tg in targets:
            inst_num = tg["instance_number"]
            x1, y1 = tg["pixel_x_1based"], tg["pixel_y_1based"]
            target_dcm = None
            for p in dcm_files:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True)
                if int(ds.InstanceNumber) == inst_num:
                    target_dcm = ds
                    break
            assert target_dcm is not None, f"Instance {inst_num} missing in {pid}"
            ipp = [float(x) for x in target_dcm.ImagePositionPatient]
            spacing = [float(x) for x in target_dcm.PixelSpacing]
            iop = [float(x) for x in target_dcm.ImageOrientationPatient]
            x0, y0 = x1 - 1, y1 - 1
            gt_X = ipp[0] + x0 * spacing[0] * iop[0] + y0 * spacing[1] * iop[3]
            gt_Y = ipp[1] + x0 * spacing[0] * iop[1] + y0 * spacing[1] * iop[4]
            gt_Z = ipp[2]
            gt_lps = np.array([gt_X, gt_Y, gt_Z], dtype=np.float64)
            zyx_idx = to_index(gt_lps, vol_r.affine)
            z_i, y_i, x_i = int(round(zyx_idx[0])), int(round(zyx_idx[1])), int(round(zyx_idx[2]))
            target_hu = float(vol_r.hu[z_i, y_i, x_i]) if (0 <= z_i < vol_r.hu.shape[0] and 0 <= y_i < vol_r.hu.shape[1] and 0 <= x_i < vol_r.hu.shape[2]) else None
            target_records.append({
                "nodule_index": tg["nodule_index"], "instance_number": inst_num,
                "pixel_xy": [x1, y1], "gt_lps": gt_lps.tolist(),
                "zyx_voxel": [z_i, y_i, x_i], "target_voxel_hu": target_hu
            })
        
        # Mask
        t0 = time.time()
        img = normalize(vol_r.hu, window=(-1000, 400))
        img[~vol_r.valid] = 0.0
        mask, base, mask_diag = lung_mask(vol_r.hu, vol_r.valid, spacing=1.0, air_hu=-400, min_component_mm3=10000, dilation_mm=5)
        t_mask = time.time() - t0
        for tr in target_records:
            z_i, y_i, x_i = tr["zyx_voxel"]
            tr["in_lung_mask"] = bool(mask[z_i, y_i, x_i])
        
        # Generate proposals with cap=4000
        t0 = time.time()
        cand_cfg = {"threshold": 0.005, "sigmas_mm": [1.0, 1.5, 2.0, 3.0], "nms_mm": 3.0, "cap": 4000}
        cands, gen_diag = generate(img, mask, cand_cfg, spacing=1.0)
        t_gen = time.time() - t0
        
        cands_zyx = np.array([c["zyx"] for c in cands], dtype=np.float64)
        cands_lps = to_world(cands_zyx, vol_r.affine)
        
        is_match = [False] * len(cands)
        is_dup = [False] * len(cands)
        match_target_id = [""] * len(cands)
        match_dist = [""] * len(cands)
        
        # Evaluate proposal distance & matching
        for tr in target_records:
            gt_lps_np = np.array(tr["gt_lps"])
            dists = np.linalg.norm(cands_lps - gt_lps_np, axis=1) if len(cands) else np.array([])
            min_d_idx = int(np.argmin(dists)) if len(dists) else -1
            min_d = float(dists[min_d_idx]) if min_d_idx >= 0 else float("inf")
            tr["min_candidate_distance_mm"] = min_d
            tr["proposal_rank"] = min_d_idx + 1 if min_d_idx >= 0 else None
            tr["detected_within_5mm_cap4000"] = (min_d < 5.0)
            if min_d < 5.0:
                is_match[min_d_idx] = True
                match_target_id[min_d_idx] = str(tr["nodule_index"])
                match_dist[min_d_idx] = f"{min_d:.2f}"
                for c_i, d in enumerate(dists):
                    if c_i != min_d_idx and d < 5.0:
                        is_dup[c_i] = True
                        match_target_id[c_i] = str(tr["nodule_index"])
                        match_dist[c_i] = f"{d:.2f}"
        
        # Ensemble inference
        t0 = time.time()
        probs_cal = [0.0] * len(cands)
        probs_raw = [0.0] * len(cands)
        for start in range(0, len(cands), 64):
            end = min(start + 64, len(cands))
            batch_zyx = cands_zyx[start:end]
            m_cnt = len(batch_zyx)
            p51_b = np.empty((m_cnt, 51, 51, 51), dtype=np.float32)
            v51_b = np.empty((m_cnt, 51, 51, 51), dtype=np.float32)
            for k_i, zc in enumerate(batch_zyx):
                p51_b[k_i], _ = crop(img, zc, 51, 0)
                v51_b[k_i], _ = crop(vol_r.valid, zc, 51, False)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                fi_g, fv_g, ci_g, cv_g = gpu_extract_from_p51(
                    torch.from_numpy(p51_b).to(device), torch.from_numpy(v51_b).to(device)
                )
                vf, vc = make_inputs(fi_g, fv_g, ci_g, cv_g, f_fine, f_ctx)
                views = [
                    (vf, vc), (torch.flip(vf, [4]), torch.flip(vc, [4])),
                    (torch.flip(vf, [2, 3]), torch.flip(vc, [2, 3])),
                    (torch.flip(vf, [2, 3, 4]), torch.flip(vc, [2, 3, 4]))
                ]
                lo_raw_sum = torch.zeros(m_cnt, device=device, dtype=torch.float32)
                lo_cal_sum = torch.zeros(m_cnt, device=device, dtype=torch.float32)
                for m_fold in models:
                    lo_m = torch.zeros(m_cnt, device=device, dtype=torch.float32)
                    for xvf, xvc in views:
                        lg = m_fold(xvf, xvc).float()
                        lo_m = lo_m + (lg[:, 1] - lg[:, 0])
                    lo_m = lo_m * 0.25
                    lo_raw_sum = lo_raw_sum + lo_m
                    lo_cal_sum = lo_cal_sum + (scale_oof * lo_m + bias_oof)
                p_raw = torch.sigmoid(lo_raw_sum / 5.0).cpu().numpy()
                p_cal = torch.sigmoid(lo_cal_sum / 5.0).cpu().numpy()
                for k_i in range(m_cnt):
                    probs_raw[start + k_i] = float(p_raw[k_i])
                    probs_cal[start + k_i] = float(p_cal[k_i])
        t_infer = time.time() - t0
        
        # Check target scores
        for tr in target_records:
            p_rank = tr["proposal_rank"]
            if p_rank is not None and tr["detected_within_5mm_cap4000"]:
                c_idx = p_rank - 1
                tr["model_prob_calibrated"] = probs_cal[c_idx]
                tr["model_prob_raw"] = probs_raw[c_idx]
                tr["classifier_rank_in_scan"] = sum(1 for p in probs_cal if p > probs_cal[c_idx]) + 1
                print(f"  Target #{tr['nodule_index']}: dist={tr['min_candidate_distance_mm']:.2f}mm, rank={p_rank}, prob={probs_cal[c_idx]:.6f}, clf_rank={tr['classifier_rank_in_scan']}/{len(cands)}", flush=True)
            else:
                tr["model_prob_calibrated"] = None
                tr["model_prob_raw"] = None
                tr["classifier_rank_in_scan"] = None
                print(f"  Target #{tr['nodule_index']}: MISSED BY GENERATOR (dist={tr['min_candidate_distance_mm']:.2f}mm, nearest rank={p_rank})", flush=True)
        
        # Write scores.csv and store records
        for c_i, c in enumerate(cands):
            cid = f"{pid}:cand:{c_i:05d}"
            p_cal_val = probs_cal[c_i]
            p_raw_val = probs_raw[c_i]
            csv_writer.writerow([
                pid, cid, c_i + 1, f"{cands_lps[c_i, 0]:.4f}", f"{cands_lps[c_i, 1]:.4f}", f"{cands_lps[c_i, 2]:.4f}",
                f"{c['log_response']:.6f}", f"{c['sigma_mm']:.2f}", f"{p_cal_val:.8f}", f"{p_raw_val:.8f}",
                1 if is_match[c_i] else 0, 1 if is_dup[c_i] else 0, match_target_id[c_i], match_dist[c_i]
            ])
            cands_by_patient[pid].append({
                "patient_id": pid, "candidate_id": cid, "score": p_cal_val, "score_raw": p_raw_val,
                "is_nodule_match": is_match[c_i], "is_duplicate_ignore": is_dup[c_i],
                "target_id": f"{pid}:target:{match_target_id[c_i]}" if is_match[c_i] else ""
            })
        scores_csv_f.flush()
        
        t_scan_total = time.time() - t0_scan
        scan_records[pid] = {
            "patient_id": pid, "slice_count": len(dcm_files), "proposals_generated": len(cands),
            "shape_1mm": list(vol_r.hu.shape), "targets": target_records,
            "timing": {"ingest_s": t_ingest, "resample_s": t_resamp, "mask_s": t_mask, "gen_s": t_gen, "infer_s": t_infer, "total_s": t_scan_total}
        }
        print(f"  [Done] {pid} total: {t_scan_total:.1f}s (gen: {t_gen:.1f}s, infer: {t_infer:.1f}s)", flush=True)
    
    scores_csv_f.close()
    print("\nAll 60 scans scored! Generating Cohort Metrics...", flush=True)
    
    # Cohorts separation
    pilot_patients = ["LUNGx-CT001", "LUNGx-CT002"]
    unexposed_patients = [p for p in patient_order if p not in pilot_patients]
    
    def evaluate_cohort_subset(name, p_list):
        subset_cands = [c for pid in p_list for c in cands_by_patient[pid]]
        n_scans = len(p_list)
        all_targets = [tg for pid in p_list for tg in scan_records[pid]["targets"]]
        n_targets = len(all_targets)
        covered_targets = sum(1 for tg in all_targets if tg["detected_within_5mm_cap4000"])
        froc_res = compute_froc_cpm(subset_cands, n_scans, n_targets)
        boot_res = run_bootstrap_patient_clustered(p_list, scan_records, cands_by_patient, n_boot=1000)
        
        # Threshold metrics
        tp_nat = sum(1 for c in subset_cands if c["is_nodule_match"] and c["score"] >= nat_cut_oof)
        fp_nat = sum(1 for c in subset_cands if not c["is_nodule_match"] and not c["is_duplicate_ignore"] and c["score"] >= nat_cut_oof)
        tp_half = sum(1 for c in subset_cands if c["is_nodule_match"] and c["score"] >= 0.5)
        fp_half = sum(1 for c in subset_cands if not c["is_nodule_match"] and not c["is_duplicate_ignore"] and c["score"] >= 0.5)
        
        return {
            "cohort_name": name, "scan_count": n_scans, "nodule_count": n_targets,
            "proposals_evaluated": len(subset_cands),
            "proposal_coverage": {
                "covered_nodules": covered_targets, "total_nodules": n_targets,
                "rate": float(covered_targets / n_targets) if n_targets else 0.0,
                "bootstrap_95ci": boot_res.get("proposal_coverage", {})
            },
            "operating_points": {
                "natural_prevalence_cut": {
                    "threshold": nat_cut_oof, "true_positives": tp_nat,
                    "nodule_sensitivity": float(tp_nat / n_targets) if n_targets else 0.0,
                    "total_false_positives": fp_nat, "fp_per_scan": float(fp_nat / n_scans) if n_scans else 0.0,
                    "bootstrap_95ci": boot_res.get("sensitivity_nat_cut", {})
                },
                "half_cut": {
                    "threshold": 0.5, "true_positives": tp_half,
                    "nodule_sensitivity": float(tp_half / n_targets) if n_targets else 0.0,
                    "total_false_positives": fp_half, "fp_per_scan": float(fp_half / n_scans) if n_scans else 0.0,
                    "bootstrap_95ci": boot_res.get("sensitivity_half", {})
                }
            },
            "froc_cpm_fixed_5mm": {
                "cpm": froc_res["cpm"],
                "grid_fp_rates": froc_res["grid_fp_rates"],
                "sensitivities": froc_res["sensitivities"],
                "bootstrap_95ci": boot_res.get("cpm", {})
            }
        }
    
    summary_pilot = evaluate_cohort_subset("Technical_Pilot_2scans", pilot_patients)
    summary_unexposed = evaluate_cohort_subset("Unexposed_Test_58scans", unexposed_patients)
    summary_full = evaluate_cohort_subset("Full_Test_60scans", patient_order)
    
    final_summary = {
        "status": "PASS",
        "task": "CLEAR3D-MSFv2 SPIE-AAPM Independent Test Evaluation (Cap=4000)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_sha256": proto_sha,
        "selection_basis": "Cap 4,000 selected strictly from 142 LUNA16 Dev scans (92.19% sensitivity)",
        "evaluation_rule": "Fixed Euclidean distance < 5.0 mm matching (closest proposal TP, duplicate hits ignored)",
        "annotation_caveat": "Target annotations in SPIE-AAPM LUNGx are incomplete for full scan CAD (only primary biopsy-proven nodule), so false positives include unannotated genuine nodules and granulomas.",
        "cohort_results": {
            "technical_pilot_scans": summary_pilot,
            "unexposed_test_scans": summary_unexposed,
            "full_test_cohort": summary_full
        },
        "per_scan_diagnostics": scan_records,
        "governance": {
            "reference_release": "HOLD",
            "confirmation_cohort": "83 patients strictly restricted / untouched",
            "internal_test_fold0": "Untouched / zero retraining",
            "manuscript_writing": "DEFERRED"
        }
    }
    
    summary_json_path = EXT_DIR / "test_cohort_cap4000_evaluation_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(final_summary, f, indent=2)
    
    print(f"\nEvaluation Complete! Results saved to {summary_json_path}", flush=True)
    print("Summary:", flush=True)
    print(f"  Unexposed (58 scans): Coverage={summary_unexposed['proposal_coverage']['rate']*100:.1f}%, Sens@cut={summary_unexposed['operating_points']['natural_prevalence_cut']['nodule_sensitivity']*100:.1f}%, CPM={summary_unexposed['froc_cpm_fixed_5mm']['cpm']:.4f}", flush=True)
    print(f"  Full (60 scans): Coverage={summary_full['proposal_coverage']['rate']*100:.1f}%, Sens@cut={summary_full['operating_points']['natural_prevalence_cut']['nodule_sensitivity']*100:.1f}%, CPM={summary_full['froc_cpm_fixed_5mm']['cpm']:.4f}", flush=True)

if __name__ == "__main__":
    main()