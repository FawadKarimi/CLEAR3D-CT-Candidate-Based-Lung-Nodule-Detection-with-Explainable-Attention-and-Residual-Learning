"""Authorized Technical Pilot: CLEAR3D-MSFv2 on SPIE-AAPM Test Cohort (LUNGx-CT001 & LUNGx-CT002)"""
from __future__ import annotations
import csv, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import pydicom
import torch

BASE = Path("/experiment-storage/chaidir/fawad-dsai")
ROOT = BASE / "CLEAR3D_CT project/02_Active_Model_and_Pipeline"
EXT_DIR = ROOT / "external_validation_spie_aapm"
RAW_TEST_DIR = EXT_DIR / "raw_test"
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

def main():
    t_start = time.time()
    print("=" * 70, flush=True)
    print("CLEAR3D-MSFv2 SPIE-AAPM TEST COHORT TECHNICAL PILOT", flush=True)
    print("=" * 70, flush=True)

    binding_path = ROOT / "primary_reference_system_binding.json"
    assert binding_path.exists(), f"Binding missing: {binding_path}"
    with open(binding_path) as f:
        binding = json.load(f)
    print(f"Reference System: {binding['system_name']}", flush=True)
    print(f"Status: reference_release={binding['restrictions']['reference_release']}", flush=True)

    device = torch.device("cuda:1")
    models = []
    for fold_name, info in binding["checkpoints"].items():
        ck_path = ROOT / info["path"]
        assert sha256_file(ck_path) == info["sha256"], f"Hash mismatch on {fold_name}"
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        m = CLEAR3DMSFv2()
        m.load_state_dict(ck["model"], strict=True)
        m.to(device).eval()
        models.append(m)
        print(f"  [OK] Loaded & verified {fold_name} ({info['sha256'][:12]}...)", flush=True)

    scale_oof = float(binding["calibration_contract"]["scale_oof"])
    bias_oof = float(binding["calibration_contract"]["bias_oof"])
    nat_cut_oof = float(binding["calibration_contract"]["nat_cut_oof"])
    f_fine, f_ctx = build_focus_grids(device)

    with open(EXT_DIR / "test_ground_truth.json") as f:
        gt_test = json.load(f)

    test_pilot_patients = ["LUNGx-CT001", "LUNGx-CT002"]
    print(f"Running technical pilot on: {test_pilot_patients}", flush=True)

    pilot_results = []
    for pid in test_pilot_patients:
        t0_scan = time.time()
        dcm_dir = RAW_TEST_DIR / pid
        dcm_files = sorted(list(dcm_dir.glob("*.dcm")))
        print(f"\n--- Processing {pid} ({len(dcm_files)} DICOM slices) ---", flush=True)
        t0 = time.time()
        ingest_cfg = {"orientation_atol": 1e-4, "position_atol_mm": 1e-2, "spacing_rtol": 1e-3, "spacing_atol_mm": 1e-3}
        vol = load_series([str(p) for p in dcm_files], ingest_cfg)
        t_ingest = time.time() - t0
        t0 = time.time()
        vol_r = resample_lps(vol, spacing=1.0)
        t_resamp = time.time() - t0
        print(f"  Ingest: {t_ingest:.2f}s, Resample to 1.0mm: {t_resamp:.2f}s (Shape: {vol_r.hu.shape})", flush=True)
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
            assert target_dcm is not None, f"Instance {inst_num} not found in {pid}"
            ipp = [float(x) for x in target_dcm.ImagePositionPatient]
            spacing = [float(x) for x in target_dcm.PixelSpacing]
            iop = [float(x) for x in target_dcm.ImageOrientationPatient]
            x0 = x1 - 1
            y0 = y1 - 1
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
            print(f"  Target Nodule #{tg['nodule_index']}: inst={inst_num}, LPS=[{gt_X:.2f}, {gt_Y:.2f}, {gt_Z:.2f}], voxel=[{z_i}, {y_i}, {x_i}], HU={target_hu}", flush=True)
        t0 = time.time()
        img = normalize(vol_r.hu, window=(-1000, 400))
        img[~vol_r.valid] = 0.0
        mask, base, mask_diag = lung_mask(vol_r.hu, vol_r.valid, spacing=1.0, air_hu=-400, min_component_mm3=10000, dilation_mm=5)
        t_mask = time.time() - t0
        for tr in target_records:
            z_i, y_i, x_i = tr["zyx_voxel"]
            in_mask = bool(mask[z_i, y_i, x_i])
            tr["in_lung_mask"] = in_mask
            print(f"    Target #{tr['nodule_index']} in lung mask: {in_mask}", flush=True)
        t0 = time.time()
        cand_cfg_uncapped = {"threshold": 0.005, "sigmas_mm": [1.0, 1.5, 2.0, 3.0], "nms_mm": 3.0, "cap": 20000}
        all_cands, gen_diag = generate(img, mask, cand_cfg_uncapped, spacing=1.0)
        t_gen = time.time() - t0
        print(f"  Candidate Gen: {len(all_cands)} proposals in {t_gen:.2f}s (Total peaks in mask: {gen_diag['positive_peaks_in_mask']}, NMS suppressed: {gen_diag['nms_suppressed']})", flush=True)
        cands_zyx = np.array([c["zyx"] for c in all_cands], dtype=np.float64)
        cands_lps = to_world(cands_zyx, vol_r.affine)
        for tr in target_records:
            gt_lps_np = np.array(tr["gt_lps"])
            dists = np.linalg.norm(cands_lps - gt_lps_np, axis=1)
            min_d_idx = int(np.argmin(dists)) if len(dists) else -1
            min_d = float(dists[min_d_idx]) if min_d_idx >= 0 else float("inf")
            tr["min_candidate_distance_mm"] = min_d
            tr["uncapped_proposal_rank"] = min_d_idx + 1 if min_d_idx >= 0 else None
            tr["proposal_log_response"] = float(all_cands[min_d_idx]["log_response"]) if min_d_idx >= 0 else None
            tr["detected_within_5mm_uncapped"] = min_d < 5.0
            tr["detected_within_5mm_cap400"] = (min_d_idx < 400) and (min_d < 5.0)
            tr["detected_within_5mm_cap1500"] = (min_d_idx < 1500) and (min_d < 5.0)
            print(f"    Target #{tr['nodule_index']}: min dist = {min_d:.2f}mm, uncapped rank = {tr['uncapped_proposal_rank']}, in top-400: {tr['detected_within_5mm_cap400']}, in top-1500: {tr['detected_within_5mm_cap1500']}", flush=True)
        eval_cands = all_cands[:min(1500, len(all_cands))]
        eval_zyx = cands_zyx[:len(eval_cands)]
        t0 = time.time()
        probs_cal = [0.0] * len(eval_cands)
        probs_raw = [0.0] * len(eval_cands)
        for start in range(0, len(eval_cands), 64):
            end = min(start + 64, len(eval_cands))
            batch_zyx = eval_zyx[start:end]
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
                views = [(vf, vc), (torch.flip(vf, [4]), torch.flip(vc, [4])), (torch.flip(vf, [2, 3]), torch.flip(vc, [2, 3])), (torch.flip(vf, [2, 3, 4]), torch.flip(vc, [2, 3, 4]))]
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
        print(f"  Ensemble Inference on {len(eval_cands)} proposals: {t_infer:.2f}s ({t_infer/len(eval_cands)*1000:.1f} ms/prop)", flush=True)
        for tr in target_records:
            prop_rank = tr["uncapped_proposal_rank"]
            if prop_rank is not None and prop_rank <= len(eval_cands) and tr["min_candidate_distance_mm"] < 5.0:
                c_idx = prop_rank - 1
                tr["model_prob_calibrated"] = probs_cal[c_idx]
                tr["model_prob_raw"] = probs_raw[c_idx]
                clf_rank = sum(1 for p in probs_cal if p > probs_cal[c_idx]) + 1
                tr["classifier_rank_in_scan"] = clf_rank
                print(f"    Target #{tr['nodule_index']}: Calibrated Prob = {probs_cal[c_idx]:.6f}, Classifier Rank = {clf_rank}/{len(eval_cands)}", flush=True)
            else:
                tr["model_prob_calibrated"] = None
                tr["model_prob_raw"] = None
                tr["classifier_rank_in_scan"] = None
        pilot_results.append({
            "patient_id": pid, "slice_count": len(dcm_files), "shape_1mm": list(vol_r.hu.shape),
            "targets": target_records,
            "timing": {"ingest_s": t_ingest, "resample_s": t_resamp, "mask_s": t_mask, "gen_s": t_gen, "infer_s": t_infer, "total_scan_s": time.time() - t0_scan}
        })
    summary = {
        "status": "PASS", "title": "CLEAR3D-MSFv2 SPIE-AAPM Test Cohort Technical Pilot",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "patients_evaluated": len(pilot_results),
        "total_targets": sum(len(p["targets"]) for p in pilot_results),
        "results": pilot_results,
        "reference_release": binding["restrictions"]["reference_release"]
    }
    out_path = EXT_DIR / "test_cohort_technical_pilot_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nTechnical Pilot Complete! Summary saved to {out_path}", flush=True)

if __name__ == "__main__":
    main()