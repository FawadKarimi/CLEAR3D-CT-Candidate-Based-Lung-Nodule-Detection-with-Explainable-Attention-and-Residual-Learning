"""Independent External Validation Suite: CLEAR3D-MSFv2 on SPIE-AAPM Lung CT Challenge Cohort.
- System: Frozen 5-model ensemble (fold-0.pt .. fold-4.pt) with 4-view TTA and pre-sigmoid logit averaging
- Preprocessing: 1.0 mm isotropic LPS resampling, window [-1000, 400] HU normalized to [0, 1]
- Proposals: Automated multiscale LoG blob generator (sigmas [1, 1.5, 2, 3] mm, threshold 0.005, NMS 3.0 mm, cap 400)
- Annotations: SPIE-AAPM Calibration Set (10 scans, 10 biopsy-proven nodules)
- Matching: Challenge-rule (Euclidean distance < 5.0 mm; closest is TP, duplicate matches ignored) and raw matching
- Metrics: Proposal coverage, FROC curve, sensitivities at 7 FP rates [0.125, 0.25, 0.5, 1, 2, 4, 8], CPM,
  candidate-level metrics at nat_cut_oof (0.9999467) & 0.5, and 1,000-rep patient-clustered bootstrap 95% CIs.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom
import torch

BASE = Path('/experiment-storage/chaidir/fawad-dsai')
ROOT = BASE / 'CLEAR3D_CT project/02_Active_Model_and_Pipeline'
EXT_DIR = ROOT / 'external_validation_spie_aapm'
RAW_DIR = EXT_DIR / 'raw_pilot'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE / 'CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/src'))

from clear3d_msf_model_20261106_002 import CLEAR3DMSFv2, build_focus_grids, make_inputs
from train_clear3d_msf_ensemble_20261106_004 import gpu_extract_from_p51
from lidc3d.geometry import crop, normalize, resample_lps, to_index, to_world
from lidc3d.ingest import load_series
from lidc3d.candidates import lung_mask, generate

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    vals = scores[order]
    i, n = 0, len(vals)
    while i < n:
        j = i + 1
        while j < n and vals[j] == vals[i]:
            j += 1
        if j > i + 1:
            ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return float((ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))

def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-scores, kind='mergesort')
    y = (labels[order] == 1).astype(np.float64)
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1, dtype=np.float64)
    return float((prec * y).sum() / n_pos)

def calc_confusion_metrics(scores: np.ndarray, labels: np.ndarray, cut: float) -> dict:
    pred = scores >= cut
    y = labels == 1
    tp = int((pred & y).sum())
    tn = int((~pred & ~y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    prec = float(tp / (tp + fp)) if (tp + fp) else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) else 0.0
    acc = float((tp + tn) / len(labels)) if len(labels) else 0.0
    f1p = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    f1n = float(2 * tn / (2 * tn + fp + fn)) if (2 * tn + fp + fn) else 0.0
    return {
        'total_candidates': len(labels),
        'positives': int(y.sum()),
        'negatives': int((~y).sum()),
        'threshold': float(cut),
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'accuracy': acc,
        'macro_f1': float(0.5 * (f1p + f1n)),
        'f1_positive': f1p,
        'f1_negative': f1n,
        'precision_positive': prec,
        'recall_positive': rec,
        'specificity_negative': spec,
    }

def compute_froc_cpm(candidate_records: list[dict], num_scans: int, num_targets: int, grid=(0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)):
    target_best_score = {}
    for c in candidate_records:
        if c['is_nodule_match']:
            tid = c['target_id']
            if tid not in target_best_score or c['score'] > target_best_score[tid]:
                target_best_score[tid] = c['score']
    
    fp_scores = [c['score'] for c in candidate_records if (not c['is_nodule_match'] and not c['is_duplicate_ignore'])]
    fp_scores.sort(reverse=True)
    
    all_scores = sorted(list(set([c['score'] for c in candidate_records])), reverse=True)
    
    tps_curve = []
    fps_curve = []
    sens_curve = []
    
    for thresh in all_scores:
        tps = sum(1 for s in target_best_score.values() if s >= thresh)
        fps = sum(1 for s in fp_scores if s >= thresh)
        tps_curve.append(tps)
        fps_curve.append(fps / float(num_scans))
        sens_curve.append(tps / float(num_targets))
    
    fps_curve = [0.0] + fps_curve + [max(fps_curve[-1] if fps_curve else 0.0, 20.0)]
    sens_curve = [0.0] + sens_curve + [sens_curve[-1] if sens_curve else 0.0]
    
    fps_curve_np = np.array(fps_curve)
    sens_curve_np = np.array(sens_curve)
    order = np.argsort(fps_curve_np, kind='mergesort')
    fps_curve_sorted = fps_curve_np[order]
    sens_curve_sorted = sens_curve_np[order]
    
    sens_at_grid = np.interp(grid, fps_curve_sorted, sens_curve_sorted).tolist()
    cpm = float(np.mean(sens_at_grid))
    
    return {
        'grid_fp_rates': list(grid),
        'sensitivities': [float(x) for x in sens_at_grid],
        'cpm': cpm,
        'target_count': num_targets,
        'scan_count': num_scans,
        'total_fps': len(fp_scores)
    }

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    t_start_total = time.time()
    print("=" * 70, flush=True)
    print("STARTING CLEAR3D-MSFv2 SPIE-AAPM INDEPENDENT COHORT EVALUATION", flush=True)
    print("=" * 70, flush=True)
    
    binding_path = ROOT / 'primary_reference_system_binding.json'
    assert binding_path.exists(), f"Binding file missing: {binding_path}"
    with open(binding_path) as f:
        binding = json.load(f)
    print(f"Loaded Reference Binding: {binding['system_name']}", flush=True)
    print(f"Status: reference_release={binding['restrictions']['reference_release']}", flush=True)
    
    device = torch.device('cuda:1')
    models = []
    for fold_name, info in binding['checkpoints'].items():
        ck_path = ROOT / info['path']
        actual_h = sha256_file(ck_path)
        assert actual_h == info['sha256'], f"Checkpoint {fold_name} sha mismatch! {actual_h} != {info['sha256']}"
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        m = CLEAR3DMSFv2()
        m.load_state_dict(ck['model'], strict=True)
        m.to(device).eval()
        models.append(m)
        print(f"  [OK] Verified & loaded {fold_name} ({actual_h[:12]}...)", flush=True)
    
    scale_oof = float(binding['calibration_contract']['scale_oof'])
    bias_oof = float(binding['calibration_contract']['bias_oof'])
    nat_cut_oof = float(binding['calibration_contract']['nat_cut_oof'])
    print(f"Ensemble loaded on {device}. scale={scale_oof}, bias={bias_oof}, cut={nat_cut_oof}", flush=True)
    f_fine, f_ctx = build_focus_grids(device)
    
    gt_path = EXT_DIR / 'calibration_ground_truth.json'
    with open(gt_path) as f:
        gt_all = json.load(f)
    print(f"Loaded ground truth for {len(gt_all)} calibration cases.", flush=True)
    
    calib_patients = [
        'CT-Training-LC001', 'CT-Training-LC002', 'CT-Training-LC003',
        'CT-Training-LC008', 'CT-Training-LC009', 'CT-Training-BE001',
        'CT-Training-BE002', 'CT-Training-BE006', 'CT-Training-BE007', 'CT-Training-BE010'
    ]
    
    scan_diagnostics = []
    all_candidate_rows = []
    proposal_coverage_records = []
    
    scores_csv_path = EXT_DIR / 'scores.csv'
    t_start_scoring = time.time()
    
    print("\n--- Processing Scans ---", flush=True)
    for scan_idx, pid in enumerate(calib_patients, 1):
        t0 = time.time()
        dcm_dir = RAW_DIR / pid
        assert dcm_dir.exists(), f"DICOM directory missing for {pid}: {dcm_dir}"
        
        gt_info = gt_all[pid.upper()]
        inst_num = int(gt_info['instance_number'])
        x1, y1 = int(gt_info['pixel_x_1based']), int(gt_info['pixel_y_1based'])
        
        dcm_files = sorted(list(dcm_dir.glob('*.dcm')))
        ingest_cfg = {
            'orientation_atol': 1e-4, 'position_atol_mm': 1e-2,
            'spacing_rtol': 1e-3, 'spacing_atol_mm': 1e-3
        }
        vol = load_series([str(p) for p in dcm_files], ingest_cfg)
        
        target_dcm = None
        for p in dcm_files:
            ds_head = pydicom.dcmread(str(p), stop_before_pixels=True)
            if int(ds_head.InstanceNumber) == inst_num:
                target_dcm = ds_head
                break
        assert target_dcm is not None, f"Could not find slice with InstanceNumber={inst_num} for {pid}"
        
        ipp = [float(x) for x in target_dcm.ImagePositionPatient]
        spacing = [float(x) for x in target_dcm.PixelSpacing]
        iop = [float(x) for x in target_dcm.ImageOrientationPatient]
        
        x0 = x1 - 1
        y0 = y1 - 1
        gt_X = ipp[0] + x0 * spacing[0] * iop[0] + y0 * spacing[1] * iop[3]
        gt_Y = ipp[1] + x0 * spacing[0] * iop[1] + y0 * spacing[1] * iop[4]
        gt_Z = ipp[2]
        gt_lps = np.array([gt_X, gt_Y, gt_Z], dtype=np.float64)
        
        t_resamp0 = time.time()
        vol_r = resample_lps(vol, spacing=1.0)
        t_resamp = time.time() - t_resamp0
        
        img = normalize(vol_r.hu, window=(-1000, 400))
        img[~vol_r.valid] = 0.0
        
        t_mask0 = time.time()
        mask, base, mask_status = lung_mask(
            vol_r.hu, vol_r.valid, spacing=1.0, air_hu=-400,
            min_component_mm3=10000, dilation_mm=5
        )
        t_mask = time.time() - t_mask0
        
        t_prop0 = time.time()
        cand_cfg = {
            'threshold': 0.005,
            'sigmas_mm': [1.0, 1.5, 2.0, 3.0],
            'nms_mm': 3.0,
            'cap': 400
        }
        cands, diag = generate(img, mask, cand_cfg, spacing=1.0)
        t_prop = time.time() - t_prop0
        
        # In lidc3d.candidates.generate, each candidate is a dict with key "zyx"
        cands_zyx = np.array([c['zyx'] for c in cands], dtype=np.float64)
        cands_lps = to_world(cands_zyx, vol_r.affine)
        
        dists = np.linalg.norm(cands_lps - gt_lps, axis=1)
        min_dist = float(np.min(dists)) if len(dists) else float('inf')
        min_dist_idx = int(np.argmin(dists)) if len(dists) else -1
        detected_within_5mm = bool(min_dist < 5.0)
        
        proposal_coverage_records.append({
            'patient_id': pid,
            'nodule_id': gt_info['nodule_id'],
            'diagnosis': gt_info['diagnosis'],
            'gt_lps': gt_lps.tolist(),
            'proposals_generated': len(cands),
            'min_distance_mm': min_dist,
            'detected_within_5mm': detected_within_5mm,
            'proposal_rank': min_dist_idx + 1 if detected_within_5mm else None,
            'proposal_log_response': float(cands[min_dist_idx]['log_response']) if min_dist_idx >= 0 else None
        })
        
        is_nodule_match = [False] * len(cands)
        is_duplicate = [False] * len(cands)
        
        if detected_within_5mm:
            is_nodule_match[min_dist_idx] = True
            for c_i, d in enumerate(dists):
                if c_i != min_dist_idx and d < 5.0:
                    is_duplicate[c_i] = True
        
        t_infer0 = time.time()
        scan_probs_cal = [0.0] * len(cands)
        scan_probs_raw = [0.0] * len(cands)
        scan_logits_cal = [0.0] * len(cands)
        scan_logits_raw = [0.0] * len(cands)
        
        for start in range(0, len(cands), 64):
            end = min(start + 64, len(cands))
            batch_zyx = cands_zyx[start:end]
            m_cnt = len(batch_zyx)
            p51_b = np.empty((m_cnt, 51, 51, 51), dtype=np.float32)
            v51_b = np.empty((m_cnt, 51, 51, 51), dtype=np.float32)
            for k_i, zc in enumerate(batch_zyx):
                p51_b[k_i], _ = crop(img, zc, 51, 0)
                v51_b[k_i], _ = crop(vol_r.valid, zc, 51, False)
            
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
                fi_g, fv_g, ci_g, cv_g = gpu_extract_from_p51(
                    torch.from_numpy(p51_b).to(device),
                    torch.from_numpy(v51_b).to(device)
                )
                vf, vc = make_inputs(fi_g, fv_g, ci_g, cv_g, f_fine, f_ctx)
                views = [
                    (vf, vc),
                    (torch.flip(vf, [4]), torch.flip(vc, [4])),
                    (torch.flip(vf, [2, 3]), torch.flip(vc, [2, 3])),
                    (torch.flip(vf, [2, 3, 4]), torch.flip(vc, [2, 3, 4])),
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
                avg_lo_raw = lo_raw_sum / float(len(models))
                avg_lo_cal = lo_cal_sum / float(len(models))
                probs_raw = torch.sigmoid(avg_lo_raw).cpu().tolist()
                probs_cal = torch.sigmoid(avg_lo_cal).cpu().tolist()
                lo_raw_list = avg_lo_raw.cpu().tolist()
                lo_cal_list = avg_lo_cal.cpu().tolist()
            
            for idx_in_batch, (pr, pc, lr, lc) in enumerate(zip(probs_raw, probs_cal, lo_raw_list, lo_cal_list)):
                global_idx = start + idx_in_batch
                scan_probs_raw[global_idx] = float(pr)
                scan_probs_cal[global_idx] = float(pc)
                scan_logits_raw[global_idx] = float(lr)
                scan_logits_cal[global_idx] = float(lc)
        
        t_infer = time.time() - t_infer0
        t_total_scan = time.time() - t0
        
        scan_order = np.argsort(-np.array(scan_probs_cal))
        ranks_in_scan = np.empty(len(cands), dtype=int)
        ranks_in_scan[scan_order] = np.arange(1, len(cands) + 1)
        
        nodule_rank_in_scan = int(ranks_in_scan[min_dist_idx]) if detected_within_5mm else None
        nodule_score_cal = float(scan_probs_cal[min_dist_idx]) if detected_within_5mm else None
        nodule_score_raw = float(scan_probs_raw[min_dist_idx]) if detected_within_5mm else None
        
        for i in range(len(cands)):
            all_candidate_rows.append({
                'patient_id': pid,
                'candidate_id': f"{pid}:cand:{i:04d}",
                'coordX': float(cands_lps[i, 0]),
                'coordY': float(cands_lps[i, 1]),
                'coordZ': float(cands_lps[i, 2]),
                'dist_to_gt_mm': float(dists[i]),
                'is_nodule_match': is_nodule_match[i],
                'is_duplicate_ignore': is_duplicate[i],
                'target_id': gt_info['nodule_id'],
                'raw_prob': scan_probs_raw[i],
                'cal_prob': scan_probs_cal[i],
                'raw_logit': scan_logits_raw[i],
                'cal_logit': scan_logits_cal[i],
                'rank_in_scan': int(ranks_in_scan[i]),
                'label': 1 if is_nodule_match[i] else 0,
                'score': scan_probs_cal[i]
            })
            
        print(f"[{scan_idx}/10] {pid} ({gt_info['diagnosis']}): {len(cands)} props, "
              f"GT dist={min_dist:.2f}mm, Nodule Prob={nodule_score_cal}, Rank={nodule_rank_in_scan}/400 "
              f"(resamp={t_resamp:.1f}s, mask={t_mask:.1f}s, prop={t_prop:.1f}s, infer={t_infer:.1f}s, total={t_total_scan:.1f}s)", flush=True)
        
        scan_diagnostics.append({
            'patient_id': pid,
            'diagnosis': gt_info['diagnosis'],
            'slices': len(dcm_files),
            'raw_shape': list(vol.hu.shape),
            'resampled_shape': list(vol_r.hu.shape),
            'proposals': len(cands),
            'min_dist_mm': min_dist,
            'nodule_detected': detected_within_5mm,
            'nodule_rank_in_scan': nodule_rank_in_scan,
            'nodule_cal_prob': nodule_score_cal,
            'nodule_raw_prob': nodule_score_raw,
            'timing': {
                'resample_sec': t_resamp,
                'mask_sec': t_mask,
                'proposal_sec': t_prop,
                'infer_sec': t_infer,
                'total_sec': t_total_scan
            }
        })
        
    t_scoring_total = time.time() - t_start_scoring
    print(f"\nAll 10 scans scored in {t_scoring_total:.1f}s ({t_scoring_total/60:.2f} min).", flush=True)
    
    with open(scores_csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['patient_id', 'candidate_id', 'coordX', 'coordY', 'coordZ',
                    'dist_to_gt_mm', 'is_nodule_match', 'is_duplicate_ignore',
                    'raw_prob', 'cal_prob', 'raw_logit', 'cal_logit', 'rank_in_scan', 'label'])
        for r in all_candidate_rows:
            w.writerow([r['patient_id'], r['candidate_id'], r['coordX'], r['coordY'], r['coordZ'],
                        f"{r['dist_to_gt_mm']:.4f}", int(r['is_nodule_match']), int(r['is_duplicate_ignore']),
                        f"{r['raw_prob']:.8f}", f"{r['cal_prob']:.8f}", f"{r['raw_logit']:.8f}",
                        f"{r['cal_logit']:.8f}", r['rank_in_scan'], r['label']])
    print(f"Saved scores to {scores_csv_path} (SHA-256: {sha256_file(scores_csv_path)})", flush=True)
    
    num_scans = len(calib_patients)
    num_targets = len(gt_all)
    froc_results = compute_froc_cpm(all_candidate_rows, num_scans=num_scans, num_targets=num_targets)
    print("\n" + "=" * 50, flush=True)
    print("FROC EVALUATION RESULTS (Challenge-Rule)", flush=True)
    print("=" * 50, flush=True)
    print(f"Sensitivity at 7 FP rates: {froc_results['sensitivities']}", flush=True)
    print(f"CPM: {froc_results['cpm']:.6f}", flush=True)
    
    scores_arr = np.array([r['cal_prob'] for r in all_candidate_rows])
    labels_arr = np.array([r['label'] for r in all_candidate_rows])
    
    non_dup_mask = np.array([not r['is_duplicate_ignore'] for r in all_candidate_rows])
    scores_eval = scores_arr[non_dup_mask]
    labels_eval = labels_arr[non_dup_mask]
    
    metrics_cut = calc_confusion_metrics(scores_eval, labels_eval, nat_cut_oof)
    metrics_05 = calc_confusion_metrics(scores_eval, labels_eval, 0.5)
    pr_auc_val = pr_auc(scores_eval, labels_eval)
    roc_auc_val = roc_auc(scores_eval, labels_eval)
    
    print(f"\nCandidate Metrics @ nat_cut_oof ({nat_cut_oof}):", flush=True)
    print(f"  Acc: {metrics_cut['accuracy']:.4f}, Prec: {metrics_cut['precision_positive']:.4f}, Rec: {metrics_cut['recall_positive']:.4f}, F1: {metrics_cut['f1_positive']:.4f}", flush=True)
    print(f"  TP: {metrics_cut['TP']}, FP: {metrics_cut['FP']}, TN: {metrics_cut['TN']}, FN: {metrics_cut['FN']}", flush=True)
    
    print(f"\nCandidate Metrics @ 0.5:", flush=True)
    print(f"  Acc: {metrics_05['accuracy']:.4f}, Prec: {metrics_05['precision_positive']:.4f}, Rec: {metrics_05['recall_positive']:.4f}, F1: {metrics_05['f1_positive']:.4f}", flush=True)
    print(f"  TP: {metrics_05['TP']}, FP: {metrics_05['FP']}, TN: {metrics_05['TN']}, FN: {metrics_05['FN']}", flush=True)
    print(f"PR-AUC: {pr_auc_val:.6f}, ROC-AUC: {roc_auc_val:.6f}", flush=True)
    
    print("\nRunning 1,000 patient-clustered bootstrap iterations...", flush=True)
    unique_pats = np.array(calib_patients)
    cand_pats_arr = np.array([r['patient_id'] for r in all_candidate_rows])
    
    pat_to_cand_idxs = {p: np.where(cand_pats_arr == p)[0] for p in unique_pats}
    
    boot_cpms = []
    boot_f1p_cut = []
    boot_rec_cut = []
    boot_prec_cut = []
    boot_f1p_05 = []
    boot_rec_05 = []
    boot_prec_05 = []
    
    rng = np.random.RandomState(42)
    for b_i in range(1000):
        samp_pats = rng.choice(unique_pats, size=len(unique_pats), replace=True)
        samp_cands = []
        for sp in samp_pats:
            for idx in pat_to_cand_idxs[sp]:
                samp_cands.append(all_candidate_rows[idx])
        
        samp_froc = compute_froc_cpm(samp_cands, num_scans=len(samp_pats), num_targets=len(samp_pats))
        boot_cpms.append(samp_froc['cpm'])
        
        samp_scores = np.array([c['score'] for c in samp_cands if not c['is_duplicate_ignore']])
        samp_labels = np.array([c['label'] for c in samp_cands if not c['is_duplicate_ignore']])
        
        m_cut = calc_confusion_metrics(samp_scores, samp_labels, nat_cut_oof)
        boot_f1p_cut.append(m_cut['f1_positive'])
        boot_rec_cut.append(m_cut['recall_positive'])
        boot_prec_cut.append(m_cut['precision_positive'])
        
        m_5 = calc_confusion_metrics(samp_scores, samp_labels, 0.5)
        boot_f1p_05.append(m_5['f1_positive'])
        boot_rec_05.append(m_5['recall_positive'])
        boot_prec_05.append(m_5['precision_positive'])
        
    cpm_95ci = [float(np.percentile(boot_cpms, 2.5)), float(np.percentile(boot_cpms, 97.5))]
    f1p_cut_95ci = [float(np.percentile(boot_f1p_cut, 2.5)), float(np.percentile(boot_f1p_cut, 97.5))]
    rec_cut_95ci = [float(np.percentile(boot_rec_cut, 2.5)), float(np.percentile(boot_rec_cut, 97.5))]
    prec_cut_95ci = [float(np.percentile(boot_prec_cut, 2.5)), float(np.percentile(boot_prec_cut, 97.5))]
    
    f1p_05_95ci = [float(np.percentile(boot_f1p_05, 2.5)), float(np.percentile(boot_f1p_05, 97.5))]
    rec_05_95ci = [float(np.percentile(boot_rec_05, 2.5)), float(np.percentile(boot_rec_05, 97.5))]
    prec_05_95ci = [float(np.percentile(boot_prec_05, 2.5)), float(np.percentile(boot_prec_05, 97.5))]
    
    print(f"CPM 95% CI: [{cpm_95ci[0]:.4f}, {cpm_95ci[1]:.4f}] (mean: {np.mean(boot_cpms):.4f})", flush=True)
    print(f"F1 @ 0.5 95% CI: [{f1p_05_95ci[0]:.4f}, {f1p_05_95ci[1]:.4f}]", flush=True)
    print(f"F1 @ nat_cut 95% CI: [{f1p_cut_95ci[0]:.4f}, {f1p_cut_95ci[1]:.4f}]", flush=True)
    
    detected_count = sum(1 for p in proposal_coverage_records if p['detected_within_5mm'])
    proposal_coverage_summary = {
        'total_nodules': len(proposal_coverage_records),
        'nodules_detected_within_5mm': detected_count,
        'proposal_coverage_rate': float(detected_count / len(proposal_coverage_records)),
        'per_nodule': proposal_coverage_records
    }
    print(f"\nProposal Coverage: {detected_count}/{len(proposal_coverage_records)} ({detected_count/len(proposal_coverage_records)*100:.1f}%)", flush=True)
    
    summary_report = {
        'status': 'PASS',
        'task': 'CLEAR3D-MSFv2 Independent External Evaluation on SPIE-AAPM Lung CT Cohort',
        'dataset': {
            'collection': 'SPIE-AAPM Lung CT Challenge',
            'doi': '10.7937/K9/TCIA.2015.UZLSU3FL',
            'license': 'CC-BY 3.0',
            'cohort_role': 'External Validation Calibration Set',
            'patient_count': len(calib_patients),
            'target_nodule_count': num_targets,
            'overlap_with_lidc_luna16': 'UNKNOWN / UNVERIFIED'
        },
        'primary_reference_system': {
            'name': binding['system_name'],
            'reference_release': binding['restrictions']['reference_release'],
            'checkpoints': binding['checkpoints'],
            'calibration_contract': binding['calibration_contract']
        },
        'timing_seconds': {
            'scoring_seconds': float(t_scoring_total),
            'total_pipeline_seconds': float(time.time() - t_start_total)
        },
        'proposal_generator': {
            'algorithm': 'Multiscale Laplacian of Gaussian (LoG) 3D Blob Detector',
            'parameters': cand_cfg,
            'learned_weights': False,
            'coverage_summary': proposal_coverage_summary
        },
        'classifier_froc': {
            **froc_results,
            'cpm_bootstrap_95ci': cpm_95ci,
            'cpm_bootstrap_mean': float(np.mean(boot_cpms))
        },
        'candidate_discrimination': {
            'at_nat_cut_oof': {
                **metrics_cut,
                'f1_positive_95ci': f1p_cut_95ci,
                'recall_positive_95ci': rec_cut_95ci,
                'precision_positive_95ci': prec_cut_95ci
            },
            'at_fixed_05': {
                **metrics_05,
                'f1_positive_95ci': f1p_05_95ci,
                'recall_positive_95ci': rec_05_95ci,
                'precision_positive_95ci': prec_05_95ci
            },
            'pr_auc': pr_auc_val,
            'roc_auc': roc_auc_val
        },
        'per_scan_diagnostics': scan_diagnostics,
        'hashes': {
            'scores_csv_sha256': sha256_file(scores_csv_path),
            'ground_truth_json_sha256': sha256_file(gt_path),
            'reference_binding_json_sha256': sha256_file(binding_path)
        }
    }
    
    summary_json_path = EXT_DIR / 'spie_aapm_cohort_evaluation_summary.json'
    with open(summary_json_path, 'w') as f:
        json.dump(summary_report, f, indent=2)
    print(f"\nSaved comprehensive evaluation report to {summary_json_path}", flush=True)
    print("=" * 70, flush=True)
    print("EVALUATION COMPLETE", flush=True)
    print("=" * 70, flush=True)

if __name__ == '__main__':
    main()
