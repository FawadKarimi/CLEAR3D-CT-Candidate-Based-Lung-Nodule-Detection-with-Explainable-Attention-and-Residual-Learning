
import json, math, os, sys, time
from pathlib import Path
import numpy as np
import pydicom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage as ndi

BASE = Path('/experiment-storage/chaidir/fawad-dsai')
ROOT = BASE / 'CLEAR3D_CT project/02_Active_Model_and_Pipeline'
EXT_DIR = ROOT / 'external_validation_spie_aapm'
RAW_DIR = EXT_DIR / 'raw_pilot'
OUT_DIR = EXT_DIR / 'pilot_audit_overlays'
OUT_DIR.mkdir(exist_ok=True, parents=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE / 'CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/src'))

from lidc3d.ingest import load_series
from lidc3d.geometry import resample_lps, to_index, to_world, normalize
from lidc3d.candidates import lung_mask, generate

print("--- Starting Detailed Target Coordinate & Miss Audit ---", flush=True)

with open(EXT_DIR / 'calibration_ground_truth.json') as f:
    gt_all = json.load(f)

calib_patients = [
    'CT-Training-LC001', 'CT-Training-LC002', 'CT-Training-LC003',
    'CT-Training-LC008', 'CT-Training-LC009', 'CT-Training-BE001',
    'CT-Training-BE002', 'CT-Training-BE006', 'CT-Training-BE007', 'CT-Training-BE010'
]

ingest_cfg = {
    'orientation_atol': 1e-4, 'position_atol_mm': 1e-2,
    'spacing_rtol': 1e-3, 'spacing_atol_mm': 1e-3
}

cand_cfg_400 = {
    'threshold': 0.005, 'sigmas_mm': [1.0, 1.5, 2.0, 3.0],
    'nms_mm': 3.0, 'cap': 400
}

cand_cfg_uncapped = {
    'threshold': 0.005, 'sigmas_mm': [1.0, 1.5, 2.0, 3.0],
    'nms_mm': 3.0, 'cap': 50000
}

audit_records = []

for idx, pid in enumerate(calib_patients, 1):
    t0 = time.time()
    dcm_dir = RAW_DIR / pid
    gt_info = gt_all[pid.upper()]
    inst_num = int(gt_info['instance_number'])
    x1, y1 = int(gt_info['pixel_x_1based']), int(gt_info['pixel_y_1based'])
    
    dcm_files = sorted(list(dcm_dir.glob('*.dcm')))
    
    target_dcm = None
    for p in dcm_files:
        ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        if int(ds.InstanceNumber) == inst_num:
            target_dcm = ds
            break
            
    assert target_dcm is not None, f"Slice with InstanceNumber={inst_num} missing in {pid}"
    
    ipp = [float(x) for x in target_dcm.ImagePositionPatient]
    spacing = [float(x) for x in target_dcm.PixelSpacing]
    iop = [float(x) for x in target_dcm.ImageOrientationPatient]
    slice_thickness = float(getattr(target_dcm, 'SliceThickness', 0.0))
    
    x0, y0 = x1 - 1, y1 - 1
    gt_X = ipp[0] + x0 * spacing[0] * iop[0] + y0 * spacing[1] * iop[3]
    gt_Y = ipp[1] + x0 * spacing[0] * iop[1] + y0 * spacing[1] * iop[4]
    gt_Z = ipp[2]
    gt_lps = np.array([gt_X, gt_Y, gt_Z], dtype=np.float64)
    
    vol = load_series([str(p) for p in dcm_files], ingest_cfg)
    vol_r = resample_lps(vol, spacing=1.0)
    
    target_zyx = to_index(gt_lps, vol_r.affine)
    tz, ty, tx = int(round(target_zyx[0])), int(round(target_zyx[1])), int(round(target_zyx[2]))
    
    mask, base, mask_status = lung_mask(
        vol_r.hu, vol_r.valid, spacing=1.0, air_hu=-400,
        min_component_mm3=10000, dilation_mm=5
    )
    
    img = normalize(vol_r.hu, window=(-1000, 400))
    img[~vol_r.valid] = 0.0
    
    in_bounds = (0 <= tz < vol_r.hu.shape[0] and 0 <= ty < vol_r.hu.shape[1] and 0 <= tx < vol_r.hu.shape[2])
    hu_target = float(vol_r.hu[tz, ty, tx]) if in_bounds else None
    in_mask_target = bool(mask[tz, ty, tx]) if in_bounds else False
    
    if in_bounds:
        zmin, zmax = max(0, tz-3), min(vol_r.hu.shape[0], tz+4)
        ymin, ymax = max(0, ty-3), min(vol_r.hu.shape[1], ty+4)
        xmin, xmax = max(0, tx-3), min(vol_r.hu.shape[2], tx+4)
        
        box_hu = vol_r.hu[zmin:zmax, ymin:ymax, xmin:xmax]
        box_mask = mask[zmin:zmax, ymin:ymax, xmin:xmax]
        
        hu_min = float(np.min(box_hu))
        hu_max = float(np.max(box_hu))
        hu_mean = float(np.mean(box_hu))
        mask_fraction = float(np.mean(box_mask))
        mask_any = bool(np.any(box_mask))
    else:
        hu_min, hu_max, hu_mean, mask_fraction, mask_any = None, None, None, 0.0, False
        
    if not in_mask_target and in_bounds:
        edt = ndi.distance_transform_edt(~mask, sampling=1.0)
        dist_to_mask = float(edt[tz, ty, tx])
    else:
        dist_to_mask = 0.0
        
    cands_400, _ = generate(img, mask, cand_cfg_400, spacing=1.0)
    zyx_400 = np.array([c['zyx'] for c in cands_400], dtype=np.float64) if len(cands_400) else np.empty((0, 3))
    lps_400 = to_world(zyx_400, vol_r.affine) if len(cands_400) else np.empty((0, 3))
    dists_400 = np.linalg.norm(lps_400 - gt_lps, axis=1) if len(cands_400) else np.array([])
    min_dist_400 = float(np.min(dists_400)) if len(dists_400) else float('inf')
    min_idx_400 = int(np.argmin(dists_400)) if len(dists_400) else -1
    detected_400 = bool(min_dist_400 < 5.0)
    
    cands_unc, _ = generate(img, mask, cand_cfg_uncapped, spacing=1.0)
    zyx_unc = np.array([c['zyx'] for c in cands_unc], dtype=np.float64) if len(cands_unc) else np.empty((0, 3))
    lps_unc = to_world(zyx_unc, vol_r.affine) if len(cands_unc) else np.empty((0, 3))
    dists_unc = np.linalg.norm(lps_unc - gt_lps, axis=1) if len(cands_unc) else np.array([])
    min_dist_unc = float(np.min(dists_unc)) if len(dists_unc) else float('inf')
    min_idx_unc = int(np.argmin(dists_unc)) if len(dists_unc) else -1
    detected_unc = bool(min_dist_unc < 5.0)
    
    if detected_400:
        cause = "DETECTED_WITHIN_5MM"
        evidence = f"Candidate rank {min_idx_400 + 1} at distance {min_dist_400:.2f} mm"
    elif not in_bounds or not (vol_r.valid[tz, ty, tx]):
        cause = "GEOMETRY_OR_BOUNDS_ERROR"
        evidence = f"Target voxel [{tz}, {ty}, {tx}] outside valid volume grid"
    elif not mask_any and dist_to_mask > 2.0:
        cause = "LUNG_MASK_EXCLUSION"
        evidence = f"Target is outside lung mask by {dist_to_mask:.1f} mm; 0% of 7x7x7 neighborhood is masked"
    elif detected_unc and not detected_400:
        cause = "RANKING_AND_CAP"
        evidence = f"Candidate within {min_dist_unc:.2f} mm found at rank {min_idx_unc + 1} (> 400 cap)"
    elif not detected_unc:
        cause = "CANDIDATE_GENERATION"
        evidence = f"No LoG peak >= 0.005 within 5 mm across scales [1.0, 1.5, 2.0, 3.0] mm (min uncapped dist: {min_dist_unc:.2f} mm)"
    else:
        cause = "UNEXPLAINED"
        evidence = "Requires further investigation"
        
    record = {
        'patient_id': pid,
        'diagnosis': gt_info['diagnosis'],
        'spreadsheet_ref': {
            'pixel_x_1based': x1,
            'pixel_y_1based': y1,
            'instance_number': inst_num
        },
        'dicom_metadata': {
            'pixel_spacing': spacing,
            'slice_thickness': slice_thickness,
            'image_position_patient': ipp,
            'image_orientation_patient': iop,
            'total_slices': len(dcm_files)
        },
        'lps_ground_truth': list(gt_lps),
        'resampled_voxel_zyx': [tz, ty, tx],
        'hu_diagnostics': {
            'target_hu': hu_target,
            'box_7x7x7_mean_hu': hu_mean,
            'box_7x7x7_max_hu': hu_max,
            'box_7x7x7_min_hu': hu_min
        },
        'mask_diagnostics': {
            'target_in_mask': in_mask_target,
            'box_7x7x7_mask_fraction': mask_fraction,
            'distance_to_mask_mm': dist_to_mask
        },
        'proposal_diagnostics': {
            'detected_in_cap400': detected_400,
            'min_dist_cap400_mm': min_dist_400,
            'rank_in_cap400': min_idx_400 + 1 if detected_400 else None,
            'detected_in_uncapped': detected_unc,
            'min_dist_uncapped_mm': min_dist_unc,
            'rank_in_uncapped': min_idx_unc + 1 if detected_unc else None,
            'total_uncapped_proposals': len(cands_unc)
        },
        'root_cause': cause,
        'root_cause_evidence': evidence
    }
    audit_records.append(record)
    
    print(f"[{idx}/10] {pid} ({gt_info['diagnosis']}): {cause} -> {evidence} ({time.time()-t0:.1f}s)", flush=True)
    
    # 2. Generate Overlay Plot
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    
    crop_radius = 60
    y0_c = max(0, ty - crop_radius)
    y1_c = min(vol_r.hu.shape[1], ty + crop_radius)
    x0_c = max(0, tx - crop_radius)
    x1_c = min(vol_r.hu.shape[2], tx + crop_radius)
    
    hu_slice = vol_r.hu[tz, y0_c:y1_c, x0_c:x1_c]
    mask_slice = mask[tz, y0_c:y1_c, x0_c:x1_c]
    
    ax.imshow(hu_slice, cmap='gray', vmin=-1000, vmax=400, origin='lower')
    
    if np.any(mask_slice):
        ax.contour(mask_slice, levels=[0.5], colors=['#00FF66'], linewidths=1.5, origin='lower')
        
    rel_tx = tx - x0_c
    rel_ty = ty - y0_c
    
    ax.plot(rel_tx, rel_ty, 'r+', markersize=14, markeredgewidth=2, label='Target GT Center')
    circle_5mm = patches.Circle((rel_tx, rel_ty), radius=5.0, edgecolor='red', facecolor='none',
                                linestyle='--', linewidth=1.5, label='5.0 mm Matching Radius')
    ax.add_patch(circle_5mm)
    
    nearby_props = []
    for c_i, (cz, cy, cx) in enumerate(zyx_400):
        if abs(cz - tz) <= 3.0:
            d_lps = np.linalg.norm(lps_400[c_i] - gt_lps)
            if d_lps < 25.0:
                rel_cx = cx - x0_c
                rel_cy = cy - y0_c
                if 0 <= rel_cx < (x1_c - x0_c) and 0 <= rel_cy < (y1_c - y0_c):
                    nearby_props.append((rel_cx, rel_cy, d_lps, c_i + 1, cands_400[c_i]['log_response']))
                    
    for rel_cx, rel_cy, d_lps, rank, score in nearby_props[:5]:
        ax.plot(rel_cx, rel_cy, 'o', color='#3399FF', markersize=8, markeredgecolor='white', markeredgewidth=1)
        ax.annotate(f"R{rank} ({d_lps:.1f}mm)", (rel_cx + 1.5, rel_cy + 1.5),
                    color='white', fontsize=8, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", fc="#003366", ec="none", alpha=0.8))
        
    title_str = f"{pid} ({gt_info['diagnosis'].upper()}) - {cause}\nGT: ({gt_lps[0]:.1f}, {gt_lps[1]:.1f}, {gt_lps[2]:.1f}) mm | Voxel: [{tz}, {ty}, {tx}]\nMin Cand Dist: {min_dist_400:.1f} mm | HU: {hu_target:.0f}"
    ax.set_title(title_str, fontsize=10, weight='bold')
    ax.set_xlabel("X (voxels / mm)")
    ax.set_ylabel("Y (voxels / mm)")
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    overlay_path = OUT_DIR / f"overlay_{pid}.png"
    plt.savefig(overlay_path)
    plt.close()

fig, axes = plt.subplots(2, 5, figsize=(25, 10), dpi=150)
axes = axes.flatten()

for idx, pid in enumerate(calib_patients):
    rec = audit_records[idx]
    img_p = OUT_DIR / f"overlay_{pid}.png"
    im = plt.imread(img_p)
    axes[idx].imshow(im)
    axes[idx].axis('off')

plt.suptitle("CLEAR3D-CT SPIE-AAPM Calibration Cohort: 10 Target Nodules & Candidate Audit Overlays", fontsize=16, weight='bold')
plt.tight_layout()
grid_path = OUT_DIR / "all_ten_cases_overlay_grid.png"
plt.savefig(grid_path)
plt.close()

summary_audit = {
    'total_cases': len(calib_patients),
    'detected_cases': sum(1 for r in audit_records if r['root_cause'] == 'DETECTED_WITHIN_5MM'),
    'missed_cases': sum(1 for r in audit_records if r['root_cause'] != 'DETECTED_WITHIN_5MM'),
    'root_cause_breakdown': {
        'DETECTED_WITHIN_5MM': sum(1 for r in audit_records if r['root_cause'] == 'DETECTED_WITHIN_5MM'),
        'GEOMETRY_OR_BOUNDS_ERROR': sum(1 for r in audit_records if r['root_cause'] == 'GEOMETRY_OR_BOUNDS_ERROR'),
        'LUNG_MASK_EXCLUSION': sum(1 for r in audit_records if r['root_cause'] == 'LUNG_MASK_EXCLUSION'),
        'CANDIDATE_GENERATION': sum(1 for r in audit_records if r['root_cause'] == 'CANDIDATE_GENERATION'),
        'RANKING_AND_CAP': sum(1 for r in audit_records if r['root_cause'] == 'RANKING_AND_CAP'),
        'UNEXPLAINED': sum(1 for r in audit_records if r['root_cause'] == 'UNEXPLAINED')
    },
    'per_patient_records': audit_records
}

with open(OUT_DIR / 'audit_misses_summary.json', 'w') as f:
    json.dump(summary_audit, f, indent=2)

print(f"\n--- Audit Complete! Saved summary to {OUT_DIR / 'audit_misses_summary.json'} and grid to {grid_path} ---", flush=True)
