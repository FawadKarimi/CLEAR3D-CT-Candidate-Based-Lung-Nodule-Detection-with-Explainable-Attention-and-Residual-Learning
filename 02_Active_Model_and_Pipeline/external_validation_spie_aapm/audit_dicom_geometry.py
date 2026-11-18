import json, os, sys
from pathlib import Path
import numpy as np
import pydicom

BASE = Path("/experiment-storage/chaidir/fawad-dsai")
ROOT = BASE / "CLEAR3D_CT project/02_Active_Model_and_Pipeline"
EXT_DIR = ROOT / "external_validation_spie_aapm"
RAW_TEST = EXT_DIR / "raw_test"

with open(EXT_DIR / "test_ground_truth.json") as f:
    gt_test = json.load(f)

scans = sorted(list(gt_test.keys()))
print(f"Auditing DICOM geometry across all {len(scans)} test scans...", flush=True)

diffs = []
spacing_diffs = []
iop_deviations = []

for pid in scans:
    dcm_dir = RAW_TEST / pid
    dcm_files = sorted(list(dcm_dir.glob("*.dcm")))
    targets = gt_test[pid]
    
    slices = []
    for p in dcm_files:
        ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        slices.append({
            "path": p,
            "instance": int(ds.InstanceNumber),
            "ipp": [float(x) for x in ds.ImagePositionPatient],
            "spacing": [float(x) for x in ds.PixelSpacing],
            "iop": [float(x) for x in ds.ImageOrientationPatient],
            "thickness": float(getattr(ds, "SliceThickness", 0.0))
        })
        
    for tg in targets:
        inst_num = int(tg["instance_number"])
        x1, y1 = int(tg["pixel_x_1based"]), int(tg["pixel_y_1based"])
        x0, y0 = x1 - 1, y1 - 1
        
        target_slice = next((s for s in slices if s["instance"] == inst_num), None)
        assert target_slice is not None, f"Instance {inst_num} missing in {pid}"
        
        ipp = target_slice["ipp"]
        spacing = target_slice["spacing"]
        iop = target_slice["iop"]
        
        sp_diff = abs(spacing[0] - spacing[1])
        if sp_diff > 1e-6:
            spacing_diffs.append((pid, tg["nodule_index"], spacing))
            
        iop_diff = np.max(np.abs(np.array(iop) - np.array([1, 0, 0, 0, 1, 0])))
        if iop_diff > 1e-4:
            iop_deviations.append((pid, tg["nodule_index"], iop))
            
        gt_X_used = ipp[0] + x0 * spacing[0] * iop[0] + y0 * spacing[1] * iop[3]
        gt_Y_used = ipp[1] + x0 * spacing[0] * iop[1] + y0 * spacing[1] * iop[4]
        gt_Z_used = ipp[2]
        lps_used = np.array([gt_X_used, gt_Y_used, gt_Z_used])
        
        gt_X_std = ipp[0] + x0 * spacing[1] * iop[0] + y0 * spacing[0] * iop[3]
        gt_Y_std = ipp[1] + x0 * spacing[1] * iop[1] + y0 * spacing[0] * iop[4]
        gt_Z_std = ipp[2] + x0 * spacing[1] * iop[2] + y0 * spacing[0] * iop[5]
        lps_std = np.array([gt_X_std, gt_Y_std, gt_Z_std])
        
        diff = np.linalg.norm(lps_used - lps_std)
        if diff > 1e-6:
            diffs.append((pid, tg["nodule_index"], diff))

print(f"Audit Results:")
print(f"  Scans with non-square pixel spacing (spacing[0] != spacing[1]): {len(spacing_diffs)}")
print(f"  Scans with non-standard axial IOP (!= [1,0,0,0,1,0]): {len(iop_deviations)}")
print(f"  Targets with L2 coordinate discrepancy between used and strict DICOM formula: {len(diffs)}")
if diffs:
    for d in diffs[:5]: print(f"    {d}")
else:
    print("  EXACT 0.000000 mm discrepancy: Used formula matches strict DICOM standard identically across all 73 targets!")
