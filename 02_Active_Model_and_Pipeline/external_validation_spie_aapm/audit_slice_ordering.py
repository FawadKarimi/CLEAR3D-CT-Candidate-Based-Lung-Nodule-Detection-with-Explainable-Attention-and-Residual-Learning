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
instance_order_issues = []
spacing_z_inconsistencies = []

for pid in scans:
    dcm_dir = RAW_TEST / pid
    dcm_files = sorted(list(dcm_dir.glob("*.dcm")))
    slices = []
    for p in dcm_files:
        ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        slices.append({
            "path": p,
            "instance": int(ds.InstanceNumber),
            "z": float(ds.ImagePositionPatient[2]),
            "thickness": float(getattr(ds, "SliceThickness", 0.0))
        })
        
    # Sort by z
    slices_by_z = sorted(slices, key=lambda s: s["z"])
    # Check if instance numbers are strictly monotonic with respect to z
    inst_forward = [s["instance"] for s in slices_by_z]
    is_mono_inc = all(inst_forward[i] < inst_forward[i+1] for i in range(len(inst_forward)-1))
    is_mono_dec = all(inst_forward[i] > inst_forward[i+1] for i in range(len(inst_forward)-1))
    
    if not (is_mono_inc or is_mono_dec):
        instance_order_issues.append((pid, "Non-monotonic instance vs Z"))
        
    # Check slice spacing consistency in Z
    z_diffs = np.diff([s["z"] for s in slices_by_z])
    if np.max(np.abs(z_diffs - np.median(z_diffs))) > 0.05:
        spacing_z_inconsistencies.append((pid, np.min(z_diffs), np.max(z_diffs), np.median(z_diffs)))

print("Slice Ordering Audit:")
print(f"  Scans with non-monotonic InstanceNumber vs Z: {len(instance_order_issues)}")
print(f"  Scans with non-uniform Z-spacing (>0.05mm deviation): {len(spacing_z_inconsistencies)}")
if spacing_z_inconsistencies:
    for s in spacing_z_inconsistencies[:5]: print(f"    {s}")
