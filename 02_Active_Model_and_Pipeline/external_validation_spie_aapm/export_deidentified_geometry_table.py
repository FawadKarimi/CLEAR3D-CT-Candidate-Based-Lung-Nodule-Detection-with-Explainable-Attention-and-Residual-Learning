import csv, json, os, sys
from pathlib import Path
import numpy as np
import pydicom

BASE = Path("/experiment-storage/chaidir/fawad-dsai")
ROOT = BASE / "CLEAR3D_CT project/02_Active_Model_and_Pipeline"
EXT_DIR = ROOT / "external_validation_spie_aapm"
RAW_TEST = EXT_DIR / "raw_test"

with open(EXT_DIR / "test_ground_truth.json") as f:
    gt_test = json.load(f)

scans = sorted(list(gt_test.keys()), key=lambda x: int(x.split("-CT")[1]))
print(f"Exporting de-identified geometry table for {len(scans)} scans, 73 targets...", flush=True)

rows = []
for pid in scans:
    dcm_dir = RAW_TEST / pid
    targets = gt_test[pid]
    target_instances = {int(tg["instance_number"]): tg for tg in targets}
    
    # Fast search: only parse slices until all target instances are found
    found_slices = {}
    for p in dcm_dir.glob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
            inst = int(ds.InstanceNumber)
            if inst in target_instances:
                found_slices[inst] = ds
                if len(found_slices) == len(target_instances):
                    break
        except Exception as e:
            continue
            
    for tg in targets:
        inst = int(tg["instance_number"])
        ds = found_slices.get(inst)
        assert ds is not None, f"Instance {inst} not found for {pid}"
        
        ipp = [float(x) for x in ds.ImagePositionPatient]
        spacing = [float(x) for x in ds.PixelSpacing]
        iop = [float(x) for x in ds.ImageOrientationPatient]
        thickness = float(getattr(ds, "SliceThickness", 0.0))
        
        x1 = int(tg["pixel_x_1based"])
        y1 = int(tg["pixel_y_1based"])
        x0 = x1 - 1
        y0 = y1 - 1
        
        # DICOM Standard mapping: P_xyz = S + x * Delta_col * X_xyz + y * Delta_row * Y_xyz
        gt_X = ipp[0] + x0 * spacing[1] * iop[0] + y0 * spacing[0] * iop[3]
        gt_Y = ipp[1] + x0 * spacing[1] * iop[1] + y0 * spacing[0] * iop[4]
        gt_Z = ipp[2] + x0 * spacing[1] * iop[2] + y0 * spacing[0] * iop[5]
        
        rows.append({
            "patient_id": pid,
            "nodule_index": int(tg["nodule_index"]),
            "instance_number": inst,
            "pixel_x_1based": x1,
            "pixel_y_1based": y1,
            "pixel_x_0based": x0,
            "pixel_y_0based": y0,
            "slice_thickness_mm": round(thickness, 4),
            "pixel_spacing_row_mm": round(spacing[0], 6),
            "pixel_spacing_col_mm": round(spacing[1], 6),
            "ipp_x_mm": round(ipp[0], 4),
            "ipp_y_mm": round(ipp[1], 4),
            "ipp_z_mm": round(ipp[2], 4),
            "iop_1": round(iop[0], 4),
            "iop_2": round(iop[1], 4),
            "iop_3": round(iop[2], 4),
            "iop_4": round(iop[3], 4),
            "iop_5": round(iop[4], 4),
            "iop_6": round(iop[5], 4),
            "target_lps_x_mm": round(gt_X, 4),
            "target_lps_y_mm": round(gt_Y, 4),
            "target_lps_z_mm": round(gt_Z, 4),
            "raw_pos_str": tg.get("raw_pos_str", "")
        })

out_csv = EXT_DIR / "portable_target_geometry.csv"
fieldnames = list(rows[0].keys())
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

out_json = EXT_DIR / "portable_target_geometry.json"
with open(out_json, "w") as f:
    json.dump({"total_targets": len(rows), "total_patients": len(scans), "targets": rows}, f, indent=2)

print(f"Successfully exported {len(rows)} targets to:")
print(f"  - {out_csv}")
print(f"  - {out_json}")
