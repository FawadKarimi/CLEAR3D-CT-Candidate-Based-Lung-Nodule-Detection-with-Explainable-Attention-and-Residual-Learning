import csv, json, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

BASE = Path("/experiment-storage/chaidir/fawad-dsai")
ROOT = BASE / "CLEAR3D_CT project/02_Active_Model_and_Pipeline"
EXT_DIR = ROOT / "external_validation_spie_aapm"

with open(EXT_DIR / "test_ground_truth.json") as f:
    gt_test = json.load(f)

with open(EXT_DIR / "frozen_evaluation_protocol_cap4000.json") as f:
    proto_data = json.load(f)
nat_cut_oof = float(proto_data["frozen_system"]["calibration"]["nat_cut_oof"])

scores_path = EXT_DIR / "scores_cap4000.csv"
print("Loading candidates from scores_cap4000.csv...", flush=True)

# Group raw candidates by patient
cands_by_patient = defaultdict(list)
with open(scores_path) as f:
    r = csv.DictReader(f)
    for row in r:
        cands_by_patient[row["patient_id"]].append({
            "patient_id": row["patient_id"],
            "candidate_id": row["candidate_id"],
            "rank": int(row["rank"]),
            "coord": np.array([float(row["coordX"]), float(row["coordY"]), float(row["coordZ"])]),
            "score": float(row["probability_calibrated"]),
            "score_raw": float(row["probability_raw"])
        })

patient_order = sorted(list(cands_by_patient.keys()), key=lambda x: int(x.split("-CT")[1]))

# Need target coordinates for all scans
# We can load them from DICOM target LPS or test_ground_truth.json + DICOM
import pydicom
RAW_TEST = EXT_DIR / "raw_test"

target_lps_dict = {}
for pid in patient_order:
    dcm_dir = RAW_TEST / pid
    dcm_files = sorted(list(dcm_dir.glob("*.dcm")))
    targets = gt_test[pid]
    target_lps_dict[pid] = []
    
    slices = {}
    for p in dcm_files:
        ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        slices[int(ds.InstanceNumber)] = ds
        
    for tg in targets:
        inst = int(tg["instance_number"])
        ds = slices[inst]
        ipp = [float(x) for x in ds.ImagePositionPatient]
        spacing = [float(x) for x in ds.PixelSpacing]
        iop = [float(x) for x in ds.ImageOrientationPatient]
        x0 = int(tg["pixel_x_1based"]) - 1
        y0 = int(tg["pixel_y_1based"]) - 1
        gt_X = ipp[0] + x0 * spacing[1] * iop[0] + y0 * spacing[0] * iop[3]
        gt_Y = ipp[1] + x0 * spacing[1] * iop[1] + y0 * spacing[0] * iop[4]
        gt_Z = ipp[2] + x0 * spacing[1] * iop[2] + y0 * spacing[0] * iop[5]
        target_lps_dict[pid].append({
            "nodule_index": int(tg["nodule_index"]),
            "lps": np.array([gt_X, gt_Y, gt_Z]),
            "instance": inst
        })

print(f"Loaded {len(patient_order)} patients and {sum(len(v) for v in target_lps_dict.values())} targets.", flush=True)

# Build matching records under both rules
records_rule_closest = defaultdict(list)
records_rule_highest = defaultdict(list)

target_diffs = []

for pid in patient_order:
    cands = cands_by_patient[pid]
    targets = target_lps_dict[pid]
    cand_coords = np.array([c["coord"] for c in cands]) # [4000, 3]
    cand_scores = np.array([c["score"] for c in cands])
    
    # Distance matrix [4000, n_targets]
    if len(targets) > 0:
        tg_coords = np.array([t["lps"] for t in targets])
        # dists[i, j] = dist from cand i to target j
        dists = np.linalg.norm(cand_coords[:, None, :] - tg_coords[None, :, :], axis=2)
    else:
        dists = np.empty((len(cands), 0))
        
    # RULE A: Closest Proposal
    matched_cand_idx_ruleA = {}
    dup_cand_indices_ruleA = set()
    for tg_idx, tg in enumerate(targets):
        tg_dists = dists[:, tg_idx]
        min_idx = int(np.argmin(tg_dists))
        if tg_dists[min_idx] < 5.0:
            matched_cand_idx_ruleA[tg_idx] = min_idx
            # All other cands within 5mm are dups
            dups = np.where((tg_dists < 5.0) & (np.arange(len(cands)) != min_idx))[0]
            for d in dups: dup_cand_indices_ruleA.add(d)
            
    # RULE B: Highest-Scoring Proposal per Target
    matched_cand_idx_ruleB = {}
    dup_cand_indices_ruleB = set()
    for tg_idx, tg in enumerate(targets):
        tg_dists = dists[:, tg_idx]
        within_5mm = np.where(tg_dists < 5.0)[0]
        if len(within_5mm) > 0:
            # find cand with max score
            best_score_cand = within_5mm[np.argmax(cand_scores[within_5mm])]
            matched_cand_idx_ruleB[tg_idx] = best_score_cand
            for d in within_5mm:
                if d != best_score_cand:
                    dup_cand_indices_ruleB.add(d)
                    
    # Compare matches for this scan
    for tg_idx, tg in enumerate(targets):
        cA = matched_cand_idx_ruleA.get(tg_idx)
        cB = matched_cand_idx_ruleB.get(tg_idx)
        if cA != cB:
            target_diffs.append({
                "patient_id": pid, "nodule_index": tg["nodule_index"],
                "cand_ruleA_idx": cA, "dist_ruleA": dists[cA, tg_idx] if cA is not None else None,
                "score_ruleA": cand_scores[cA] if cA is not None else None,
                "cand_ruleB_idx": cB, "dist_ruleB": dists[cB, tg_idx] if cB is not None else None,
                "score_ruleB": cand_scores[cB] if cB is not None else None
            })
            
    # Format candidate lists for FROC
    for c_i, c in enumerate(cands):
        # Rule A
        is_match_A = False
        tid_A = ""
        for tg_idx, m_idx in matched_cand_idx_ruleA.items():
            if m_idx == c_i:
                is_match_A = True
                tid_A = f"{pid}:target:{targets[tg_idx]['nodule_index']}"
                break
        records_rule_closest[pid].append({
            "patient_id": pid, "candidate_id": c["candidate_id"], "score": c["score"], "score_raw": c["score_raw"],
            "is_nodule_match": is_match_A, "is_duplicate_ignore": (c_i in dup_cand_indices_ruleA),
            "target_id": tid_A
        })
        
        # Rule B
        is_match_B = False
        tid_B = ""
        for tg_idx, m_idx in matched_cand_idx_ruleB.items():
            if m_idx == c_i:
                is_match_B = True
                tid_B = f"{pid}:target:{targets[tg_idx]['nodule_index']}"
                break
        records_rule_highest[pid].append({
            "patient_id": pid, "candidate_id": c["candidate_id"], "score": c["score"], "score_raw": c["score_raw"],
            "is_nodule_match": is_match_B, "is_duplicate_ignore": (c_i in dup_cand_indices_ruleB),
            "target_id": tid_B
        })

print(f"\n--- Target Matching Comparison ---")
print(f"Total targets where Rule A (closest) and Rule B (highest score) selected DIFFERENT proposals: {len(target_diffs)} / 73")
for td in target_diffs:
    print(f"  {td['patient_id']} Target #{td['nodule_index']}:")
    print(f"    Rule A (Closest): cand #{td['cand_ruleA_idx']+1}, dist={td['dist_ruleA']:.2f}mm, score={td['score_ruleA']:.6f}")
    print(f"    Rule B (Highest): cand #{td['cand_ruleB_idx']+1}, dist={td['dist_ruleB']:.2f}mm, score={td['score_ruleB']:.6f}")
