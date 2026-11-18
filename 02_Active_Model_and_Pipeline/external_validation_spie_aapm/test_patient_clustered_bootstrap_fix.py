import json, os, sys
from pathlib import Path
import numpy as np

def compute_froc_cpm(candidate_records, num_scans, num_targets, grid=(0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)):
    if num_scans == 0 or num_targets == 0:
        return {"cpm": 0.0, "sensitivities": [0.0]*len(grid)}
    target_best_score = {}
    fp_scores = []
    for c in candidate_records:
        if c["is_nodule_match"]:
            tid = c["target_id"]
            if tid not in target_best_score or c["score"] > target_best_score[tid]:
                target_best_score[tid] = c["score"]
        elif not c["is_duplicate_ignore"]:
            fp_scores.append(c["score"])
            
    target_scores_np = np.sort(np.array(list(target_best_score.values()), dtype=np.float64))[::-1] if target_best_score else np.array([])
    fp_scores_np = np.sort(np.array(fp_scores, dtype=np.float64))[::-1] if fp_scores else np.array([])
    
    if len(fp_scores_np) > 2000:
        eval_scores = np.unique(np.concatenate([target_scores_np, fp_scores_np[:2000], fp_scores_np[2000::20]]))
        eval_scores = np.sort(eval_scores)[::-1]
    else:
        eval_scores = np.sort(np.unique(np.concatenate([target_scores_np, fp_scores_np])))[::-1] if len(fp_scores_np) else target_scores_np
        
    tps_curve = np.searchsorted(-target_scores_np, -eval_scores, side="right") if len(target_scores_np) else np.zeros(len(eval_scores))
    fps_curve = np.searchsorted(-fp_scores_np, -eval_scores, side="right") if len(fp_scores_np) else np.zeros(len(eval_scores))
    
    fps_rate = fps_curve / float(num_scans)
    sens_rate = tps_curve / float(num_targets)
    
    fps_curve_ext = np.concatenate(([0.0], fps_rate, [max(fps_rate[-1] if len(fps_rate) else 0.0, 20.0)]))
    sens_curve_ext = np.concatenate(([0.0], sens_rate, [sens_rate[-1] if len(sens_rate) else 0.0]))
    
    order = np.argsort(fps_curve_ext, kind="mergesort")
    fps_sorted = fps_curve_ext[order]
    sens_sorted = sens_curve_ext[order]
    
    sens_at_grid = np.interp(grid, fps_sorted, sens_sorted).tolist()
    cpm = float(np.mean(sens_at_grid))
    return {
        "grid_fp_rates": list(grid),
        "sensitivities": [float(x) for x in sens_at_grid],
        "cpm": cpm,
        "total_fps": len(fp_scores_np),
        "targets_detected": len(target_best_score),
        "targets_total": num_targets
    }

print("=== Running Synthetic Test: Single Scan vs. Scan Sampled Twice ===", flush=True)

# Define Patient 1: 1 true nodule (prob=0.98), 4 false positives (probs=0.6, 0.4, 0.2, 0.05)
p1_targets = [{"nodule_index": 1, "detected": True}]
p1_cands = [
    {"candidate_id": "c1", "score": 0.98, "is_nodule_match": True, "is_duplicate_ignore": False, "target_id": "P1:target:1"},
    {"candidate_id": "c2", "score": 0.60, "is_nodule_match": False, "is_duplicate_ignore": False, "target_id": ""},
    {"candidate_id": "c3", "score": 0.40, "is_nodule_match": False, "is_duplicate_ignore": False, "target_id": ""},
    {"candidate_id": "c4", "score": 0.20, "is_nodule_match": False, "is_duplicate_ignore": False, "target_id": ""},
    {"candidate_id": "c5", "score": 0.05, "is_nodule_match": False, "is_duplicate_ignore": False, "target_id": ""}
]

# Baseline evaluation on [P1] (1 scan, 1 target)
froc_single = compute_froc_cpm(p1_cands, num_scans=1, num_targets=1)
print(f"Single scan evaluation:")
print(f"  CPM: {froc_single['cpm']:.6f}")
print(f"  Sensitivities: {froc_single['sensitivities']}")
print(f"  Targets detected: {froc_single['targets_detected']}/{froc_single['targets_total']}")

# BUGGY evaluation: Sample [P1, P1] without distinct copy IDs
buggy_cands = []
for p in [p1_cands, p1_cands]:
    for c in p:
        buggy_cands.append(dict(c))
froc_buggy = compute_froc_cpm(buggy_cands, num_scans=2, num_targets=2)
print(f"\nBuggy duplicate sample [P1, P1] (shared target IDs):")
print(f"  CPM: {froc_buggy['cpm']:.6f} (ERROR: sensitivity halved!)")
print(f"  Sensitivities: {froc_buggy['sensitivities']}")
print(f"  Targets detected: {froc_buggy['targets_detected']}/{froc_buggy['targets_total']}")

# FIXED evaluation: Sample [P1, P1] with distinct copy IDs
fixed_cands = []
for copy_idx in range(2):
    for c in p1_cands:
        c_copy = dict(c)
        c_copy["candidate_id"] = f"{c['candidate_id']}#copy{copy_idx}"
        if c_copy["is_nodule_match"]:
            c_copy["target_id"] = f"P1#copy{copy_idx}:target:1"
        fixed_cands.append(c_copy)
froc_fixed = compute_froc_cpm(fixed_cands, num_scans=2, num_targets=2)
print(f"\nFixed duplicate sample [P1, P1] (distinct copy IDs):")
print(f"  CPM: {froc_fixed['cpm']:.6f} (MATCHES SINGLE SCAN EXACTLY!)")
print(f"  Sensitivities: {froc_fixed['sensitivities']}")
print(f"  Targets detected: {froc_fixed['targets_detected']}/{froc_fixed['targets_total']}")

# Assertions
assert abs(froc_fixed['cpm'] - froc_single['cpm']) < 1e-9, "CPM mismatch between single and duplicate copy!"
for s_fix, s_sin in zip(froc_fixed['sensitivities'], froc_single['sensitivities']):
    assert abs(s_fix - s_sin) < 1e-9, "Sensitivity mismatch at grid point!"
print("\nALL ASSERTIONS PASSED! Patient-clustered bootstrap fix verified with zero discrepancy!")
