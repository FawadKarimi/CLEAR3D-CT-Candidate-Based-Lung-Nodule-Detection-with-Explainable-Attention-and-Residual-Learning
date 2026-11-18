import json, os, sys, time
from pathlib import Path
import numpy as np

def compute_froc_cpm_reference(candidate_records, num_scans, num_targets, grid=(0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)):
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

def compute_froc_cpm_unsubsampled_fast(candidate_records, num_scans, num_targets, grid=(0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)):
    if num_scans == 0 or num_targets == 0:
        return {
            "grid_fp_rates": list(grid),
            "sensitivities": [0.0]*len(grid),
            "cpm": 0.0,
            "total_fps": 0,
            "targets_detected": 0,
            "targets_total": num_targets
        }
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
    
    # UN-SUBSAMPLED: Evaluate ALL unique scores in candidate_records without any downsampling
    all_scores = np.sort(np.unique(np.concatenate([target_scores_np, fp_scores_np])))[::-1] if (len(target_scores_np) or len(fp_scores_np)) else np.array([])
    
    tps_curve = np.searchsorted(-target_scores_np, -all_scores, side="right") if len(target_scores_np) else np.zeros(len(all_scores))
    fps_curve = np.searchsorted(-fp_scores_np, -all_scores, side="right") if len(fp_scores_np) else np.zeros(len(all_scores))
    
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

print("=== Testing Un-subsampled FROC with >2,000 FPs, Duplicate Patients, Tied Scores, Missed & Overlapping Targets ===", flush=True)

# Construct challenging stress test dataset:
# 10 patients, 15 targets, 3,500 false positives (>2000!)
# Patient 1 has overlapping targets T1 and T2
# Patient 2 is sampled twice (duplicate patient with distinct copy ID)
# Patient 3 has tied scores (TP and FP have identical 0.7500000)
# Patient 4 has missed target T4
rng = np.random.RandomState(12345)
cands = []

# Targets:
# T1 (prob=0.99), T2 (prob=0.92, overlapping near T1 with duplicate cand at 0.85)
cands.append({"candidate_id": "c_t1", "score": 0.99, "is_nodule_match": True, "is_duplicate_ignore": False, "target_id": "P1:target:1"})
cands.append({"candidate_id": "c_t2", "score": 0.92, "is_nodule_match": True, "is_duplicate_ignore": False, "target_id": "P1:target:2"})
cands.append({"candidate_id": "c_t2_dup", "score": 0.85, "is_nodule_match": False, "is_duplicate_ignore": True, "target_id": "P1:target:2"})

# Tied scores: T3 has prob 0.75, FP also has prob 0.75
cands.append({"candidate_id": "c_t3", "score": 0.75, "is_nodule_match": True, "is_duplicate_ignore": False, "target_id": "P3:target:3"})
cands.append({"candidate_id": "c_fp_tied", "score": 0.75, "is_nodule_match": False, "is_duplicate_ignore": False, "target_id": ""})

# Missed target: T4 is in the denominator (num_targets includes T4) but has NO candidate hits!

# Duplicate patient P2 sampled twice with distinct IDs
for copy_idx in range(2):
    cands.append({"candidate_id": f"c_t_p2_copy{copy_idx}", "score": 0.88, "is_nodule_match": True, "is_duplicate_ignore": False, "target_id": f"P2#copy{copy_idx}:target:1"})
    for j in range(25):
        cands.append({"candidate_id": f"fp_p2_copy{copy_idx}_{j}", "score": 0.10 + 0.02*j, "is_nodule_match": False, "is_duplicate_ignore": False, "target_id": ""})

# Add 3,450 additional false positives with varied scores (>2000 total FPs)
fp_scores = rng.beta(0.5, 5.0, size=3450) # realistic long-tail false positive distribution
for i, s in enumerate(fp_scores):
    cands.append({"candidate_id": f"fp_{i}", "score": float(s), "is_nodule_match": False, "is_duplicate_ignore": False, "target_id": ""})

total_targets = 1 + 1 + 1 + 1 + 2 # T1, T2, T3, T4(missed), P2 copy0, P2 copy1 = 6 targets
total_scans = 10 # 10 scans

print(f"Total candidates created: {len(cands)}")
print(f"Total false positives: {sum(1 for c in cands if not c['is_nodule_match'] and not c['is_duplicate_ignore'])} (>2,000 threshold)")
print(f"Total targets in denominator: {total_targets}")

t0 = time.time()
res_ref = compute_froc_cpm_reference(cands, total_scans, total_targets)
t_ref = time.time() - t0
print(f"Reference FROC completed in {t_ref:.3f}s: CPM = {res_ref['cpm']:.8f}")

t0 = time.time()
res_fast = compute_froc_cpm_unsubsampled_fast(cands, total_scans, total_targets)
t_fast = time.time() - t0
print(f"Unsubsampled Fast FROC completed in {t_fast:.3f}s: CPM = {res_fast['cpm']:.8f} (Speedup: {t_ref/t_fast:.1f}x)")

cpm_diff = abs(res_ref["cpm"] - res_fast["cpm"])
sens_diffs = [abs(r - f) for r, f in zip(res_ref["sensitivities"], res_fast["sensitivities"])]
max_diff = max(sens_diffs)

print(f"\nDiscrepancy Metrics:")
print(f"  CPM Absolute Difference: {cpm_diff:.2e}")
print(f"  Max Sensitivity Difference across grid: {max_diff:.2e}")
for rate, s_r, s_f in zip(res_ref["grid_fp_rates"], res_ref["sensitivities"], res_fast["sensitivities"]):
    print(f"    @ {rate:5.3f} FP/scan: Ref={s_r*100:6.3f}% | Fast={s_f*100:6.3f}% | Diff={abs(s_r-s_f):.2e}")

assert cpm_diff < 1e-10, f"CPM mismatch: {cpm_diff}"
assert max_diff < 1e-10, f"Sensitivity mismatch: {max_diff}"
print("\nTEST PASSED WITH EXACT 0.0000000000 DISCREPANCY (< 1e-10)!", flush=True)
