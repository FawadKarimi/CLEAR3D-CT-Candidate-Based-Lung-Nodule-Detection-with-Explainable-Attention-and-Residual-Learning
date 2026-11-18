import json, os, sys
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

def compute_froc_cpm_fast(candidate_records, num_scans, num_targets, grid=(0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)):
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

print("=== Independent FROC Validation Suite ===", flush=True)

test_cases = [
    # Case 1: Duplicate candidates for single target
    {
        "name": "Duplicate candidates matching same target",
        "num_scans": 2, "num_targets": 1,
        "cands": [
            {"candidate_id": "c1", "score": 0.95, "is_nodule_match": True, "is_duplicate_ignore": False, "target_id": "T1"},
            {"candidate_id": "c2", "score": 0.85, "is_nodule_match": False, "is_duplicate_ignore": True, "target_id": "T1"},
            {"candidate_id": "c3", "score": 0.70, "is_nodule_match": False, "is_duplicate_ignore": True, "target_id": "T1"},
            {"candidate_id": "f1", "score": 0.80, "is_nodule_match": False, "is_duplicate_ignore": False, "target_id": ""},
            {"candidate_id": "f2", "score": 0.50, "is_nodule_match": False, "is_duplicate_ignore": False, "target_id": ""},
            {"candidate_id": "f3", "score": 0.30, "is_nodule_match": False, "is_duplicate_ignore": False, "target_id": ""},
            {"candidate_id": "f4", "score": 0.10, "is_nodule_match": False, "is_duplicate_ignore": False, "target_id": ""}
        ]
    },
    # Case 2: Overlapping targets
    {
        "name": "Overlapping targets in close proximity",
        "num_scans": 1, "num_targets": 2,
        "cands": [
            {"candidate_id": "c1", "score": 0.92, "is_nodule_match": True, "is_duplicate_ignore": False, "target_id": "T1"},
            {"candidate_id": "c2", "score": 0.88, "is_nodule_match": True, "is_duplicate_ignore": False, "target_id": "T2"},
            {"candidate_id": "c3", "score": 0.75, "is_nodule_match": False, "is_duplicate_ignore": True, "target_id": "T1"},
            {"candidate_id": "f1", "score": 0.60, "is_nodule_match": False, "is_duplicate_ignore": False, "target_id": ""},
            {"candidate_id": "f2", "score": 0.40, "is_nodule_match": False, "is_duplicate_ignore": False, "target_id": ""}
        ]
    },
    # Case 3: Tied scores between TP and FP
    {
        "name": "Tied scores between TPs and FPs",
        "num_scans": 2, "num_targets": 2,
        "cands": [
            {"candidate_id": "c1", "score": 0.50, "is_nodule_match": True, "is_duplicate_ignore": False, "target_id": "T1"},
            {"candidate_id": "c2", "score": 0.50, "is_nodule_match": True, "is_duplicate_ignore": False, "target_id": "T2"},
            {"candidate_id": "f1", "score": 0.50, "is_nodule_match": False, "is_duplicate_ignore": False, "target_id": ""},
            {"candidate_id": "f2", "score": 0.50, "is_nodule_match": False, "is_duplicate_ignore": False, "target_id": ""}
        ]
    },
    # Case 4: Missed targets (proposals failed to cover target)
    {
        "name": "Missed targets with zero matching candidates",
        "num_scans": 3, "num_targets": 3,
        "cands": [
            {"candidate_id": "c1", "score": 0.99, "is_nodule_match": True, "is_duplicate_ignore": False, "target_id": "T1"},
            # T2 and T3 have no proposals within 5mm
            {"candidate_id": "f1", "score": 0.60, "is_nodule_match": False, "is_duplicate_ignore": False, "target_id": ""},
            {"candidate_id": "f2", "score": 0.30, "is_nodule_match": False, "is_duplicate_ignore": False, "target_id": ""}
        ]
    },
    # Case 5: Realistic multi-scan case with 5 scans and varied FP rates
    {
        "name": "Multi-scan realistic scenario (5 scans, 6 targets)",
        "num_scans": 5, "num_targets": 6,
        "cands": [
            {"candidate_id": f"tp_{i}", "score": 0.95 - 0.08*i, "is_nodule_match": True, "is_duplicate_ignore": False, "target_id": f"T{i+1}"} for i in range(5)
        ] + [
            # T6 is missed
            {"candidate_id": f"dup_{i}", "score": 0.80 - 0.05*i, "is_nodule_match": False, "is_duplicate_ignore": True, "target_id": f"T1"} for i in range(3)
        ] + [
            {"candidate_id": f"fp_{i}", "score": 0.90 / (1.0 + 0.15*i), "is_nodule_match": False, "is_duplicate_ignore": False, "target_id": ""} for i in range(40)
        ]
    }
]

for idx, tc in enumerate(test_cases, 1):
    ref_res = compute_froc_cpm_reference(tc["cands"], tc["num_scans"], tc["num_targets"])
    fast_res = compute_froc_cpm_fast(tc["cands"], tc["num_scans"], tc["num_targets"])
    
    cpm_diff = abs(ref_res["cpm"] - fast_res["cpm"])
    sens_diffs = [abs(r - f) for r, f in zip(ref_res["sensitivities"], fast_res["sensitivities"])]
    max_sens_diff = max(sens_diffs)
    
    print(f"[{idx}/5] Test: {tc['name']}")
    print(f"  Reference CPM: {ref_res['cpm']:.6f} | Fast CPM: {fast_res['cpm']:.6f} | Diff: {cpm_diff:.2e}")
    print(f"  Max Sensitivity Diff across grid: {max_sens_diff:.2e}")
    
    assert cpm_diff < 1e-6, f"CPM mismatch in test {idx}: diff={cpm_diff}"
    assert max_sens_diff < 1e-6, f"Sensitivity mismatch in test {idx}: diff={max_sens_diff}"
    assert ref_res["total_fps"] == fast_res["total_fps"], "FP count mismatch!"
    assert ref_res["targets_detected"] == fast_res["targets_detected"], "Target count mismatch!"

print("\nALL 5 INDEPENDENT FROC IMPLEMENTATION TESTS PASSED WITH ZERO DISCREPANCY (< 1e-6)!", flush=True)
