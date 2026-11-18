
import numpy as np

p1_cands = [
    {"candidate_id": "c1", "score": 0.92, "is_nodule_match": True, "is_duplicate_ignore": False, "target_id": "P1:t1"},
    {"candidate_id": "c2", "score": 0.8, "is_nodule_match": False, "is_duplicate_ignore": False, "target_id": ""},
    {"candidate_id": "c3", "score": 0.7, "is_nodule_match": False, "is_duplicate_ignore": False, "target_id": ""},
    {"candidate_id": "c4", "score": 0.6, "is_nodule_match": False, "is_duplicate_ignore": False, "target_id": ""},
    {"candidate_id": "c5", "score": 0.5, "is_nodule_match": False, "is_duplicate_ignore": False, "target_id": ""},
    {"candidate_id": "c6", "score": 0.4, "is_nodule_match": False, "is_duplicate_ignore": False, "target_id": ""},
]
p2_cands = [
    {"candidate_id": "c7", "score": 0.85, "is_nodule_match": True, "is_duplicate_ignore": False, "target_id": "P2:t1"},
    {"candidate_id": "c8", "score": 0.60, "is_nodule_match": True, "is_duplicate_ignore": False, "target_id": "P2:t2"},
    {"candidate_id": "c9", "score": 0.9, "is_nodule_match": False, "is_duplicate_ignore": False, "target_id": ""},
    {"candidate_id": "c10", "score": 0.3, "is_nodule_match": False, "is_duplicate_ignore": False, "target_id": ""},
    {"candidate_id": "c11", "score": 0.2, "is_nodule_match": False, "is_duplicate_ignore": False, "target_id": ""},
]

from test_froc_unsubsampled_validation import compute_froc_cpm_unsubsampled_fast

draw = [0, 0, 1]
cands_dict = [p1_cands, p2_cands]
all_b_cands = []
for copy_idx, idx in enumerate(draw):
    for c in cands_dict[idx]:
        c_copy = dict(c)
        c_copy["candidate_id"] = f"{c['candidate_id']}#copy{copy_idx}"
        if c_copy["is_nodule_match"]:
            c_copy["target_id"] = f"{c['target_id']}#copy{copy_idx}"
        all_b_cands.append(c_copy)

res_dict = compute_froc_cpm_unsubsampled_fast(all_b_cands, 3, 4)

p1_tg = np.array([0.92])
p1_fps = np.array([0.8, 0.7, 0.6, 0.5, 0.4])
p2_tg = np.array([0.85, 0.60])
p2_fps = np.array([0.9, 0.3, 0.2])

all_tg = np.concatenate([p1_tg, p1_tg, p2_tg])
all_fps = np.concatenate([p1_fps, p1_fps, p2_fps])

grid = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
target_scores_np = np.sort(all_tg)[::-1]
fp_scores_np = np.sort(all_fps)[::-1]
all_scores = np.sort(np.unique(np.concatenate([target_scores_np, fp_scores_np])))[::-1]
tps_curve = np.searchsorted(-target_scores_np, -all_scores, side="right")
fps_curve = np.searchsorted(-fp_scores_np, -all_scores, side="right")
fps_rate = fps_curve / 3.0
sens_rate = tps_curve / 4.0
fps_curve_ext = np.concatenate(([0.0], fps_rate, [max(fps_rate[-1], 20.0)]))
sens_curve_ext = np.concatenate(([0.0], sens_rate, [sens_rate[-1]]))
order = np.argsort(fps_curve_ext, kind="mergesort")
sens_at_grid = np.interp(grid, fps_curve_ext[order], sens_curve_ext[order]).tolist()
cpm_arr = float(np.mean(sens_at_grid))

print("CPM Dict:", res_dict["cpm"])
print("CPM Array:", cpm_arr)
print("Sens Dict:", res_dict["sensitivities"])
print("Sens Array:", sens_at_grid)
assert abs(res_dict["cpm"] - cpm_arr) < 1e-12, "Mismatch!"
print("EXACT MATCH VERIFIED!")
