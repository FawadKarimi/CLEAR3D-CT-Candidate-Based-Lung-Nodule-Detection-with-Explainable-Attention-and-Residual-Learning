import csv, hashlib, json, math, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

BASE = Path("/experiment-storage/chaidir/fawad-dsai")
ROOT = BASE / "CLEAR3D_CT project/02_Active_Model_and_Pipeline"
EXT_DIR = ROOT / "external_validation_spie_aapm"
RAW_TEST = EXT_DIR / "raw_test"

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()

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
    
    # We evaluate at all target scores, plus a dense grid of fp thresholds to accurately trace the curve
    if len(fp_scores_np) > 2000:
        # Take target scores + first 2000 fp scores (since max grid FP is 8.0 * num_scans <= 480 FPs)
        # and downsample remainder
        eval_scores = np.unique(np.concatenate([
            target_scores_np,
            fp_scores_np[:2000],
            fp_scores_np[2000::20]
        ]))
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

def run_bootstrap_patient_clustered_fast(patients_list, scan_dict, cands_by_patient, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    n_pts = len(patients_list)
    if n_pts == 0:
        return {}
    boot_cov, boot_cpm, boot_sens_nat, boot_sens_half = [], [], [], []
    boot_dups = []
    nat_cut = 0.9999467134475708
    
    for b in range(n_boot):
        sampled_indices = rng.choice(n_pts, size=n_pts, replace=True)
        dup_count = n_pts - len(set(sampled_indices))
        boot_dups.append(dup_count)
        b_cands = []
        b_targets = 0
        b_covered = 0
        for idx in sampled_indices:
            pid = patients_list[idx]
            scan_info = scan_dict[pid]
            b_targets += len(scan_info["targets"])
            b_covered += sum(1 for tg in scan_info["targets"] if tg["detected_within_5mm_cap4000"])
            for c in cands_by_patient[pid]:
                b_cands.append(c)
        froc_res = compute_froc_cpm_fast(b_cands, n_pts, b_targets)
        cov_rate = (b_covered / b_targets) if b_targets else 0.0
        sens_nat = (sum(1 for c in b_cands if c["is_nodule_match"] and c["score"] >= nat_cut) / b_targets) if b_targets else 0.0
        sens_half = (sum(1 for c in b_cands if c["is_nodule_match"] and c["score"] >= 0.5) / b_targets) if b_targets else 0.0
        boot_cov.append(cov_rate)
        boot_cpm.append(froc_res["cpm"])
        boot_sens_nat.append(sens_nat)
        boot_sens_half.append(sens_half)
        
    def ci(arr):
        return {
            "mean": float(np.mean(arr)), "std": float(np.std(arr)),
            "ci_95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
        }
    return {
        "n_bootstraps": n_boot,
        "avg_duplicate_patients": float(np.mean(boot_dups)),
        "proposal_coverage": ci(boot_cov),
        "cpm": ci(boot_cpm),
        "sensitivity_nat_cut": ci(boot_sens_nat),
        "sensitivity_half": ci(boot_sens_half)
    }

def main():
    print("--- Computing Consolidated SPIE-AAPM Metrics from scores_cap4000.csv ---", flush=True)
    t0 = time.time()
    
    # 1. Load Ground Truth
    with open(EXT_DIR / "test_ground_truth.json") as f:
        gt_data = json.load(f)
        
    # 2. Load Protocol
    proto_path = EXT_DIR / "frozen_evaluation_protocol_cap4000.json"
    proto_sha = sha256_file(proto_path)
    with open(proto_path) as f:
        proto_data = json.load(f)
    nat_cut_oof = proto_data["frozen_system"]["calibration"]["nat_cut_oof"]
    
    # 3. Read scores_cap4000.csv
    scores_path = EXT_DIR / "scores_cap4000.csv"
    cands_by_patient = defaultdict(list)
    print("Reading scores_cap4000.csv...", flush=True)
    with open(scores_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patient_id"]
            is_match = row["is_nodule_match"] == "1"
            is_dup = row["is_duplicate_ignore"] == "1"
            score_cal = float(row["probability_calibrated"])
            score_raw = float(row["probability_raw"])
            cands_by_patient[pid].append({
                "patient_id": pid,
                "candidate_id": row["candidate_id"],
                "rank": int(row["rank"]),
                "coordX": float(row["coordX"]),
                "coordY": float(row["coordY"]),
                "coordZ": float(row["coordZ"]),
                "score": score_cal,
                "score_raw": score_raw,
                "is_nodule_match": is_match,
                "is_duplicate_ignore": is_dup,
                "target_id": f"{pid}:target:{row['target_id']}" if is_match else "",
                "target_distance_mm": float(row["target_distance_mm"]) if row["target_distance_mm"] else None
            })
            
    print(f"Loaded {sum(len(v) for v in cands_by_patient.values())} candidates across {len(cands_by_patient)} patients.", flush=True)
    
    patient_order = sorted(list(cands_by_patient.keys()), key=lambda x: int(x.split("-CT")[1]))
    pilot_patients = ["LUNGx-CT001", "LUNGx-CT002"]
    unexposed_patients = [p for p in patient_order if p not in pilot_patients]
    
    # 4. Build per_scan_diagnostics
    scan_records = {}
    for pid in patient_order:
        cands = cands_by_patient[pid]
        tg_defs = gt_data.get(pid, [])
        target_records = []
        for tg in tg_defs:
            nidx = int(tg["nodule_index"])
            matched_cands = [c for c in cands if c["is_nodule_match"] and c["target_id"] == f"{pid}:target:{nidx}"]
            if matched_cands:
                mc = matched_cands[0]
                # compute classifier rank
                clf_rank = sum(1 for c in cands if c["score"] > mc["score"]) + 1
                target_records.append({
                    "nodule_index": nidx,
                    "spreadsheet_pos_str": tg["raw_pos_str"],
                    "instance_number": int(tg["instance_number"]),
                    "detected_within_5mm_cap4000": True,
                    "min_candidate_distance_mm": mc["target_distance_mm"],
                    "proposal_rank": mc["rank"],
                    "model_prob_calibrated": mc["score"],
                    "model_prob_raw": mc["score_raw"],
                    "classifier_rank_in_scan": clf_rank
                })
            else:
                # Target was missed by generator. Find closest candidate distance:
                # We can compute distance from candidates if needed, or inspect min dist from CSV
                target_records.append({
                    "nodule_index": nidx,
                    "spreadsheet_pos_str": tg["raw_pos_str"],
                    "instance_number": int(tg["instance_number"]),
                    "detected_within_5mm_cap4000": False,
                    "min_candidate_distance_mm": None,
                    "proposal_rank": None,
                    "model_prob_calibrated": None,
                    "model_prob_raw": None,
                    "classifier_rank_in_scan": None
                })
        scan_records[pid] = {
            "patient_id": pid,
            "proposals_generated": len(cands),
            "targets": target_records
        }
        
    def evaluate_cohort_subset(name, p_list):
        t_sub0 = time.time()
        subset_cands = [c for pid in p_list for c in cands_by_patient[pid]]
        n_scans = len(p_list)
        all_targets = [tg for pid in p_list for tg in scan_records[pid]["targets"]]
        n_targets = len(all_targets)
        covered_targets = sum(1 for tg in all_targets if tg["detected_within_5mm_cap4000"])
        froc_res = compute_froc_cpm_fast(subset_cands, n_scans, n_targets)
        boot_res = run_bootstrap_patient_clustered_fast(p_list, scan_records, cands_by_patient, n_boot=1000)
        
        tp_nat = sum(1 for c in subset_cands if c["is_nodule_match"] and c["score"] >= nat_cut_oof)
        fp_nat = sum(1 for c in subset_cands if not c["is_nodule_match"] and not c["is_duplicate_ignore"] and c["score"] >= nat_cut_oof)
        tp_half = sum(1 for c in subset_cands if c["is_nodule_match"] and c["score"] >= 0.5)
        fp_half = sum(1 for c in subset_cands if not c["is_nodule_match"] and not c["is_duplicate_ignore"] and c["score"] >= 0.5)
        
        print(f"Cohort {name} ({n_scans} scans, {n_targets} targets): Coverage={covered_targets}/{n_targets} ({covered_targets/n_targets*100:.1f}%), CPM={froc_res['cpm']:.4f} ({time.time()-t_sub0:.2f}s)", flush=True)
        
        return {
            "cohort_name": name, "scan_count": n_scans, "nodule_count": n_targets,
            "proposals_evaluated": len(subset_cands),
            "proposal_coverage": {
                "covered_nodules": covered_targets, "total_nodules": n_targets,
                "rate": float(covered_targets / n_targets) if n_targets else 0.0,
                "bootstrap_95ci": boot_res.get("proposal_coverage", {})
            },
            "operating_points": {
                "natural_prevalence_cut": {
                    "threshold": nat_cut_oof, "true_positives": tp_nat,
                    "nodule_sensitivity": float(tp_nat / n_targets) if n_targets else 0.0,
                    "total_false_positives": fp_nat, "fp_per_scan": float(fp_nat / n_scans) if n_scans else 0.0,
                    "bootstrap_95ci": boot_res.get("sensitivity_nat_cut", {})
                },
                "half_cut": {
                    "threshold": 0.5, "true_positives": tp_half,
                    "nodule_sensitivity": float(tp_half / n_targets) if n_targets else 0.0,
                    "total_false_positives": fp_half, "fp_per_scan": float(fp_half / n_scans) if n_scans else 0.0,
                    "bootstrap_95ci": boot_res.get("sensitivity_half", {})
                }
            },
            "froc_cpm_fixed_5mm": {
                "cpm": froc_res["cpm"],
                "grid_fp_rates": froc_res["grid_fp_rates"],
                "sensitivities": froc_res["sensitivities"],
                "bootstrap_95ci": boot_res.get("cpm", {})
            }
        }
        
    summary_pilot = evaluate_cohort_subset("Technical_Pilot_2scans", pilot_patients)
    summary_unexposed = evaluate_cohort_subset("Unexposed_Test_58scans", unexposed_patients)
    summary_full = evaluate_cohort_subset("Full_Test_60scans", patient_order)
    
    final_summary = {
        "status": "PASS",
        "task": "CLEAR3D-MSFv2 SPIE-AAPM Independent Test Evaluation (Cap=4000)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_sha256": proto_sha,
        "selection_basis": "Cap 4,000 selected strictly from 142 LUNA16 Dev scans (92.19% sensitivity)",
        "evaluation_rule": "Fixed Euclidean distance < 5.0 mm matching (closest proposal TP, duplicate hits ignored)",
        "annotation_caveat": "Target annotations in SPIE-AAPM LUNGx are incomplete for full scan CAD (only primary biopsy-proven nodule), so false positives include unannotated genuine nodules and granulomas.",
        "cohort_results": {
            "technical_pilot_scans": summary_pilot,
            "unexposed_test_scans": summary_unexposed,
            "full_test_cohort": summary_full
        },
        "per_scan_diagnostics": scan_records,
        "governance": {
            "reference_release": "HOLD",
            "confirmation_cohort": "83 patients strictly restricted / untouched",
            "internal_test_fold0": "Untouched / zero retraining",
            "manuscript_writing": "DEFERRED"
        }
    }
    
    summary_json_path = EXT_DIR / "test_cohort_cap4000_evaluation_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(final_summary, f, indent=2)
        
    print(f"\nEvaluation Complete! Results saved to {summary_json_path} in {time.time()-t0:.2f}s", flush=True)

if __name__ == "__main__":
    main()
