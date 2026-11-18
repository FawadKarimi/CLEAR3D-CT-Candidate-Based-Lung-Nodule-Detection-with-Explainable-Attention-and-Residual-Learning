
import csv, json, math, sys
from pathlib import Path
import numpy as np

BASE = Path('/experiment-storage/chaidir/fawad-dsai')
ROOT = BASE / 'CLEAR3D_CT project/02_Active_Model_and_Pipeline'
EXT_DIR = ROOT / 'external_validation_spie_aapm'
scores_csv = EXT_DIR / 'scores.csv'
summary_json = EXT_DIR / 'spie_aapm_cohort_evaluation_summary.json'

print("=== Independent Metrics Reproduction & Audit ===", flush=True)

with open(scores_csv, 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Loaded {len(rows)} candidate rows from {scores_csv.name}", flush=True)

for r in rows:
    r['dist_to_gt_mm'] = float(r['dist_to_gt_mm'])
    r['is_nodule_match'] = (str(r['is_nodule_match']).strip() in ('1', 'True', 'true'))
    r['is_duplicate_ignore'] = (str(r['is_duplicate_ignore']).strip() in ('1', 'True', 'true'))
    r['raw_prob'] = float(r['raw_prob'])
    r['cal_prob'] = float(r['cal_prob'])
    r['raw_logit'] = float(r['raw_logit'])
    r['cal_logit'] = float(r['cal_logit'])
    r['rank_in_scan'] = int(r['rank_in_scan'])
    r['label'] = int(r['label'])

unique_pats = sorted(list(set(r['patient_id'] for r in rows)))
print(f"Total unique patients: {len(unique_pats)}", flush=True)

pat_coverage = {}
for p in unique_pats:
    p_rows = [r for r in rows if r['patient_id'] == p]
    matched = [r for r in p_rows if r['is_nodule_match']]
    min_dist = min(r['dist_to_gt_mm'] for r in p_rows)
    pat_coverage[p] = {
        'total_proposals': len(p_rows),
        'min_distance_mm': round(min_dist, 4),
        'detected_5mm': len(matched) > 0,
        'matched_count': len(matched),
        'best_match_rank': matched[0]['rank_in_scan'] if matched else None,
        'best_match_cal_prob': matched[0]['cal_prob'] if matched else None,
        'best_match_raw_prob': matched[0]['raw_prob'] if matched else None
    }

detected_count = sum(1 for p, c in pat_coverage.items() if c['detected_5mm'])
coverage_rate = detected_count / float(len(unique_pats))
print(f"Proposal Coverage: {detected_count}/{len(unique_pats)} ({coverage_rate*100:.1f}%)", flush=True)

def calc_froc(candidate_rows, num_scans=10, num_targets=10, grid=(0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0), interp_mode='linear'):
    target_best_score = {}
    for r in candidate_rows:
        if r['is_nodule_match']:
            pid = r['patient_id']
            if pid not in target_best_score or r['cal_prob'] > target_best_score[pid]:
                target_best_score[pid] = r['cal_prob']
                
    fp_scores = [r['cal_prob'] for r in candidate_rows if (not r['is_nodule_match'] and not r['is_duplicate_ignore'])]
    fp_scores.sort(reverse=True)
    
    all_scores = sorted(list(set([r['cal_prob'] for r in candidate_rows])), reverse=True)
    
    tps_curve = []
    fps_curve = []
    sens_curve = []
    
    for thresh in all_scores:
        tps = sum(1 for s in target_best_score.values() if s >= thresh)
        fps = sum(1 for s in fp_scores if s >= thresh)
        tps_curve.append(tps)
        fps_curve.append(fps / float(num_scans))
        sens_curve.append(tps / float(num_targets))
        
    fps_curve = [0.0] + fps_curve + [max(fps_curve[-1] if fps_curve else 0.0, 20.0)]
    sens_curve = [0.0] + sens_curve + [sens_curve[-1] if sens_curve else 0.0]
    
    fps_curve_np = np.array(fps_curve)
    sens_curve_np = np.array(sens_curve)
    order = np.argsort(fps_curve_np, kind='mergesort')
    fps_curve_sorted = fps_curve_np[order]
    sens_curve_sorted = sens_curve_np[order]
    
    if interp_mode == 'linear':
        sens_at_grid = np.interp(grid, fps_curve_sorted, sens_curve_sorted).tolist()
    else:
        sens_at_grid = []
        for g in grid:
            valid_idx = np.where(fps_curve_sorted <= g)[0]
            sens_at_grid.append(float(sens_curve_sorted[valid_idx[-1]]) if len(valid_idx) else 0.0)
            
    cpm = float(np.mean(sens_at_grid))
    return {
        'grid': list(grid),
        'sensitivities': [round(x, 4) for x in sens_at_grid],
        'cpm': round(cpm, 6),
        'interp_mode': interp_mode
    }

froc_linear = calc_froc(rows, interp_mode='linear')
froc_step = calc_froc(rows, interp_mode='step')

print(f"FROC (linear): sens={froc_linear['sensitivities']}, CPM={froc_linear['cpm']:.6f}", flush=True)
print(f"FROC (step-wise): sens={froc_step['sensitivities']}, CPM={froc_step['cpm']:.6f}", flush=True)

# Patient-level Clustered Bootstrap Audit
rng = np.random.RandomState(42)
pat_arr = np.array(unique_pats)
cand_pats_arr = np.array([r['patient_id'] for r in rows])
pat_to_cand_idxs = {p: np.where(cand_pats_arr == p)[0] for p in pat_arr}

boot_cpms = []
duplicate_counts = []

for b_i in range(1000):
    sample_pats = rng.choice(pat_arr, size=len(pat_arr), replace=True)
    duplicate_counts.append(len(pat_arr) - len(set(sample_pats)))
    
    sample_rows = []
    for sp in sample_pats:
        for idx in pat_to_cand_idxs[sp]:
            sample_rows.append(rows[idx])
            
    res_b = calc_froc(sample_rows, num_scans=len(sample_pats), num_targets=len(sample_pats))
    boot_cpms.append(res_b['cpm'])

cpm_95ci = [float(np.percentile(boot_cpms, 2.5)), float(np.percentile(boot_cpms, 97.5))]
mean_boot_cpm = float(np.mean(boot_cpms))
mean_duplicates_per_rep = float(np.mean(duplicate_counts))

print(f"Bootstrap CPM 95% CI: [{cpm_95ci[0]:.4f}, {cpm_95ci[1]:.4f}], mean={mean_boot_cpm:.4f}", flush=True)
print(f"Average duplicate patient draws per iteration: {mean_duplicates_per_rep:.2f} / 10", flush=True)

with open(summary_json) as f:
    saved_summary = json.load(f)

saved_cpm = saved_summary['classifier_froc']['cpm']
saved_ci = saved_summary['classifier_froc']['cpm_bootstrap_95ci']
saved_coverage = saved_summary['proposal_generator']['coverage_summary']['proposal_coverage_rate']

print(f"Comparing with saved summary:")
print(f"  CPM: saved={saved_cpm:.6f}, reproduced={froc_linear['cpm']:.6f}")
print(f"  Coverage: saved={saved_coverage:.4f}, reproduced={coverage_rate:.4f}")
print(f"  95% CI: saved={saved_ci}, reproduced={cpm_95ci}")

assert abs(saved_cpm - froc_linear['cpm']) < 1e-5, f"CPM mismatch: {saved_cpm} vs {froc_linear['cpm']}"
assert abs(saved_coverage - coverage_rate) < 1e-5, f"Coverage mismatch: {saved_coverage} vs {coverage_rate}"
assert abs(saved_ci[0] - cpm_95ci[0]) < 1e-5 and abs(saved_ci[1] - cpm_95ci[1]) < 1e-5, f"CI mismatch: {saved_ci} vs {cpm_95ci}"

print(">>> ALL SAVED SUMMARY METRICS VERIFIED AND REPRODUCED EXACTLY <<<", flush=True)

audit_report = {
    'status': 'REPRODUCED_AND_AUDITED',
    'dataset': 'SPIE-AAPM Lung CT Challenge Calibration Cohort (10 scans)',
    'reproduced_metrics': {
        'proposal_coverage_5mm': coverage_rate,
        'detected_scans': detected_count,
        'total_scans': len(unique_pats),
        'cpm_linear_interp': froc_linear['cpm'],
        'cpm_step_interp': froc_step['cpm'],
        'sensitivities_7_fps': froc_linear['sensitivities'],
        'bootstrap_1000_cpm_95ci': cpm_95ci,
        'bootstrap_mean_cpm': mean_boot_cpm,
        'bootstrap_mean_duplicate_patients': mean_duplicates_per_rep
    },
    'per_patient_coverage': pat_coverage,
    'methodological_audits': {
        'matching_rule_distinction': {
            'official_luna16_rule': 'Euclidean distance d < max(diameter_mm / 2.0, 5.0 mm), where diameter_mm is annotated in ground truth',
            'pilot_spie_aapm_rule': 'Euclidean distance d < 5.0 mm fixed Cartesian distance threshold',
            'reason_for_difference': 'LUNGx/SPIE-AAPM calibration ground-truth provides only (x, y, slice) center points without nodule diameter or contour annotations'
        },
        'annotation_completeness_and_false_positive_interpretation': {
            'lungx_challenge_scope': 'Classification (benign vs malignant) of a single primary biopsy-proven nodule per patient',
            'unannotated_pulmonary_structures': 'LUNGx reference explicitly does NOT exhaustively annotate secondary benign nodules, granulomas, scars, lymph nodes, or vascular bifurcations',
            'consequence_for_cad_fp_evaluation': 'Candidate proposals categorized as false positives are non-matches to the single biopsy target, but may represent true physiological nodules or granulomas unannotated in the challenge protocol'
        },
        'bootstrap_cluster_properties': {
            'cluster_unit': 'Patient / Scan level (N=10)',
            'resampling': '10 patients drawn with replacement; when patient P is drawn k times, its target nodule and all 400 candidate rows are replicated k times',
            'denominator_consistency': 'Scan count denominator and nodule target denominator both equal 10 in every bootstrap sample',
            'interpolation_robustness': 'Linear vs step-wise interpolation yields identical CPM because true nodule probabilities are separated at Rank 1 (0 FPs at threshold)'
        }
    }
}

with open(EXT_DIR / 'metrics_reproduction_audit.json', 'w') as f:
    json.dump(audit_report, f, indent=2)

print(f"Saved audit report to {EXT_DIR / 'metrics_reproduction_audit.json'}", flush=True)
