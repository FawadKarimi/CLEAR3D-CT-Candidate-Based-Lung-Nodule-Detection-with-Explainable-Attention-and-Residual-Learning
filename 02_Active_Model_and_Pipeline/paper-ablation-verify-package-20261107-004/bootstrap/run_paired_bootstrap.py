#!/usr/bin/env python3
"""
Patient-Level Paired Bootstrap Resampling for CLEAR3D-CT Development Ablations.

Computes paired bootstrap confidence intervals for:
1. Full Fold 0 Baseline vs. Control B (No-Mining): Hard-negative mining contribution.
2. Control B (No-Mining) vs. Control A (Fine-Only): Multi-scale context fusion contribution.

Resampling unit: Patient ID (142 development patients, each corresponding 1:1 to a CT series).
Evaluation: Official LUNA16 FROC CAD evaluation at [0.125, 0.25, 0.5, 1, 2, 4, 8] FP/scan grid.
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import sklearn.metrics as skl_metrics

SCRIPT_DIR = Path(__file__).resolve().parent
GRID = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=np.float64)


def precompute_matches(pred_csv: Path, annots_csv: Path, annots_exc_csv: Path, series_uids: list[str]) -> list[dict]:
    annots = {}
    with annots_csv.open(newline='') as f:
        for row in csv.DictReader(f):
            suid = row['seriesuid']
            diam = float(row.get('diameter_mm', -1.0))
            if diam < 0: diam = 10.0
            annots.setdefault(suid, []).append({
                'x': float(row['coordX']), 'y': float(row['coordY']), 'z': float(row['coordZ']),
                'rad_sq': (diam / 2.0) ** 2, 'state': 'Included'
            })
            
    with annots_exc_csv.open(newline='') as f:
        for row in csv.DictReader(f):
            suid = row['seriesuid']
            diam = float(row.get('diameter_mm', -1.0))
            if diam < 0: diam = 10.0
            annots.setdefault(suid, []).append({
                'x': float(row['coordX']), 'y': float(row['coordY']), 'z': float(row['coordZ']),
                'rad_sq': (diam / 2.0) ** 2, 'state': 'Excluded'
            })

    cands_by_suid = defaultdict(list)
    with pred_csv.open(newline='') as f:
        for row in csv.DictReader(f):
            cands_by_suid[row['seriesuid']].append({
                'x': float(row['coordX']), 'y': float(row['coordY']), 'z': float(row['coordZ']),
                'prob': float(row['probability'])
            })
            
    scan_list = []
    for suid in series_uids:
        cands = cands_by_suid.get(suid, [])
        cands_dict = {i: c for i, c in enumerate(cands)}
        cands_rem = set(cands_dict.keys())
        scan_annots = annots.get(suid, [])
        
        detected_probs = []
        n_inc = 0
        
        for a in scan_annots:
            if a['state'] != 'Included': continue
            n_inc += 1
            ax, ay, az, rsq = a['x'], a['y'], a['z'], a['rad_sq']
            matches = []
            for cid in list(cands_rem):
                c = cands_dict[cid]
                if (ax - c['x'])**2 + (ay - c['y'])**2 + (az - c['z'])**2 < rsq:
                    matches.append(c['prob'])
                    cands_rem.remove(cid)
            if matches:
                detected_probs.append(max(matches))
                
        for a in scan_annots:
            if a['state'] != 'Excluded': continue
            ax, ay, az, rsq = a['x'], a['y'], a['z'], a['rad_sq']
            for cid in list(cands_rem):
                c = cands_dict[cid]
                if (ax - c['x'])**2 + (ay - c['y'])**2 + (az - c['z'])**2 < rsq:
                    cands_rem.remove(cid)
                    
        fp_probs = np.array([cands_dict[cid]['prob'] for cid in cands_rem], dtype=np.float32)
        scan_list.append({
            'total_nodules': n_inc,
            'detected_probs': np.array(detected_probs, dtype=np.float32),
            'fp_probs': fp_probs
        })
    return scan_list


def eval_indices(scan_data: list[dict], indices: list[int] | np.ndarray) -> tuple[float, np.ndarray]:
    total_nods = sum(scan_data[i]['total_nodules'] for i in indices)
    if total_nods == 0:
        return 0.0, np.zeros(len(GRID))
        
    det_list = [scan_data[i]['detected_probs'] for i in indices if len(scan_data[i]['detected_probs']) > 0]
    fp_list = [scan_data[i]['fp_probs'] for i in indices if len(scan_data[i]['fp_probs']) > 0]
    
    det_probs = np.concatenate(det_list) if det_list else np.empty(0, dtype=np.float32)
    fp_probs = np.concatenate(fp_list) if fp_list else np.empty(0, dtype=np.float32)
    
    y_true = np.concatenate([np.ones(len(det_probs)), np.zeros(len(fp_probs))])
    y_score = np.concatenate([det_probs, fp_probs])
    
    fpr, tpr, thresholds = skl_metrics.roc_curve(y_true, y_score)
    fps = fpr * len(fp_probs) / float(len(indices))
    sens = tpr * len(det_probs) / float(total_nods)
    
    order = np.argsort(fps, kind='mergesort')
    sens_grid = np.interp(GRID, fps[order], sens[order])
    cpm = float(np.mean(sens_grid))
    return cpm, sens_grid


def main():
    parser = argparse.ArgumentParser(description='Run patient-level paired bootstrap resampling.')
    parser.add_argument('--pkg-root', type=Path, default=SCRIPT_DIR.parent)
    parser.add_argument('--b', type=int, default=2000, help='Number of bootstrap replicates (default: 2000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--recompute', action='store_true', help='Force recomputation of candidate matching from CSVs')
    parser.add_argument('--out', type=Path, default=None, help='Output path for bootstrap_results.json')
    args = parser.parse_args()

    pkg_root = args.pkg_root
    preds_dir = pkg_root / 'predictions'
    ref_dir = pkg_root / 'reference'
    cache_file = pkg_root / 'bootstrap' / 'precomputed_scan_matches.npz'
    out_file = args.out or (pkg_root / 'bootstrap' / 'bootstrap_results.json')
    
    mapping_file = ref_dir / 'patient_series_mapping.json'
    mapping = json.load(mapping_file.open())
    p2s = mapping['patient_to_series']
    patients_list = sorted(list(p2s.keys()))
    series_uids = [p2s[p] for p in patients_list]
    N = len(patients_list)
    
    models = [
        ('fold0_baseline', preds_dir / 'fold0_baseline_scores.csv'),
        ('control_b_nomining', preds_dir / 'control_b_nomining_scores.csv'),
        ('control_a_fineonly', preds_dir / 'control_a_fineonly_scores.csv')
    ]
    
    model_scan_data = {}
    
    if cache_file.exists() and not args.recompute:
        print('Loading precomputed matches from ' + cache_file.name + '...')
        c = np.load(cache_file, allow_pickle=True)
        for mtag, _ in models:
            nods = c[mtag + '_nodules']
            dets = [c[mtag + '_det_' + str(i)] for i in range(N)]
            fps = [c[mtag + '_fp_' + str(i)] for i in range(N)]
            model_scan_data[mtag] = [
                {'total_nodules': int(nods[i]), 'detected_probs': dets[i], 'fp_probs': fps[i]}
                for i in range(N)
            ]
    else:
        print('Precomputing matches from prediction CSVs across ' + str(N) + ' patients...')
        annots_csv = ref_dir / 'annotations.csv'
        annots_exc_csv = ref_dir / 'annotations_excluded.csv'
        save_dict = {}
        for mtag, mpath in models:
            t0 = time.time()
            sdata = precompute_matches(mpath, annots_csv, annots_exc_csv, series_uids)
            model_scan_data[mtag] = sdata
            save_dict[mtag + '_nodules'] = np.array([d['total_nodules'] for d in sdata], dtype=np.int32)
            for i, d in enumerate(sdata):
                save_dict[mtag + '_det_' + str(i)] = d['detected_probs']
                save_dict[mtag + '_fp_' + str(i)] = d['fp_probs']
            print('  Precomputed ' + mtag + ' in ' + f'{time.time() - t0:.2f} s')
        np.savez_compressed(cache_file, **save_dict)
        print('Saved cache to ' + str(cache_file))

    print('')
    print('=' * 80)
    print('DEVELOPMENT COHORT POINT ESTIMATES (N = 142 patients, 192 nodules)')
    print('=' * 80)
    point_cpms = {}
    point_sens = {}
    for mtag, _ in models:
        cpm, sens = eval_indices(model_scan_data[mtag], list(range(N)))
        point_cpms[mtag] = cpm
        point_sens[mtag] = sens.tolist()
        print(f'{mtag:20s}: CPM = {cpm:.6f} | Sensitivities: {[round(s, 6) for s in sens]}')
        
    delta_point_1 = point_cpms['fold0_baseline'] - point_cpms['control_b_nomining']
    delta_point_2 = point_cpms['control_b_nomining'] - point_cpms['control_a_fineonly']
    print(f'Point Estimate Delta 1 (Fold0 - NoMining):  {delta_point_1:+.6f} CPM')
    print(f'Point Estimate Delta 2 (NoMining - FineOnly): {delta_point_2:+.6f} CPM')

    B = args.b
    seed = args.seed
    print('')
    print('=' * 80)
    print(f'PATIENT-LEVEL PAIRED BOOTSTRAP (B = {B} replicates, seed = {seed}, unit = Patient ID)')
    print('=' * 80)
    
    rng = np.random.default_rng(seed)
    bootstrap_cpms = {mtag: np.zeros(B, dtype=np.float64) for mtag, _ in models}
    
    t0 = time.time()
    for b in range(B):
        sample_idx = rng.integers(0, N, size=N)
        for mtag, _ in models:
            cpm_b, _ = eval_indices(model_scan_data[mtag], sample_idx)
            bootstrap_cpms[mtag][b] = cpm_b
            
    elapsed = time.time() - t0
    print(f'Completed {B} paired replicates in {elapsed:.2f} s ({elapsed / B * 1000:.2f} ms/replicate).')
    
    diff1 = bootstrap_cpms['fold0_baseline'] - bootstrap_cpms['control_b_nomining']
    diff2 = bootstrap_cpms['control_b_nomining'] - bootstrap_cpms['control_a_fineonly']
    
    def analyze_diff(diff: np.ndarray, point_val: float, name: str) -> dict:
        mean_val = float(np.mean(diff))
        std_err = float(np.std(diff, ddof=1))
        ci_95_low = float(np.percentile(diff, 2.5))
        ci_95_high = float(np.percentile(diff, 97.5))
        ci_90_low = float(np.percentile(diff, 5.0))
        ci_90_high = float(np.percentile(diff, 95.0))
        ci_99_low = float(np.percentile(diff, 0.5))
        ci_99_high = float(np.percentile(diff, 99.5))
        p_gt_0 = float(np.mean(diff > 0))
        p_le_0 = float(np.mean(diff <= 0))
        
        quantiles = {
            'p01': float(np.percentile(diff, 1.0)),
            'p025': ci_95_low,
            'p05': ci_90_low,
            'p10': float(np.percentile(diff, 10.0)),
            'p25': float(np.percentile(diff, 25.0)),
            'p50': float(np.percentile(diff, 50.0)),
            'p75': float(np.percentile(diff, 75.0)),
            'p90': float(np.percentile(diff, 90.0)),
            'p95': ci_90_high,
            'p975': ci_95_high,
            'p99': float(np.percentile(diff, 99.0))
        }
        
        print('')
        print('--- ' + name + ' ---')
        print(f'  Cohort Point Estimate: {point_val:+.6f}')
        print(f'  Bootstrap Mean:        {mean_val:+.6f}')
        print(f'  Bootstrap Std Error:   {std_err:.6f}')
        print(f'  95% Percentile CI:     [{ci_95_low:+.6f}, {ci_95_high:+.6f}]')
        print(f'  90% Percentile CI:     [{ci_90_low:+.6f}, {ci_90_high:+.6f}]')
        print(f'  99% Percentile CI:     [{ci_99_low:+.6f}, {ci_99_high:+.6f}]')
        print(f'  Empirical Pr(Diff > 0): {p_gt_0:.4f} ({int(p_gt_0 * len(diff))}/{len(diff)})')
        sig_str = 'YES (Not statistically significant at p < 0.05)' if (ci_95_low <= 0 <= ci_95_high) else 'NO (Statistically significant)'
        print(f'  Zero in 95% CI:        {sig_str}')
        
        return {
            'point_estimate': point_val,
            'bootstrap_mean': mean_val,
            'std_error': std_err,
            'ci_95_percentile': [ci_95_low, ci_95_high],
            'ci_90_percentile': [ci_90_low, ci_90_high],
            'ci_99_percentile': [ci_99_low, ci_99_high],
            'p_gt_0': p_gt_0,
            'p_le_0': p_le_0,
            'zero_in_95_ci': bool(ci_95_low <= 0 <= ci_95_high),
            'quantiles': quantiles
        }
        
    analysis1 = analyze_diff(diff1, delta_point_1, '1. Full Fold 0 Baseline vs. Control B (No-Mining)')
    analysis2 = analyze_diff(diff2, delta_point_2, '2. Control B (No-Mining) vs. Control A (Fine-Only)')
    
    results = {
        'status': 'PASS',
        'scope': 'exploratory development cohort paired bootstrap (142 patients, 192 nodules)',
        'limitations': (
            'Evaluated on repeatedly used development cohort (142 scans). '
            'Does NOT claim proof of generalizable isolated component effects. '
            'Both 95% percentile confidence intervals cross zero, indicating exploratory nature.'
        ),
        'configuration': {
            'replicate_count': B,
            'random_seed': seed,
            'resampling_unit': 'Patient ID (142 unique LIDC-IDRI patients)',
            'interpolation_method': 'np.interp linear piecewise over sorted FROC points',
            'evaluation_grid': GRID.tolist(),
            'total_patients': N,
            'total_ground_truth_nodules': 192
        },
        'point_estimates': {
            'fold0_baseline': {'cpm': point_cpms['fold0_baseline'], 'sensitivities': point_sens['fold0_baseline']},
            'control_b_nomining': {'cpm': point_cpms['control_b_nomining'], 'sensitivities': point_sens['control_b_nomining']},
            'control_a_fineonly': {'cpm': point_cpms['control_a_fineonly'], 'sensitivities': point_sens['control_a_fineonly']}
        },
        'paired_differences': {
            'diff1_fold0_vs_nomining': analysis1,
            'diff2_nomining_vs_fineonly': analysis2
        }
    }
    
    with out_file.open('w') as f:
        json.dump(results, f, indent=2)
    print('')
    print('Saved bootstrap summary to ' + str(out_file))
    print('=' * 80)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
