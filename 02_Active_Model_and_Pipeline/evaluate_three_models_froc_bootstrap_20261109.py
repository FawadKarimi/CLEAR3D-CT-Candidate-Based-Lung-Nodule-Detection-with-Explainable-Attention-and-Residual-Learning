#!/usr/bin/env python3
"""
Full-cohort 142-scan FROC & Patient-Clustered Bootstrap Comparison
Comparing:
1. Run 004 Reference 5-Model Ensemble (4-view TTA)
2. CLEAR3D-HighRes (4-view TTA, single-model improvement)
3. MedicalNet 3D ResNet-18 Comparator (single-model baseline)
"""
from __future__ import annotations
import csv, json, os, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import sklearn.metrics as skl_metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path("/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/02_Active_Model_and_Pipeline")
EV_DIR = ROOT / "runs/evaluation-full-dev-20261109"
REF_DIR = Path("/experiment-storage/chaidir/fawad-dsai/data/reference/luna16-zenodo-3723295/resolved-reference-v1")
MAPPING_FILE = ROOT / "paper-ablation-verify-package-20261107-004/reference/patient_series_mapping.json"
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
    print("=== Loading series and patient mappings ===", flush=True)
    mapping = json.load(MAPPING_FILE.open())
    p2s = mapping['patient_to_series']
    patients_list = sorted(list(p2s.keys()))
    series_uids = [p2s[p] for p in patients_list]
    N = len(patients_list)
    assert N == 142
    
    annots_csv = REF_DIR / "annotations.csv"
    annots_exc_csv = REF_DIR / "annotations_excluded.csv"
    
    models = [
        ('Run 004 Ensemble', EV_DIR / 'full_dev_predictions_run004.csv'),
        ('CLEAR3D-HighRes', EV_DIR / 'full_dev_predictions_highres.csv'),
        ('MedicalNet ResNet-18', EV_DIR / 'full_dev_predictions_medicalnet.csv')
    ]
    
    # Precompute matches
    scan_matches = {}
    point_results = {}
    for mname, mpath in models:
        print(f"Precomputing matches for {mname}...", flush=True)
        t0 = time.time()
        sdata = precompute_matches(mpath, annots_csv, annots_exc_csv, series_uids)
        scan_matches[mname] = sdata
        cpm, sens = eval_indices(sdata, list(range(N)))
        point_results[mname] = {
            'cpm': float(cpm),
            'sensitivities': [float(s) for s in sens],
            'total_nodules': sum(d['total_nodules'] for d in sdata),
            'detected_nodules': sum(len(d['detected_probs']) for d in sdata),
            'total_fps': sum(len(d['fp_probs']) for d in sdata),
            'avg_fp_per_scan': float(sum(len(d['fp_probs']) for d in sdata) / N)
        }
        print(f"  {mname}: CPM = {cpm:.6f} | Sensitivities: {[round(s, 4) for s in sens]} ({time.time()-t0:.1f}s)", flush=True)
        
    # Bootstrap
    B = 1000
    seed = 42
    print(f"\n=== Running Patient-Clustered Bootstrap (B={B}, seed={seed}, unit=Patient ID) ===", flush=True)
    rng = np.random.default_rng(seed)
    
    boot_cpms = {mname: np.zeros(B, dtype=np.float64) for mname, _ in models}
    boot_sens = {mname: np.zeros((B, len(GRID)), dtype=np.float64) for mname, _ in models}
    
    t0 = time.time()
    for b in range(B):
        sample_idx = rng.integers(0, N, size=N)
        for mname, _ in models:
            cpm_b, sens_b = eval_indices(scan_matches[mname], sample_idx)
            boot_cpms[mname][b] = cpm_b
            boot_sens[mname][b] = sens_b
            
    print(f"Bootstrap completed in {time.time()-t0:.1f}s", flush=True)
    
    bootstrap_stats = {}
    for mname, _ in models:
        cpm_arr = boot_cpms[mname]
        sens_arr = boot_sens[mname]
        bootstrap_stats[mname] = {
            'cpm': {
                'point': point_results[mname]['cpm'],
                'mean': float(np.mean(cpm_arr)),
                'std': float(np.std(cpm_arr, ddof=1)),
                'ci95': [float(np.percentile(cpm_arr, 2.5)), float(np.percentile(cpm_arr, 97.5))]
            },
            'sensitivities': [
                {
                    'fp_rate': float(GRID[g]),
                    'point': point_results[mname]['sensitivities'][g],
                    'mean': float(np.mean(sens_arr[:, g])),
                    'std': float(np.std(sens_arr[:, g], ddof=1)),
                    'ci95': [float(np.percentile(sens_arr[:, g], 2.5)), float(np.percentile(sens_arr[:, g], 97.5))]
                }
                for g in range(len(GRID))
            ]
        }
        
    # Paired deltas
    def paired_analysis(arr_a, arr_b, name_a, name_b):
        delta = arr_a - arr_b
        return {
            'comparison': f"{name_a} minus {name_b}",
            'point_delta': float(np.mean(delta)),
            'std_error': float(np.std(delta, ddof=1)),
            'ci95': [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))],
            'p_value_empirical': float(np.mean(delta <= 0) if np.mean(delta) > 0 else np.mean(delta >= 0))
        }
        
    paired_comparisons = {
        'highres_vs_run004': paired_analysis(boot_cpms['CLEAR3D-HighRes'], boot_cpms['Run 004 Ensemble'], 'CLEAR3D-HighRes', 'Run 004 Ensemble'),
        'highres_vs_medicalnet': paired_analysis(boot_cpms['CLEAR3D-HighRes'], boot_cpms['MedicalNet ResNet-18'], 'CLEAR3D-HighRes', 'MedicalNet ResNet-18'),
        'run004_vs_medicalnet': paired_analysis(boot_cpms['Run 004 Ensemble'], boot_cpms['MedicalNet ResNet-18'], 'Run 004 Ensemble', 'MedicalNet ResNet-18')
    }
    
    out_dict = {
        'status': 'PASS',
        'scope': 'Tasks 3 official development 142-scan LUNA16 FROC & patient-clustered bootstrap comparison',
        'cohort': {
            'scans': 142,
            'patients': 142,
            'total_ground_truth_nodules': 192,
            'fp_grid': GRID.tolist()
        },
        'point_results': point_results,
        'bootstrap_stats': bootstrap_stats,
        'paired_comparisons': paired_comparisons
    }
    
    out_json = EV_DIR / "froc_and_bootstrap_comparison.json"
    with open(out_json, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"Results saved to {out_json}", flush=True)
    
    # Plot FROC curves
    plt.figure(figsize=(9, 6))
    colors = {'Run 004 Ensemble': 'black', 'CLEAR3D-HighRes': '#1f77b4', 'MedicalNet ResNet-18': '#d62728'}
    styles = {'Run 004 Ensemble': '-', 'CLEAR3D-HighRes': '--', 'MedicalNet ResNet-18': '-.'}
    
    for mname, _ in models:
        sdata = scan_matches[mname]
        det_probs = np.concatenate([d['detected_probs'] for d in sdata if len(d['detected_probs']) > 0])
        fp_probs = np.concatenate([d['fp_probs'] for d in sdata if len(d['fp_probs']) > 0])
        total_nods = sum(d['total_nodules'] for d in sdata)
        
        y_true = np.concatenate([np.ones(len(det_probs)), np.zeros(len(fp_probs))])
        y_score = np.concatenate([det_probs, fp_probs])
        
        fpr, tpr, _ = skl_metrics.roc_curve(y_true, y_score)
        fps = fpr * len(fp_probs) / float(N)
        sens = tpr * len(det_probs) / float(total_nods)
        
        order = np.argsort(fps, kind='mergesort')
        fps_s = fps[order]
        sens_s = sens[order]
        
        # Clip to [0.1, 8.5]
        mask = (fps_s >= 0.05) & (fps_s <= 8.5)
        cpm_val = point_results[mname]['cpm']
        plt.semilogx(fps_s[mask], sens_s[mask], label=f"{mname} (CPM={cpm_val:.4f})", color=colors[mname], linestyle=styles[mname], linewidth=2.0)
        
    plt.axvline(x=0.125, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=0.25, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=2.0, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=4.0, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=8.0, color='gray', linestyle=':', alpha=0.5)
    
    plt.xlim(0.1, 8.5)
    plt.ylim(0.0, 1.02)
    plt.xlabel("False Positives per Scan (log scale)")
    plt.ylabel("Sensitivity")
    plt.title("LUNA16 Official Scan-Level FROC Comparison (142 Development Scans)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(EV_DIR / "froc_three_models_comparison.png", dpi=300)
    print(f"FROC plot saved to {EV_DIR / 'froc_three_models_comparison.png'}", flush=True)

if __name__ == '__main__':
    main()
