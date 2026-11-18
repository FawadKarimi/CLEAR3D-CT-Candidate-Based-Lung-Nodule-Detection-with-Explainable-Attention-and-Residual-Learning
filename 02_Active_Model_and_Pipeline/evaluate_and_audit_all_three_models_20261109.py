#!/usr/bin/env python3
"""
Comprehensive LUNA16 Candidate Metrics, Official FROC, and Patient-Clustered Bootstrap Audit
For:
1. Run 004 Reference 5-Model Ensemble
2. CLEAR3D-HighRes (Single Model)
3. MedicalNet 3D ResNet-18 (Single Model Comparator, Corrected 48x48x48 valid-mask normalization)

Generates:
- metrics_summary_audit.json
- froc_and_bootstrap_comparison_audit.json
- Official LUNA16 evaluator output directories:
  * official_luna16_eval_run004/
  * official_luna16_eval_highres/
  * official_luna16_eval_medicalnet/
- froc_three_models_comparison_audit.png
- clear3d_full_development_comparison_audit_evidence_20261109.zip
"""
import os, sys, json, csv, time, math, zipfile, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, average_precision_score
import sklearn.metrics as skl_metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path("/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/02_Active_Model_and_Pipeline")
EV_DIR = ROOT / "runs/evaluation-full-dev-20261109"
REF_DIR = Path("/experiment-storage/chaidir/fawad-dsai/data/reference/luna16-zenodo-3723295/resolved-reference-v1")
MAPPING_FILE = ROOT / "paper-ablation-verify-package-20261107-004/reference/patient_series_mapping.json"
SUIDS_FILE = ROOT / "runs/clear3d-msf-official-dev-eval-20261106-004/seriesuids.csv"
GRID = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=np.float64)

# Import official LUNA16 evaluator
sys.path.insert(0, str(ROOT / "paper-ablation-verify-package-20261107-004/evaluator"))
import noduleCADEvaluationLUNA16 as ev
ev.csvTools.readCSV = lambda f: list(csv.reader(open(f, newline='')))

def compute_candidate_metrics(y_true, y_pred, threshold=0.5):
    ap = float(average_precision_score(y_true, y_pred))
    p, r, _ = precision_recall_curve(y_true, y_pred)
    pr_auc_trap = float(auc(r, p))
    roc_auc = float(roc_auc_score(y_true, y_pred))
    
    pred_bin = (y_pred >= threshold).astype(int)
    tp = int(np.sum((pred_bin == 1) & (y_true == 1)))
    tn = int(np.sum((pred_bin == 0) & (y_true == 0)))
    fp = int(np.sum((pred_bin == 1) & (y_true == 0)))
    fn = int(np.sum((pred_bin == 0) & (y_true == 1)))
    
    n = len(y_true)
    acc = float((tp + tn) / n)
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1_pos = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    f1_neg = float(2 * tn / (2 * tn + fp + fn)) if (2 * tn + fp + fn) > 0 else 0.0
    macro_f1 = float((f1_pos + f1_neg) / 2.0)
    fp_scan = float(fp / 142.0)
    
    return {
        'threshold': float(threshold),
        'ap': ap,
        'pr_auc': pr_auc_trap,
        'roc_auc': roc_auc,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'specificity': spec,
        'f1_positive': f1_pos,
        'f1_negative': f1_neg,
        'macro_f1': macro_f1,
        'fp_per_scan': fp_scan
    }

def find_optimal_threshold(y_true, y_pred):
    best_th = 0.5
    best_f1 = -1.0
    best_res = None
    for th in np.linspace(0.01, 0.99, 99):
        m = compute_candidate_metrics(y_true, y_pred, threshold=th)
        if m['f1_positive'] > best_f1:
            best_f1 = m['f1_positive']
            best_th = float(th)
            best_res = m
    return best_res

def precompute_luna16_matches(pred_csv: Path, annots_csv: Path, annots_exc_csv: Path, series_uids: list[str]) -> list[dict]:
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
    total_tps = 0
    total_fps = 0
    total_fns = 0
    total_nods = 0
    total_cands = 0
    total_irrelevant = 0
    total_double_ignored = 0
    
    for suid in series_uids:
        cands = cands_by_suid.get(suid, [])
        cands_dict = {i: c for i, c in enumerate(cands)}
        cands_rem = set(cands_dict.keys())
        scan_annots = annots.get(suid, [])
        total_cands += len(cands)
        
        detected_probs = []
        n_inc = 0
        
        # Included nodules
        for a in scan_annots:
            if a['state'] != 'Included': continue
            n_inc += 1
            total_nods += 1
            ax, ay, az, rsq = a['x'], a['y'], a['z'], a['rad_sq']
            matches = []
            for cid in list(cands_rem):
                c = cands_dict[cid]
                if (ax - c['x'])**2 + (ay - c['y'])**2 + (az - c['z'])**2 < rsq:
                    matches.append(c['prob'])
                    cands_rem.remove(cid)
            if matches:
                detected_probs.append(max(matches))
                total_tps += 1
                if len(matches) > 1:
                    total_double_ignored += (len(matches) - 1)
            else:
                total_fns += 1
                
        # Excluded nodules
        for a in scan_annots:
            if a['state'] != 'Excluded': continue
            ax, ay, az, rsq = a['x'], a['y'], a['z'], a['rad_sq']
            for cid in list(cands_rem):
                c = cands_dict[cid]
                if (ax - c['x'])**2 + (ay - c['y'])**2 + (az - c['z'])**2 < rsq:
                    cands_rem.remove(cid)
                    total_irrelevant += 1
                    
        fp_probs = np.array([cands_dict[cid]['prob'] for cid in cands_rem], dtype=np.float32)
        total_fps += len(fp_probs)
        scan_list.append({
            'suid': suid,
            'total_nodules': n_inc,
            'detected_probs': np.array(detected_probs, dtype=np.float32),
            'fp_probs': fp_probs
        })
        
    cad_stats = {
        'TP': total_tps, 'FP': total_fps, 'FN': total_fns,
        'total_nodules': total_nods, 'total_candidates': total_cands,
        'ignored_excluded': total_irrelevant, 'ignored_double': total_double_ignored,
        'sensitivity': total_tps / total_nods if total_nods > 0 else 0.0,
        'avg_fp_per_scan': total_fps / len(series_uids)
    }
    return scan_list, cad_stats

def compute_froc_curve(scan_data: list[dict], indices: list[int] | np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    total_nods = sum(scan_data[i]['total_nodules'] for i in indices)
    if total_nods == 0:
        return 0.0, np.zeros(len(GRID)), np.zeros(1), np.zeros(1)
        
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
    fps_sorted = fps[order]
    sens_sorted = sens[order]
    sens_grid = np.interp(GRID, fps_sorted, sens_sorted)
    cpm = float(np.mean(sens_grid))
    return cpm, sens_grid, fps_sorted, sens_sorted

def main():
    print("=== STARTING FULL COHORT AUDIT & EVALUATION ===", flush=True)
    mapping = json.load(MAPPING_FILE.open())
    p2s = mapping['patient_to_series']
    patients_list = sorted(list(p2s.keys()))
    series_uids = [p2s[p] for p in patients_list]
    N = len(patients_list)
    assert N == 142
    print(f"Loaded {N} patients and series UIDs.", flush=True)

    # 1. Load predictions
    models_config = [
        ('Run 004 Ensemble', EV_DIR / 'full_dev_predictions_run004.csv', 'run004'),
        ('CLEAR3D-HighRes', EV_DIR / 'full_dev_predictions_highres.csv', 'highres'),
        ('MedicalNet ResNet-18', EV_DIR / 'full_dev_predictions_medicalnet.csv', 'medicalnet'),
    ]

    dfs = {}
    for name, path, code in models_config:
        print(f"Loading predictions for {name} from {path}...", flush=True)
        dfs[name] = pd.read_csv(path)
        assert len(dfs[name]) == 122689

    # 2. Recalculate Candidate-Level Metrics
    print("\n=== RECALCULATING CANDIDATE METRICS DIRECTLY FROM PREDICTION FILES ===", flush=True)
    sel_1060_cids = set(json.load(open(ROOT / 'selection-001.json'))['dev']['selected'][i]['id'] for i in range(1060))
    sub_mask = dfs['Run 004 Ensemble']['candidate_id'].isin(sel_1060_cids).values
    assert np.sum(sub_mask) == 1060

    metrics_audit = {}
    for name, path, code in models_config:
        df = dfs[name]
        y_true_nat = df['label'].values
        y_pred_nat = df['probability'].values
        
        m_nat_05 = compute_candidate_metrics(y_true_nat, y_pred_nat, threshold=0.5)
        m_nat_opt = find_optimal_threshold(y_true_nat, y_pred_nat)
        
        y_true_sub = y_true_nat[sub_mask]
        y_pred_sub = y_pred_nat[sub_mask]
        m_sub_05 = compute_candidate_metrics(y_true_sub, y_pred_sub, threshold=0.5)
        
        metrics_audit[name] = {
            'natural_population_122689': {
                'average_precision': m_nat_05['ap'],
                'pr_auc': m_nat_05['pr_auc'],
                'roc_auc': m_nat_05['roc_auc'],
                'threshold_0_5': m_nat_05,
                'optimal_threshold': m_nat_opt,
            },
            'dev_1060_subset': {
                'average_precision': m_sub_05['ap'],
                'pr_auc': m_sub_05['pr_auc'],
                'roc_auc': m_sub_05['roc_auc'],
                'threshold_0_5': m_sub_05,
            }
        }
        print(f"{name}:")
        print(f"  Natural: AP={m_nat_05['ap']:.4f}, PR-AUC={m_nat_05['pr_auc']:.4f}, ROC-AUC={m_nat_05['roc_auc']:.4f}, Th=0.5: TP={m_nat_05['TP']}, TN={m_nat_05['TN']}, FP={m_nat_05['FP']}, FN={m_nat_05['FN']}, Prec={m_nat_05['precision']:.4f}, Rec={m_nat_05['recall']:.4f}, PosF1={m_nat_05['f1_positive']:.4f}, MacroF1={m_nat_05['macro_f1']:.4f}, FP/scan={m_nat_05['fp_per_scan']:.2f}")
        print(f"  1,060 Sub: AP={m_sub_05['ap']:.4f}, PR-AUC={m_sub_05['pr_auc']:.4f}, ROC-AUC={m_sub_05['roc_auc']:.4f}, Th=0.5: TP={m_sub_05['TP']}, TN={m_sub_05['TN']}, FP={m_sub_05['FP']}, FN={m_sub_05['FN']}, Prec={m_sub_05['precision']:.4f}, Rec={m_sub_05['recall']:.4f}, PosF1={m_sub_05['f1_positive']:.4f}, MacroF1={m_sub_05['macro_f1']:.4f}")

    metrics_audit_file = EV_DIR / "metrics_summary_audit.json"
    metrics_audit_file.write_text(json.dumps(metrics_audit, indent=2))
    print(f"Saved {metrics_audit_file}", flush=True)

    # 3. Official LUNA16 Evaluator & Numerical Equivalence
    print("\n=== RUNNING OFFICIAL LUNA16 EVALUATOR & VERIFYING NUMERICAL EQUIVALENCE ===", flush=True)
    annots_csv = REF_DIR / "annotations.csv"
    annots_exc_csv = REF_DIR / "annotations_excluded.csv"
    
    allNodules, seriesUIDs = ev.collect(str(annots_csv), str(annots_exc_csv), str(SUIDS_FILE))
    
    point_results = {}
    froc_curves = {}
    official_eval_results = {}
    
    for name, path, code in models_config:
        eval_out_dir = EV_DIR / f"official_luna16_eval_{code}"
        eval_out_dir.mkdir(exist_ok=True)
        
        # Run official evaluateCAD
        t0 = time.time()
        ev.evaluateCAD(seriesUIDs, str(path), str(eval_out_dir) + "/", allNodules, f"eval_{code}",
                       maxNumberOfCADMarks=-1, performBootstrapping=False)
        t_eval = time.time() - t0
        
        # Load official froc curve from disk
        froc_disk_path = eval_out_dir / f"froc_eval_{code}.txt"
        curves_disk = np.loadtxt(froc_disk_path, delimiter=',')
        order_disk = np.argsort(curves_disk[:, 0], kind='mergesort')
        sens_disk = np.interp(GRID, curves_disk[order_disk, 0], curves_disk[order_disk, 1])
        cpm_disk = float(np.mean(sens_disk))
        
        # Also run vectorized matcher
        scan_data, cad_stats = precompute_luna16_matches(path, annots_csv, annots_exc_csv, series_uids)
        cpm_vec, sens_vec, fps_sorted, sens_sorted = compute_froc_curve(scan_data, np.arange(N))
        
        # Check numerical equivalence
        cpm_diff = abs(cpm_disk - cpm_vec)
        sens_diff = np.max(np.abs(sens_disk - sens_vec))
        print(f"{name} Official LUNA16 Evaluator ({t_eval:.2f}s):")
        print(f"  Official Disk CPM: {cpm_disk:.6f} | Vectorized CPM: {cpm_vec:.6f} | Diff: {cpm_diff:.8e}")
        print(f"  Official Disk Sens: {[round(float(s), 4) for s in sens_disk]}")
        print(f"  Vectorized Sens:    {[round(float(s), 4) for s in sens_vec]} | Max Diff: {sens_diff:.8e}")
        assert cpm_diff < 1e-4 and sens_diff < 1e-4, f"Numerical equivalence check failed for {name}!"
        print("  NUMERICAL EQUIVALENCE VERIFIED!")
        
        point_results[name] = {
            'cpm': float(cpm_disk),
            'sensitivities': [float(s) for s in sens_disk],
            'cad_stats': cad_stats,
            'scan_data': scan_data
        }
        froc_curves[name] = (fps_sorted, sens_sorted)
        official_eval_results[name] = {
            'cpm': float(cpm_disk),
            'sensitivities': [float(s) for s in sens_disk],
            'output_dir': str(eval_out_dir),
            'cad_analysis_file': str(eval_out_dir / 'CADAnalysis.txt'),
            'froc_txt_file': str(froc_disk_path),
        }

    # 4. Patient-Clustered Bootstrap Resampling (B=1000)
    print("\n=== RUNNING PATIENT-CLUSTERED BOOTSTRAP (B=1000, seed=42) ===", flush=True)
    B = 1000
    rng = np.random.default_rng(42)
    
    bootstrap_samples = [rng.integers(0, N, size=N) for _ in range(B)]
    
    model_boot_cpms = {name: [] for name, _, _ in models_config}
    model_boot_sens = {name: [] for name, _, _ in models_config}
    
    for b_idx, idxs in enumerate(bootstrap_samples):
        for name, _, _ in models_config:
            cpm_b, sens_b, _, _ = compute_froc_curve(point_results[name]['scan_data'], idxs)
            model_boot_cpms[name].append(cpm_b)
            model_boot_sens[name].append(sens_b)
            
    bootstrap_stats = {}
    for name, _, _ in models_config:
        pt_cpm = point_results[name]['cpm']
        cpms_arr = np.array(model_boot_cpms[name], dtype=np.float64)
        sens_arr = np.array(model_boot_sens[name], dtype=np.float64) # (B, 7)
        
        cpm_boot_mean = float(np.mean(cpms_arr))
        cpm_boot_se = float(np.std(cpms_arr, ddof=1))
        cpm_ci95 = [float(np.quantile(cpms_arr, 0.025)), float(np.quantile(cpms_arr, 0.975))]
        
        sens_stats = []
        for g_idx, fp_rate in enumerate(GRID):
            s_vec = sens_arr[:, g_idx]
            sens_stats.append({
                'fp_rate': float(fp_rate),
                'point': float(point_results[name]['sensitivities'][g_idx]),
                'bootstrap_mean': float(np.mean(s_vec)),
                'bootstrap_se': float(np.std(s_vec, ddof=1)),
                'ci95': [float(np.quantile(s_vec, 0.025)), float(np.quantile(s_vec, 0.975))]
            })
            
        bootstrap_stats[name] = {
            'cpm': {
                'point': pt_cpm,
                'bootstrap_mean': cpm_boot_mean,
                'bootstrap_se': cpm_boot_se,
                'ci95': cpm_ci95
            },
            'sensitivities_grid': sens_stats
        }

    # 5. Paired Comparisons (Point Diff from Point CPMs, Bootstrap Mean Separately)
    print("\n=== PAIRED COMPARISONS (Calculated from Point CPMs & Patient-Clustered Bootstrap) ===", flush=True)
    comparisons = [
        ('highres_vs_run004', 'CLEAR3D-HighRes', 'Run 004 Ensemble'),
        ('highres_vs_medicalnet', 'CLEAR3D-HighRes', 'MedicalNet ResNet-18'),
        ('run004_vs_medicalnet', 'Run 004 Ensemble', 'MedicalNet ResNet-18'),
    ]
    
    paired_results = {}
    for comp_key, name_a, name_b in comparisons:
        pt_a = point_results[name_a]['cpm']
        pt_b = point_results[name_b]['cpm']
        point_delta = float(pt_a - pt_b) # POINT DIFFERENCE DIRECTLY FROM POINT CPMS
        
        arr_a = np.array(model_boot_cpms[name_a], dtype=np.float64)
        arr_b = np.array(model_boot_cpms[name_b], dtype=np.float64)
        deltas = arr_a - arr_b
        
        boot_mean_delta = float(np.mean(deltas))
        boot_se = float(np.std(deltas, ddof=1))
        ci95 = [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))]
        
        paired_results[comp_key] = {
            'comparison': f"{name_a} minus {name_b}",
            'point_delta': point_delta,
            'bootstrap_mean_delta': boot_mean_delta,
            'bootstrap_se': boot_se,
            'paired_patient_ci95': ci95
        }
        print(f"Comparison: {name_a} vs {name_b}")
        print(f"  Point Delta (from point CPMs): {point_delta:+.6f}")
        print(f"  Bootstrap Mean Delta:          {boot_mean_delta:+.6f}")
        print(f"  Bootstrap Standard Error:      {boot_se:.6f}")
        print(f"  Paired Patient-Level 95% CI:   [{ci95[0]:+.6f}, {ci95[1]:+.6f}]")

    # Save FROC and Bootstrap JSON Audit
    froc_audit_data = {
        'status': 'PASS',
        'scope': 'Tasks 3 official development 142-scan LUNA16 FROC & patient-clustered bootstrap comparison (Corrected MedicalNet)',
        'cohort': {
            'scans': N,
            'patients': N,
            'total_ground_truth_nodules': 192,
            'fp_grid': GRID.tolist(),
        },
        'point_results': {
            name: {
                'cpm': point_results[name]['cpm'],
                'sensitivities': point_results[name]['sensitivities'],
                'cad_stats': point_results[name]['cad_stats']
            } for name in point_results
        },
        'bootstrap_stats': bootstrap_stats,
        'paired_comparisons': paired_results,
        'official_luna16_evaluator': official_eval_results
    }
    
    froc_audit_file = EV_DIR / "froc_and_bootstrap_comparison_audit.json"
    froc_audit_file.write_text(json.dumps(froc_audit_data, indent=2))
    print(f"Saved {froc_audit_file}", flush=True)

    # 6. Generate Publication Plot
    plt.figure(figsize=(9, 6.5))
    colors = {'Run 004 Ensemble': '#1f77b4', 'CLEAR3D-HighRes': '#2ca02c', 'MedicalNet ResNet-18': '#d62728'}
    styles = {'Run 004 Ensemble': '-', 'CLEAR3D-HighRes': '--', 'MedicalNet ResNet-18': '-.'}
    
    for name, _, _ in models_config:
        fps_s, sens_s = froc_curves[name]
        cpm = point_results[name]['cpm']
        plt.plot(fps_s, sens_s, styles[name], color=colors[name], lw=2.2, label=f"{name} (CPM = {cpm:.4f})")
        
    for fp_val in GRID:
        plt.axvline(x=fp_val, color='gray', linestyle=':', alpha=0.45)
        
    plt.xscale('log')
    plt.xlim(0.125, 8.0)
    plt.ylim(0.0, 1.02)
    plt.xlabel('Average False Positives per Scan (log scale)', fontsize=12, fontweight='bold')
    plt.ylabel('Nodule Detection Sensitivity', fontsize=12, fontweight='bold')
    plt.title('LUNA16 Official Scan-Level FROC (142 Development Scans)', fontsize=13, fontweight='bold')
    plt.xticks(GRID, [str(x) for x in GRID])
    plt.grid(True, which='both', linestyle='--', alpha=0.35)
    plt.legend(loc='lower right', fontsize=11, framealpha=0.95)
    plt.tight_layout()
    plot_path = EV_DIR / "froc_three_models_comparison_audit.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved publication FROC figure to {plot_path}", flush=True)

    # 7. Create New Small Audit Verification Package
    print("\n=== CREATING NEW SMALL AUDIT VERIFICATION PACKAGE (WITHOUT CHECKPOINTS/IMAGES) ===", flush=True)
    zip_path = ROOT / "_archive_v2/evidence_zips_and_7z/clear3d_full_development_comparison_audit_evidence_20261109.zip"
    
    # Copy inference log
    mednet_log_src = Path("/tmp/mednet_corrected.log")
    mednet_log_dst = EV_DIR / "mednet_corrected_inference.log"
    if mednet_log_src.exists():
        import shutil
        shutil.copy2(mednet_log_src, mednet_log_dst)

    files_to_pack = [
        EV_DIR / "full_dev_predictions_run004.csv",
        EV_DIR / "full_dev_predictions_highres.csv",
        EV_DIR / "full_dev_predictions_medicalnet.csv",
        EV_DIR / "full_dev_predictions_medicalnet_mismatched_raw32_backup.csv",
        EV_DIR / "metrics_summary_audit.json",
        EV_DIR / "froc_and_bootstrap_comparison_audit.json",
        EV_DIR / "froc_three_models_comparison_audit.png",
        mednet_log_dst,
        ROOT / "score_full_dev_cohort_20261109.py",
        ROOT / "train_medicalnet_resnet18_comparator.py",
        ROOT / "run_medicalnet_full_dev_corrected.py",
        ROOT / "evaluate_and_audit_all_three_models_20261109.py",
        ROOT / "selection-001.json",
        ROOT / "balanced-candidate-classification-20261106-001.json",
        MAPPING_FILE,
        SUIDS_FILE,
        ROOT / "runs/comparator-medicalnet-resnet18-20261109/dev_predictions_1060.csv",
        ROOT / "runs/comparator-medicalnet-resnet18-20261109/medicalnet_resnet18_evaluation_summary.json",
    ]
    
    # Also add the official evaluator output directories
    for code in ['run004', 'highres', 'medicalnet']:
        d = EV_DIR / f"official_luna16_eval_{code}"
        for f in d.glob("*"):
            if f.is_file():
                files_to_pack.append(f)
                
    manifest = {}
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files_to_pack:
            if not f.exists():
                print(f"Warning: {f} does not exist, skipping")
                continue
            arcname = str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else f.name
            zf.write(f, arcname=arcname)
            h = hashlib.sha256(f.read_bytes()).hexdigest()
            manifest[arcname] = {
                'size_bytes': f.stat().st_size,
                'sha256': h
            }
            
    zip_size = zip_path.stat().st_size
    zip_sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    
    manifest_audit_file = EV_DIR / "SHA256SUMS_AUDIT.json"
    manifest_audit_file.write_text(json.dumps(manifest, indent=2))
    
    print(f"Small Audit Archive created: {zip_path}")
    print(f"Size: {zip_size:,} bytes ({zip_size / 1024**2:.2f} MB)")
    print(f"SHA-256: {zip_sha}")
    print("=== AUDIT AND EVALUATION COMPLETE ===", flush=True)

if __name__ == '__main__':
    main()
