"""Locked CLEAR3D-MSFv2 Run 004 Internal Test Evaluation on LUNA16 Fold 0 (89 scans).
- Model: Frozen Run 004 5-model ensemble (fold-0.pt .. fold-4.pt)
- Preprocessing: 1.0 mm isotropic LPS resampling
- Candidates: Official LUNA16 candidates_V2.csv (79,135 candidates across 89 test series)
- Inference: 4-view TTA (Identity, X flip, YZ flip, XYZ flip), pre-sigmoid logit averaging across 5 models
- Evaluator: Adapted official LUNA16 noduleCADEvaluationLUNA16.py (strict inequality distance < radius)
- Metrics: CPM, 7 FROC sensitivities, candidate-level discrimination at nat_cut_oof (0.9999467), patient-level bootstrap CIs
"""
import sys, os, time, json, csv, math, tempfile, zipfile, hashlib
from collections import defaultdict
from pathlib import Path
from lib2to3.refactor import RefactoringTool, get_fixers_from_package

import numpy as np
import torch

BASE = Path('/experiment-storage/chaidir/fawad-dsai')
ROOT = BASE / 'CLEAR3D_CT project/02_Active_Model_and_Pipeline'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE / 'CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/src'))

import train_clear3d_msf_ensemble_20261106_004 as train_mod
from clear3d_msf_model_20261106_002 import CLEAR3DMSFv2, build_focus_grids, make_inputs
from train_clear3d_msf_ensemble_20261106_004 import gpu_extract_from_p51, load_scan_arrays
from lidc3d.geometry import crop

TRAIN_OUT = ROOT / 'runs/clear3d-msf-ensemble-20261106-004'
CHECKPOINTS = [
    ('fold-0.pt', '739d1d7883d66922f568be4cdc66f491bdabe8eb9e1c2c22ecb8c5bf7b8f5494'),
    ('fold-1.pt', '547c56eef21b55569e611a5d2a05f0418c59b8258b0ac45d1e1d814a4623d6e4'),
    ('fold-2.pt', '9e93f0474463b3a15953391ec7adf1ac0ed6d3ed7a3d80712aef240022625290'),
    ('fold-3.pt', '2ad01868acb7a3a008b8f26584f556dffb9b05d4dccb7f137039ad09f6782758'),
    ('fold-4.pt', '90d4fb48ecba46ef95e9a55563ba7d4c7aa0d19236f922d96f2f80431901727a'),
]

TEST_MANIFEST = BASE / 'CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-cohort-fold0-v1/cohort_test_WITHHOLD.jsonl'
CAND_FILE = BASE / 'data/reference/luna16-zenodo-3723295/resolved-reference-v1/candidates_V2.csv'
REFD = BASE / 'data/reference/luna16-zenodo-3723295/resolved-reference-v1'
EVAL_ZIP = BASE / 'data/reference/luna16-zenodo-3723295/evaluationScript.zip'

PREP = BASE / 'CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-test-fold0-v1/prepared'
train_mod.PREP = PREP

OUT = ROOT / 'runs/clear3d-msf-official-test-fold0-001'
OUT.mkdir(parents=True, exist_ok=True)

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def main():
    t_start_total = time.time()
    print("=== STARTING CLEAR3D-CT TEST-FOLD0 EVALUATION ===")
    print("Verifying checkpoint integrity...")
    device = torch.device('cuda:1')
    models = []
    scale_oof, bias_oof, nat_cut_oof = None, None, None
    for name, expected_h in CHECKPOINTS:
        p = TRAIN_OUT / 'checkpoints' / name
        actual_h = sha256_file(p)
        assert actual_h == expected_h, f"Checkpoint {name} sha mismatch"
        ck = torch.load(p, map_location='cpu', weights_only=False)
        m = CLEAR3DMSFv2()
        m.load_state_dict(ck['model'], strict=True)
        m.to(device).eval()
        models.append(m)
        scale_oof = float(ck['scale_oof'])
        bias_oof = float(ck['bias_oof'])
        nat_cut_oof = float(ck['nat_cut_oof'])
    print(f"Loaded 5 models on {device}. scale={scale_oof}, bias={bias_oof}, cut={nat_cut_oof}")
    f_fine, f_ctx = build_focus_grids(device)

    print("Loading test cohort manifest...")
    test_scans = {}
    with open(TEST_MANIFEST) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                test_scans[d['series_uid']] = d
    print(f"Test scans in manifest: {len(test_scans)}")
    assert len(test_scans) == 89, f"Expected 89 test scans, got {len(test_scans)}"

    print("Reading candidates from candidates_V2.csv...")
    groups = defaultdict(list)
    total_cands = 0
    with open(CAND_FILE) as f:
        r = csv.reader(f)
        next(r)
        for row_idx, row in enumerate(r, 1):
            ser = row[0]
            if ser in test_scans:
                meta = test_scans[ser]
                groups[meta['scan_id']].append({
                    'series': ser,
                    'patient': meta['patient_id'],
                    'scan': meta['scan_id'],
                    'x': float(row[1]),
                    'y': float(row[2]),
                    'z': float(row[3]),
                    'label': int(row[4]),
                    'id': f"test:{row_idx}"
                })
                total_cands += 1
    print(f"Loaded {total_cands} candidates across {len(groups)} test scans.")
    assert total_cands == 79135, f"Expected 79135 candidates, got {total_cands}"

    scores_csv = OUT / 'scores.csv'
    all_scores = []
    all_labels = []
    all_series_coords = []
    series_uids = []
    patient_ids = []
    candidate_patient_map = []
    total_scored = 0

    t_start_scoring = time.time()
    with scores_csv.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability', 'candidate_id', 'rank', 'label'])
        
        for scan_idx, (sid, rr) in enumerate(sorted(groups.items()), 1):
            t_scan0 = time.time()
            r0 = rr[0]
            series_uids.append(r0['series'])
            patient_ids.append(r0['patient'])
            
            img, val, inv = load_scan_arrays(sid, r0['patient'], r0['series'], use_mmap=True)
            xyz1 = np.array([[r['x'], r['y'], r['z'], 1.0] for r in rr], dtype=np.float64)
            zyxs = (xyz1 @ inv.T)[:, :3]
            z_ord = np.argsort(zyxs[:, 0], kind='mergesort')
            zyxs_ord = zyxs[z_ord]
            scan_probs = [0.0] * len(rr)

            for start in range(0, len(rr), 64):
                czyx = zyxs_ord[start:start + 64]
                orig_idx = z_ord[start:start + 64]
                m_cnt = len(czyx)
                p51_b = np.empty((m_cnt, 51, 51, 51), dtype=np.float32)
                v51_b = np.empty((m_cnt, 51, 51, 51), dtype=np.float32)
                for k_i, zc in enumerate(czyx):
                    p51_b[k_i], _ = crop(img, zc, 51, 0)
                    v51_b[k_i], _ = crop(val, zc, 51, False)
                with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
                    fi_g, fv_g, ci_g, cv_g = gpu_extract_from_p51(
                        torch.from_numpy(p51_b).to(device),
                        torch.from_numpy(v51_b).to(device),
                    )
                    vf, vc = make_inputs(fi_g, fv_g, ci_g, cv_g, f_fine, f_ctx)
                    views = [
                        (vf, vc),
                        (torch.flip(vf, [4]), torch.flip(vc, [4])),
                        (torch.flip(vf, [2, 3]), torch.flip(vc, [2, 3])),
                        (torch.flip(vf, [2, 3, 4]), torch.flip(vc, [2, 3, 4])),
                    ]
                    lo_sum = torch.zeros(m_cnt, device=device, dtype=torch.float32)
                    for m_fold in models:
                        lo_m = torch.zeros(m_cnt, device=device, dtype=torch.float32)
                        for xvf, xvc in views:
                            lg = m_fold(xvf, xvc).float()
                            lo_m = lo_m + (lg[:, 1] - lg[:, 0])
                        lo_m = lo_m * 0.25
                        lo_sum = lo_sum + (scale_oof * lo_m + bias_oof)
                    probs = torch.sigmoid(lo_sum / float(len(models))).cpu().tolist()
                for oi, pv in zip(orig_idx, probs):
                    scan_probs[int(oi)] = float(pv)
            del img, val
            
            for rank, (r, p_val) in enumerate(zip(rr, scan_probs), 1):
                writer.writerow([r['series'], r['x'], r['y'], r['z'], p_val, r['id'], rank, r['label']])
                all_scores.append(p_val)
                all_labels.append(int(r['label']))
                all_series_coords.append((r['series'], r['x'], r['y'], r['z']))
                candidate_patient_map.append(r['patient'])
            f.flush()
            total_scored += len(rr)
            t_scan1 = time.time()
            if scan_idx % 10 == 0 or scan_idx == len(groups):
                print(f"Scored {scan_idx}/89 scans ({total_scored}/{total_cands} candidates) - Last scan: {t_scan1-t_scan0:.2f}s, Total: {t_scan1-t_start_scoring:.1f}s")

    t_scoring_total = time.time() - t_start_scoring
    print(f"=== SCORING COMPLETE in {t_scoring_total:.1f}s ({t_scoring_total/60:.2f} min) ===")

    # Candidate metrics
    scores_np = np.asarray(all_scores, dtype=np.float64)
    labels_np = np.asarray(all_labels, dtype=np.int64)

    def calc_metrics(scores, labels, cut):
        pred = scores >= cut
        y = labels == 1
        tp = int((pred & y).sum())
        tn = int((~pred & ~y).sum())
        fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum())
        prec = float(tp / (tp + fp)) if (tp + fp) else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) else 0.0
        acc = float((tp + tn) / len(labels))
        f1p = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
        f1n = float(2 * tn / (2 * tn + fp + fn)) if (2 * tn + fp + fn) else 0.0
        return {
            'total_candidates': len(labels),
            'positives': int(y.sum()),
            'negatives': int((~y).sum()),
            'threshold': float(cut),
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'accuracy': acc,
            'macro_f1': float(0.5 * (f1p + f1n)),
            'f1_positive': f1p,
            'f1_negative': f1n,
            'precision_positive': prec,
            'recall_positive': rec,
            'specificity_negative': spec,
        }

    metrics_cut = calc_metrics(scores_np, labels_np, nat_cut_oof)
    metrics_05 = calc_metrics(scores_np, labels_np, 0.5)
    print("Candidate metrics at nat_cut_oof:", metrics_cut)
    print("Candidate metrics at 0.5:", metrics_05)

    # Candidate metrics patient-level clustered bootstrap (1000 samples)
    print("Computing candidate-level patient-clustered bootstrap (1,000 samples)...")
    unique_patients = np.array(sorted(list(set(patient_ids))))
    cand_patients_arr = np.array(candidate_patient_map)
    patient_to_indices = {p: np.where(cand_patients_arr == p)[0] for p in unique_patients}
    rng = np.random.RandomState(42)
    boot_f1p_cut = []
    boot_rec_cut = []
    boot_prec_cut = []
    boot_f1p_05 = []
    boot_rec_05 = []
    boot_prec_05 = []
    for _ in range(1000):
        samp_pats = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        samp_indices = np.concatenate([patient_to_indices[p] for p in samp_pats])
        m_c = calc_metrics(scores_np[samp_indices], labels_np[samp_indices], nat_cut_oof)
        boot_f1p_cut.append(m_c['f1_positive'])
        boot_rec_cut.append(m_c['recall_positive'])
        boot_prec_cut.append(m_c['precision_positive'])
        m_5 = calc_metrics(scores_np[samp_indices], labels_np[samp_indices], 0.5)
        boot_f1p_05.append(m_5['f1_positive'])
        boot_rec_05.append(m_5['recall_positive'])
        boot_prec_05.append(m_5['precision_positive'])

    metrics_cut['f1_positive_95ci'] = [float(np.percentile(boot_f1p_cut, 2.5)), float(np.percentile(boot_f1p_cut, 97.5))]
    metrics_cut['recall_positive_95ci'] = [float(np.percentile(boot_rec_cut, 2.5)), float(np.percentile(boot_rec_cut, 97.5))]
    metrics_cut['precision_positive_95ci'] = [float(np.percentile(boot_prec_cut, 2.5)), float(np.percentile(boot_prec_cut, 97.5))]

    metrics_05['f1_positive_95ci'] = [float(np.percentile(boot_f1p_05, 2.5)), float(np.percentile(boot_f1p_05, 97.5))]
    metrics_05['recall_positive_95ci'] = [float(np.percentile(boot_rec_05, 2.5)), float(np.percentile(boot_rec_05, 97.5))]
    metrics_05['precision_positive_95ci'] = [float(np.percentile(boot_prec_05, 2.5)), float(np.percentile(boot_prec_05, 97.5))]

    # Run official LUNA16 evaluator
    print("Running official LUNA16 FROC evaluator...")
    froc_dir = OUT / 'froc_eval'
    froc_dir.mkdir(parents=True, exist_ok=True)
    sub_csv = froc_dir / 'submission_all.csv'
    with scores_csv.open(newline='') as fin, sub_csv.open('w', newline='') as fout:
        w_csv = csv.writer(fout)
        w_csv.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'])
        for row in csv.DictReader(fin):
            w_csv.writerow([row['seriesuid'], row['coordX'], row['coordY'], row['coordZ'], row['probability']])

    uids_csv = froc_dir / 'seriesuids.csv'
    with uids_csv.open('w') as f:
        for u in sorted(series_uids):
            f.write(u + '\n')

    z = zipfile.ZipFile(EVAL_ZIP)
    with tempfile.TemporaryDirectory() as temp:
        t = Path(temp)
        (t / 'tools').mkdir()
        fix = RefactoringTool(get_fixers_from_package('lib2to3.fixes'))
        for name in ['noduleCADEvaluationLUNA16.py', 'NoduleFinding.py', 'tools/csvTools.py', 'tools/__init__.py']:
            src = z.read('evaluationScript/' + name).decode()
            txt = str(fix.refactor_string(src, name)).replace('basex=', 'base=').replace('plt.grid(b=', 'plt.grid(visible=') if src.strip() else ''
            (t / name).write_text(txt)
        sys.path.insert(0, temp)
        import matplotlib
        matplotlib.use('Agg')
        import noduleCADEvaluationLUNA16 as ev
        ev.csvTools.readCSV = lambda f: list(csv.reader(open(f, newline='')))
        nodules, uids = ev.collect(str(REFD / 'annotations.csv'), str(REFD / 'annotations_excluded.csv'), str(uids_csv))
        cad_out = froc_dir / 'cad'
        cad_out.mkdir(parents=True, exist_ok=True)
        
        # Run main evaluation with bootstrapping enabled (1000 samples)
        ev.evaluateCAD(uids, str(sub_csv), str(cad_out) + '/', nodules, 'clear3d_msf_test0', maxNumberOfCADMarks=-1, performBootstrapping=True, numberOfBootstrapSamples=1000)
        
        curves = np.loadtxt(cad_out / 'froc_clear3d_msf_test0.txt', delimiter=',')
        grid = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        order = np.argsort(curves[:, 0], kind='mergesort')
        sens = np.interp(grid, curves[order, 0], curves[order, 1]).tolist()
        cpm = float(np.mean(sens))

    # Read the bootstrap FROC curve
    boot_csv = cad_out / 'froc_clear3d_msf_test0_bootstrapping.csv'
    boot_data = np.loadtxt(boot_csv, delimiter=',', skiprows=1)
    # boot_data has columns: FPrate, Sensivity[Mean], Sensivity[Lower bound], Sensivity[Upper bound]
    fp_rate = boot_data[:, 0]
    sens_mean_curve = boot_data[:, 1]
    sens_lb_curve = boot_data[:, 2]
    sens_ub_curve = boot_data[:, 3]

    sens_mean_grid = np.interp(grid, fp_rate, sens_mean_curve).tolist()
    sens_lb_grid = np.interp(grid, fp_rate, sens_lb_curve).tolist()
    sens_ub_grid = np.interp(grid, fp_rate, sens_ub_curve).tolist()
    cpm_mean = float(np.mean(sens_mean_grid))
    cpm_lb = float(np.mean(sens_lb_grid))
    cpm_ub = float(np.mean(sens_ub_grid))

    froc_results = {
        'fp_per_scan_grid': grid,
        'sensitivities': [float(x) for x in sens],
        'cpm': cpm,
        'series_count': len(uids),
        'bootstrap_1000_mean_sensitivities': [float(x) for x in sens_mean_grid],
        'bootstrap_1000_lower_bound_sensitivities': [float(x) for x in sens_lb_grid],
        'bootstrap_1000_upper_bound_sensitivities': [float(x) for x in sens_ub_grid],
        'cpm_bootstrap_mean': cpm_mean,
        'cpm_bootstrap_95ci': [cpm_lb, cpm_ub]
    }
    print("FROC Results:", froc_results)
    print(f"CPM: {cpm:.6f} (95% CI: [{cpm_lb:.6f}, {cpm_ub:.6f}])")

    final_report = {
        'status': 'PASS',
        'task': 'LUNA16 False-Positive Reduction on supplied candidates_V2',
        'cohort': 'test-fold0 (89 scans, 89 patients)',
        'scope': 'Internal LUNA16 subset 0 evaluation; prior-exposure uncertainty acknowledged; not external clinical validation',
        'authorization': 'Recorded as user report of supervisor confirmation; no written signature/authority fabricated',
        'other_prepared_val_authorization': 'UNSETTLED',
        'runtime_seconds': {
            'scoring_seconds': float(t_scoring_total),
            'total_seconds': float(time.time() - t_start_total)
        },
        'model': {
            'ensemble': 'Run 004 (5 checkpoints: fold-0.pt .. fold-4.pt)',
            'scale_oof': scale_oof,
            'bias_oof': bias_oof,
            'nat_cut_oof': nat_cut_oof,
            'tta': '4-view (Identity, X, YZ, XYZ)',
            'aggregation': 'pre-sigmoid average across models'
        },
        'froc': froc_results,
        'candidate_metrics_nat_cut': metrics_cut,
        'candidate_metrics_05': metrics_05,
        'hashes': {
            'scores_csv_sha256': sha256_file(scores_csv),
            'test_manifest_sha256': sha256_file(TEST_MANIFEST),
            'candidates_v2_sha256': sha256_file(CAND_FILE),
            'annotations_sha256': sha256_file(REFD / 'annotations.csv'),
            'annotations_excluded_sha256': sha256_file(REFD / 'annotations_excluded.csv'),
        }
    }
    
    with open(OUT / 'final_test_report.json', 'w') as f:
        json.dump(final_report, f, indent=2)
    print("Report written to:", OUT / 'final_test_report.json')
    print("=== FINISHED ALL STEPS SUCCESSFULLY ===")

if __name__ == '__main__':
    main()
