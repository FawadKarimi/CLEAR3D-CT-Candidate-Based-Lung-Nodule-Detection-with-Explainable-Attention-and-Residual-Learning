#!/usr/bin/env python3
"""
Regenerate LUNA16 FROC Curves & CPM Scores from Prediction CSVs.
Pure Python 3 implementation adapted from official LUNA16 CAD evaluation script.
"""
from __future__ import annotations
import argparse, csv, os, sys, time
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import noduleCADEvaluationLUNA16 as ev
ev.csvTools.readCSV = lambda f: list(csv.reader(open(f, newline='')))

GRID = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

EXPECTED = {
    'fold0_baseline': {
        'cpm': 0.904018,
        'sensitivities': [0.781250, 0.864583, 0.890625, 0.911458, 0.953125, 0.963542, 0.963542]
    },
    'control_b_nomining': {
        'cpm': 0.875000,
        'sensitivities': [0.677083, 0.796875, 0.869792, 0.911458, 0.932292, 0.968750, 0.968750]
    },
    'control_a_fineonly': {
        'cpm': 0.865327,
        'sensitivities': [0.671875, 0.776042, 0.864583, 0.890625, 0.932292, 0.947917, 0.973958]
    }
}


def run_evaluation(pred_csv: Path, annots_csv: Path, annots_exc_csv: Path, series_uids_csv: Path, out_dir: Path, model_tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    cad_out = out_dir / model_tag
    cad_out.mkdir(parents=True, exist_ok=True)
    
    sub_csv = cad_out / f'submission_{model_tag}.csv'
    with pred_csv.open(newline='') as fin, sub_csv.open('w', newline='') as fout:
        w = csv.writer(fout)
        w.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'])
        for row in csv.DictReader(fin):
            w.writerow([row['seriesuid'], row['coordX'], row['coordY'], row['coordZ'], row['probability']])
            
    nodules, uids = ev.collect(str(annots_csv), str(annots_exc_csv), str(series_uids_csv))
    
    t0 = time.time()
    ev.evaluateCAD(
        uids,
        str(sub_csv),
        str(cad_out) + '/',
        nodules,
        model_tag,
        maxNumberOfCADMarks=-1,
        performBootstrapping=False
    )
    elapsed = time.time() - t0
    
    froc_file = cad_out / f'froc_{model_tag}.txt'
    curves = np.loadtxt(froc_file, delimiter=',')
    order = np.argsort(curves[:, 0], kind='mergesort')
    sens = np.interp(GRID, curves[order, 0], curves[order, 1]).tolist()
    cpm = float(np.mean(sens))
    
    return {
        'model_tag': model_tag,
        'cpm': cpm,
        'sensitivities': sens,
        'elapsed_sec': elapsed,
        'froc_file': str(froc_file)
    }


def main():
    parser = argparse.ArgumentParser(description='Regenerate FROC curves and CPM scores.')
    parser.add_argument('--pkg-root', type=Path, default=SCRIPT_DIR.parent)
    parser.add_argument('--out-dir', type=Path, default=None)
    args = parser.parse_args()
    
    pkg_root = args.pkg_root
    preds_dir = pkg_root / 'predictions'
    ref_dir = pkg_root / 'reference'
    out_dir = args.out_dir or (pkg_root / 'reproduced_froc')
    
    annots_csv = ref_dir / 'annotations.csv'
    annots_exc_csv = ref_dir / 'annotations_excluded.csv'
    series_uids_csv = ref_dir / 'seriesuids.csv'
    
    models = [
        ('fold0_baseline', preds_dir / 'fold0_baseline_scores.csv'),
        ('control_b_nomining', preds_dir / 'control_b_nomining_scores.csv'),
        ('control_a_fineonly', preds_dir / 'control_a_fineonly_scores.csv'),
    ]
    
    print('=' * 80)
    print('CLEAR3D-CT FROC REGENERATION AUDIT')
    print('Reference Annotations: ' + str(annots_csv))
    print('Excluded Annotations:  ' + str(annots_exc_csv))
    print('Series UIDs:           ' + str(series_uids_csv))
    print('Output Directory:      ' + str(out_dir))
    print('=' * 80)
    
    all_passed = True
    for tag, pred_path in models:
        print('')
        print('Evaluating ' + tag + ' from ' + pred_path.name + '...')
        res = run_evaluation(pred_path, annots_csv, annots_exc_csv, series_uids_csv, out_dir, tag)
        exp = EXPECTED[tag]
        
        cpm_diff = abs(res['cpm'] - exp['cpm'])
        passed = (cpm_diff < 1e-5)
        if not passed:
            all_passed = False
            
        status_str = 'MATCH [PASS]' if passed else 'MISMATCH [FAIL]'
        print('  Status:         ' + status_str)
        rcpm = res["cpm"]
        ecpm = exp["cpm"]
        print(f"  Regenerated CPM: {rcpm:.6f} (Expected: {ecpm:.6f}, diff: {cpm_diff:.6e})")
        rsens = [round(s, 6) for s in res["sensitivities"]]
        print(f"  Sensitivities:   {rsens}")
        relap = res["elapsed_sec"]
        print(f"  Elapsed Time:    {relap:.2f} s")
        
    print('')
    print('=' * 80)
    if all_passed:
        print('ALL 3 FROC CURVES AND CPM METRICS EXACTLY REPRODUCED!')
    else:
        print('ERROR: Metric discrepancies detected!')
    print('=' * 80)
    return 0 if all_passed else 1

if __name__ == '__main__':
    raise SystemExit(main())
