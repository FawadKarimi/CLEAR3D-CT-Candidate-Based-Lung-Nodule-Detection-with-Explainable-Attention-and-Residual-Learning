"""Tasks 2 & 3 Official Development Scoring & LUNA16 FROC Evaluation for CLEAR3D-MSFv2 (Run 004):
- Monitored single-process GPU worker via user_watchdog.
- Scores all 122,689 official candidates_V2 rows across all 142 dev scans (0 confirm/test access).
- Evaluates Task 2 (natural imbalanced population: 122,689 unfiltered & excluded-filtered) at 0.5 and fit-only nat_cut_oof.
- Evaluates Task 3 (official LUNA16 evaluateCAD scan-level FROC at [0.125, 0.25, 0.5, 1, 2, 4, 8] FP/scan & CPM).
"""
from __future__ import annotations
import argparse, csv, ctypes, hashlib, json, math, os, signal, sys, tempfile, time, zipfile
from collections import Counter, defaultdict
from lib2to3.refactor import RefactoringTool, get_fixers_from_package
from pathlib import Path

BASE = Path('/experiment-storage/chaidir/fawad-dsai')
ROOT = BASE / 'CLEAR3D_CT project/02_Active_Model_and_Pipeline'
WORK = BASE / 'CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64/source'
sys.path[:0] = [str(ROOT), str(WORK / 'src'), str(WORK / 'scripts')]

import numpy as np
import user_watchdog as w
import loader

G = 1024 ** 3
LIMITS = {
    'setup_seconds': 600,
    'run_seconds': 14400,
    'validation_seconds': 1800,
    'total_seconds': 15600,
    'rss_bytes': 8 * G,
    'allocator_bytes': 4 * G,
    'gpu_bytes': 6 * G,
    'sample_seconds': 2.0,
    'min_ram_bytes': 32 * G,
    'reserve_ram_bytes': 8 * G,
    'min_gpu_bytes': 10 * G,
    'min_disk_bytes': 20 * G,
}

SPLIT_PATH = BASE / 'CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-comparison-protocol-20261025/runs/phase-cf/split.json'
DEV_JSONL = BASE / 'CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-comparison-protocol-20261025/runs/phase-cf/dev.jsonl'
REFD = BASE / 'data/reference/luna16-zenodo-3723295/resolved-reference-v1'
EVAL_ZIP = BASE / 'data/reference/luna16-zenodo-3723295/evaluationScript.zip'
BIND = BASE / 'CLEAR3D_CT project/04_Earlier_Phases/CLEAR3D-parallel-review-20261014-2f1a64/D-long/current-fit-binding.json'
PREP = BASE / 'CLEAR3D_CT project/03_Phase1_LIDC3D/LIDC3D/artifacts/luna16-prepare-development-fold0-v1/prepared'
TRAIN_OUT = ROOT / 'runs/clear3d-msf-ensemble-20261106-004'
MODEL_FILE = ROOT / 'clear3d_msf_model_20261106_002.py'
MODEL_SHA = 'e83899960b095eab9f92fe104f13e3380c9c5bc48e6ebc82b173003a520914e7'
OUT = ROOT / 'runs/clear3d-msf-official-dev-20261106-004'


def require(ok: bool, msg: str):
    if not ok:
        raise ValueError(msg)


def roc_auc_np(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    vals = scores[order]
    i, n = 0, len(vals)
    while i < n:
        j = i + 1
        while j < n and vals[j] == vals[i]:
            j += 1
        if j > i + 1:
            ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return float((ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def pr_auc_np(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = int((labels == 1).sum())
    order = np.argsort(-scores, kind='mergesort')
    y = (labels[order] == 1).astype(np.float64)
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1, dtype=np.float64)
    return float((prec * y).sum() / n_pos)


def nat_cls_metrics(scores: np.ndarray, labels: np.ndarray, cut: float) -> dict:
    pred = scores >= cut
    y = labels == 1
    tp = int((pred & y).sum())
    tn = int((~pred & ~y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    f1p = (2.0 * tp / (2.0 * tp + fp + fn)) if tp > 0 else 0.0
    f1n = (2.0 * tn / (2.0 * tn + fn + fp)) if tn > 0 else 0.0
    return {
        'rows': int(len(labels)),
        'positives': int(y.sum()),
        'negatives': int((~y).sum()),
        'threshold': float(cut),
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'fp_per_scan_142': float(fp / 142.0),
        'accuracy': float((tp + tn) / len(labels)),
        'macro_f1': float(0.5 * (f1p + f1n)),
        'f1_positive': float(f1p),
        'f1_negative': float(f1n),
        'precision_positive': float(tp / (tp + fp)) if (tp + fp) else 0.0,
        'recall_positive': float(tp / (tp + fn)) if (tp + fn) else 0.0,
        'specificity_negative': float(tn / (tn + fp)) if (tn + fp) else 0.0,
    }


def run_luna16_froc(scores_csv: Path, series_uids: list[str], froc_dir: Path) -> dict:
    froc_dir.mkdir(parents=True, exist_ok=False)
    sub_csv = froc_dir / 'submission_all.csv'
    with scores_csv.open(newline='') as fin, sub_csv.open('x', newline='') as fout:
        w_csv = csv.writer(fout)
        w_csv.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'])
        for row in csv.DictReader(fin):
            w_csv.writerow([row['seriesuid'], row['coordX'], row['coordY'], row['coordZ'], row['probability']])

    uids_csv = froc_dir / 'seriesuids.csv'
    uids_csv.write_text(''.join(u + '\n' for u in sorted(series_uids)))

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
        cad_out.mkdir()
        ev.evaluateCAD(uids, str(sub_csv), str(cad_out) + '/', nodules, 'clear3d_msf_004', maxNumberOfCADMarks=-1, performBootstrapping=False)
        curves = np.loadtxt(cad_out / 'froc_clear3d_msf_004.txt', delimiter=',')
        grid = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        # Note: np.interp requires increasing xp; sort curves by FP/scan ascending
        order = np.argsort(curves[:, 0], kind='mergesort')
        sens = np.interp(grid, curves[order, 0], curves[order, 1]).tolist()
        return {
            'fp_per_scan_grid': grid,
            'sensitivities': [float(x) for x in sens],
            'cpm': float(np.mean(sens)),
            'series_count': len(uids),
        }


def work():
    import torch
    from lidc3d.provenance import seed_all
    from clear3d_msf_model_20261106_002 import CLEAR3DMSFv2, build_focus_grids, make_inputs
    from lidc3d.geometry import crop
    from train_clear3d_msf_ensemble_20261106_004 import gpu_extract_from_p51, load_scan_arrays

    t_start = time.time()
    require(loader.hash_file(MODEL_FILE) == MODEL_SHA, 'model file changed')
    b_bind = json.loads(BIND.read_text())
    w.bindings(b_bind['parent_binding'], b_bind['parent_binding_sha256'])
    split = loader.get_split(SPLIT_PATH)
    require(not (split['fit'] & split['dev']), 'fit/dev overlap')

    train_res = json.loads((TRAIN_OUT / 'worker-result.json').read_text())
    require(train_res['status'] == 'PASS' and len(train_res['checkpoint_sha256']) == 5, 'invalid train worker-result')

    seed_all(17)
    require(torch.cuda.device_count() == 1, 'GPU visibility must be 1')
    device = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(
        LIMITS['allocator_bytes'] / torch.cuda.get_device_properties(0).total_memory
    )
    f_fine, f_ctx = build_focus_grids(device)

    models = []
    scale_oof = None
    bias_oof = None
    nat_cut_oof = None
    for k, expected_sha in enumerate(train_res['checkpoint_sha256']):
        cp = TRAIN_OUT / 'checkpoints' / f'fold-{k}.pt'
        require(loader.hash_file(cp) == expected_sha, f'fold-{k} checkpoint sha mismatch')
        ck = torch.load(cp, map_location='cpu', weights_only=False)
        m = CLEAR3DMSFv2()
        m.load_state_dict(ck['model'], strict=True)
        m.to(device).eval()
        models.append(m)
        scale_oof = float(ck['scale_oof'])
        bias_oof = float(ck['bias_oof'])
        nat_cut_oof = float(ck['nat_cut_oof'])

    w.save(OUT / 'phase.json', {'name': 'training'})
    rows = loader.candidate_rows(DEV_JSONL, 'dev', split['dev'])
    require(len(rows) == 122689, f'unexpected dev candidate count: {len(rows)}')
    groups = defaultdict(list)
    for r in rows:
        groups[r['scan']].append(r)
    require(len(groups) == 142, f'unexpected dev scan count: {len(groups)}')

    all_scores = []
    all_labels = []
    all_series_coords = []
    series_uids = []
    total_scored = 0

    scores_csv = OUT / 'scores.csv'
    with scores_csv.open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability', 'candidate_id', 'rank', 'label'])
        for scan_idx, (sid, rr) in enumerate(sorted(groups.items()), 1):
            r0 = rr[0]
            series_uids.append(r0['series'])
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
            require(len(scan_probs) == len(rr), 'scan prob count mismatch')
            for rank, (r, p_val) in enumerate(zip(rr, scan_probs), 1):
                require(math.isfinite(p_val) and 0.0 <= p_val <= 1.0, 'invalid prob')
                writer.writerow([r['series'], r['x'], r['y'], r['z'], p_val, r['id'], rank, r['label']])
                all_scores.append(p_val)
                all_labels.append(int(r['label']))
                all_series_coords.append((r['series'], r['x'], r['y'], r['z']))
            f.flush()
            total_scored += len(rr)
            if scan_idx % 20 == 0 or scan_idx == 142:
                print(f'SCORED_DEV_SCANS {scan_idx}/142 rows={total_scored} elapsed={time.time()-t_start:.1f}s', flush=True)

    require(total_scored == 122689, 'total scored mismatch')
    w.save(OUT / 'phase.json', {'name': 'validation'})

    scores_arr = np.array(all_scores, dtype=np.float64)
    labels_arr = np.array(all_labels, dtype=np.int64)

    # Excluded-filtered mask (matching selection.py official excluded radius)
    exc_refs = defaultdict(list)
    with (REFD / 'annotations_excluded.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            d = float(r['diameter_mm'])
            exc_refs[r['seriesuid']].append((
                [float(r['coordX']), float(r['coordY']), float(r['coordZ'])],
                5.0 if d < 0 else d / 2.0,
            ))
    keep_mask = np.ones(len(all_scores), dtype=bool)
    for i, (suid, cx, cy, cz) in enumerate(all_series_coords):
        if labels_arr[i] == 0:
            regs = exc_refs.get(suid, ())
            if regs and any(math.dist((cx, cy, cz), rc) <= rr_rad for rc, rr_rad in regs):
                keep_mask[i] = False

    froc_res = run_luna16_froc(scores_csv, series_uids, OUT / 'froc_eval')

    torch.cuda.synchronize()
    peak_gpu = int(torch.cuda.max_memory_allocated())
    require(peak_gpu <= LIMITS['allocator_bytes'], 'GPU allocator cap exceeded')

    summary = {
        'status': 'PASS',
        'scope': 'exploratory official dev candidates (Task 2 natural prevalence & Task 3 LUNA16 FROC); prior dev exposure; zero dev tuning; not untouched test',
        'reference_release': 'HOLD',
        'cap4000': 'NOT_ADOPTED',
        'scans': 142,
        'rows_unfiltered': int(len(scores_arr)),
        'rows_excluded_filtered': int(keep_mask.sum()),
        'seconds': float(time.time() - t_start),
        'gpu_peak_bytes': peak_gpu,
        'scores_csv_sha256': loader.hash_file(scores_csv),
        'checkpoint_sha256': train_res['checkpoint_sha256'],
        'scale_oof': scale_oof,
        'bias_oof': bias_oof,
        'nat_cut_oof': nat_cut_oof,
        'task2_natural_unfiltered_122689': {
            'roc_auc': roc_auc_np(scores_arr, labels_arr),
            'pr_auc': pr_auc_np(scores_arr, labels_arr),
            'at_0_5': nat_cls_metrics(scores_arr, labels_arr, 0.5),
            'at_fit_oof_nat_threshold': nat_cls_metrics(scores_arr, labels_arr, nat_cut_oof),
        },
        'task2_natural_excluded_filtered': {
            'roc_auc': roc_auc_np(scores_arr[keep_mask], labels_arr[keep_mask]),
            'pr_auc': pr_auc_np(scores_arr[keep_mask], labels_arr[keep_mask]),
            'at_0_5': nat_cls_metrics(scores_arr[keep_mask], labels_arr[keep_mask], 0.5),
            'at_fit_oof_nat_threshold': nat_cls_metrics(scores_arr[keep_mask], labels_arr[keep_mask], nat_cut_oof),
        },
        'task3_luna16_froc_142_dev_series': froc_res,
    }
    w.save(OUT / 'worker-result.json', summary)
    w.save(OUT / 'phase.json', {'name': 'complete'})
    print('OFFICIAL_DEV_SUMMARY', json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--worker', action='store_true')
    p.add_argument('--gate-fd', type=int)
    p.add_argument('--parent-pid', type=int)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()

    if a.worker:
        require(ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) == 0 and os.getppid() == a.parent_pid, 'parent gate')
        require(os.read(a.gate_fd, 1) == b'1', 'gate')
        os.close(a.gate_fd)
        try:
            work()
        except BaseException as e:
            w.save(OUT / 'worker-failure.json', {'error': repr(e)})
            raise
        return 0

    if not a.execute:
        print('PLAN ONLY', LIMITS)
        return 0

    b_bind = json.loads(BIND.read_text())
    w.bindings(b_bind['parent_binding'], b_bind['parent_binding_sha256'])
    require(not OUT.exists() and OUT.parent.is_dir(), 'output exists or parent absent')
    require(w.available_ram() >= LIMITS['min_ram_bytes'] and w.disk_free(OUT.parent) >= LIMITS['min_disk_bytes'], 'insufficient host RAM/disk')
    gpu = w.reviewed.gpu_snapshot()
    require(gpu['free_bytes'] >= LIMITS['min_gpu_bytes'], f"GPU free memory below 10 GiB: {gpu['free_bytes']}")

    OUT.mkdir()
    w.save(OUT / 'phase.json', {'name': 'setup'})
    w.save(OUT / 'gpu-before.json', gpu)
    w.save(OUT / 'invocation.json', {
        'script_sha256': loader.hash_file(__file__),
        'model_sha256': MODEL_SHA,
        'limits': LIMITS,
        'reference_release': 'HOLD',
        'cap4000': 'NOT_ADOPTED',
        'scope': 'official dev 122,689 scoring & LUNA16 FROC; zero dev tuning',
    })

    def validate(path, device, limits):
        r = json.loads((path / 'worker-result.json').read_text())
        require(r['status'] == 'PASS' and r['rows_unfiltered'] == 122689, 'worker evidence invalid')
        return r

    stop = [False]
    handlers = {s: signal.signal(s, lambda *args: stop.__setitem__(0, True)) for s in (signal.SIGINT, signal.SIGTERM)}
    child = None
    try:
        with (OUT / 'worker.log').open('xb', buffering=0) as log:
            child = w.OwnedWorker([sys.executable, '-B', str(Path(__file__).resolve()), '--worker'], w.child_env('cuda'), log)
            w.save(OUT / 'owned-worker.json', child.identity)
            result = w.supervise(child, OUT, 'cuda', LIMITS, observe=w.reviewed.gpu_snapshot, stop=lambda: stop[0], validate_evidence=validate, completion_reason='OFFICIAL_DEV_EVAL_COMPLETE')
        return 0 if result['status'] == 'PASS' else 1
    finally:
        if child is not None and child.proc.poll() is None:
            child.cleanup()
        for s, h in handlers.items():
            signal.signal(s, h)


if __name__ == '__main__':
    raise SystemExit(main())
