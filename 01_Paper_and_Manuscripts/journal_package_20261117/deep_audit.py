import os, json, re, hashlib, zipfile
from pathlib import Path

BASE = Path('/experiment-storage/chaidir/fawad-dsai')
V2 = BASE / 'CLEAR3D_CT project/02_Active_Model_and_Pipeline'
OUT = V2 / 'journal_package_20261117'

report = {}

# 1. Inspect dirs like CLEAR3D-comparison-protocol-20261025 and CLEAR3D-MedicalNet*
dirs_of_interest = sorted([str(p) for p in BASE.glob('CLEAR3D-*') if p.is_dir()])
report['clear3d_top_dirs'] = dirs_of_interest

comp_files = []
for dname in ['CLEAR3D-comparison-protocol-20261025', 'CLEAR3D-parallel-review-20261014-2f1a64', 'CLEAR3D-matched-execution-20261013-4e9a7d', 'CLEAR3D-MedicalNet-review-20261013-6b802e']:
    dp = BASE / dname
    if dp.exists():
        for p in dp.rglob('*'):
            if p.is_file() and p.suffix in ('.json', '.md', '.txt', '.py', '.csv', '.log') and p.stat().st_size < 2_000_000:
                comp_files.append(str(p))
report['comp_files_count'] = len(comp_files)
report['comp_files_sample'] = [f for f in comp_files if any(k in f.lower() for k in ['result', 'summary', 'metric', 'report', 'zuo', 'compare', 'audit', 'ledger', 'manifest', 'close', 'eval', 'model'])]

# 2. Grep for provisional source-of-truth numbers across python/json/md/txt/log/docx
needles = ['0.4878', '0.1697', '0.3967', '0.1606', '0.9751', '0.8143', '52.9', '64 of 121', '0.9129', '0.8006', '0.0975', '0.9043', '0.6768', '0.4279', 'two_branch', 'two-branch', 'TwoBranch', 'DualBranch']
hits = {n: [] for n in needles}

scan_roots = [BASE / d for d in os.listdir(BASE) if 'CLEAR3D' in d and (BASE / d).is_dir()]
for root in scan_roots:
    for p in root.rglob('*'):
        if p.is_file() and p.suffix in ('.json', '.md', '.txt', '.py', '.log', '.sh', '.yaml', '.yml') and p.stat().st_size < 5_000_000:
            try:
                txt = p.read_text(errors='ignore')
            except Exception:
                continue
            for n in needles:
                if n in txt:
                    # capture first matching line
                    line = next((ln.strip()[:200] for ln in txt.splitlines() if n in ln), '')
                    hits[n].append({'file': str(p), 'line': line})

# Also check docx files in V2
docx_info = {}
for dp in sorted(V2.glob('*.docx')):
    info = {'size': dp.stat().st_size, 'sha256': hashlib.sha256(dp.read_bytes()).hexdigest()}
    try:
        import docx
        doc = docx.Document(str(dp))
        full_txt = '\n'.join([par.text for par in doc.paragraphs if par.text.strip()])
        for tbl in doc.tables:
            for row in tbl.rows:
                full_txt += '\n' + ' | '.join([c.text.strip() for c in row.cells])
        info['paragraphs'] = len(doc.paragraphs)
        info['tables'] = len(doc.tables)
        info['sections'] = len(doc.sections)
        sec = doc.sections[0]
        info['page_mm'] = [round(sec.page_width.mm, 2), round(sec.page_height.mm, 2)]
        info['margins_mm'] = {
            'top': round(sec.top_margin.mm, 2),
            'bottom': round(sec.bottom_margin.mm, 2),
            'left': round(sec.left_margin.mm, 2),
            'right': round(sec.right_margin.mm, 2),
            'header': round(sec.header_distance.mm, 2),
            'footer': round(sec.footer_distance.mm, 2),
        }
        info['cols'] = sec._sectPr.xpath('.//w:cols/@w:num')
        info['col_space'] = sec._sectPr.xpath('.//w:cols/@w:space')
        info['needle_hits'] = [n for n in needles if n in full_txt]
        if 'JOIV' in dp.name:
            info['sample_paras'] = [(par.style.name if par.style else None, par.alignment, par.text[:140]) for par in doc.paragraphs[:60] if par.text.strip()]
            info['headers_footers'] = {
                'header_paras': [p.text for p in sec.header.paragraphs if p.text.strip()],
                'footer_paras': [p.text for p in sec.footer.paragraphs if p.text.strip()],
            }
            (OUT / 'joiv_template_extracted.txt').write_text(full_txt)
        else:
            (OUT / (dp.stem + '_extracted.txt')).write_text(full_txt)
    except Exception as e:
        info['error'] = str(e)
    docx_info[dp.name] = info

report['needle_hits'] = {k: v[:15] for k, v in hits.items()}
report['needle_counts'] = {k: len(v) for k, v in hits.items()}
report['docx_info'] = docx_info

# 3. Inspect all runs in V2/runs
runs_summary = {}
for rd in sorted((V2 / 'runs').iterdir()):
    if rd.is_dir():
        files = [f.name for f in rd.iterdir()]
        summary_file = rd / 'results_summary.json'
        status_file = rd / 'status.json'
        entry = {'files_count': len(files), 'sample_files': sorted(files)[:12]}
        if summary_file.exists():
            try:
                entry['results_summary'] = json.loads(summary_file.read_text())
            except Exception as e:
                entry['results_summary_err'] = str(e)
        if status_file.exists():
            try:
                entry['status'] = json.loads(status_file.read_text())
            except Exception as e:
                entry['status_err'] = str(e)
        runs_summary[rd.name] = entry
report['v2_runs'] = runs_summary

# 4. Inspect model code structure of clear3d_msf_model_20261106_002.py and m3.py and medicalnet comparator
code_summaries = {}
for cp in [
    V2 / 'clear3d_msf_model_20261106_002.py',
    V2 / 'train_medicalnet_resnet18_comparator.py',
    V2 / 'run_medicalnet_full_dev_corrected.py',
    V2 / 'train_clear3d_intervention_fold0_20261108.py',
    BASE / 'CLEAR3D-parallel-review-20261014-2f1a64/source/m3.py',
    BASE / 'CLEAR3D-parallel-review-20261014-2f1a64/accepted/source/src/lidc3d/models.py',
    V2 / 'nodule_coverage_audit.py',
]:
    if cp.exists():
        txt = cp.read_text(errors='ignore')
        code_summaries[str(cp)] = {
            'lines': len(txt.splitlines()),
            'classes_defs': [ln.strip() for ln in txt.splitlines() if ln.strip().startswith(('class ', 'def '))],
            'imports_medicalnet_or_transformer': [ln.strip() for ln in txt.splitlines() if any(w in ln.lower() for w in ['medicalnet', 'transformer', 'multihead', 'attention', 'resnet', 'clear3d', 'stem', 'branch', 'backbone', 'pretrained'])][:40]
        }
report['code_summaries'] = code_summaries

# 5. Inspect key JSON audit/evidence files in V2
key_jsons = {}
for jp in [
    V2 / 'reconcile_audit_and_exposure_20261107.json',
    V2 / 'published_benchmarks_comparison.json',
    V2 / 'candidate_generation_comparison_dev.json',
    V2 / 'patient_to_fold_manifest_20261109.json',
]:
    if jp.exists():
        try:
            d = json.loads(jp.read_text())
            if jp.name == 'patient_to_fold_manifest_20261109.json':
                from collections import Counter
                folds = Counter()
                if isinstance(d, dict) and 'patients' in d:
                    for k, v in d['patients'].items():
                        folds[str(v)] += 1
                    key_jsons[jp.name] = {'top_keys': list(d.keys()), 'fold_counts': dict(folds)}
                elif isinstance(d, dict):
                    vals = Counter(str(v) for v in d.values())
                    key_jsons[jp.name] = {'len': len(d), 'val_counts': dict(vals), 'sample': list(d.items())[:5]}
            else:
                key_jsons[jp.name] = d
        except Exception as e:
            key_jsons[jp.name] = str(e)
report['key_jsons'] = key_jsons

(OUT / 'deep_audit_report.json').write_text(json.dumps(report, indent=2, default=str))
print('AUDIT_WRITTEN', (OUT / 'deep_audit_report.json').stat().st_size)
