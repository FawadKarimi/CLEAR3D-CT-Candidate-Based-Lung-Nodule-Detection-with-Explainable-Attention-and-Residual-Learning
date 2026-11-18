import json, zipfile, shutil, xml.etree.ElementTree as ET
from pathlib import Path

V2 = Path('/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/02_Active_Model_and_Pipeline')
OUT = V2 / 'journal_package_20261117'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'

res = {}

# 1. Check JOIV styles & Table 1 formatting & word counts
with zipfile.ZipFile(V2 / 'JOIV_Template_2025.docx', 'r') as z:
    doc = ET.fromstring(z.read('word/document.xml'))
    styles = ET.fromstring(z.read('word/styles.xml'))
    tbl = doc.find(f'.//{W}tbl')
    trows = []
    if tbl is not None:
        for tr in tbl.findall(f'{W}tr'):
            trows.append([''.join(t.text or '' for t in tc.iter(f'{W}t')).strip() for tc in tr.findall(f'{W}tc')])
    res['joiv_table1'] = trows
    # Extract full text paragraphs of JOIV
    body = doc.find(f'{W}body')
    res['joiv_full_paras'] = [
        {
            'idx': i,
            'jc': p.find(f'{W}pPr/{W}jc').attrib.get(f'{W}val') if p.find(f'{W}pPr/{W}jc') is not None else None,
            'ind': p.find(f'{W}pPr/{W}ind').attrib if p.find(f'{W}pPr/{W}ind') is not None else None,
            'spacing': p.find(f'{W}pPr/{W}spacing').attrib if p.find(f'{W}pPr/{W}spacing') is not None else None,
            'text': ''.join(t.text or '' for t in p.iter(f'{W}t'))
        }
        for i, p in enumerate(body) if p.tag == f'{W}p'
    ]

# 2. Extract ORIG docx text to verify all claims in previous draft
with zipfile.ZipFile(V2 / 'CLEAR3D_CT_IEEE_Expert_Review_DRAFT.ORIG.docx', 'r') as z:
    doc = ET.fromstring(z.read('word/document.xml'))
    body = doc.find(f'{W}body')
    res['orig_docx_paras'] = [''.join(t.text or '' for t in p.iter(f'{W}t')) for p in body if p.tag == f'{W}p']

# 3. Inspect evaluate_and_audit_all_three_models_20261109.py & clear3d_definitive_evidence_package_20261109.zip summary files
for fn in ['evaluate_and_audit_all_three_models_20261109.py', 'clear3d-msf-task2-task3-evaluation-20261106-004.json']:
    p = V2 / fn
    if p.exists():
        res[fn] = p.read_text(errors='ignore')

zpath = V2 / 'clear3d_definitive_evidence_package_20261109.zip'
if zpath.exists():
    with zipfile.ZipFile(zpath, 'r') as z:
        res['definitive_zip_files'] = z.namelist()
        for nm in ['internal_test_report_fold0.json', 'test_cohort_cap4000_corrected_evaluation_summary.json', 'published_benchmarks_comparison.json']:
            if nm in z.namelist():
                res['zip:' + nm] = json.loads(z.read(nm).decode('utf-8'))

# 4. Inspect crossval-10fold-clear3d-20261109 CADAnalysis & summary files + ablation runs
for ap in sorted((V2 / 'runs').glob('clear3d-msf-ablation-*')):
    for sf in ap.rglob('*'):
        if sf.is_file() and sf.name in ('summary.json', 'CADAnalysis.txt', 'results_summary.json', 'paired_cpm_bootstrap.json'):
            res[str(sf.relative_to(V2))] = sf.read_text(errors='ignore')[:4000]

cv_cad = V2 / 'runs/crossval-10fold-clear3d-20261109/official_cad_eval/CADAnalysis.txt'
if cv_cad.exists():
    res['cv797_CADAnalysis'] = cv_cad.read_text()

# 5. Check tools availability
import importlib.util
res['tools'] = {
    'soffice': shutil.which('soffice') or shutil.which('libreoffice'),
    'pdftoppm': shutil.which('pdftoppm'),
    'python_docx': importlib.util.find_spec('docx') is not None,
    'matplotlib': importlib.util.find_spec('matplotlib') is not None,
}
# Check env-310-fawad packages
import sys
res['python_exe'] = sys.executable

(OUT / 'audit_remaining.json').write_text(json.dumps(res, indent=2))
print('REMAINING_OK')
