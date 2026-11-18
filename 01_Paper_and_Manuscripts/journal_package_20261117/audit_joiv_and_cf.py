import json, zipfile, xml.etree.ElementTree as ET
from pathlib import Path

V2 = Path('/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/02_Active_Model_and_Pipeline')
BASE = Path('/experiment-storage/chaidir/fawad-dsai')
OUT = V2 / 'journal_package_20261117'

# A. Full JOIV_Template_2025.docx XML audit
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
joiv_path = V2 / 'JOIV_Template_2025.docx'
joiv_audit = {}
with zipfile.ZipFile(joiv_path, 'r') as z:
    joiv_audit['zip_files'] = z.namelist()
    doc_xml = ET.fromstring(z.read('word/document.xml'))
    styles_xml = ET.fromstring(z.read('word/styles.xml'))
    body = doc_xml.find(f'{W}body')
    paras = []
    for idx, ch in enumerate(body):
        tag = ch.tag.split('}')[-1]
        if tag == 'p':
            texts = ''.join(t.text or '' for t in ch.iter(f'{W}t'))
            pStyle = ch.find(f'{W}pPr/{W}pStyle')
            jc = ch.find(f'{W}pPr/{W}jc')
            sectPr = ch.find(f'{W}pPr/{W}sectPr')
            r_fonts = []
            for r in ch.findall(f'{W}r')[:4]:
                sz = r.find(f'{W}rPr/{W}sz')
                rFonts = r.find(f'{W}rPr/{W}rFonts')
                b = r.find(f'{W}rPr/{W}b')
                it = r.find(f'{W}rPr/{W}i')
                sc = r.find(f'{W}rPr/{W}smallCaps')
                r_fonts.append({
                    'sz': sz.attrib.get(f'{W}val') if sz is not None else None,
                    'font': rFonts.attrib.get(f'{W}ascii') if rFonts is not None else None,
                    'b': b is not None, 'i': it is not None, 'smallCaps': sc is not None,
                    't': ''.join(t.text or '' for t in r.findall(f'{W}t'))[:40]
                })
            has_img = len(list(ch.iter('{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}inline'))) + len(list(ch.iter('{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}anchor')))
            paras.append({
                'idx': idx, 'tag': 'p',
                'style': pStyle.attrib.get(f'{W}val') if pStyle is not None else None,
                'jc': jc.attrib.get(f'{W}val') if jc is not None else None,
                'has_sectPr': sectPr is not None,
                'has_img': has_img,
                'runs': r_fonts,
                'text': texts[:250]
            })
        elif tag == 'tbl':
            rows = ch.findall(f'{W}tr')
            first_row = [''.join(t.text or '' for t in tc.iter(f'{W}t'))[:40] for tc in rows[0].findall(f'{W}tc')] if rows else []
            paras.append({'idx': idx, 'tag': 'tbl', 'nrows': len(rows), 'first_row': first_row})
        elif tag == 'sectPr':
            pgSz = ch.find(f'{W}pgSz')
            pgMar = ch.find(f'{W}pgMar')
            cols = ch.find(f'{W}cols')
            hdrs = [h.attrib for h in ch.findall(f'{W}headerReference')]
            ftrs = [f.attrib for f in ch.findall(f'{W}footerReference')]
            paras.append({
                'idx': idx, 'tag': 'sectPr',
                'pgSz': pgSz.attrib if pgSz is not None else None,
                'pgMar': pgMar.attrib if pgMar is not None else None,
                'cols': cols.attrib if cols is not None else None,
                'hdrs': hdrs, 'ftrs': ftrs
            })
    joiv_audit['body_elements'] = paras
    style_names = []
    for st in styles_xml.findall(f'{W}style'):
        nm = st.find(f'{W}name')
        style_names.append((st.attrib.get(f'{W}styleId'), st.attrib.get(f'{W}type'), nm.attrib.get(f'{W}val') if nm is not None else None))
    joiv_audit['styles'] = style_names
    for hf in [n for n in z.namelist() if 'header' in n or 'footer' in n]:
        hf_xml = ET.fromstring(z.read(hf))
        joiv_audit[hf] = ''.join(t.text or '' for t in hf_xml.iter(f'{W}t'))

(OUT / 'joiv_template_audit.json').write_text(json.dumps(joiv_audit, indent=2))

# B. Phase-CF (2026-10-25) & dev optimization history & Phase 708 CV & primary ref
extra = {}
cf_dir = BASE / 'CLEAR3D-comparison-protocol-20261025'
extra['cf_all_files'] = [str(p.relative_to(cf_dir)) for p in cf_dir.rglob('*') if p.is_file() and p.stat().st_size < 2000000 and '.ipynb_checkpoints' not in str(p)]
for fname in ['primary_reference_system_binding.json', 'clear3d_msf_model_20261106_002.py', 'runs/cv_improved_10fold_dev708_20261114/prespecified_protocol_dev708.json', 'runs/cv_improved_10fold_dev708_20261114/run_10fold_dev708.py']:
    fp = V2 / fname
    if fp.exists():
        extra[fname] = fp.read_text(errors='ignore')

# Check isolated/ or phase-cf/ scripts in CLEAR3D-comparison-protocol-20261025
for p in sorted(cf_dir.rglob('*.py')):
    if '.ipynb_checkpoints' not in str(p):
        txt = p.read_text(errors='ignore')
        extra['py:' + str(p.relative_to(cf_dir))] = txt[:8000]

(OUT / 'extra_audit_details.json').write_text(json.dumps(extra, indent=2))
print('EXTRA_AUDIT_OK')
