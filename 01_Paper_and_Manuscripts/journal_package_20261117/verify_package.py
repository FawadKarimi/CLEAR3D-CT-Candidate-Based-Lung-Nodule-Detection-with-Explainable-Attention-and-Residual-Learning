# -*- coding: utf-8 -*-
"""Automated forensic verification across CLEAR3D-CT JIIM (Double-Blind), Title Page, and JOIV packages."""
import json
import re
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import io

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manuscript_data as MD

ROOT = Path("/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/01_Paper_and_Manuscripts")
PKG = ROOT / "journal_package_20261117"
JIIM_DOCX = PKG / "CLEAR3D_CT_JIIM_DoubleBlind_Manuscript.docx"
TP_DOCX = PKG / "JIIM_Title_Page_and_Declarations.docx"
JOIV_DOCX = PKG / "CLEAR3D_CT_JOIV_Manuscript.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

def check_docx_common(path: Path, expected_imgs: int):
    assert path.exists() and path.stat().st_size > 10000, f"Missing or tiny docx: {path}"
    with zipfile.ZipFile(path, "r") as z:
        names = z.namelist()
        for req in ("word/document.xml", "word/_rels/document.xml.rels", "[Content_Types].xml"):
            assert req in names, f"{path.name}: missing {req}"
        # Parse all XML files cleanly
        for fn in names:
            if fn.endswith(".xml") or fn.endswith(".rels"):
                ET.fromstring(z.read(fn))
        doc_root = ET.fromstring(z.read("word/document.xml"))
        rels_root = ET.fromstring(z.read("word/_rels/document.xml.rels"))
        ct_xml = z.read("[Content_Types].xml").decode("utf-8", "replace")
        core_xml = z.read("docProps/core.xml").decode("utf-8", "replace") if "docProps/core.xml" in names else ""

        rel_map = {}
        hyperlinks = []
        for ch in rels_root:
            rid = ch.get("Id")
            tp = ch.get("Type", "")
            tgt = ch.get("Target", "")
            rel_map[rid] = (tp, tgt)
            if tp.endswith("/hyperlink"):
                hyperlinks.append((rid, tgt))

        # Check all drawing embeds resolve and images load cleanly
        embeds = []
        for el in doc_root.iter():
            for attr_k, attr_v in el.attrib.items():
                if attr_k.endswith("}embed") or attr_k == "embed":
                    embeds.append(attr_v)
        for rid in embeds:
            assert rid in rel_map, f"{path.name}: missing rel {rid}"
            _, tgt = rel_map[rid]
            img_zip_path = f"word/{tgt}"
            assert img_zip_path in names, f"{path.name}: missing embedded image {img_zip_path}"
            im = Image.open(io.BytesIO(z.read(img_zip_path)))
            im.verify()

        assert len(embeds) >= expected_imgs, f"{path.name}: expected >= {expected_imgs} images, found {len(embeds)}"
        assert 'Extension="png"' in ct_xml, f"{path.name}: [Content_Types].xml missing png default"

        body = doc_root.find(f"{{{W_NS}}}body")
        paras = list(body.iter(f"{{{W_NS}}}p"))
        tbls = list(body.iter(f"{{{W_NS}}}tbl"))
        full_text = "\n".join("".join(t.text or "" for t in p.iter(f"{{{W_NS}}}t")) for p in body.iter(f"{{{W_NS}}}p"))
        sect_prs = list(body.iter(f"{{{W_NS}}}sectPr"))
        hdrs = [n for n in names if "header" in n.lower()]
        ftrs = [n for n in names if "footer" in n.lower()]

    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "embeds": len(embeds),
        "tables": len(tbls),
        "paras": len(paras),
        "sect_prs": len(sect_prs),
        "hdrs": hdrs,
        "ftrs": ftrs,
        "hyperlinks": hyperlinks,
        "full_text": full_text,
        "core_xml": core_xml,
        "doc_root": doc_root,
    }

def check_citations_and_refs(full_text: str, label: str):
    # Separate main body from References list (lines starting with [k])
    lines = full_text.splitlines()
    ref_lines = [ln.strip() for ln in lines if re.match(r"^\[\d+\]\s", ln.strip())]
    body_text = "\n".join(ln for ln in lines if not re.match(r"^\[\d+\]\s", ln.strip()))
    assert len(ref_lines) == 42, f"{label}: expected 42 reference entries, found {len(ref_lines)}"
    ref_nums = [int(re.match(r"^\[(\d+)\]", ln).group(1)) for ln in ref_lines]
    assert ref_nums == list(range(1, 43)), f"{label}: non-sequential ref numbers"
    cited = set()
    for m in re.finditer(r"\[(\d+)(?:[\u2013–-](\d+))?\]", body_text):
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        for k in range(a, b + 1):
            cited.add(k)
    missing = set(range(1, 43)) - cited
    extra = cited - set(range(1, 43))
    assert not missing and not extra, f"{label}: citation sync error missing={missing}, extra={extra}"
    # Recency check across 42 refs
    y20_25 = 0
    y26 = 0
    for ln in ref_lines:
        ys = re.findall(r"\b(19\d\d|20\d\d)\b", ln)
        y = int(ys[-1]) if ys else 0
        if 2020 <= y <= 2025:
            y20_25 += 1
        elif y == 2026:
            y26 += 1
    body_words = len(body_text.split())
    return {
        "ref_count": len(ref_lines),
        "cited_count": len(cited),
        "years_2020_2025": y20_25,
        "pct_2020_2025": round(100.0 * y20_25 / 42.0, 2),
        "years_2020_2026": y20_25 + y26,
        "pct_2020_2026": round(100.0 * (y20_25 + y26) / 42.0, 2),
        "body_words_excl_refs": body_words,
    }

def check_jiim_anonymity(info):
    banned = ["karimi", "muchtar", "riza", "fawad", "syiah kuala", "banda aceh", "kemitraan negara", "mfkarimi", "kahlil", "fawadkarimi"]
    combined = (info["full_text"] + "\n" + info["core_xml"]).lower()
    leaks = [b for b in banned if re.search(rf"\b{re.escape(b)}\b", combined) or (len(b) > 5 and b in combined)]
    assert not leaks, f"JIIM DOUBLE-BLIND LEAK DETECTED: {leaks}"
    assert not info["hyperlinks"], f"JIIM unexpected external hyperlinks: {info['hyperlinks']}"
    return leaks

def check_joiv_rules(info):
    assert info["hdrs"] == [], f"JOIV headers forbidden: {info['hdrs']}"
    assert info["ftrs"] == [], f"JOIV footers forbidden: {info['ftrs']}"
    assert info["hyperlinks"] == [], f"JOIV hyperlinks forbidden: {info['hyperlinks']}"
    # Check no pgNumSeq or PAGE fields
    doc_s = ET.tostring(info["doc_root"], encoding="unicode")
    assert "fldSimple" not in doc_s and "instrText" not in doc_s, "JOIV page numbers/fields forbidden"
    # Check A4 size in first & final sectPr
    pgszs = list(info["doc_root"].iter(f"{{{W_NS}}}pgSz"))
    assert all(p.get(f"{{{W_NS}}}w") == "11906" and p.get(f"{{{W_NS}}}h") == "16838" for p in pgszs), "JOIV non-A4 page size detected"
    return True

def main():
    rep = {}
    # 1. Self-check figures
    for fg in MD.FIGURES:
        fp = ROOT / fg["path"]
        assert fp.exists(), f"Missing figure {fp}"
        w, h = Image.open(fp).size
        rep[fp.name] = {"px": [w, h], "bytes": fp.stat().st_size}

    # 2. Check JIIM double-blind manuscript
    jiim = check_docx_common(JIIM_DOCX, expected_imgs=4)
    leaks = check_jiim_anonymity(jiim)
    jiim_cit = check_citations_and_refs(jiim["full_text"], "JIIM")
    jiim_abs_words = len(" ".join(l + t for l, t in MD.JIIM_ABSTRACT_SECTIONS).split())
    assert jiim_abs_words <= 250
    rep["JIIM_DoubleBlind"] = {**{k: jiim[k] for k in ("file", "bytes", "embeds", "tables", "paras", "sect_prs")}, "leaks": leaks, "abs_words": jiim_abs_words, **jiim_cit}

    # 3. Check JIIM Title Page
    tp = check_docx_common(TP_DOCX, expected_imgs=0)
    assert "Mohammad Fawad Karimi" in tp["full_text"]
    assert "Universitas Syiah Kuala" in tp["full_text"]
    rep["JIIM_TitlePage"] = {k: tp[k] for k in ("file", "bytes", "paras")}

    # 4. Check JOIV Camera-Ready Template-Native manuscript
    joiv = check_docx_common(JOIV_DOCX, expected_imgs=5)  # 4 figures + 1 CC-BY-SA badge (rId9)
    check_joiv_rules(joiv)
    joiv_cit = check_citations_and_refs(joiv["full_text"], "JOIV")
    joiv_abs_words = len(MD.JOIV_ABSTRACT_TEXT.split())
    assert 230 <= joiv_abs_words <= 250
    assert joiv_cit["years_2020_2025"] >= 30 and joiv_cit["pct_2020_2025"] >= 70.0
    # Rough page count check: 5,200+ words in 10pt 2-col + 7 tables + 4 figs easily spans >= 7 content pages excl. refs
    assert joiv_cit["body_words_excl_refs"] >= 4500
    rep["JOIV_Manuscript"] = {**{k: joiv[k] for k in ("file", "bytes", "embeds", "tables", "paras", "sect_prs", "hdrs", "ftrs", "hyperlinks")}, "abs_words": joiv_abs_words, **joiv_cit}

    # 5. Check Pandoc readability smoke test on both main docx files
    for p in (JIIM_DOCX, TP_DOCX, JOIV_DOCX):
        proc = subprocess.run(["pandoc", str(p), "-t", "plain"], capture_output=True, text=True, check=True)
        rep[f"pandoc_{p.name}_chars"] = len(proc.stdout)

    # 6. Numeric truth spot-checks against verified_numbers_compact.json
    snap = json.loads((PKG / "verified_numbers_compact.json").read_text())
    hold = snap["hold_arm_e_8v"]["arm_e_hold_8view"]
    dev = snap["dev708"]["pooled_oof"]
    expected_strings = [
        f"{dev['baseline_dev708']['cpm']:.4f}",
        f"{dev['improved_dev708']['cpm']:.4f}",
        f"{dev['improved_dev708']['pr_auc']:.4f}",
        f"{hold['roc_auc']:.4f}",
        f"{hold['pr_auc']:.4f}",
        f"{hold['froc']['cpm']:.4f}",
        "75.54%",
        "67.14%",
        "0.4279",
        "0.4340",
        "0.0975",
        "0.8006",
        "0.9129",
    ]
    for label, txt in [("JIIM", jiim["full_text"]), ("JOIV", joiv["full_text"])]:
        for s in expected_strings:
            assert s in txt, f"{label}: missing verified metric string {s}"
    rep["numeric_truth_assertions"] = f"ALL_{len(expected_strings)}_STRINGS_VERIFIED_IN_BOTH_DOCX"

    (PKG / "verification_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print("VERIFY_PACKAGE_ALL_PASSED")

if __name__ == "__main__":
    main()
