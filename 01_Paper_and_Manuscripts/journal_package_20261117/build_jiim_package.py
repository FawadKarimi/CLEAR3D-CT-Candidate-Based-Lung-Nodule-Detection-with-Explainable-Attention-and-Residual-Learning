# -*- coding: utf-8 -*-
"""Builds CLEAR3D_CT_JIIM_DoubleBlind_Manuscript.docx and JIIM_Title_Page_and_Declarations.docx."""
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manuscript_data as MD

ROOT = Path("/experiment-storage/chaidir/fawad-dsai/CLEAR3D_CT project/01_Paper_and_Manuscripts")
PKG = ROOT / "journal_package_20261117"
BASE_DOCX = ROOT / "legacy_drafts_and_templates/CLEAR3D_CT_IEEE_Updated_20261116.docx"
OUT_MAIN = PKG / "CLEAR3D_CT_JIIM_DoubleBlind_Manuscript.docx"
OUT_TP = PKG / "JIIM_Title_Page_and_Declarations.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

def make_run(text: str, sz_half: int = 22, bold: bool = False, italic: bool = False, color: str = "000000", sup: bool = False) -> str:
    b_tag = "<w:b/>" if bold else '<w:b w:val="0"/>'
    i_tag = "<w:i/>" if italic else '<w:i w:val="0"/>'
    v_tag = '<w:vertAlign w:val="superscript"/>' if sup else ""
    rpr = (
        '<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>'
        f'{b_tag}{i_tag}<w:color w:val="{color}"/><w:sz w:val="{sz_half}"/><w:szCs w:val="{sz_half}"/>{v_tag}</w:rPr>'
    )
    return f'<w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'

def make_rich_runs(text: str, sz_half: int = 22, bold: bool = False, italic: bool = False, color: str = "000000") -> str:
    # Convert ^2, ^3 into crisp superscript or Unicode safely where not inside words
    # Note: replace 21^3, 33^3, 25^3, 48^3, 51^3, 16^3, 32^3, 20^3, 30^3, 40^3 with proper superscript runs!
    parts = re.split(r"(\^[0-9]+)", text)
    out = []
    for pt in parts:
        if not pt:
            continue
        if pt.startswith("^") and pt[1:].isdigit():
            out.append(make_run(pt[1:], sz_half=sz_half, bold=bold, italic=italic, color=color, sup=True))
        else:
            out.append(make_run(pt, sz_half=sz_half, bold=bold, italic=italic, color=color, sup=False))
    return "".join(out)

def make_para(runs_xml: str, jc: str = "both", before: int = 0, after: int = 120, line: int = 320, first_line: int = 0, keep_next: bool = False, pstyle: str = "Normal", border_bottom: bool = False, hanging: int = 0, left: int = 0) -> str:
    kn = "<w:keepNext/>" if keep_next else ""
    pb = '<w:pBdr><w:bottom w:val="single" w:sz="6" w:space="4" w:color="334155"/></w:pBdr>' if border_bottom else ""
    ind_attrs = []
    if left:
        ind_attrs.append(f'w:left="{left}"')
    if hanging:
        ind_attrs.append(f'w:hanging="{hanging}"')
    elif first_line:
        ind_attrs.append(f'w:firstLine="{first_line}"')
    ind = f'<w:ind {" ".join(ind_attrs)}/>' if ind_attrs else '<w:ind w:firstLine="0" w:left="0"/>'
    return (
        f'<w:p><w:pPr><w:pStyle w:val="{pstyle}"/>{kn}{pb}'
        f'<w:spacing w:before="{before}" w:after="{after}" w:line="{line}" w:lineRule="auto"/>'
        f'{ind}<w:jc w:val="{jc}"/></w:pPr>{runs_xml}</w:p>'
    )

def make_image_para(rid: str, doc_id: int, img_name: str, cx_emu: int, cy_emu: int) -> str:
    return (
        '<w:p><w:pPr><w:keepNext/><w:spacing w:before="140" w:after="60" w:line="240" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>'
        f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx_emu}" cy="{cy_emu}"/>'
        f'<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{doc_id}" name="Picture {doc_id}"/>'
        f'<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        f'<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        f'<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:nvPicPr><pic:cNvPr id="0" name="{escape(img_name)}"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx_emu}" cy="{cy_emu}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        f'</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'
    )

def make_table(headers: list[str], rows: list[list[str]], widths_dxa: list[int]) -> str:
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths_dxa)
    total_w = sum(widths_dxa)
    tbl_pr = (
        f'<w:tblPr><w:tblW w:w="{total_w}" w:type="dxa"/><w:jc w:val="center"/><w:tblLayout w:type="fixed"/>'
        '<w:tblBorders>'
        '<w:top w:val="single" w:sz="8" w:space="0" w:color="1E293B"/>'
        '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="1E293B"/>'
        '<w:insideH w:val="single" w:sz="2" w:space="0" w:color="CBD5E1"/>'
        '<w:left w:val="none"/><w:right w:val="none"/><w:insideV w:val="none"/>'
        '</w:tblBorders>'
        '<w:tblCellMar><w:top w:w="50" w:type="dxa"/><w:bottom w:w="50" w:type="dxa"/>'
        '<w:left w:w="70" w:type="dxa"/><w:right w:w="70" w:type="dxa"/></w:tblCellMar></w:tblPr>'
    )
    trs = []
    # Header row
    h_cells = []
    for i, h in enumerate(headers):
        jc = "left" if i == 0 else "center"
        h_cells.append(
            f'<w:tc><w:tcPr><w:tcW w:w="{widths_dxa[i]}" w:type="dxa"/><w:shd w:val="clear" w:color="auto" w:fill="F1F5F9"/>'
            '<w:tcBorders><w:bottom w:val="single" w:sz="6" w:space="0" w:color="334155"/></w:tcBorders><w:vAlign w:val="center"/></w:tcPr>'
            + make_para(make_rich_runs(h, sz_half=18, bold=True), jc=jc, before=0, after=0, line=240)
            + "</w:tc>"
        )
    trs.append("<w:tr><w:trPr><w:cantSplit/><w:tblHeader/></w:trPr>" + "".join(h_cells) + "</w:tr>")
    for r_idx, row in enumerate(rows):
        b_cells = []
        is_summary = (r_idx >= len(rows) - 2 and any(k in row[0] for k in ("Fold Mean", "Pooled OOF", "CLEAR3D-CT")))
        shd = '<w:shd w:val="clear" w:color="auto" w:fill="F8FAFC"/>' if is_summary or r_idx % 2 == 1 else ""
        for i, val in enumerate(row):
            jc = "left" if i == 0 or (i == len(row) - 1 and len(val) > 26) else "center"
            b_cells.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{widths_dxa[i]}" w:type="dxa"/>{shd}<w:vAlign w:val="center"/></w:tcPr>'
                + make_para(make_rich_runs(val, sz_half=17, bold=is_summary), jc=jc, before=0, after=0, line=230)
                + "</w:tc>"
            )
        trs.append("<w:tr><w:trPr><w:cantSplit/></w:trPr>" + "".join(b_cells) + "</w:tr>")
    return f"<w:tbl>{tbl_pr}<w:tblGrid>{grid}</w:tblGrid>{''.join(trs)}</w:tbl>"

def extract_doc_wrapper(base_docx: Path):
    with zipfile.ZipFile(base_docx, "r") as z:
        doc_bytes = z.read("word/document.xml").decode("utf-8", "replace")
    m = re.search(r"<w:document\b[^>]*>", doc_bytes)
    assert m, "Could not find <w:document> open tag"
    return doc_bytes[: m.end()]

def package_clean_docx(out_path: Path, body_inner_xml: str, sect_pr_xml: str, media_map: dict[str, tuple[str, Path]], author_meta: str):
    doc_open = extract_doc_wrapper(BASE_DOCX)
    final_doc_xml = (
        doc_open
        + "<w:body>"
        + body_inner_xml
        + sect_pr_xml
        + "</w:body></w:document>"
    )
    # Parse check before zipping
    ET.fromstring(final_doc_xml.encode("utf-8"))

    with zipfile.ZipFile(BASE_DOCX, "r") as zin:
        rels_xml = zin.read("word/_rels/document.xml.rels").decode("utf-8", "replace")
        # Remove old image/hyperlink rels cleanly to prevent dangling or non-blind URIs
        rels_root = ET.fromstring(rels_xml)
        rel_ns = rels_root.tag.split("}")[0].strip("{")
        ET.register_namespace("", rel_ns)
        for ch in list(rels_root):
            t = ch.get("Type", "")
            if t.endswith("/image") or t.endswith("/hyperlink"):
                rels_root.remove(ch)
        for rid, (target_name, _) in media_map.items():
            el = ET.Element(f"{{{rel_ns}}}Relationship")
            el.set("Id", rid)
            el.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
            el.set("Target", f"media/{target_name}")
            rels_root.append(el)
        new_rels_bytes = ET.tostring(rels_root, encoding="utf-8", xml_declaration=True)

        core_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            f'<dc:title>{escape(MD.TITLE_CASE)}</dc:title>'
            f'<dc:creator>{escape(author_meta)}</dc:creator>'
            f'<cp:lastModifiedBy>{escape(author_meta)}</cp:lastModifiedBy>'
            '</cp:coreProperties>'
        ).encode("utf-8")

        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                fn = item.filename
                if fn.startswith("word/media/") or fn.startswith("word/header") or fn.startswith("word/footer"):
                    continue
                if fn == "word/document.xml":
                    zout.writestr(fn, final_doc_xml.encode("utf-8"))
                elif fn == "word/_rels/document.xml.rels":
                    zout.writestr(fn, new_rels_bytes)
                elif fn == "docProps/core.xml":
                    zout.writestr(fn, core_xml)
                else:
                    zout.writestr(fn, zin.read(fn))
            for rid, (target_name, img_path) in media_map.items():
                zout.write(img_path, f"word/media/{target_name}")

def build_main_double_blind():
    media_map = {}
    fig_dims = {}
    max_w_emu = int(6.2 * 914400)  # 6.2 inches inside 6.5 in printable width
    for idx, fg in enumerate(MD.FIGURES, start=1):
        rid = f"rId{50 + idx}"
        p = ROOT / fg["path"]
        w_px, h_px = Image.open(p).size
        cx = max_w_emu
        cy = int(cx * (h_px / w_px))
        # Clamp tall fig2 to max 4.4 in if needed
        max_h = int(4.6 * 914400)
        if cy > max_h:
            cy = max_h
            cx = int(cy * (w_px / h_px))
        tname = p.name
        media_map[rid] = (tname, p)
        fig_dims[fg["num"]] = (rid, tname, cx, cy)

    parts = []
    # Top banner + Title + Running Head (100% Double-Blind)
    parts.append(make_para(make_run("Original Research Article — Double-Blind Review Copy (Authors & Affiliations Redacted)", sz_half=18, bold=True, italic=True, color="475569"), jc="left", before=0, after=80, line=240))
    parts.append(make_para(make_run(MD.TITLE_CASE, sz_half=32, bold=True, color="0F172A"), jc="left", before=40, after=120, line=280))
    parts.append(make_para(make_run("Running Title: ", sz_half=20, bold=True) + make_run(MD.RUNNING_TITLE, sz_half=20, italic=True), jc="left", before=0, after=160, line=240, border_bottom=True))

    # Structured Abstract
    parts.append(make_para(make_run("Abstract", sz_half=24, bold=True, color="0F172A"), jc="left", before=140, after=80, line=260, keep_next=True))
    for lbl, txt in MD.JIIM_ABSTRACT_SECTIONS:
        parts.append(make_para(make_run(lbl, sz_half=21, bold=True) + make_rich_runs(txt, sz_half=21), jc="both", before=0, after=80, line=280))
    parts.append(make_para(make_run("Keywords: ", sz_half=20, bold=True) + make_run("; ".join(MD.KEYWORDS_LIST) + ".", sz_half=20, italic=False), jc="left", before=40, after=180, line=260, border_bottom=True))

    def h1(txt: str):
        parts.append(make_para(make_run(txt, sz_half=26, bold=True, color="0F172A"), jc="left", before=220, after=100, line=260, keep_next=True))

    def h2(txt: str):
        parts.append(make_para(make_run(txt, sz_half=22, bold=True, italic=True, color="1E293B"), jc="left", before=160, after=70, line=260, keep_next=True))

    def add_fig(fnum: str):
        fg = [f for f in MD.FIGURES if f["num"] == fnum][0]
        rid, tname, cx, cy = fig_dims[fnum]
        parts.append(make_image_para(rid, int(fnum), tname, cx, cy))
        parts.append(make_para(make_rich_runs(fg["caption"], sz_half=19, italic=False, color="1E293B"), jc="both", before=20, after=140, line=250))

    def add_tbl(idx_0: int):
        tb = MD.TABLES[idx_0]
        parts.append(make_para(make_run(f"Table {tb['num_arab']}. ", sz_half=19, bold=True) + make_rich_runs(tb["title"], sz_half=19, bold=False), jc="left", before=140, after=60, line=240, keep_next=True))
        parts.append(make_table(tb["headers"], tb["rows"], tb["widths_dxa_jiim"]))
        parts.append(make_para(make_run("", sz_half=16), jc="left", before=0, after=80, line=220))

    # 1. Introduction
    h1("1. Introduction")
    for p in MD.SEC_INTRO:
        parts.append(make_para(make_rich_runs(p, sz_half=22), jc="both", before=0, after=120, line=320))

    # 2. Related Work
    h1("2. Related Work")
    for idx, (st, ps) in enumerate(MD.SEC_RELWORK_SUBS, start=1):
        h2(f"2.{idx} {st}")
        for p in ps:
            parts.append(make_para(make_rich_runs(p, sz_half=22), jc="both", before=0, after=120, line=320))

    # 3. Materials and Methods
    h1("3. Materials and Methods")
    for idx, (st, ps) in enumerate(MD.SEC_METHODS_SUBS, start=1):
        h2(f"3.{idx} {st}")
        for p in ps:
            parts.append(make_para(make_rich_runs(p, sz_half=22), jc="both", before=0, after=120, line=320))
        if idx == 1:
            add_tbl(0)
        elif idx == 2:
            add_fig("1")

    # 4. Results
    h1("4. Results")
    for idx, (st, ps) in enumerate(MD.SEC_RESULTS_SUBS, start=1):
        h2(f"4.{idx} {st}")
        for p in ps:
            parts.append(make_para(make_rich_runs(p, sz_half=22), jc="both", before=0, after=120, line=320))
        if idx == 1:
            add_tbl(1)
        elif idx == 2:
            add_tbl(2)
            add_fig("4")
        elif idx == 3:
            add_tbl(3)
        elif idx == 4:
            add_tbl(4)
            add_fig("2")
            add_fig("3")
        elif idx == 5:
            add_tbl(5)

    # 5. Discussion
    h1("5. Discussion")
    for idx, (st, ps) in enumerate(MD.SEC_DISCUSSION_ONLY, start=1):
        h2(f"5.{idx} {st}")
        for p in ps:
            parts.append(make_para(make_rich_runs(p, sz_half=22), jc="both", before=0, after=120, line=320))
        if idx == 1:
            add_tbl(6)

    # 6. Conclusion
    h1("6. Conclusion")
    for p in MD.SEC_CONCLUSION:
        parts.append(make_para(make_rich_runs(p, sz_half=22), jc="both", before=0, after=120, line=320))

    # Acknowledgments & Declarations (Blinded)
    h1("Acknowledgments")
    parts.append(make_para(make_run(MD.ACKNOWLEDGMENT_BLIND, sz_half=21, italic=True), jc="both", before=0, after=120, line=280))

    h1("Declarations")
    for lbl, txt in MD.DECLARATIONS_BLIND:
        parts.append(make_para(make_run(lbl, sz_half=21, bold=True) + make_run(txt, sz_half=21), jc="both", before=0, after=80, line=280))

    # References
    h1("References")
    for ref in MD.REFS:
        parts.append(make_para(make_run(ref, sz_half=19), jc="left", before=0, after=60, line=240, left=420, hanging=420))

    # Clean single-column Letter layout with 1.0 in (1440 dxa) margins
    sect_pr = (
        '<w:sectPr>'
        '<w:pgSz w:w="12240" w:h="15840" w:orient="portrait"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>'
        '<w:cols w:num="1" w:space="720"/>'
        '</w:sectPr>'
    )
    package_clean_docx(OUT_MAIN, "".join(parts), sect_pr, media_map, "Anonymous Authors (Double-Blind Review)")
    print("BUILT_JIIM_MAIN:", OUT_MAIN, OUT_MAIN.stat().st_size)

def build_title_page():
    parts = []
    parts.append(make_para(make_run("TITLE PAGE & SUBMISSION DECLARATIONS", sz_half=22, bold=True, color="475569"), jc="left", before=0, after=100, line=240, border_bottom=True))
    parts.append(make_para(make_run("Target Venue: ", sz_half=21, bold=True) + make_run("Journal of Imaging Informatics in Medicine (JIIM, Springer Nature) — Zero-APC Subscription Publishing Model", sz_half=21), jc="left", before=60, after=100, line=260))
    parts.append(make_para(make_run("Manuscript Title: ", sz_half=22, bold=True) + make_run(MD.TITLE_CASE, sz_half=22, bold=False), jc="left", before=40, after=80, line=280))
    parts.append(make_para(make_run("Running Title: ", sz_half=21, bold=True) + make_run(MD.RUNNING_TITLE, sz_half=21, italic=True), jc="left", before=0, after=140, line=260))

    parts.append(make_para(make_run("Authors and Affiliations", sz_half=24, bold=True, color="0F172A"), jc="left", before=120, after=80, line=260))
    for i, a in enumerate(MD.AUTHORS, start=1):
        star = " (Corresponding Author)" if a["corr"] else ""
        parts.append(make_para(
            make_run(f"{i}. {a['name']}{star}", sz_half=22, bold=True)
            + make_run(f"\n   Affiliation: {a['dept']}, {a['inst']}\n   Email: {a['email']} | ORCID: [INSERT ORCID iD]", sz_half=20),
            jc="left", before=0, after=90, line=260
        ))
    parts.append(make_para(make_run("Corresponding Author: ", sz_half=21, bold=True) + make_run("Mohammad Fawad Karimi, Department of Informatics, Faculty of Mathematics and Natural Sciences, Universitas Syiah Kuala, Banda Aceh 23111, Indonesia (mfkarimi@mhs.usk.ac.id).", sz_half=21), jc="both", before=40, after=160, line=260, border_bottom=True))

    parts.append(make_para(make_run("Full Acknowledgments", sz_half=24, bold=True, color="0F172A"), jc="left", before=120, after=60, line=260))
    parts.append(make_para(make_run(MD.ACKNOWLEDGMENT_FULL, sz_half=21), jc="both", before=0, after=140, line=280))

    parts.append(make_para(make_run("Mandatory Submission Statements & Disclosures", sz_half=24, bold=True, color="0F172A"), jc="left", before=120, after=80, line=260))
    tp_decls = [
        ("Funding: ", "Supported by the Kemitraan Negara Berkembang (KNB) Scholarship Program, Ministry of Education, Culture, Research, and Technology, Republic of Indonesia, and institutional compute resources at Universitas Syiah Kuala. [Authors: verify exact KNB contract/award reference number if applicable.]"),
        ("Competing Interests: ", "The authors declare that they have no competing financial or personal interests that could influence the work reported in this manuscript."),
        ("Ethical Approval and Consent to Participate: ", "This retrospective computational study exclusively used anonymized public CT scans from LIDC-IDRI / LUNA16 (TCIA) and SPIE-AAPM-NCI LUNGx. Formal institutional ethics exemption confirmation is archived by the corresponding author [Authors: insert institutional exemption reference/date before final submission]."),
        ("Author Contributions (CRediT): ", "[Authors: confirm exact role mapping before submission.] Provisional draft attribution: Mohammad Fawad Karimi — Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Writing - Original Draft, Visualization. Kahlil Muchtar — Supervision, Methodology, Writing - Review & Editing, Project Administration. Hammam Riza — Supervision, Writing - Review & Editing."),
        ("Data Availability: ", "All imaging benchmarks analyzed in this study are publicly hosted via official LUNA16 (Zenodo record 3723295 / grand-challenge.org) and The Cancer Imaging Archive (TCIA: LIDC-IDRI and SPIE-AAPM LUNGx)."),
        ("Code Availability: ", "Reproducible CLEAR3D-MSFv2 architecture definitions, patient-disjoint cohort manifests, nested Platt calibration scripts, and evaluation verification routines will be released via a clean public repository link upon peer-review acceptance (withheld during double-blind review)."),
        ("Generative AI Disclosure: ", "Generative AI tools were used strictly for Python script auditing, structured literature cross-checking, and English academic copy-editing; all experimental runs, statistical computations, numerical values, and scientific conclusions were generated and verified from primary logs by the authors."),
        ("Pre-Submission Administrative Checklist: ", "(1) Confirm ORCIDs for all three authors; (2) Confirm Kahlil Muchtar vs. Mohammad Fawad Karimi corresponding author designation across JIIM vs. JOIV coursework; (3) Attach LUNA16 Subset 1 / confirm_83 internal data provenance memo only if requested by editor; (4) Verify zero identifying strings in CLEAR3D_CT_JIIM_DoubleBlind_Manuscript.docx."),
    ]
    for lbl, txt in tp_decls:
        parts.append(make_para(make_run(lbl, sz_half=21, bold=True) + make_run(txt, sz_half=21), jc="both", before=0, after=85, line=280))

    sect_pr = (
        '<w:sectPr>'
        '<w:pgSz w:w="12240" w:h="15840" w:orient="portrait"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>'
        '<w:cols w:num="1" w:space="720"/>'
        '</w:sectPr>'
    )
    package_clean_docx(OUT_TP, "".join(parts), sect_pr, {}, "Mohammad Fawad Karimi")
    print("BUILT_JIIM_TITLEPAGE:", OUT_TP, OUT_TP.stat().st_size)

if __name__ == "__main__":
    build_main_double_blind()
    build_title_page()
