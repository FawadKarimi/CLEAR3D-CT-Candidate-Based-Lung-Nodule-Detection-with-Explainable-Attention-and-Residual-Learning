# -*- coding: utf-8 -*-
"""Builds CLEAR3D_CT_JOIV_Manuscript.docx natively cloned from JOIV_Template_2025.docx XML."""
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
TEMPLATE_DOCX = ROOT / "legacy_drafts_and_templates/JOIV_Template_2025.docx"
OUT_JOIV = PKG / "CLEAR3D_CT_JOIV_Manuscript.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# Exact A4 JOIV dimensions (from JOIV_Template_2025.docx p[15] & p[212])
# A4 = 11906 x 16838 dxa; margins top=1077(19mm), bottom=1588(28mm), left=right=811(14.32mm)
# Printable width = 10284 dxa; 2 cols w=4901 + gap=482(8.5mm) + w=4901 = 10284 dxa.
SECT_1COL_OPEN = (
    '<w:sectPr>'
    '<w:pgSz w:h="16838" w:w="11906" w:orient="portrait"/>'
    '<w:pgMar w:bottom="1588" w:top="1077" w:left="811" w:right="811" w:header="709" w:footer="709"/>'
    '<w:pgNumType w:start="1"/>'
    '</w:sectPr>'
)
SECT_1COL_CONT = (
    '<w:sectPr>'
    '<w:type w:val="continuous"/>'
    '<w:pgSz w:h="16838" w:w="11906" w:orient="portrait"/>'
    '<w:pgMar w:bottom="1588" w:top="1077" w:left="811" w:right="811" w:header="709" w:footer="709"/>'
    '<w:cols w:equalWidth="1" w:num="1" w:space="0"/>'
    '</w:sectPr>'
)
SECT_2COL_CONT = (
    '<w:sectPr>'
    '<w:type w:val="continuous"/>'
    '<w:pgSz w:h="16838" w:w="11906" w:orient="portrait"/>'
    '<w:pgMar w:bottom="1588" w:top="1077" w:left="811" w:right="811" w:header="709" w:footer="709"/>'
    '<w:cols w:equalWidth="0" w:num="2">'
    '<w:col w:space="482" w:w="4901"/>'
    '<w:col w:space="0" w:w="4901"/>'
    '</w:cols>'
    '</w:sectPr>'
)

def make_run(text: str, sz_half: int = 20, bold: bool = False, italic: bool = False, small_caps: bool = False, sup: bool = False, br: bool = False) -> str:
    b_tag = '<w:b w:val="1"/>' if bold else '<w:b w:val="0"/>'
    i_tag = '<w:i w:val="1"/>' if italic else '<w:i w:val="0"/>'
    sc_tag = '<w:smallCaps w:val="1"/>' if small_caps else '<w:smallCaps w:val="0"/>'
    va = "superscript" if sup else "baseline"
    rpr = (
        '<w:rPr>'
        '<w:rFonts w:ascii="Times New Roman" w:cs="Times New Roman" w:eastAsia="Times New Roman" w:hAnsi="Times New Roman"/>'
        f'{b_tag}{i_tag}{sc_tag}<w:strike w:val="0"/><w:color w:val="000000"/>'
        f'<w:sz w:val="{sz_half}"/><w:szCs w:val="{sz_half}"/><w:u w:val="none"/>'
        f'<w:shd w:fill="auto" w:val="clear"/><w:vertAlign w:val="{va}"/><w:rtl w:val="0"/>'
        '</w:rPr>'
    )
    br_xml = '<w:br w:type="textWrapping"/>' if br else ""
    t_xml = f'<w:t xml:space="preserve">{escape(text)}</w:t>' if text else ""
    return f"<w:r>{rpr}{t_xml}{br_xml}</w:r>"

def make_rich_runs(text: str, sz_half: int = 20, bold: bool = False, italic: bool = False, small_caps: bool = False) -> str:
    parts = re.split(r"(\^[0-9]+)", text)
    out = []
    for pt in parts:
        if not pt:
            continue
        if pt.startswith("^") and pt[1:].isdigit():
            out.append(make_run(pt[1:], sz_half=sz_half, bold=bold, italic=italic, small_caps=small_caps, sup=True))
        else:
            out.append(make_run(pt, sz_half=sz_half, bold=bold, italic=italic, small_caps=small_caps, sup=False))
    return "".join(out)

def make_para(runs_xml: str, jc: str = "both", before: int = 0, after: int = 0, line: int = 240, first_line: int = 216, left: int = 0, hanging: int = 0, num_id: int = None, keep_next: bool = False, sect_pr: str = "", bottom_border: bool = False) -> str:
    kn = '<w:keepNext w:val="1"/>' if keep_next else '<w:keepNext w:val="0"/>'
    num_xml = f'<w:numPr><w:ilvl w:val="0"/><w:numId w:val="{num_id}"/></w:numPr>' if num_id is not None else ""
    pb = '<w:pBdr><w:bottom w:color="000000" w:space="1" w:sz="4" w:val="single"/></w:pBdr>' if bottom_border else ""
    if hanging:
        ind = f'<w:ind w:left="{left}" w:right="0" w:hanging="{hanging}"/>'
    else:
        ind = f'<w:ind w:left="{left}" w:right="0" w:firstLine="{first_line}"/>'
    sp = f'<w:sectPr>{sect_pr[10:]}' if sect_pr else ""
    if sect_pr:
        sp = sect_pr
    return (
        f'<w:p><w:pPr>{kn}<w:keepLines w:val="0"/><w:pageBreakBefore w:val="0"/><w:widowControl w:val="1"/>'
        f'{num_xml}{pb}<w:shd w:fill="auto" w:val="clear"/>'
        f'<w:spacing w:after="{after}" w:before="{before}" w:line="{line}" w:lineRule="auto"/>'
        f'{ind}<w:jc w:val="{jc}"/>{sp}</w:pPr>{runs_xml}</w:p>'
    )

def make_heading1(title: str, num_id: int = 5) -> str:
    # Level-1: 10pt Small Caps Centered Regular, before=240 after=80, numId=5 for I/II/III/IV, None for Acknowledgment/References
    return make_para(
        make_run(title, sz_half=20, bold=False, italic=False, small_caps=True),
        jc="center", before=240, after=80, line=240, first_line=0,
        left=288, hanging=288, num_id=num_id, keep_next=True
    )

def make_heading2(title: str, num_id: int = 2) -> str:
    # Level-2: 10pt Italic Left-justified, uppercase letter A., B., ... before=150 after=60
    return make_para(
        make_run(title, sz_half=20, bold=False, italic=True, small_caps=False),
        jc="left", before=150, after=60, line=240, first_line=0,
        left=288, hanging=288, num_id=num_id, keep_next=True
    )

def make_equation(eq_text: str, eq_num: str, col_w_dxa: int = 4901) -> str:
    center_tab = col_w_dxa // 2
    right_tab = col_w_dxa - 25
    rpr_tab = '<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>'
    runs = (
        f'<w:r>{rpr_tab}<w:tab/></w:r>'
        + make_run(eq_text, sz_half=20, italic=True)
        + f'<w:r>{rpr_tab}<w:tab/></w:r>'
        + make_run(f"({eq_num})", sz_half=20, italic=False)
    )
    return (
        '<w:p><w:pPr><w:keepNext w:val="0"/><w:widowControl w:val="1"/>'
        f'<w:tabs><w:tab w:val="center" w:leader="none" w:pos="{center_tab}"/><w:tab w:val="right" w:leader="none" w:pos="{right_tab}"/></w:tabs>'
        '<w:spacing w:before="80" w:after="80" w:line="240" w:lineRule="auto"/>'
        '<w:ind w:left="0" w:right="0" w:firstLine="0"/><w:jc w:val="left"/></w:pPr>'
        f'{runs}</w:p>'
    )

def make_image_para(rid: str, doc_id: int, img_name: str, cx_emu: int, cy_emu: int) -> str:
    return (
        '<w:p><w:pPr><w:keepNext w:val="1"/><w:spacing w:after="40" w:before="100" w:line="240" w:lineRule="auto"/>'
        '<w:ind w:left="0" w:right="0" w:firstLine="0"/><w:jc w:val="center"/></w:pPr>'
        f'<w:r><w:drawing><wp:inline distB="0" distT="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx_emu}" cy="{cy_emu}"/><wp:effectExtent b="0" l="0" r="0" t="0"/>'
        f'<wp:docPr id="{doc_id}" name="{escape(img_name)}"/>'
        '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="{escape(img_name)}"/><pic:cNvPicPr preferRelativeResize="0"/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:srcRect b="0" l="0" r="0" t="0"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx_emu}" cy="{cy_emu}"/></a:xfrm>'
        '<a:prstGeom prst="rect"/><a:ln/></pic:spPr>'
        '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'
    )

def make_joiv_table(headers: list[str], rows: list[list[str]], widths_dxa: list[int], sz_cell_half: int = 16) -> str:
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths_dxa)
    total_w = sum(widths_dxa)
    tbl_pr = (
        f'<w:tblPr><w:tblW w:w="{total_w}" w:type="dxa"/><w:jc w:val="center"/><w:tblLayout w:type="fixed"/>'
        '<w:tblBorders>'
        '<w:top w:color="000000" w:space="0" w:sz="6" w:val="single"/>'
        '<w:bottom w:color="000000" w:space="0" w:sz="6" w:val="single"/>'
        '<w:insideH w:color="CCCCCC" w:space="0" w:sz="2" w:val="single"/>'
        '<w:left w:val="none"/><w:right w:val="none"/><w:insideV w:val="none"/>'
        '</w:tblBorders>'
        '<w:tblCellMar><w:top w:w="35" w:type="dxa"/><w:bottom w:w="35" w:type="dxa"/>'
        '<w:left w:w="50" w:type="dxa"/><w:right w:w="50" w:type="dxa"/></w:tblCellMar></w:tblPr>'
    )
    trs = []
    h_cells = []
    for i, h in enumerate(headers):
        jc = "left" if i == 0 else "center"
        h_cells.append(
            f'<w:tc><w:tcPr><w:tcW w:w="{widths_dxa[i]}" w:type="dxa"/>'
            '<w:tcBorders><w:bottom w:color="000000" w:space="0" w:sz="4" w:val="single"/></w:tcBorders>'
            '<w:vAlign w:val="center"/></w:tcPr>'
            + make_para(make_rich_runs(h, sz_half=sz_cell_half, bold=True), jc=jc, before=0, after=0, line=220, first_line=0)
            + "</w:tc>"
        )
    trs.append('<w:tr><w:trPr><w:cantSplit/><w:tblHeader/></w:trPr>' + "".join(h_cells) + "</w:tr>")
    for r_idx, row in enumerate(rows):
        is_sum = (r_idx >= len(rows) - 2 and any(k in row[0] for k in ("Fold Mean", "Pooled OOF", "CLEAR3D-CT")))
        b_cells = []
        for i, val in enumerate(row):
            jc = "left" if i == 0 or (i == len(row) - 1 and len(val) > 26) else "center"
            b_cells.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{widths_dxa[i]}" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr>'
                + make_para(make_rich_runs(val, sz_half=sz_cell_half, bold=is_sum), jc=jc, before=0, after=0, line=215, first_line=0)
                + "</w:tc>"
            )
        trs.append('<w:tr><w:trPr><w:cantSplit/></w:trPr>' + "".join(b_cells) + "</w:tr>")
    return f"<w:tbl>{tbl_pr}<w:tblGrid>{grid}</w:tblGrid>{''.join(trs)}</w:tbl>"

def build_joiv():
    with zipfile.ZipFile(TEMPLATE_DOCX, "r") as zin:
        doc_bytes = zin.read("word/document.xml").decode("utf-8", "replace")
        root = ET.fromstring(doc_bytes.encode("utf-8"))
        body = root.find(f"{{{W_NS}}}body")
        orig_children = list(body)
        # Keep p[0] (top spacing / VML frame) and p[12..14] (CC license text, CC image rId9, divider line) verbatim
        m_doc = re.search(r"<w:document\b[^>]*>", doc_bytes)
        assert m_doc
        doc_open = doc_bytes[: m_doc.end()]
        # Strip ns0 prefixes when serializing preserved template nodes by re-registering standard prefixes
        for m in re.finditer(r'xmlns:([A-Za-z0-9_]+)="([^"]+)"', doc_open):
            ET.register_namespace(m.group(1), m.group(2))
        p0_xml = ET.tostring(orig_children[0], encoding="unicode")
        p12_xml = ET.tostring(orig_children[12], encoding="unicode")
        p13_xml = ET.tostring(orig_children[13], encoding="unicode")
        p14_xml = ET.tostring(orig_children[14], encoding="unicode")

    # Map figures cleanly for column vs full-width spanning
    media_map = {}
    fig_dims = {}
    col_w_emu = int((4901 / 1440.0) * 914400)   # 3.403 in
    page_w_emu = int((9900 / 1440.0) * 914400)  # 6.875 in within 7.14 in printable A4 width
    for idx, fg in enumerate(MD.FIGURES, start=1):
        rid = f"rId{40 + idx}"
        p = ROOT / fg["path"]
        w_px, h_px = Image.open(p).size
        # Fig 2 stays single-column (compact clear curve); Figs 1, 3, 4 span full 2-column A4 width for maximum legibility!
        span_2col = fg["num"] in ("1", "3", "4")
        cx = page_w_emu if span_2col else col_w_emu
        cy = int(cx * (h_px / w_px))
        media_map[rid] = (f"joiv_fig{fg['num']}.png", p)
        fig_dims[fg["num"]] = (rid, p.name, cx, cy, span_2col)

    parts = []
    # --- SECTION 1: 1-COLUMN HEADER BLOCK ---
    parts.append(p0_xml)
    # Title: 18pt Regular Centered, spacing before=2560
    parts.append(make_para(make_run(MD.TITLE_CASE, sz_half=36, bold=False, italic=False), jc="center", before=2560, after=0, line=240, first_line=0))
    # Authors: 12pt Centered with superscripts (a,*, b)
    auth_runs = (
        make_run("Mohammad Fawad Karimi", sz_half=24) + make_run("a,*", sz_half=24, sup=True)
        + make_run(", Kahlil Muchtar", sz_half=24) + make_run("b", sz_half=24, sup=True)
        + make_run(", Hammam Riza", sz_half=24) + make_run("a", sz_half=24, sup=True)
    )
    parts.append(make_para(auth_runs, jc="center", before=120, after=120, line=240, first_line=0))
    # Affiliations: 9pt Italic Centered
    aff_runs = (
        make_run("a", sz_half=18, italic=True, sup=True)
        + make_run(" Department of Informatics, Faculty of Mathematics and Natural Sciences, Universitas Syiah Kuala, Banda Aceh 23111, Indonesia", sz_half=18, italic=True, br=True)
        + make_run("b", sz_half=18, italic=True, sup=True)
        + make_run(" Department of Electrical and Computer Engineering, Universitas Syiah Kuala, Banda Aceh 23111, Indonesia", sz_half=18, italic=True)
    )
    parts.append(make_para(aff_runs, jc="center", before=0, after=60, line=240, first_line=0))
    # Corresponding Email: 9pt Italic Centered (plain text, NO hyperlink!)
    parts.append(make_para(make_run(MD.CORR_EMAIL_TEXT, sz_half=18, italic=True), jc="center", before=0, after=60, line=240, first_line=0))
    # Divider line + spacer
    parts.append(make_para(make_run("", sz_half=18), jc="left", before=0, after=0, line=240, first_line=0, bottom_border=True))
    parts.append(make_para(make_run("", sz_half=18), jc="left", before=0, after=0, line=240, first_line=0))
    # Abstract: 9pt Bold+Italic "Abstract", Regular "—", 9pt Bold body (exact match to p[7] in JOIV_Template_2025.docx!)
    abs_runs = (
        make_run("Abstract", sz_half=18, bold=True, italic=True)
        + make_run("— ", sz_half=18, bold=False, italic=False)
        + make_rich_runs(MD.JOIV_ABSTRACT_TEXT, sz_half=18, bold=True, italic=False)
    )
    parts.append(make_para(abs_runs, jc="both", before=0, after=0, line=240, first_line=0))
    parts.append(make_para(make_run("", sz_half=18), jc="left", before=0, after=0, line=240, first_line=0))
    # Keywords: 9pt Bold+Italic "Keywords", Regular "—", 9pt Bold semicolon-separated keywords
    kw_runs = (
        make_run("Keywords", sz_half=18, bold=True, italic=True)
        + make_run("— ", sz_half=18, bold=False, italic=False)
        + make_run("; ".join(MD.KEYWORDS_LIST) + ".", sz_half=18, bold=True, italic=False)
    )
    parts.append(make_para(kw_runs, jc="both", before=0, after=0, line=240, first_line=0))
    parts.append(make_para(make_run("", sz_half=18), jc="left", before=0, after=0, line=240, first_line=0))
    # Editorial dates line + CC license + CC icon (rId9 -> image4.png) + divider + Section 1 Break (p[15])
    parts.append(make_para(make_run("Manuscript received —; revised —; accepted —. Date of publication —.", sz_half=16, italic=True), jc="right", before=0, after=0, line=240, first_line=0, bottom_border=True))
    parts.append(p12_xml)
    parts.append(p13_xml)
    parts.append(p14_xml)
    parts.append(make_para(make_run("", sz_half=20), jc="left", before=0, after=0, line=240, first_line=0, sect_pr=SECT_1COL_OPEN))

    # --- HELPERS FOR SECTION 2 BODY ---
    def add_body(txt: str):
        parts.append(make_para(make_rich_runs(txt, sz_half=20), jc="both", before=0, after=0, line=240, first_line=216))

    def add_h3_inline(label: str, body_txt: str):
        # JOIV Level-3 Heading: indented, Italic, ends with colon, body immediately follows in same paragraph!
        runs = make_run(label + ": ", sz_half=20, italic=True) + make_rich_runs(body_txt, sz_half=20, italic=False)
        parts.append(make_para(runs, jc="both", before=60, after=0, line=240, first_line=216))

    def enter_span():
        parts.append(make_para(make_run("", sz_half=16), jc="left", before=0, after=0, line=120, first_line=0, sect_pr=SECT_2COL_CONT))

    def exit_span():
        parts.append(make_para(make_run("", sz_half=16), jc="left", before=0, after=0, line=120, first_line=0, sect_pr=SECT_1COL_CONT))

    def add_fig_raw(fnum: str):
        fg = [f for f in MD.FIGURES if f["num"] == fnum][0]
        rid, tname, cx, cy, _ = fig_dims[fnum]
        parts.append(make_image_para(rid, int(fnum) + 10, tname, cx, cy))
        jc_cap = "center" if len(fg["caption"]) < 85 else "both"
        parts.append(make_para(make_rich_runs(fg["caption"], sz_half=16), jc=jc_cap, before=0, after=80, line=240, first_line=0))

    def add_fig(fnum: str):
        _, _, _, _, span2 = fig_dims[fnum]
        if span2:
            enter_span()
        add_fig_raw(fnum)
        if span2:
            exit_span()

    def add_tbl_raw(idx_0: int, span2: bool = True):
        tb = MD.TABLES[idx_0]
        widths = [int(w * 9900 / 9360) for w in tb["widths_dxa_jiim"]] if span2 else tb["widths_dxa_joiv"]
        cap_runs = make_run(f"TABLE {tb['num_roman']}", sz_half=16, small_caps=True, br=True) + make_rich_runs(tb["title"], sz_half=16, small_caps=True)
        parts.append(make_para(cap_runs, jc="center", before=120, after=60, line=240, first_line=0, keep_next=True))
        parts.append(make_joiv_table(tb["headers"], tb["rows"], widths, sz_cell_half=16))
        parts.append(make_para(make_run("", sz_half=16), jc="left", before=0, after=60, line=200, first_line=0))

    def add_tbl(idx_0: int, span2: bool = True):
        if span2:
            enter_span()
        add_tbl_raw(idx_0, span2=span2)
        if span2:
            exit_span()

    # --- SECTION I. INTRODUCTION (integrating Related Work cleanly as subtopics) ---
    parts.append(make_heading1("Introduction", num_id=5))
    for p in MD.SEC_INTRO:
        add_body(p)
    # Related Work integrated cleanly with JOIV Level-2 headings (numId=7 or fresh letter A., B.)
    for idx, (st, ps) in enumerate(MD.SEC_RELWORK_SUBS, start=1):
        letter = "A" if idx == 1 else "B"
        # Use explicit Level-2 formatting matching template when inside I. so numbering never collides with II. A.
        parts.append(make_para(make_run(f"{letter}. {st}", sz_half=20, italic=True), jc="left", before=150, after=60, line=240, first_line=0, keep_next=True))
        for p in ps:
            add_body(p)

    # --- SECTION II. THE MATERIALS AND METHOD ---
    parts.append(make_heading1("The Materials and Method", num_id=5))
    for idx, (st, ps) in enumerate(MD.SEC_METHODS_SUBS, start=1):
        parts.append(make_heading2(st, num_id=2))
        for p in ps:
            add_body(p)
        if idx == 1:
            add_tbl(0, span2=True)
        elif idx == 2:
            add_fig("1")
        elif idx == 4:
            # Flush-left centered equations (1) and (2) without image format
            add_body("Specifically, eight-view log-odds aggregation and matched Inner Fold 1 Platt probability calibration follow (1) and (2):")
            parts.append(make_equation("l_TTA8(x) = (1/8) \u03a3_{v=1..8} [z\u2081(T_v(x)) \u2212 z\u2080(T_v(x))]", "1"))
            parts.append(make_equation("p_cal(x) = \u03c3(T\u2080 \u00b7 l_TTA8(x) + b\u2080)", "2"))

    # --- SECTION III. RESULTS AND DISCUSSION ---
    parts.append(make_heading1("Results and Discussion", num_id=5))
    for idx, (st, ps) in enumerate(MD.SEC_RESULTS_SUBS, start=1):
        parts.append(make_heading2(st, num_id=3))
        for p in ps:
            add_body(p)
        if idx == 1:
            add_tbl(1, span2=True)
        elif idx == 2:
            enter_span()
            add_tbl_raw(2, span2=True)
            add_fig_raw("4")
            exit_span()
        elif idx == 3:
            add_tbl(3, span2=True)
        elif idx == 4:
            add_fig("2")
            enter_span()
            add_tbl_raw(4, span2=True)
            add_fig_raw("3")
            exit_span()
        elif idx == 5:
            add_tbl(5, span2=True)

    # Integrate Discussion subsections seamlessly into III. Results and Discussion as F. and G.
    for idx, (st, ps) in enumerate(MD.SEC_DISCUSSION_ONLY, start=6):
        parts.append(make_heading2(st, num_id=3))
        for j, p in enumerate(ps):
            if idx == 7 and j == 2:
                add_h3_inline("1) Planned Validation and Future Directions", p)
            else:
                add_body(p)
        if idx == 6:
            add_tbl(6, span2=True)

    # --- SECTION IV. CONCLUSION ---
    parts.append(make_heading1("Conclusion", num_id=5))
    for p in MD.SEC_CONCLUSION:
        add_body(p)

    # --- UNNUMBERED ACKNOWLEDGMENT & REFERENCES ---
    parts.append(make_heading1("Acknowledgment", num_id=None))
    add_body(MD.ACKNOWLEDGMENT_FULL)
    add_body(
        "Ethical and Disclosure Notes: This study exclusively analyzed de-identified public thoracic CT datasets (LIDC-IDRI, LUNA16, SPIE-AAPM-NCI LUNGx). "
        "The authors declare no competing financial or non-financial interests. Generative AI coding and editorial assistants aided script auditing and manuscript polishing under strict author verification of every numerical result and reference."
    )

    parts.append(make_heading1("References", num_id=None))
    for ref in MD.REFS:
        # 8pt (sz=16) Times New Roman, hanging indent, Regular/Italic styling cleanly separated, ZERO hyperlinks
        # Split reference cleanly at title closing quote to italicize venue subtly if comma-delimited
        parts.append(make_para(make_run(ref, sz_half=16), jc="both", before=0, after=30, line=240, first_line=0, left=380, hanging=380))

    final_xml = doc_open + "<w:body>" + "".join(parts) + SECT_2COL_CONT + "</w:body></w:document>"
    ET.fromstring(final_xml.encode("utf-8"))

    with zipfile.ZipFile(TEMPLATE_DOCX, "r") as zin:
        rels_xml = zin.read("word/_rels/document.xml.rels").decode("utf-8", "replace")
        rels_root = ET.fromstring(rels_xml)
        rel_ns = rels_root.tag.split("}")[0].strip("{")
        ET.register_namespace("", rel_ns)
        # Keep existing rId9 (CC-BY-SA logo used in p13_xml) and strip hyperlinks / unused ole/sample images
        unused_targets = {"media/image1.wmf", "media/image2.png", "media/image3.png", "media/image5.png", "embeddings/oleObject1.bin"}
        for ch in list(rels_root):
            tp = ch.get("Type", "")
            tgt = ch.get("Target", "")
            if tp.endswith("/hyperlink") or tgt in unused_targets:
                rels_root.remove(ch)
        for rid, (tname, _) in media_map.items():
            el = ET.Element(f"{{{rel_ns}}}Relationship")
            el.set("Id", rid)
            el.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
            el.set("Target", f"media/{tname}")
            rels_root.append(el)
        new_rels_bytes = ET.tostring(rels_root, encoding="utf-8", xml_declaration=True)

        core_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            f'<dc:title>{escape(MD.TITLE_CASE)}</dc:title>'
            '<dc:creator>Mohammad Fawad Karimi</dc:creator>'
            '<cp:lastModifiedBy>Mohammad Fawad Karimi</cp:lastModifiedBy>'
            '</cp:coreProperties>'
        ).encode("utf-8")

        unused_zip = {"word/media/image1.wmf", "word/media/image2.png", "word/media/image3.png", "word/media/image5.png", "word/embeddings/oleObject1.bin"}
        with zipfile.ZipFile(OUT_JOIV, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                fn = item.filename
                if fn in unused_zip:
                    continue
                if fn == "word/document.xml":
                    zout.writestr(fn, final_xml.encode("utf-8"))
                elif fn == "word/_rels/document.xml.rels":
                    zout.writestr(fn, new_rels_bytes)
                elif fn == "docProps/core.xml":
                    zout.writestr(fn, core_xml)
                else:
                    zout.writestr(fn, zin.read(fn))
            for rid, (tname, img_p) in media_map.items():
                zout.write(img_p, f"word/media/{tname}")

    print("BUILT_JOIV_DOCX:", OUT_JOIV, OUT_JOIV.stat().st_size)

if __name__ == "__main__":
    build_joiv()
