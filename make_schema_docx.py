"""Generate the development-schema Word document from ARCHITECTURE.md.

Parses the markdown (headings, tables, code blocks, lists) and renders a
formatted .docx with the GitHub repository link in the header and footer.

Usage:  python make_schema_docx.py [output.docx]
"""
import re
import sys
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Inches, Pt, RGBColor

REPO_URL = "https://github.com/jasanjoseph/CAD.ai"
SOURCE_MD = Path(__file__).parent / "ARCHITECTURE.md"

BODY_FONT = "Calibri"
MONO_FONT = "Consolas"
ACCENT = RGBColor(0x1F, 0x3B, 0x73)


def shade(el, fill: str) -> None:
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    el.append(shd)


def add_hyperlink(paragraph, url: str, text: str) -> None:
    rid = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), rid)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    rpr.append(color)
    rpr.append(underline)
    run.append(rpr)
    t = OxmlElement("w:t")
    t.text = text
    run.append(t)
    link.append(run)
    paragraph._p.append(link)


_INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")


def add_runs(paragraph, text: str, size: float = 10.5, mono: bool = False) -> None:
    """Render text with **bold** and `code` inline formatting."""
    for token in _INLINE.split(text):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**"):
            run = paragraph.add_run(token[2:-2])
            run.bold = True
        elif token.startswith("`") and token.endswith("`"):
            run = paragraph.add_run(token[1:-1])
            run.font.name = MONO_FONT
            run.font.size = Pt(size - 1)
            run.font.color.rgb = RGBColor(0xA3, 0x1D, 0x1D)
        else:
            run = paragraph.add_run(token)
        run.font.size = run.font.size or Pt(size)
        if mono:
            run.font.name = MONO_FONT


def add_code_block(doc: Document, lines: list[str]) -> None:
    for i, line in enumerate(lines):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(6 if i == 0 else 0)
        p.paragraph_format.space_after = Pt(6 if i == len(lines) - 1 else 0)
        p.paragraph_format.left_indent = Inches(0.15)
        p.paragraph_format.line_spacing = 1.0
        shade(p._p.get_or_add_pPr(), "F2F4F7")
        run = p.add_run(line if line.strip() else " ")
        run.font.name = MONO_FONT
        run.font.size = Pt(7.5)
        run.font.color.rgb = RGBColor(0x21, 0x26, 0x2D)


def add_table(doc: Document, rows: list[list[str]]) -> None:
    if not rows:
        return
    header, body = rows[0], rows[1:]
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    table.autofit = True
    for cell, text in zip(table.rows[0].cells, header):
        cell.text = ""
        p = cell.paragraphs[0]
        add_runs(p, text.replace("**", ""), size=9.5)
        for run in p.runs:
            run.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade(cell._tc.get_or_add_tcPr(), "1F3B73")
    for row in body:
        cells = table.add_row().cells
        for cell, text in zip(cells, row):
            cell.text = ""
            add_runs(cell.paragraphs[0], text, size=9)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def parse_markdown(md: str):
    """Yield (kind, payload) tuples for headings, code blocks, tables, lists, paragraphs."""
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            yield "code", block
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            yield "table", rows
            continue
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            yield ("h1" if level == 1 else "h2" if level == 2 else "h3"), line.lstrip("# ").strip()
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(lines[i][2:].strip())
                i += 1
            yield "list", items
            continue
        elif re.match(r"^\d+\.\s", line):
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s", lines[i]):
                items.append(re.sub(r"^\d+\.\s", "", lines[i]).strip())
                i += 1
            yield "numlist", items
            continue
        elif line.strip() and line.strip() != "---":
            para = line.strip()
            while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(
                r"^(#|\||```|- |\d+\.\s)", lines[i + 1]
            ):
                i += 1
                para += " " + lines[i].strip()
            yield "p", para
        i += 1


def build(out_path: Path) -> Path:
    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = section.page_height, section.page_width
    for attr in ("left_margin", "right_margin"):
        setattr(section, attr, Inches(0.7))
    section.top_margin = Inches(0.6)
    section.bottom_margin = Inches(0.6)

    style = doc.styles["Normal"]
    style.font.name = BODY_FONT
    style.font.size = Pt(10.5)

    # Running footer with the repository link
    footer_p = section.footer.paragraphs[0]
    footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer_p.add_run("Repository: ")
    run.font.size = Pt(8)
    add_hyperlink(footer_p, REPO_URL, REPO_URL)
    for r in footer_p.runs:
        r.font.size = Pt(8)

    # ---- Title block ----
    title = doc.add_heading("Mechanical CAD Companion AI", level=0)
    for run in title.runs:
        run.font.color.rgb = ACCENT
    sub = doc.add_paragraph()
    add_runs(sub, "Development Schema & Architecture Reference", size=13)
    sub.runs[0].bold = True
    sub.runs[0].font.color.rgb = ACCENT

    link_p = doc.add_paragraph()
    link_p.add_run("Source repository: ").bold = True
    add_hyperlink(link_p, REPO_URL, REPO_URL)
    meta = doc.add_paragraph()
    meta.add_run(f"Generated {date.today().isoformat()} | branch main | language: Python | stack: LangGraph + trimesh + Gemini").italic = True
    doc.add_paragraph("_" * 118)

    # ---- Body from ARCHITECTURE.md ----
    for kind, payload in parse_markdown(SOURCE_MD.read_text(encoding="utf-8")):
        if kind == "h1":
            continue  # markdown title already rendered above
        if kind in ("h2", "h3"):
            h = doc.add_heading(level=2 if kind == "h2" else 3)
            run = h.add_run(payload)
            run.font.color.rgb = ACCENT
        elif kind == "code":
            add_code_block(doc, payload)
        elif kind == "table":
            add_table(doc, payload)
        elif kind == "list":
            for item in payload:
                p = doc.add_paragraph(style="List Bullet")
                add_runs(p, item)
        elif kind == "numlist":
            for item in payload:
                p = doc.add_paragraph(style="List Number")
                add_runs(p, item)
        elif kind == "p":
            p = doc.add_paragraph()
            add_runs(p, payload)

    # ---- Quick-start appendix ----
    h = doc.add_heading(level=2)
    run = h.add_run("Appendix - Clone & Run")
    run.font.color.rgb = ACCENT
    add_code_block(
        doc,
        [
            "git clone https://github.com/jasanjoseph/CAD.ai.git",
            "cd CAD.ai",
            "py -m venv .venv",
            ".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt shapely mapbox_earcut python-docx",
            "copy .env.example .env        # then paste your GOOGLE_API_KEY (free tier) into .env",
            ".\\.venv\\Scripts\\python.exe make_sample_part.py",
            ".\\.venv\\Scripts\\python.exe cad_companion.py ingest .\\rulebook\\baja_sae_2026.txt",
            '.\\.venv\\Scripts\\python.exe cad_companion.py run .\\parts\\bracket.stl --units mm --material aisi_4130 --part "suspension mounting bracket"',
        ],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)
    return out_path


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "CAD.ai_Development_Schema.docx"
    path = build(dest)
    print(f"wrote {path} ({path.stat().st_size} bytes)")
