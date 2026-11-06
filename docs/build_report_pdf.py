"""Create a typeset PDF companion from the editable report-template DOCX.

This is an independent ReportLab typesetting pass, not a Word-export fidelity
claim. Run with the workspace Python that provides python-docx and reportlab.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import re
from pathlib import Path
from xml.etree import ElementTree as ET

from docx import Document
from docx.oxml.ns import qn
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    Image as RLImage,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph


NAVY = colors.HexColor("#16283F")
TEAL = colors.HexColor("#237C83")
PALE = colors.HexColor("#EAF2F3")
LIGHT = colors.HexColor("#F2F5F8")
MUTED = colors.HexColor("#596A7C")
GRID = colors.HexColor("#D6DCE3")
BODY = colors.HexColor("#273444")
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def locate_paths(script: Path):
    repo = None
    for parent in script.resolve().parents:
        if (parent / "docs").is_dir() and (parent / "ml").is_dir() and (parent / "models").is_dir():
            repo = parent
            break
        if (parent / "outputs" / "ObjectLab").is_dir():
            repo = parent / "outputs" / "ObjectLab"
            break
    if repo is None:
        raise RuntimeError("Could not locate ObjectLab relative to this script")
    out = repo.parent if repo.parent.name == "outputs" else repo
    return repo, out


def register_fonts():
    candidates = [Path(r"C:\Windows\Fonts\arial.ttf"), Path(r"C:\Windows\Fonts\calibri.ttf")]
    bold_candidates = [Path(r"C:\Windows\Fonts\arialbd.ttf"), Path(r"C:\Windows\Fonts\calibrib.ttf")]
    regular = next((p for p in candidates if p.exists()), None)
    bold = next((p for p in bold_candidates if p.exists()), None)
    if regular and bold:
        pdfmetrics.registerFont(TTFont("ReportSans", str(regular)))
        pdfmetrics.registerFont(TTFont("ReportSans-Bold", str(bold)))
        return "ReportSans", "ReportSans-Bold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = register_fonts()


def normalize(text: str) -> str:
    text = (text or "").replace("\u00a0", " ")
    for char in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"):
        text = text.replace(char, "-")
    return text.replace("\u00d7", "x").replace("\u00b1", "+/-").replace("\u2192", "->").replace("\u2022", "|")


def esc(text: str) -> str:
    from xml.sax.saxutils import escape

    return escape(normalize(text)).replace("\n", "<br/>")


def blocks(document):
    body = document.element.body
    for element in body.iterchildren():
        if element.tag == qn("w:p"):
            yield DocxParagraph(element, document)
        elif element.tag == qn("w:tbl"):
            yield DocxTable(element, document)


def has_page_break(paragraph: DocxParagraph) -> bool:
    return bool(paragraph._p.xpath('.//w:br[@w:type="page"]')) or bool(
        paragraph._p.xpath("./w:pPr/w:pageBreakBefore")
    )


def parse_rgb(hex_value: str | None):
    if not hex_value or hex_value in ("auto", "FFFFFF"):
        return None
    try:
        return colors.HexColor("#" + hex_value)
    except Exception:
        return None


def table_flowable(table: DocxTable, width: float):
    rows = []
    fills = []
    is_pipeline = len(table.rows) == 1 and len(table.columns) == 5 and "EXIF" in table.cell(0, 0).text
    for r_index, row in enumerate(table.rows):
        cells = []
        for c_index, cell in enumerate(row.cells):
            txt = esc(cell.text)
            shade = cell._tc.xpath("./w:tcPr/w:shd/@w:fill")
            bg = parse_rgb(shade[0] if shade else None)
            if is_pipeline:
                dark = bg in (TEAL, NAVY)
                style = ParagraphStyle(
                    "diagram-cell",
                    fontName=FONT_BOLD,
                    fontSize=8.7,
                    leading=11,
                    textColor=colors.white if dark else NAVY,
                    alignment=TA_CENTER,
                    spaceAfter=0,
                )
                cells.append(Paragraph(txt, style))
                fills.append((r_index, c_index, bg or PALE))
            else:
                header = r_index == 0
                style = ParagraphStyle(
                    "cell-header" if header else "cell",
                    fontName=FONT_BOLD if header else FONT,
                    fontSize=8.8 if header else 9.0,
                    leading=10.8,
                    textColor=colors.white if header else BODY,
                    alignment=TA_LEFT,
                    spaceAfter=0,
                    wordWrap="CJK",
                )
                cells.append(Paragraph(txt, style))
                if header:
                    fills.append((r_index, c_index, NAVY))
                else:
                    fills.append((r_index, c_index, LIGHT if r_index % 2 else colors.white))
        rows.append(cells)

    if not rows:
        return Spacer(1, 2)
    cols = len(rows[0])
    grid_cols = table._tbl.tblGrid.gridCol_lst
    raw_widths = []
    for col in grid_cols[:cols]:
        try:
            raw_widths.append(float(col.get(qn("w:w"), "0")))
        except (TypeError, ValueError):
            raw_widths.append(0.0)
    if len(raw_widths) != cols or not sum(raw_widths):
        raw_widths = [1.0] * cols
    col_widths = [width * value / sum(raw_widths) for value in raw_widths]
    if is_pipeline:
        t = Table(rows, colWidths=col_widths, hAlign="CENTER")
    else:
        t = Table(rows, colWidths=col_widths, repeatRows=1, hAlign="CENTER")
    cmds = [
        ("GRID", (0, 0), (-1, -1), 0.55, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for r, c, bg in fills:
        cmds.append(("BACKGROUND", (c, r), (c, r), bg))
    if is_pipeline:
        cmds.append(("BOX", (0, 0), (-1, -1), 0.8, GRID))
    t.setStyle(TableStyle(cmds))
    return t


def main():
    repo, out = locate_paths(Path(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=out / "Report_Template.docx")
    parser.add_argument("--output", type=Path, default=out / "Report_Template.pdf")
    args = parser.parse_args()
    source = args.source.resolve()
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    document = Document(source)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("DocTitle", fontName=FONT_BOLD, fontSize=26, leading=29, textColor=colors.black, spaceAfter=7)
    body_style = ParagraphStyle("DocBody", fontName=FONT, fontSize=10.2, leading=13.2, textColor=BODY, spaceAfter=5, wordWrap="CJK")
    h1_style = ParagraphStyle("H1", fontName=FONT_BOLD, fontSize=18, leading=21, textColor=NAVY, spaceBefore=4, spaceAfter=5)
    h2_style = ParagraphStyle("H2", fontName=FONT_BOLD, fontSize=11.5, leading=14, textColor=TEAL, spaceBefore=5, spaceAfter=3)
    eyebrow_style = ParagraphStyle("Eyebrow", fontName=FONT_BOLD, fontSize=8.5, leading=10, textColor=TEAL, spaceAfter=1)
    caption_style = ParagraphStyle("CaptionSmall", fontName=FONT, fontSize=8.5, leading=10, textColor=MUTED, spaceAfter=3)
    placeholder_style = ParagraphStyle("Placeholder", fontName=FONT, fontSize=9.2, leading=12, textColor=MUTED, italic=True, spaceAfter=5, leftIndent=7, borderColor=GRID, borderWidth=0.45, borderPadding=4)

    story = []
    section_started = False
    for block in blocks(document):
        if isinstance(block, DocxParagraph):
            if has_page_break(block):
                continue
            image_ids = block._p.xpath('.//a:blip/@r:embed')
            if image_ids:
                for rel_id in image_ids:
                    blob = block.part.related_parts[rel_id].blob
                    from PIL import Image as PILImage
                    with PILImage.open(BytesIO(blob)) as image_meta:
                        img_ratio = image_meta.width / image_meta.height
                    target_width = (5.5 if image_meta.width > 2000 else 6.55) * inch
                    img = RLImage(BytesIO(blob), width=target_width, height=4.2 * inch, kind="proportional")
                    img._restrictSize(target_width, 4.2 * inch)
                    img.hAlign = "CENTER"
                    story.append(img)
                    story.append(Spacer(1, 3))
                continue
            text = normalize(block.text.strip())
            if not text:
                continue
            style_name = block.style.name if block.style else "Normal"
            if re.match(r"^\d\d\s+/", text):
                if section_started or text.startswith("01"):
                    story.append(PageBreak())
                section_started = True
                flow = Paragraph(esc(text), eyebrow_style)
            elif style_name == "Title":
                flow = Paragraph(esc(text), title_style)
            elif style_name == "Heading 2":
                flow = Paragraph(esc(text), h2_style)
            elif style_name == "Caption":
                flow = Paragraph(esc(text), caption_style)
            elif text.startswith("[") and any(run.italic for run in block.runs):
                flow = Paragraph(esc(text), placeholder_style)
            elif "ASTANA IT UNIVERSITY" in text:
                flow = Paragraph(esc(text), eyebrow_style)
            else:
                flow = Paragraph(esc(text), body_style)
            story.append(flow)
        else:
            story.append(table_flowable(block, 7.06 * inch))
            story.append(Spacer(1, 4))

    class HeaderDoc(BaseDocTemplate):
        def __init__(self, filename, **kw):
            super().__init__(filename, pagesize=letter, leftMargin=0.72 * inch, rightMargin=0.72 * inch, topMargin=0.68 * inch, bottomMargin=0.65 * inch, **kw)
            frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
            self.addPageTemplates([PageTemplate(id="all", frames=frame, onPage=self.paint_page)])

        def paint_page(self, canvas, doc):
            canvas.saveState()
            canvas.setTitle("Household Object Recognition Assignment 2 Report Template")
            canvas.setAuthor("Alina Kovalenko")
            canvas.setFillColor(TEAL)
            canvas.setFont(FONT_BOLD, 8)
            canvas.drawString(self.leftMargin, letter[1] - 0.42 * inch, "OBJECTLAB  /  COMPUTER VISION")
            canvas.setStrokeColor(GRID)
            canvas.setLineWidth(0.5)
            canvas.line(self.leftMargin, letter[1] - 0.5 * inch, letter[0] - self.rightMargin, letter[1] - 0.5 * inch)
            canvas.setFillColor(MUTED)
            canvas.setFont(FONT, 8)
            canvas.drawRightString(letter[0] - self.rightMargin, 0.38 * inch, f"ASSIGNMENT 2  |  {doc.page}")
            canvas.restoreState()

    HeaderDoc(str(target)).build(story)
    print(target)


if __name__ == "__main__":
    main()


