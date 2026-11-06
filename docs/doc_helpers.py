"""
VERITAS-MM 2.0 Academic Project Documentation - Helper Primitives and Styling.
Provides formatting, typography, table styling, and figure embedding utilities
matching the Somaiya Vidyavihar University M.Sc. Data Science academic layout.
"""

from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

# Color Palette (Somaiya Academic Identity & Modern Publication Standards)
COLOR_PRIMARY_NAVY = RGBColor(30, 58, 138)    # #1E3A8A - Headings & Primary Accents
COLOR_SECONDARY_BLUE = RGBColor(37, 99, 235)  # #2563EB - Subheadings
COLOR_BODY_TEXT = RGBColor(30, 41, 59)        # #1E293B - Slate 800 Body Text
COLOR_MUTED_TEXT = RGBColor(100, 116, 139)    # #64748B - Slate 500 Captions & Footers
HEX_NAVY_BG = "1E3A8A"
HEX_LIGHT_BG = "F8FAFC"
HEX_ALT_ROW = "F1F5F9"
HEX_BORDER = "CBD5E1"
HEX_CALLOUT_BORDER = "2563EB"
HEX_CALLOUT_BG = "EFF6FF"

def set_cell_background(cell, fill_hex):
    """Set background color of a table cell."""
    tcPr = cell._element.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    """Set inner cell padding in dxa (1 pt = 20 dxa)."""
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for margin_name, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{margin_name}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table, color="CBD5E1", sz="4", val="single"):
    """Set clean thin borders on a table."""
    tblPr = table._element.xpath('w:tblPr')
    if tblPr:
        borders = OxmlElement('w:tblBorders')
        for border_name in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
            border = OxmlElement(f'w:{border_name}')
            border.set(qn('w:val'), val)
            border.set(qn('w:sz'), sz)
            border.set(qn('w:space'), '0')
            border.set(qn('w:color'), color)
            borders.append(border)
        tblPr[0].append(borders)

def format_paragraph(p, space_before=0, space_after=6, line_spacing=1.15, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    """Format paragraph spacing and alignment."""
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    p.alignment = align

def add_heading_1(doc, text, page_break_before=True):
    """Add formal Chapter Heading matching SAGE style."""
    if page_break_before:
        doc.add_page_break()
    h = doc.add_paragraph()
    format_paragraph(h, space_before=14, space_after=10, line_spacing=1.15, align=WD_ALIGN_PARAGRAPH.LEFT)
    run = h.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(16)
    run.font.bold = True
    run.font.color.rgb = COLOR_PRIMARY_NAVY
    return h

def add_heading_2(doc, text):
    """Add formal Section Heading matching SAGE style."""
    h = doc.add_paragraph()
    format_paragraph(h, space_before=11, space_after=6, line_spacing=1.15, align=WD_ALIGN_PARAGRAPH.LEFT)
    run = h.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = COLOR_SECONDARY_BLUE
    return h

def add_heading_3(doc, text):
    """Add formal Subsection Heading matching SAGE style."""
    h = doc.add_paragraph()
    format_paragraph(h, space_before=8, space_after=4, line_spacing=1.15, align=WD_ALIGN_PARAGRAPH.LEFT)
    run = h.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11.5)
    run.font.bold = True
    run.font.color.rgb = COLOR_BODY_TEXT
    return h

def add_body_p(doc, text, bold_prefix=None, space_after=6):
    """Add formal body paragraph in Times New Roman 11pt."""
    p = doc.add_paragraph()
    format_paragraph(p, space_before=0, space_after=space_after, line_spacing=1.15, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = 'Times New Roman'
        r_pre.font.size = Pt(11)
        r_pre.font.bold = True
        r_pre.font.color.rgb = COLOR_BODY_TEXT
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)
    run.font.color.rgb = COLOR_BODY_TEXT
    return p

def add_bullet_p(doc, text, bold_prefix=None):
    """Add formal bullet list item."""
    p = doc.add_paragraph(style='List Bullet')
    format_paragraph(p, space_before=1, space_after=3, line_spacing=1.15, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = 'Times New Roman'
        r_pre.font.size = Pt(11)
        r_pre.font.bold = True
        r_pre.font.color.rgb = COLOR_BODY_TEXT
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)
    run.font.color.rgb = COLOR_BODY_TEXT
    return p

def add_callout_box(doc, text, bold_title="GOVERNING EPISTEMIC PRINCIPLE:"):
    """Add academic callout box with colored left border."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, HEX_CALLOUT_BG)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    # Left border only
    tcPr = cell._element.get_or_add_tcPr()
    borders = OxmlElement('w:tcBorders')
    left_b = OxmlElement('w:left')
    left_b.set(qn('w:val'), 'single')
    left_b.set(qn('w:sz'), '24') # 3pt
    left_b.set(qn('w:space'), '0')
    left_b.set(qn('w:color'), HEX_CALLOUT_BORDER)
    borders.append(left_b)
    for b_name in ['top', 'bottom', 'right']:
        b = OxmlElement(f'w:{b_name}')
        b.set(qn('w:val'), 'none')
        borders.append(b)
    tcPr.append(borders)

    p = cell.paragraphs[0]
    format_paragraph(p, space_before=2, space_after=2, line_spacing=1.15, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    r1 = p.add_run(bold_title + " ")
    r1.font.name = 'Times New Roman'
    r1.font.size = Pt(10.5)
    r1.font.bold = True
    r1.font.color.rgb = COLOR_PRIMARY_NAVY
    r2 = p.add_run(text)
    r2.font.name = 'Times New Roman'
    r2.font.size = Pt(10.5)
    r2.font.color.rgb = COLOR_BODY_TEXT

    p_sp = doc.add_paragraph()
    format_paragraph(p_sp, space_before=0, space_after=4)

def add_figure(doc, img_path, caption, fig_num_str, width=Inches(6.2)):
    """Embed high-resolution figure with formal caption."""
    if not Path(img_path).exists():
        print(f"WARNING: Image not found: {img_path}")
        return
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(4)
    run = p_img.add_run()
    run.add_picture(str(img_path), width=width)

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(10)
    r_fig = p_cap.add_run(f"{fig_num_str}: ")
    r_fig.font.name = 'Times New Roman'
    r_fig.font.size = Pt(9.5)
    r_fig.font.bold = True
    r_fig.font.color.rgb = COLOR_BODY_TEXT
    r_txt = p_cap.add_run(caption)
    r_txt.font.name = 'Times New Roman'
    r_txt.font.size = Pt(9.5)
    r_txt.font.italic = True
    r_txt.font.color.rgb = COLOR_MUTED_TEXT

def create_table(doc, headers, rows, col_widths=None):
    """Create professionally formatted academic table."""
    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table, color=HEX_BORDER, sz="4")

    # Header Row
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        set_cell_background(hdr_cells[i], HEX_NAVY_BG)
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=140, right=140)
        p = hdr_cells[i].paragraphs[0]
        format_paragraph(p, space_before=2, space_after=2, line_spacing=1.0, align=WD_ALIGN_PARAGRAPH.CENTER)
        for r in p.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(9.5)
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)

    # Data Rows
    for r_idx, row in enumerate(rows):
        row_cells = table.rows[r_idx + 1].cells
        bg_hex = HEX_ALT_ROW if (r_idx % 2 == 1) else "FFFFFF"
        for c_idx, val in enumerate(row):
            row_cells[c_idx].text = str(val)
            set_cell_background(row_cells[c_idx], bg_hex)
            set_cell_margins(row_cells[c_idx], top=90, bottom=90, left=120, right=120)
            p = row_cells[c_idx].paragraphs[0]
            align = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 or len(str(val)) > 20 else WD_ALIGN_PARAGRAPH.CENTER
            format_paragraph(p, space_before=1, space_after=1, line_spacing=1.05, align=align)
            for r in p.runs:
                r.font.name = 'Times New Roman'
                r.font.size = Pt(9)
                r.font.color.rgb = COLOR_BODY_TEXT

    # Set column widths if provided
    if col_widths:
        for r in table.rows:
            for c_idx, w in enumerate(col_widths):
                r.cells[c_idx].width = w

    p_after = doc.add_paragraph()
    format_paragraph(p_after, space_before=0, space_after=6)
    return table
