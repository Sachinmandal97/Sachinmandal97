"""
VERITAS-MM 2.0 Academic Project Documentation - Master Document Assembler.
Orchestrates the construction of the formal 50-page Master of Science academic documentation,
integrating all preliminary pages, 16 chapters, 14 high-resolution figures, 9 comprehensive tables,
and university seals matching SAGE Documentation.pdf.
"""

import sys
import os
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

# Add docs dir to path for imports
WORKSPACE = Path(r"C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-submission")
DOCS_DIR = WORKSPACE / "docs"
ASSETS_DIR = DOCS_DIR / "assets"
OUTPUT_DOCX = WORKSPACE / "VERITAS_MM_FINAL_PROJECT_DOCUMENTATION.docx"

sys.path.insert(0, str(DOCS_DIR))

from doc_helpers import (
    COLOR_PRIMARY_NAVY, COLOR_MUTED_TEXT, format_paragraph
)
from doc_sections_prelim import render_preliminary_pages
from doc_sections_ch1_to_ch4 import render_chapters_1_to_4
from doc_sections_ch5_to_ch8 import render_chapters_5_to_8
from doc_sections_ch9_to_ch11 import render_chapters_9_to_11
from doc_sections_ch12_to_ch16 import render_chapters_12_to_16

def add_page_number_to_run(run):
    """Word XML dynamic page number field."""
    fldChar1 = OxmlElement('w:fldChar')
    fldChar1.set(qn('w:fldCharType'), 'begin')
    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = "PAGE"
    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'separate')
    fldChar3 = OxmlElement('w:fldChar')
    fldChar3.set(qn('w:fldCharType'), 'end')
    r = run._r
    r.append(fldChar1)
    r.append(instrText)
    r.append(fldChar2)
    r.append(fldChar3)

def build_veritas_documentation():
    """Build the complete 50-page academic project documentation."""
    print("=" * 70)
    print("VERITAS-MM 2.0: Academic Project Documentation Builder")
    print(f"Workspace: {WORKSPACE}")
    print(f"Output:    {OUTPUT_DOCX}")
    print("=" * 70)

    doc = Document()

    # Configure Page Setup (A4 dimensions with standard academic margins)
    section = doc.sections[0]
    section.page_width = Inches(8.27)   # 210 mm (A4)
    section.page_height = Inches(11.69) # 297 mm (A4)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.25)  # Binding gutter margin
    section.right_margin = Inches(1.0)

    # Enable distinct first page so cover page remains clean
    section.different_first_page_header_footer = True

    # Configure Running Header (Pages 2+)
    header = section.header
    p_head = header.paragraphs[0]
    format_paragraph(p_head, space_before=0, space_after=4, align=WD_ALIGN_PARAGRAPH.RIGHT)
    r_head = p_head.add_run("VERITAS-MM 2.0: Multimodal Misinformation & Cheapfake Verification Framework")
    r_head.font.name = 'Times New Roman'
    r_head.font.size = Pt(8.5)
    r_head.font.italic = True
    r_head.font.color.rgb = COLOR_MUTED_TEXT

    # Configure Running Footer (Pages 2+)
    footer = section.footer
    p_foot = footer.paragraphs[0]
    format_paragraph(p_foot, space_before=4, space_after=0, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_foot_pre = p_foot.add_run("Somaiya Vidyavihar University — M.Sc. Data Science | Page ")
    r_foot_pre.font.name = 'Times New Roman'
    r_foot_pre.font.size = Pt(9)
    r_foot_pre.font.color.rgb = COLOR_MUTED_TEXT
    add_page_number_to_run(p_foot.add_run())

    # Configure Base Normal Style
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(30, 41, 59)

    # 1. Preliminary Pages (Pages 1-6)
    print("\n[1/5] Rendering Preliminary Pages (Cover, Certificate, Declaration, Ack, Index, Use Case Profile)...")
    render_preliminary_pages(doc, ASSETS_DIR)

    # 2. Chapters 1 to 4 (Pages 7-18)
    print("[2/5] Rendering Chapters 1 to 4 (Problem Statement, Background, Objectives, Solution Approach)...")
    render_chapters_1_to_4(doc, ASSETS_DIR)

    # 3. Chapters 5 to 8 (Pages 19-31)
    print("[3/5] Rendering Chapters 5 to 8 (Architecture, Workflow UI, Expected Outputs, Operational Challenges)...")
    render_chapters_5_to_8(doc, ASSETS_DIR)

    # 4. Chapters 9 to 11 (Pages 32-39)
    print("[4/5] Rendering Chapters 9 to 11 (Research Process, AI Governance, Technology Mapping)...")
    render_chapters_9_to_11(doc, ASSETS_DIR)

    # 5. Chapters 12 to 16 & Closing (Pages 40-50)
    print("[5/5] Rendering Chapters 12 to 16 & University Closing (Evaluation, Limitations, Scope, Conclusion, References)...")
    render_chapters_12_to_16(doc, ASSETS_DIR)

    # Save DOCX
    print(f"\nSaving generated document to: {OUTPUT_DOCX}")
    doc.save(str(OUTPUT_DOCX))
    size_mb = OUTPUT_DOCX.stat().st_size / (1024 * 1024)
    print(f"SUCCESS: Document generated successfully! Size: {size_mb:.2f} MB")
    print("=" * 70)

if __name__ == "__main__":
    build_veritas_documentation()
