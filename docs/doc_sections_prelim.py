"""
VERITAS-MM 2.0 Academic Project Documentation - Preliminary Pages.
Renders Pages 1 to 6:
- Page 1: Cover Page
- Page 2: Certificate
- Page 3: Declaration by the Student
- Page 4: Acknowledgement
- Page 5: Index (Table of Contents)
- Page 6: Part A: AI Use Case Profile
"""

from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

from doc_helpers import (
    COLOR_PRIMARY_NAVY, COLOR_SECONDARY_BLUE, COLOR_BODY_TEXT, COLOR_MUTED_TEXT,
    HEX_NAVY_BG, HEX_LIGHT_BG, HEX_ALT_ROW, HEX_BORDER,
    format_paragraph, add_heading_1, add_heading_2, add_heading_3,
    add_body_p, add_bullet_p, create_table, set_cell_background, set_cell_margins
)

def render_preliminary_pages(doc, assets_dir):
    """Render the full preliminary suite matching SAGE Documentation layout."""
    
    # -------------------------------------------------------------------------
    # PAGE 1: COVER PAGE
    # -------------------------------------------------------------------------
    logo_top = assets_dir / "page1_img_0.png"
    if logo_top.exists():
        p_logo = doc.add_paragraph()
        p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        format_paragraph(p_logo, space_before=0, space_after=12)
        p_logo.add_run().add_picture(str(logo_top), width=Inches(3.4))

    # Main Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_title, space_before=12, space_after=8, line_spacing=1.2)
    r_title = p_title.add_run("VERITAS-MM 2.0: EXPLAINABLE MULTIMODAL VISION-LANGUAGE FRAMEWORK FOR AI-GENERATED FAKE NEWS AND MISINFORMATION DETECTION")
    r_title.font.name = 'Times New Roman'
    r_title.font.size = Pt(17)
    r_title.font.bold = True
    r_title.font.color.rgb = COLOR_PRIMARY_NAVY

    # Subtitle
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_sub, space_before=4, space_after=18, line_spacing=1.15)
    r_sub = p_sub.add_run("A Decoupled Tri-Pillar Architecture for Evidence-Grounded Verification, Out-of-Context Cheapfake Detection, and Rhetorical Framing Analysis")
    r_sub.font.name = 'Times New Roman'
    r_sub.font.size = Pt(11.5)
    r_sub.font.italic = True
    r_sub.font.color.rgb = COLOR_MUTED_TEXT

    # Fulfillment statement
    p_subm = doc.add_paragraph()
    p_subm.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_subm, space_before=10, space_after=14)
    r_subm = p_subm.add_run("Project Report Submitted in Partial Fulfilment for the Award of the Degree of\n")
    r_subm.font.name = 'Times New Roman'
    r_subm.font.size = Pt(11)
    r_deg = p_subm.add_run("Master of Science in Data Science")
    r_deg.font.name = 'Times New Roman'
    r_deg.font.size = Pt(13)
    r_deg.font.bold = True
    r_deg.font.color.rgb = COLOR_PRIMARY_NAVY

    # Candidate Block
    p_by = doc.add_paragraph()
    p_by.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_by, space_before=10, space_after=14)
    r_by = p_by.add_run("By\nCandidate for Master of Science in Data Science\nSeat No: 44033025002 | Roll No: 2025-DS-014")
    r_by.font.name = 'Times New Roman'
    r_by.font.size = Pt(11)
    r_by.font.bold = True

    # Department and Year
    p_dept = doc.add_paragraph()
    p_dept.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_dept, space_before=8, space_after=10)
    r_dept = p_dept.add_run("DEPARTMENT OF INFORMATION TECHNOLOGY AND COMPUTER SCIENCE\nACADEMIC YEAR 2026–2027")
    r_dept.font.name = 'Times New Roman'
    r_dept.font.size = Pt(11.5)
    r_dept.font.bold = True
    r_dept.font.color.rgb = COLOR_PRIMARY_NAVY

    # Supervision
    p_sup = doc.add_paragraph()
    p_sup.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_sup, space_before=8, space_after=14)
    r_sup = p_sup.add_run("Under the Supervision of\n")
    r_sup.font.name = 'Times New Roman'
    r_sup.font.size = Pt(11)
    r_guide = p_sup.add_run("Dr. Swati Maurya")
    r_guide.font.name = 'Times New Roman'
    r_guide.font.size = Pt(12)
    r_guide.font.bold = True

    # Bottom Logo and Institution
    logo_bot = assets_dir / "page1_img_1.png"
    if logo_bot.exists():
        p_lbot = doc.add_paragraph()
        p_lbot.alignment = WD_ALIGN_PARAGRAPH.CENTER
        format_paragraph(p_lbot, space_before=8, space_after=6)
        p_lbot.add_run().add_picture(str(logo_bot), width=Inches(1.8))

    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_inst, space_before=2, space_after=0)
    r_inst = p_inst.add_run("Somaiya Vidyavihar University\nVidyavihar (East), Mumbai 400077, India\nwww.somaiya.edu")
    r_inst.font.name = 'Times New Roman'
    r_inst.font.size = Pt(10)
    r_inst.font.color.rgb = COLOR_MUTED_TEXT

    # -------------------------------------------------------------------------
    # PAGE 2: CERTIFICATE
    # -------------------------------------------------------------------------
    doc.add_page_break()
    p_cert_title = doc.add_paragraph()
    p_cert_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_cert_title, space_before=24, space_after=18)
    r_ct = p_cert_title.add_run("Certificate")
    r_ct.font.name = 'Times New Roman'
    r_ct.font.size = Pt(18)
    r_ct.font.bold = True
    r_ct.font.color.rgb = COLOR_PRIMARY_NAVY

    add_body_p(
        doc,
        "This is to certify that the project entitled \"VERITAS-MM 2.0: Explainable Multimodal Vision-Language "
        "Framework for AI-Generated Fake News and Misinformation Detection\" has been satisfactorily completed "
        "by the student of Master of Science in Data Science program (Seat No: 44033025002) as prescribed by the "
        "School of Science and Humanities, Somaiya Vidyavihar University, Mumbai, in partial fulfilment for the award "
        "of the degree of Master of Science in Data Science during the Academic Year 2026–2027.",
        space_after=20
    )

    add_body_p(
        doc,
        "The project demonstrates rigorous scientific methodology, ethical decoupling of multi-modal evidence streams, "
        "and reproducible experimental evaluation on standard benchmarks including NewsCLIPpings and VLDBench, "
        "adhering strictly to university academic integrity standards.",
        space_after=36
    )

    # Signature blocks table
    sig_table = doc.add_table(rows=2, cols=3)
    sig_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r in sig_table.rows:
        for c in r.cells:
            set_cell_margins(c, top=80, bottom=80, left=80, right=80)

    sig_titles = [
        "_____________________\nSignature of Project Guide\nDr. Swati Maurya",
        "_____________________\nSignature of Head of Department\nDept of IT & CS",
        "_____________________\nName & Sign of Examiner\nExternal Examiner"
    ]
    for idx, text in enumerate(sig_titles):
        p = sig_table.rows[0].cells[idx].paragraphs[0]
        format_paragraph(p, space_before=0, space_after=4, align=WD_ALIGN_PARAGRAPH.CENTER)
        r = p.add_run(text)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(10)
        r.font.bold = True

    p_dt = sig_table.rows[1].cells[0].paragraphs[0]
    format_paragraph(p_dt, space_before=16, space_after=0, align=WD_ALIGN_PARAGRAPH.LEFT)
    r_dt = p_dt.add_run("Date: November 02, 2026\nPlace: Mumbai, India")
    r_dt.font.name = 'Times New Roman'
    r_dt.font.size = Pt(10)

    # Departmental Seal in right cell
    seal_img = assets_dir / "page2_seal_0.png"
    if seal_img.exists():
        p_seal = sig_table.rows[1].cells[2].paragraphs[0]
        p_seal.alignment = WD_ALIGN_PARAGRAPH.CENTER
        format_paragraph(p_seal, space_before=8, space_after=0)
        p_seal.add_run().add_picture(str(seal_img), width=Inches(1.5))
        r_sl = p_seal.add_run("\nDepartmental Seal")
        r_sl.font.name = 'Times New Roman'
        r_sl.font.size = Pt(9)
        r_sl.font.italic = True
        r_sl.font.color.rgb = COLOR_MUTED_TEXT

    # -------------------------------------------------------------------------
    # PAGE 3: DECLARATION BY THE STUDENT
    # -------------------------------------------------------------------------
    doc.add_page_break()
    p_dec_title = doc.add_paragraph()
    p_dec_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_dec_title, space_before=24, space_after=18)
    r_dt = p_dec_title.add_run("Declaration by the Student")
    r_dt.font.name = 'Times New Roman'
    r_dt.font.size = Pt(18)
    r_dt.font.bold = True
    r_dt.font.color.rgb = COLOR_PRIMARY_NAVY

    add_body_p(doc, "I hereby certify and declare that:", space_after=12)

    add_bullet_p(doc, "The work contained in this Project Report is original and has been completed by myself under the direct academic supervision of Dr. Swati Maurya, Department of Information Technology and Computer Science, Somaiya Vidyavihar University.", bold_prefix="a) ")
    add_bullet_p(doc, "The work has not been submitted previously to any other university, institute, or examining body for the award of any degree, diploma, fellowship, or professional associate-ship.", bold_prefix="b) ")
    add_bullet_p(doc, "I have strictly conformed to the academic norms, ethical codes of conduct, and scholarly research guidelines established by Somaiya Vidyavihar University.", bold_prefix="c) ")
    add_bullet_p(doc, "Whenever materials (source code, algorithmic architectures, pre-trained weights, benchmark datasets, empirical evaluation protocols, and academic literature) have been drawn from external sources, they have been fully and properly acknowledged through rigorous citations in the text and comprehensive bibliographical listings in Chapter 16.", bold_prefix="d) ")
    add_bullet_p(doc, "All benchmark evaluation metrics, confidence intervals, rejection coverage curves, and software tests reported herein represent genuine, unmanipulated experimental executions without synthetic fabrication.", bold_prefix="e) ")

    p_sig_stud = doc.add_paragraph()
    format_paragraph(p_sig_stud, space_before=40, space_after=4, align=WD_ALIGN_PARAGRAPH.RIGHT)
    r_ss = p_sig_stud.add_run("_____________________________________\nSignature of the Student\nCandidate for M.Sc. Data Science\nSeat No: 44033025002\nDate: November 02, 2026")
    r_ss.font.name = 'Times New Roman'
    r_ss.font.size = Pt(10.5)
    r_ss.font.bold = True

    # -------------------------------------------------------------------------
    # PAGE 4: ACKNOWLEDGEMENT
    # -------------------------------------------------------------------------
    doc.add_page_break()
    p_ack_title = doc.add_paragraph()
    p_ack_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_ack_title, space_before=24, space_after=18)
    r_at = p_ack_title.add_run("Acknowledgement")
    r_at.font.name = 'Times New Roman'
    r_at.font.size = Pt(18)
    r_at.font.bold = True
    r_at.font.color.rgb = COLOR_PRIMARY_NAVY

    add_body_p(
        doc,
        "The completion of this Master of Science dissertation represents the culmination of intensive research "
        "at the intersection of multimodal representation learning, automated fact-checking, and responsible AI governance. "
        "I express my deepest gratitude to all individuals and institutions whose invaluable guidance made this work possible.",
        space_after=10
    )

    add_body_p(
        doc,
        "First and foremost, I express my profound gratitude to my project supervisor, Dr. Swati Maurya, for her exceptional "
        "intellectual stewardship, meticulous feedback, and unwavering encouragement throughout every phase of this research. "
        "Her rigorous commitment to methodological transparency, refusal to accept statistical conflations, and guidance on "
        "epistemic decoupling served as the foundational bedrock of this project.",
        space_after=10
    )

    add_body_p(
        doc,
        "I extend my sincere thanks to the Head of the Department and the esteemed faculty members of the Department of "
        "Information Technology and Computer Science, Somaiya Vidyavihar University, for fostering a demanding yet supportive "
        "academic environment and providing the computational infrastructure required to conduct large-scale vision-language benchmarking.",
        space_after=10
    )

    add_body_p(
        doc,
        "I am profoundly indebted to the open-source artificial intelligence and data science communities. This research "
        "builds upon open benchmarks, architectures, and libraries provided by Hugging Face (Transformers), OpenCLIP (LAION), "
        "spaCy (Explosion AI), Streamlit, and PyTorch. Special appreciation is extended to the authors of the NewsCLIPpings "
        "benchmark and the VLDBench consortium for making their curated datasets available to the scientific community.",
        space_after=10
    )

    add_body_p(
        doc,
        "Finally, I extend my heartfelt appreciation to my family, peers, and fellow researchers whose patience, moral support, "
        "and critical perspectives sustained me throughout the rigorous demands of this graduate dissertation.",
        space_after=16
    )

    # -------------------------------------------------------------------------
    # PAGE 5: INDEX (TABLE OF CONTENTS)
    # -------------------------------------------------------------------------
    doc.add_page_break()
    p_idx_title = doc.add_paragraph()
    p_idx_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_idx_title, space_before=20, space_after=14)
    r_it = p_idx_title.add_run("Index")
    r_it.font.name = 'Times New Roman'
    r_it.font.size = Pt(18)
    r_it.font.bold = True
    r_it.font.color.rgb = COLOR_PRIMARY_NAVY

    index_headers = ["Sr. No.", "Contents", "Page No."]
    index_rows = [
        ["—", "Part A: AI Use Case Profile", "6"],
        ["1", "Problem Statement", "7 – 9"],
        ["2", "Background and Context", "10 – 12"],
        ["3", "Objectives", "13 – 14"],
        ["4", "Solution Approach", "15 – 16"],
        ["5", "Conceptual System Architecture", "17 – 19"],
        ["6", "High-Level Workflow", "20 – 25"],
        ["7", "Expected and Generated Outputs", "26 – 27"],
        ["8", "Operational Challenges", "28 – 30"],
        ["9", "End-to-End Research Process", "31 – 32"],
        ["10", "Modern AI Governance and Policies", "33 – 35"],
        ["11", "Technology Domains Mapped", "36 – 37"],
        ["12", "Evaluation Methodology and Experimental Results", "38 – 42"],
        ["13", "Limitations", "43 – 44"],
        ["14", "Future Scope", "45 – 46"],
        ["15", "Conclusion", "47"],
        ["16", "References", "48"],
        ["—", "Turnitin Originality Report", "49"],
        ["—", "University Closing Details", "50"]
    ]
    create_table(doc, index_headers, index_rows, [Inches(1.0), Inches(4.5), Inches(1.2)])

    # -------------------------------------------------------------------------
    # PAGE 6: PART A: AI USE CASE PROFILE
    # -------------------------------------------------------------------------
    doc.add_page_break()
    p_uc_h = doc.add_paragraph()
    format_paragraph(p_uc_h, space_before=16, space_after=4, align=WD_ALIGN_PARAGRAPH.LEFT)
    r_uc_main = p_uc_h.add_run("VERITAS-MM 2.0: MULTIMODAL MISINFORMATION & CHEAPFAKE VERIFIER")
    r_uc_main.font.name = 'Times New Roman'
    r_uc_main.font.size = Pt(13)
    r_uc_main.font.bold = True
    r_uc_main.font.color.rgb = COLOR_PRIMARY_NAVY

    p_uc_title = doc.add_paragraph()
    format_paragraph(p_uc_title, space_before=2, space_after=12, align=WD_ALIGN_PARAGRAPH.LEFT)
    r_uct = p_uc_title.add_run("Part A: AI Use Case Profile")
    r_uct.font.name = 'Times New Roman'
    r_uct.font.size = Pt(16)
    r_uct.font.bold = True
    r_uct.font.color.rgb = COLOR_SECONDARY_BLUE

    profile_headers = ["Attribute Dimension", "Specification Details"]
    profile_rows = [
        ["Use Case Title", "VERITAS-MM 2.0: Explainable Multimodal Vision-Language Framework for AI-Generated Fake News and Misinformation Detection"],
        ["AI Application Vertical", "Applied AI, Multimodal Representation Learning, NLP, Computer Vision, Digital Forensics, Automated Fact-Checking"],
        ["Collaborators & Guidance", "Department of Information Technology & Computer Science, Somaiya Vidyavihar University (Supervisor: Dr. Swati Maurya)"],
        ["Target Beneficiaries", "Journalistic Newsrooms, Independent Fact-Checking Consortia, OSINT Intelligence Analysts, Platform Integrity Teams"],
        ["Data Modalities", "Bimodal & Multimodal: Raw Headline Text, Photographic Imagery, Entity Knowledge Graphs, External Web Evidence"],
        ["Core Model Architectures", "1. Natural Language Inference: Fine-Tuned DeBERTa-v3 Cross-Encoder (mdeberta-v3-base)\n2. Vision-Language Semantic Matching: OpenCLIP ViT-B/32 (laion2b_s34b_b79k)\n3. Rhetorical Framing Classifier: Parameter-Efficient Gated Multimodal Fusion Adapter (1.58M parameters)"],
        ["Epistemic Methodology", "Strict Decoupled Tri-Pillar Architecture: Orthogonal bifurcation of Factual Verification (Pillar 1), Cross-Modal Cheapfake Mismatch (Pillar 2), and Rhetorical Packaging (Pillar 3)"],
        ["Selective Rejection", "Chow's Selective Classification Reject Option with Validation-Tuned Margin (|s - 0.28| >= 0.07); defers ambiguous cases for human review"],
        ["Compute & Hardware", "Local CPU & GPU Execution (Apple Silicon MPS / Intel Core i7 / NVIDIA CUDA); Zero-redundancy frozen feature caching"],
        ["Operational Governance", "Strict Non-Claims Protocol: Never asserts 'False' due to search absence; Never claims rhetorical spin proves factual fabrication"]
    ]
    create_table(doc, profile_headers, profile_rows, [Inches(2.2), Inches(4.5)])
