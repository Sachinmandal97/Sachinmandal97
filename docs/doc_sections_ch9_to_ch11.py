"""
VERITAS-MM 2.0 Academic Project Documentation - Chapters 9 to 11.
Calibrated to exact academic depth and pagination matching SAGE Documentation:
- Chapter 9: End-to-End Research Process (Pages 31 – 32, Figure 9.1, Table 9.1)
- Chapter 10: Modern AI Governance and Policies (Pages 33 – 35, Table 10.1)
- Chapter 11: Technology Domains Mapped (Pages 36 – 37, Table 11.1)
"""

from pathlib import Path
from docx.shared import Inches, Pt
from doc_helpers import (
    add_heading_1, add_heading_2, add_heading_3,
    add_body_p, add_bullet_p, add_figure, create_table
)

def render_chapters_9_to_11(doc, assets_dir):
    """Render Chapters 9 through 11 with lifecycle timelines, governance tables, and tech stacks."""
    
    # =========================================================================
    # CHAPTER 9: END-TO-END RESEARCH PROCESS (Pages 31-32)
    # =========================================================================
    add_heading_1(doc, "9. End-to-End Research Process", page_break_before=True)

    add_heading_2(doc, "9.1 Research Methodology & 9-Phase Lifecycle")
    add_body_p(
        doc,
        "The development and empirical validation of VERITAS-MM 2.0 was executed through a rigorous, nine-phase chronological "
        "research lifecycle adhering to open science and reproducible machine learning principles. The project transitioned "
        "systematically from initial baseline diagnostic audits to large-scale benchmark caching, architectural adapter "
        "optimization, full-spectrum experimental evaluation, and human-in-the-loop application integration.",
        space_after=6
    )

    fig9_1 = assets_dir / "fig9_1_research_lifecycle.png"
    add_figure(doc, fig9_1, "Chronological Nine-Phase Research Lifecycle Timeline of VERITAS-MM 2.0.", "Fig. 9.1", width=Inches(5.4))

    doc.add_page_break()
    add_heading_2(doc, "9.2 Detailed Chronological Phase Execution")
    add_body_p(
        doc,
        "Table 9.1 details the objective, computational methodology, formal deliverables, and verified experimental outcomes "
        "across each phase of the research program:",
        space_after=6
    )

    lifecycle_headers = ["Phase Identifier", "Research Focus & Core Activity", "Formal Deliverable / Milestone", "Key Empirical / Technical Outcome"]
    lifecycle_rows = [
        ["Phase 1: Baseline Diagnostic", "Comprehensive architectural & performance audit of initial repository", "PHASE_1_RESEARCH_ALIGNMENT_REPORT.md", "Identified monolithic conflation & missing out-of-context verification"],
        ["Phase 2: Data Acquisition", "Acquired official NewsCLIPpings (7,264 test) & VLDBench (31,339 records)", "DATASET_ACQUISITION_REPORT.md", "Verified split integrity, image decodes, and absence of data corruption"],
        ["Phase 3: Ingestion & Quality", "Diagnostic audit of multi-modal features, token lengths, and publishers", "VLDBENCH_INGESTION_REPORT.md", "Discovered trailing publisher token leakage in raw news headlines"],
        ["Phase 4: Adapter Design", "Mathematical design of parameter-efficient multimodal fusion adapters", "VLDBENCH_PHASE4_ADAPTER_DESIGN.md", "Selected 1.58M Gated Multimodal Adapter; Late Concatenation as baseline"],
        ["Phase 5: Feature Caching", "Offline feature extraction of DeBERTa-v3 and OpenCLIP representations", "VLDBENCH_PHASE5_FEATURE_CACHE.md", "Extracted 31,339 records; executed headline sanitization regex audit"],
        ["Phase 6: Experiments", "Fast completion training of Gated Adapter, Concatenation, and Baselines", "VLDBENCH_PHASE6_RESULTS.md", "Trained on Protocol A, Protocol B Strict, and Publisher-Held-Out splits"],
        ["Phase 7: Model Selection", "Multi-metric evaluation, cluster-bootstrap CIs, and error analysis", "VLDBENCH_PHASE7_SELECTION.md", "Selected Gated Adapter (Macro-F1 64.54%) outperforming unimodal models"],
        ["Phase 8: UI Integration", "Integrated Tri-Pillar engine into interactive Streamlit application", "VERITAS_MM_PHASE8_REPORT.md", "Preserved SQLite database (288 articles); added telemetry & disclaimers"],
        ["Phase 9: Performance Audit", "Formal audit of selective classification & thesis documentation", "VERITAS_MM_FINAL_RESULTS.md", "Audited 87.44% precision at 10.85% coverage; verified 122/122 test suite"]
    ]
    create_table(doc, lifecycle_headers, lifecycle_rows, [Inches(1.5), Inches(2.2), Inches(2.0), Inches(1.5)])

    add_heading_2(doc, "9.3 Methodological Safeguards Against Data Contamination")
    add_body_p(
        doc,
        "To ensure uncompromising scientific integrity, four strict methodological controls were enforced throughout:",
        space_after=6
    )
    add_bullet_p(doc, "Zero Test-Set Hyperparameter Tuning: All decision boundaries (tau* = 0.28) and margin rejection half-widths (Delta* = 0.07) were tuned exclusively on the official validation splits. The test sets were evaluated exactly once.", bold_prefix="1. Separation of Splits: ")
    add_bullet_p(doc, "Publisher-Aware Group Stratification: In VLDBench Protocol A, news publishers were partitioned using group-aware stratified sampling to ensure no newsroom overlap between training and testing.", bold_prefix="2. Group Stratification: ")
    add_bullet_p(doc, "Protected Baseline Workspace: An untouched reference baseline directory (multimodal-misinfo-verifier) was preserved under read-only locks to verify that code refactorings introduced zero performance regressions.", bold_prefix="3. Protected Baseline: ")
    add_bullet_p(doc, "Automated Regression Harness: Every commit was validated against 122 automated unit and integration tests covering NLI scoring, CLIP geometry, SQLite transactions, and Streamlit components.", bold_prefix="4. Regression Testing: ")

    # =========================================================================
    # CHAPTER 10: MODERN AI GOVERNANCE AND POLICIES (Pages 33-35)
    # =========================================================================
    add_heading_1(doc, "10. Modern AI Governance and Policies", page_break_before=True)

    add_heading_2(doc, "10.1 Responsible AI Principles and Epistemic Integrity")
    add_body_p(
        doc,
        "The automated classification of journalistic content intersects directly with sensitive public discourse, media credibility, "
        "and democratic integrity. Deploying irresponsible or opaque AI models that label news organizations as 'fake' creates severe "
        "reputational harm, chills legitimate public debate, and erodes trust in computational technology. VERITAS-MM 2.0 is built "
        "upon strict Responsible AI governance frameworks governing transparency, epistemic humility, and accountability.",
        space_after=6
    )
    add_body_p(
        doc,
        "In algorithmic governance, the principle of proportionality mandates that automated systems operating in high-stakes informational "
        "ecosystems must not possess unconstrained unilateral authority. VERITAS-MM 2.0 embeds strict algorithmic checks and balances, "
        "ensuring that every automated inference is bound to verifiable, tamper-evident provenance trails and that borderline uncertainty "
        "is explicitly surfaced to human experts rather than concealed beneath forced binary classifications.",
        space_after=10
    )

    add_heading_2(doc, "10.2 Strict Epistemic Non-Claims Framework")
    add_body_p(
        doc,
        "The core governance contribution of VERITAS-MM 2.0 is its formal Epistemic Non-Claims Framework. The system enforces "
        "three inviolable boundaries across all outputs and programmatic interfaces:",
        space_after=6
    )
    add_bullet_p(doc, "Absence of retrieved corroborating evidence must never be presented as proof of falsehood. If external search engines return zero results, the system reports 'Uncertain / Insufficient Evidence'.", bold_prefix="Boundary 1: Epistemic Caution: ")
    add_bullet_p(doc, "An image-text mismatch (cheapfake) must never be presented as proof that the textual narrative is factually untrue. Authentic stories frequently deploy uncalibrated stock photos due to photo desk errors rather than intentional disinformation.", bold_prefix="Boundary 2: Decoupled Modality: ")
    add_bullet_p(doc, "Rhetorical or sensationalist framing must never be presented as proof that the underlying reported event did not occur. Clickbait framing is an editorial style, not evidence of fabrication.", bold_prefix="Boundary 3: Rhetorical Separation: ")

    doc.add_page_break()
    add_heading_2(doc, "10.3 Scientific Distinction: Modality Salience vs. Causal Attribution")
    add_body_p(
        doc,
        "In deep learning literature, attention weights and gating activations are frequently mischaracterized as 'causal explanations' "
        "of model decisions. Extensive empirical critiques (e.g., Jain & Wallace, 2019; Wiegreffe & Pinter, 2019) demonstrate that "
        "attention weights do not correspond to causal feature importance. In VERITAS-MM 2.0, the sigmoid gate outputs of Pillar 3 "
        "(w_T = 38.2%, w_I = 61.8%) are strictly designated as 'Modality Salience Weights'—reflecting internal feature scaling in the "
        "forward pass—and are explicitly disclaimed as causal claims regarding publisher intent.",
        space_after=10
    )

    add_heading_2(doc, "10.4 Auditable Provenance and Source Verification")
    add_body_p(
        doc,
        "Every factual claim verified by Pillar 1 must be accompanied by an immutable cryptographic or URI provenance link. "
        "The framework rejects black-box internal knowledge representations for fact verification. If a factual claim is flagged "
        "as 'Refuted', the interface presents the exact external news wire article, publication timestamp, and cross-encoder "
        "entailment breakdown, allowing journalists to verify the corroborating evidence firsthand before taking editorial action.",
        space_after=10
    )

    doc.add_page_break()
    add_heading_2(doc, "10.5 Human-in-the-Loop Operational Workflows")
    add_body_p(
        doc,
        "Rather than functioning as an autonomous content filter or censor, VERITAS-MM 2.0 is architected as an investigative decision-support "
        "workstation. The system establishes clear escalation paths where low-confidence and borderline predictions are deferred for "
        "manual human review. Editorial personnel retain ultimate authority over public corrections and retraction decisions.",
        space_after=6
    )

    add_heading_2(doc, "10.6 Governance and Ethical Boundary Commitments")
    add_body_p(doc, "Table 10.1 outlines the formal governance commitments implemented in the system:", space_after=6)

    gov_headers = ["Governance Dimension", "Institutional Policy Commitment", "Technical Enforcement Mechanism", "Auditing Strategy"]
    gov_rows = [
        ["Algorithmic Transparency", "Every verdict must provide complete source provenance links", "Evidence Provenance Cards with direct URLs & timestamps", "Independent manual inspection of retrieved URLs"],
        ["Humility in Uncertainty", "System must abstain rather than hallucinate confident errors", "Chow's Reject Option & Uncertain NLI categories", "Rejection coverage auditing across validation splits"],
        ["Non-Conflation of Spin", "Stylistic spin must never be branded as factual fabrication", "Complete architectural decoupling of Pillar 1 and Pillar 3", "Unit tests ensuring distinct non-overlapping outputs"],
        ["Data Privacy & Rights", "Zero storage of sensitive user PII; open-access news indexing", "Local SQLite database restricted to public news articles", "Compliance with CC-BY, MIT, and fair use guidelines"],
        ["Human Primacy", "AI outputs serve as investigative aids, not autonomous censors", "Streamlit UI prominently flags 'Human Editorial Review Required'", "Operational hand-off queues for deferred predictions"]
    ]
    create_table(doc, gov_headers, gov_rows, [Inches(1.5), Inches(2.2), Inches(2.0), Inches(1.5)])

    # =========================================================================
    # CHAPTER 11: TECHNOLOGY DOMAINS MAPPED (Pages 36-37)
    # =========================================================================
    add_heading_1(doc, "11. Technology Domains Mapped", page_break_before=True)

    add_heading_2(doc, "11.1 Layer-by-Layer Technology Ecosystem")
    add_body_p(
        doc,
        "VERITAS-MM 2.0 integrates modern, industry-standard open-source libraries, deep learning frameworks, and architectural "
        "paradigms spanning natural language processing, computer vision, information retrieval, and web application engineering. "
        "Table 11.1 maps every technology component to its primary architectural domain, operational layer, and functional purpose:",
        space_after=6
    )

    tech_headers = ["Technology / Library", "Version / Source", "Primary Functional Purpose", "Architectural Layer"]
    tech_rows_1 = [
        ["Python", "3.10 / 3.11", "Primary execution language and runtime environment", "Core Infrastructure"],
        ["PyTorch", "2.1+ / torchvision", "Deep neural network execution, autograd, and GPU acceleration", "Layer 2 & Layer 3"],
        ["Hugging Face Transformers", "4.35+", "Cross-encoder DeBERTa-v3 model loading and tokenization", "Layer 2 & Layer 3"],
        ["OpenCLIP", "2.24+ (LAION-2B)", "Pre-trained ViT-B/32 vision-language hypersphere embeddings", "Layer 2 & Layer 3"],
        ["spaCy", "3.7+ (en_core_web_sm)", "Linguistic syntactic dependency parsing & claim decomposition", "Layer 2"],
        ["Google News RSS", "Public News API", "Live real-time breaking news evidence retrieval", "Layer 1 & Layer 3"]
    ]
    create_table(doc, tech_headers, tech_rows_1, [Inches(1.5), Inches(1.4), Inches(2.8), Inches(1.5)])

    doc.add_page_break()
    add_body_p(doc, "Table 11.1 (Continued): Technology Domains Mapped Across Persistence, Application, and Testing Layers:", space_after=6)
    tech_rows_2 = [
        ["Wikipedia API", "1.4+", "Authoritative encyclopedic entity background knowledge retrieval", "Layer 1 & Layer 3"],
        ["Rank-BM25", "0.2+", "Local lexical BM25Okapi inverted index search for offline fallback", "Layer 3"],
        ["Streamlit", "1.32+", "Multi-tab interactive reactive web dashboard and workstation UI", "Layer 4"],
        ["SQLite3", "Built-in / Thread-Safe", "ACID-compliant relational persistence for verified article logs", "Layer 4"],
        ["Scikit-Learn", "1.4+", "Metrics computation, ROC-AUC curves, and statistical evaluation", "Evaluation Layer"],
        ["Pytest", "8.0+", "Comprehensive automated unit and regression test harness (122 tests)", "Testing Framework"]
    ]
    create_table(doc, tech_headers, tech_rows_2, [Inches(1.5), Inches(1.4), Inches(2.8), Inches(1.5)])

    add_heading_2(doc, "11.2 Architectural Integration and Hardware Execution")
    add_body_p(
        doc,
        "The system architecture is engineered to run seamlessly across both consumer workstations and high-performance server clusters. "
        "By enforcing a global model registry, neural foundation weights (DeBERTa-v3 435MB, OpenCLIP ViT-B/32 350MB) are loaded into host "
        "RAM or VRAM exactly once, permitting high-throughput sub-second inference without out-of-memory overhead during sustained multi-tab usage.",
        space_after=10
    )
