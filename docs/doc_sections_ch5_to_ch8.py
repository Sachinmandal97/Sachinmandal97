"""
VERITAS-MM 2.0 Academic Project Documentation - Chapters 5 to 8.
Calibrated to exact academic depth and pagination matching SAGE Documentation:
- Chapter 5: Conceptual System Architecture (Pages 17 – 19, Figures 5.1, 5.2, Table 5.1)
- Chapter 6: High-Level Workflow (Pages 20 – 25, Figures 6.1, 6.2, 6.3, 6.4)
- Chapter 7: Expected and Generated Outputs (Pages 26 – 27, Table 7.1)
- Chapter 8: Operational Challenges (Pages 28 – 30, Table 8.1)
"""

from pathlib import Path
from docx.shared import Inches, Pt
from doc_helpers import (
    add_heading_1, add_heading_2, add_heading_3,
    add_body_p, add_bullet_p, add_figure, create_table
)

def render_chapters_5_to_8(doc, assets_dir):
    """Render Chapters 5 through 8 with architecture diagrams, workflow UI, and risk tables."""
    
    # =========================================================================
    # CHAPTER 5: CONCEPTUAL SYSTEM ARCHITECTURE (Pages 17-19)
    # =========================================================================
    add_heading_1(doc, "5. Conceptual System Architecture", page_break_before=True)

    add_heading_2(doc, "5.1 Multi-Layer Conceptual Architecture")
    add_body_p(
        doc,
        "VERITAS-MM 2.0 is architected as an enterprise-grade, four-tier decoupled system engineered for reproducibility, "
        "modularity, and computational efficiency. The architecture decouples data ingestion and neural representation "
        "from verification logic, ensuring that heavy foundation models are instantiated exactly once and shared across downstream tasks.",
        space_after=6
    )
    add_bullet_p(doc, "Handles multi-source article intake via live RSS feeds, direct URL scraping via Newspaper3k/BeautifulSoup4, manual investigator inputs, and benchmark batch loaders (NewsCLIPpings, VLDBench).", bold_prefix="Layer 1: Data Ingestion & News Aggregation: ")
    add_bullet_p(doc, "Executes linguistic syntactic dependency parsing via spaCy, text tokenization, headline sanitization, image tensor decoding, and feature projection via frozen foundation backbones.", bold_prefix="Layer 2: Multi-Modal Preprocessing & Representation: ")
    add_bullet_p(doc, "Houses the decoupled verification engines: Pillar 1 (RAG NLI Fact Verification), Pillar 2 (OpenCLIP Cosine Alignment & Selective Classification), and Pillar 3 (Gated Multimodal Fusion Adapter).", bold_prefix="Layer 3: Decoupled Verification Engine: ")
    add_bullet_p(doc, "Renders the interactive Streamlit dashboard, manages persistent SQLite audit trails, generates structured JSON export payloads, and coordinates human-in-the-loop review queues.", bold_prefix="Layer 4: User Interface, Persistence & Governance: ")

    fig5_1 = assets_dir / "fig5_1_system_architecture.png"
    add_figure(doc, fig5_1, "VERITAS-MM 2.0 Four-Layer Conceptual System Architecture.", "Fig. 5.1", width=Inches(5.4))

    add_heading_2(doc, "5.2 Decoupled Tri-Pillar Verification Engine")
    add_body_p(
        doc,
        "The core innovation of the framework resides in Layer 3, where three independent analytical pipelines execute in parallel. "
        "Unlike end-to-end multi-task networks where gradients from text-image discordance interfere with factual entailment representations, "
        "our decoupled design guarantees complete epistemic isolation.",
        space_after=6
    )

    fig5_2 = assets_dir / "fig5_2_tri_pillar_engine.png"
    add_figure(doc, fig5_2, "Detailed Architecture of the Decoupled Tri-Pillar Verification Engine with Shared Foundation Backbones.", "Fig. 5.2", width=Inches(5.4))

    add_heading_2(doc, "5.3 Component Responsibilities and Invariant Contracts")
    add_body_p(
        doc,
        "To enforce software robustness across hundreds of benchmark runs, each architectural subsystem is bound by strict invariant contracts (Table 5.1):",
        space_after=6
    )

    contract_headers = ["Subsystem Component", "Primary Operational Responsibility", "Mathematical / Architectural Contract", "Failure Fallback Mode"]
    contract_rows = [
        ["Claim Decomposition", "Extract syntactic claims from body text", "spaCy en_core_web_sm dependency parse; filters non-predicates", "Fallback to raw headline sentence"],
        ["Evidence Retriever", "Fetch external corroborating articles", "Google News RSS + Wikipedia API + BM25Okapi lexical index", "Mark 'Insufficient Evidence' if zero match"],
        ["Cross-Encoder NLI", "Evaluate claim-evidence entailment", "DeBERTa-v3 cross-attention; output softmax [Entail, Neut, Contra]", "Assign Uncertain if max probability < 0.55"],
        ["Cheapfake Comparator", "Evaluate image-text cross-modal match", "OpenCLIP ViT-B/32 cosine similarity on unit hypersphere S^511", "Calibrated decision threshold tau* = 0.28"],
        ["Selective Classifier", "Filter low-confidence predictions", "Reject Option: Accept if |s - 0.28| >= 0.07; Defer if |s - 0.28| < 0.07", "Defer for human review (0% auto error)"],
        ["Rhetorical Adapter", "Quantify sensationalist framing", "Gated Fusion Adapter (1.58M params) over frozen embeddings", "Output neutral baseline if visual absent"]
    ]
    create_table(doc, contract_headers, contract_rows, [Inches(1.5), Inches(2.2), Inches(2.2), Inches(1.3)])

    # =========================================================================
    # CHAPTER 6: HIGH-LEVEL WORKFLOW (Pages 20-25)
    # =========================================================================
    add_heading_1(doc, "6. High-Level Workflow", page_break_before=True)

    add_heading_2(doc, "6.1 End-to-End Operational Lifecycle")
    add_body_p(
        doc,
        "The operational lifecycle of an article through VERITAS-MM 2.0 encompasses nine discrete processing stages, "
        "guaranteeing complete reproducibility and transparent traceability from initial intake to final human-assisted adjudication:",
        space_after=6
    )
    add_bullet_p(doc, "Stage 1: Intake & Normalization — Article URL, text headline, body text, and digital image are ingested and normalized.", bold_prefix="1. Intake: ")
    add_bullet_p(doc, "Stage 2: Syntactic Claim Decomposition — Dependency parsing decomposes narrative text into verifiable predicate claims.", bold_prefix="2. Decomposition: ")
    add_bullet_p(doc, "Stage 3: Multi-Source Evidence Retrieval — Google News RSS and Wikipedia are queried in parallel for real-time reporting.", bold_prefix="3. Retrieval: ")
    add_bullet_p(doc, "Stage 4: Cross-Encoder NLI Inference — DeBERTa-v3 assesses entailment against retrieved evidence passages.", bold_prefix="4. Fact NLI: ")
    add_bullet_p(doc, "Stage 5: Cross-Modal Feature Encoding — OpenCLIP ViT-B/32 encodes image and text into 512-d unit hypersphere vectors.", bold_prefix="5. Encoding: ")
    add_bullet_p(doc, "Stage 6: Cheapfake Cosine & Margin Evaluation — Cosine alignment is computed and compared against calibrated margin band.", bold_prefix="6. Alignment: ")
    add_bullet_p(doc, "Stage 7: Rhetorical Gated Fusion — Sanitized text and image embeddings pass through the 1.58M Gated Adapter.", bold_prefix="7. Framing: ")
    add_bullet_p(doc, "Stage 8: Intelligence Synthesis & Audit Logging — Results are persisted to SQLite and formatted for UI rendering.", bold_prefix="8. Synthesis: ")
    add_bullet_p(doc, "Stage 9: Human-in-the-Loop Review — Deferred borderline predictions are queued for manual editorial adjudication.", bold_prefix="9. Editorial Review: ")

    fig6_1 = assets_dir / "fig6_1_end_to_end_workflow.png"
    add_figure(doc, fig6_1, "End-to-End Operational Lifecycle and Dual Automated-Human Adjudication Workflow.", "Fig. 6.1", width=Inches(5.4))

    doc.add_page_break()
    add_heading_2(doc, "6.2 Data Ingestion, Extraction, and Preprocessing Workflow")
    add_body_p(
        doc,
        "The ingestion pipeline accommodates diverse input streams. When an analyst provides an article URL, Newspaper3k extracts the main "
        "body text and high-resolution lead photograph while stripping advertisements and navigation boilerplate. In batch benchmark mode, "
        "custom JSON loaders parse official NewsCLIPpings and VLDBench records, validating image integrity and checksum alignment.",
        space_after=6
    )
    add_body_p(
        doc,
        "Text preprocessing applies spaCy linguistic parsing (`en_core_web_sm`) to decompose multi-paragraph narratives into candidate "
        "factual claims. The parser filters out non-assertive sentences (e.g., rhetorical questions, exclamations, opinions) by verifying "
        "the presence of subject-verb-object predicate triplets. Concurrently, headlines undergo anchor-bounded sanitization to eliminate "
        "trailing publisher tokens before feature caching.",
        space_after=6
    )
    add_body_p(
        doc,
        "Image preprocessing loads JPEG/PNG files via Pillow, converts them to RGB color space, applies center cropping and bicubic "
        "resizing to 224x224 pixels, and normalizes pixel values using standard ImageNet mean and standard deviation tensors. Preprocessed "
        "tensors are cached in memory to ensure sub-millisecond multi-task feature sharing.",
        space_after=10
    )

    doc.add_page_break()
    add_heading_2(doc, "6.3 Claim Investigation Workstation Interface")
    add_body_p(
        doc,
        "The Claim Investigation Workstation (`tab_verify`) represents the primary operational cockpit for investigative journalists. "
        "The interface enables analysts to submit raw text, upload image assets, or provide live URLs. Upon triggering verification, "
        "the UI asynchronously streams progress indicators across all three pillars:",
        space_after=6
    )
    add_bullet_p(doc, "Presents collapsible evidence provenance cards displaying matched external articles, source logos, publication timestamps, and DeBERTa-v3 entailment probabilities [Entailment, Neutral, Contradiction].", bold_prefix="Evidence Provenance Panel: ")
    add_bullet_p(doc, "Visualizes OpenCLIP cosine alignment on an interactive color-coded radial meter, accompanied by a dynamic status badge ('High-Confidence Match', 'High-Confidence Cheapfake', or 'Borderline — Deferred for Human Review').", bold_prefix="Cross-Modal Alignment Gauge: ")
    add_bullet_p(doc, "Displays predicted framing classification ('Objective' vs. 'Sensationalist') alongside modality salience contribution sliders (w_T = 38.2%, w_I = 61.8%) and a prominent epistemic disclaimer.", bold_prefix="Rhetorical Framing Panel: ")

    fig6_2 = assets_dir / "fig6_2_ui_verify_workspace.png"
    add_figure(doc, fig6_2, "Claim Investigation Workstation Interface (tab_verify) displaying Tri-Pillar verification outputs.", "Fig. 6.2", width=Inches(5.4))

    doc.add_page_break()
    add_heading_2(doc, "6.4 Live News Intelligence & Background Processing Pipeline")
    add_body_p(
        doc,
        "The Live News Intelligence Monitor (`tab_live`) operates as an autonomous background sentinel. A dedicated worker thread "
        "polls public RSS feeds from global news organizations (Reuters, Associated Press, BBC World, Al Jazeera) at configurable intervals. "
        "Inbound articles are queued and processed asynchronously through the Tri-Pillar verification engine without blocking UI interactivity.",
        space_after=6
    )
    add_body_p(
        doc,
        "The live view displays real-time telemetry metrics: total articles monitored, average verification latency (ms), distribution "
        "of factual verdicts, and current queue depth. Crucially, articles where cross-modal alignment falls within the rejection margin "
        "(|s - 0.28| < 0.07) are automatically routed into an interactive Human Review Deferral Queue, allowing editors to filter, inspect, "
        "and manually adjudicate ambiguous stories with a single click.",
        space_after=6
    )

    fig6_3 = assets_dir / "fig6_3_ui_live_intelligence.png"
    add_figure(doc, fig6_3, "Live News Intelligence Monitor Interface (tab_live) tracking real-time streams and human review queues.", "Fig. 6.3", width=Inches(5.4))

    doc.add_page_break()
    add_heading_2(doc, "6.5 Empirical Benchmark Evaluation Dashboard")
    add_body_p(
        doc,
        "The Benchmark Evaluation Dashboard (`tab_eval`) provides complete empirical transparency for academic researchers and data science auditors. "
        "The dashboard directly connects to cached benchmark test tensors (NewsCLIPpings test N=7,264; VLDBench Protocol A N=1,094), allowing "
        "users to dynamically explore evaluation metrics and inspect error distributions.",
        space_after=6
    )
    add_bullet_p(doc, "Renders interactive confusion matrices, per-class precision/recall metrics, and statistical significance tests comparing calibrated thresholds (tau=0.28) against naive baselines.", bold_prefix="Benchmark Confusion Matrices: ")
    add_bullet_p(doc, "Features an interactive slider adjusting the rejection half-width Delta in real time, dynamically rendering accuracy-vs-coverage and risk-vs-coverage curves to illustrate the trade-off between coverage and precision.", bold_prefix="Selective Classification Curves: ")
    add_bullet_p(doc, "Provides full breakdown tables comparing Gated Fusion Adapter performance across all four NewsCLIPpings generator splits (Person, Scene, Text-Clip, Semantics).", bold_prefix="Generator Type Breakdowns: ")

    fig6_4 = assets_dir / "fig6_4_ui_benchmark_dashboard.png"
    add_figure(doc, fig6_4, "Empirical Benchmark Evaluation Dashboard (tab_eval) for auditing datasets, curves, and confusion matrices.", "Fig. 6.4", width=Inches(5.4))

    doc.add_page_break()
    add_heading_2(doc, "6.6 Human-in-the-Loop Operational Adjudication Protocol")
    add_body_p(
        doc,
        "The foundational philosophy of VERITAS-MM 2.0 is that computational models should empower, rather than supplant, human editorial "
        "judgment. When selective classification flags an article as borderline (|s - 0.28| < 0.07), the system dispatches the record to "
        "the human adjudication protocol:",
        space_after=6
    )
    add_bullet_p(doc, "The deferred article appears in the newsroom review queue with an amber triage badge, displaying the exact margin distance (e.g., s_TI = 0.264, Delta_dist = -0.016) and highlighting why automated confidence was insufficient.", bold_prefix="1. Triage Alert: ")
    add_bullet_p(doc, "The human analyst is provided one-click access to Google Reverse Image Search, TinEye, and the retrieved external news articles to verify whether the photograph was recirculated from a prior event.", bold_prefix="2. Investigative Deep-Dive: ")
    add_bullet_p(doc, "The editor submits an affirmative judgment ('Confirmed Pristine', 'Confirmed Cheapfake', or 'Benign Stock Illustration'), committing the verified record to the SQLite audit log and clearing the alert.", bold_prefix="3. Adjudication Commit: ")

    # =========================================================================
    # CHAPTER 7: EXPECTED AND GENERATED OUTPUTS (Pages 26-27)
    # =========================================================================
    add_heading_1(doc, "7. Expected and Generated Outputs", page_break_before=True)

    add_heading_2(doc, "7.1 Multi-Dimensional Output Specification")
    add_body_p(
        doc,
        "Traditional fake news tools output an opaque score between 0.0 and 1.0. VERITAS-MM 2.0 rejects this oversimplification, "
        "generating a multi-dimensional diagnostic report across all three decoupled verification axes (Table 7.1). Every generated output "
        "is accompanied by evidentiary provenance, confidence metrics, and explicit action directives.",
        space_after=6
    )

    output_headers = ["Output Dimension", "Underlying Metric / Tensor", "Permitted Category Values", "Operational Interpretation"]
    output_rows = [
        ["Factual Veracity", "DeBERTa-v3 Softmax Entailment [p_e, p_n, p_c]", "Supported, Refuted, Partially Supported, Conflicting, Uncertain", "Determines factual alignment against authoritative external web documents"],
        ["Cross-Modal Match", "OpenCLIP Cosine Similarity s_TI in [-1.0, 1.0]", "Pristine (s >= 0.28) vs. Falsified (s < 0.28)", "Determines if visual image authentically illustrates the described event"],
        ["Selective Rejection", "Margin Distance |s_TI - 0.28| relative to Delta=0.07", "Automated Prediction Accepted vs. Deferred for Human Review", "Guarantees 87.44% precision on accepted cases; defers ambiguous samples"],
        ["Rhetorical Framing", "Gated Fusion Logits over Frozen Representations", "Neutral / Objective vs. Sensationalist / Manipulative", "Quantifies editorial spin without conflating styling with factual falsity"],
        ["Modality Salience", "Sigmoid Gate Activations [w_T, w_I] summing to 1.0", "Text Weight w_T (38.2%) vs. Image Weight w_I (61.8%)", "Reports relative neural feature contribution; NOT causal journalistic intent"]
    ]
    create_table(doc, output_headers, output_rows, [Inches(1.5), Inches(2.0), Inches(1.8), Inches(1.9)])

    add_heading_2(doc, "7.2 Evidence Provenance Cards and Linguistic Breakdown")
    add_body_p(
        doc,
        "For every analyzed claim, Pillar 1 renders interactive Evidence Provenance Cards. Each card displays the retrieved passage, "
        "direct source URL, publishing organization, retrieval timestamp, and DeBERTa-v3 entailment probabilities. Analysts can "
        "inspect the exact linguistic rationale behind every factual verdict, eliminating the opacity of black-box language models.",
        space_after=10
    )

    add_heading_2(doc, "7.3 Cross-Modal Alignment Scores and Rejection Chips")
    add_body_p(
        doc,
        "Pillar 2 generates high-visibility visual status chips in the Streamlit UI: a green 'High-Confidence Pristine' badge for "
        "s_TI >= 0.35, a red 'High-Confidence Cheapfake' badge for s_TI <= 0.21, and an amber 'Borderline — Deferred for Human Review' "
        "chip for s_TI in (0.21, 0.35). This tri-state output ensures editorial staff immediately recognize which cases require human oversight.",
        space_after=10
    )

    add_heading_2(doc, "7.4 Modality Salience Weights vs. Causal Attribution Disclaimers")
    add_body_p(
        doc,
        "To uphold scientific integrity, Pillar 3 outputs a mandatory epistemic disclaimer alongside its modality gate breakdown: "
        "'NOTICE: Modality Salience reflects internal neural adapter feature weighting during classification; it does NOT constitute "
        "a causal claim regarding publisher intent or evidentiary sufficiency.' This safeguards analysts against over-interpreting attention weights.",
        space_after=10
    )

    add_heading_2(doc, "7.5 Structured JSON Export and SQLite Audit Logging Schema")
    add_body_p(
        doc,
        "Every verification session is serialized into a standards-compliant JSON payload and committed to an ACID-compliant "
        "SQLite database (`data/articles.db`). The schema records 22 distinct relational fields including article_id, headline, "
        "image_path, factual_verdict, entailment_prob, clip_similarity, rejection_margin, selective_decision, rhetorical_label, "
        "modality_salience_text, modality_salience_img, and timestamp, enabling longitudinal auditability across 288+ verified records.",
        space_after=10
    )

    # =========================================================================
    # CHAPTER 8: OPERATIONAL CHALLENGES (Pages 28-30)
    # =========================================================================
    add_heading_1(doc, "8. Operational Challenges", page_break_before=True)

    add_heading_2(doc, "8.1 In-the-Wild Real-Time Web Retrieval Latency and API Rate Limits")
    add_body_p(
        doc,
        "Deploying RAG verification against the open web introduces severe operational vulnerabilities. External search endpoints "
        "frequently throttle traffic, impose rate limits, or suffer network timeouts. Furthermore, breaking news events exhibit "
        "indexing delays where authoritative fact-checks do not appear until hours after a viral claim has spread.",
        space_after=6
    )
    add_body_p(
        doc,
        "VERITAS-MM 2.0 addresses this through multi-tiered retrieval fallbacks: if Google News RSS exceeds latency thresholds, "
        "the pipeline cascades to the Wikipedia API and subsequently to a local, pre-indexed BM25Okapi lexical corpus. If all "
        "external sources return zero corroborating passages, the system safely reports 'Insufficient Evidence' rather than failing.",
        space_after=10
    )

    add_heading_2(doc, "8.2 Asymmetric Directional Semantics in Image-Text Alignment")
    add_body_p(
        doc,
        "Image-text cross-modal alignment is inherently asymmetric. An image contains vastly richer spatial, background, and "
        "atmospheric details than a concise 12-word news headline. Conversely, a headline may describe abstract financial metrics "
        "or political treaties that have no direct pictorial representation, accompanied legitimately by a symbolic stock photo.",
        space_after=6
    )
    add_body_p(
        doc,
        "OpenCLIP cosine similarity measures global semantic co-occurrence rather than directional entailment. Consequently, "
        "legitimate symbolic illustrations (e.g., a gavel illustrating a supreme court ruling) yield moderate cosine scores "
        "(0.22 - 0.27) that lie precisely inside the ambiguous boundary zone. Calibrating the margin rejection band (|s - 0.28| < 0.07) "
        "is critical to prevent automated systems from falsely flagging legitimate symbolic stock photography as malicious cheapfakes.",
        space_after=10
    )

    add_heading_2(doc, "8.3 The 'Gen 0' Hard Mismatch Attack (Adversarial Retrieval Inversion)")
    add_body_p(
        doc,
        "In the NewsCLIPpings benchmark, the 'Semantics / Person' generator represents the most challenging cheapfake attack: "
        "adversaries retrieve images depicting the exact same public individual (e.g., a political leader) but from a completely "
        "different historical event. Because OpenCLIP recognizes the high-level semantic facial entity, cosine similarity remains "
        "elevated, causing uncalibrated zero-shot models to drop to near-chance accuracy (58.00% on the Person split).",
        space_after=10
    )

    add_heading_2(doc, "8.4 The High Rejection Rate vs. Operational Capacity Trade-off")
    add_body_p(
        doc,
        "Achieving 87.44% precision via selective classification incurs a steep operational cost: 89.15% of test samples "
        "(N=6,476) fall within the margin band and are deferred for human review. In a resource-constrained newsroom, editorial "
        "staff cannot manually inspect 89% of inbound content. Newsroom managers must actively configure the margin threshold Delta "
        "based on available human staff, balancing automated precision against manual review capacity.",
        space_after=10
    )

    add_heading_2(doc, "8.5 Publisher Name Memorization and Headline Sanitization")
    add_body_p(
        doc,
        "As established in Chapter 1, training adapters on uncurated headlines leads to severe shortcut learning where models "
        "memorize publisher strings. To eliminate this vulnerability, we implemented an anchor-bounded regex sanitization protocol "
        "stripping patterns such as `r'\\s*[-|–—:]\\s*(Reuters|CNN|BBC|Daily Mail|Fox News).*$'`. Audits confirmed zero false "
        "removals across 100 inspected samples, forcing the Gated Adapter to rely genuinely on multimodal semantic framing.",
        space_after=10
    )

    doc.add_page_break()
    add_heading_2(doc, "8.6 Operational Risk-Control Matrix")
    add_body_p(doc, "Table 8.1 summarizes the core operational risks identified in deployment and their corresponding architectural controls:", space_after=6)

    risk_headers = ["Identified Operational Risk", "Potential Real-World Impact", "Architectural Mitigation / Control", "Verification Status"]
    risk_rows = [
        ["Absence of Evidence Fallacy", "Breaking news falsely flagged as fake", "Mandatory 'Uncertain / Insufficient Evidence' verdict", "Verified across 100% of zero-retrieval tests"],
        ["Publisher Token Leakage", "Model memorizes tabloid names as shortcuts", "Anchor-bounded regex headline sanitization", "Audited on 100 random samples; 0% false strip"],
        ["Low-Margin Ambiguity", "Legitimate symbolic photos flagged as cheapfakes", "Chow's Selective Classification Reject Option", "Defers 89.15% ambiguous cases for human review"],
        ["Adversarial Person Mismatch", "Same-entity cheapfakes bypass global CLIP", "DeBERTa-v3 entity grounding & NLI verification", "Pillar 1 catches claim falsehood independently"],
        ["Foundation Memory Bloat", "System crashes due to OOM on shared GPU", "Global in-memory model registry with frozen weights", "Verified: 0 redundant encoder instantiations"]
    ]
    create_table(doc, risk_headers, risk_rows, [Inches(1.5), Inches(2.0), Inches(2.2), Inches(1.5)])
