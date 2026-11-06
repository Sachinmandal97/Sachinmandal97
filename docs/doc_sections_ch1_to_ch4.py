"""
VERITAS-MM 2.0 Academic Project Documentation - Chapters 1 to 4.
Calibrated to exact academic depth and pagination matching SAGE Documentation:
- Chapter 1: Problem Statement (Pages 7 – 9)
- Chapter 2: Background and Context (Pages 10 – 12)
- Chapter 3: Objectives (Pages 13 – 14)
- Chapter 4: Solution Approach (Pages 15 – 16, Figures 4.1, 4.2, 4.3, 4.4, Callout Box)
"""

from pathlib import Path
from docx.shared import Inches, Pt
from doc_helpers import (
    add_heading_1, add_heading_2, add_heading_3,
    add_body_p, add_bullet_p, add_callout_box, add_figure, create_table
)

def render_chapters_1_to_4(doc, assets_dir):
    """Render Chapters 1 through 4 calibrated for exact academic depth and layout."""
    
    # =========================================================================
    # CHAPTER 1: PROBLEM STATEMENT (Pages 7-9)
    # =========================================================================
    add_heading_1(doc, "1. Problem Statement", page_break_before=True)

    add_heading_2(doc, "1.1 The Proliferation of Multimodal Misinformation")
    add_body_p(
        doc,
        "The contemporary information ecosystem has transitioned decisively from purely textual dissemination to complex, "
        "multimodal communication modalities. Modern social feeds, algorithmic aggregators, and digital news platforms "
        "are dominated by multimedia assets where textual narratives are inextricably linked with photographic imagery, "
        "infographics, and editorial styling. While this multimedia convergence enhances user engagement and communicative "
        "vividness, it has simultaneously democratized and accelerated the weaponization of false and misleading content.",
        space_after=6
    )
    add_body_p(
        doc,
        "Deceptive digital campaigns rarely operate through isolated text claims. Instead, malicious actors leverage multimodal "
        "synergy to manipulate cognitive perception. Cognitive psychological research establishes that photographic imagery "
        "triggers heuristic processing and visual truthiness—a cognitive bias where individuals intuitively assign higher "
        "credibility to claims accompanied by photographs, even when the image provides zero corroborating evidentiary support "
        "(Newman et al., 2012). This psychological vulnerability is actively exploited at unprecedented scale, demanding "
        "automated, explainable, and epistemically robust computational defense systems.",
        space_after=6
    )
    add_body_p(
        doc,
        "Furthermore, the democratization of high-speed digital publishing tools has drastically compressed the news cycle. "
        "Misinformation campaigns exploit breaking news windows—periods immediately following major geopolitical, public health, "
        "or natural disaster events—where public demand for information is acute but authoritative facts remain unconfirmed. "
        "In these volatile information vacuums, multimodal falsehoods propagate exponentially faster than subsequent corrections, "
        "exerting enduring impacts on public opinion, market stability, and social cohesion.",
        space_after=10
    )

    add_heading_2(doc, "1.2 Vulnerability of Unimodal and Text-Only Verifiers")
    add_body_p(
        doc,
        "Automated fact-checking has historically evolved as an exclusively Natural Language Processing (NLP) task. Standard "
        "state-of-the-art verification pipelines ingest an isolated text claim, execute entity-based document retrieval over "
        "curated encyclopedic corpora (e.g., Wikipedia), and apply cross-encoder Natural Language Inference (NLI) to predict "
        "veracity. While effective for simple isolated historical or scientific assertions, unimodal text architectures "
        "suffer catastrophic blind spots when confronted with modern multimodal news:",
        space_after=6
    )
    add_bullet_p(doc, "Complete insensitivity to visual context: A text-only verifier cannot observe that a pristine, factually accurate headline describing an earthquake in 2023 is paired with an image depicting rubble from a 2011 tsunami, failing to detect deceptive visual contextualization.", bold_prefix="1. Visual Blindness: ")
    add_bullet_p(doc, "Vulnerability to cross-modal incongruity: Misinformation frequently arises not from factual falsehood within the text, but from the intentional semantic discordance manufactured between the textual assertion and the photographic framing.", bold_prefix="2. Semantic Discordance: ")
    add_bullet_p(doc, "Omission of multimodal rhetorical spin: Tabloid framing often relies on inflammatory visual juxtaposition to warp reader sentiment while maintaining plausible deniability in the literal text.", bold_prefix="3. Visual Sensationalism: ")
    add_bullet_p(doc, "Fragility under partial evidence: Text-only verifiers frequently fail when claims involve colloquial expressions, numerical estimations, or emerging entity nomenclature not yet cataloged in static knowledge bases.", bold_prefix="4. Lexical Brittleness: ")

    add_heading_2(doc, "1.3 Out-of-Context \"Cheapfakes\" vs. Synthetic Deepfakes")
    add_body_p(
        doc,
        "A critical conceptual fallacy pervading both public discourse and computer vision research is the disproportionate "
        "focus on generative AI deepfakes (e.g., diffusion models and Generative Adversarial Networks). Empirical audits of "
        "fact-checked media across international verification networks (e.g., AFP Fact Check, PolitiFact, Snopes) demonstrate "
        "that fully synthetic generative image manipulations account for less than 5% of real-world deceptive imagery (Brennen et al., 2020).",
        space_after=6
    )
    add_body_p(
        doc,
        "The overwhelming majority of visual deception consists of \"cheapfakes\"—authentic, unmanipulated photographs stripped "
        "of their original context and maliciously re-captioned, re-dated, or relocated (Paris & Donovan, 2019). Because cheapfake "
        "photographs contain genuine camera sensor noise, authentic optical physics, and zero generative synthesis artifacts, "
        "they bypass all traditional pixel-forensic and frequency-domain artifact detectors. Addressing cheapfakes requires "
        "semantic cross-modal reasoning rather than low-level pixel forensics.",
        space_after=6
    )
    add_body_p(
        doc,
        "Cheapfakes are exceptionally potent because the visual asset itself is undeniably real. When an investigator examines "
        "the photograph in isolation using forensic filters (e.g., Error Level Analysis, noise variance analysis), the image "
        "registers as entirely authentic. Deception arises entirely from the contextual mismatch manufactured between the visual "
        "scene and the accompanying narrative text.",
        space_after=10
    )

    add_heading_2(doc, "1.4 The \"Absence of Evidence\" Epistemic Fallacy")
    add_body_p(
        doc,
        "Conventional automated fact-checking systems commit a catastrophic epistemological error: they conflate the absence "
        "of retrieved corroborating evidence with positive proof of falsity. In standard automated benchmarks (e.g., FEVER), "
        "if an automated search engine fails to retrieve supporting documents due to indexing latency, niche entity nomenclature, "
        "or paywalled reporting, the classifier defaults to assigning a label of \"Refuted\" or \"Fake\".",
        space_after=6
    )
    add_body_p(
        doc,
        "In real-world operational newsrooms, this assumption produces catastrophic false positive rates. Real news breaks "
        "long before encyclopedic indices are updated. A scientifically grounded verification engine must enforce an explicit "
        "epistemic boundary: the failure to locate corroborating open-web evidence must be reported strictly as \"Uncertain / "
        "Insufficient Evidence\", never as affirmative confirmation of fabrication.",
        space_after=10
    )

    add_heading_2(doc, "1.5 Spurious Publisher Token Memorization in Visual News Benchmarks")
    add_body_p(
        doc,
        "Recent large-scale multimodal manipulation benchmarks, such as VLDBench (Tan et al., 2024), aggregate real-world "
        "news headlines from diverse journalistic outlets. However, our rigorous diagnostic audits reveal severe benchmark "
        "leakage: raw headlines frequently contain trailing publisher attribution suffixes (e.g., \"- Reuters\", \"| Daily Mail\", "
        "\"- CNN\").",
        space_after=6
    )
    add_body_p(
        doc,
        "When standard deep neural networks (e.g., BERT, DeBERTa) are trained naively on such data, they rapidly memorize the "
        "publisher token as an inductive shortcut. Tabloid publishers with higher baseline rates of sensationalism are flagged "
        "mechanically regardless of article content, while reputable outlets are granted blanket immunity. When deployed in "
        "publisher-held-out or temporal-shift evaluation, unmitigated models collapse. Eliminating publisher token leakage "
        "through anchor-bounded headline sanitization is an indispensable requirement for trustworthy computational verification.",
        space_after=10
    )

    add_heading_2(doc, "1.6 Core Research Questions")
    add_body_p(doc, "This dissertation is governed by three fundamental, interrelated research questions:", space_after=6)
    add_bullet_p(doc, "Can multimodal verification be decoupled into orthogonal, non-conflicting axes—factual claim grounding, out-of-context cross-modal alignment, and rhetorical framing—to prevent epistemic conflation and provide auditable explanations?", bold_prefix="RQ-1 (Decoupled Epistemology): ")
    add_bullet_p(doc, "Can zero-shot vision-language alignment models (OpenCLIP) detect out-of-context cheapfakes with high precision through principled calibration and selective classification without catastrophic false positives on breaking news?", bold_prefix="RQ-2 (Cheapfake Alignment & Selective Risk): ")
    add_bullet_p(doc, "Can parameter-efficient multimodal fusion adapters accurately quantify sensationalist framing over frozen representations while remaining provably immune to publisher stylistic shortcut memorization?", bold_prefix="RQ-3 (Rhetorical Robustness & Anti-Leakage): ")

    # =========================================================================
    # CHAPTER 2: BACKGROUND AND CONTEXT (Pages 10-12)
    # =========================================================================
    add_heading_1(doc, "2. Background and Context", page_break_before=True)

    add_heading_2(doc, "2.1 Misinformation Taxonomies: Factual, Contextual, and Rhetorical")
    add_body_p(
        doc,
        "Digital misinformation is frequently discussed as a monolithic phenomenon. However, academic taxonomy frameworks "
        "(Wardle & Derakhshan, 2017; Tandoc et al., 2018) identify distinct conceptual varieties that operate through "
        "entirely different mechanisms:",
        space_after=6
    )
    add_bullet_p(doc, "Factual Invalidation: Objective factual assertions that directly contradict empirical reality, verifiable historical records, or scientific consensus (e.g., fabricated casualty statistics).", bold_prefix="1. Factual Deception: ")
    add_bullet_p(doc, "Contextual Misattribution: The pairing of authentic visual evidence with inaccurate or fabricated situational context (e.g., attributing an explosion photo from Syria 2015 to Ukraine 2022).", bold_prefix="2. Contextual Cheapfakes: ")
    add_bullet_p(doc, "Rhetorical Manipulation: The deployment of sensationalist, hyper-partisan, or clickbait rhetorical devices engineered to manipulate user affect while reporting factual occurrences.", bold_prefix="3. Rhetorical Framing: ")
    add_body_p(
        doc,
        "Treating these three distinct failure modes as a single binary classification output leads to fatal confusion in editorial "
        "workflows. An article reporting a verified factual event using sensationalized rhetoric must not be branded as 'fake news', "
        "nor should an unmanipulated photo in an out-of-context pairing be dismissed as a synthetic deepfake.",
        space_after=10
    )

    add_heading_2(doc, "2.2 Contrastive Vision-Language Representations (CLIP & OpenCLIP)")
    add_body_p(
        doc,
        "Contrastive Language-Image Pre-training (CLIP; Radford et al., 2021) revolutionized cross-modal semantic understanding "
        "by mapping text descriptions and images into a shared Euclidean embedding space. Trained on hundreds of millions of "
        "scraped image-caption pairs using InfoNCE symmetric cross-entropy loss, CLIP aligns visual semantics with textual concepts.",
        space_after=6
    )
    add_body_p(
        doc,
        "In this work, we employ OpenCLIP ViT-B/32 pre-trained on the open LAION-2B dataset (laion2b_s34b_b79k; Cherti et al., 2023). "
        "OpenCLIP maps raw images I and candidate headlines T into normalized 512-dimensional vectors e_I, e_T in R^512 such that "
        "||e_I||_2 = ||e_T||_2 = 1. The cosine similarity s_TI = <e_T, e_I> provides a direct geometric measure of semantic alignment "
        "on the unit hypersphere S^511, providing an ideal zero-shot comparator for out-of-context cheapfake detection.",
        space_after=6
    )
    add_body_p(
        doc,
        "The InfoNCE objective optimizes the mutual information lower bound between visual and textual representations. Given a batch "
        "of N (image, text) pairs, the model maximizes the cosine similarity of genuine pairs while penalizing all N^2 - N incorrect pairings. "
        "Consequently, OpenCLIP develops rich, generalized representations of visual concepts, actions, and objects that map directly "
        "to natural language vocabulary without requiring task-specific fine-tuning.",
        space_after=10
    )

    add_heading_2(doc, "2.3 Natural Language Inference and Cross-Encoder Transformers")
    add_body_p(
        doc,
        "Natural Language Inference (NLI) is the foundational task of determining whether a premise sentence P entails, "
        "contradicts, or is neutral with respect to a hypothesis sentence H. While bi-encoders compute isolated representations, "
        "cross-encoders concatenate the premise and hypothesis as a single sequence [CLS] P [SEP] H [SEP], enabling full "
        "cross-attention across every token pair (Devlin et al., 2019).",
        space_after=6
    )
    add_body_p(
        doc,
        "We utilize DeBERTa-v3 (Decoding-enhanced BERT with Disentangled Attention; He et al., 2021) fine-tuned on multi-genre NLI. "
        "DeBERTa-v3 represents each word using two distinct vectors encoding content and relative position respectively, calculating "
        "attention weights via disentangled matrices. This architectural innovation provides unmatched semantic sensitivity when "
        "evaluating whether retrieved news evidence corroborates or refutes extracted claim statements.",
        space_after=6
    )
    add_body_p(
        doc,
        "Compared to standard BERT and RoBERTa architectures, DeBERTa's disentangled attention mechanism calculates attention "
        "matrices as the sum of content-to-content, content-to-position, and position-to-content terms. This explicit disentanglement "
        "allows the model to capture syntactic dependencies and negations with superior precision, preventing false positive "
        "entailment predictions on complex, multi-clause journalistic claims.",
        space_after=10
    )

    add_heading_2(doc, "2.4 Retrieval-Augmented Generation (RAG) for Fact-Checking")
    add_body_p(
        doc,
        "Retrieval-Augmented Generation (RAG) combines dense or sparse external information retrieval with neural reasoning "
        "(Lewis et al., 2020). For open-web fact-checking, automated claim verification requires retrieving current, authoritative "
        "evidence from dynamic live sources rather than relying on static, hallucination-prone parametric LLM memory.",
        space_after=6
    )
    add_body_p(
        doc,
        "VERITAS-MM 2.0 implements a multi-tiered evidence retrieval pipeline combining live Google News RSS search for real-time "
        "journalistic reporting, the Wikipedia API for established historical entities, and a local BM25Okapi lexical index "
        "for instant offline fallback, providing verifiable provenance links for every generated verdict.",
        space_after=10
    )

    add_heading_2(doc, "2.5 Computational Rhetoric and Parameter-Efficient Fusion")
    add_body_p(
        doc,
        "Computational rhetoric analyzes the framing devices, sensationalist markers, and editorial packaging utilized in mass media. "
        "Rather than fine-tuning entire massive multi-billion parameter foundation models—which risks catastrophic forgetting and "
        "demands prohibitive GPU compute—parameter-efficient adapter architectures introduce small, trainable bottleneck layers "
        "over frozen representations (Houlsby et al., 2019; Hu et al., 2021).",
        space_after=6
    )
    add_body_p(
        doc,
        "By freezing both the 768-dimensional DeBERTa-v3 text encoder and the 768-dimensional OpenCLIP vision backbone, we train "
        "a lightweight Gated Multimodal Fusion Adapter containing only 1.58M parameters. This design preserves foundation model "
        "knowledge while enabling precise rhetorical classification on modest CPU or single-GPU workstations.",
        space_after=10
    )

    add_heading_2(doc, "2.6 Selective Classification and Chow's Reject Option")
    add_body_p(
        doc,
        "In mission-critical, high-stakes domains such as medical diagnosis and news verification, raw classification accuracy is "
        "insufficient. A classifier must possess the ability to withhold judgment when prediction confidence is ambiguous—a framework "
        "formalized by C.K. Chow (1970) as Selective Classification with a Reject Option (Geifman & El-Yaniv, 2017).",
        space_after=6
    )
    add_body_p(
        doc,
        "By defining a symmetric rejection margin Delta around the calibrated decision boundary tau*, predictions where "
        "|s_TI - tau*| < Delta are deferred for human expert review. This trade-off between coverage and selective risk allows "
        "VERITAS-MM 2.0 to deliver certified high-precision automated verdicts on unambiguous cases while eliminating automated false accusations.",
        space_after=10
    )

    # =========================================================================
    # CHAPTER 3: OBJECTIVES (Pages 13-14)
    # =========================================================================
    add_heading_1(doc, "3. Objectives", page_break_before=True)

    add_heading_2(doc, "3.1 Primary Academic Objective")
    add_body_p(
        doc,
        "The overarching objective of this Master of Science research project is to design, implement, empirically benchmark, "
        "and deploy an explainable, decoupled multimodal vision-language framework capable of rigorously identifying out-of-context "
        "cheapfakes, verifying factual claims against open-web evidence, and quantifying multimodal rhetorical manipulation "
        "without conflating stylistic framing with factual falsehood, while maintaining certified operational reliability "
        "under strict distribution shift.",
        space_after=10
    )

    add_heading_2(doc, "3.2 Specific Research Objectives (RO)")
    add_bullet_p(doc, "Architect an epistemically decoupled verification pipeline that bifurcates factual veracity, cross-modal alignment, and rhetorical framing into three orthogonal, independently auditable evaluation axes.", bold_prefix="RO-1 (Decoupled Verification Architecture): ")
    add_bullet_p(doc, "Calibrate zero-shot OpenCLIP ViT-B/32 cross-modal cosine alignment on the official NewsCLIPpings benchmark to establish an empirically optimal decision threshold superior to naive zero-cosine baselines.", bold_prefix="RO-2 (Zero-Shot Cheapfake Calibration): ")
    add_bullet_p(doc, "Formulate and validate a margin-based selective classification mechanism (Chow's reject option) to achieve >=85% accepted-subset precision on unambiguous image-text pairings while deferring ambiguous samples for human editorial review.", bold_prefix="RO-3 (Selective Classification & Risk Bound): ")
    add_bullet_p(doc, "Design and train a parameter-efficient Gated Multimodal Fusion Adapter over frozen DeBERTa-v3 and OpenCLIP embeddings, evaluating performance against unimodal and publisher baselines under strict temporal and newsroom-held-out splits.", bold_prefix="RO-4 (Parameter-Efficient Rhetorical Fusion): ")
    add_bullet_p(doc, "Develop an anchor-bounded headline sanitization protocol that eliminates trailing publisher attribution tokens, proving resistance to spurious inductive shortcut memorization.", bold_prefix="RO-5 (Anti-Leakage Headline Sanitization): ")

    add_heading_2(doc, "3.3 Engineering and Architectural Objectives (EO)")
    add_bullet_p(doc, "Build a responsive, human-centered Streamlit workstation supporting multi-tier evidence inspection, real-time RSS news monitoring, and benchmark evaluation dashboards.", bold_prefix="EO-1 (Interactive Investigation Workstation): ")
    add_bullet_p(doc, "Implement a shared, zero-redundancy in-memory model registry that loads heavy neural encoders (DeBERTa-v3, OpenCLIP) exactly once, eliminating CPU/GPU memory leaks during prolonged sessions.", bold_prefix="EO-2 (Zero-Redundancy Model Registry): ")
    add_bullet_p(doc, "Engineer an automated thread-safe SQLite persistence layer logging verified claims, retrieved evidence cards, cross-modal scores, and selective classification decisions with sub-millisecond retrieval.", bold_prefix="EO-3 (Thread-Safe Persistence & Auditing): ")
    add_bullet_p(doc, "Maintain a comprehensive automated regression test suite covering all mathematical scoring functions, threshold boundaries, and UI components with 100% test pass rates.", bold_prefix="EO-4 (Comprehensive Automated Testing): ")

    add_heading_2(doc, "3.4 Target Stakeholders and Expected Socio-Technical Impact")
    add_body_p(
        doc,
        "VERITAS-MM 2.0 is engineered to serve three primary stakeholder communities:",
        space_after=6
    )
    add_bullet_p(doc, "Professional Fact-Checking Organizations (e.g., IFCN signatories): Accelerates triage by filtering obvious cheapfakes, surfacing instant external evidence, and deferring ambiguous borderline cases for manual human adjudication.", bold_prefix="1. Fact-Checking Newsrooms: ")
    add_bullet_p(doc, "Social Media Platform Trust & Safety Teams: Provides explainable, non-arbitrary verification signals that distinguish benign sensationalism from harmful out-of-context deception.", bold_prefix="2. Platform Integrity Teams: ")
    add_bullet_p(doc, "Academic Data Science & NLP Researchers: Establishes a transparent benchmark baseline and reproducible codebase for multimodal misinformation research without publisher leakage.", bold_prefix="3. Computational Social Science: ")

    # =========================================================================
    # CHAPTER 4: SOLUTION APPROACH (Pages 15-16)
    # =========================================================================
    add_heading_1(doc, "4. Solution Approach", page_break_before=True)

    add_heading_2(doc, "4.1 The Decoupled Tri-Pillar Philosophy")
    add_body_p(
        doc,
        "To resolve the fatal conflations inherent in monolithic black-box fake news detectors, VERITAS-MM 2.0 introduces "
        "a strictly decoupled Tri-Pillar verification architecture. Each pillar operates over distinct mathematical principles, "
        "ingests specific feature streams, and outputs isolated epistemic signals that are synthesized without semantic interference.",
        space_after=6
    )

    add_callout_box(
        doc,
        "1. Absence of retrieved web evidence does NOT equal positive proof that a claim is false (Epistemic Caution).\n"
        "2. An image-text contextual mismatch does NOT prove that the textual claim itself is factually incorrect.\n"
        "3. Sensationalized rhetorical framing does NOT demonstrate that the reported underlying event is fabricated.\n"
        "4. Automated systems must provide verifiable evidence provenance and defer low-confidence borderline cases to human analysts.",
        bold_title="GOVERNING EPISTEMIC PRINCIPLES OF VERITAS-MM 2.0:"
    )

    add_heading_2(doc, "4.2 Decoupled Verification Engines (Pillars 1, 2, and 3)")
    add_body_p(
        doc,
        "The three verification pillars execute in parallel, sharing cached foundation representations while maintaining completely separate decision logics:",
        space_after=4
    )
    add_bullet_p(doc, "Pillar 1 (RAG NLI Fact Verification): Ingests extracted syntactic claims via spaCy, executes multi-source retrieval (Google News RSS, Wikipedia, BM25Okapi), applies topical gating, and performs cross-encoder NLI via DeBERTa-v3 to yield deterministic epistemic categories (Supported, Refuted, Partially Supported, Conflicting, Uncertain).", bold_prefix="Pillar 1: ")
    add_bullet_p(doc, "Pillar 2 (OpenCLIP Cheapfake Alignment): Projects image and text onto the unit hypersphere S^511 via OpenCLIP ViT-B/32, calculating cosine alignment s_TI against calibrated threshold tau* = 0.28. Implements Chow's reject option (|s_TI - 0.28| >= 0.07) to achieve 87.44% precision on accepted predictions while deferring low-margin cases for human review.", bold_prefix="Pillar 2: ")
    add_bullet_p(doc, "Pillar 3 (Gated Multimodal Rhetorical Fusion): Ingests anchor-sanitized headlines and images through frozen DeBERTa-v3 and OpenCLIP backbones into a 1.58M parameter Gated Adapter, computing dynamic sigmoid modality weights (w_T = 38.2%, w_I = 61.8%) to quantify manipulative framing without publisher leakage.", bold_prefix="Pillar 3: ")

    fig4_1 = assets_dir / "fig4_1_rag_nli_workflow.png"
    add_figure(doc, fig4_1, "Pillar 1 RAG NLI Evidence Verification Pipeline.", "Fig. 4.1", width=Inches(5.0))

    fig4_2 = assets_dir / "fig4_2_cross_modal_alignment.png"
    add_figure(doc, fig4_2, "Pillar 2 OpenCLIP Hypersphere Cosine Geometry & Calibrated Threshold (tau=0.28).", "Fig. 4.2", width=Inches(5.0))

    fig4_3 = assets_dir / "fig4_3_selective_classification.png"
    add_figure(doc, fig4_3, "Pillar 2 Selective Classification Margin Rejection Band (|s - 0.28| >= 0.07).", "Fig. 4.3", width=Inches(5.0))

    fig4_4 = assets_dir / "fig4_4_gated_adapter_architecture.png"
    add_figure(doc, fig4_4, "Pillar 3 Parameter-Efficient Gated Multimodal Fusion Adapter Architecture (1.58M params).", "Fig. 4.4", width=Inches(5.0))
