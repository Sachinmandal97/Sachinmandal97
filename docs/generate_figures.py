"""
VERITAS-MM 2.0 Academic Project Documentation Figure Generator.
Generates all 14 publication-quality figures and empirical charts at 300 DPI.
"""

import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Arrow
import numpy as np

# Configure matplotlib for academic publication styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

out_dir = Path(r"C:\Users\94in\.gemini\antigravity\scratch\multimodal-misinfo-submission\docs\assets")
out_dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# FIG 5.1: High-Level Conceptual System Architecture
# ─────────────────────────────────────────────────────────────────────────────
def generate_fig5_1():
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Colors
    c_ui = "#1E293B"       # Slate 800
    c_layer1 = "#2563EB"   # Blue
    c_layer2 = "#0D9488"   # Teal
    c_layer3 = "#D97706"   # Amber
    c_layer4 = "#4F46E5"   # Indigo
    c_card_bg = "#F8FAFC"  # Slate 50
    c_border = "#CBD5E1"   # Slate 300

    # Layer 1: User Interface Layer
    p1 = FancyBboxPatch((5, 78), 90, 18, boxstyle="round,pad=0.5", ec=c_layer1, fc=c_card_bg, lw=1.5)
    ax.add_patch(p1)
    ax.text(7, 92, "1. USER INTERFACE LAYER (Streamlit Multi-Tab Web Application)", fontsize=11, fontweight='bold', color=c_layer1)
    tabs = ["Live News Intelligence\n(tab_live: RSS & GDELT)", "Claim Verification\n(tab_verify: Tri-Pillar)", "Intelligence Store\n(tab_evidence: SQLite)", "System Architecture\n(tab_arch: Diagrams)", "Benchmark Evaluation\n(tab_eval: Metrics)"]
    for i, t in enumerate(tabs):
        bx = 7 + i * 17.5
        box = FancyBboxPatch((bx, 80), 16.5, 9.5, boxstyle="round,pad=0.3", ec=c_border, fc="#EFF6FF", lw=1)
        ax.add_patch(box)
        ax.text(bx + 8.25, 84.75, t, fontsize=7.5, ha='center', va='center', color="#1E3A8A", weight='semibold')

    # Layer 2: Orchestration & Service Layer
    p2 = FancyBboxPatch((5, 54), 90, 18, boxstyle="round,pad=0.5", ec=c_layer2, fc=c_card_bg, lw=1.5)
    ax.add_patch(p2)
    ax.text(7, 68, "2. APPLICATION ORCHESTRATION & SERVICE LAYER", fontsize=11, fontweight='bold', color=c_layer2)
    services = [
        "Pipeline Service\n(src/pipeline.py)",
        "Claim Decomposition\n(src/decomposition.py)",
        "Evidence Retriever\n(src/retrieval.py)",
        "Verification Engine\n(src/verification.py)",
        "Background Scheduler\n(src/scheduler.py)"
    ]
    for i, s in enumerate(services):
        bx = 7 + i * 17.5
        box = FancyBboxPatch((bx, 56), 16.5, 9.5, boxstyle="round,pad=0.3", ec=c_border, fc="#F0FDFA", lw=1)
        ax.add_patch(box)
        ax.text(bx + 8.25, 60.75, s, fontsize=7.5, ha='center', va='center', color="#115E59", weight='semibold')

    # Layer 3: Tri-Pillar Decoupled Verification Engine
    p3 = FancyBboxPatch((5, 27), 90, 22, boxstyle="round,pad=0.5", ec=c_layer3, fc=c_card_bg, lw=1.5)
    ax.add_patch(p3)
    ax.text(7, 45, "3. TRI-PILLAR DECOUPLED MULTIMODAL VERIFICATION ENGINE", fontsize=11, fontweight='bold', color=c_layer3)
    
    # Pillar 1
    box_p1 = FancyBboxPatch((7, 29), 28, 13.5, boxstyle="round,pad=0.3", ec="#F59E0B", fc="#FFFBEB", lw=1.2)
    ax.add_patch(box_p1)
    ax.text(21, 39, "PILLAR 1: Fact Verification", fontsize=8.5, fontweight='bold', ha='center', color="#92400E")
    ax.text(21, 33.5, "spaCy Dependency Clauses\nGoogle News RSS + Wikipedia + BM25\nDeBERTa-v3 Cross-Encoder NLI", fontsize=7, ha='center', color="#78350F")

    # Pillar 2
    box_p2 = FancyBboxPatch((36, 29), 28, 13.5, boxstyle="round,pad=0.3", ec="#10B981", fc="#ECFDF5", lw=1.2)
    ax.add_patch(box_p2)
    ax.text(50, 39, "PILLAR 2: Cheapfake Detection", fontsize=8.5, fontweight='bold', ha='center', color="#065F46")
    ax.text(50, 33.5, "OpenCLIP ViT-B/32 Encoders\nUnit-Hypersphere Cosine Alignment\nSelective Margin Band (|s - 0.28| >= 0.07)", fontsize=7, ha='center', color="#064E3B")

    # Pillar 3
    box_p3 = FancyBboxPatch((65, 29), 28, 13.5, boxstyle="round,pad=0.3", ec="#8B5CF6", fc="#F5F3FF", lw=1.2)
    ax.add_patch(box_p3)
    ax.text(79, 39, "PILLAR 3: Rhetorical Framing", fontsize=8.5, fontweight='bold', ha='center', color="#5B21B6")
    ax.text(79, 33.5, "Anchor-Bounded Headline Sanitization\nGated Multimodal Fusion Adapter (1.58M)\nSigmoid Modality Salience Gate", fontsize=7, ha='center', color="#4C1D95")

    # Layer 4: Foundation Encoders & Data Persistence
    p4 = FancyBboxPatch((5, 4), 90, 18, boxstyle="round,pad=0.5", ec=c_layer4, fc=c_card_bg, lw=1.5)
    ax.add_patch(p4)
    ax.text(7, 18, "4. FOUNDATION ENCODERS, PERSISTENCE & DATA LAYER", fontsize=11, fontweight='bold', color=c_layer4)
    data_items = [
        "DeBERTa-v3 NLI\n(MoritzLaurer/...)\nIn-Memory Singleton",
        "OpenCLIP ViT-B/32\n(openai/clip-...)\nIn-Memory Singleton",
        "SQLite Persistence\n(data/news_intelligence.db\nWAL Mode, 325 Records)",
        "Evidence Corpus\n(evidence_corpus.jsonl\nBM25 Index, 6 Docs)",
        "Live Ingestion Feeds\n(BBC, Reuters, Guardian,\nHindu, GDELT 2.0)"
    ]
    for i, d in enumerate(data_items):
        bx = 7 + i * 17.5
        box = FancyBboxPatch((bx, 6), 16.5, 9.5, boxstyle="round,pad=0.3", ec=c_border, fc="#EEF2FF", lw=1)
        ax.add_patch(box)
        ax.text(bx + 8.25, 10.75, d, fontsize=6.8, ha='center', va='center', color="#312E81", weight='semibold')

    # Vertical Connecting Arrows
    for x in [21, 50, 79]:
        ax.annotate('', xy=(x, 78), xytext=(x, 72), arrowprops=dict(arrowstyle="<->", color="#64748B", lw=1.5))
        ax.annotate('', xy=(x, 54), xytext=(x, 49), arrowprops=dict(arrowstyle="<->", color="#64748B", lw=1.5))
        ax.annotate('', xy=(x, 27), xytext=(x, 22), arrowprops=dict(arrowstyle="<->", color="#64748B", lw=1.5))

    plt.tight_layout()
    fig.savefig(out_dir / "fig5_1_system_architecture.png", bbox_inches='tight')
    plt.close(fig)
    print("Generated Fig 5.1")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 5.2: Tri-Pillar Decoupled Verification Engine
# ─────────────────────────────────────────────────────────────────────────────
def generate_fig5_2():
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Input Box
    in_box = FancyBboxPatch((35, 84), 30, 12, boxstyle="round,pad=0.5", ec="#334155", fc="#F1F5F9", lw=2)
    ax.add_patch(in_box)
    ax.text(50, 92, "INPUT NEWS CLAIM & MEDIA", fontsize=10, fontweight='bold', ha='center', color="#0F172A")
    ax.text(50, 87, "Text Headline / Claim Statement  +  Optional Photographic Context", fontsize=7.5, ha='center', color="#334155")

    # Arrows splitting from Input
    ax.annotate('', xy=(20, 74), xytext=(40, 84), arrowprops=dict(arrowstyle="->", color="#F59E0B", lw=2))
    ax.annotate('', xy=(50, 74), xytext=(50, 84), arrowprops=dict(arrowstyle="->", color="#10B981", lw=2))
    ax.annotate('', xy=(80, 74), xytext=(60, 84), arrowprops=dict(arrowstyle="->", color="#8B5CF6", lw=2))

    # Pillar 1 Box
    p1 = FancyBboxPatch((5, 26), 28, 48, boxstyle="round,pad=0.5", ec="#F59E0B", fc="#FFFBEB", lw=1.8)
    ax.add_patch(p1)
    ax.text(19, 70, "PILLAR 1: FACT VERIFICATION", fontsize=9, fontweight='bold', ha='center', color="#B45309")
    ax.text(19, 65, "Analytical Goal:\nFactual Truth vs. Falsehood", fontsize=7.5, ha='center', color="#78350F", style='italic')
    p1_items = [
        "1. spaCy Linguistic Parse:\nIsolates atomic sub-clauses",
        "2. Multi-Source Search:\nGoogle News RSS + Wiki + BM25",
        "3. Entity-Relevance Gating:\nFilters off-topic evidence",
        "4. DeBERTa-v3 Cross-Encoder:\nDirectional NLI Stance Estimation",
        "5. Epistemic Aggregation Rule:\nEvaluates refutation threshold"
    ]
    for i, it in enumerate(p1_items):
        ax.text(7, 57 - i * 6.5, it, fontsize=7, color="#451A03")
    
    # Pillar 1 Output Box
    out1 = FancyBboxPatch((7, 8), 24, 14, boxstyle="round,pad=0.3", ec="#F59E0B", fc="#FEF3C7", lw=1.2)
    ax.add_patch(out1)
    ax.text(19, 18, "Deterministic Epistemic Verdict", fontsize=7.5, fontweight='bold', ha='center', color="#92400E")
    ax.text(19, 12, "Supported · Partially Supported\nMisleading · Conflicting · Uncertain", fontsize=6.8, ha='center', color="#78350F")

    # Pillar 2 Box
    p2 = FancyBboxPatch((36, 26), 28, 48, boxstyle="round,pad=0.5", ec="#10B981", fc="#ECFDF5", lw=1.8)
    ax.add_patch(p2)
    ax.text(50, 70, "PILLAR 2: CHEAPFAKE DETECTION", fontsize=9, fontweight='bold', ha='center', color="#047857")
    ax.text(50, 65, "Analytical Goal:\nOut-of-Context Misattribution", fontsize=7.5, ha='center', color="#064E3B", style='italic')
    p2_items = [
        "1. OpenCLIP ViT-B/32:\nDual 512-d unit embeddings",
        "2. Cosine Hypersphere Metric:\ns_TI = <v_T, v_I> in [-1, 1]",
        "3. Calibrated Boundary:\nOptimal threshold tau = 0.28",
        "4. Selective Reject Option:\nMargin band |s_TI - 0.28| >= 0.07",
        "5. Selective Classification:\n87.44% acc on accepted (10.8% cov)"
    ]
    for i, it in enumerate(p2_items):
        ax.text(38, 57 - i * 6.5, it, fontsize=7, color="#022C22")

    # Pillar 2 Output Box
    out2 = FancyBboxPatch((38, 8), 24, 14, boxstyle="round,pad=0.3", ec="#10B981", fc="#D1FAE5", lw=1.2)
    ax.add_patch(out2)
    ax.text(50, 18, "Semantic Alignment & Triage", fontsize=7.5, fontweight='bold', ha='center', color="#065F46")
    ax.text(50, 12, "Accepted Pristine / Falsified  OR\nDeferred for Subsequent Review", fontsize=6.8, ha='center', color="#064E3B")

    # Pillar 3 Box
    p3 = FancyBboxPatch((67, 26), 28, 48, boxstyle="round,pad=0.5", ec="#8B5CF6", fc="#F5F3FF", lw=1.8)
    ax.add_patch(p3)
    ax.text(81, 70, "PILLAR 3: RHETORICAL FRAMING", fontsize=9, fontweight='bold', ha='center', color="#6D28D9")
    ax.text(81, 65, "Analytical Goal:\nSensationalist Packaging & Spin", fontsize=7.5, ha='center', color="#4C1D95", style='italic')
    p3_items = [
        "1. Headline Sanitization:\nPurges publisher suffixes (- Reuters)",
        "2. Frozen Dual Backbones:\nDeBERTa-v3 (768) + OpenCLIP (768)",
        "3. Gated Fusion Adapter (1.58M):\nSigmoid channel gate g in [0,1]^512",
        "4. Modality Salience Routing:\nw_text = 38.2%, w_vision = 61.8%",
        "5. VLDBench Rhetoric Predictor:\n64.54% Macro-F1, 0.7086 ROC-AUC"
    ]
    for i, it in enumerate(p3_items):
        ax.text(69, 57 - i * 6.5, it, fontsize=7, color="#2E1065")

    # Pillar 3 Output Box
    out3 = FancyBboxPatch((69, 8), 24, 14, boxstyle="round,pad=0.3", ec="#8B5CF6", fc="#EDE9FE", lw=1.2)
    ax.add_patch(out3)
    ax.text(81, 18, "Framing & Salience Weights", fontsize=7.5, fontweight='bold', ha='center', color="#5B21B6")
    ax.text(81, 12, "Likely vs. Unlikely Manipulation\nw_T = 38.2%, w_I = 61.8%", fontsize=6.8, ha='center', color="#4C1D95")

    # Governing Epistemic Banner at Bottom
    gov_box = FancyBboxPatch((10, 0.5), 80, 5, boxstyle="round,pad=0.3", ec="#EF4444", fc="#FEF2F2", lw=1)
    ax.add_patch(gov_box)
    ax.text(50, 3, "GOVERNING PRINCIPLE:  Factual Falsehood  !=  Image-Context Mismatch  !=  Sensationalist Rhetoric", fontsize=8, fontweight='bold', ha='center', color="#991B1B")

    plt.tight_layout()
    fig.savefig(out_dir / "fig5_2_tri_pillar_engine.png", bbox_inches='tight')
    plt.close(fig)
    print("Generated Fig 5.2")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4.1: Pillar 1 RAG NLI Workflow
# ─────────────────────────────────────────────────────────────────────────────
def generate_fig4_1():
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    steps = [
        ("Input Claim\nStatement", 6, 50, 12, 28, "#1E293B", "#F8FAFC"),
        ("spaCy Linguistic\nDecomposition", 22, 50, 14, 28, "#2563EB", "#EFF6FF"),
        ("Multi-Source\nEvidence Retrieval", 40, 50, 15, 28, "#0D9488", "#F0FDFA"),
        ("DeBERTa-v3\nCross-Encoder NLI", 59, 50, 15, 28, "#D97706", "#FFFBEB"),
        ("Epistemic Decision\nAggregation", 78, 50, 16, 28, "#4F46E5", "#EEF2FF")
    ]

    for title, x, y, w, h, ec, fc in steps:
        box = FancyBboxPatch((x, y - h/2), w, h, boxstyle="round,pad=0.4", ec=ec, fc=fc, lw=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + 6, title, fontsize=7.5, fontweight='bold', ha='center', va='center', color=ec)

    # Subtext annotations inside boxes
    ax.text(12, 44, "Raw headline or\nsocial caption", fontsize=6.5, ha='center', color="#64748B")
    ax.text(29, 43, "Dependency clauses\nNamed entities\nAnchors extracted", fontsize=6.2, ha='center', color="#1E40AF")
    ax.text(47.5, 43, "Google News RSS\nWikipedia API\nBM25 Local Corpus", fontsize=6.2, ha='center', color="#115E59")
    ax.text(66.5, 43, "Pairwise Stance:\nP(Entailment)\nP(Neutral)\nP(Contradiction)", fontsize=6.2, ha='center', color="#92400E")
    ax.text(86, 43, "Supported\nPartially Supported\nMisleading\nConflicting\nUncertain", fontsize=6.2, ha='center', color="#3730A3")

    # Arrows between steps
    for i in range(len(steps) - 1):
        x1 = steps[i][1] + steps[i][3]
        x2 = steps[i+1][1]
        ax.annotate('', xy=(x2, 50), xytext=(x1, 50), arrowprops=dict(arrowstyle="->", color="#475569", lw=1.5))

    # Evidence Filtering Callout below retrieval
    callout = FancyBboxPatch((35, 8), 25, 18, boxstyle="round,pad=0.3", ec="#CBD5E1", fc="#F8FAFC", lw=1, ls="--")
    ax.add_patch(callout)
    ax.text(47.5, 21, "Relevance Gating (retrieval.py)", fontsize=7, fontweight='bold', ha='center', color="#334155")
    ax.text(47.5, 14, "Exact anchor overlap >= 1\nTopical lexical similarity >= 0.25\nUncorroborated -> Uncertain", fontsize=6.2, ha='center', color="#64748B")
    ax.annotate('', xy=(47.5, 36), xytext=(47.5, 26), arrowprops=dict(arrowstyle="->", color="#94A3B8", lw=1, ls="--"))

    plt.tight_layout()
    fig.savefig(out_dir / "fig4_1_rag_nli_workflow.png", bbox_inches='tight')
    plt.close(fig)
    print("Generated Fig 4.1")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4.2: Pillar 2 OpenCLIP Hypersphere Alignment
# ─────────────────────────────────────────────────────────────────────────────
def generate_fig4_2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

    # Subplot 1: Hypersphere Geometry
    theta = np.linspace(0, 2*np.pi, 200)
    ax1.plot(np.cos(theta), np.sin(theta), color="#94A3B8", lw=1.5, ls="--", label="Unit Hypersphere S^{511}")
    ax1.scatter([0], [0], color="#64748B", s=30)
    ax1.text(0.05, 0.05, "Origin (0)", fontsize=8, color="#64748B")

    # Vectors
    v_text = np.array([0.707, 0.707])
    v_img_pristine = np.array([0.866, 0.500])
    v_img_falsified = np.array([0.259, 0.966])

    ax1.annotate('', xy=v_text, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#2563EB", lw=2))
    ax1.text(0.72, 0.75, "f_T(T) Text", fontsize=9, fontweight='bold', color="#2563EB")

    ax1.annotate('', xy=v_img_pristine, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#10B981", lw=2))
    ax1.text(0.88, 0.48, "f_V(I_Pristine)", fontsize=9, fontweight='bold', color="#10B981")

    ax1.annotate('', xy=v_img_falsified, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#EF4444", lw=2))
    ax1.text(0.15, 1.02, "f_V(I_Falsified)", fontsize=9, fontweight='bold', color="#EF4444")

    ax1.set_xlim(-1.2, 1.3)
    ax1.set_ylim(-1.2, 1.3)
    ax1.set_aspect('equal')
    ax1.set_title("A. OpenCLIP Unit Hypersphere Geometry", fontweight='bold', fontsize=10)
    ax1.axis('off')

    # Subplot 2: Score Distribution & Decision Boundaries
    x = np.linspace(0.10, 0.45, 500)
    p_pristine = (1 / (0.052 * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - 0.298)/0.052)**2)
    p_falsified = (1 / (0.052 * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - 0.269)/0.052)**2)

    ax2.plot(x, p_pristine, color="#10B981", lw=2, label="Pristine Pairs (mu=0.298)")
    ax2.plot(x, p_falsified, color="#EF4444", lw=2, label="Falsified Pairs (mu=0.269)")
    ax2.axvline(0.22, color="#94A3B8", lw=1.8, ls=":", label="Heuristic Baseline tau=0.22 (Acc 54.68%)")
    ax2.axvline(0.28, color="#2563EB", lw=2.2, ls="-", label="Calibrated Optimal tau=0.28 (Acc 63.53%)")

    # Margin Shading
    ax2.axvspan(0.28 - 0.07, 0.28 + 0.07, color="#F59E0B", alpha=0.18, label="Ambiguity Margin Delta=0.07 (Deferred)")

    ax2.set_xlabel("OpenCLIP Cosine Similarity Score s_TI")
    ax2.set_ylabel("Probability Density")
    ax2.set_title("B. Score Distributions & Decision Boundaries", fontweight='bold', fontsize=10)
    ax2.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(out_dir / "fig4_2_cross_modal_alignment.png", bbox_inches='tight')
    plt.close(fig)
    print("Generated Fig 4.2")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4.3: Selective Classification Margin Rejection Mechanism
# ─────────────────────────────────────────────────────────────────────────────
def generate_fig4_3():
    fig, ax = plt.subplots(figsize=(9, 4), dpi=300)
    ax.set_xlim(0.10, 0.46)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Base line
    ax.plot([0.12, 0.44], [4, 4], color="#475569", lw=2)

    # Zones
    # Zone 1: Accepted Falsified
    z1 = FancyBboxPatch((0.12, 4.3), 0.09, 4.5, boxstyle="round,pad=0.01", ec="#DC2626", fc="#FEE2E2", lw=1.5)
    ax.add_patch(z1)
    ax.text(0.165, 7.2, "ACCEPTED PREDICTION\nFALSIFIED (Cheapfake)", fontsize=8, fontweight='bold', ha='center', color="#991B1B")
    ax.text(0.165, 5.2, "s_TI <= tau - Delta (0.21)\nHigh Confidence Mismatch\nFalsified Prec: 87.57%", fontsize=7, ha='center', color="#7F1D1D")

    # Zone 2: Deferred Ambiguity Band
    z2 = FancyBboxPatch((0.21, 4.3), 0.14, 4.5, boxstyle="round,pad=0.01", ec="#D97706", fc="#FEF3C7", lw=1.5)
    ax.add_patch(z2)
    ax.text(0.28, 7.2, "DEFERRED COHORT\nSubsequent Human Review", fontsize=8, fontweight='bold', ha='center', color="#92400E")
    ax.text(0.28, 5.2, "|s_TI - 0.28| < Delta (0.07)\nHigh Error Risk Region\n89.15% Test Records Deferred", fontsize=7, ha='center', color="#78350F")

    # Zone 3: Accepted Pristine
    z3 = FancyBboxPatch((0.35, 4.3), 0.09, 4.5, boxstyle="round,pad=0.01", ec="#16A34A", fc="#DCFCE7", lw=1.5)
    ax.add_patch(z3)
    ax.text(0.395, 7.2, "ACCEPTED PREDICTION\nPRISTINE (Authentic)", fontsize=8, fontweight='bold', ha='center', color="#166534")
    ax.text(0.395, 5.2, "s_TI >= tau + Delta (0.35)\nHigh Confidence Concordance\nPristine Prec: 87.33%", fontsize=7, ha='center', color="#14532D")

    # Ticks and Labels on axis
    ticks = [
        (0.21, "tau - Delta\n(0.21)"),
        (0.28, "Operating Boundary\ntau = 0.28"),
        (0.35, "tau + Delta\n(0.35)")
    ]
    for pos, label in ticks:
        ax.plot([pos, pos], [3.6, 4.4], color="#0F172A", lw=2)
        ax.text(pos, 2.5, label, fontsize=8, ha='center', va='top', fontweight='bold', color="#1E293B")

    # Performance annotation box
    perf_box = FancyBboxPatch((0.14, 0.2), 0.28, 1.6, boxstyle="round,pad=0.02", ec="#3B82F6", fc="#EFF6FF", lw=1)
    ax.add_patch(perf_box)
    ax.text(0.28, 1.0, "Accepted-Subset Test Acc: 87.44%  ·  Coverage: 10.85% (N=788)  ·  95% CI: [84.94%, 89.57%]\nFull Benchmark Baseline Acc: 63.53% (N=7,264 across 100% Coverage)", fontsize=7.2, ha='center', va='center', color="#1E3A8A", fontweight='semibold')

    plt.tight_layout()
    fig.savefig(out_dir / "fig4_3_selective_classification.png", bbox_inches='tight')
    plt.close(fig)
    print("Generated Fig 4.3")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4.4: Pillar 3 Gated Multimodal Fusion Adapter Architecture
# ─────────────────────────────────────────────────────────────────────────────
def generate_fig4_4():
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Encoders (Top)
    enc_t = FancyBboxPatch((15, 80), 30, 14, boxstyle="round,pad=0.4", ec="#2563EB", fc="#EFF6FF", lw=1.5)
    ax.add_patch(enc_t)
    ax.text(30, 89, "Frozen DeBERTa-v3 Text Encoder", fontsize=8.5, fontweight='bold', ha='center', color="#1E40AF")
    ax.text(30, 83, "Sanitized Headline -> 768-d Pooled Representation", fontsize=7, ha='center', color="#3B82F6")

    enc_v = FancyBboxPatch((55, 80), 30, 14, boxstyle="round,pad=0.4", ec="#0D9488", fc="#F0FDFA", lw=1.5)
    ax.add_patch(enc_v)
    ax.text(70, 89, "Frozen OpenCLIP Vision Encoder", fontsize=8.5, fontweight='bold', ha='center', color="#115E59")
    ax.text(70, 83, "Visual Image -> 768-d Pooled Representation", fontsize=7, ha='center', color="#14B8A6")

    # Linear Projections
    proj_t = FancyBboxPatch((17, 60), 26, 11, boxstyle="round,pad=0.3", ec="#94A3B8", fc="#F8FAFC", lw=1.2)
    ax.add_patch(proj_t)
    ax.text(30, 65.5, "Text Projection Layer\nLinear(768->512) + LayerNorm", fontsize=7.2, ha='center', color="#334155")

    proj_v = FancyBboxPatch((57, 60), 26, 11, boxstyle="round,pad=0.3", ec="#94A3B8", fc="#F8FAFC", lw=1.2)
    ax.add_patch(proj_v)
    ax.text(70, 65.5, "Vision Projection Layer\nLinear(768->512) + LayerNorm", fontsize=7.2, ha='center', color="#334155")

    ax.annotate('', xy=(30, 71), xytext=(30, 80), arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.5))
    ax.annotate('', xy=(70, 71), xytext=(70, 80), arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.5))

    # Gating Mechanism (Center)
    gate_box = FancyBboxPatch((32, 36), 36, 16, boxstyle="round,pad=0.4", ec="#8B5CF6", fc="#F5F3FF", lw=1.8)
    ax.add_patch(gate_box)
    ax.text(50, 48, "DYNAMIC SIGMOID CHANNEL GATE", fontsize=9, fontweight='bold', ha='center', color="#5B21B6")
    ax.text(50, 42, "g = Sigmoid(Linear(Concat[u_T; u_V], 1024->512))", fontsize=7.5, ha='center', color="#6D28D9", family='monospace')
    ax.text(50, 38, "Elementwise Modality Routing: f = g * u_T + (1 - g) * u_V", fontsize=7, ha='center', color="#4C1D95", style='italic')

    # Connection lines to Gate
    ax.annotate('', xy=(42, 52), xytext=(30, 60), arrowprops=dict(arrowstyle="->", color="#8B5CF6", lw=1.5))
    ax.annotate('', xy=(58, 52), xytext=(70, 60), arrowprops=dict(arrowstyle="->", color="#8B5CF6", lw=1.5))

    # Output Modality Salience Callout
    sal_box = FancyBboxPatch((72, 38), 24, 12, boxstyle="round,pad=0.3", ec="#C084FC", fc="#FAF5FF", lw=1, ls=":")
    ax.add_patch(sal_box)
    ax.text(84, 46, "Empirical Modality Salience", fontsize=7.2, fontweight='bold', ha='center', color="#7E22CE")
    ax.text(84, 41, "Text Gate: 38.2%\nVisual Gate: 61.8%", fontsize=6.8, ha='center', color="#6B21A8")

    # Classification Head (Bottom)
    head_box = FancyBboxPatch((35, 12), 30, 15, boxstyle="round,pad=0.4", ec="#4F46E5", fc="#EEF2FF", lw=1.5)
    ax.add_patch(head_box)
    ax.text(50, 23, "Classification MLP & Output", fontsize=8.5, fontweight='bold', ha='center', color="#3730A3")
    ax.text(50, 16, "LayerNorm + Dropout(0.3) + Linear(512->2)\n4-Class / Binary Rhetorical Likelihood", fontsize=7, ha='center', color="#312E81")

    ax.annotate('', xy=(50, 27), xytext=(50, 36), arrowprops=dict(arrowstyle="->", color="#4F46E5", lw=1.8))

    # Footnote
    ax.text(50, 4, "Total Trainable Parameters: 1,582,594 (1.58M)  ·  Backbones Frozen During Training & Inference", fontsize=7.5, ha='center', color="#64748B", style='italic')

    plt.tight_layout()
    fig.savefig(out_dir / "fig4_4_gated_adapter_architecture.png", bbox_inches='tight')
    plt.close(fig)
    print("Generated Fig 4.4")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 6.1: End-to-End Operational Workflow
# ─────────────────────────────────────────────────────────────────────────────
def generate_fig6_1():
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    stages = [
        ("1. Ingestion / Input", "User Submission or\nLive RSS/GDELT Feed", 5, 50, 15, "#1E293B", "#F1F5F9"),
        ("2. Decomposition", "spaCy Dependency Parse\nExtract Atomic Claims", 24, 50, 16, "#2563EB", "#EFF6FF"),
        ("3. Parallel Verification", "Pillar 1: RAG + DeBERTa\nPillar 2: OpenCLIP Cosine\nPillar 3: Gated Adapter", 44, 50, 18, "#0D9488", "#F0FDFA"),
        ("4. Decision Triage", "Margin Check Delta=0.07\nAccepted vs. Deferred\nVerdict Synthesis", 66, 50, 16, "#D97706", "#FFFBEB"),
        ("5. Presentation & Store", "Streamlit UI Rendering\nSQLite Audit Log\nHuman Review Queue", 86, 50, 13, "#4F46E5", "#EEF2FF")
    ]

    for title, desc, x, y, w, ec, fc in stages:
        box = FancyBboxPatch((x, y - 18), w, 36, boxstyle="round,pad=0.4", ec=ec, fc=fc, lw=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + 12, title, fontsize=7.5, fontweight='bold', ha='center', color=ec)
        ax.text(x + w/2, y - 2, desc, fontsize=6.8, ha='center', color="#334155")

    for i in range(len(stages) - 1):
        x1 = stages[i][2] + stages[i][4]
        x2 = stages[i+1][2]
        ax.annotate('', xy=(x2, 50), xytext=(x1, 50), arrowprops=dict(arrowstyle="->", color="#64748B", lw=2))

    # Top banner: Multi-User Scenarios
    sc_box = FancyBboxPatch((15, 82), 70, 12, boxstyle="round,pad=0.3", ec="#CBD5E1", fc="#F8FAFC", lw=1)
    ax.add_patch(sc_box)
    ax.text(50, 90, "DUAL OPERATIONAL MODES: INTERACTIVE INVESTIGATION & CONTINUOUS SURVEILLANCE", fontsize=8, fontweight='bold', ha='center', color="#1E293B")
    ax.text(50, 85, "Interactive Mode: Single Claim/Image Analysis (< 2.5s CPU)  |  Surveillance Mode: Background RSS Queue Ingestion", fontsize=7, ha='center', color="#475569")

    # Bottom banner: Human-in-the-Loop Safeguard
    hl_box = FancyBboxPatch((15, 6), 70, 12, boxstyle="round,pad=0.3", ec="#F59E0B", fc="#FEF3C7", lw=1)
    ax.add_patch(hl_box)
    ax.text(50, 14, "HUMAN-IN-THE-LOOP ACTIVE LEARNING & EDITORIAL GOVERNANCE", fontsize=8, fontweight='bold', ha='center', color="#92400E")
    ax.text(50, 9, "Ambiguous cases (|s - 0.28| < 0.07) and uncorroborated claims (Uncertain) are routed for expert human review.", fontsize=7, ha='center', color="#78350F")

    plt.tight_layout()
    fig.savefig(out_dir / "fig6_1_end_to_end_workflow.png", bbox_inches='tight')
    plt.close(fig)
    print("Generated Fig 6.1")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 9.1: End-to-End Research Process & Timeline
# ─────────────────────────────────────────────────────────────────────────────
def generate_fig9_1():
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    phases = [
        ("Phase 1: Formulation", "Problem identification, Tri-Pillar decoupling conception, academic non-claims defined", 5, 75, 27, "#2563EB", "#EFF6FF"),
        ("Phase 2: Baseline Audit", "NewsCLIPpings acquisition, zero-shot OpenCLIP audit, threshold calibration tau=0.28", 36, 75, 27, "#0D9488", "#F0FDFA"),
        ("Phase 3: Syndication Split", "Group-aware bipartite graph clustering, Protocol A (70/15/15) split generation", 67, 75, 27, "#D97706", "#FFFBEB"),
        ("Phase 4: Adapter Design", "Strict temporal Protocol B (129 groups purged), Gated Adapter Option 3 selected", 5, 45, 27, "#4F46E5", "#EEF2FF"),
        ("Phase 5: Feature Caching", "Sanitized DeBERTa + OpenCLIP caching, publisher leakage audit, baseline models", 36, 45, 27, "#7C3AED", "#F5F3FF"),
        ("Phase 6: Empirical Training", "Option 1 vs Option 3 training, bootstrap CIs, Protocol B Strict & Held-Out evaluation", 67, 45, 27, "#DB2777", "#FDF2F8"),
        ("Phase 7: Model Selection", "Validation-based multi-criteria model selection, error analysis, limitations review", 5, 15, 27, "#059669", "#ECFDF5"),
        ("Phase 8: System Integration", "Full Streamlit integration (app.py), shared singleton memory caching, 122 tests", 36, 15, 27, "#EA580C", "#FFF7ED"),
        ("Phase 9: Governance Audit", "Selective classification audit (Delta=0.07, 87.44% acc), dissertation documentation", 67, 15, 27, "#DC2626", "#FEF2F2")
    ]

    for title, desc, x, y, w, ec, fc in phases:
        box = FancyBboxPatch((x, y - 10), w, 20, boxstyle="round,pad=0.3", ec=ec, fc=fc, lw=1.2)
        ax.add_patch(box)
        ax.text(x + w/2, y + 6, title, fontsize=7.5, fontweight='bold', ha='center', color=ec)
        ax.text(x + w/2, y - 2, desc, fontsize=6.3, ha='center', color="#334155")

    # Flow arrows
    ax.annotate('', xy=(36, 75), xytext=(32, 75), arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.5))
    ax.annotate('', xy=(67, 75), xytext=(63, 75), arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.5))
    ax.annotate('', xy=(18.5, 55), xytext=(80.5, 65), arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.5, connectionstyle="angle,angleA=0,angleB=90,rad=5"))
    ax.annotate('', xy=(36, 45), xytext=(32, 45), arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.5))
    ax.annotate('', xy=(67, 45), xytext=(63, 45), arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.5))
    ax.annotate('', xy=(18.5, 25), xytext=(80.5, 35), arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.5, connectionstyle="angle,angleA=0,angleB=90,rad=5"))
    ax.annotate('', xy=(36, 15), xytext=(32, 15), arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.5))
    ax.annotate('', xy=(67, 15), xytext=(63, 15), arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.5))

    ax.text(50, 95, "CHRONOLOGICAL END-TO-END RESEARCH & DEVELOPMENT TIMELINE", fontsize=9.5, fontweight='bold', ha='center', color="#0F172A")

    plt.tight_layout()
    fig.savefig(out_dir / "fig9_1_research_lifecycle.png", bbox_inches='tight')
    plt.close(fig)
    print("Generated Fig 9.1")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 12.1: NewsCLIPpings Confusion Matrix & Adversarial Breakdown
# ─────────────────────────────────────────────────────────────────────────────
def generate_fig12_1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

    # Subplot 1: Full Test Confusion Matrix (tau=0.28, N=7,264)
    cm = np.array([[2835, 797], [1852, 1780]])
    im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title("A. Full Test Confusion Matrix (tau=0.28, N=7,264)", fontsize=9.5, fontweight='bold')
    
    classes = ['Pristine (0)', 'Falsified (1)']
    tick_marks = np.arange(len(classes))
    ax1.set_xticks(tick_marks)
    ax1.set_xticklabels(classes, fontsize=8.5)
    ax1.set_yticks(tick_marks)
    ax1.set_yticklabels(classes, fontsize=8.5)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, f"{cm[i, j]:,}\n({cm[i, j]/3632*100:.1f}%)",
                     ha="center", va="center", fontsize=9, fontweight='bold',
                     color="white" if cm[i, j] > thresh else "black")

    ax1.set_ylabel('True Benchmark Class')
    ax1.set_xlabel('Predicted Class')
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # Subplot 2: Generator Robustness Breakdown
    gens = ['Gen 1: Semantics\nText-Text', 'Gen 2: Scene\nResNet Place', 'Gen 3: Person\nSBERT Text', 'Gen 0: Semantics\nText-Image (NN)']
    acc_tau22 = [55.12, 60.35, 54.52, 48.73]
    acc_tau28 = [81.44, 67.79, 61.07, 43.83]

    y = np.arange(len(gens))
    h = 0.35

    ax2.barh(y - h/2, acc_tau22, h, label='Baseline (tau=0.22)', color='#94A3B8')
    ax2.barh(y + h/2, acc_tau28, h, label='Calibrated (tau=0.28)', color='#2563EB')

    ax2.set_yticks(y)
    ax2.set_yticklabels(gens, fontsize=8)
    ax2.set_xlabel('Accuracy (%)')
    ax2.set_xlim(0, 100)
    ax2.set_title('B. Adversarial Generator Breakdown (N=1,816 each)', fontsize=9.5, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    fig.savefig(out_dir / "fig12_1_newsclippings_performance.png", bbox_inches='tight')
    plt.close(fig)
    print("Generated Fig 12.1")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 12.2: Risk-Coverage & Accuracy-Coverage Curves
# ─────────────────────────────────────────────────────────────────────────────
def generate_fig12_2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

    # Empirical test & val sweep data
    cov_test = np.array([100.0, 82.03, 66.42, 50.73, 36.78, 26.10, 16.97, 10.85, 6.95, 4.35, 2.62])
    acc_test = np.array([63.53, 66.91, 70.24, 74.22, 78.52, 82.23, 85.24, 87.44, 91.09, 93.04, 92.11])
    risk_test = 100.0 - acc_test

    cov_val = np.array([100.0, 83.61, 66.20, 50.17, 36.77, 25.97, 17.75, 11.30, 6.98, 4.02, 2.22])
    acc_val = np.array([63.51, 65.95, 69.40, 73.70, 77.82, 81.14, 84.28, 86.40, 87.35, 90.43, 91.03])

    # Subplot A: Accuracy vs Coverage
    ax1.plot(cov_test, acc_test, marker='o', color='#2563EB', lw=2, label='Test Set (N=7,264)')
    ax1.plot(cov_val, acc_val, marker='s', color='#10B981', lw=1.8, ls='--', label='Validation Set (N=7,024)')
    
    # Highlight operating point Delta=0.07
    ax1.scatter([10.85], [87.44], color='#DC2626', s=90, zorder=5, label='Test Operating Point Delta*=0.07 (87.44%)')
    ax1.scatter([11.30], [86.40], color='#D97706', s=80, zorder=5, label='Val Operating Point Delta*=0.07 (86.40%)')

    ax1.axhline(85.0, color='#94A3B8', ls=':', lw=1.5, label='High-Precision Target (85%)')
    ax1.set_xlabel('Coverage Phi (%)')
    ax1.set_ylabel('Accepted Subset Accuracy (%)')
    ax1.set_title('A. Selective Accuracy vs. Coverage', fontsize=9.5, fontweight='bold')
    ax1.set_xlim(0, 105)
    ax1.set_ylim(60, 96)
    ax1.legend(loc='lower left', fontsize=7.5)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Subplot B: Selective Risk vs Coverage
    ax2.plot(cov_test, risk_test, marker='o', color='#DC2626', lw=2, label='Test Selective Risk (1 - Acc)')
    ax2.scatter([10.85], [12.56], color='#2563EB', s=90, zorder=5, label='Test Risk at Delta*=0.07 (12.56%)')

    ax2.set_xlabel('Coverage Phi (%)')
    ax2.set_ylabel('Selective Error Risk (%)')
    ax2.set_title('B. Empirical Risk-Coverage Trade-off', fontsize=9.5, fontweight='bold')
    ax2.set_xlim(0, 105)
    ax2.set_ylim(5, 40)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    fig.savefig(out_dir / "fig12_2_risk_coverage_curve.png", bbox_inches='tight')
    plt.close(fig)
    print("Generated Fig 12.2")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 12.3: VLDBench Comparison & Generalization
# ─────────────────────────────────────────────────────────────────────────────
def generate_fig12_3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

    # Subplot 1: Architectural Comparison on Protocol A Test (N=1,094)
    models = ['Publisher-Only\n(Metadata)', 'Sanitized Text\n(DeBERTa-v3)', 'Image-Only\n(OpenCLIP)', 'Option 1\nLate Concat', 'Option 3\nGated Fusion*']
    f1_scores = [44.36, 55.00, 64.30, 63.22, 64.54]
    colors = ['#94A3B8', '#38BDF8', '#34D399', '#FBBF24', '#818CF8']

    bars = ax1.bar(models, f1_scores, color=colors, width=0.55, edgecolor='#334155', lw=1)
    ax1.set_ylabel('Test Macro-F1 (%)')
    ax1.set_ylim(35, 72)
    ax1.set_title('A. Architecture Comparison (Protocol A Test, N=1,094)', fontsize=9.5, fontweight='bold')
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 1, f"{h:.2f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax1.grid(True, axis='y', linestyle=':', alpha=0.6)

    # Subplot 2: Generalization Across Distribution Shifts
    eval_splits = ['Protocol A Test\n(Group-Aware)', 'Protocol B Strict\n(Temporal Shift)', 'Publisher-Held-Out\n(Unseen Newsrooms)']
    opt3_f1 = [64.54, 62.43, 61.42]

    bars2 = ax2.bar(eval_splits, opt3_f1, color='#6366F1', width=0.45, edgecolor='#312E81', lw=1)
    ax2.set_ylabel('Macro-F1 (%)')
    ax2.set_ylim(50, 72)
    ax2.set_title('B. Option 3 Robustness Across Distribution Shifts', fontsize=9.5, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.8, f"{h:.2f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax2.grid(True, axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    fig.savefig(out_dir / "fig12_3_vldbench_comparison.png", bbox_inches='tight')
    plt.close(fig)
    print("Generated Fig 12.3")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 6.2 - 6.4: UI Workstation Visual Mockups
# ─────────────────────────────────────────────────────────────────────────────
def generate_ui_mockups():
    # Fig 6.2: Claim Investigation Workspace (tab_verify)
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Window frame
    win = FancyBboxPatch((2, 2), 96, 96, boxstyle="round,pad=0.5", ec="#334155", fc="#0F172A", lw=1.5)
    ax.add_patch(win)
    ax.text(5, 94, "VERITAS-MM 2.0  ·  Claim Investigation Workspace (tab_verify)", fontsize=9, fontweight='bold', color="#F8FAFC")

    # Verdict Banner
    vb = FancyBboxPatch((5, 78), 90, 12, boxstyle="round,pad=0.3", ec="#10B981", fc="#064E3B", lw=1.2)
    ax.add_patch(vb)
    ax.text(7, 85, "SUPPORTED  ·  Ground Truth Verified", fontsize=10, fontweight='bold', color="#34D399")
    ax.text(7, 80.5, "Independent cross-referenced evidence corroborates all extracted propositions with entailment confidence 0.94.", fontsize=7, color="#A7F3D0")

    # Selective Classification Status Banner
    sb = FancyBboxPatch((5, 63), 90, 12, boxstyle="round,pad=0.3", ec="#059669", fc="#022C22", lw=1.2)
    ax.add_patch(sb)
    ax.text(7, 70, "ACCEPTED PREDICTION (Pillar 2 High-Margin Confident)  ·  Margin: 0.112 >= Delta (0.07)", fontsize=8.5, fontweight='bold', color="#6EE7B7")
    ax.text(7, 65.5, "OpenCLIP score (0.392) clears high-confidence threshold. Accepted-subset test accuracy is 87.44% (Coverage: 10.85%, N=788). Full baseline: 63.53%.", fontsize=6.8, color="#D1FAE5")

    # Column Left: Cross-Modal Alignment Card
    cm_box = FancyBboxPatch((5, 22), 43, 38, boxstyle="round,pad=0.4", ec="#334155", fc="#1E293B", lw=1)
    ax.add_patch(cm_box)
    ax.text(7, 55, "Pillar 2: Cross-Modal Alignment (OpenCLIP)", fontsize=8, fontweight='bold', color="#38BDF8")
    ax.text(7, 49, "Cosine Alignment Score: s_TI = 0.392", fontsize=7.5, color="#F8FAFC")
    # Progress bar
    pb_bg = FancyBboxPatch((7, 43), 39, 4, boxstyle="round,pad=0.1", ec="#475569", fc="#334155", lw=1)
    ax.add_patch(pb_bg)
    pb_fg = FancyBboxPatch((7, 43), 26, 4, boxstyle="round,pad=0.1", ec="#10B981", fc="#10B981", lw=1)
    ax.add_patch(pb_fg)
    ax.text(7, 36, "Decision Boundary: tau = 0.28\nStatus: Visual and textual contexts concordant.\nNon-claim: Does not detect pixel-level deepfake artifacts.", fontsize=6.5, color="#94A3B8")

    # Column Right: Rhetorical Manipulation Card
    rh_box = FancyBboxPatch((52, 22), 43, 38, boxstyle="round,pad=0.4", ec="#334155", fc="#1E293B", lw=1)
    ax.add_patch(rh_box)
    ax.text(54, 55, "Pillar 3: Rhetorical Framing Analysis", fontsize=8, fontweight='bold', color="#C084FC")
    ax.text(54, 49, "Manipulation Verdict: UNLIKELY (0.184)", fontsize=7.5, color="#F8FAFC")
    ax.text(54, 43, "Modality Salience Weight Gating (1.58M Adapter):", fontsize=7, color="#E2E8F0")
    # Gate bar
    gb_t = FancyBboxPatch((54, 37), 14.8, 3.5, boxstyle="round,pad=0.1", ec="#3B82F6", fc="#3B82F6", lw=1)
    ax.add_patch(gb_t)
    gb_v = FancyBboxPatch((68.8, 37), 24.2, 3.5, boxstyle="round,pad=0.1", ec="#8B5CF6", fc="#8B5CF6", lw=1)
    ax.add_patch(gb_v)
    ax.text(54, 31, "Text Channel: 38.2%   |   Visual Channel: 61.8%\nSanitization: Reuters attribution purged before encoding.\nEpistemic boundary: Rhetoric score != factual falsity.", fontsize=6.5, color="#94A3B8")

    # Bottom Evidence Strip
    ev_box = FancyBboxPatch((5, 5), 90, 14, boxstyle="round,pad=0.3", ec="#334155", fc="#1E293B", lw=1)
    ax.add_patch(ev_box)
    ax.text(7, 14, "Retrieved External Evidence (Pillar 1 RAG)", fontsize=7.5, fontweight='bold', color="#FBBF24")
    ax.text(7, 8, "[Google News RSS] Reuters (2026): 'London Fire Brigade confirms warehouse blaze under control with zero casualties.'\nNLI Stance: ENTAILMENT (Probability: 0.942)  ·  Topical Relevance Score: 0.88", fontsize=6.5, color="#CBD5E1")

    plt.tight_layout()
    fig.savefig(out_dir / "fig6_2_ui_verify_workspace.png", bbox_inches='tight')
    plt.close(fig)
    print("Generated Fig 6.2")

    # Fig 6.3: Live News Intelligence Monitor (tab_live)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    win = FancyBboxPatch((2, 2), 96, 96, boxstyle="round,pad=0.5", ec="#334155", fc="#0F172A", lw=1.5)
    ax.add_patch(win)
    ax.text(5, 94, "VERITAS-MM 2.0  ·  Live News Intelligence Monitor (tab_live)", fontsize=9, fontweight='bold', color="#F8FAFC")

    # KPI Header Strip
    kpis = [("Total Articles Stored", "325"), ("Pending Verification", "18"), ("Supported Claims", "142"), ("Misleading Claims", "98"), ("Active RSS Feeds", "8")]
    for i, (k, v) in enumerate(kpis):
        bx = 5 + i * 18.2
        box = FancyBboxPatch((bx, 76), 17.2, 14, boxstyle="round,pad=0.3", ec="#334155", fc="#1E293B", lw=1)
        ax.add_patch(box)
        ax.text(bx + 8.6, 84.5, v, fontsize=11, fontweight='bold', ha='center', color="#38BDF8")
        ax.text(bx + 8.6, 79, k, fontsize=6.5, ha='center', color="#94A3B8")

    # Table Mockup
    tbl = FancyBboxPatch((5, 10), 90, 62, boxstyle="round,pad=0.4", ec="#334155", fc="#1E293B", lw=1)
    ax.add_patch(tbl)
    ax.text(7, 67, "Real-Time Ingestion Queue & Collector Health Registry", fontsize=8, fontweight='bold', color="#F8FAFC")
    
    headers = ["Timestamp", "Publisher", "Region", "Headline / Claim Text", "Status", "Pillar 1", "Pillar 2"]
    ax.text(7, 61, f"{headers[0]:<12} {headers[1]:<12} {headers[2]:<10} {headers[3]:<42} {headers[4]:<12} {headers[5]:<10} {headers[6]}", fontsize=6.8, fontweight='bold', color="#38BDF8", family='monospace')
    
    rows = [
        ("14:32:10", "Reuters", "Intl", "Fed holds benchmark rates steady amid inflation cools", "VERIFIED", "Supported", "Accepted"),
        ("14:28:45", "Daily Mail", "UK", "Chaos in downtown as protest escalates violently", "VERIFIED", "Misleading", "Accepted"),
        ("14:21:02", "Times of India", "India", "ISRO schedules heavy rocket test flight for Sunday", "VERIFIED", "Supported", "Deferred"),
        ("14:15:33", "BBC News", "Intl", "New climate accord signed by 45 nations in Geneva", "VERIFIED", "Supported", "Accepted"),
        ("14:02:18", "The Guardian", "UK", "Hospital trusts report IT disruption following update", "PENDING", "Processing", "Queued")
    ]
    for idx, r in enumerate(rows):
        y_pos = 54 - idx * 7.5
        ax.text(7, y_pos, f"{r[0]:<12} {r[1]:<12} {r[2]:<10} {r[3]:<42} {r[4]:<12} {r[5]:<10} {r[6]}", fontsize=6.5, color="#CBD5E1", family='monospace')

    plt.tight_layout()
    fig.savefig(out_dir / "fig6_3_ui_live_intelligence.png", bbox_inches='tight')
    plt.close(fig)
    print("Generated Fig 6.3")

    # Fig 6.4: Empirical Evaluation Dashboard (tab_eval)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    win = FancyBboxPatch((2, 2), 96, 96, boxstyle="round,pad=0.5", ec="#334155", fc="#0F172A", lw=1.5)
    ax.add_patch(win)
    ax.text(5, 94, "VERITAS-MM 2.0  ·  Empirical Benchmark Evaluation Dashboard (tab_eval)", fontsize=9, fontweight='bold', color="#F8FAFC")

    # Benchmark Cards
    # NewsCLIPpings Card
    c1 = FancyBboxPatch((5, 12), 43, 78, boxstyle="round,pad=0.4", ec="#334155", fc="#1E293B", lw=1)
    ax.add_patch(c1)
    ax.text(7, 85, "Task A: NewsCLIPpings Cheapfake Benchmark", fontsize=8.5, fontweight='bold', color="#38BDF8")
    ax.text(7, 78, "Full Test Accuracy (tau=0.28): 63.53%\nBalanced Accuracy: 63.53%  ·  Macro-F1: 62.75%\n95% Wilson CI: [62.42%, 64.63%]  (N=7,264)\nMcNemar Test: chi^2 = 196.36, p = 1.30e-44", fontsize=7, color="#E2E8F0")

    ax.text(7, 56, "Selective Classification (Delta*=0.07):", fontsize=7.5, fontweight='bold', color="#FBBF24")
    ax.text(7, 44, "Accepted Accuracy: 87.44% (N=788, 10.85% cov)\n95% Wilson CI: [84.94%, 89.57%] (Point Estimate)\nBalanced Acc: 87.10%  ·  Macro-F1: 87.24%\nSelective Risk: 12.56% (vs 36.47% baseline)\nDeferred for Review: 6,476 records (89.15%)", fontsize=7, color="#CBD5E1")

    # VLDBench Card
    c2 = FancyBboxPatch((52, 12), 43, 78, boxstyle="round,pad=0.4", ec="#334155", fc="#1E293B", lw=1)
    ax.add_patch(c2)
    ax.text(54, 85, "VLDBench Multimodal Rhetoric Benchmark", fontsize=8.5, fontweight='bold', color="#C084FC")
    ax.text(54, 78, "Option 3 Gated Multimodal Fusion Adapter (1.58M)\nProtocol A Test Macro-F1: 64.54% (95% CI: [61.7, 67.2])\nBalanced Accuracy: 64.80%  ·  ROC-AUC: 0.7086\nAccuracy: 64.72% (N=1,094 across 4,285 clusters)", fontsize=7, color="#E2E8F0")

    ax.text(54, 56, "Generalization Across Distribution Shifts:", fontsize=7.5, fontweight='bold', color="#34D399")
    ax.text(54, 44, "Protocol B Strict (Temporal): 62.43% Macro-F1\n(129 cross-cutoff clusters purged)\nPublisher-Held-Out (Unseen Outlets): 61.42%\n(Headline sanitization verified effective)\nModality Salience Gate: w_T=38.2%, w_I=61.8%", fontsize=7, color="#CBD5E1")

    plt.tight_layout()
    fig.savefig(out_dir / "fig6_4_ui_benchmark_dashboard.png", bbox_inches='tight')
    plt.close(fig)
    print("Generated Fig 6.4")

if __name__ == "__main__":
    generate_fig5_1()
    generate_fig5_2()
    generate_fig4_1()
    generate_fig4_2()
    generate_fig4_3()
    generate_fig4_4()
    generate_fig6_1()
    generate_fig9_1()
    generate_fig12_1()
    generate_fig12_2()
    generate_fig12_3()
    generate_ui_mockups()
    print("ALL 14 FIGURES GENERATED SUCCESSFULLY!")
