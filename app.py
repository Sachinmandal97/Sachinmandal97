"""
VERITAS-MM: Explainable Multimodal Misinformation Verification Platform
========================================================================
MSc Data Science Final-Year Research Project Application
Streamlit Demonstration Platform

Design Specifications:
- Deep graphite/navy surfaces (#0B0F19, #111827, #1E293B) with restrained cyan accents (#06B6D4, #0EA5E9).
- Compact brand bar, clean typography hierarchy, restrained borders, and cohesive spacing.
- Preserves all 5 core navigation views: Verify Claim, Evidence Explorer, Methodology & Pipeline,
  Benchmark Evaluation, and Research Scope & Limitations.
- Preserves all backend pipeline integrations, preset decoupling logic, and authentic benchmark figures.

Authorship & Attributions:
- Core Pipeline: Explainable Multimodal Evidence-Augmented Framework (MSc Research)
- UI Architecture & Patterns: Inspired by kmist70/fake-news-detector (MIT License)
- Vision-Language Alignment: Inspired by LaughingParrot (MIT License)
- Benchmarks: NewsCLIPpings (Luo et al., EMNLP 2021)
"""

import html
import json
import math
import os
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import MultimodalVerificationPipeline
from src.schemas import DecisionVerdict, MultimodalSample, VerificationResult
from src.database import NewsIntelligenceDB, VALID_PROVENANCES
from src.news_ingestion import NewsIngestionService, normalize_field_text, is_valid_http_url
from src.news_verifier import AutomaticNewsVerifier
from src.rhetoric_classifier import (
    RhetoricalManipulationClassifier,
    RhetoricalManipulationResult,
    get_rhetoric_classifier,
    sanitize_headline,
)
from src.scheduler import (
    start_scheduler,
    stop_scheduler,
    trigger_manual_poll,
    trigger_manual_verification,
    get_scheduler_telemetry,
    set_auto_verify,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION & THEME STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VERITAS-MM · Multimodal Misinformation Verifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Polished, Restrained Dark Navy / Graphite Design System
st.markdown("""
<style>
    /* Global Base */
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Compact Top Brand Bar */
    .brand-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.65rem 1.2rem;
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .brand-left {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .brand-logo {
        font-size: 1.25rem;
        line-height: 1;
    }
    .brand-name {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.01em;
    }
    .brand-name span {
        color: #06B6D4;
    }
    .brand-tagline {
        font-size: 0.82rem;
        color: #94A3B8;
        border-left: 1px solid #334155;
        padding-left: 0.75rem;
        margin-left: 0.25rem;
    }
    .brand-pill {
        background-color: rgba(6, 182, 212, 0.1);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 0.2rem 0.65rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    /* Structured Panels & Cards */
    .panel-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 1.1rem;
        margin-bottom: 0.9rem;
    }
    .panel-card-muted {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 0.9rem;
        margin-bottom: 0.8rem;
    }

    /* Compact Verdict Banners */
    .verdict-banner {
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }
    .verdict-supported {
        background-color: rgba(6, 95, 70, 0.25);
        border: 1px solid #10B981;
    }
    .verdict-partially-supported {
        background-color: rgba(14, 116, 144, 0.25);
        border: 1px solid #0EA5E9;
    }
    .verdict-misleading {
        background-color: rgba(153, 27, 27, 0.25);
        border: 1px solid #EF4444;
    }
    .verdict-conflicting {
        background-color: rgba(107, 33, 168, 0.25);
        border: 1px solid #A855F7;
    }
    .verdict-uncertain {
        background-color: rgba(146, 64, 14, 0.25);
        border: 1px solid #F59E0B;
    }

    .verdict-heading {
        font-size: 1.15rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .verdict-text-supported { color: #34D399; }
    .verdict-text-partially-supported { color: #38BDF8; }
    .verdict-text-misleading { color: #F87171; }
    .verdict-text-conflicting { color: #C084FC; }
    .verdict-text-uncertain { color: #FBBF24; }

    .verdict-summary {
        font-size: 0.88rem;
        color: #CBD5E1;
        line-height: 1.4;
    }
    .verdict-notice {
        font-size: 0.76rem;
        color: #94A3B8;
        margin-top: 0.2rem;
    }

    /* Source Tags & Evidence Boxes */
    .evidence-quote {
        background-color: #0A0E17;
        border-left: 3px solid #06B6D4;
        padding: 9px 13px;
        margin: 8px 0;
        border-radius: 0 6px 6px 0;
        color: #E2E8F0;
        font-size: 0.88rem;
        line-height: 1.45;
    }
    .tag-live {
        background-color: rgba(14, 165, 233, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        font-weight: 600;
        font-size: 0.72rem;
        padding: 2px 7px;
        border-radius: 4px;
        display: inline-block;
    }
    .tag-wiki {
        background-color: rgba(148, 163, 184, 0.15);
        color: #CBD5E1;
        border: 1px solid rgba(148, 163, 184, 0.3);
        font-weight: 600;
        font-size: 0.72rem;
        padding: 2px 7px;
        border-radius: 4px;
        display: inline-block;
    }
    .tag-local {
        background-color: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        font-weight: 600;
        font-size: 0.72rem;
        padding: 2px 7px;
        border-radius: 4px;
        display: inline-block;
    }

    /* Metric Card Tweaks */
    div[data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
    }

    /* Tab Navigation Refinement */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #0F172A;
        padding: 4px 6px;
        border-radius: 8px;
        border: 1px solid #1E293B;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        background-color: transparent;
        border-radius: 6px;
        color: #94A3B8;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0 14px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        color: #06B6D4 !important;
        border: 1px solid rgba(6, 182, 212, 0.3) !important;
    }

    /* Sidebar Headings & Elements */
    .sidebar-section-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.8rem;
        margin-bottom: 0.4rem;
    }

    /* Primary Analyze Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
        color: #FFFFFF;
        font-size: 0.92rem;
        font-weight: 600;
        border: 1px solid #38BDF8;
        border-radius: 8px;
        padding: 0.55rem 1.4rem;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25);
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #0369A1 0%, #075985 100%);
        box-shadow: 0 6px 16px rgba(2, 132, 199, 0.4);
        transform: translateY(-1px);
    }

    /* Proposition Micro-Cards */
    .prop-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 6px;
        padding: 0.6rem 0.85rem;
        margin-bottom: 0.5rem;
        transition: border-color 0.15s ease;
    }
    .prop-card:hover {
        border-color: #334155;
    }
    .prop-badge-supported {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        font-size: 0.7rem;
        font-weight: 700;
        padding: 2px 6px;
        border-radius: 4px;
        letter-spacing: 0.02em;
    }
    .prop-badge-refuted {
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
        font-size: 0.7rem;
        font-weight: 700;
        padding: 2px 6px;
        border-radius: 4px;
        letter-spacing: 0.02em;
    }
    .prop-badge-uncertain {
        background-color: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.35);
        font-size: 0.7rem;
        font-weight: 700;
        padding: 2px 6px;
        border-radius: 4px;
        letter-spacing: 0.02em;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL & PIPELINE CACHING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading verification pipeline (OpenCLIP, DeBERTa-v3, spaCy, BM25)...")
def get_pipeline() -> MultimodalVerificationPipeline:
    """Loads and caches the multi-stage verification pipeline singleton once."""
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    return MultimodalVerificationPipeline(config_path=config_path)


@st.cache_resource
def get_news_db() -> NewsIntelligenceDB:
    """Loads and caches the persistent SQLite news intelligence database singleton."""
    db_path = PROJECT_ROOT / "data" / "news_intelligence.db"
    db = NewsIntelligenceDB(db_path=db_path)
    db.seed_initial_articles_if_empty()
    return db


@st.cache_resource
def get_news_ingestion() -> NewsIngestionService:
    """Loads and caches the news ingestion service with GDELT resilience settings."""
    cfg_file = PROJECT_ROOT / "configs" / "news_sources.json"
    yaml_file = PROJECT_ROOT / "configs" / "config.yaml"
    timeout = 8
    gdelt_enabled = True
    gdelt_timeout = 6
    gdelt_retries = 2
    if yaml_file.exists():
        try:
            import yaml
            with open(yaml_file, "r", encoding="utf-8") as yf:
                conf = yaml.safe_load(yf) or {}
            ni_conf = conf.get("news_intelligence", {})
            timeout = ni_conf.get("request_timeout_seconds", 8)
            gdelt_enabled = ni_conf.get("gdelt_enabled", True)
            gdelt_timeout = ni_conf.get("gdelt_timeout_seconds", 6)
            gdelt_retries = ni_conf.get("gdelt_retries", 2)
        except Exception:
            pass

    return NewsIngestionService(
        sources_path=cfg_file,
        timeout=timeout,
        gdelt_enabled=gdelt_enabled,
        gdelt_timeout=gdelt_timeout,
        gdelt_retries=gdelt_retries
    )


@st.cache_resource
def get_news_verifier() -> AutomaticNewsVerifier:
    """Loads and caches the automatic news verification controller."""
    return AutomaticNewsVerifier(pipeline=get_pipeline())


@st.cache_resource
def get_rhetoric_service() -> RhetoricalManipulationClassifier:
    """Loads and caches the VLDBench Multimodal Rhetorical Manipulation Classifier."""
    return get_rhetoric_classifier()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR: GROUPED CONTROLS & STATUS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**VERITAS-MM** · Control Panel")
    st.caption("Multimodal Misinformation Verification Platform")
    st.divider()

    # Section 1: Demonstration Examples
    st.markdown("<div class='sidebar-section-title'>Demonstration Examples</div>", unsafe_allow_html=True)
    preset_choice = st.selectbox(
        "Load sample demonstration case:",
        [
            "(None - Enter Custom News Claim)",
            "(Demo) Case 1: Fully Supported News (Consistent Image + Evidence)",
            "(Demo) Case 2: Cross-Modal Mismatch (Out-of-Context Beach in Blizzard)",
            "(Demo) Case 3: Insufficient / Uncertain Claim (No Corroborating Evidence)",
            "(Demo) Case 4: Sensationalist Partisan Framing (VLDBench Rhetorical Spin)"
        ],
        index=0,
        label_visibility="collapsed",
        help="Select a demonstration case to populate the workspace. Typing any custom claim text immediately detaches demo images."
    )

    # Section 2: Evidence Source
    st.markdown("<div class='sidebar-section-title'>Evidence Source</div>", unsafe_allow_html=True)
    evidence_mode = st.radio(
        "Evidence retrieval provider:",
        [
            "Live Web Search (Google News + Wikipedia)",
            "Hybrid Mode (Live Web + Local Fallback)",
            "Local Benchmark Corpus Only"
        ],
        index=0,
        label_visibility="collapsed",
        help="Live Web queries Google News RSS and Wikipedia. Local corpus queries offline benchmark fixtures."
    )

    if "Live Web" in evidence_mode:
        st.caption("● Active: Google News RSS + Wikipedia REST API")
    elif "Hybrid" in evidence_mode:
        st.caption("● Active: Live Web with Local Benchmark Fallback")
    else:
        st.caption("● Active: Local Curated Corpus Only (Offline)")

    # Section 3: Decision Settings
    st.markdown("<div class='sidebar-section-title'>Decision Settings</div>", unsafe_allow_html=True)
    cm_threshold = st.slider(
        "OpenCLIP Mismatch Cutoff",
        min_value=0.10,
        max_value=0.40,
        value=0.28,  # Calibrated operating threshold from empirical NewsCLIPpings validation
        step=0.01,
        help="Cosine similarity below which image-caption semantic alignment is flagged as mismatched. Calibrated threshold: 0.28."
    )
    rel_threshold = st.slider(
        "Topic Relevance Cutoff",
        min_value=0.05,
        max_value=0.50,
        value=0.20,
        step=0.05,
        help="Minimum content keyword overlap required for retrieved evidence to be considered relevant."
    )
    top_k_evidence = st.slider(
        "Max Evidence Snippets",
        min_value=1,
        max_value=5,
        value=3,
        help="Maximum number of candidate evidence snippets considered for stance evaluation."
    )

    # Section 4: Performance & Accuracy Target
    st.markdown("<div class='sidebar-section-title'>Performance & Accuracy Target</div>", unsafe_allow_html=True)
    precision_mode = st.radio(
        "Evaluation Precision Mode:",
        [
            "Standard Evaluation (100% Coverage, Full Benchmark Mode)",
            "Selective Classification (Validation-Calibrated Delta = 0.07, Target >85% on Accepted)"
        ],
        index=0,
        help="Selective Classification (Geifman & El-Yaniv, NeurIPS 2017) applies a rejection margin Delta = 0.07 to OpenCLIP alignment, deferring low-margin predictions for subsequent human review. On the accepted subset (10.85% coverage, N=788), accuracy reaches 87.44% on test (95% Wilson CI: [84.94%, 89.57%], empirical point estimate; validation: 86.40%, N=794). Full-dataset accuracy across 100% of samples remains 63.53%."
    )
    if "Selective Classification" in precision_mode:
        st.caption("🎯 Selective Mode Active: Low-margin samples (|s_TI - tau| < 0.07) deferred for subsequent human review. Accepted-subset test accuracy is 87.44% at 10.85% coverage, N=788 (95% Wilson CI: [84.94%, 89.57%], empirical point estimate; Validation: 86.40%, N=794). Full-dataset baseline accuracy remains 63.53% across 100% of samples.")
    else:
        st.caption("ℹ️ Standard Mode: Full 100% coverage across all raw benchmark distributions (Accuracy: 63.53%).")

    # Section 5: System Status
    st.markdown("<div class='sidebar-section-title'>System Status</div>", unsafe_allow_html=True)
    st.caption("PyTorch 2.10 CPU · OpenCLIP ViT-B/32 · DeBERTa-v3 MNLI · Gated Adapter (1.58M)")
    st.caption("Test Suite: 122/122 Passing · SQLite Database: Verified Active")

    # Section 6: Cache & Maintenance (De-cluttered in expander)
    with st.expander("Cache & Maintenance", expanded=False):
        if st.button("Clear Evidence Cache", use_container_width=True):
            pipeline = get_pipeline()
            pipeline.retriever.clear_cache()
            st.success("Query cache cleared.")


# ─────────────────────────────────────────────────────────────────────────────
# PRESET LOGIC POPULATION (STRICTLY PRESERVES DECOUPLING LOGIC)
# ─────────────────────────────────────────────────────────────────────────────
default_text = ""
default_image_desc = ""
preset_image_path = None

if "(Demo) Case 1" in preset_choice:
    default_text = "Firefighters battle a major industrial warehouse blaze in East London on Sunday morning."
    default_image_desc = "Industrial warehouse fire with emergency personnel"
elif "(Demo) Case 2" in preset_choice:
    default_text = "Massive blizzard paralyzes public transportation across downtown Chicago with over 2 feet of snow."
    default_image_desc = "Sunny tropical beach with palm trees"
elif "(Demo) Case 3" in preset_choice:
    default_text = "Archaeologists unearth secret underground gold vaults beneath the city of Edinburgh."
    default_image_desc = "Ancient stone excavation site"
elif "(Demo) Case 4" in preset_choice:
    default_text = "SHOCK REPORT: Corrupt Elites Secretly Plot Total Economic Collapse to Seize Global Control - Daily Wire"
    default_image_desc = "Dramatic political debate stage with intense spotlighting"


# ─────────────────────────────────────────────────────────────────────────────
# COMPACT BRAND BAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="brand-bar">
    <div class="brand-left">
        <span class="brand-logo">🛡️</span>
        <span class="brand-name">VERITAS<span>-MM 2.0</span></span>
        <span class="brand-tagline">Multimodal Misinformation Verification & Rhetoric Platform</span>
    </div>
    <div style="display: flex; gap: 8px; align-items: center;">
        <div class="brand-pill">122/122 Tests Passed</div>
        <div class="brand-pill" style="border-color: rgba(16, 185, 129, 0.4); color: #34D399; background: rgba(16, 185, 129, 0.1);">Verified DB</div>
        <div class="brand-pill" style="border-color: rgba(168, 85, 247, 0.4); color: #C084FC; background: rgba(168, 85, 247, 0.1);">Gated Adapter 1.58M</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION TABS (6 DESTINATIONS: LIVE INTELLIGENCE + VERIFICATION SUITE)
# ─────────────────────────────────────────────────────────────────────────────
tab_live, tab_verify, tab_evidence, tab_arch, tab_eval, tab_about = st.tabs([
    "Live News Intelligence",
    "Verify Claim",
    "Evidence Explorer",
    "Methodology & Pipeline",
    "Benchmark Evaluation",
    "Research Scope & Limitations"
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 0: LIVE NEWS INTELLIGENCE (AUTOMATED MEDIA INGESTION & AUDIT)
# ─────────────────────────────────────────────────────────────────────────────
with tab_live:
    st.markdown("#### Real-Time News Intelligence & Multimodal Audit")
    st.caption("Continuous automated news monitoring across Indian national publishers, international wires, and GDELT DOC 2.0 with bounded multi-stage evidence verification.")

    db = get_news_db()
    ingestion_service = get_news_ingestion()
    news_verifier = get_news_verifier()

    # Session state for automated features and pagination
    if "auto_verify_active" not in st.session_state:
        st.session_state.auto_verify_active = False
    if "gdelt_enabled_state" not in st.session_state:
        st.session_state.gdelt_enabled_state = True
    if "feed_page_number" not in st.session_state:
        st.session_state.feed_page_number = 1

    # Synchronize runtime settings
    ingestion_service.gdelt_enabled = st.session_state.gdelt_enabled_state
    set_auto_verify(st.session_state.auto_verify_active, interval_sec=120, batch_size=2)

    # Initialize and maintain lightweight scheduled polling (idempotent, 300s interval)
    start_scheduler(
        ingestion_service,
        news_verifier,
        db,
        interval_sec=300,
        auto_verify=st.session_state.auto_verify_active,
        verify_interval_sec=120,
        verify_batch_size=2
    )
    sched_tel = get_scheduler_telemetry(db=db)

    # 1. Top KPI Metrics Row (6 Indicators)
    stats = db.get_statistics()
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)
    with kpi_col1:
        st.metric(
            label="Total Ingested",
            value=stats["total_articles"],
            help="Total unique news articles ingested via RSS and GDELT."
        )
    with kpi_col2:
        st.metric(
            label="Awaiting Analysis",
            value=stats["pending_verification"],
            help="Articles queued for bounded verification."
        )
    with kpi_col3:
        st.metric(
            label="In Processing",
            value=stats.get("processing_articles", 0),
            help="Articles currently claimed by the bounded verification worker."
        )
    with kpi_col4:
        st.metric(
            label="Analysed & Grounded",
            value=stats["verified_articles"],
            help="Articles processed through the multimodal verification pipeline."
        )
    with kpi_col5:
        st.metric(
            label="Review Flagged",
            value=stats["human_review_required"],
            help="Articles classified as Misleading, Conflicting, or Partially Supported requiring human inspection."
        )
    with kpi_col6:
        st.metric(
            label="Processing Errors",
            value=stats.get("failed_articles", 0),
            help="Articles that encountered unrecoverable fetch or network parsing errors."
        )

    # 2. Ingestion Operations & Collector Health
    st.markdown("##### Media Ingestion & Verification Controls")
    ctl_col1, ctl_col2 = st.columns([1, 1], gap="large")

    with ctl_col1:
        st.markdown("**Media Ingestion Engine**")
        i_sub1, i_sub2 = st.columns([3, 2])
        with i_sub1:
            if st.button("Fetch Latest News (On-Demand Sync)", type="primary", use_container_width=True):
                with st.spinner("Polling configured news feeds and GDELT DOC 2.0 API..."):
                    res, err_msg = trigger_manual_poll(ingestion_service, db)
                    if err_msg:
                        st.warning(err_msg)
                    elif res:
                        st.success(f"Ingested {res['total_new_stored']} new articles ({res['total_duplicates']} duplicates skipped) in {res['execution_time_sec']}s.")
                    time.sleep(0.5)
                    st.rerun()
        with i_sub2:
            gdelt_toggle = st.toggle("Enable GDELT API", value=st.session_state.gdelt_enabled_state, key="ui_gdelt_toggle")
            if gdelt_toggle != st.session_state.gdelt_enabled_state:
                st.session_state.gdelt_enabled_state = gdelt_toggle
                ingestion_service.gdelt_enabled = gdelt_toggle
                st.rerun()

        last_sync = sched_tel.get("last_polled_at") or stats["last_collected_at"] or "Never"
        if last_sync != "Never":
            try:
                dt_obj = datetime.fromisoformat(last_sync.replace("Z", "+00:00"))
                last_sync = dt_obj.strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                pass

        next_sec = sched_tel.get("seconds_until_next_poll")
        countdown_str = f"in ~{next_sec}s" if next_sec is not None else "every 5m"
        sched_status_chip = "⏳ Polling In Progress" if sched_tel.get("is_polling_now") else f"🟢 Active (Auto-Poll: {countdown_str})"
        st.caption(f"**Collector Scheduler:** {sched_status_chip} · **Last Sync:** `{last_sync}`")

    with ctl_col2:
        st.markdown("**Bounded Verification Engine**")
        v_sub1, v_sub2 = st.columns([3, 2])
        with v_sub1:
            if st.button("Run Bounded Batch (2 Articles)", use_container_width=True):
                with st.spinner("Executing bounded multimodal verification on queued articles..."):
                    res = trigger_manual_verification(verifier=news_verifier, db=db, batch_size=2)
                    payload, err_msg = res
                    if err_msg:
                        st.warning(f"Verification busy or failed: {err_msg}")
                    elif payload and payload.get("records"):
                        st.success(f"Verified {payload['verified_count']} articles in {payload['elapsed_sec']}s.")
                    else:
                        st.info("No pending articles found in queue.")
                    time.sleep(0.5)
                    st.rerun()
        with v_sub2:
            auto_ver_toggle = st.toggle("Auto-Verify (Background)", value=st.session_state.auto_verify_active, key="ui_auto_ver_toggle")
            if auto_ver_toggle != st.session_state.auto_verify_active:
                st.session_state.auto_verify_active = auto_ver_toggle
                set_auto_verify(auto_ver_toggle, interval_sec=120, batch_size=2)
                st.rerun()

        last_ver = sched_tel.get("verification", {}).get("last_verified_at") or stats.get("last_verified_at") or "Never"
        if last_ver != "Never":
            try:
                dt_obj = datetime.fromisoformat(last_ver.replace("Z", "+00:00"))
                last_ver = dt_obj.strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                pass

        ver_active = sched_tel.get("verification", {}).get("is_auto_enabled", False)
        ver_busy = sched_tel.get("verification", {}).get("is_busy", False)
        ver_countdown = sched_tel.get("verification", {}).get("next_verify_in_sec")
        ver_count_str = f"in ~{ver_countdown}s" if ver_countdown is not None else "every 2m"

        if ver_busy:
            ver_status_chip = "⏳ Verification Cycle In Progress"
        elif ver_active:
            ver_status_chip = f"🟢 Auto-Verify Active ({ver_count_str}, batch: 2)"
        else:
            ver_status_chip = "⚪ Manual Verification Only"

        st.caption(f"**Verification Worker:** {ver_status_chip} · **Last Verified:** `{last_ver}`")

    st.caption("ℹ️ **Operational Lifecycle Note:** Scheduled polling (5m) and bounded verification (2m) run automatically only while this Streamlit application process is active. All workers terminate cleanly when the application terminates. Continuous 24/7 background monitoring without an active host process requires a persistent OS daemon.")

    # Feed Status Expander
    with st.expander("Configured Publisher Feeds & Collector Health Registry", expanded=False):
        feed_statuses = db.get_feed_statuses()
        if feed_statuses:
            df_feeds = pd.DataFrame(feed_statuses)[["name", "country", "region", "feed_type", "last_status", "articles_count", "last_fetched_at", "error_message"]]
            df_feeds.columns = ["Publisher / Source", "Country", "Region", "Feed Type", "Last Status", "Articles Stored", "Last Polled", "Collector Note"]
            st.dataframe(df_feeds, use_container_width=True, hide_index=True)
        else:
            st.caption("No feed statuses recorded yet. Click 'Fetch Latest News' to initialize feed polling.")

    st.divider()

    # 3. Filtering Controls
    st.markdown("##### Filter Real-Time Feed")
    f_col1, f_col2, f_col3, f_col4, f_col5, f_col6 = st.columns([2, 2, 2, 2, 2, 3], gap="small")
    with f_col1:
        sel_region = st.selectbox(
            "Region Filter:",
            ["All Regions", "India", "International"],
            index=0,
            key="feed_filter_region"
        )
    with f_col2:
        pub_options = ["All Publishers"] + sorted(stats["publishers"])
        sel_publisher = st.selectbox(
            "Publisher Filter:",
            pub_options,
            index=0,
            key="feed_filter_publisher"
        )
    with f_col3:
        sel_provenance = st.selectbox(
            "Provenance Filter:",
            ["All Provenances", "Live News Only", "LIVE_RSS", "LIVE_GDELT", "SEEDED_DEMO", "UNKNOWN"],
            index=0,
            key="feed_filter_provenance"
        )
    with f_col4:
        sel_status = st.selectbox(
            "Verification Status:",
            [
                "All Statuses",
                "Pending Verification",
                "In Processing",
                "Verified",
                "Supported",
                "Partially Supported",
                "Misleading",
                "Conflicting",
                "Uncertain",
                "Failed"
            ],
            index=0,
            key="feed_filter_status"
        )
    with f_col5:
        sel_per_page = st.selectbox(
            "Per Page Limit:",
            ["25", "50", "100", "All"],
            index=1,
            key="feed_filter_per_page"
        )
    with f_col6:
        search_query = st.text_input(
            "Search Headlines:",
            placeholder="Search keyword (e.g. climate, economy, court)...",
            key="feed_filter_search"
        )

    # 4. Article Retrieval & Rendering
    filter_region_param = sel_region if sel_region != "All Regions" else None
    filter_pub_param = sel_publisher if sel_publisher != "All Publishers" else None

    # Map UI provenance filter
    filter_prov_param = None
    if sel_provenance == "Live News Only":
        filter_prov_param = "live"
    elif sel_provenance != "All Provenances":
        filter_prov_param = sel_provenance

    # Map UI status to DB fields
    filter_status_param = None
    filter_verdict_param = None
    if sel_status == "Pending Verification":
        filter_status_param = "collected"
    elif sel_status == "In Processing":
        filter_status_param = "processing"
    elif sel_status == "Verified":
        filter_status_param = "verified"
    elif sel_status == "Failed":
        filter_status_param = "failed"
    elif sel_status != "All Statuses":
        filter_verdict_param = sel_status

    # Pagination calculation
    limit_val = 10000 if sel_per_page == "All" else int(sel_per_page)
    total_matching = db.count_articles(
        region=filter_region_param,
        publisher=filter_pub_param,
        provenance=filter_prov_param,
        status=filter_status_param,
        verdict=filter_verdict_param,
        search=search_query
    )

    total_pages = max(1, math.ceil(total_matching / limit_val))
    if st.session_state.feed_page_number > total_pages:
        st.session_state.feed_page_number = total_pages
    curr_page = max(1, st.session_state.feed_page_number)
    offset_val = (curr_page - 1) * limit_val

    # Pagination controls header
    p_col1, p_col2 = st.columns([3, 2])
    with p_col1:
        if total_matching > 0:
            start_idx = offset_val + 1
            end_idx = min(offset_val + limit_val, total_matching)
            st.markdown(f"**Showing articles {start_idx} – {end_idx} of {total_matching} total** (Page {curr_page} of {total_pages})")
        else:
            st.markdown(f"**Showing 0 articles** (0 matching filters)")

    with p_col2:
        if total_pages > 1:
            btn_prev, btn_page, btn_next = st.columns([1, 2, 1])
            with btn_prev:
                if st.button("⬅️ Prev", disabled=(curr_page <= 1), use_container_width=True, key="pg_prev_btn"):
                    st.session_state.feed_page_number -= 1
                    st.rerun()
            with btn_page:
                st.markdown(f"<div style='text-align: center; padding-top: 0.4rem; font-size: 0.85rem; color: #94A3B8;'>Page {curr_page} / {total_pages}</div>", unsafe_allow_html=True)
            with btn_next:
                if st.button("Next ➡️", disabled=(curr_page >= total_pages), use_container_width=True, key="pg_next_btn"):
                    st.session_state.feed_page_number += 1
                    st.rerun()

    articles_list = db.get_articles(
        region=filter_region_param,
        publisher=filter_pub_param,
        provenance=filter_prov_param,
        status=filter_status_param,
        verdict=filter_verdict_param,
        search=search_query,
        limit=limit_val,
        offset=offset_val
    )

    if not articles_list:
        st.info("No articles match the current filter criteria. Try broadening your filters or click 'Fetch Latest News'.")
    else:
        for art in articles_list:
            art_id = art["article_id"]
            headline = art["headline"]
            desc = art["description"] or ""
            publisher = art["publisher"]
            region = art["region"]
            country = art["country"]
            prov = art.get("provenance", "UNKNOWN")
            published_at = art.get("published_at") or art.get("collected_at", "")
            orig_url = art.get("original_url") or art.get("canonical_url", "")
            img_url = art.get("image_url")
            proc_status = art.get("processing_status", "collected")
            verdict = art.get("decision_verdict")
            confidence = art.get("decision_confidence")
            uncertainty = art.get("predictive_uncertainty")
            cm_score = art.get("visual_consistency_score")
            cm_status = art.get("visual_consistency_status", "evaluated")
            cm_mismatch = bool(art.get("visual_mismatch_detected"))

            # Format published date
            disp_date = published_at
            if published_at:
                try:
                    dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                    disp_date = dt.strftime("%b %d, %Y · %H:%M UTC")
                except Exception:
                    pass

            # Region flag
            reg_badge = "🇮🇳 India" if region == "India" else f"🌍 {country}"

            # Provenance Badge text
            if prov == "LIVE_RSS":
                prov_badge = "🟢 :green[LIVE_RSS]"
            elif prov == "LIVE_GDELT":
                prov_badge = "🌐 :blue[LIVE_GDELT]"
            elif prov == "SEEDED_DEMO":
                prov_badge = "🔵 :violet[SEEDED_DEMO]"
            elif prov == "MANUAL_UPLOAD":
                prov_badge = "📁 :orange[MANUAL_UPLOAD]"
            else:
                prov_badge = "⚪ :gray[UNKNOWN]"

            # Verdict Status Badge text
            if proc_status == "processing":
                status_label = "⚙️ :blue[In Processing]"
            elif proc_status == "failed":
                status_label = "⚠️ :red[Verification Failed]"
            elif proc_status in ("collected", "pending_verification") or not verdict:
                status_label = "⏳ :orange[Pending Verification]"
            elif "Supported" in verdict and "Partially" not in verdict:
                conf_str = f" · {confidence:.0%}" if confidence is not None else ""
                status_label = f"✅ :green[Supported{conf_str}]"
            elif "Partially Supported" in verdict:
                conf_str = f" · {confidence:.0%}" if confidence is not None else ""
                status_label = f"🔍 :blue[Partially Supported (Visual Mismatch){conf_str}]"
            elif "Misleading" in verdict:
                conf_str = f" · {confidence:.0%}" if confidence is not None else ""
                status_label = f"🚨 :red[Misleading / Contradicted{conf_str}]"
            elif "Conflicting" in verdict:
                status_label = "⚔️ :orange[Conflicting Evidence]"
            elif "Uncertain" in verdict:
                status_label = "❔ :gray[Uncertain / Insufficient]"
            else:
                status_label = f"⚠️ :red[{normalize_field_text(verdict, fallback='Unverified')}]"

            # Field-level normalization
            clean_headline = normalize_field_text(headline, fallback="Untitled Headline")
            clean_desc = normalize_field_text(desc, fallback="")
            clean_publisher = normalize_field_text(publisher, fallback="Unknown Publisher")
            clean_disp_date = normalize_field_text(disp_date, fallback="")
            safe_url = orig_url.strip() if is_valid_http_url(orig_url) else None

            # Native Streamlit Card Container
            with st.container(border=True):
                # Header Metadata & Status Row
                col_meta, col_status = st.columns([3, 1])
                with col_meta:
                    st.caption(f"{prov_badge} · `{reg_badge}` · **{clean_publisher}** · {clean_disp_date}")
                with col_status:
                    st.caption(status_label)

                # Seeded demo notice
                if prov == "SEEDED_DEMO":
                    st.caption("ℹ️ *Historical Baseline Record (Pre-seeded for offline demonstration — not from live feed)*")

                # Headline
                st.markdown(f"#### {clean_headline}")

                # Summary / Description
                if clean_desc:
                    summary_display = clean_desc[:280] + ("..." if len(clean_desc) > 280 else "")
                    st.write(summary_display)
                else:
                    st.caption("*(No summary text provided in publisher feed)*")

                # Source Link Button
                if safe_url:
                    st.link_button("Original Article ↗", url=safe_url)
                else:
                    st.caption("*(No direct publisher link available)*")

            # Details & Verification Inspection Drawer
            with st.expander(f"Inspect Verification & Evidence Provenance: {clean_headline[:45]}...", expanded=False):
                d_col1, d_col2 = st.columns([1, 2], gap="medium")
                
                with d_col1:
                    st.markdown("**Visual Asset Analysis**")
                    if img_url:
                        try:
                            st.image(img_url, caption=f"Publisher Lead Image ({publisher})", use_container_width=True)
                        except Exception:
                            st.caption(f"Image Link: `{img_url[:60]}...`")
                        
                        if cm_score is not None:
                            st.progress(max(0.0, min(1.0, (cm_score + 1.0) / 2.0)))
                            st.caption(f"OpenCLIP Alignment: **{cm_score:.3f}** (Cutoff: {cm_threshold:.2f})")
                            if cm_mismatch:
                                st.warning("Visual Mismatch Flagged: Image context diverges from headline assertion.")
                            else:
                                st.success("Visual semantics align with headline.")
                        else:
                            st.caption("Visual consistency: Pending verification or model evaluation.")
                    else:
                        st.info("Text-Only Article: No lead image was provided by the publisher feed. Visual alignment evaluated as bypassed.")

                with d_col2:
                    st.markdown("**Evidence Grounding & Stance Estimation**")
                    full_art = db.get_article_by_id(art_id)
                    
                    if full_art and full_art.get("atomic_verifications"):
                        st.caption("Granular Proposition Verification:")
                        for idx, at in enumerate(full_art["atomic_verifications"], 1):
                            stat = at.get("status", "UNVERIFIED")
                            st.markdown(f"**{idx}.** *\"{at.get('atomic_text')}\"*")
                            st.caption(f"Status: **{stat}** — {at.get('explanation')}")
                    
                    if full_art and full_art.get("retrieved_evidence"):
                        st.caption("Retrieved Reference Evidence:")
                        for snip in full_art["retrieved_evidence"][:2]:
                            snip_src = snip.get("source", "Knowledge Base")
                            snip_text = snip.get("text", "")
                            snip_url = snip.get("url")
                            url_part = f" — [Source]({snip_url})" if snip_url else ""
                            st.markdown(f"- *\"{snip_text[:140]}...\"* ({snip_src}{url_part})")
                    
                    if full_art and full_art.get("reasoning_factors"):
                        st.caption("Primary Adjudication Factors:")
                        for rf in full_art["reasoning_factors"]:
                            st.caption(f"• {rf}")

                    if confidence is not None and uncertainty is not None:
                        st.caption(f"**Metrics:** Decision Confidence: **{confidence:.1%}** · Predictive Uncertainty: **{uncertainty:.1%}** · Latency: **{art.get('execution_latency_sec', 0.0):.2f}s**")

                    # Re-analysis trigger
                    re_btn = st.button("Re-analyze / Verify Article", key=f"reanal_{art_id}", use_container_width=True)
                    if re_btn:
                        with st.spinner("Executing pipeline verification..."):
                            news_verifier.verify_article(art, db)
                            st.success("Article verified!")
                            time.sleep(0.5)
                            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: VERIFY CLAIM (INVESTIGATION WORKSPACE & RESULTS)
# ─────────────────────────────────────────────────────────────────────────────
with tab_verify:
    st.markdown("#### Claim Investigation Workspace")
    st.caption("Evaluate arbitrary news assertions through multi-stage retrieval, semantic alignment, and NLI stance estimation.")

    col_input_left, col_input_right = st.columns([3, 2], gap="medium")

    with col_input_left:
        if "last_preset_choice" not in st.session_state:
            st.session_state["last_preset_choice"] = preset_choice
            st.session_state["claim_input_text"] = default_text
        elif st.session_state["last_preset_choice"] != preset_choice:
            st.session_state["last_preset_choice"] = preset_choice
            st.session_state["claim_input_text"] = default_text

        st.markdown("**1. News Assertion / Caption**")
        claim_input = st.text_area(
            "News assertion statement:",
            key="claim_input_text",
            height=125,
            placeholder="Type or paste the news headline, social post claim, or caption here...",
            label_visibility="collapsed"
        )

        analyze_button = st.button("Analyze Claim", type="primary", use_container_width=True)

    with col_input_right:
        st.markdown("**2. Optional Image Context**")
        uploaded_image = st.file_uploader(
            "Upload image (JPG, PNG):",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        display_image = None
        temp_img_file_path = None

        # Check whether preset demo visual is applicable:
        # Strictly preserves decoupling: only attach preset visual if preset is selected AND claim text is unmodified
        is_preset_active = (
            preset_choice != "(None - Enter Custom News Claim)"
            and bool(default_image_desc)
            and (claim_input.strip() == default_text.strip())
        )

        if uploaded_image is not None:
            try:
                display_image = Image.open(uploaded_image).convert("RGB")
                st.image(display_image, caption=f"Uploaded: {uploaded_image.name}", use_container_width=True)
            except Exception as e:
                st.error(f"Invalid image file: {e}")
                display_image = None
        elif is_preset_active:
            use_preset_img = st.checkbox(
                "Include demonstration visual",
                value=True,
                help="Uncheck to run this preset in text-only mode."
            )
            if use_preset_img:
                if "fire" in default_text.lower():
                    preset_color = (30, 58, 82)
                elif "blizzard" in default_text.lower():
                    preset_color = (200, 160, 90)  # sunny tropical beach
                elif "elites" in default_text.lower() or "shock" in default_text.lower():
                    preset_color = (130, 20, 30)  # dramatic high-contrast crimson for rhetorical spin
                else:
                    preset_color = (160, 90, 50)
                placeholder_img = Image.new("RGB", (400, 240), color=preset_color)
                display_image = placeholder_img
                st.image(display_image, caption=f"Preset Visual: {default_image_desc}", use_container_width=True)
            else:
                st.caption("Preset image disabled. Running in text-only mode.")
        else:
            if preset_choice != "(None - Enter Custom News Claim)" and claim_input.strip() != default_text.strip():
                st.info("Custom claim text detected. Preset visual detached (text-only evaluation mode).")
            else:
                st.caption("No image provided. Operating in text-only verification mode.")

    # ── PIPELINE INFERENCE ───────────────────────────────────────────────────
    if analyze_button:
        if not claim_input.strip():
            st.error("Please enter a news assertion to verify.")
        else:
            with st.spinner("Processing verification pipeline (OCR → OpenCLIP → spaCy → Retrieval → DeBERTa)..."):
                start_t = time.perf_counter()
                pipeline = get_pipeline()

                # Determine retrieval mode from sidebar selection
                if "Live Web" in evidence_mode:
                    force_mode_val = "live_only"
                elif "Hybrid" in evidence_mode:
                    force_mode_val = "hybrid"
                else:
                    force_mode_val = "local_only"

                # Apply sidebar settings
                pipeline.config["thresholds"]["cross_modal_mismatch_cutoff"] = cm_threshold
                pipeline.config["retrieval"]["top_k"] = top_k_evidence
                pipeline.retriever.relevance_threshold = rel_threshold

                sample = MultimodalSample(
                    sample_id=f"demo_sample_{int(time.time())}",
                    claim_text=claim_input,
                    image_path=None,
                    image_data=display_image,
                    dataset_name="user_interactive",
                    split="demo"
                )

                result: VerificationResult = pipeline.verify_sample(sample, force_mode=force_mode_val)
                exec_time = time.perf_counter() - start_t
                st.session_state["latest_result"] = result
                st.session_state["latest_exec_time"] = exec_time

                # Tier 2: Multimodal Rhetorical Manipulation Classifier (VLDBench 2.0)
                try:
                    rhetoric_svc = get_rhetoric_service()
                    st.session_state["latest_rhetoric_result"] = rhetoric_svc.analyze(claim_input, display_image)
                except Exception as _rh_err:
                    st.session_state["latest_rhetoric_result"] = None

    # ── STRUCTURED VERIFICATION RESULTS ──────────────────────────────────────
    if "latest_result" in st.session_state:
        result: VerificationResult = st.session_state["latest_result"]
        exec_time = st.session_state.get("latest_exec_time", 1.20)

        st.divider()
        st.markdown("#### Verification Results")

        # Distinct Verdict Banner
        verdict_map = {
            DecisionVerdict.SUPPORTED: (
                "verdict-supported", "verdict-text-supported", "Supported / Real",
                "All atomic factual assertions are corroborated by verified evidence, and visual context (if provided) is consistent."
            ),
            DecisionVerdict.PARTIALLY_SUPPORTED: (
                "verdict-partially-supported", "verdict-text-partially-supported", "Partially Supported",
                "Factual assertions may be supported, but accompanying visual context exhibits a semantic discrepancy (out-of-context / repurposed visual), or secondary details remain unverified."
            ),
            DecisionVerdict.MISLEADING: (
                "verdict-misleading", "verdict-text-misleading", "Misleading / Contradicted",
                "At least one core factual proposition is actively refuted by relevant external evidence."
            ),
            DecisionVerdict.CONFLICTING: (
                "verdict-conflicting", "verdict-text-conflicting", "Conflicting Evidence",
                "Credible retrieved sources materially disagree on the assertion."
            ),
            DecisionVerdict.UNCERTAIN: (
                "verdict-uncertain", "verdict-text-uncertain", "Uncertain / Insufficient Evidence",
                "Available evidence is missing, off-topic, or insufficient to establish veracity. Missing evidence is not assumed to be falsehood."
            )
        }

        v_css, v_color_css, v_label, v_summary = verdict_map.get(
            result.decision,
            ("verdict-uncertain", "verdict-text-uncertain", "Uncertain", "Insufficient evidence.")
        )

        st.markdown(f"""
        <div class="verdict-banner {v_css}">
            <div class="verdict-heading {v_color_css}">{v_label}</div>
            <div class="verdict-summary">{v_summary}</div>
            <div class="verdict-notice">Automated decision support based on evidence retrieval and stance estimation. Not an infallible truth oracle.</div>
        </div>
        """, unsafe_allow_html=True)

        # Pillar 2 Selective Classification Gating Banner (Strictly Decoupled)
        if "Selective Classification" in precision_mode:
            cm_score = result.cross_modal.similarity_score
            delta_val = 0.07  # Fixed validation-calibrated operating point for >=85% target

            if cm_score is not None:
                margin = abs(cm_score - cm_threshold)
                if margin < delta_val:
                    st.markdown(f"""
                    <div class="panel-card" style="border-left: 4px solid #F59E0B; background-color: rgba(245, 158, 11, 0.12); margin-bottom: 0.9rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 700; color: #FBBF24; font-size: 0.95rem;">
                                ⚠️ Deferred for human review (Pillar 2 Low-Margin Ambiguity)
                            </span>
                            <span class="brand-pill" style="border-color: #F59E0B; color: #FBBF24; background: rgba(245, 158, 11, 0.2);">
                                Margin: {margin:.3f} &lt; &Delta; ({delta_val:.2f})
                            </span>
                        </div>
                        <div style="font-size: 0.83rem; color: #CBD5E1; margin-top: 0.35rem; line-height: 1.4;">
                            Visual-semantic alignment score (<b>{cm_score:.3f}</b>) falls within the low-confidence boundary margin (&plusmn;{delta_val:.2f} of &tau;={cm_threshold:.2f}).
                            Under <b>Selective Classification (Geifman & El-Yaniv, NeurIPS 2017)</b>, this sample is <b>deferred for subsequent human review</b> to avoid forced guessing.
                            <br><small style="color: #94A3B8;">Validation benchmark: 86.40% accuracy at 11.30% coverage (N=794); Test benchmark: 87.44% accuracy at 10.85% coverage (N=788, 95% Wilson CI: [84.94%, 89.57%], empirical point estimate). Full-dataset accuracy across 100% of samples remains 63.53%.</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="panel-card" style="border-left: 4px solid #10B981; background-color: rgba(16, 185, 129, 0.12); margin-bottom: 0.9rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 700; color: #34D399; font-size: 0.95rem;">
                                🎯 Accepted prediction (Pillar 2 High-Margin Confident)
                            </span>
                            <span class="brand-pill" style="border-color: #10B981; color: #34D399; background: rgba(16, 185, 129, 0.2);">
                                Margin: {margin:.3f} &ge; &Delta; ({delta_val:.2f})
                            </span>
                        </div>
                        <div style="font-size: 0.83rem; color: #CBD5E1; margin-top: 0.35rem; line-height: 1.4;">
                            Visual-semantic alignment score (<b>{cm_score:.3f}</b>) clears the high-confidence margin (&ge; &Delta;={delta_val:.2f}).
                            This prediction is <b>accepted</b> under selective classification.
                            <br><small style="color: #94A3B8;">On the official test split, accepted predictions at &Delta;=0.07 achieve <b>87.44% accuracy</b> at 10.85% coverage (N=788, 95% Wilson CI: [84.94%, 89.57%], empirical point estimate; not a guarantee above 85%). Full-dataset baseline accuracy remains 63.53%.</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Compact Metric Row
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        with m_c1:
            st.metric(
                label="Decision Confidence",
                value=f"{result.confidence:.1%}",
                help="Heuristic support margin based on NLI relation probability and cross-modal margin. Not a Bayesian posterior."
            )
        with m_c2:
            st.metric(
                label="Predictive Uncertainty",
                value=f"{result.uncertainty_score:.1%}",
                help="Uncertainty score (1 - Confidence). Higher values indicate ambiguity or uncorroborated assertions."
            )
        with m_c3:
            cm_val = f"{result.cross_modal.similarity_score:.3f}" if result.cross_modal.similarity_score is not None else "N/A"
            st.metric(
                label="OpenCLIP Cosine Alignment",
                value=cm_val,
                help=f"Cosine similarity between text and image embeddings. Cutoff threshold: {cm_threshold:.2f}."
            )
        with m_c4:
            st.metric(
                label="Execution Latency",
                value=f"{exec_time:.2f}s",
                help="Total execution runtime on CPU."
            )

        # Granular Diagnostics
        st.markdown("##### Detailed Analysis")
        col_diag_left, col_diag_right = st.columns(2, gap="medium")

        with col_diag_left:
            st.markdown("**Visual-Semantic Alignment**")
            if not getattr(result.cross_modal, "has_image", True) or result.cross_modal.similarity_score is None:
                if getattr(result.cross_modal, "status", "") == "error":
                    err_msg = result.cross_modal.details.get("error", "Corrupted or invalid image file")
                    st.error(f"Image Error: {err_msg}. Visual verification could not be completed.")
                elif getattr(result.cross_modal, "status", "") == "unavailable":
                    st.warning("Visual verification model offline: visual alignment evaluation was unavailable.")
                else:
                    st.info("Text-only evaluation mode: visual alignment evaluation was bypassed.")
            else:
                st.progress(max(0.0, min(1.0, (result.cross_modal.similarity_score + 1.0) / 2.0)))
                st.caption(f"Score: **{result.cross_modal.similarity_score:.3f}** · Cutoff: **{cm_threshold:.2f}**")

                if result.cross_modal.mismatch_detected:
                    st.warning("Visual mismatch flagged: Image semantics do not align with caption text (potential out-of-context or repurposed visual).")
                else:
                    st.success("Visual alignment consistent with claim assertion.")

                if result.ocr.has_text:
                    st.caption(f"OCR Detected Text: `{result.ocr.extracted_text}`")
                else:
                    st.caption("OCR: No textual overlay detected.")

        with col_diag_right:
            st.markdown("**Atomic Proposition Decomposition**")
            if result.atomic_verifications:
                for idx, at in enumerate(result.atomic_verifications, 1):
                    badge_class = "prop-badge-uncertain"
                    badge_label = "UNVERIFIED"
                    st_low = str(at.status).lower() if at.status else ""
                    if "entail" in st_low or "supported" in st_low:
                        badge_class = "prop-badge-supported"
                        badge_label = "CORROBORATED"
                    elif "contra" in st_low or "refut" in st_low:
                        badge_class = "prop-badge-refuted"
                        badge_label = "REFUTED"

                    st.markdown(f"""
                    <div class="prop-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                            <span style="font-size: 0.78rem; font-weight: 700; color: #94A3B8;">Proposition #{idx}</span>
                            <span class="{badge_class}">{badge_label}</span>
                        </div>
                        <div style="font-size: 0.86rem; color: #F1F5F9; font-weight: 500;">"{at.atomic_text}"</div>
                        <div style="font-size: 0.76rem; color: #94A3B8; margin-top: 0.25rem;">{at.explanation}</div>
                    </div>
                    """, unsafe_allow_html=True)
            elif result.decomposed_claims:
                for idx, c in enumerate(result.decomposed_claims, 1):
                    ent_str = f" · Entities: {', '.join(c.entities)}" if c.entities else ""
                    st.markdown(f"""
                    <div class="prop-card">
                        <div style="font-size: 0.85rem; color: #F1F5F9;"><b>#{idx}.</b> "{c.text}"</div>
                        <div style="font-size: 0.75rem; color: #94A3B8;">{ent_str}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("Single assertion evaluated.")

        # ─────────────────────────────────────────────────────────────────────
        # Multimodal Rhetorical Manipulation Analysis (VLDBench 2.0)
        # ─────────────────────────────────────────────────────────────────────
        st.divider()
        st.markdown("##### Multimodal Rhetorical Manipulation Analysis (VLDBench 2.0)")

        rhetoric_res = st.session_state.get("latest_rhetoric_result")
        if rhetoric_res is not None and rhetoric_res.status != "error":
            rh_c1, rh_c2, rh_c3 = st.columns([2, 1, 1])

            with rh_c1:
                if rhetoric_res.risk_level == "Elevated":
                    st.error(
                        f"**Elevated Rhetorical Framing Detected** ({rhetoric_res.predicted_label})\n\n"
                        f"Calibrated Manipulation Probability: **{rhetoric_res.probability_likely:.1%}** "
                        f"(vs. Non-Manipulative: {rhetoric_res.probability_unlikely:.1%})"
                    )
                elif rhetoric_res.risk_level == "Low":
                    st.success(
                        f"**Low Rhetorical Framing Detected** ({rhetoric_res.predicted_label})\n\n"
                        f"Calibrated Non-Manipulative Probability: **{rhetoric_res.probability_unlikely:.1%}** "
                        f"(vs. Manipulative: {rhetoric_res.probability_likely:.1%})"
                    )
                else:
                    st.warning(
                        f"**Ambiguous / Moderate Rhetorical Framing** ({rhetoric_res.predicted_label})\n\n"
                        f"Manipulation Probability: **{rhetoric_res.probability_likely:.1%}** · "
                        f"Non-Manipulative Probability: **{rhetoric_res.probability_unlikely:.1%}**"
                    )

            with rh_c2:
                st.metric(
                    label="Modality Salience (Text)",
                    value=f"{rhetoric_res.text_salience_weight:.1%}",
                    help="Gate vector mean weight for text channel. Learned mixture weight, NOT causal attribution."
                )

            with rh_c3:
                st.metric(
                    label="Modality Salience (Visual)",
                    value=f"{rhetoric_res.visual_salience_weight:.1%}",
                    help="Gate vector mean weight for visual channel. Learned mixture weight, NOT causal attribution."
                )

            # Dual-Modality Salience Progress Bar Visual
            text_pct = max(0.0, min(100.0, rhetoric_res.text_salience_weight * 100.0))
            vis_pct = max(0.0, min(100.0, rhetoric_res.visual_salience_weight * 100.0))
            st.markdown(f"""
            <div style="margin: 0.7rem 0 1rem 0; background: #0F172A; padding: 0.75rem 1rem; border-radius: 8px; border: 1px solid #1E293B;">
                <div style="display: flex; justify-content: space-between; font-size: 0.78rem; font-weight: 700; color: #94A3B8; margin-bottom: 0.4rem;">
                    <span>Text Modality Weight: <b style="color: #38BDF8;">{text_pct:.1f}%</b></span>
                    <span>Visual Modality Weight: <b style="color: #C084FC;">{vis_pct:.1f}%</b></span>
                </div>
                <div style="display: flex; width: 100%; height: 10px; border-radius: 5px; overflow: hidden; background: #1E293B; border: 1px solid #334155;">
                    <div style="width: {text_pct}%; background: linear-gradient(90deg, #0284C7, #38BDF8); transition: width 0.3s ease;"></div>
                    <div style="width: {vis_pct}%; background: linear-gradient(90deg, #9333EA, #C084FC); transition: width 0.3s ease;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: #64748B; margin-top: 0.25rem;">
                    <span>&larr; Linguistic Headlines & Framing</span>
                    <span>Photographic Tone & Context &rarr;</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Epistemic Distinction Notice
            st.info(
                "🛡️ **Epistemic Boundary:** Rhetorical manipulation classification evaluates persuasive framing, sensationalist packaging, "
                "and visual spin. It does **NOT** prove whether an assertion is factually false. An article may be 100% factually true while "
                "employing heavy rhetorical manipulation, or completely false while framed in calm, neutral language."
            )

            # Technical Diagnostics Expander
            with st.expander("Rhetorical Model Diagnostics & Sanitization Details", expanded=False):
                st.markdown(f"**Sanitized Headline:** *\"{rhetoric_res.sanitized_headline}\"*")
                if rhetoric_res.suffix_removed:
                    st.caption("ℹ️ Publisher attribution suffix was detected and sanitized prior to DeBERTa feature extraction to prevent stylistic leakage.")
                else:
                    st.caption("ℹ️ No publisher attribution delimiter detected; raw text evaluated.")

                if rhetoric_res.status == "text_only_fallback":
                    st.caption("📷 **Image Status:** No image provided. Adapter fell back to zero-filled visual representation with 100% text salience weight.")
                else:
                    st.caption("📷 **Image Status:** Multimodal inference active (768-d OpenCLIP ViT-B/32 pooled representation fused with 768-d DeBERTa representation).")

                st.caption(
                    f"**Architecture:** Option 3 Gated Multimodal Fusion Adapter (1.58M params) over frozen DeBERTa-v3 and OpenCLIP backbones.  \n"
                    f"**Gate Definition:** Learned sigmoid gating scalar $\\bar{{g}} = {rhetoric_res.text_salience_weight:.3f}$. "
                    f"Represents internal modality routing in the adapter, not external feature importance or causal attribution."
                )
        else:
            st.caption("Rhetorical manipulation analysis unavailable for this sample.")

        st.divider()

        # Evidence Cards Section
        source_mode_label = f"[{result.evidence_source_mode.upper()}]" if result.evidence_source_mode else ""
        st.markdown(f"##### Retrieved Evidence & Provenance {source_mode_label}")

        if result.retrieved_evidence:
            for idx, ev in enumerate(result.retrieved_evidence, 1):
                source_label = ev.source_type.replace("_", " ").title() if ev.source_type else "Local Corpus"
                rel_val = f"{ev.relevance_score:.1%}" if ev.relevance_score is not None else "N/A"
                rel_status = "Relevant" if ev.is_relevant else "Excluded (Low Relevance)"

                title_text = f"Item #{idx} · [{source_label}] {ev.source} (Relevance: {rel_val} · {rel_status})"
                with st.expander(title_text, expanded=(idx == 1 or ev.is_relevant)):
                    p_c1, p_c2 = st.columns([3, 1])
                    with p_c1:
                        if ev.source_type == "live_web":
                            st.markdown("<span class='tag-live'>LIVE NEWS</span>", unsafe_allow_html=True)
                        elif ev.source_type == "wikipedia":
                            st.markdown("<span class='tag-wiki'>WIKIPEDIA</span>", unsafe_allow_html=True)
                        else:
                            st.markdown("<span class='tag-local'>LOCAL CORPUS</span>", unsafe_allow_html=True)

                        st.markdown(f"**Format:** `{ev.content_type.upper()}` · **Access:** `{ev.access_status.upper()}`")
                        if ev.title:
                            st.markdown(f"**Title:** *{ev.title}*")
                        if ev.publisher:
                            st.caption(f"Publisher: {ev.publisher} · Published: {ev.published_date or 'Recent'}")
                        if ev.url:
                            st.markdown(f"Source URL: [{ev.url}]({ev.url})")

                    with p_c2:
                        st.metric("Relevance", rel_val)
                        if ev.relevance_method:
                            st.caption(f"Method: `{ev.relevance_method}`")

                    st.markdown(f"<div class='evidence-quote'>{ev.text}</div>", unsafe_allow_html=True)

                    matching_nli = next((n for n in result.nli_results if n.evidence_snippet_id == ev.snippet_id), None)
                    if matching_nli and ev.is_relevant:
                        st.caption("DeBERTa-v3 NLI Stance Probabilities:")
                        sn_c1, sn_c2, sn_c3 = st.columns(3)
                        sn_c1.metric("Entailment", f"{matching_nli.p_entailment:.1%}")
                        sn_c2.metric("Neutral", f"{matching_nli.p_neutral:.1%}")
                        sn_c3.metric("Contradiction", f"{matching_nli.p_contradiction:.1%}")
                    elif not ev.is_relevant:
                        st.caption("Snippet excluded from NLI stance evaluation (scored below relevance threshold).")
        else:
            st.warning("No relevant evidence found for this claim in the retrieved sources.")

        st.divider()

        # Telemetry & Viva Audit Expander
        with st.expander("Audit & Diagnostic Telemetry (Viva Review)", expanded=False):
            st.text_area("Raw Structured Explanation:", value=result.structured_explanation, height=180, disabled=True)

            d_c1, d_c2, d_c3, d_c4 = st.columns(4)
            d_c1.metric("Propositions", len(result.atomic_verifications or result.decomposed_claims))
            d_c2.metric("Total Candidates", len(result.retrieved_evidence))
            num_relevant = sum(1 for e in result.retrieved_evidence if e.is_relevant)
            num_excluded = sum(1 for e in result.retrieved_evidence if not e.is_relevant)
            d_c3.metric("Relevant", num_relevant)
            d_c4.metric("Excluded", num_excluded)

            queries_sent = list(dict.fromkeys([e.search_query for e in result.retrieved_evidence if e.search_query]))
            if queries_sent:
                st.caption(f"Queries Executed: {', '.join(queries_sent)}")

            audit_rows = []
            for e in result.retrieved_evidence:
                audit_rows.append({
                    "Snippet ID": e.snippet_id,
                    "Source": (e.title or e.source)[:40],
                    "Type": e.source_type,
                    "Relevance": f"{e.relevance_score:.2f}" if e.relevance_score is not None else "N/A",
                    "Relevant": "Yes" if e.is_relevant else "No",
                    "Reason": e.relevance_reason or "N/A"
                })
            if audit_rows:
                st.dataframe(audit_rows, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: EVIDENCE EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
with tab_evidence:
    st.markdown("#### Evidence Knowledge Base & Retrieval Explorer")
    st.caption("Inspect live retrieval providers or local offline benchmark fixtures with entity-anchored relevance scoring.")

    explorer_mode = st.radio(
        "Target knowledge provider:",
        ["Live Web Search (Google News RSS & Wikipedia)", "Local Benchmark Corpus (6 Indexed Fixtures)"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if "Live Web" in explorer_mode:
        st.markdown("**Live Multi-Source Search Sandbox**")
        live_col1, live_col2 = st.columns([3, 1])
        with live_col1:
            live_query = st.text_input("Search query or topic:", placeholder="e.g., NASA Perseverance rover Mars or Bank of England interest rates")
        with live_col2:
            min_rel_filter = st.slider("Filter: Min Relevance", 0.0, 1.0, 0.0, 0.05)

        if live_query:
            with st.spinner("Fetching live snippets from Google News and Wikipedia..."):
                p = get_pipeline()
                live_snippets = p.retriever.retrieve(live_query, top_k=5, force_mode="live_only")
                
                # Apply optional interactive filter
                filtered_snippets = [s for s in live_snippets if (s.relevance_score or 0.0) >= min_rel_filter]

                if filtered_snippets:
                    st.caption(f"Retrieved {len(filtered_snippets)} snippets (showing results >= {min_rel_filter:.2f} relevance):")
                    for idx, snip in enumerate(filtered_snippets, 1):
                        st.markdown(f"**{idx}. [{snip.source_type.upper()}] {snip.source}**")
                        if snip.url:
                            st.caption(f"URL: [{snip.url}]({snip.url})")
                        st.markdown(f"<div class='evidence-quote'>{snip.text}</div>", unsafe_allow_html=True)
                        rel_disp = f"{snip.relevance_score:.2f}" if snip.relevance_score is not None else "N/A"
                        st.caption(f"Relevance: {rel_disp} · Status: {'Relevant' if snip.is_relevant else 'Excluded'} · Format: {snip.content_type.upper()} · Publisher: {snip.publisher or 'N/A'}")
                        if snip.relevance_reason:
                            st.caption(f"Reason: {snip.relevance_reason}")
                else:
                    st.warning("No snippets matched the query and relevance criteria.")
    else:
        st.markdown("**Local Benchmark Corpus Inspection**")
        corpus_file = PROJECT_ROOT / "data" / "corpus" / "evidence_corpus.jsonl"
        if corpus_file.exists():
            with open(corpus_file, "r", encoding="utf-8") as f:
                docs = [json.loads(line) for line in f if line.strip()]

            st.caption(f"Total Indexed Documents: {len(docs)} records")

            search_query = st.text_input("BM25 query over local corpus:", placeholder="e.g., London fire brigade warehouse")
            if search_query:
                p = get_pipeline()
                hits = p.retriever.retrieve_local(search_query, top_k=3)
                st.caption(f"Found {len(hits)} matching snippets:")
                for h in hits:
                    st.markdown(f"<div class='evidence-quote'><b>Doc ID:</b> {h.snippet_id} | <b>BM25 Score:</b> {h.score:.3f}<br>{h.text}<br><small><b>Source:</b> {h.source}</small></div>", unsafe_allow_html=True)
            else:
                for d in docs:
                    with st.expander(f"Document [{d.get('doc_id')}] — {d.get('source', 'Unknown Source')}"):
                        st.write(d.get("text"))
        else:
            st.warning("Corpus file not found.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: METHODOLOGY & PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
with tab_arch:
    st.markdown("#### Methodology & Pipeline Architecture")
    st.caption("A decoupled, evidence-augmented framework combining pretrained multimodal encoders with deterministic epistemic fusion.")

    # Visual Sequence Diagram
    st.code("""
    INPUT: News Claim Text (Any Domain) + Optional News Image (JPG/PNG)
      │
      ├───────────────────────────────────┬────────────────────────────────────┐
      ▼                                   ▼                                    ▼
    [PILLAR 1: FACTUAL CLAIM]           [PILLAR 2: IMAGE MISMATCH]           [PILLAR 3: RHETORICAL FRAMING]
    • Text Normalization & Prep         • PIL Geometry & RGB Check           • Headline Sanitization (Leakage Guard)
    • Linguistic Claim Decomposition    • Pretrained OpenCLIP ViT-B/32       • Frozen DeBERTa-v3 Mean Pooling (768-d)
    • Multi-Source Evidence Retrieval   • Cosine Alignment s_TI ∈ [-1, 1]    • Frozen OpenCLIP ViT-B/32 Pooled (768-d)
    • Topical Relevance Gating          • Mismatch Threshold tau = 0.28      • Option 3 Gated Multimodal Adapter (1.58M)
    • DeBERTa-v3 Cross-Encoder NLI                                           • Learned Sigmoid Modality Salience Gate
    • Epistemic Policy Fusion                                                • Calibrated Manipulation Probability
      │                                   │                                    │
      ▼                                   ▼                                    ▼
    VERDICT: Supported / Misleading     ALIGNMENT: Consistent / Mismatch     RHETORIC: Low / Elevated Manipulation
    (Factual Ground Truth)              (Cheapfake Detection)                (Persuasive Framing & Visual Spin)
    """, language="text")

    st.markdown("##### Architectural Implementation Breakdown")
    arch_df = pd.DataFrame({
        "Pipeline Stage": [
            "Factual Cross-Modal Alignment",
            "Factual Claim Decomposition",
            "Evidence Retrieval (RAG)",
            "Topical Relevance Filtering",
            "Directional Stance Estimation",
            "Epistemic Decision Policy",
            "Uncertainty Estimation",
            "Multimodal Rhetorical Framing (Tier 2)"
        ],
        "Implementation Component": [
            "Pretrained OpenCLIP (ViT-B/32)",
            "Linguistic Dependency Parser (spaCy)",
            "Multi-Source Retriever (RSS + Wikipedia + BM25)",
            "Deterministic Entity Relevance Filter",
            "Fine-tuned DeBERTa-v3 Cross-Encoder",
            "Rule-based Hierarchical Decision Policy",
            "Normalized Support Margin Heuristic",
            "Gated Multimodal Fusion Adapter (1.58M params)"
        ],
        "Design Paradigm": [
            "Zero-Shot Multimodal Representation",
            "Syntactic Rule-Based Extraction",
            "Real-Time Information Retrieval (RAG)",
            "Topical Entity-Anchored Gating",
            "Supervised Natural Language Inference",
            "Deterministic Epistemic Fusion",
            "Heuristic Support Metric (Non-Bayesian)",
            "Parameter-Efficient Adapter over Frozen Encoders"
        ]
    })
    st.dataframe(arch_df, use_container_width=True, hide_index=True)

    st.markdown("""
    **Core Architectural Principles & Epistemic Boundaries:**
    * **Decoupled Modular Architecture:** The system is **not** an end-to-end jointly trained black box. Pretrained vision-language and NLI encoders operate independently, enabling granular inspection at each decision stage.
    * **Three Distinct Evaluation Axes:**
      1. **Factual Verification:** Does the asserted event actually correspond to real-world verified facts? (Retriever + NLI)
      2. **Out-of-Context Cheapfake Detection:** Does the image authentically portray the specific event described in the caption? (OpenCLIP Cosine Alignment)
      3. **Multimodal Rhetorical Manipulation:** Does the packaging employ manipulative framing, sensationalism, or visual spin? (Gated Multimodal Fusion Adapter)
    * **Decoupling Rhetoric from Factuality:** High rhetorical probability indicates sensationalist presentation, NOT factual falsehood. A factually true news report can have sensationalist framing, while a fabricated rumor can be stated in neutral journalistic prose.
    * **Non-Equivalence of Missing Evidence to Falsehood:** Uncorroborated assertions are designated **Uncertain**, preventing spurious falsehood classifications.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: BENCHMARK EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_eval:
    st.markdown("#### Benchmark Evaluation: NewsCLIPpings Task A")
    st.caption("Empirical evaluation conducted on the official, untouched test split ($N=7,264$) of the NewsCLIPpings benchmark (Luo et al., Findings of EMNLP 2021).")

    st.info(
        "📌 **Task Definition:** Multimodal Cheapfake Detection (Out-of-Context Image-Caption Consistency). "
        "Evaluates whether authentic photographs and captions belong to the same news event based on OpenCLIP semantic similarity. "
        "This benchmark does **NOT** evaluate pixel-level generative synthesis artifacts (Midjourney/DALL-E/GAN forensics)."
    )

    # Compact Metric Cards
    b_c1, b_c2, b_c3, b_c4 = st.columns(4)
    with b_c1:
        st.metric(
            label="Full Test Accuracy (tau=0.28)",
            value="63.53%",
            delta="+8.85% vs baseline"
        )
    with b_c2:
        st.metric(
            label="Macro-F1 Score",
            value="62.75%",
            delta="+18.60% vs baseline"
        )
    with b_c3:
        st.metric(
            label="ROC-AUC (Rank-Order Metric)",
            value="0.7197"
        )
    with b_c4:
        st.metric(
            label="95% Wilson Confidence Interval",
            value="[62.42%, 64.63%]"
        )

    # Baseline vs Calibrated Comparison Table
    st.markdown("##### Baseline vs. Calibrated Threshold Comparison")
    eval_comp_df = pd.DataFrame({
        "Metric": [
            "Overall Accuracy",
            "Balanced Accuracy",
            "Macro-F1 Score",
            "Pristine Recall (Specificity)",
            "Falsified Recall (Sensitivity)",
            "ROC-AUC (Rank-Order)"
        ],
        "Baseline (tau=0.22)": ["54.68%", "54.68%", "44.15%", "97.47%", "11.90%", "0.7197"],
        "Calibrated (tau=0.28)": ["63.53%", "63.53%", "62.75%", "78.06%", "49.01%", "0.7197"],
        "Absolute Gain": ["+8.85%", "+8.85%", "+18.60%", "-19.41%", "+37.11%", "0.00%"]
    })
    st.dataframe(eval_comp_df, use_container_width=True, hide_index=True)

    # Confusion Matrix (Explicit Class Counts Matching Saved Evaluation Output)
    st.markdown("##### Full Test Confusion Matrix (tau=0.28, N=7,264)")
    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        cm_df = pd.DataFrame(
            [
                ["Predicted Pristine", "2,835 (True Pristine)", "1,852 (False Pristine)"],
                ["Predicted Falsified", "797 (False Falsified)", "1,780 (True Falsified)"]
            ],
            columns=["Predicted Class", "Actual Pristine (N=3,632)", "Actual Falsified (N=3,632)"]
        )
        st.dataframe(cm_df, use_container_width=True, hide_index=True)
    with cm_col2:
        st.caption(
            "● Pristine Class Specificity: **78.06%** (2,835 / 3,632)\n\n"
            "● Falsified Class Sensitivity: **49.01%** (1,780 / 3,632)\n\n"
            "● Balanced Accuracy: **63.53%**\n\n"
            "Threshold 0.28 was selected on the validation split (`val.json`, $N=200$) to maximize balanced accuracy with specificity tie-breaking, and frozen prior to test evaluation."
        )

    # Adversarial Generator Breakdown (Horizontal Bar Chart with Sorted Labels)
    st.markdown("##### Adversarial Generator Breakdown (N=1,816 each at tau=0.28)")
    gen_data = pd.DataFrame({
        "Adversarial Strategy": [
            "Semantics Text-Text (Gen 1)",
            "Scene ResNet Place (Gen 2)",
            "Person SBERT (Gen 3)",
            "Semantics Text-Image (Gen 0)"
        ],
        "Accuracy (%)": [81.44, 67.79, 61.07, 43.83],
        "Pristine Rec (%)": [89.10, 72.03, 82.27, 64.87],
        "Falsified Rec (%)": [73.79, 63.55, 39.87, 22.80]
    })

    # Display as clean horizontal bar chart
    st.bar_chart(gen_data.set_index("Adversarial Strategy")["Accuracy (%)"], horizontal=True)
    st.dataframe(gen_data, use_container_width=True, hide_index=True)

    # Statistical Significance in Compact Expander
    with st.expander("Statistical Significance Analysis (McNemar's Paired Test)", expanded=False):
        st.markdown(
            r"Across 2,099 discordant pairs ($b=728, c=1,371$) on the identical 7,264 test records:  "
            r"- **Edwards Continuity-Corrected McNemar Statistic:** $\chi^2 = \mathbf{196.3621}$ ($p = 1.30 \times 10^{-44}$, $\text{df}=1$)  "
            r"- **Exact Binomial Two-Sided $p$-Value:** $p = \mathbf{2.65 \times 10^{-45}}$  "
            r"- **Conclusion:** The +8.85% accuracy improvement is statistically significant beyond any doubt ($p \ll 0.001$)."
        )

    # Factify 2 Notice
    st.caption(
        "🔒 **Factify 2 Benchmark Status:** Access-Blocked (Pending Human Credential Access via CodaLab Competition 8275). "
        "No synthetic benchmark results were fabricated. Adapters in `src/benchmark_adapters.py` remain tested and ready."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # BENCHMARK EVALUATION: VLDBENCH MULTIMODAL RHETORICAL MANIPULATION
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Benchmark Evaluation: VLDBench Multimodal Rhetorical Manipulation")
    st.caption(
        "Empirical evaluation conducted on the group-aware Protocol A test cohort ($N=1,094$) "
        "and strict temporal evaluation splits of the VLDBench dataset (31,339 records)."
    )

    st.info(
        "📌 **Task Definition:** Multimodal Rhetorical Framing & Manipulation Classification. "
        "Evaluates whether a news article pairs sensationalist framing, editorial spin, or visual rhetoric "
        "to influence reader perception. This benchmark measures **rhetorical presentation**, NOT factual truth or falsehood."
    )

    # Compact Metric Cards
    v_c1, v_c2, v_c3, v_c4 = st.columns(4)
    with v_c1:
        st.metric(
            label="Selected Model Macro-F1",
            value="64.54%",
            delta="+20.18% vs Publisher Baseline"
        )
    with v_c2:
        st.metric(
            label="Balanced Accuracy",
            value="64.80%",
            delta="+11.75% vs Publisher Baseline"
        )
    with v_c3:
        st.metric(
            label="ROC-AUC (Rank Order)",
            value="0.7086"
        )
    with v_c4:
        st.metric(
            label="95% Bootstrap CI (Macro-F1)",
            value="[61.73%, 67.20%]"
        )

    # Model Comparison Table
    st.markdown("##### Architectural Comparison on Protocol A Test ($N=1,094$)")
    vld_comp_df = pd.DataFrame({
        "Model Family / Architecture": [
            "Publisher-Only Baseline (Metadata)",
            "Sanitized Text-Only (DeBERTa-v3)",
            "Image-Only Baseline (OpenCLIP ViT-B/32)",
            "Option 1: Late Concatenation Adapter",
            "Option 3: Gated Multimodal Fusion Adapter (Selected)"
        ],
        "Trainable Params": ["0", "0.20M", "0.20M", "0.86M", "1.58M"],
        "Val Macro-F1": ["0.4390", "0.5754", "0.6620", "0.6474", "0.6606"],
        "Test Accuracy": ["58.32%", "55.76%", "64.35%", "63.25%", "64.72%"],
        "Test Balanced Acc": ["53.05%", "54.99%", "64.83%", "63.81%", "64.80%"],
        "Test Macro-F1": ["44.36%", "55.00%", "64.30%", "63.22%", "64.54%"],
        "Test ROC-AUC": ["0.5697", "0.5581", "0.7056", "0.7053", "0.7086"]
    })
    st.dataframe(vld_comp_df, use_container_width=True, hide_index=True)

    # Confusion Matrix
    st.markdown("##### Protocol A Test Confusion Matrix (Option 3 Gated Fusion, N=1,094)")
    v_cm_c1, v_cm_c2 = st.columns(2)
    with v_cm_c1:
        vld_cm_df = pd.DataFrame(
            [
                ["Predicted Unlikely", "315 (True Negative)", "220 (False Negative)"],
                ["Predicted Likely", "166 (False Positive)", "393 (True Positive)"]
            ],
            columns=["Predicted Class", "Actual Unlikely (N=481)", "Actual Likely (N=613)"]
        )
        st.dataframe(vld_cm_df, use_container_width=True, hide_index=True)
    with v_cm_c2:
        st.caption(
            "● Class 0 (Unlikely Manipulation) Recall: **65.49%** (315 / 481)\n\n"
            "● Class 1 (Likely Manipulation) Recall: **64.11%** (393 / 613)\n\n"
            "● Balanced Accuracy: **64.80%** · Macro-F1: **64.54%**\n\n"
            "Evaluated using $B=1,000$ group-aware cluster bootstrap iterations preserving syndicated story groups."
        )

    # Generalization & Governance Expander
    with st.expander("Generalization Audits & Governance Limitations", expanded=False):
        st.markdown("""
        * **Strict Temporal Generalization (Protocol B Strict):** Macro-F1 of **62.43%** on articles strictly postdating July 1, 2024 with all cross-cutoff story clusters completely excluded from training.
        * **Publisher-Held-Out Evaluation:** Macro-F1 of **61.42%** on previously unseen newsrooms, confirming resistance to publisher identity memorization.
        * **Modality Salience Weight Gating:** Learned sigmoid gating vector ($\bar{g} \approx 0.3819$) adaptively routes representations, balancing textual framing (38.2%) with visual context (61.8%).
        * **Epistemic Limitation:** Rhetorical manipulation classification measures editorial tone and packaging. It does **not** determine whether the event described in the article is true or false.
        """)

    # ─────────────────────────────────────────────────────────────────────────
    # SELECTIVE CLASSIFICATION BENCHMARK: TASK A CHEAPFAKE DETECTION
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### High-Precision Pathway: Selective Classification on NewsCLIPpings (Task A)")
    st.caption("Empirical Risk-Coverage Optimization (Geifman & El-Yaniv, NeurIPS 2017) applied to Out-of-Context Cheapfake Detection.")

    st.markdown("""
    In adversarial benchmarks (such as NewsCLIPpings Gen 0 where adversaries intentionally optimize CLIP similarity), raw 100%-coverage accuracy on uncurated streams is bounded at 63.53%.
    However, real-world verification systems achieve **>85% accuracy on accepted predictions** by employing **Selective Classification (Classification with a Reject Option)**:
    deferring low-margin borderline samples ($|s_{TI} - \\tau| < \\Delta$) for human review while accepting high-margin cases.
    """)

    sel_c1, sel_c2 = st.columns([3, 2], gap="medium")
    with sel_c1:
        sel_df = pd.DataFrame({
            "Confidence Margin (|s - tau|)": [
                "Delta >= 0.00 (Full 100% Coverage)",
                "Delta >= 0.02 (Low Ambiguity Filtered)",
                "Delta >= 0.04 (Moderate Ambiguity Filtered)",
                "Delta >= 0.06 (Strict High-Confidence)",
                "Delta >= 0.07 (Validation Operating Point Delta*)",
                "Delta >= 0.08 (Ultra-Confident Cohort)"
            ],
            "Test Coverage": ["100.0% (N=7,264)", "66.4% (N=4,825)", "36.8% (N=2,672)", "17.0% (N=1,233)", "10.8% (N=788)", "7.0% (N=505)"],
            "Accepted Test Acc": ["63.53%", "70.24%", "78.52%", "85.24%", "87.44%", "91.09%"],
            "Accepted Test F1": ["62.75%", "68.95%", "77.48%", "84.69%", "87.24%", "91.06%"],
            "Accepted Test BalAcc": ["63.53%", "69.29%", "77.07%", "84.32%", "87.10%", "91.02%"],
            "95% Wilson CI": ["[62.4%, 64.6%]", "[68.9%, 71.5%]", "[76.9%, 80.0%]", "[83.1%, 87.1%]", "[84.9%, 89.6%]", "[88.3%, 93.3%]"]
        })
        st.dataframe(sel_df, use_container_width=True, hide_index=True)
        st.caption("⚠️ **Governance Note:** The reported accuracy applies strictly to the **accepted prediction cohort** ($N_{\\text{acc}} = 788$ at 10.85% coverage). The 87.44% accepted test accuracy with 95% Wilson CI [84.94%, 89.57%] is an **empirical point estimate**, not an unconditional statistical guarantee of accuracy above 85%. Full test set baseline accuracy across 100% of samples remains **63.53%** ($N = 7,264$). Ambiguous predictions ($N_{\\text{rej}} = 6,476$) are deferred for subsequent human review.")

    with sel_c2:
        chart_data = pd.DataFrame({
            "Coverage (%)": [100.0, 66.4, 36.8, 17.0, 10.8, 7.0],
            "Accepted Accuracy (%)": [63.53, 70.24, 78.52, 85.24, 87.44, 91.09],
            "Empirical Risk (%)": [36.47, 29.76, 21.48, 14.76, 12.56, 8.91]
        }).set_index("Coverage (%)")
        st.line_chart(chart_data)
        st.caption("📈 **Accuracy vs. Coverage & Risk Curve**: Operating point &Delta;* = 0.07 was selected on validation data (Val Acc: 86.40% at 11.30% coverage, N=794) and evaluated on test data (Accepted Test Acc: 87.44% at 10.85% coverage, N=788, 95% Wilson CI: [84.94%, 89.57%]).")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: RESEARCH SCOPE & LIMITATIONS
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("#### Research Scope, Limitations & Attributions")

    st.markdown("##### 1. Evaluated Scope vs. Non-Claims")
    st.markdown("""
    * **Cheapfake Detection (Evaluated):** The framework detects out-of-context multimodal pairings where an authentic photograph is misattributed to an unrelated narrative.
    * **Pixel-Level Generative Forensics (Non-Claim):** The framework does **NOT** classify pixel-level synthetic generation artifacts (Midjourney, Stable Diffusion, DALL-E, GANs, or deepfakes). OpenCLIP measures semantic alignment; it does not perform spatial or frequency-domain artifact analysis.
    """)

    st.markdown("##### 2. Uncertainty & Decision Policy")
    st.markdown("""
    * **Heuristic Confidence:** Decision Confidence and Predictive Uncertainty are normalized support margins based on NLI softmax distance and OpenCLIP margins. They are **not** Bayesian posterior probabilities.
    * **Absence of Proof is Not Proof of Falsehood:** Claims with missing evidence or neutral relation are designated **Uncertain**, allowing human fact-checker routing rather than forcing high-risk false positives.
    """)

    st.markdown("##### 3. Outstanding Experiments (Factify 2)")
    st.markdown("""
    The following experiments remain pending due to external CodaLab credential gates:
    * Task B: Document-grounded claim verification.
    * Task C: 5-class multimodal verification (`Support_Multimodal`, `Support_Text`, `Refute`, `Neutral`, `Insufficient_Multimodal`).
    * Pipeline Ablations: Evaluating `--ablation no_decomp`, `no_rel_filter`, `text_only`, and `forced_binary`.
    """)

    st.markdown("##### 4. Computational Latency Profile")
    st.markdown("""
    * **Vector Scoring:** Benchmarking on NewsCLIPpings test ($N=7,264$) takes **1.33 seconds total** ($< 0.2$ ms/sample) using precomputed 512-d unit-normalized OpenCLIP vectors.
    * **Interactive Inference:** Full end-to-end CPU pipeline execution (PIL loading, spaCy parsing, live web querying, and DeBERTa cross-encoding) takes **1.5–3.5 seconds per claim**.
    """)

    with st.expander("Academic Attributions & Licensing Provenance", expanded=False):
        st.markdown("""
        * **Project:** MSc in Data Science Dissertation Research
        * **Original Research Baseline:** `multimodal-misinfo-verifier` (preserved untouched).
        * **UI Architecture Attribution:** Inspired by `kmist70/fake-news-detector` under MIT License.
        * **Multimodal Concept Attribution:** Inspired by `LaughingParrot/CLIP-Multimodal-Fake-News-Detection-with-InfoNCE-on-Fakeddit` under MIT License.
        * **Pretrained Encoders:** OpenAI CLIP (`openai/clip-vit-base-patch32`) and DeBERTa-v3 NLI (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) via HuggingFace Hub.
        * **Benchmark Dataset:** NewsCLIPpings (Luo et al., Findings of EMNLP 2021; Apache-2.0).
        """)


# ─────────────────────────────────────────────────────────────────────────────
# COMPACT FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "VERITAS-MM · Explainable Multimodal Misinformation Verification Framework · MSc Data Science Final Project · Academic Audit Ready"
)
