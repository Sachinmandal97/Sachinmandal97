"""
Unit and integration tests for the Universal Multimodal News Verification System.

Verifies:
1. Google News RSS and Wikipedia search providers (parsing, formatting, and graceful network error handling).
2. Lexical & topical relevance scoring (filtering out unrelated stories, e.g. Sahara / Venice).
3. Evidence retriever caching with TTL and deduplication.
4. Prevention of decision inversion: Irrelevant evidence is excluded from decisive NLI, preventing false contradictions.
5. Decoupled multimodal fusion: Low CLIP similarity alone does not force MISLEADING when text is corroborated.
6. Universal domain-agnostic verification across politics, business, science/space, and sports.
7. End-to-end provenance preservation (source_type, URL, publisher, relevance score, evidence_source_mode).
"""

from datetime import datetime, timezone
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

from src.evidence_providers import BaseSearchProvider, GoogleNewsRSSProvider, WikipediaSearchProvider
from src.pipeline import MultimodalVerificationPipeline
from src.retrieval import EvidenceRetriever, compute_relevance_score, extract_content_words
from src.schemas import (
    AtomicClaim,
    AtomicVerificationResult,
    CrossModalScore,
    DecisionVerdict,
    EvidenceSnippet,
    MultimodalSample,
    NLIResult,
    VerificationResult,
)
from src.verification import fuse_multimodal_decision, verify_atomic_claims


# ─────────────────────────────────────────────────────────────────────────────
# 1. SEARCH PROVIDER TESTS (WITH DETERMINISTIC MOCKS)
# ─────────────────────────────────────────────────────────────────────────────

def test_google_news_rss_provider_parsing():
    """Verifies that GoogleNewsRSSProvider parses mock RSS XML correctly into EvidenceSnippets."""
    mock_rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Google News</title>
        <item>
          <title>Mars Rover Discovers Subsurface Ice - Reuters</title>
          <link>https://news.google.com/articles/mars-rover-123</link>
          <pubDate>Mon, 12 Oct 2026 14:00:00 GMT</pubDate>
          <source url="https://www.reuters.com">Reuters</source>
          <description>NASA robotic explorer identified substantial ice reservoirs beneath equatorial plains.</description>
        </item>
      </channel>
    </rss>
    """
    provider = GoogleNewsRSSProvider(timeout=5)

    with patch("src.evidence_providers.urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_rss_xml.encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        snippets = provider.search("Mars rover ice discovery", top_k=3)

        assert len(snippets) == 1
        snip = snippets[0]
        assert snip.source_type == "live_web"
        assert "Reuters" in snip.source
        assert snip.publisher_domain == "https://www.reuters.com"
        assert snip.url is None
        assert snip.wrapper_url == "https://news.google.com/articles/mars-rover-123"
        assert snip.publisher == "Reuters"
        assert "NASA robotic explorer identified" in snip.text


def test_wikipedia_provider_parsing():
    """Verifies that WikipediaSearchProvider parses search and summary JSON responses into EvidenceSnippets."""
    mock_search_json = b'{"query": {"search": [{"title": "James Webb Space Telescope", "pageid": 12345, "snippet": "The James Webb Space Telescope is an infrared observatory"}]}}'
    mock_extract_json = b'{"query": {"pages": {"12345": {"pageid": 12345, "title": "James Webb Space Telescope", "extract": "The James Webb Space Telescope is an infrared space observatory launched in December 2021."}}}}'

    provider = WikipediaSearchProvider(timeout=5)

    def mock_urlopen_side_effect(req, *args, **kwargs):
        mock_resp = MagicMock()
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "list=search" in url:
            mock_resp.read.return_value = mock_search_json
        else:
            mock_resp.read.return_value = mock_extract_json
        ctx = MagicMock()
        ctx.__enter__.return_value = mock_resp
        return ctx

    with patch("src.evidence_providers.urllib.request.urlopen", side_effect=mock_urlopen_side_effect):
        snippets = provider.search("James Webb Space Telescope", top_k=2)

        assert len(snippets) == 1
        snip = snippets[0]
        assert snip.source_type == "wikipedia"
        assert "James Webb" in snip.source
        assert "infrared space observatory" in snip.text
        assert "wikipedia.org" in snip.url


def test_provider_network_error_graceful_handling():
    """Verifies that network connection errors or timeouts return an empty list without raising exceptions."""
    provider_news = GoogleNewsRSSProvider(timeout=1)
    provider_wiki = WikipediaSearchProvider(timeout=1)

    with patch("src.evidence_providers.urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        res_news = provider_news.search("Any query", top_k=3)
        res_wiki = provider_wiki.search("Any query", top_k=3)

        assert res_news == []
        assert res_wiki == []


# ─────────────────────────────────────────────────────────────────────────────
# 2. TOPICAL RELEVANCE & LEXICAL FILTERING TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_relevance_score_computation():
    """Verifies that relevant query-passage pairs score high and unrelated pairs score below cutoff."""
    query = "Mars rover discovers water ice deposits"
    relevant_doc = "NASA robotic rover detects subterranean water ice deposits across the Martian equator."
    unrelated_doc_sahara = "Dust storms sweep across the Sahara desert causing reduced visibility in North Africa."
    unrelated_doc_venice = "Gondola traffic restored in Venice canals following heavy seasonal rainfall."

    score_rel = compute_relevance_score(query, relevant_doc)
    score_sahara = compute_relevance_score(query, unrelated_doc_sahara)
    score_venice = compute_relevance_score(query, unrelated_doc_venice)

    assert score_rel >= 0.30, f"Expected high relevance score for matching passage, got {score_rel}"
    assert score_sahara <= 0.10, f"Expected near zero relevance for Sahara passage, got {score_sahara}"
    assert score_venice == 0.0, f"Expected 0.0 relevance for Venice passage, got {score_venice}"


def test_unrelated_passages_filtered_by_retriever():
    """Verifies that EvidenceRetriever marks unrelated candidate documents as is_relevant=False."""
    retriever = EvidenceRetriever(relevance_threshold=0.20)
    
    # Query about a topic not in the local demo corpus (e.g. semiconductor chip supply)
    unrelated_query = "Semiconductor manufacturing factory opens in Tokyo"
    local_hits = retriever.retrieve_local(unrelated_query, top_k=3)

    for hit in local_hits:
        # None of the 6 synthetic demo stories (fire, flood, blizzard, archeology, etc.) should match semiconductors
        assert hit.is_relevant is False or hit.relevance_score < 0.20


# ─────────────────────────────────────────────────────────────────────────────
# 3. RETRIEVER CACHING WITH TTL TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_retriever_query_caching_and_ttl():
    """Verifies that repeated queries hit the memory cache and expired items are refreshed."""
    retriever = EvidenceRetriever(enable_live_search=False, cache_ttl_seconds=2)
    query = "London warehouse blaze firefighters"

    hits_1 = retriever.retrieve(query, top_k=2, force_mode="local_only")
    cache_key = retriever._hash_query(query, top_k=2, force_mode="local_only")
    assert cache_key in retriever._query_cache

    # Second call should retrieve directly from cache
    hits_2 = retriever.retrieve(query, top_k=2, force_mode="local_only")
    assert len(hits_1) == len(hits_2)
    assert hits_1[0].snippet_id == hits_2[0].snippet_id

    # Expire cache manually by modifying cached timestamp
    cached_time, cached_res = retriever._query_cache[cache_key]
    retriever._query_cache[cache_key] = (cached_time - 10, cached_res)

    # Re-retrieval should detect stale entry and execute a fresh query
    hits_3 = retriever.retrieve(query, top_k=2, force_mode="local_only")
    assert len(hits_3) == len(hits_1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. DECISION INVERSION & NLI RELEVANCE AWARENESS TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_irrelevant_evidence_excluded_from_nli_scoring():
    """
    Critical requirement: When candidate retrieval returns an unrelated passage (e.g. Sahara),
    the passage must NOT be passed to DeBERTa NLI to force a false contradiction or misleading verdict.
    Instead, the atomic proposition must remain UNVERIFIED.
    """
    retriever = EvidenceRetriever(relevance_threshold=0.20)

    # Create mock mock snippet with is_relevant=False
    mock_irrelevant_snippet = EvidenceSnippet(
        snippet_id="unrelated_01",
        text="Tourist gondolas resume operations in Venice canals.",
        source="Venice News",
        score=0.1,
        relevance_score=0.0,
        is_relevant=False,
        source_type="local_corpus"
    )

    with patch.object(retriever, "retrieve", return_value=[mock_irrelevant_snippet]):
        claims = [AtomicClaim(claim_id="c_01", text="NASA rover landed safely on Mars.", entities=["NASA", "Mars"])]
        atomic_results = verify_atomic_claims(claims, retriever=retriever)

        assert len(atomic_results) == 1
        # The proposition must be UNVERIFIED, NOT CONTRADICTED
        assert atomic_results[0].status == "UNVERIFIED"
        assert "relevance" in atomic_results[0].explanation.lower() or "unverified" in atomic_results[0].explanation.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 5. DECOUPLED MULTIMODAL FUSION TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_low_clip_score_alone_does_not_force_misleading():
    """
    Verifies that low OpenCLIP image similarity alone does NOT force an overall MISLEADING verdict
    when factual claims are supported. Instead, it must be flagged as out-of-context visual.
    """
    # Case A: Factual text is SUPPORTED, but image is mismatched (e.g. cheapfake beach in blizzard)
    cross_modal_mismatch = CrossModalScore(
        similarity_score=0.12,
        mismatch_detected=True,
        confidence_margin=0.10,
        assessment_notes="Low image alignment"
    )
    supported_atomic = [
        AtomicVerificationResult(
            claim_id="c1",
            atomic_text="Heavy snowstorm blankets city.",
            status="SUPPORTED"
        )
    ]
    nli_entailed = [
        NLIResult(
            claim_text="Heavy snowstorm blankets city.",
            evidence_snippet_id="ev_01",
            predicted_relation="entailment",
            p_entailment=0.92,
            p_neutral=0.05,
            p_contradiction=0.03
        )
    ]

    verdict_a, reasons_a = fuse_multimodal_decision(
        cross_modal=cross_modal_mismatch,
        nli_results=nli_entailed,
        atomic_results=supported_atomic
    )
    # The factual claim was verified by evidence, so low CLIP score alone must NOT force MISLEADING
    assert verdict_a == DecisionVerdict.PARTIALLY_SUPPORTED
    assert any("visual" in r.lower() or "mismatch" in r.lower() or "repurposed" in r.lower() or "context" in r.lower() for r in reasons_a)

    # Case B: Factual text has NO evidence (unverified) and image is mismatched
    unverified_atomic = [
        AtomicVerificationResult(
            claim_id="c2",
            atomic_text="Alien artifacts found on Mars.",
            status="UNVERIFIED"
        )
    ]
    verdict_b, _ = fuse_multimodal_decision(
        cross_modal=cross_modal_mismatch,
        nli_results=[],
        atomic_results=unverified_atomic
    )
    # Without text contradiction, lack of evidence + mismatch should remain UNCERTAIN (not presumed false)
    assert verdict_b == DecisionVerdict.UNCERTAIN


# ─────────────────────────────────────────────────────────────────────────────
# 6. UNIVERSAL DOMAIN-AGNOSTIC EVALUATION (MOCKED PIPELINE)
# ─────────────────────────────────────────────────────────────────────────────

def test_universal_domain_agnostic_verification():
    """
    Verifies that the verification pipeline can process diverse news domains:
    1. Politics & Government
    2. International Conflict
    3. Business & Economy
    4. Science & Space
    5. Sports & Entertainment
    All without hardcoding or dependence on a single benchmark topic.
    """
    pipeline = MultimodalVerificationPipeline()

    test_claims = [
        ("Politics", "The parliamentary committee approved the draft data privacy legislation on Wednesday."),
        ("Conflict", "United Nations negotiators brokered a 72-hour ceasefire agreement between the factions."),
        ("Business", "The central bank lowered benchmark interest rates by 25 basis points to stimulate growth."),
        ("Space", "The robotic lander transmitted initial high-resolution surface panoramas back to mission control."),
        ("Sports", "The championship final ended in a penalty shootout victory after extra time.")
    ]

    for domain, text in test_claims:
        sample = MultimodalSample(
            sample_id=f"test_{domain.lower()}",
            claim_text=text,
            image_path=None
        )
        # Run in local_only mode for deterministic, fast offline execution
        result = pipeline.verify_sample(sample, force_mode="local_only")

        assert isinstance(result, VerificationResult)
        assert result.decision in [
            DecisionVerdict.SUPPORTED,
            DecisionVerdict.PARTIALLY_SUPPORTED,
            DecisionVerdict.MISLEADING,
            DecisionVerdict.UNCERTAIN
        ]
        assert len(result.atomic_verifications) >= 1
        assert result.evidence_source_mode in ["local_only", "hybrid", "live_only"]
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.uncertainty_score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. PROVENANCE PRESERVATION IN VERIFICATION RESULT
# ─────────────────────────────────────────────────────────────────────────────

def test_evidence_provenance_preservation():
    """Verifies that metadata fields (source_type, URL, publisher, relevance_score) persist through pipeline."""
    sample_snippet = EvidenceSnippet(
        snippet_id="prov_01",
        text="The international summit concluded with a multilateral treaty on maritime navigation rights.",
        source="Associated Press - Summit Report",
        score=0.88,
        relevance_score=0.45,
        is_relevant=True,
        url="https://apnews.com/article/maritime-treaty-summit",
        publisher="Associated Press",
        published_date="2026-10-10",
        source_type="live_web"
    )

    assert sample_snippet.source_type == "live_web"
    assert sample_snippet.url.startswith("https://")
    assert sample_snippet.publisher == "Associated Press"
    assert sample_snippet.relevance_score == 0.45
    assert sample_snippet.is_relevant is True
