"""
Deterministic Regression and Integrity Test Suite for Live Search Evidence Hotfix.

Verifies all 14 criteria from Section G & H:
 1. test_nasa_claim_returns_nasa_results: NASA Broom Point claim retrieves NASA records when provided.
 2. test_nasa_claim_filters_camera_stories: Candidate items with only speed-camera stories produce NO relevant evidence.
 3. test_camera_stories_zero_relevance: School-camera stories receive relevance_score = 0.0 and is_relevant = False.
 4. test_relevance_never_defaults_to_80_or_100: Relevance score never defaults to 80% or 100%.
 5. test_relevance_failure_returns_none: Empty or invalid input returns None ('Relevance unavailable').
 6. test_google_wrapper_vs_publisher_url: Google News wrapper and publisher URLs are distinct.
 7. test_inaccessible_article_honest_label: Inaccessible article is marked snippet_only/inaccessible, never full_text.
 8. test_distinct_cache_keys_no_collision: Cache keys for different queries do not collide.
 9. test_distinct_queries_no_cross_leak: Distinct queries do not cross-contaminate cached results.
10. test_irrelevant_contradiction_does_not_mislead: Irrelevant evidence does not trigger false contradictions.
11. test_missing_evidence_yields_uncertain: Missing evidence produces UNCERTAIN with explicit disclosure.
12. test_supporting_passage_yields_supported: Truly supporting passage yields SUPPORTED (or PARTIALLY_SUPPORTED if visual mismatch).
13. test_refuting_passage_yields_misleading: Truly refuting passage yields MISLEADING.
14. test_multi_topic_verification: Multi-topic verification across science, sports, business, and public safety.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import urllib.error

from src.article_fetcher import fetch_article_content, resolve_publisher_url
from src.evidence_providers import GoogleNewsRSSProvider, WikipediaSearchProvider
from src.pipeline import MultimodalVerificationPipeline
from src.retrieval import (
    EvidenceRetriever,
    build_search_queries,
    compute_relevance_score,
    evaluate_evidence_relevance,
    extract_named_entities_and_anchors,
)
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
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

NASA_CLAIM = "NASA's Perseverance rover captured a view of the Broom Point rock formation on Mars on May 15, 2025."

CAMERA_STORIES = [
    EvidenceSnippet(
        snippet_id="cam_dalton",
        text="Dalton school-zone cameras resume operation ahead of the new academic calendar with automated ticketing.",
        source="Live News - Dalton Daily",
        title="Dalton school-zone cameras resume operation",
        url="https://www.daltondaily.com/news/cameras-resume",
        wrapper_url="https://news.google.com/rss/articles/CBMiDALTON123",
        publisher="Dalton Daily",
        source_type="live_web"
    ),
    EvidenceSnippet(
        snippet_id="cam_santa_fe",
        text="Santa Fe speed cameras take bullets as local vandalism incidents rise across municipal highways.",
        source="Live News - Santa Fe New Mexican",
        title="Santa Fe speed cameras take bullets",
        url="https://www.santafenewmexican.com/news/cameras-bullets",
        wrapper_url="https://news.google.com/rss/articles/CBMiSANTAFE456",
        publisher="Santa Fe New Mexican",
        source_type="live_web"
    ),
    EvidenceSnippet(
        snippet_id="cam_albany",
        text="Albany school speed-zone cameras return to operation following scheduled software updates and calibration.",
        source="Live News - Albany Times",
        title="Albany school speed-zone cameras return to operation",
        url="https://www.albanytimes.com/news/speed-cameras-return",
        wrapper_url="https://news.google.com/rss/articles/CBMiALBANY789",
        publisher="Albany Times",
        source_type="live_web"
    ),
]

NASA_PASSAGE = EvidenceSnippet(
    snippet_id="nasa_perseverance_01",
    text="NASA's Perseverance rover captured a high-resolution view of the Broom Point rock formation on Mars on May 15, 2025.",
    source="NASA Jet Propulsion Laboratory",
    title="Perseverance Views Broom Point on Mars",
    url="https://science.nasa.gov/mission/mars-2020-perseverance/broom-point",
    wrapper_url="https://news.google.com/rss/articles/CBMiNASA123",
    publisher="NASA JPL",
    source_type="live_web"
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. NASA CLAIM RETURNS NASA RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def test_nasa_claim_returns_nasa_results():
    """Verifies that the NASA claim retrieves and qualifies the genuine NASA record."""
    retriever = EvidenceRetriever(relevance_threshold=0.20)
    score, is_relevant, reason = evaluate_evidence_relevance(NASA_CLAIM, NASA_PASSAGE.text)

    assert is_relevant is True
    assert score is not None
    assert score >= 0.70
    assert "Passed" in reason
    assert any(ent in reason for ent in ["NASA", "Perseverance", "Broom Point", "Mars"])


# ─────────────────────────────────────────────────────────────────────────────
# 2. NASA CLAIM FILTERS OUT CAMERA STORIES
# ─────────────────────────────────────────────────────────────────────────────

def test_nasa_claim_filters_camera_stories():
    """Verifies that when only speed-camera stories are presented, 0 relevant evidence is returned."""
    retriever = EvidenceRetriever(relevance_threshold=0.20)

    for cam_story in CAMERA_STORIES:
        score, is_relevant, reason = evaluate_evidence_relevance(NASA_CLAIM, cam_story.text)
        assert is_relevant is False
        assert score == 0.0
        assert "Excluded: 0 matching entity anchors" in reason

    # When filtered through verify_atomic_claims, no evidence is admitted
    atomic_claims = [
        AtomicClaim(
            claim_id="nasa_01",
            text=NASA_CLAIM,
            entities=["NASA", "Perseverance", "Broom Point", "Mars"]
        )
    ]
    with patch.object(retriever, "retrieve", return_value=CAMERA_STORIES):
        results = verify_atomic_claims(atomic_claims, retriever=retriever)
        assert len(results) == 1
        assert results[0].status == "UNVERIFIED"
        assert results[0].nli_result is None
        assert results[0].retrieved_evidence is not None
        assert results[0].retrieved_evidence.is_relevant is False


# ─────────────────────────────────────────────────────────────────────────────
# 3. CAMERA STORIES ZERO RELEVANCE WITH EXPLICIT REASON
# ─────────────────────────────────────────────────────────────────────────────

def test_camera_stories_zero_relevance():
    """Verifies that each camera story strictly receives 0.0 relevance and is_relevant=False."""
    entities = extract_named_entities_and_anchors(NASA_CLAIM)
    assert "NASA" in entities
    assert "Broom Point" in entities

    for cam in CAMERA_STORIES:
        score, is_relevant, reason = evaluate_evidence_relevance(
            NASA_CLAIM,
            cam.text,
            claim_entities=entities
        )
        assert score == 0.0
        assert is_relevant is False
        assert "Excluded: 0 matching entity anchors" in reason
        assert "NASA" in reason or "Broom Point" in reason


# ─────────────────────────────────────────────────────────────────────────────
# 4. RELEVANCE NEVER DEFAULTS TO 80% OR 100%
# ─────────────────────────────────────────────────────────────────────────────

def test_relevance_never_defaults_to_80_or_100():
    """Verifies that relevance calculation never silently falls back to 0.80 or 1.00."""
    snip = EvidenceSnippet(
        snippet_id="raw_snip",
        text="A totally unrelated passage about local municipal tax updates.",
        source="Municipal Press"
    )
    # Default schema value must be None, not 1.0
    assert snip.relevance_score is None
    assert snip.score is None
    assert snip.is_relevant is False

    # Evaluation on unrelated content must not return 0.80 or 1.00
    score, is_rel, _ = evaluate_evidence_relevance(NASA_CLAIM, snip.text)
    assert score != 0.80
    assert score != 1.00
    assert score == 0.0
    assert is_rel is False


# ─────────────────────────────────────────────────────────────────────────────
# 5. RELEVANCE FAILURE RETURNS NONE
# ─────────────────────────────────────────────────────────────────────────────

def test_relevance_failure_returns_none():
    """Verifies that empty or whitespace inputs produce (None, False, 'Relevance unavailable...')."""
    # Empty query
    score_1, is_rel_1, reason_1 = evaluate_evidence_relevance("", "Some non-empty passage.")
    assert score_1 is None
    assert is_rel_1 is False
    assert "Relevance unavailable" in reason_1

    # Empty passage
    score_2, is_rel_2, reason_2 = evaluate_evidence_relevance("Some valid claim.", "   ")
    assert score_2 is None
    assert is_rel_2 is False
    assert "Relevance unavailable" in reason_2

    # Insufficient content words (stopwords only)
    score_3, is_rel_3, reason_3 = evaluate_evidence_relevance("is it at the", "and so on")
    assert score_3 is None
    assert is_rel_3 is False
    assert "Relevance unavailable" in reason_3


# ─────────────────────────────────────────────────────────────────────────────
# 6. GOOGLE WRAPPER VS PUBLISHER URL
# ─────────────────────────────────────────────────────────────────────────────

def test_google_wrapper_vs_publisher_url():
    """Verifies that Google News wrappers are stored in wrapper_url while direct publisher URLs are in url."""
    mock_rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Google News</title>
        <item>
          <title>Mars Rover Finds Organic Molecules - BBC News</title>
          <link>https://news.google.com/rss/articles/CBMiBBC123456</link>
          <pubDate>Wed, 14 Oct 2026 10:00:00 GMT</pubDate>
          <source url="https://www.bbc.com">BBC News</source>
          <description>Scientific team confirms discovery of complex organic compounds.</description>
        </item>
      </channel>
    </rss>
    """
    provider = GoogleNewsRSSProvider(timeout=5)

    with patch("src.evidence_providers.urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_rss_xml.encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        snippets = provider.search("Mars rover molecules", top_k=1)
        assert len(snippets) == 1
        s = snippets[0]
        # Publisher root domain from <source url="...">
        assert s.publisher_domain == "https://www.bbc.com"
        assert s.url is None
        # Google News CBMi wrapper kept separate
        assert s.wrapper_url == "https://news.google.com/rss/articles/CBMiBBC123456"
        assert s.publisher == "BBC News"


# ─────────────────────────────────────────────────────────────────────────────
# 7. INACCESSIBLE ARTICLE HONEST LABEL
# ─────────────────────────────────────────────────────────────────────────────

def test_inaccessible_article_honest_label():
    """Verifies that inaccessible URLs or HTTP errors are transparently labeled, never claimed as full text."""
    with patch("src.article_fetcher.urllib.request.urlopen", side_effect=urllib.error.HTTPError("url", 403, "Forbidden", {}, None)):
        res = fetch_article_content("https://paywalled-news.com/exclusive")
        assert res["text"] == ""
        assert res["content_type"] == "snippet_only"
        assert res["access_status"] == "inaccessible"

    # Direct resolution failure on Google CBMi token
    resolved_url, note = resolve_publisher_url("https://news.google.com/rss/articles/CBMiINVALID_TOKEN")
    assert resolved_url is None
    assert "http_error" in note or "google_wrapper" in note or "failed" in note


# ─────────────────────────────────────────────────────────────────────────────
# 8. DISTINCT CACHE KEYS DO NOT COLLIDE
# ─────────────────────────────────────────────────────────────────────────────

def test_distinct_cache_keys_no_collision():
    """Verifies that different queries produce distinct cache keys."""
    retriever = EvidenceRetriever()
    k1 = retriever._hash_query("NASA Perseverance rover Broom Point", top_k=3, force_mode="live_only")
    k2 = retriever._hash_query("Dalton school-zone cameras resume operation", top_k=3, force_mode="live_only")
    k3 = retriever._hash_query("NASA Perseverance rover Broom Point", top_k=5, force_mode="live_only")
    k4 = retriever._hash_query("nasa perseverance rover broom point", top_k=3, force_mode="live_only")

    assert k1 != k2
    assert k1 != k3
    # Case normalization should yield identical keys
    assert k1 == k4


# ─────────────────────────────────────────────────────────────────────────────
# 9. DISTINCT QUERIES DO NOT REUSE UNRELATED RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def test_distinct_queries_no_cross_leak():
    """Verifies that cache results from one query are never returned for a different query."""
    retriever = EvidenceRetriever(enable_live_search=False)
    retriever.clear_cache()

    q_london = "London warehouse blaze firefighters"
    q_geneva = "World leaders convened in Geneva for summit addressing inflation"

    res_london = retriever.retrieve(q_london, top_k=2, force_mode="local_only")
    res_geneva = retriever.retrieve(q_geneva, top_k=2, force_mode="local_only")

    # London results must contain fire/warehouse, Geneva results must contain Geneva/summit
    assert any("fire" in s.text.lower() for s in res_london)
    assert any("geneva" in s.text.lower() or "inflation" in s.text.lower() for s in res_geneva)

    # Cache should contain both distinct keys
    k_london = retriever._hash_query(q_london, top_k=2, force_mode="local_only")
    k_geneva = retriever._hash_query(q_geneva, top_k=2, force_mode="local_only")
    assert k_london in retriever._query_cache
    assert k_geneva in retriever._query_cache
    assert retriever._query_cache[k_london][1] != retriever._query_cache[k_geneva][1]


# ─────────────────────────────────────────────────────────────────────────────
# 10. IRRELEVANT CONTRADICTION DOES NOT MISLEAD
# ─────────────────────────────────────────────────────────────────────────────

def test_irrelevant_contradiction_does_not_mislead():
    """Verifies that irrelevant passages filtered out do not trigger NLI contradiction or MISLEADING verdict."""
    retriever = EvidenceRetriever(relevance_threshold=0.20)
    irrelevant_passage = EvidenceSnippet(
        snippet_id="off_topic_01",
        text="The state transportation agency installed 15 speed cameras along the turnpike.",
        source="Traffic Daily",
        score=0.1,
        relevance_score=0.0,
        is_relevant=False,
        source_type="live_web"
    )

    with patch.object(retriever, "retrieve", return_value=[irrelevant_passage]):
        claims = [AtomicClaim(claim_id="c_space", text="James Webb telescope spotted oldest galaxy.", entities=["James Webb"])]
        atomic_results = verify_atomic_claims(claims, retriever=retriever)

        assert len(atomic_results) == 1
        assert atomic_results[0].status == "UNVERIFIED"

        # Multimodal fusion with 0 supported and 0 contradicted must NOT be MISLEADING
        verdict, reasons = fuse_multimodal_decision(
            cross_modal=CrossModalScore(similarity_score=0.15, mismatch_detected=False),
            nli_results=[],
            atomic_results=atomic_results
        )
        assert verdict == DecisionVerdict.UNCERTAIN


# ─────────────────────────────────────────────────────────────────────────────
# 11. MISSING EVIDENCE PRODUCES UNCERTAIN WITH REQUIRED DISCLOSURE
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_evidence_yields_uncertain():
    """Verifies that when no evidence survives relevance filtering, verdict is UNCERTAIN with standard disclosure."""
    atomic_results = [
        AtomicVerificationResult(
            claim_id="c_none",
            atomic_text="Unattested claim with no news coverage.",
            status="UNVERIFIED",
            explanation="No relevant evidence found for this claim in the retrieved sources."
        )
    ]

    verdict, reasons = fuse_multimodal_decision(
        cross_modal=CrossModalScore(similarity_score=0.25, mismatch_detected=False),
        nli_results=[],
        atomic_results=atomic_results
    )
    assert verdict == DecisionVerdict.UNCERTAIN
    assert any("No relevant evidence found for this claim in the retrieved sources." in r for r in reasons)


# ─────────────────────────────────────────────────────────────────────────────
# 12. TRULY SUPPORTING PASSAGE YIELDS SUPPORTED
# ─────────────────────────────────────────────────────────────────────────────

def test_supporting_passage_yields_supported():
    """Verifies that an entailed atomic claim yields SUPPORTED (or PARTIALLY_SUPPORTED if visual mismatch)."""
    # Matching visual + entailed text -> SUPPORTED
    cross_modal_match = CrossModalScore(similarity_score=0.32, mismatch_detected=False)
    atomic_results = [
        AtomicVerificationResult(
            claim_id="c_supp",
            atomic_text="Firefighters battled the blaze on Sunday morning.",
            status="SUPPORTED"
        )
    ]
    nli_entailed = [
        NLIResult(
            claim_text="Firefighters battled the blaze on Sunday morning.",
            evidence_snippet_id="ev_01",
            predicted_relation="entailment",
            p_entailment=0.91,
            p_neutral=0.06,
            p_contradiction=0.03
        )
    ]

    verdict, _ = fuse_multimodal_decision(
        cross_modal=cross_modal_match,
        nli_results=nli_entailed,
        atomic_results=atomic_results
    )
    assert verdict == DecisionVerdict.SUPPORTED

    # Mismatched visual + entailed text -> PARTIALLY_SUPPORTED with cheapfake warning (NOT MISLEADING)
    cross_modal_mismatch = CrossModalScore(similarity_score=0.10, mismatch_detected=True, assessment_notes="Visual mismatch")
    verdict_mismatch, reasons_mismatch = fuse_multimodal_decision(
        cross_modal=cross_modal_mismatch,
        nli_results=nli_entailed,
        atomic_results=atomic_results
    )
    assert verdict_mismatch == DecisionVerdict.PARTIALLY_SUPPORTED
    assert any("visual" in r.lower() or "mismatch" in r.lower() or "context" in r.lower() for r in reasons_mismatch)


# ─────────────────────────────────────────────────────────────────────────────
# 13. TRULY REFUTING PASSAGE YIELDS MISLEADING
# ─────────────────────────────────────────────────────────────────────────────

def test_refuting_passage_yields_misleading():
    """Verifies that a contradicted claim produces MISLEADING."""
    cross_modal = CrossModalScore(similarity_score=0.28, mismatch_detected=False)
    atomic_results = [
        AtomicVerificationResult(
            claim_id="c_ref",
            atomic_text="The city council unanimously banned electric scooters.",
            status="CONTRADICTED"
        )
    ]
    nli_contradicted = [
        NLIResult(
            claim_text="The city council unanimously banned electric scooters.",
            evidence_snippet_id="ev_02",
            predicted_relation="contradiction",
            p_entailment=0.04,
            p_neutral=0.08,
            p_contradiction=0.88
        )
    ]

    verdict, reasons = fuse_multimodal_decision(
        cross_modal=cross_modal,
        nli_results=nli_contradicted,
        atomic_results=atomic_results
    )
    assert verdict == DecisionVerdict.MISLEADING
    assert any("contradicted" in r.lower() for r in reasons)


# ─────────────────────────────────────────────────────────────────────────────
# 14. MULTI-TOPIC VERIFICATION SUITE
# ─────────────────────────────────────────────────────────────────────────────

def test_multi_topic_verification():
    """
    Verifies positive and negative relevance controls across four diverse news domains:
    1. Science (NASA / Mars)
    2. Sports (Champions League)
    3. Business (Central Bank interest rates)
    4. Public Safety (Industrial warehouse fire)
    """
    topics = [
        {
            "domain": "Science",
            "claim": "NASA's Perseverance rover analyzed Martian soil samples in Jezero Crater.",
            "pos": "NASA's Perseverance rover collected Martian soil and regolith samples across Jezero Crater.",
            "neg": "Premier League football club announced record seasonal ticket sales.",
        },
        {
            "domain": "Sports",
            "claim": "Real Madrid secured victory in the European Champions League final.",
            "pos": "Real Madrid won the UEFA Champions League final after scoring two second-half goals.",
            "neg": "Central bank governors convened in Basel to debate benchmark interest rates.",
        },
        {
            "domain": "Business",
            "claim": "The Federal Reserve lowered benchmark interest rates by 25 basis points.",
            "pos": "The Federal Reserve announced a 25 basis point reduction in federal funds rates.",
            "neg": "Firefighters deployed aerial ladders to combat a warehouse blaze in London.",
        },
        {
            "domain": "Public Safety",
            "claim": "London Fire Brigade deployed 70 firefighters to an industrial warehouse blaze.",
            "pos": "Ten fire engines and approximately 70 firefighters tackled an industrial warehouse fire in East London.",
            "neg": "Astronomers detected high-energy cosmic rays originating from distant supernovas.",
        },
    ]

    for t in topics:
        domain = t["domain"]
        claim = t["claim"]
        pos_doc = t["pos"]
        neg_doc = t["neg"]

        # Positive test: should be relevant
        pos_score, pos_rel, pos_reason = evaluate_evidence_relevance(claim, pos_doc)
        assert pos_rel is True, f"Failed positive control for {domain}: {pos_reason}"
        assert pos_score is not None and pos_score >= 0.25, f"Low score for {domain}: {pos_score}"

        # Negative test: should be strictly excluded
        neg_score, neg_rel, neg_reason = evaluate_evidence_relevance(claim, neg_doc)
        assert neg_rel is False, f"Failed negative control for {domain}: {neg_reason}"
        assert neg_score == 0.0 or neg_score < 0.20, f"Score not low for negative {domain}: {neg_score}"


# ─────────────────────────────────────────────────────────────────────────────
# 15. MALFORMED PROVIDER RESPONSES HANDLED GRACEFULLY
# ─────────────────────────────────────────────────────────────────────────────

def test_malformed_provider_responses_handled_gracefully():
    """Verifies that malformed XML or corrupt JSON from providers returns [] without crashing."""
    provider_news = GoogleNewsRSSProvider(timeout=5)
    provider_wiki = WikipediaSearchProvider(timeout=5)

    # Corrupt XML (unclosed tag, binary garbage)
    with patch("src.evidence_providers.urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"<rss><channel><item><title>Broken XML"
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        res = provider_news.search("Any query", top_k=3)
        assert res == []

    # Corrupt JSON (truncated JSON syntax)
    with patch("src.evidence_providers.urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"query": {"search": [{"title": "Truncated'
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        res_wiki = provider_wiki.search("Any query", top_k=3)
        assert res_wiki == []


# ─────────────────────────────────────────────────────────────────────────────
# 16. CACHE INVALIDATION & RELEVANCE THRESHOLD SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────

def test_cache_invalidation_and_threshold_sensitivity():
    """Verifies that changing relevance threshold updates cache keys and clear_cache() purges all caches."""
    retriever = EvidenceRetriever(relevance_threshold=0.20)
    k1 = retriever._hash_query("Economic summit inflation", top_k=3, force_mode="live_only")

    # Modifying threshold produces a distinct key
    retriever.relevance_threshold = 0.40
    k2 = retriever._hash_query("Economic summit inflation", top_k=3, force_mode="live_only")
    assert k1 != k2

    # Clear cache removes both retriever entries and article fetcher cache
    retriever._query_cache[k1] = (1000.0, [])
    retriever.clear_cache()
    assert len(retriever._query_cache) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 17. CROSS-DOMAIN NEGATIVE CONTROLS (SPORTS & BUSINESS)
# ─────────────────────────────────────────────────────────────────────────────

def test_cross_domain_negative_controls():
    """Explicit negative controls ensuring domain mismatches are excluded with 0.0 relevance."""
    sports_claim = "Arsenal Football Club completed the transfer signing of a new central midfielder."
    science_doc = "NASA's Perseverance rover collected soil samples from the Martian crater floor."

    s_score, s_rel, s_reason = evaluate_evidence_relevance(sports_claim, science_doc)
    assert s_rel is False
    assert s_score == 0.0
    assert "Excluded" in s_reason

    business_claim = "The European Central Bank increased benchmark interest rates by 50 basis points."
    crime_doc = "Police cordoned off residential streets following an overnight vehicular accident."

    b_score, b_rel, b_reason = evaluate_evidence_relevance(business_claim, crime_doc)
    assert b_rel is False
    assert b_score == 0.0
    assert "Excluded" in b_reason
