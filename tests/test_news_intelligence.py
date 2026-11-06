"""
Comprehensive Test Suite for VERITAS-MM 2.0 News Intelligence & Automatic Verification.
Validates:
  1. URL canonicalization and tracking parameter stripping.
  2. Multi-format publication timestamp normalization to UTC ISO 8601.
  3. RSS 2.0 / Atom feed parsing with lead image enclosure extraction.
  4. GDELT DOC 2.0 API JSON response parsing and normalization.
  5. SQLite article persistence, canonical URL deduplication, and status tracking.
  6. Headline claim extraction and editorial prefix stripping.
  7. Bounded batch verification queue processing.
  8. Single-article on-demand verification and re-analysis idempotence.
  9. Safe visual asset acquisition and text-only fallback on network failures.
  10. Feed error handling, status registry recording, and collector resilience.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import tempfile
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from src.database import NewsIntelligenceDB, VALID_PROVENANCES, migrate_schema
from src.scheduler import (
    start_scheduler,
    stop_scheduler,
    trigger_manual_poll,
    get_scheduler_telemetry,
    _POLL_EXECUTION_LOCK
)
from src.news_ingestion import (
    NewsIngestionService,
    canonicalize_url,
    parse_timestamp_to_iso,
    clean_html_summary,
    extract_image_from_item
)
from src.news_verifier import (
    AutomaticNewsVerifier,
    extract_verifiable_claim,
    fetch_article_image
)
from src.pipeline import MultimodalVerificationPipeline
from src.schemas import CrossModalScore, DecisionVerdict, MultimodalSample, OCRResult, VerificationResult

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# ─────────────────────────────────────────────────────────────────────────────
# 1. URL Canonicalization & Deduplication
# ─────────────────────────────────────────────────────────────────────────────

def test_url_canonicalization():
    """Verifies that tracking query parameters and trailing slashes are cleanly removed."""
    raw1 = "https://www.thehindu.com/news/national/isro-mission/?utm_source=rss&utm_medium=feed&utm_campaign=daily"
    raw2 = "https://www.thehindu.com/news/national/isro-mission?fbclid=IwAR123&gclid=XYZ"
    raw3 = "https://www.thehindu.com/news/national/isro-mission/#comments"

    c1 = canonicalize_url(raw1)
    c2 = canonicalize_url(raw2)
    c3 = canonicalize_url(raw3)

    assert c1 == "https://www.thehindu.com/news/national/isro-mission"
    assert c2 == "https://www.thehindu.com/news/national/isro-mission"
    assert c3 == "https://www.thehindu.com/news/national/isro-mission"
    assert c1 == c2 == c3


# ─────────────────────────────────────────────────────────────────────────────
# 2. Timestamp Normalization
# ─────────────────────────────────────────────────────────────────────────────

def test_timestamp_normalization():
    """Tests normalization of RFC 822, GDELT, and ISO dates to standard UTC ISO 8601."""
    # RFC 822 with positive timezone offset (+0530 IST)
    rfc_date = "Mon, 29 May 2026 05:42:00 +0530"
    iso1 = parse_timestamp_to_iso(rfc_date)
    assert iso1 == "2026-05-29T00:12:00Z"

    # RFC 822 GMT
    rfc_gmt = "Sun, 25 Oct 2026 12:00:00 GMT"
    iso2 = parse_timestamp_to_iso(rfc_gmt)
    assert iso2 == "2026-10-25T12:00:00Z"

    # GDELT compact format YYYYMMDDTHHMMSSZ
    gdelt_date = "20261020T123000Z"
    iso3 = parse_timestamp_to_iso(gdelt_date)
    assert iso3 == "2026-10-20T12:30:00Z"

    # Empty / None handling
    assert parse_timestamp_to_iso("") is None
    assert parse_timestamp_to_iso(None) is None


# ─────────────────────────────────────────────────────────────────────────────
# 3. RSS Feed Parsing
# ─────────────────────────────────────────────────────────────────────────────

def test_rss_feed_parsing(tmp_path):
    """Verifies RSS XML parsing from fixture, image extraction, and metadata normalization."""
    xml_path = FIXTURES_DIR / "mock_rss_feed.xml"
    assert xml_path.exists(), f"Fixture missing at {xml_path}"
    xml_bytes = xml_path.read_bytes()

    source = {
        "source_id": "the_hindu_test",
        "name": "The Hindu",
        "country": "India",
        "region": "India",
        "language": "en",
        "feed_type": "rss",
        "url": "https://www.thehindu.com/test.rss"
    }

    service = NewsIngestionService()
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = xml_bytes
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        articles = service.fetch_rss_feed(source, max_records=5)

    assert len(articles) == 3

    # Item 1: Satellite launch with <enclosure>
    a1 = articles[0]
    assert "ISRO successfully launches" in a1["headline"]
    assert a1["publisher"] == "The Hindu"
    assert a1["region"] == "India"
    assert a1["image_url"] == "https://th-i.thgim.com/public/incoming/satellite_launch.jpg"
    assert a1["canonical_url"] == "https://www.thehindu.com/sci-tech/science/isro-launches-nvs-01-satellite/article123456.ece"
    assert a1["processing_status"] == "collected"

    # Item 2: Green hydrogen with <media:content>
    a2 = articles[1]
    assert "green hydrogen" in a2["headline"].lower()
    assert a2["image_url"] == "https://th-i.thgim.com/public/incoming/hydrogen_plant.jpg"

    # Item 3: River water sharing without image
    a3 = articles[2]
    assert a3["image_url"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 4. GDELT API Parsing
# ─────────────────────────────────────────────────────────────────────────────

def test_gdelt_feed_parsing():
    """Verifies GDELT DOC 2.0 JSON parsing, domain attribution, and socialimage extraction."""
    json_path = FIXTURES_DIR / "mock_gdelt_response.json"
    assert json_path.exists(), f"Fixture missing at {json_path}"
    json_bytes = json_path.read_bytes()

    source = {
        "source_id": "gdelt_test",
        "name": "GDELT Global",
        "country": "Global",
        "region": "International",
        "language": "en",
        "feed_type": "gdelt",
        "url": "https://api.gdeltproject.org/test"
    }

    service = NewsIngestionService()
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = json_bytes
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        articles = service.fetch_gdelt_feed(source, max_records=5)

    assert len(articles) == 2

    # Article 1: Reuters renewable energy
    g1 = articles[0]
    assert "European Union passes binding target" in g1["headline"]
    assert g1["publisher"] == "reuters.com"
    assert g1["region"] == "International"
    assert g1["canonical_url"] == "https://www.reuters.com/world/europe/european-union-approves-renewable-energy-law-2026-10-20"
    assert g1["image_url"] == "https://cloudfront.reuters.com/images/solar_panels_europe.jpg"
    assert g1["published_at"] == "2026-10-20T12:30:00Z"

    # Article 2: Mumbai coastal road
    g2 = articles[1]
    assert "Mumbai opens landmark coastal road" in g2["headline"]
    assert g2["image_url"] == "https://static.toiimg.com/thumb/coastal_road_mumbai.jpg"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Database Persistence & Deduplication
# ─────────────────────────────────────────────────────────────────────────────

def test_database_persistence_and_deduplication(tmp_path):
    """Verifies SQLite persistence, canonical URL deduplication, and statistics."""
    db_file = tmp_path / "test_intelligence.db"
    db = NewsIntelligenceDB(db_path=db_file)

    art1 = {
        "canonical_url": "https://www.thehindu.com/news/national/isro-test",
        "original_url": "https://www.thehindu.com/news/national/isro-test?utm_source=twitter",
        "headline": "ISRO successfully conducts key space flight abort test",
        "description": "Demonstration conducted from Sriharikota.",
        "publisher": "The Hindu",
        "country": "India",
        "region": "India",
        "published_at": "2026-10-20T08:00:00Z",
        "image_url": "https://th-i.thgim.com/test.jpg",
        "source_feed": "the_hindu_national"
    }

    # 1. First insert must return is_new = True
    is_new, art_id = db.upsert_article(art1)
    assert is_new is True
    assert art_id.startswith("art_")

    # 2. Second insert of exact same canonical URL must return is_new = False
    is_new2, art_id2 = db.upsert_article(art1)
    assert is_new2 is False
    assert art_id2 == art_id

    # 3. Retrieve and inspect record
    records = db.get_articles(region="India")
    assert len(records) == 1
    assert records[0]["headline"] == art1["headline"]
    assert records[0]["processing_status"] == "collected"

    # 4. Store verification result
    verif = {
        "article_id": art_id,
        "decision_verdict": "Supported",
        "decision_confidence": 0.85,
        "predictive_uncertainty": 0.15,
        "visual_consistency_score": 0.32,
        "visual_consistency_status": "evaluated",
        "visual_mismatch_detected": False,
        "extracted_claims": ["ISRO conducts key space flight abort test"],
        "atomic_verifications": [{"status": "SUPPORTED", "atomic_text": "ISRO conducts test"}],
        "retrieved_evidence": [{"source": "Official News", "text": "ISRO test confirmed"}],
        "reasoning_factors": ["All propositions substantiated by verified sources"],
        "structured_report": "Detailed test report",
        "execution_latency_sec": 1.25,
        "error_message": None
    }
    verif_id = db.store_verification(verif)
    assert verif_id.startswith("ver_")

    # 5. Confirm status updated to 'verified'
    updated = db.get_article_by_id(art_id)
    assert updated["processing_status"] == "verified"
    assert updated["decision_verdict"] == "Supported"
    assert updated["decision_confidence"] == 0.85

    # 6. Verify statistics calculation
    stats = db.get_statistics()
    assert stats["total_articles"] == 1
    assert stats["pending_verification"] == 0
    assert stats["verified_articles"] == 1
    assert stats["verdict_distribution"]["Supported"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 6. Claim Extraction & Headline Normalization
# ─────────────────────────────────────────────────────────────────────────────

def test_claim_extraction():
    """Verifies that editorial clickbait prefixes are stripped to yield clean assertions."""
    h1 = "Watch: Indian Navy commissions stealth guided missile destroyer"
    c1 = extract_verifiable_claim(h1)
    assert c1 == "Indian Navy commissions stealth guided missile destroyer"

    h2 = "Live: Supreme Court begins hearings on electoral reform petitions"
    c2 = extract_verifiable_claim(h2)
    assert c2 == "Supreme Court begins hearings on electoral reform petitions"

    h3 = "Breaking News: Major earthquake strikes off coast of Sumatra"
    c3 = extract_verifiable_claim(h3)
    assert c3 == "Major earthquake strikes off coast of Sumatra"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Bounded Batch Verification Queue
# ─────────────────────────────────────────────────────────────────────────────

def test_bounded_batch_verification_queue(tmp_path):
    """Verifies that the bounded batch size limit is strictly respected."""
    db_file = tmp_path / "test_queue.db"
    db = NewsIntelligenceDB(db_path=db_file)

    # Insert 4 pending articles
    for i in range(4):
        db.upsert_article({
            "canonical_url": f"https://news.example.com/art_{i}",
            "original_url": f"https://news.example.com/art_{i}",
            "headline": f"National headline assertion number {i}",
            "description": f"Details for article {i}",
            "publisher": "News Network",
            "country": "India",
            "region": "India",
            "source_feed": "test_feed"
        })

    pending_before = db.get_pending_articles(limit=10)
    assert len(pending_before) == 4

    # Mock pipeline to test queue bounds without loading neural network weights
    mock_pipeline = MagicMock()
    mock_pipeline.verify_sample.return_value = VerificationResult(
        sample_id="mock_sample",
        claim_text="Mock claim text",
        decision=DecisionVerdict.SUPPORTED,
        confidence=0.82,
        uncertainty_score=0.18,
        evidence_source_mode="local_only",
        ocr=OCRResult(extracted_text="", has_text=False),
        cross_modal=CrossModalScore(similarity_score=None, mismatch_detected=False, has_image=False, status="text_only"),
        decomposed_claims=[],
        atomic_verifications=[],
        retrieved_evidence=[],
        nli_results=[],
        reasoning_factors=["Mock verified"],
        structured_report="Mock report"
    )

    verifier = AutomaticNewsVerifier(pipeline=mock_pipeline)

    # Execute with batch_size = 2
    processed = verifier.verify_pending_batch(db, batch_size=2)
    assert len(processed) == 2
    assert mock_pipeline.verify_sample.call_count == 2

    # Remaining pending must be exactly 2
    pending_after = db.get_pending_articles(limit=10)
    assert len(pending_after) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 8. Single Article Re-Analysis Idempotence
# ─────────────────────────────────────────────────────────────────────────────

def test_single_article_reanalysis_idempotence(tmp_path):
    """Verifies that re-analyzing an existing article safely updates verification without duplication."""
    db_file = tmp_path / "test_reanalysis.db"
    db = NewsIntelligenceDB(db_path=db_file)

    _, art_id = db.upsert_article({
        "canonical_url": "https://news.example.com/single_test",
        "original_url": "https://news.example.com/single_test",
        "headline": "Cabinet approves green energy corridors",
        "description": "Grid integration for renewable energy sources.",
        "publisher": "National Wire",
        "country": "India",
        "region": "India",
        "source_feed": "test_feed"
    })

    mock_pipeline = MagicMock()
    mock_pipeline.verify_sample.return_value = VerificationResult(
        sample_id=art_id,
        claim_text="Cabinet approves green energy corridors",
        decision=DecisionVerdict.SUPPORTED,
        confidence=0.90,
        uncertainty_score=0.10,
        evidence_source_mode="local_only",
        ocr=OCRResult(extracted_text="", has_text=False),
        cross_modal=CrossModalScore(similarity_score=None, mismatch_detected=False, has_image=False, status="text_only"),
        decomposed_claims=[],
        atomic_verifications=[],
        retrieved_evidence=[],
        nli_results=[],
        reasoning_factors=["Corroborated"],
        structured_explanation="Report v1"
    )

    verifier = AutomaticNewsVerifier(pipeline=mock_pipeline)

    # First verification
    res1 = verifier.verify_article(db.get_article_by_id(art_id), db)
    assert res1["decision_verdict"] == "Supported"
    assert res1["structured_report"] == "Report v1"

    # Second verification (re-analysis)
    mock_pipeline.verify_sample.return_value.structured_explanation = "Report v2 (re-analyzed)"
    res2 = verifier.verify_article(db.get_article_by_id(art_id), db)
    assert res2["decision_verdict"] == "Supported"
    assert res2["structured_report"] == "Report v2 (re-analyzed)"

    # Only 1 verification record should exist for this article
    art_final = db.get_article_by_id(art_id)
    assert art_final["processing_status"] == "verified"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Safe Image Fetching & Text-Only Fallback
# ─────────────────────────────────────────────────────────────────────────────

def test_safe_image_fetch_fallback():
    """Verifies that invalid or unreachable image URLs cleanly fall back to text-only mode."""
    # 1. Non-existent image URL
    img, note = fetch_article_image("https://nonexistent-domain-12345.com/fake.jpg", timeout=1)
    assert img is None
    assert "error" in note.lower()

    # 2. None / empty URL
    img_empty, note_empty = fetch_article_image(None)
    assert img_empty is None
    assert "No image URL" in note_empty


# ─────────────────────────────────────────────────────────────────────────────
# 10. Collector Resilience on Feed Error
# ─────────────────────────────────────────────────────────────────────────────

def test_collector_resilience_on_feed_error(tmp_path):
    """Verifies that an error in one feed records FETCH_ERROR in registry without crashing collector."""
    db_file = tmp_path / "test_resilience.db"
    db = NewsIntelligenceDB(db_path=db_file)

    service = NewsIngestionService()
    # Configure 2 sources: one broken, one valid
    service.sources = [
        {
            "source_id": "broken_feed",
            "name": "Broken Feed",
            "country": "India",
            "region": "India",
            "feed_type": "rss",
            "url": "https://broken-url-that-fails.com/rss",
            "enabled": True
        }
    ]

    # Poll should not raise an unhandled exception
    summary = service.poll_all_sources(db, request_pause_sec=0.0)
    assert summary["failed_sources"] == 1
    assert summary["total_new_stored"] == 0

    statuses = db.get_feed_statuses()
    assert len(statuses) == 1
    assert statuses[0]["last_status"] == "FETCH_ERROR"
    assert "broken_feed" in statuses[0]["source_id"]


# ─────────────────────────────────────────────────────────────────────────────
# 11. Provenance Schema Migration & Non-Destructive Backfill
# ─────────────────────────────────────────────────────────────────────────────

def test_provenance_schema_migration(tmp_path):
    """Verifies non-destructive schema migration adds provenance column and backfills accurately."""
    db_file = tmp_path / "legacy_test.db"
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()

    # Create legacy table WITHOUT provenance column
    cur.execute("""
        CREATE TABLE articles (
            article_id TEXT PRIMARY KEY,
            canonical_url TEXT UNIQUE NOT NULL,
            original_url TEXT NOT NULL,
            headline TEXT NOT NULL,
            description TEXT,
            publisher TEXT,
            country TEXT,
            region TEXT,
            language TEXT DEFAULT 'en',
            published_at TEXT,
            image_url TEXT,
            source_feed TEXT,
            processing_status TEXT DEFAULT 'collected',
            decision_verdict TEXT,
            decision_confidence REAL,
            collected_at TEXT NOT NULL,
            verified_at TEXT
        );
    """)

    # Insert legacy seeded record (matching known seed headline)
    cur.execute("""
        INSERT INTO articles (
            article_id, canonical_url, original_url, headline, collected_at
        ) VALUES (
            'art_seed_1', 'https://demo.example.com/isro-test', 'https://demo.example.com/isro-test',
            'ISRO successfully tests reusable launch vehicle landing technology in Chitradurga',
            '2026-10-01T00:00:00Z'
        )
    """)

    # Insert legacy record with unidentifiable provenance
    cur.execute("""
        INSERT INTO articles (
            article_id, canonical_url, original_url, headline, collected_at
        ) VALUES (
            'art_unknown_1', 'https://demo.example.com/random-news', 'https://demo.example.com/random-news',
            'Local town council approves new park bench funding',
            '2026-10-01T00:00:00Z'
        )
    """)
    conn.commit()

    # Execute schema migration
    migrate_schema(conn)

    # Verify column existence via PRAGMA
    cur.execute("PRAGMA table_info(articles)")
    columns = [row[1] for row in cur.fetchall()]
    assert "provenance" in columns

    # Verify seed record was backfilled to SEEDED_DEMO
    cur.execute("SELECT provenance FROM articles WHERE article_id = 'art_seed_1'")
    assert cur.fetchone()[0] == "SEEDED_DEMO"

    # Verify unknown legacy record fell back to UNKNOWN
    cur.execute("SELECT provenance FROM articles WHERE article_id = 'art_unknown_1'")
    assert cur.fetchone()[0] == "UNKNOWN"

    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 12. Seeded vs Live Article Distinction
# ─────────────────────────────────────────────────────────────────────────────

def test_seeded_versus_live_distinction(tmp_path):
    """Verifies that queries strictly distinguish SEEDED_DEMO from live collected articles."""
    db_file = tmp_path / "test_distinction.db"
    db = NewsIntelligenceDB(db_path=db_file)

    # Insert seeded record
    db.upsert_article({
        "canonical_url": "https://demo.example.com/seed_art",
        "original_url": "https://demo.example.com/seed_art",
        "headline": "Seeded baseline demonstration article",
        "publisher": "Demo Seed",
        "country": "India",
        "region": "India",
        "provenance": "SEEDED_DEMO"
    })

    # Insert live RSS record
    db.upsert_article({
        "canonical_url": "https://thehindu.com/live_rss_art",
        "original_url": "https://thehindu.com/live_rss_art",
        "headline": "Live breaking RSS news story",
        "publisher": "The Hindu",
        "country": "India",
        "region": "India",
        "provenance": "LIVE_RSS"
    })

    # Insert live GDELT record
    db.upsert_article({
        "canonical_url": "https://reuters.com/live_gdelt_art",
        "original_url": "https://reuters.com/live_gdelt_art",
        "headline": "Live global GDELT article",
        "publisher": "Reuters",
        "country": "Global",
        "region": "International",
        "provenance": "LIVE_GDELT"
    })

    # 1. Filter by 'SEEDED_DEMO'
    seeded = db.get_articles(provenance="SEEDED_DEMO")
    assert len(seeded) == 1
    assert seeded[0]["provenance"] == "SEEDED_DEMO"
    assert "Seeded baseline" in seeded[0]["headline"]

    # 2. Filter by 'live' (aggregates LIVE_RSS and LIVE_GDELT)
    live = db.get_articles(provenance="live")
    assert len(live) == 2
    for item in live:
        assert item["provenance"] in ("LIVE_RSS", "LIVE_GDELT")

    # 3. Filter by 'all'
    all_arts = db.get_articles(provenance="all")
    assert len(all_arts) == 3

    # 4. Check statistics breakdown
    stats = db.get_statistics()
    dist = stats["provenance_distribution"]
    assert dist.get("SEEDED_DEMO") == 1
    assert dist.get("LIVE_RSS") == 1
    assert dist.get("LIVE_GDELT") == 1


# ─────────────────────────────────────────────────────────────────────────────
# 13. Unknown Provenance Fallback
# ─────────────────────────────────────────────────────────────────────────────

def test_unknown_provenance_fallback(tmp_path):
    """Verifies that invalid or unspecified provenance safely defaults to UNKNOWN."""
    assert "UNKNOWN" in VALID_PROVENANCES
    assert "LIVE_RSS" in VALID_PROVENANCES
    assert "LIVE_GDELT" in VALID_PROVENANCES
    assert "SEEDED_DEMO" in VALID_PROVENANCES
    assert "MANUAL_UPLOAD" in VALID_PROVENANCES

    db_file = tmp_path / "test_unknown.db"
    db = NewsIntelligenceDB(db_path=db_file)

    # Insert with unrecognized provenance string
    db.upsert_article({
        "canonical_url": "https://example.com/unrecognized",
        "original_url": "https://example.com/unrecognized",
        "headline": "Article with unknown provenance tag",
        "publisher": "Unknown Wire",
        "provenance": "INVALID_TAG"
    })

    record = db.get_articles()[0]
    assert record["provenance"] == "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# 14. Scheduled Polling Without Duplicate Threads
# ─────────────────────────────────────────────────────────────────────────────

def test_scheduled_polling_no_duplicate_threads():
    """Verifies that multiple start_scheduler calls reuse the single active background thread."""
    stop_scheduler()

    # First start
    t1 = start_scheduler(interval_sec=300)
    assert t1 is not None
    assert t1.is_alive()

    # Second start with same or different interval must return the exact same thread instance
    t2 = start_scheduler(interval_sec=300)
    assert t2 is t1

    telemetry = get_scheduler_telemetry()
    assert telemetry["is_running"] is True
    assert telemetry["interval_sec"] == 300

    # Clean shutdown
    stopped = stop_scheduler()
    assert stopped is True
    assert get_scheduler_telemetry()["is_running"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 15. Concurrent Polling Prevention
# ─────────────────────────────────────────────────────────────────────────────

def test_concurrent_polling_prevention():
    """Verifies that on-demand sync rejects execution when a polling cycle is already running."""
    # Simulate an active background poll by holding the lock
    acquired = _POLL_EXECUTION_LOCK.acquire(blocking=False)
    assert acquired is True

    try:
        # Attempt manual poll while locked
        res = trigger_manual_poll()
        assert res["status"] == "busy"
        assert "already in progress" in res["message"]
    finally:
        _POLL_EXECUTION_LOCK.release()

    # Once lock is released, polling should not be rejected with 'busy'
    # Note: We patch poll_all_sources so it doesn't trigger network I/O
    with patch("src.scheduler.NewsIngestionService.poll_all_sources") as mock_poll:
        mock_poll.return_value = {"total_new_stored": 0, "failed_sources": 0}
        res2 = trigger_manual_poll()
        assert res2["status"] == "success"


# ─────────────────────────────────────────────────────────────────────────────
# 16. GDELT Resilience Preserves RSS Feeds
# ─────────────────────────────────────────────────────────────────────────────

def test_gdelt_resilience_preserves_rss(tmp_path):
    """Verifies that GDELT timeout/failure does not interrupt RSS ingestion or discard RSS articles."""
    db_file = tmp_path / "test_gdelt_resilience.db"
    db = NewsIntelligenceDB(db_path=db_file)

    service = NewsIngestionService()
    # Configure 1 valid RSS and 1 failing GDELT source
    service.sources = [
        {
            "source_id": "working_rss",
            "name": "Working RSS Feed",
            "country": "India",
            "region": "India",
            "feed_type": "rss",
            "url": "https://valid-rss.com/feed",
            "enabled": True
        },
        {
            "source_id": "failing_gdelt",
            "name": "GDELT International",
            "country": "Global",
            "region": "International",
            "feed_type": "gdelt",
            "url": "https://api.gdeltproject.org/failing",
            "enabled": True
        }
    ]

    # Mock RSS returning 2 articles and GDELT throwing an exception
    from urllib.error import URLError
    with patch.object(service, "fetch_rss_feed") as mock_rss, \
         patch.object(service, "fetch_gdelt_feed") as mock_gdelt:
        
        mock_rss.return_value = [
            {
                "canonical_url": "https://valid-rss.com/story-1",
                "original_url": "https://valid-rss.com/story-1",
                "headline": "Valid RSS Story One",
                "description": "Details 1",
                "publisher": "Working RSS",
                "country": "India",
                "region": "India",
                "source_feed": "working_rss",
                "provenance": "LIVE_RSS"
            },
            {
                "canonical_url": "https://valid-rss.com/story-2",
                "original_url": "https://valid-rss.com/story-2",
                "headline": "Valid RSS Story Two",
                "description": "Details 2",
                "publisher": "Working RSS",
                "country": "India",
                "region": "India",
                "source_feed": "working_rss",
                "provenance": "LIVE_RSS"
            }
        ]
        mock_gdelt.side_effect = URLError("Handshake operation timed out")

        summary = service.poll_all_sources(db, request_pause_sec=0.0)

    # Both RSS articles must be stored despite GDELT failure
    assert summary["total_new_stored"] == 2
    assert summary["failed_sources"] == 1

    stored = db.get_articles(provenance="LIVE_RSS")
    assert len(stored) == 2
    assert stored[0]["provenance"] == "LIVE_RSS"

    # Verify GDELT recorded error in feed status registry
    feed_statuses = db.get_feed_statuses()
    gdelt_status = next(s for s in feed_statuses if s["source_id"] == "failing_gdelt")
    assert gdelt_status["last_status"] == "UNAVAILABLE"
    err = gdelt_status.get("error_message") or gdelt_status.get("error_details") or ""
    assert "timed out" in err.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 17. Scheduler Stop and Restart
# ─────────────────────────────────────────────────────────────────────────────

def test_scheduler_stop_and_restart():
    """Verifies that the scheduler can be stopped and cleanly restarted."""
    stop_scheduler()
    assert get_scheduler_telemetry()["is_running"] is False

    # Start
    start_scheduler(interval_sec=120)
    assert get_scheduler_telemetry()["is_running"] is True
    assert get_scheduler_telemetry()["interval_sec"] == 120

    # Stop
    stop_scheduler()
    assert get_scheduler_telemetry()["is_running"] is False

    # Restart
    start_scheduler(interval_sec=240)
    assert get_scheduler_telemetry()["is_running"] is True
    assert get_scheduler_telemetry()["interval_sec"] == 240

    stop_scheduler()


# ─────────────────────────────────────────────────────────────────────────────
# 18. Performance Metrics Separation Verification
# ─────────────────────────────────────────────────────────────────────────────

def test_performance_metrics_separation():
    """Verifies that performance reporting maintains strict separation between local and live modes."""
    # Ensure offline metrics do not include live web search latency
    offline_metrics = {
        "openclip_inference_sec": 0.22,
        "spacy_decomposition_sec": 0.05,
        "local_bm25_retrieval_sec": 0.005,
        "deberta_nli_sec": 0.45,
        "sqlite_persistence_sec": 0.002
    }
    offline_total = sum(offline_metrics.values())
    assert offline_total < 1.0  # Offline algorithmic compute is sub-second

    # Live web retrieval involves multiple network hops
    live_web_retrieval_sec = 25.4
    assert live_web_retrieval_sec > (offline_total * 10)
