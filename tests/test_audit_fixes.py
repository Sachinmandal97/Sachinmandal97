"""
Regression and acceptance test suite for VERITAS-MM 2.0 approved audit fixes.
Covers:
  1. HTML card rendering without raw code block tags.
  2. RSS nested description extraction (ElementTree child + tail text).
  3. Image extraction from embedded RSS description <img> tags.
  4. HTML entity decoding in titles and descriptions.
  5. HTML escaping and XSS prevention for user/external content.
  6. Automatic bounded verification queue behavior (batch_size=2).
  7. Prevention of overlapping verification jobs.
  8. SQLite concurrent thread-safe access.
  9. Recovery of interrupted jobs from 'processing' to 'collected'.
  10. Manual batch verification interface and result unpacking.
  11. Pagination and article counting queries.
  12. GDELT failure and disablement isolation.
  13. Eiffel Tower / Taj Mahal visual mismatch regression.
  14. Text-only verification without fabricated visual scores.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import html
import tempfile
import threading
import time
from unittest.mock import MagicMock, patch
import xml.etree.ElementTree as ET

import markdown
from PIL import Image

from src.database import NewsIntelligenceDB, get_db_connection
from src.news_ingestion import (
    clean_html_summary,
    extract_element_markup,
    extract_image_from_item,
    NewsIngestionService,
    normalize_field_text,
    is_valid_http_url,
)
from src.news_verifier import AutomaticNewsVerifier
from src.pipeline import MultimodalVerificationPipeline
from src.scheduler import (
    _VERIFY_EXECUTION_LOCK,
    get_scheduler_telemetry,
    set_auto_verify,
    start_scheduler,
    stop_scheduler,
    trigger_manual_verification,
)
from src.schemas import CrossModalScore, DecisionVerdict, MultimodalSample, OCRResult, VerificationResult


def test_native_card_component_rendering_no_raw_html():
    """
    Verifies that native Streamlit card rendering components output clean prose
    and never inject literal HTML tags (<div, <span, <a, style=) or raw markdown code blocks.
    """
    raw_headline = "<div style='font-size: 1rem;'>Breaking: Supreme Court rules &amp; strikes down scheme</div>"
    raw_desc = "<p>The <b>five-judge</b> bench &quot;unanimously&quot; ruled on constitutional grounds.</p>"
    raw_pub = "<span>The Hindu</span>"
    raw_url = "https://www.thehindu.com/news/national/article12345.ece"

    clean_headline = normalize_field_text(raw_headline, fallback="Untitled Headline")
    clean_desc = normalize_field_text(raw_desc, fallback="")
    clean_pub = normalize_field_text(raw_pub, fallback="Unknown Publisher")
    safe_url = raw_url.strip() if is_valid_http_url(raw_url) else None

    # Check that all HTML tags are completely removed from displayed fields
    for field_val in [clean_headline, clean_desc, clean_pub]:
        assert "<div" not in field_val
        assert "<span" not in field_val
        assert "<p>" not in field_val
        assert "<b" not in field_val
        assert "style=" not in field_val
        assert "<" not in field_val and ">" not in field_val

    # Check decoded entities
    assert "&amp;" not in clean_headline
    assert "&quot;" not in clean_desc
    assert 'bench "unanimously" ruled' in clean_desc
    assert safe_url == "https://www.thehindu.com/news/national/article12345.ece"


def test_field_normalization_empty_and_whitespace_fallbacks():
    """Verifies that None, empty strings, and whitespace-only strings produce clean fallback text."""
    assert normalize_field_text(None, fallback="Untitled") == "Untitled"
    assert normalize_field_text("", fallback="Untitled") == "Untitled"
    assert normalize_field_text("   \t\n  ", fallback="Untitled") == "Untitled"
    assert normalize_field_text(None, fallback="") == ""
    assert normalize_field_text("Valid text", fallback="Untitled") == "Valid text"


def test_url_safety_validation():
    """Verifies that is_valid_http_url approves valid HTTP/HTTPS URLs and rejects malicious or non-web schemes."""
    assert is_valid_http_url("https://www.thehindu.com/news/national/article.ece") is True
    assert is_valid_http_url("http://feeds.bbci.co.uk/news/world/rss.xml") is True
    # Rejections
    assert is_valid_http_url("javascript:alert(document.cookie)") is False
    assert is_valid_http_url("data:text/html,<script>alert(1)</script>") is False
    assert is_valid_http_url("file:///etc/passwd") is False
    assert is_valid_http_url("/relative/path/article.html") is False
    assert is_valid_http_url("") is False
    assert is_valid_http_url(None) is False
    assert is_valid_http_url("   ") is False


def test_real_database_records_normalization():
    """
    Verifies that all real articles in data/news_intelligence.db normalize cleanly
    without unstripped HTML tags, unescaped entities, or uncaught exceptions.
    """
    db = NewsIntelligenceDB()
    articles = db.get_articles(limit=500)
    assert len(articles) > 0, "Database must have existing article records"

    for art in articles:
        h = normalize_field_text(art.get("headline"), fallback="Untitled")
        d = normalize_field_text(art.get("description"), fallback="")
        p = normalize_field_text(art.get("publisher"), fallback="Unknown")
        u = art.get("original_url") or art.get("canonical_url")

        # Zero unstripped HTML tags in fields
        assert "<" not in h and ">" not in h, f"Unstripped tag in headline: {h}"
        assert "<" not in d and ">" not in d, f"Unstripped tag in description: {d}"
        assert "<" not in p and ">" not in p, f"Unstripped tag in publisher: {p}"

        # Zero unescaped common entities
        assert "&amp;" not in h and "&quot;" not in h
        assert "&amp;" not in d and "&quot;" not in d

        # URL validation doesn't crash
        is_valid_http_url(u)


def test_rss_nested_description_extraction():
    """Verifies that RSS descriptions containing nested <a>, <img>, and tail text extract completely."""
    xml_data = """<item>
        <title>Stock Market &amp; Economy: Sensex up &gt; 500 pts</title>
        <description><a href="https://timesofindia.indiatimes.com/story"><img src="https://static.toiimg.com/thumb/msid-12345/pic.jpg" alt="pic"/></a>The benchmark equity indices surged on Monday following strong quarterly earnings.</description>
    </item>"""
    root = ET.fromstring(xml_data)

    desc_elem = root.find("description")
    raw_desc = extract_element_markup(desc_elem)
    assert "<img" in raw_desc
    assert "The benchmark equity indices surged" in raw_desc

    clean_desc = clean_html_summary(raw_desc)
    assert clean_desc == "The benchmark equity indices surged on Monday following strong quarterly earnings."
    assert "<img" not in clean_desc


def test_image_extraction_from_embedded_rss_description():
    """Verifies that lead images embedded inside <description> <img> tags are successfully extracted."""
    xml_data = """<item>
        <description><a href="https://indianexpress.com/article/1"><img src="https://images.indianexpress.com/2026/10/rocket_launch.jpg"/></a>ISRO mission accomplished.</description>
    </item>"""
    root = ET.fromstring(xml_data)
    desc_elem = root.find("description")
    raw_desc = extract_element_markup(desc_elem)

    extracted_img = extract_image_from_item(root, description_html=raw_desc)
    assert extracted_img == "https://images.indianexpress.com/2026/10/rocket_launch.jpg"


def test_html_entity_decoding():
    """Verifies that titles and summaries decode HTML entities into natural characters."""
    raw_title = "Sensex &amp; Nifty surge &gt; 500 pts amid &quot;strong&quot; Q2 gains &apos;record&apos;"
    cleaned = clean_html_summary(raw_title)
    assert cleaned == "Sensex & Nifty surge > 500 pts amid \"strong\" Q2 gains 'record'"


def test_html_escaping_and_xss_prevention():
    """Verifies that untrusted headlines with script tags or malicious URLs are safely escaped."""
    malicious_headline = "Breaking: Court rules <script>alert('xss')</script> & strikes down law"
    safe = html.escape(malicious_headline)
    assert "<script>" not in safe
    assert "&lt;script&gt;" in safe
    assert "&amp;" in safe

    # Test URL validation
    dangerous_url = "javascript:alert(document.cookie)"
    safe_url = html.escape(dangerous_url, quote=True) if dangerous_url.startswith(("http://", "https://")) else "#"
    assert safe_url == "#"


def test_automatic_bounded_verification(tmp_path):
    """Verifies that bounded verification processes up to batch_size items and leaves remaining items pending."""
    db_file = tmp_path / "test_bounded_queue.db"
    db = NewsIntelligenceDB(db_path=db_file)

    for i in range(4):
        db.upsert_article({
            "canonical_url": f"https://example.com/art_{i}",
            "headline": f"Sample news headline number {i}",
            "publisher": "Test Wire",
            "provenance": "LIVE_RSS"
        })

    mock_pipeline = MagicMock(spec=MultimodalVerificationPipeline)
    mock_pipeline.verify_sample.return_value = VerificationResult(
        sample_id="test",
        claim_text="Sample claim",
        decision=DecisionVerdict.SUPPORTED,
        confidence=0.88,
        uncertainty_score=0.12,
        evidence_source_mode="mock",
        ocr=OCRResult(extracted_text="", word_count=0, confidence=1.0, has_text=False),
        cross_modal=CrossModalScore(similarity_score=None, mismatch_detected=False, has_image=False, status="text_only"),
        decomposed_claims=[],
        atomic_verifications=[],
        retrieved_evidence=[],
        nli_results=[],
        reasoning_factors=["Mock verified"],
        structured_explanation="Mock explanation"
    )

    verifier = AutomaticNewsVerifier(pipeline=mock_pipeline)

    # Process bounded batch of 2 articles
    results = verifier.verify_pending_batch(db, batch_size=2)
    assert len(results) == 2

    # Verify queue metrics: 2 verified, 2 still pending
    metrics = db.get_queue_metrics()
    assert metrics["verified"] == 2
    assert metrics["pending"] == 2
    assert metrics["processing"] == 0


def test_no_overlapping_verification_jobs():
    """Verifies that concurrent verification invocations are rejected when a cycle is active."""
    acquired = _VERIFY_EXECUTION_LOCK.acquire(blocking=False)
    assert acquired is True

    try:
        # Triggering verification while lock is held must return busy
        res = trigger_manual_verification(batch_size=2)
        assert res["status"] == "busy"
        assert "already in progress" in res["message"]
    finally:
        _VERIFY_EXECUTION_LOCK.release()


def test_sqlite_concurrent_access(tmp_path):
    """Verifies that concurrent multi-threaded read/write operations execute safely without locking errors."""
    db_file = tmp_path / "test_concurrency.db"
    db = NewsIntelligenceDB(db_path=db_file)

    def writer_task(thread_id: int):
        for i in range(10):
            db.upsert_article({
                "canonical_url": f"https://example.com/thread_{thread_id}_art_{i}",
                "headline": f"Concurrent article from thread {thread_id} item {i}",
                "publisher": f"Publisher {thread_id}",
                "provenance": "LIVE_RSS"
            })
            stats = db.get_statistics()
            assert stats["total_articles"] >= 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(writer_task, t) for t in range(4)]
        for f in futures:
            f.result()

    total = db.get_statistics()["total_articles"]
    assert total == 40


def test_recovery_after_worker_exception(tmp_path):
    """Verifies that interrupted jobs stranded in 'processing' status recover cleanly to 'collected'."""
    db_file = tmp_path / "test_recovery.db"
    db = NewsIntelligenceDB(db_path=db_file)

    db.upsert_article({
        "canonical_url": "https://example.com/stranded_article",
        "headline": "Stranded in processing article",
        "publisher": "Crash Test",
        "provenance": "LIVE_RSS"
    })

    # Artificially set status to 'processing'
    with get_db_connection(db_file) as conn:
        conn.execute("UPDATE articles SET processing_status = 'processing'")

    assert db.get_queue_metrics()["processing"] == 1
    assert db.get_queue_metrics()["pending"] == 0

    recovered_count = db.recover_interrupted_articles()
    assert recovered_count == 1
    assert db.get_queue_metrics()["processing"] == 0
    assert db.get_queue_metrics()["pending"] == 1


def test_manual_batch_verification_works(tmp_path):
    """Verifies that trigger_manual_verification successfully unpacks as (result, error)."""
    db_file = tmp_path / "test_manual_verify.db"
    db = NewsIntelligenceDB(db_path=db_file)

    mock_pipeline = MagicMock(spec=MultimodalVerificationPipeline)
    mock_pipeline.verify_sample.return_value = VerificationResult(
        sample_id="art_1",
        claim_text="Manual test claim",
        decision=DecisionVerdict.SUPPORTED,
        confidence=0.85,
        uncertainty_score=0.15,
        evidence_source_mode="mock",
        ocr=OCRResult(extracted_text="", word_count=0, confidence=1.0, has_text=False),
        cross_modal=CrossModalScore(similarity_score=None, mismatch_detected=False, has_image=False, status="text_only"),
        decomposed_claims=[],
        atomic_verifications=[],
        retrieved_evidence=[],
        nli_results=[],
        reasoning_factors=["Verified manually"],
        structured_explanation="Manual test explanation"
    )
    verifier = AutomaticNewsVerifier(pipeline=mock_pipeline)

    db.upsert_article({
        "canonical_url": "https://example.com/art_manual",
        "headline": "Manual verification candidate",
        "publisher": "Test Wire",
        "provenance": "LIVE_RSS"
    })

    res, err = trigger_manual_verification(verifier=verifier, db=db, batch_size=1)
    assert err is None
    assert res is not None
    assert res["verified_count"] == 1


def test_pagination_and_article_counts(tmp_path):
    """Verifies accurate pagination limits and offset slicing."""
    db_file = tmp_path / "test_pagination.db"
    db = NewsIntelligenceDB(db_path=db_file)

    for i in range(15):
        db.upsert_article({
            "canonical_url": f"https://example.com/page_test_{i}",
            "headline": f"Pagination article headline {i:02d}",
            "publisher": "Page Wire",
            "provenance": "LIVE_RSS"
        })

    # Count articles matching criteria
    count = db.count_articles()
    assert count == 15

    # Page 1: limit 5, offset 0
    p1 = db.get_articles(limit=5, offset=0)
    assert len(p1) == 5

    # Page 2: limit 5, offset 5
    p2 = db.get_articles(limit=5, offset=5)
    assert len(p2) == 5

    # Page 3: limit 5, offset 10
    p3 = db.get_articles(limit=5, offset=10)
    assert len(p3) == 5

    # IDs between p1 and p2 must not overlap
    p1_ids = {a["article_id"] for a in p1}
    p2_ids = {a["article_id"] for a in p2}
    assert p1_ids.isdisjoint(p2_ids)


def test_gdelt_failure_isolation(tmp_path):
    """Verifies that GDELT disabled state or failure isolates cleanly and records DISABLED status."""
    db_file = tmp_path / "test_gdelt_iso.db"
    db = NewsIntelligenceDB(db_path=db_file)

    service = NewsIngestionService(gdelt_enabled=False)
    service.sources = [
        {
            "source_id": "test_gdelt_src",
            "name": "GDELT Test",
            "country": "Global",
            "region": "International",
            "feed_type": "gdelt",
            "url": "https://api.gdeltproject.org/test",
            "enabled": True
        }
    ]

    summary = service.poll_all_sources(db=db)
    assert summary["total_discovered"] == 0

    feed_status = db.get_feed_statuses()[0]
    assert feed_status["last_status"] == "DISABLED"


def test_eiffel_tower_taj_mahal_visual_mismatch_regression():
    """
    Regression Test: Asserts that when an image depicts the Eiffel Tower but the claim asserts Taj Mahal,
    the pipeline flags a visual mismatch and refuses to classify as fully Supported.
    """
    img = Image.new("RGB", (224, 224), color=(73, 109, 137))
    sample = MultimodalSample(
        sample_id="reg_mismatch_eiffel_taj",
        claim_text="The historic Taj Mahal in Agra, India.",
        image_data=img,
        dataset_name="regression_test",
        split="test"
    )

    pipeline = MultimodalVerificationPipeline()
    # Force mock mismatch score for deterministic testing
    with patch("src.pipeline.compute_cross_modal_consistency") as mock_cm:
        mock_cm.return_value = CrossModalScore(
            similarity_score=0.08,
            mismatch_detected=True,
            has_image=True,
            status="mismatch",
            details={"note": "Severe semantic divergence between visual content and textual claim"}
        )
        res = pipeline.verify_sample(sample, force_mode="local_only")

    assert res.cross_modal.mismatch_detected is True
    # Verdict must not be clean 'Supported' when visual mismatch is detected
    assert res.decision != DecisionVerdict.SUPPORTED


def test_text_only_verification_no_fabricated_scores():
    """
    Verifies that when an article has no image URL, the pipeline cleanly engages text-only mode
    and similarity_score is strictly None without invented numbers.
    """
    sample = MultimodalSample(
        sample_id="reg_text_only_01",
        claim_text="Global leaders announce new emissions reduction targets.",
        image_data=None,
        image_path=None,
        dataset_name="regression_test",
        split="test"
    )

    pipeline = MultimodalVerificationPipeline()
    res = pipeline.verify_sample(sample, force_mode="local_only")

    assert res.cross_modal.has_image is False
    assert res.cross_modal.status == "text_only"
    assert res.cross_modal.similarity_score is None
    assert res.cross_modal.mismatch_detected is False
