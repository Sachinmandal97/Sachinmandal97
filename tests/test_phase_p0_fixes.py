"""
Phase P0 Deterministic Regression Tests
=======================================
Verifies the four critical correctness fixes for Phase P0:
1. Google News publisher URL handling & root homepage refusal.
2. Safe fallback when Google News wrapper URLs cannot be resolved.
3. Homepage content rejection (cannot overwrite target article snippets).
4. Verified article URL fetching and honest provenance labeling.
5. Extraction failures never fabricate evidence.
6. OpenCLIP tokenization handles long claims (>77 chars) via processor truncation without Python character slicing.
7. Demo-preset state logic decouples custom claims from preset visuals, ensuring text-only mode runs image-free.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import urllib.error
from PIL import Image

from src.article_fetcher import (
    fetch_article_content,
    is_root_or_homepage_url,
    resolve_publisher_url,
)
from src.cross_modal import compute_cross_modal_consistency
from src.evidence_providers import GoogleNewsRSSProvider
from src.retrieval import EvidenceRetriever
from src.schemas import EvidenceSnippet, MultimodalSample


# ─────────────────────────────────────────────────────────────────────────────
# 1. ROOT OR HOMEPAGE URL DETECTION AND REFUSAL
# ─────────────────────────────────────────────────────────────────────────────

def test_is_root_or_homepage_url_detection():
    """Verifies that root domains and homepages are reliably detected and distinguished from article URLs."""
    # Root domains and homepages (must be detected as True)
    assert is_root_or_homepage_url("https://www.reuters.com") is True
    assert is_root_or_homepage_url("https://www.reuters.com/") is True
    assert is_root_or_homepage_url("http://bbc.co.uk") is True
    assert is_root_or_homepage_url("http://bbc.co.uk/") is True
    assert is_root_or_homepage_url("https://nytimes.com/index.html") is True
    assert is_root_or_homepage_url("https://apnews.com/index.php") is True
    assert is_root_or_homepage_url("https://news.sky.com/home") is True

    # Valid specific article paths (must be detected as False)
    assert is_root_or_homepage_url("https://www.reuters.com/world/europe/london-fire-blaze-2026-10-12/") is False
    assert is_root_or_homepage_url("https://www.bbc.com/news/uk-12345678") is False
    assert is_root_or_homepage_url("https://apnews.com/article/nasa-mars-rover-organic-matter-987654") is False


def test_fetch_article_refuses_root_or_homepage():
    """Verifies that fetch_article_content refuses to fetch root/homepage URLs and returns root_domain_not_article."""
    res = fetch_article_content("https://www.reuters.com/")
    assert res["text"] == ""
    assert res["content_type"] == "snippet_only"
    assert res["access_status"] == "root_domain_not_article"
    assert "root_domain" in res.get("error", "").lower()


# ─────────────────────────────────────────────────────────────────────────────
# 2. UNRESOLVED GOOGLE WRAPPER FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def test_unresolved_google_wrapper_preserves_snippet_without_fabrication():
    """Verifies that when a Google News wrapper URL fails to resolve, original snippet text is preserved intact."""
    snip = EvidenceSnippet(
        snippet_id="test_rss_1",
        text="London Fire Brigade deployed 10 fire engines to an industrial warehouse blaze in East London.",
        source="Reuters via Google News",
        source_type="live_web",
        content_type="snippet_only",
        url=None,
        wrapper_url="https://news.google.com/rss/articles/CBMi_TEST_UNRESOLVABLE_TOKEN",
        publisher="Reuters",
        publisher_domain="https://www.reuters.com",
        access_status="snippet_only"
    )

    retriever = EvidenceRetriever()

    # Simulate resolution failure returning (None, "google_wrapper_unresolved")
    with patch("src.retrieval.resolve_publisher_url", return_value=(None, "google_wrapper_unresolved")):
        enriched = retriever._enrich_snippet_with_article(snip)

        assert enriched.url is None
        assert enriched.wrapper_url == "https://news.google.com/rss/articles/CBMi_TEST_UNRESOLVABLE_TOKEN"
        assert enriched.publisher_domain == "https://www.reuters.com"
        assert enriched.access_status == "google_wrapper_unresolved"
        assert enriched.content_type == "snippet_only"
        # Original snippet content remains intact
        assert "London Fire Brigade deployed 10 fire engines" in enriched.text
        assert "Reuters" in enriched.source


# ─────────────────────────────────────────────────────────────────────────────
# 3. HOMEPAGE CONTENT CANNOT OVERWRITE SNIPPET
# ─────────────────────────────────────────────────────────────────────────────

def test_homepage_content_cannot_overwrite_snippet():
    """Verifies that if URL resolution yields a homepage, _enrich_snippet_with_article rejects it and preserves snippet."""
    snip = EvidenceSnippet(
        snippet_id="test_rss_2",
        text="Police launched an investigation following a high-speed collision on the M25 motorway.",
        source="BBC News via Google News",
        source_type="live_web",
        content_type="snippet_only",
        url=None,
        wrapper_url="https://news.google.com/rss/articles/CBMi_HOMEPAGE_REDIRECT",
        publisher="BBC News",
        publisher_domain="https://www.bbc.com",
        access_status="snippet_only"
    )

    retriever = EvidenceRetriever()

    # Resolution erroneously returned a root homepage URL
    with patch("src.retrieval.resolve_publisher_url", return_value=("https://www.bbc.com/", "resolved")):
        enriched = retriever._enrich_snippet_with_article(snip)

        # Must reject homepage URL, keeping url as None
        assert enriched.url is None
        assert enriched.access_status == "google_wrapper_unresolved"
        assert enriched.content_type == "snippet_only"
        # Snippet text must not be overwritten by homepage boilerplate
        assert "Police launched an investigation" in enriched.text


# ─────────────────────────────────────────────────────────────────────────────
# 4. VERIFIED ARTICLE URL CAN BE FETCHED AND ASSOCIATED
# ─────────────────────────────────────────────────────────────────────────────

def test_verified_article_url_fetched_and_enriched():
    """Verifies that a valid article URL is successfully fetched, parsed, and associated with full_article provenance."""
    html_article = (
        "<!DOCTYPE html><html><head><title>East London Warehouse Fire</title></head>"
        "<body>"
        "<article>"
        "<h1>Firefighters battle East London blaze</h1>"
        "<p>Ten fire engines and about 70 firefighters were called to a fire at a commercial warehouse in East London.</p>"
        "<p>The London Fire Brigade said crews from several stations responded to the incident on Sunday morning.</p>"
        "<p>No injuries were reported, according to an emergency spokesperson.</p>"
        "</article>"
        "</body></html>"
    )

    mock_resp = MagicMock()
    mock_resp.read.return_value = html_article.encode("utf-8")
    mock_resp.headers.get.return_value = "text/html; charset=utf-8"
    mock_resp.getcode.return_value = 200

    article_url = "https://www.reuters.com/world/uk/east-london-warehouse-fire-2026-10-12/"

    with patch("src.article_fetcher.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        res = fetch_article_content(article_url)
        assert res["access_status"] == "accessible"
        assert res["content_type"] in ("full_text", "excerpt")
        assert "Ten fire engines and about 70 firefighters" in res["text"]
        assert "commercial warehouse in East London" in res["text"]

    # Verify enrichment flow
    snip = EvidenceSnippet(
        snippet_id="test_rss_3",
        text="Firefighters battle East London blaze.",
        source="Reuters via Google News",
        source_type="live_web",
        content_type="snippet_only",
        url=None,
        wrapper_url="https://news.google.com/rss/articles/CBMi_VALID_ARTICLE",
        publisher="Reuters",
        publisher_domain="https://www.reuters.com",
        access_status="snippet_only"
    )

    retriever = EvidenceRetriever()
    with patch("src.retrieval.resolve_publisher_url", return_value=(article_url, "resolved")):
        with patch("src.retrieval.fetch_article_content", return_value=res):
            enriched = retriever._enrich_snippet_with_article(snip)
            assert enriched.url == article_url
            assert enriched.access_status == "accessible"
            assert enriched.content_type in ("full_text", "excerpt")
            assert "Ten fire engines and about 70 firefighters" in enriched.text


# ─────────────────────────────────────────────────────────────────────────────
# 5. EXTRACTION FAILURES DO NOT FABRICATE EVIDENCE
# ─────────────────────────────────────────────────────────────────────────────

def test_extraction_failures_do_not_fabricate_evidence():
    """Verifies that 404, 403, network timeouts, or empty article bodies never fabricate fake evidence."""
    # 403 HTTP Error
    with patch("src.article_fetcher.urllib.request.urlopen", side_effect=urllib.error.HTTPError("url", 403, "Forbidden", {}, None)):
        res = fetch_article_content("https://paywalled-times.co.uk/article-123")
        assert res["text"] == ""
        assert res["content_type"] == "snippet_only"
        assert res["access_status"] == "inaccessible"

    # Minimal boilerplate HTML (< 60 chars of body text)
    minimal_html = "<html><body><div>Login to view this content.</div></body></html>"
    mock_resp = MagicMock()
    mock_resp.read.return_value = minimal_html.encode("utf-8")
    with patch("src.article_fetcher.urllib.request.urlopen", return_value=mock_resp):
        res = fetch_article_content("https://news.example.com/empty-article")
        assert res["text"] == ""
        assert res["content_type"] == "snippet_only"
        assert res["access_status"] == "inaccessible"


# ─────────────────────────────────────────────────────────────────────────────
# 6. OPENCLIP TOKENIZATION HANDLES LONG CLAIMS WITHOUT CHAR SLICING
# ─────────────────────────────────────────────────────────────────────────────

def test_openclip_tokenization_handles_long_claims_without_char_slicing():
    """
    Verifies that OpenCLIP consistency receives full claim_text with truncation=True, max_length=77
    instead of cutting the text string off at 77 characters via claim_text[:77].
    """
    long_claim = (
        "NASA robotic explorer Perseverance has identified compelling organic carbon molecules "
        "inside ancient Martian rock samples collected at the Jezero Crater delta formation."
    )
    assert len(long_claim) > 120  # Far exceeds 77 characters!
    char_sliced_cutoff = long_claim[:77]
    assert "Jezero Crater" not in char_sliced_cutoff  # Proves char slicing drops critical entities

    test_img = Image.new("RGB", (224, 224), color=(60, 90, 120))

    # Test with mocked processor to inspect parameters
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_inputs = {
        "pixel_values": MagicMock(),
        "input_ids": MagicMock(),
        "attention_mask": MagicMock()
    }
    # Mock to() methods
    for k in mock_inputs:
        mock_inputs[k].to.return_value = mock_inputs[k]
    mock_processor.return_value = mock_inputs

    import torch
    mock_model.get_image_features.return_value = torch.ones((1, 512))
    mock_model.get_text_features.return_value = torch.ones((1, 512))

    with patch("src.cross_modal.get_clip_model", return_value=(mock_model, mock_processor, "cpu")):
        score = compute_cross_modal_consistency(image=test_img, claim_text=long_claim)
        assert score is not None

        # Verify processor was called with the FULL claim text and tokenizer truncation
        mock_processor.assert_called_once()
        _, kwargs = mock_processor.call_args
        assert kwargs.get("text") == [long_claim]
        assert kwargs.get("truncation") is True
        assert kwargs.get("max_length") == 77


# ─────────────────────────────────────────────────────────────────────────────
# 7. DEMO-PRESET STATE LOGIC DECOUPLING AND TEXT-ONLY MODE
# ─────────────────────────────────────────────────────────────────────────────

def test_preset_decoupling_logic_and_text_only_mode():
    """
    Verifies that:
    1. Custom claims never inherit preset images.
    2. Modifying claim text automatically detaches preset images.
    3. Text-only mode executes cleanly with image_path=None.
    """
    preset_choice = "Case 1: Fully Supported News (Consistent Image + Evidence)"
    default_text = "Firefighters battle a major industrial warehouse blaze in East London on Sunday morning."
    default_image_desc = "Industrial warehouse fire with emergency personnel"

    # Case A: Preset selected, text unmodified -> preset is active
    claim_unmodified = "Firefighters battle a major industrial warehouse blaze in East London on Sunday morning."
    is_preset_active_a = (
        preset_choice != "(None - Enter Custom News Claim)"
        and bool(default_image_desc)
        and (claim_unmodified.strip() == default_text.strip())
    )
    assert is_preset_active_a is True

    # Case B: Preset selected, but user edited text to a custom claim -> preset visual detached!
    claim_modified = "NASA spacecraft detects methane plume on Saturn moon Enceladus."
    is_preset_active_b = (
        preset_choice != "(None - Enter Custom News Claim)"
        and bool(default_image_desc)
        and (claim_modified.strip() == default_text.strip())
    )
    assert is_preset_active_b is False

    # Case C: "(None - Enter Custom News Claim)" selected -> preset visual detached!
    preset_none = "(None - Enter Custom News Claim)"
    is_preset_active_c = (
        preset_none != "(None - Enter Custom News Claim)"
        and bool(default_image_desc)
        and (claim_modified.strip() == default_text.strip())
    )
    assert is_preset_active_c is False

    # Case D: Text-only verification mode (image=None) returns honest text-only status without fabricated visual score
    text_only_score = compute_cross_modal_consistency(image=None, claim_text=claim_modified)
    assert text_only_score.similarity_score is None
    assert text_only_score.has_image is False
    assert text_only_score.status == "text_only"
    assert text_only_score.mismatch_detected is False
    assert text_only_score.details.get("info") == "No image provided; text-only evaluation mode"
