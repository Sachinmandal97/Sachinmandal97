"""
Automatic News Ingestion Service for VERITAS-MM 2.0.
Collects and normalizes news articles from:
  1. Indian and International Publisher RSS 2.0 / Atom feeds.
  2. The GDELT Project DOC 2.0 Global News API.
Features:
  - Strict URL canonicalization and tracking parameter stripping.
  - Multi-format publication timestamp normalization to UTC ISO 8601.
  - Media & image enclosure extraction for multimodal verification.
  - Error isolation, retry backoff, and respectful request pacing.
"""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import html
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
from src.database import NewsIntelligenceDB
from src.utils import setup_logger

logger = setup_logger("news_ingestion")

DEFAULT_SOURCES_FILE = Path("./configs/news_sources.json")
DEFAULT_USER_AGENT = "VERITAS-MM/2.0 (Academic Multimodal Misinformation Research; +https://github.com/veritas-mm)"

# Query parameters commonly used for marketing/tracking that should be removed
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "msclkid", "ref", "ocid", "src", "ito", "rss",
    "cmpid", "feed", "feature", "source"
}


def canonicalize_url(raw_url: str) -> str:
    """
    Normalizes article URL for reliable cross-publisher duplicate detection.
    Strips tracking queries, standardizes scheme, and removes fragment.
    """
    if not raw_url:
        return ""
    try:
        parsed = urllib.parse.urlparse(raw_url.strip())
        # Clean query parameters
        queries = urllib.parse.parse_qsl(parsed.query, keep_blank_values=False)
        clean_queries = [(k, v) for k, v in queries if k.lower() not in TRACKING_PARAMS]
        clean_query_str = urllib.parse.urlencode(clean_queries)

        # Standardize path
        clean_path = parsed.path
        if clean_path.endswith("/") and len(clean_path) > 1:
            clean_path = clean_path[:-1]

        canonical = urllib.parse.urlunparse((
            parsed.scheme.lower() or "https",
            parsed.netloc.lower(),
            clean_path,
            parsed.params,
            clean_query_str,
            ""  # Strip fragment
        ))
        return canonical
    except Exception as e:
        logger.debug(f"URL canonicalization fallback for '{raw_url}': {e}")
        return raw_url.strip()


def parse_timestamp_to_iso(date_str: Optional[str]) -> Optional[str]:
    """
    Normalizes various feed date formats (RFC 822, RFC 2822, ISO 8601, GDELT)
    to a standardized UTC ISO 8601 string: YYYY-MM-DDTHH:MM:SSZ.
    """
    if not date_str or not date_str.strip():
        return None

    s = date_str.strip()

    # 1. Try RFC 2822 / RFC 822 (standard RSS date)
    try:
        dt = parsedate_to_datetime(s)
        if dt is not None:
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        pass

    # 2. Try GDELT compact format (YYYYMMDDTHHMMSSZ or YYYYMMDDHHMMSS)
    gdelt_match = re.match(r"^(\d{4})(\d{2})(\d{2})T?(\d{2})(\d{2})(\d{2})Z?$", s)
    if gdelt_match:
        y, m, d, hh, mm, ss = gdelt_match.groups()
        return f"{y}-{m}-{d}T{hh}:{mm}:{ss}Z"

    # 3. Try standard ISO 8601 / Atom formats
    iso_candidates = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d"
    ]
    for fmt in iso_candidates:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            continue

    return None


def extract_element_markup(elem: Optional[ET.Element]) -> str:
    """
    Extracts complete inner content of an XML element including nested child tags and tail text.
    Correctly handles nested markup like <description><a href="..."><img src="..."/></a>Text...</description>.
    """
    if elem is None:
        return ""
    if len(elem) > 0:
        parts = []
        if elem.text:
            parts.append(elem.text)
        for child in elem:
            parts.append(ET.tostring(child, encoding="unicode"))
        return "".join(parts).strip()
    return (elem.text or "").strip()


def normalize_field_text(raw_text: Optional[str], fallback: str = "") -> str:
    """
    Normalizes article text fields (headline, description, publisher).
    Strips any HTML tags, decomposes script/style/figure tags, decodes all HTML entities,
    replaces unprintable replacement characters, collapses whitespace, and provides fallback.
    """
    if not raw_text:
        return fallback
    text = str(raw_text)
    if "<" in text:
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style", "figure", "figcaption"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
    text = html.unescape(text)
    text = text.replace("\ufffd", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else fallback


def is_valid_http_url(url: Optional[str]) -> bool:
    """
    Validates that a URL is a valid absolute HTTP or HTTPS web address.
    Strictly rejects javascript:, data:, file:, and relative links.
    """
    if not url or not isinstance(url, str):
        return False
    u = url.strip()
    if not (u.startswith("http://") or u.startswith("https://")):
        return False
    try:
        parsed = urllib.parse.urlparse(u)
        return bool(parsed.scheme in ("http", "https") and parsed.netloc)
    except Exception:
        return False


def clean_html_summary(html_text: Optional[str]) -> str:
    """Removes HTML markup, tracking beacons, decodes HTML entities, and collapses whitespace."""
    return normalize_field_text(html_text, fallback="")


def extract_image_from_item(item_elem: ET.Element, description_html: str = "") -> Optional[str]:
    """
    Extracts high-resolution article lead image URL from RSS XML item:
    Checks <enclosure>, <media:content>, <media:thumbnail>, or <img> in description.
    """
    # 1. Check <enclosure type="image/...">
    for enc in item_elem.findall("enclosure"):
        enc_type = enc.attrib.get("type", "").lower()
        enc_url = enc.attrib.get("url")
        if enc_url and ("image" in enc_type or enc_url.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))):
            return enc_url

    # 2. Check media namespace (<media:content>, <media:thumbnail>)
    ns = {
        "media": "http://search.yahoo.com/mrss/",
        "atom": "http://www.w3.org/2005/Atom"
    }
    for tag in ["media:content", "media:thumbnail", "{http://search.yahoo.com/mrss/}content", "{http://search.yahoo.com/mrss/}thumbnail"]:
        elem = item_elem.find(tag)
        if elem is not None and elem.attrib.get("url"):
            return elem.attrib["url"]

    # 3. Look for <img> src in raw description HTML
    if description_html and "<img" in description_html.lower():
        match = re.search(r'<img[^>]+src=["\'](https?://[^"\']+)["\']', description_html, re.I)
        if match:
            return match.group(1)

    return None


class NewsIngestionService:
    """
    Orchestrates continuous and on-demand news ingestion across configured RSS feeds and GDELT.
    Provides bounded retries with exponential backoff and independent feed error isolation.
    """

    def __init__(
        self,
        sources_path: Optional[Path] = None,
        timeout: int = 8,
        gdelt_enabled: bool = True,
        gdelt_timeout: int = 6,
        gdelt_retries: int = 2
    ):
        self.sources_path = sources_path or DEFAULT_SOURCES_FILE
        self.timeout = timeout
        self.gdelt_enabled = gdelt_enabled
        self.gdelt_timeout = gdelt_timeout
        self.gdelt_retries = gdelt_retries
        self.sources: List[Dict[str, Any]] = self._load_sources()

    def _load_sources(self) -> List[Dict[str, Any]]:
        """Loads feed definitions from JSON registry."""
        if not self.sources_path.exists():
            logger.warning(f"Sources registry not found at {self.sources_path}. Using empty registry.")
            return []
        try:
            with open(self.sources_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read sources config: {e}")
            return []

    def fetch_rss_feed(self, source: Dict[str, Any], max_records: int = 10) -> List[Dict[str, Any]]:
        """
        Fetches and parses a single RSS/Atom publisher feed.
        """
        url = source["url"]
        headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.8"
        }
        req = urllib.request.Request(url, headers=headers)
        articles: List[Dict[str, Any]] = []

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            content_bytes = resp.read()

        root = ET.fromstring(content_bytes)

        # Handle standard RSS <channel><item> and Atom <feed><entry>
        items = root.findall(".//item")
        if not items:
            items = root.findall(".//{http://www.w3.org/2005/Atom}entry")

        for item in items[:max_records]:
            # Title
            title_elem = item.find("title")
            if title_elem is None:
                title_elem = item.find("{http://www.w3.org/2005/Atom}title")
            raw_title = extract_element_markup(title_elem)
            headline = clean_html_summary(raw_title)
            if not headline:
                continue

            # Link
            link = ""
            link_elem = item.find("link")
            if link_elem is not None:
                link = link_elem.text.strip() if link_elem.text else link_elem.attrib.get("href", "")
            if not link:
                atom_link = item.find("{http://www.w3.org/2005/Atom}link")
                if atom_link is not None:
                    link = atom_link.attrib.get("href", "")
            if not link:
                continue

            # Description / Summary
            desc_elem = item.find("description")
            if desc_elem is None:
                desc_elem = item.find("{http://www.w3.org/2005/Atom}summary")
            if desc_elem is None:
                desc_elem = item.find("{http://www.w3.org/2005/Atom}content")
            raw_desc = extract_element_markup(desc_elem)
            clean_desc = clean_html_summary(raw_desc)

            # Date
            date_elem = item.find("pubDate")
            if date_elem is None:
                date_elem = item.find("{http://www.w3.org/2005/Atom}published")
            if date_elem is None:
                date_elem = item.find("{http://www.w3.org/2005/Atom}updated")
            raw_date = date_elem.text.strip() if date_elem is not None and date_elem.text else ""
            iso_date = parse_timestamp_to_iso(raw_date)

            # Lead Image
            lead_image = extract_image_from_item(item, description_html=raw_desc)

            canonical = canonicalize_url(link)
            content_hash = hashlib.sha256(f"{canonical}::{headline}".encode("utf-8")).hexdigest()[:16]

            article_record = {
                "article_id": f"art_{hashlib.md5(canonical.encode('utf-8')).hexdigest()[:12]}",
                "canonical_url": canonical,
                "original_url": link,
                "headline": headline,
                "description": clean_desc,
                "publisher": source["name"],
                "country": source.get("country", "Global"),
                "region": source.get("region", "International"),
                "language": source.get("language", "en"),
                "published_at": iso_date,
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "image_url": lead_image,
                "content_hash": content_hash,
                "source_feed": source["source_id"],
                "provenance": "LIVE_RSS",
                "processing_status": "collected"
            }
            articles.append(article_record)

        return articles

    def fetch_gdelt_feed(self, source: Dict[str, Any], max_records: int = 15) -> List[Dict[str, Any]]:
        """
        Queries GDELT DOC 2.0 API with bounded retry and exponential backoff.
        Normalizes discovered news records with provenance='LIVE_GDELT'.
        Never fabricates results on network failure.
        """
        if not self.gdelt_enabled:
            logger.info(f"GDELT ingestion is disabled by configuration. Skipping '{source['name']}'.")
            return []

        url = source["url"]
        headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/json",
            "Connection": "close"
        }
        articles: List[Dict[str, Any]] = []

        last_error = None
        data = None
        for attempt in range(self.gdelt_retries + 1):
            t_attempt = datetime.now(timezone.utc).isoformat()
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=self.gdelt_timeout) as resp:
                    content = resp.read().decode("utf-8")
                    if not content.strip():
                        raise ValueError("Empty response received from GDELT DOC 2.0 API")
                    data = json.loads(content)
                    break
            except Exception as e:
                last_error = e
                if attempt < self.gdelt_retries:
                    delay = (2 ** attempt) * 1.0  # 1.0s, 2.0s
                    logger.warning(
                        f"GDELT fetch attempt {attempt+1}/{self.gdelt_retries+1} failed at {t_attempt}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"GDELT DOC 2.0 request failed permanently after {self.gdelt_retries+1} attempts at {t_attempt}: {e}."
                    )
                    raise e

        if not data:
            return []

        raw_articles = data.get("articles", [])
        for item in raw_articles[:max_records]:
            link = item.get("url")
            headline = item.get("title", "").strip()
            if not link or not headline:
                continue

            canonical = canonicalize_url(link)
            domain = item.get("domain", source["name"])
            seendate = item.get("seendate")
            iso_date = parse_timestamp_to_iso(seendate)
            social_image = item.get("socialimage")
            content_hash = hashlib.sha256(f"{canonical}::{headline}".encode("utf-8")).hexdigest()[:16]

            article_record = {
                "article_id": f"art_{hashlib.md5(canonical.encode('utf-8')).hexdigest()[:12]}",
                "canonical_url": canonical,
                "original_url": link,
                "headline": headline,
                "description": f"Discovered via GDELT DOC 2.0 API from publisher domain '{domain}'.",
                "publisher": domain or source["name"],
                "country": source.get("country", "Global"),
                "region": source.get("region", "International"),
                "language": source.get("language", "en"),
                "published_at": iso_date,
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "image_url": social_image if social_image and social_image.startswith("http") else None,
                "content_hash": content_hash,
                "source_feed": source["source_id"],
                "provenance": "LIVE_GDELT",
                "processing_status": "collected"
            }
            articles.append(article_record)

        return articles

    def poll_all_sources(
        self,
        db: NewsIntelligenceDB,
        max_per_source: int = 10,
        request_pause_sec: float = 0.2
    ) -> Dict[str, Any]:
        """
        Iterates over all configured sources, fetches articles, and records metrics.
        Returns execution summary.
        """
        t0 = time.perf_counter()
        total_discovered = 0
        total_new_stored = 0
        total_duplicates = 0
        failed_feeds = 0
        feed_results: List[Dict[str, Any]] = []

        for source in self.sources:
            if not source.get("enabled", True):
                continue

            source_id = source["source_id"]
            feed_type = source.get("feed_type", "rss").lower()
            name = source["name"]
            country = source.get("country", "Global")
            region = source.get("region", "International")
            url = source["url"]

            if feed_type == "gdelt" and not self.gdelt_enabled:
                logger.info(f"GDELT source '{name}' disabled by configuration.")
                db.update_feed_status(
                    source_id=source_id,
                    name=name,
                    country=country,
                    region=region,
                    feed_type=feed_type,
                    url=url,
                    status="DISABLED",
                    error="GDELT disabled by operator configuration",
                    articles_added=0
                )
                feed_results.append({
                    "source_id": source_id,
                    "name": name,
                    "status": "DISABLED",
                    "discovered": 0,
                    "new_stored": 0,
                    "duplicates": 0
                })
                continue

            logger.info(f"Polling {feed_type.upper()} source: {name} ({country})...")
            try:
                if feed_type == "gdelt":
                    articles = self.fetch_gdelt_feed(source, max_records=max_per_source)
                else:
                    articles = self.fetch_rss_feed(source, max_records=max_per_source)

                source_new = 0
                source_dups = 0
                for a in articles:
                    is_new, _ = db.upsert_article(a)
                    if is_new:
                        source_new += 1
                    else:
                        source_dups += 1

                total_discovered += len(articles)
                total_new_stored += source_new
                total_duplicates += source_dups

                db.update_feed_status(
                    source_id=source_id,
                    name=name,
                    country=country,
                    region=region,
                    feed_type=feed_type,
                    url=url,
                    status="OK",
                    error=None,
                    articles_added=source_new
                )

                feed_results.append({
                    "source_id": source_id,
                    "name": name,
                    "status": "OK",
                    "discovered": len(articles),
                    "new_stored": source_new,
                    "duplicates": source_dups
                })

            except Exception as e:
                err_msg = str(e)
                fail_time = datetime.now(timezone.utc).isoformat()
                status_label = "UNAVAILABLE" if feed_type == "gdelt" else "FETCH_ERROR"
                logger.warning(f"Failed to poll {name} ({url}) at {fail_time}: {err_msg}")
                failed_feeds += 1
                db.update_feed_status(
                    source_id=source_id,
                    name=name,
                    country=country,
                    region=region,
                    feed_type=feed_type,
                    url=url,
                    status=status_label,
                    error=f"[{fail_time}] {err_msg[:170]}",
                    articles_added=0
                )
                feed_results.append({
                    "source_id": source_id,
                    "name": name,
                    "status": status_label,
                    "error": f"[{fail_time}] {err_msg[:170]}",
                    "failed_at": fail_time
                })

            # Polite pacing between feeds
            time.sleep(request_pause_sec)

        elapsed = time.perf_counter() - t0
        summary = {
            "sources_polled": len(self.sources),
            "failed_sources": failed_feeds,
            "total_discovered": total_discovered,
            "total_new_stored": total_new_stored,
            "total_duplicates": total_duplicates,
            "execution_time_sec": round(elapsed, 2),
            "feed_results": feed_results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        logger.info(
            f"Ingestion cycle completed: {total_new_stored} new articles stored, "
            f"{total_duplicates} duplicates skipped across {len(self.sources)} feeds in {elapsed:.2f}s."
        )
        return summary
