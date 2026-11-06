"""
Article fetching, link resolution, and source content extraction module.
Handles:
  1. Safe URL redirect resolution.
  2. Distinction between Google News wrapper URLs and original publisher URLs.
  3. Polite HTML page fetching with timeouts and user-agent headers.
  4. Content extraction: title, publisher, publication date, and body text.
  5. Fallback labeling: transparently tags content as 'full_text', 'excerpt',
     'snippet_only', or 'headline_only', and access_status as 'accessible' or 'inaccessible'.
"""

from datetime import datetime, timezone
import logging
import re
from typing import Any, Dict, Optional, Tuple
import urllib.parse
import urllib.request
from bs4 import BeautifulSoup

from src.utils import setup_logger

logger = setup_logger("article_fetcher")

DEFAULT_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Cache for resolved URLs and fetched articles to prevent redundant network hits: {url: result_dict}
_ARTICLE_CACHE: Dict[str, Dict[str, Any]] = {}


def is_root_or_homepage_url(url: Optional[str]) -> bool:
    """
    Checks if a URL points only to a root domain or publisher homepage path,
    e.g., 'https://www.reuters.com', 'https://www.reuters.com/', 'http://bbc.co.uk/index.html'.
    Never treats a publisher homepage as a valid specific news article URL.
    """
    if not url or not isinstance(url, str) or not url.strip():
        return False
    try:
        parsed = urllib.parse.urlparse(url.strip())
        if not parsed.netloc:
            return False
        path = (parsed.path or "").strip().rstrip("/")
        # Paths typically representing homepages or section roots rather than distinct articles
        homepage_paths = {
            "",
            "/",
            "/index.html",
            "/index.htm",
            "/index.php",
            "/home",
            "/en",
            "/en/",
            "/en-us",
            "/default.aspx",
            "/main",
        }
        return path.lower() in homepage_paths and not parsed.query
    except Exception:
        return False


def resolve_publisher_url(url: str, timeout: int = 4) -> Tuple[Optional[str], str]:
    """
    Attempts to follow redirects and resolve wrapper URLs to the original publisher article URL.
    Returns: (resolved_url, status_note)
      - If resolved successfully to a specific article: (final_url, "resolved")
      - If resolved URL is just a publisher homepage: (None, "resolved_homepage_rejected")
      - If Google News wrapper returns 400/403 or blocks: (None, "google_wrapper_unresolved")
      - If connection fails: (None, "connection_error")
    """
    if not url:
        return None, "empty_url"

    if "news.google.com" not in url:
        if is_root_or_homepage_url(url):
            return None, "root_domain_rejected"
        return url, "direct_url"

    req = urllib.request.Request(url, headers=DEFAULT_BROWSER_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            final_url = resp.geturl()
            if "news.google.com" not in final_url:
                if is_root_or_homepage_url(final_url):
                    return None, "resolved_homepage_rejected"
                return final_url, "resolved"
            # In some cases Google News returns an interstitial HTML with canonical link
            html = resp.read(8192).decode("utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            canonical = soup.find("link", rel="canonical")
            if canonical and canonical.get("href") and "google.com" not in canonical["href"]:
                cand = canonical["href"].strip()
                if is_root_or_homepage_url(cand):
                    return None, "resolved_homepage_rejected"
                return cand, "resolved_canonical"
            return None, "google_wrapper_unresolved"
    except urllib.error.HTTPError as e:
        logger.debug(f"HTTPError {e.code} resolving Google News URL {url[:60]}")
        return None, "google_wrapper_unresolved"
    except Exception as e:
        logger.debug(f"Error resolving Google News URL {url[:60]}: {e}")
        return None, "google_wrapper_unresolved"


def fetch_article_content(
    url: str,
    timeout: int = 5,
    max_paragraphs: int = 8,
    max_words: int = 800
) -> Dict[str, Any]:
    """
    Fetches article content from a web URL if accessible and permitted.
    Safely extracts readable body text, title, publisher, and timestamp.
    Never fetches publisher homepage as a substitute for an unavailable article.
    Never crashes on timeouts, 403/429/404, or non-HTML responses.
    """
    if not url or not url.strip():
        return {
            "success": False,
            "text": "",
            "title": None,
            "publisher": None,
            "published_date": None,
            "content_type": "headline_only",
            "access_status": "inaccessible",
            "error": "empty_url"
        }

    if is_root_or_homepage_url(url):
        logger.warning(f"Refusing to fetch publisher homepage as article: {url}")
        return {
            "success": False,
            "text": "",
            "title": None,
            "publisher": None,
            "published_date": None,
            "content_type": "snippet_only",
            "access_status": "root_domain_not_article",
            "error": "root_domain_not_article"
        }

    # Check cache
    if url in _ARTICLE_CACHE:
        return _ARTICLE_CACHE[url]

    req = urllib.request.Request(url, headers=DEFAULT_BROWSER_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type_header = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type_header.lower():
                res = {
                    "success": False,
                    "text": "",
                    "title": None,
                    "publisher": None,
                    "published_date": None,
                    "content_type": "snippet_only",
                    "access_status": "inaccessible",
                    "error": f"non_html_content_type_{content_type_header}"
                }
                _ARTICLE_CACHE[url] = res
                return res

            html_data = resp.read().decode("utf-8", errors="ignore")
            soup = BeautifulSoup(html_data, "html.parser")

            # 1. Title
            title_elem = soup.find("title") or soup.find("h1")
            page_title = title_elem.get_text().strip() if title_elem else None

            # 2. Publisher
            meta_pub = soup.find("meta", property="og:site_name") or soup.find("meta", attrs={"name": "publisher"})
            publisher_name = meta_pub.get("content").strip() if meta_pub and meta_pub.get("content") else None

            # 3. Date
            meta_date = (
                soup.find("meta", property="article:published_time") or
                soup.find("meta", attrs={"name": "publication_date"}) or
                soup.find("meta", attrs={"name": "date"})
            )
            pub_date = meta_date.get("content").strip() if meta_date and meta_date.get("content") else None

            # 4. Extract paragraphs
            for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
                tag.decompose()

            paragraphs = []
            for p in soup.find_all("p"):
                txt = p.get_text().strip()
                # Filter out short disclaimers, cookie notices, and nav links
                if len(txt) > 40 and not re.search(r"cookie|terms of use|subscribe|all rights reserved", txt, re.I):
                    paragraphs.append(txt)

            if paragraphs:
                combined_text = " ".join(paragraphs[:max_paragraphs])
                words = combined_text.split()
                if len(words) > max_words:
                    combined_text = " ".join(words[:max_words]) + "..."
                    c_type = "excerpt"
                else:
                    c_type = "full_text" if len(words) >= 60 else "excerpt"

                res = {
                    "success": True,
                    "text": combined_text,
                    "title": page_title,
                    "publisher": publisher_name,
                    "published_date": pub_date,
                    "content_type": c_type,
                    "access_status": "accessible",
                    "error": None
                }
            else:
                res = {
                    "success": False,
                    "text": "",
                    "title": page_title,
                    "publisher": publisher_name,
                    "published_date": pub_date,
                    "content_type": "snippet_only",
                    "access_status": "snippet_only",
                    "error": "no_paragraphs_extracted"
                }

            _ARTICLE_CACHE[url] = res
            return res

    except urllib.error.HTTPError as e:
        logger.debug(f"HTTPError {e.code} fetching article from {url}")
        res = {
            "success": False,
            "text": "",
            "title": None,
            "publisher": None,
            "published_date": None,
            "content_type": "snippet_only",
            "access_status": "inaccessible",
            "error": f"http_error_{e.code}"
        }
        _ARTICLE_CACHE[url] = res
        return res

    except Exception as e:
        logger.debug(f"Failed to fetch article from {url}: {e}")
        res = {
            "success": False,
            "text": "",
            "title": None,
            "publisher": None,
            "published_date": None,
            "content_type": "snippet_only",
            "access_status": "inaccessible",
            "error": str(e)
        }
        _ARTICLE_CACHE[url] = res
        return res


def clear_article_cache():
    """Clears the in-memory article fetch cache."""
    global _ARTICLE_CACHE
    _ARTICLE_CACHE.clear()
