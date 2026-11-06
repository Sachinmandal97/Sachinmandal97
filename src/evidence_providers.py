"""
Modular live evidence acquisition provider interfaces and implementations.
Supports:
  - Google News RSS (real-time news coverage across politics, business, science, sports, disaster, etc.)
  - Wikipedia REST API (encyclopedic / factual verification across historical, scientific, institutional topics)
  - Extensible BaseSearchProvider abstraction for adding Tavily, Bing, SearXNG, or custom endpoints.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import hashlib
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

from src.article_fetcher import is_root_or_homepage_url
from src.schemas import EvidenceSnippet
from src.utils import setup_logger

logger = setup_logger("evidence_providers")

# Standard HTTP request headers
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def clean_html_text(raw_html: str) -> str:
    """Removes HTML markup and normalizes whitespace."""
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(separator=" ").strip()
    return re.sub(r"\s+", " ", text)


class BaseSearchProvider(ABC):
    """
    Abstract interface for evidence search providers.
    """

    @abstractmethod
    def search(self, query: str, top_k: int = 5, max_results: Optional[int] = None, **kwargs) -> List[EvidenceSnippet]:
        """Searches provider for snippets relevant to query."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider."""
        pass


class GoogleNewsRSSProvider(BaseSearchProvider):
    """
    Retrieves real-time topical news coverage via Google News RSS feed.
    Requires no paid API keys. Covers all international news domains:
    politics, science, business, sports, disasters, health, etc.
    """

    def __init__(self, timeout: int = 6):
        self.timeout = timeout

    @property
    def provider_name(self) -> str:
        return "GoogleNewsRSS"

    def search(self, query: str, top_k: int = 5, max_results: Optional[int] = None, **kwargs) -> List[EvidenceSnippet]:
        if max_results is not None:
            top_k = max_results
        if not query or not query.strip():
            return []

        clean_query = query.strip()
        encoded = urllib.parse.quote(clean_query)
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"

        req = urllib.request.Request(url, headers=DEFAULT_HEADERS)
        snippets: List[EvidenceSnippet] = []

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                xml_data = resp.read()

            root = ET.fromstring(xml_data)
            items = root.findall(".//item")
            retrieval_time = datetime.now(timezone.utc).isoformat()

            for idx, item in enumerate(items[:top_k]):
                raw_title = item.find("title").text if item.find("title") is not None else ""
                source_elem = item.find("source")
                source_name = source_elem.text if source_elem is not None else "News Wire"
                publisher_domain = source_elem.attrib.get("url") if source_elem is not None else None
                pub_date = item.find("pubDate").text if item.find("pubDate") is not None else None
                link = item.find("link").text if item.find("link") is not None else ""
                desc = item.find("description").text if item.find("description") is not None else ""

                cleaned_desc = clean_html_text(desc)
                if cleaned_desc and cleaned_desc.lower() != raw_title.lower():
                    if cleaned_desc.lower().startswith(raw_title.lower()):
                        extra = cleaned_desc[len(raw_title):].strip(" .-")
                        combined_text = f"{raw_title}. {extra}".strip(". ") if extra else raw_title
                    else:
                        combined_text = f"{raw_title}. {cleaned_desc}".strip()
                else:
                    combined_text = raw_title.strip()

                if not combined_text:
                    continue

                # Generate unique content-addressed snippet ID to prevent collisions across queries
                hash_basis = f"{clean_query}_{link or raw_title}_{idx}"
                unique_snip_id = f"live_{hashlib.md5(hash_basis.encode('utf-8')).hexdigest()[:10]}"

                is_wrapper = bool(link and "news.google.com" in link)
                snippet = EvidenceSnippet(
                    snippet_id=unique_snip_id,
                    text=combined_text,
                    source=f"{source_name} (via Google News)",
                    title=raw_title,
                    search_query=query,
                    score=None,
                    relevance_score=None,
                    is_relevant=False,
                    relevance_method="uncalculated",
                    relevance_reason="",
                    url=None if is_wrapper else (link if not is_root_or_homepage_url(link) else None),
                    wrapper_url=link if is_wrapper else None,
                    publisher_domain=publisher_domain,
                    publisher=source_name,
                    published_date=pub_date,
                    retrieval_timestamp=retrieval_time,
                    source_type="live_web",
                    content_type="snippet_only",
                    access_status="snippet_only",
                    metadata={"provider": "GoogleNewsRSS", "query": query, "publisher_domain": publisher_domain, "raw_link": link}
                )
                snippets.append(snippet)

            logger.info(f"GoogleNewsRSS returned {len(snippets)} snippets for query: '{query}'")
            return snippets

        except Exception as e:
            logger.warning(f"GoogleNewsRSS fetch failed for '{query}': {e}")
            return []


class WikipediaSearchProvider(BaseSearchProvider):
    """
    Retrieves encyclopedic, scientific, historical, and institutional evidence
    via the official Wikimedia REST API. Requires no API keys.
    """

    def __init__(self, timeout: int = 6):
        self.timeout = timeout
        self.headers = {"User-Agent": "MultimodalMisinfoVerifier/1.0 (academic; data science research)"}

    @property
    def provider_name(self) -> str:
        return "WikipediaAPI"

    def search(self, query: str, top_k: int = 3, max_results: Optional[int] = None, **kwargs) -> List[EvidenceSnippet]:
        if max_results is not None:
            top_k = max_results
        if not query or not query.strip():
            return []

        now = time.time()
        if getattr(self, "_cooldown_until", 0) > now:
            logger.debug("WikipediaAPI in cooldown following 429 rate limit; skipping.")
            return []

        clean_query = query.strip()
        encoded_q = urllib.parse.quote(clean_query)
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_q}&format=json&utf8=1"
        req = urllib.request.Request(search_url, headers=self.headers)

        snippets: List[EvidenceSnippet] = []
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            search_results = data.get("query", {}).get("search", [])
            retrieval_time = datetime.now(timezone.utc).isoformat()

            # Batch fetch extracts for all candidate titles in one HTTP request
            target_titles = [r.get("title", "") for r in search_results[:top_k] if r.get("title")]
            extracts_map = self._fetch_extracts(target_titles)

            for idx, res in enumerate(search_results[:top_k]):
                title = res.get("title", "")
                snippet_html = res.get("snippet", "")
                cleaned_snippet = clean_html_text(snippet_html)

                # Use batch-fetched extract or fallback to snippet
                extract_text = extracts_map.get(title)
                full_text = f"{title}: {extract_text}" if extract_text else f"{title}: {cleaned_snippet}"
                page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"

                # Generate unique content-addressed snippet ID
                hash_basis = f"{clean_query}_{title}_{idx}"
                unique_snip_id = f"wiki_{hashlib.md5(hash_basis.encode('utf-8')).hexdigest()[:10]}"

                snippet = EvidenceSnippet(
                    snippet_id=unique_snip_id,
                    text=full_text,
                    source=f"Wikipedia: {title}",
                    title=title,
                    search_query=query,
                    score=None,
                    relevance_score=None,
                    is_relevant=False,
                    relevance_method="uncalculated",
                    relevance_reason="",
                    url=page_url,
                    wrapper_url=None,
                    publisher="Wikimedia Foundation",
                    published_date=res.get("timestamp"),
                    retrieval_timestamp=retrieval_time,
                    source_type="wikipedia",
                    content_type="excerpt" if extract_text else "snippet_only",
                    access_status="accessible",
                    metadata={"provider": "WikipediaAPI", "page_title": title, "query": query}
                )
                snippets.append(snippet)

            logger.info(f"WikipediaAPI returned {len(snippets)} snippets for query: '{query}'")
            return snippets

        except Exception as e:
            if isinstance(e, urllib.error.HTTPError) and e.code == 429:
                self._cooldown_until = time.time() + 15
                logger.warning(f"WikipediaAPI 429 rate limit reached; pausing Wikipedia queries for 15s.")
            else:
                logger.warning(f"WikipediaAPI search failed for '{query}': {e}")
            return []

    def _fetch_extracts(self, titles: List[str]) -> Dict[str, str]:
        """Fetches plain-text intro extracts for multiple Wikipedia titles in a single batch request."""
        if not titles:
            return {}
        try:
            joined = "|".join(titles)
            encoded_titles = urllib.parse.quote(joined)
            url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1&titles={encoded_titles}&format=json&utf8=1"
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            extracts = {}
            for pid, pdata in pages.items():
                t = pdata.get("title", "")
                extract = pdata.get("extract", "")
                if extract:
                    sentences = extract.split(". ")
                    extracts[t] = ". ".join(sentences[:3]).strip() + "."
            return extracts
        except Exception:
            return {}

    def _fetch_extract(self, title: str) -> Optional[str]:
        """Fetches plain-text intro extract for a single title (compatibility wrapper)."""
        res = self._fetch_extracts([title])
        return res.get(title)
