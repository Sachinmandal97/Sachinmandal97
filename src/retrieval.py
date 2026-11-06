"""
Universal evidence retrieval and ranking module.
Integrates:
  1. BM25Okapi lexical retrieval over local corpus
  2. Live Web News & Wikipedia evidence acquisition
  3. Strict topical & lexical relevance scoring and thresholding
  4. Multi-query aggregation and passage deduplication
  5. Caching with time-to-live (TTL) to avoid redundant network queries
"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from src.article_fetcher import fetch_article_content, is_root_or_homepage_url, resolve_publisher_url
from src.evidence_providers import BaseSearchProvider, GoogleNewsRSSProvider, WikipediaSearchProvider
from src.schemas import AtomicClaim, EvidenceSnippet
from src.utils import setup_logger

logger = setup_logger("retrieval")

# Standard English stopwords to filter when assessing topical overlap
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "arent", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "cant", "cannot", "could", "couldnt", "did", "didnt", "do", "does", "doesnt",
    "doing", "dont", "down", "during", "each", "few", "for", "from", "further", "had", "hadnt",
    "has", "hasnt", "have", "havent", "having", "he", "hed", "hell", "hes", "her", "here", "heres",
    "hers", "herself", "him", "himself", "his", "how", "hows", "i", "id", "ill", "im", "ive",
    "if", "in", "into", "is", "isnt", "it", "its", "itself", "lets", "me", "more", "most", "mustnt",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shant", "she", "shed", "shell", "shes",
    "should", "shouldnt", "so", "some", "such", "than", "that", "thats", "the", "their", "theirs",
    "them", "themselves", "then", "there", "theres", "these", "they", "theyd", "thell", "theyre",
    "theyve", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasnt",
    "we", "wed", "well", "were", "weve", "werent", "what", "whats", "when", "whens", "where", "wheres",
    "which", "while", "who", "whos", "whom", "why", "whys", "with", "wont", "would", "wouldnt", "you",
    "youd", "youll", "youre", "youve", "your", "yours", "yourself", "yourselves"
}

# Generic news terms that carry low discriminating weight
GENERIC_NEWS_TERMS = {
    "news", "report", "reported", "reporting", "says", "said", "saying", "according",
    "view", "views", "captured", "operation", "operations", "operate", "operating",
    "resumes", "resume", "resumed", "return", "returns", "returned", "returning",
    "state", "states", "stated", "official", "officials", "statement", "latest",
    "breaking", "update", "updates", "today", "yesterday", "morning", "sunday"
}


def tokenize_text(text: str) -> List[str]:
    """Simple alphanumeric lowercase tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


def extract_content_words(text: str) -> List[str]:
    """Extracts non-stopword alphanumeric content tokens."""
    words = re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower())
    return [w for w in words if w not in STOPWORDS]


def extract_named_entities_and_anchors(text: str) -> List[str]:
    """
    Extracts high-priority named entities, proper nouns, and distinctive phrases.
    Recognizes capitalized sequences, acronyms, and multi-word names.
    Works universally across politics, sports, science, conflict, and business.
    """
    if not text:
        return []
    entities = []
    # Acronyms (e.g. NASA, BBC, CNN, JPL, UK, US, WHO, ECB, EU)
    acronyms = re.findall(r"\b[A-Z]{2,}\b", text)
    entities.extend(acronyms)

    # Capitalized word sequences (e.g. 'Perseverance rover', 'Broom Point', 'Real Madrid')
    cap_phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    for p in cap_phrases:
        if p.lower() not in STOPWORDS and p not in entities and len(p) > 2:
            entities.append(p)
        if " " in p:
            for sub_p in p.split():
                if (
                    sub_p.lower() not in STOPWORDS
                    and sub_p not in entities
                    and len(sub_p) > 2
                    and sub_p.lower() not in GENERIC_NEWS_TERMS
                ):
                    entities.append(sub_p)

    # Dates (e.g. 'May 15, 2025', 'August 4')
    dates = re.findall(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,\s+\d{4})?\b",
        text,
        re.I
    )
    entities.extend(dates)

    return list(dict.fromkeys(entities))


def build_search_queries(claim_text: str, atomic_entities: Optional[List[str]] = None) -> List[str]:
    """
    Builds concise, prioritized search queries from the claim and its entities.
    Prioritizes named entities and distinctive terms to avoid generic noise.
    Does NOT hardcode any domain or story terms.
    """
    if not claim_text or not claim_text.strip():
        return []

    entities = atomic_entities or extract_named_entities_and_anchors(claim_text)
    content_words = extract_content_words(claim_text)

    queries = []
    # 1. Primary entity-targeted query (combines key entities)
    if entities:
        quoted_entities = [f'"{e}"' if " " in e else e for e in entities[:4]]
        q1 = " ".join(quoted_entities)
        queries.append(q1)

        # 2. Entity + top distinctive non-generic content words
        distinctive_words = [
            w for w in content_words
            if w.lower() not in [e.lower() for e in entities] and w.lower() not in GENERIC_NEWS_TERMS
        ]
        if distinctive_words:
            q2 = f"{' '.join(entities[:3])} {' '.join(distinctive_words[:3])}"
            queries.append(q2)

    # 3. Clean content words query (top 6 distinctive content words)
    filtered_words = [w for w in content_words if w.lower() not in GENERIC_NEWS_TERMS]
    q_words = filtered_words[:6] if filtered_words else content_words[:6]
    if q_words:
        queries.append(" ".join(q_words))

    # Fallback to cleaned claim
    clean_claim = re.sub(r"[^\w\s\-\"\']", " ", claim_text)
    clean_claim = re.sub(r"\s+", " ", clean_claim).strip()
    if clean_claim and clean_claim not in queries:
        queries.append(clean_claim)

    return list(dict.fromkeys([q for q in queries if q.strip()]))


ENTITY_SYNONYMS: Dict[str, Set[str]] = {
    "mars": {"martian"},
    "martian": {"mars"},
    "moon": {"lunar"},
    "lunar": {"moon"},
    "earth": {"terrestrial"},
    "sun": {"solar"},
    "britain": {"british", "uk"},
    "british": {"britain", "uk"},
    "scotland": {"scottish"},
    "scottish": {"scotland"},
    "america": {"american", "us", "usa"},
    "american": {"america", "us", "usa"},
    "france": {"french"},
    "french": {"france"},
    "germany": {"german"},
    "german": {"germany"},
    "china": {"chinese"},
    "chinese": {"china"},
    "japan": {"japanese"},
    "japanese": {"japan"},
    "russia": {"russian"},
    "russian": {"russia"},
    "london": {"londoner", "londoners"},
}


def evaluate_evidence_relevance(
    query_or_claim: str,
    passage: str,
    cutoff: float = 0.20,
    claim_entities: Optional[List[str]] = None
) -> Tuple[Optional[float], bool, str]:
    """
    Evaluates whether an evidence passage is truly topically relevant to a query/claim.
    Enforces:
      1. Specificity weighting (entities / technical terms weighted higher than generic news terms).
      2. Entity anchor check: If claim has specific entities, passages matching 0 entities are capped at 0.0.
      3. Honest fallback: Returns (None, False, 'Relevance unavailable') on failure/empty input.
    Returns: (score, is_relevant, reason)
    """
    if not query_or_claim or not query_or_claim.strip() or not passage or not passage.strip():
        return None, False, "Relevance unavailable: empty input"

    q_words = extract_content_words(query_or_claim)
    t_words = set(extract_content_words(passage))

    if not q_words or not t_words:
        return None, False, "Relevance unavailable: insufficient content words"

    entities = claim_entities if claim_entities is not None else extract_named_entities_and_anchors(query_or_claim)

    # Check entity anchor match
    matched_entities = []
    if entities:
        p_lower = passage.lower()
        for ent in entities:
            ent_lower = ent.lower()
            if ent_lower in p_lower:
                matched_entities.append(ent)
            elif " " in ent:
                parts = [p for p in ent_lower.split() if p not in STOPWORDS]
                if parts and all(part in p_lower for part in parts):
                    matched_entities.append(ent)
            elif ent_lower in ENTITY_SYNONYMS and any(syn in p_lower for syn in ENTITY_SYNONYMS[ent_lower]):
                matched_entities.append(ent)
            elif len(ent_lower) > 3 and (ent_lower.rstrip("s") in p_lower or f"{ent_lower}s" in p_lower):
                matched_entities.append(ent)

        # If the claim had strong discriminating entities and ZERO matched, the article is completely off-topic
        strong_entities = [e for e in entities if len(e) > 2 and e.lower() not in GENERIC_NEWS_TERMS]
        if strong_entities and len(matched_entities) == 0:
            return 0.0, False, f"Excluded: 0 matching entity anchors (none of {strong_entities[:3]} found in passage)"

    # Specificity weighting for content words
    weights = {}
    for w in q_words:
        w_lower = w.lower()
        if any(w_lower in ent.lower() or ent.lower() in w_lower for ent in entities):
            weights[w] = 3.0
        elif w_lower in GENERIC_NEWS_TERMS:
            weights[w] = 0.4
        else:
            weights[w] = 1.0

    total_q_weight = sum(weights.values())
    matched_q_weight = 0.0
    for w in q_words:
        w_lower = w.lower()
        if w in t_words:
            matched_q_weight += weights[w]
        elif any(ent.lower() == w_lower and ent in matched_entities for ent in entities):
            matched_q_weight += weights[w]

    weighted_coverage = matched_q_weight / max(1e-6, total_q_weight)

    # Word set Jaccard
    q_set = set(q_words)
    jaccard = len(q_set.intersection(t_words)) / max(1, len(q_set.union(t_words)))

    # Entity boost if entities were matched
    entity_boost = min(0.20, 0.10 * len(matched_entities)) if matched_entities else 0.0

    raw_score = 0.60 * weighted_coverage + 0.40 * jaccard + entity_boost
    final_score = round(min(1.0, max(0.0, raw_score)), 4)

    is_relevant = final_score >= cutoff
    if is_relevant:
        reason = f"Passed: score {final_score:.2f} >= cutoff {cutoff:.2f} (matched terms: {len(q_set.intersection(t_words))}, entities: {matched_entities})"
    else:
        reason = f"Excluded: score {final_score:.2f} < cutoff {cutoff:.2f} (insufficient topical overlap)"

    return final_score, is_relevant, reason


def compute_relevance_score(query: str, text: str) -> float:
    """
    Backward-compatible wrapper for evaluate_evidence_relevance.
    Returns float score in range [0.0, 1.0].
    """
    score, is_rel, reason = evaluate_evidence_relevance(query, text)
    return score if score is not None else 0.0


class SimpleBM25:
    """
    Self-contained BM25Okapi implementation.
    Requires no external C++ binaries or vector store dependencies.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_docs: List[Dict[str, Any]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = Counter()
        self.total_docs: int = 0
        self.idf: Dict[str, float] = {}

    def fit(self, documents: List[Dict[str, Any]]) -> None:
        """Indexes a list of documents: [{'doc_id': str, 'text': str, 'source': str, ...}]"""
        self.corpus_docs = documents
        self.total_docs = len(documents)
        if self.total_docs == 0:
            return

        self.doc_lengths = []
        self.doc_freqs = Counter()

        for doc in documents:
            tokens = tokenize_text(doc.get("text", ""))
            self.doc_lengths.append(len(tokens))
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1

        self.avg_doc_len = sum(self.doc_lengths) / self.total_docs if self.total_docs > 0 else 1.0

        # Compute Robertson-Spärck Jones IDF
        self.idf = {}
        for token, freq in self.doc_freqs.items():
            self.idf[token] = math.log(1.0 + (self.total_docs - freq + 0.5) / (freq + 0.5))

    def score(self, query_tokens: List[str]) -> List[float]:
        """Scores all indexed documents against query tokens."""
        scores = [0.0] * self.total_docs
        for token in query_tokens:
            if token not in self.idf:
                continue
            token_idf = self.idf[token]
            for doc_idx, doc in enumerate(self.corpus_docs):
                doc_tokens = tokenize_text(doc.get("text", ""))
                tf = doc_tokens.count(token)
                doc_len = self.doc_lengths[doc_idx]

                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[doc_idx] += token_idf * (numerator / (denominator + 1e-9))
        return scores


class EvidenceRetriever:
    """
    Universal evidence retriever managing:
      - Local BM25 corpus retrieval
      - Live search providers (Google News RSS, Wikipedia)
      - Entity-anchored topical relevance filtering
      - LRU memory caching with TTL
    """

    def __init__(
        self,
        corpus_path: Optional[Path] = None,
        enable_live_search: bool = True,
        relevance_threshold: float = 0.20,
        cache_ttl_seconds: int = 3600
    ):
        self.bm25 = SimpleBM25()
        if corpus_path is not None:
            self.corpus_path = corpus_path
        else:
            primary = Path("./data/corpus/evidence_corpus.jsonl")
            fallback = Path("./data/corpus/mock_evidence.jsonl")
            self.corpus_path = primary if primary.exists() else fallback

        self.enable_live_search = enable_live_search
        self.relevance_threshold = relevance_threshold
        self.cache_ttl = cache_ttl_seconds
        self.is_indexed = False

        # Live search providers
        self.providers: List[BaseSearchProvider] = [
            GoogleNewsRSSProvider(timeout=5),
            WikipediaSearchProvider(timeout=5)
        ]

        # Simple in-memory query cache: {query_hash: (timestamp, results)}
        self._query_cache: Dict[str, Tuple[float, List[EvidenceSnippet]]] = {}
        self._corpus_mtime: Optional[float] = None

        self._load_and_index_corpus()

    def clear_cache(self) -> None:
        """Clears memory query cache and article fetcher cache."""
        self._query_cache.clear()
        from src.article_fetcher import clear_article_cache
        clear_article_cache()
        logger.info("Retriever query cache and article fetcher cache cleared.")

    def _load_and_index_corpus() -> None:
        pass  # Loaded below

    def _load_and_index_corpus(self) -> None:
        """Loads corpus JSONL file and indexes documents."""
        if not self.corpus_path.exists():
            logger.warning(f"Corpus path {self.corpus_path} does not exist. Retriever initialized empty.")
            return

        docs = []
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))

        self.bm25.fit(docs)
        self.is_indexed = True
        self._corpus_mtime = self.corpus_path.stat().st_mtime
        logger.info(f"[OK] Indexed {len(docs)} documents from {self.corpus_path}")

    def check_and_reload_corpus(self) -> None:
        """Re-indexes if corpus file modification time changed on disk."""
        if self.corpus_path.exists():
            current_mtime = self.corpus_path.stat().st_mtime
            if self._corpus_mtime is None or current_mtime > self._corpus_mtime:
                logger.info(f"Detected update to corpus {self.corpus_path}; re-indexing.")
                self._load_and_index_corpus()

    def retrieve_local(self, query: str, top_k: int = 3) -> List[EvidenceSnippet]:
        """Retrieves top passages from the local corpus using BM25."""
        self.check_and_reload_corpus()
        if not self.is_indexed or not query.strip():
            return []

        tokens = tokenize_text(query)
        if not tokens:
            return []

        scores = self.bm25.score(tokens)
        scored_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: List[EvidenceSnippet] = []
        for idx in scored_indices[:top_k]:
            if scores[idx] <= 0:
                continue
            doc = self.bm25.corpus_docs[idx]
            rel_score, is_relevant, rel_reason = evaluate_evidence_relevance(
                query, doc.get("text", ""), cutoff=self.relevance_threshold
            )

            snippet = EvidenceSnippet(
                snippet_id=str(doc.get("doc_id", f"doc_{idx}")),
                text=doc.get("text", ""),
                source=doc.get("source", "Local Demonstration Corpus"),
                title=doc.get("title", doc.get("source")),
                search_query=query,
                score=round(float(scores[idx]), 4),
                relevance_score=rel_score,
                is_relevant=is_relevant,
                relevance_method="bm25_lexical",
                relevance_reason=rel_reason,
                url=doc.get("url"),
                wrapper_url=None,
                publisher=doc.get("publisher", doc.get("source")),
                published_date=doc.get("published_date"),
                retrieval_timestamp=datetime.now(timezone.utc).isoformat(),
                source_type="local_corpus",
                content_type="full_text",
                access_status="accessible",
                metadata=doc
            )
            results.append(snippet)

        return results

    def retrieve_live(self, query: str, top_k: int = 3) -> List[EvidenceSnippet]:
        """Queries live providers for news articles and encyclopedic entries with targeted query reformulation."""
        if not self.enable_live_search or not query.strip():
            return []

        queries_to_try = build_search_queries(query)
        logger.info(f"Targeted search queries for '{query[:60]}': {queries_to_try}")

        live_results: List[EvidenceSnippet] = []
        for provider in self.providers:
            try:
                snippets_for_provider: List[EvidenceSnippet] = []
                # Google News RSS is the primary real-time news searcher (tries prioritized queries).
                # Wikipedia is an encyclopedic entity searcher; only query top entity query to prevent 429 rate limits.
                provider_queries = queries_to_try if provider.provider_name != "WikipediaAPI" else queries_to_try[:1]
                for q_attempt in provider_queries:
                    logger.info(f"Executing search with {provider.provider_name} for query: '{q_attempt}'")
                    found = provider.search(query=q_attempt, top_k=top_k)
                    if found:
                        snippets_for_provider.extend(found)
                        break

                for s in snippets_for_provider:
                    # Evaluate relevance against the original assertion/query
                    score, is_rel, reason = evaluate_evidence_relevance(
                        query, s.text, cutoff=self.relevance_threshold
                    )
                    s.relevance_score = score
                    s.is_relevant = is_rel
                    s.relevance_reason = reason
                    s.relevance_method = "entity_anchored_lexical"

                    # Attempt safe resolution and article enrichment if snippet is relevant
                    if s.is_relevant:
                        s = self._enrich_snippet_with_article(s, query=query)

                    live_results.append(s)
            except Exception as e:
                logger.warning(f"Error querying provider {provider.provider_name}: {e}")

        # Sort by relevance score
        return sorted(
            live_results,
            key=lambda s: (1 if s.is_relevant else 0, s.relevance_score if s.relevance_score is not None else -1.0),
            reverse=True
        )

    def _enrich_snippet_with_article(self, s: EvidenceSnippet, query: str = "") -> EvidenceSnippet:
        """
        Safely attempts to resolve wrapper URLs and enrich snippet with full article content.
        Guarantees:
        - Rejects publisher homepages / root domains.
        - Preserves snippet text intact on resolution or fetch failure.
        - Accurately updates access_status and content_type.
        """
        # 1. Attempt safe resolution of Google News wrapper
        if s.wrapper_url and not s.url:
            resolved_url, res_status = resolve_publisher_url(s.wrapper_url, timeout=3)
            if resolved_url and not is_root_or_homepage_url(resolved_url):
                s.url = resolved_url
                s.access_status = "resolved"
            else:
                s.access_status = "google_wrapper_unresolved"

        # 2. Attempt fetching specific article content only if a verified non-homepage URL exists
        if s.url and not is_root_or_homepage_url(s.url) and "news.google.com" not in s.url:
            article_data = fetch_article_content(s.url, timeout=4)
            if article_data.get("success") and article_data.get("text"):
                s.text = article_data["text"]
                s.content_type = article_data["content_type"]
                s.access_status = article_data["access_status"]
                # Re-score against richer text if query provided
                if query:
                    score_full, is_rel_full, reason_full = evaluate_evidence_relevance(
                        query, s.text, cutoff=self.relevance_threshold
                    )
                    s.relevance_score = score_full
                    s.is_relevant = is_rel_full
                    s.relevance_reason = reason_full
            else:
                # Preserve original snippet text; do NOT overwrite or fabricate
                if s.access_status != "google_wrapper_unresolved":
                    s.access_status = article_data.get("access_status", "inaccessible")

        return s

    def _hash_query(self, query: str, top_k: int = 3, force_mode: Optional[str] = None) -> str:
        """Computes deterministic cache key for query including relevance threshold."""
        clean = query.strip().lower()
        thresh = round(float(self.relevance_threshold), 2)
        return hashlib.md5(f"{clean}_{top_k}_{force_mode}_{thresh}".encode("utf-8")).hexdigest()

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        force_mode: Optional[str] = None  # 'local_only', 'live_only', 'hybrid'
    ) -> List[EvidenceSnippet]:
        """
        Universal retrieval method.
        Checks cache, retrieves from local corpus and/or live web, filters by relevance.
        """
        if not query or not query.strip():
            return []

        clean_query = query.strip()
        cache_key = self._hash_query(clean_query, top_k=top_k, force_mode=force_mode)

        # Cache check
        now = time.time()
        if cache_key in self._query_cache:
            ts, cached_res = self._query_cache[cache_key]
            if (now - ts) < self.cache_ttl:
                return cached_res

        all_candidates: List[EvidenceSnippet] = []

        # 1. Fetch live results if enabled
        if force_mode != "local_only" and self.enable_live_search:
            live_hits = self.retrieve_live(clean_query, top_k=top_k)
            all_candidates.extend(live_hits)

        # 2. Fetch local corpus results
        if force_mode != "live_only":
            local_hits = self.retrieve_local(clean_query, top_k=top_k)
            all_candidates.extend(local_hits)

        # 3. Deduplicate by text similarity or ID
        seen_texts = set()
        deduped: List[EvidenceSnippet] = []
        for cand in all_candidates:
            sig = cand.text[:40].lower()
            if sig not in seen_texts:
                seen_texts.add(sig)
                deduped.append(cand)

        # 4. Rank candidates:
        ranked = sorted(
            deduped,
            key=lambda s: (
                1 if s.is_relevant else 0,
                s.relevance_score if s.relevance_score is not None else -1.0,
                s.score if s.score is not None else 0.0
            ),
            reverse=True
        )[:top_k]

        # Store in cache
        self._query_cache[cache_key] = (now, ranked)
        return ranked

    def retrieve_for_claims(
        self,
        claims: List[AtomicClaim],
        top_k: int = 3,
        force_mode: Optional[str] = None
    ) -> List[EvidenceSnippet]:
        """
        Retrieves and deduplicates top relevant evidence across atomic claims.
        """
        all_snippets: Dict[str, EvidenceSnippet] = {}
        for claim in claims:
            snippets = self.retrieve(claim.text, top_k=top_k, force_mode=force_mode)
            for s in snippets:
                s_rel = s.relevance_score if s.relevance_score is not None else 0.0
                if s.snippet_id not in all_snippets:
                    all_snippets[s.snippet_id] = s
                else:
                    curr_rel = all_snippets[s.snippet_id].relevance_score or 0.0
                    if s_rel > curr_rel:
                        all_snippets[s.snippet_id] = s

        return sorted(
            all_snippets.values(),
            key=lambda s: (1 if s.is_relevant else 0, s.relevance_score if s.relevance_score is not None else -1.0),
            reverse=True
        )[:top_k]
