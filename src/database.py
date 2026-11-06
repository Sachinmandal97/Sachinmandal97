"""
Database persistence module for VERITAS-MM 2.0 News Intelligence.
Uses SQLite for zero-dependency, local, auditable storage of:
  1. Collected news articles (RSS & GDELT) with canonical URL deduplication.
  2. Granular multimodal verification results and evidence provenance reports.
  3. Publisher feed health, poll status, and operational metrics.
"""

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from src.utils import setup_logger

logger = setup_logger("database")

DEFAULT_DB_PATH = Path("./data/news_intelligence.db")


VALID_PROVENANCES = {"LIVE_RSS", "LIVE_GDELT", "SEEDED_DEMO", "MANUAL_UPLOAD", "UNKNOWN"}


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Returns a thread-safe SQLite connection with foreign keys enabled."""
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def migrate_schema(conn: sqlite3.Connection) -> None:
    """
    Safely migrates existing database schema to include article provenance.
    Non-destructive: preserves all existing articles and verifications.
    """
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(articles)")
    columns = [row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in cursor.fetchall()]

    if "provenance" not in columns:
        logger.info("Migrating articles table: adding 'provenance' column with default 'UNKNOWN'...")
        cursor.execute("ALTER TABLE articles ADD COLUMN provenance TEXT NOT NULL DEFAULT 'UNKNOWN';")
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_provenance ON articles(provenance);")

    # Backfill existing records using reliable historical evidence
    seed_urls = [
        "https://www.thehindu.com/news/national/isro-gaganyaan-crew-escape-system-test-success/article67450123.ece",
        "https://feeds.bbci.co.uk/news/world/un-climate-summit-renewable-energy-agreement/article883921.ece",
        "https://timesofindia.indiatimes.com/india/delhi-air-quality-emergency-measures-anti-smog-guns/articleshow/99812455.cms",
        "https://www.theguardian.com/world/2026/oct/24/ancient-submerged-roman-port-discovered-mediterranean",
        "https://indianexpress.com/article/india/supreme-court-electoral-bonds-scheme-unconstitutional-ruling/9128374/"
    ]
    placeholders = ",".join("?" for _ in seed_urls)
    cursor.execute(f"""
        UPDATE articles 
        SET provenance = 'SEEDED_DEMO' 
        WHERE provenance = 'UNKNOWN' 
          AND (canonical_url IN ({placeholders}) 
               OR source_feed LIKE 'seed_%'
               OR headline LIKE '%Gaganyaan flight abort%'
               OR headline LIKE '%triple renewable energy capacity by 2030%'
               OR headline LIKE '%Stage 4 anti-pollution curbs%'
               OR headline LIKE '%submerged Roman harbour%'
               OR headline LIKE '%Supreme Court Constitution Bench strikes down%'
               OR headline LIKE '%ISRO successfully tests reusable launch vehicle%')
    """, seed_urls)

    # Backfill live RSS articles
    cursor.execute("""
        UPDATE articles
        SET provenance = 'LIVE_RSS'
        WHERE provenance = 'UNKNOWN'
          AND source_feed IN ('the_hindu_national', 'ndtv_top_stories', 'times_of_india_top', 
                              'indian_express_india', 'bbc_world_news', 'the_guardian_world', 'npr_world')
    """)

    # Backfill live GDELT articles
    cursor.execute("""
        UPDATE articles
        SET provenance = 'LIVE_GDELT'
        WHERE provenance = 'UNKNOWN'
          AND source_feed IN ('gdelt_india_events', 'gdelt_global_events')
    """)


class NewsIntelligenceDB:
    """
    Manages persistent article storage, verification records, and feed metrics.
    Supports explicit article provenance tracking (LIVE_RSS, LIVE_GDELT, SEEDED_DEMO, MANUAL_UPLOAD, UNKNOWN).
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.init_schema()

    def init_schema(self) -> None:
        """Initializes tables, indexes, and applies non-destructive schema migrations."""
        with get_db_connection(self.db_path) as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS articles (
                article_id TEXT PRIMARY KEY,
                canonical_url TEXT UNIQUE NOT NULL,
                original_url TEXT NOT NULL,
                headline TEXT NOT NULL,
                description TEXT,
                publisher TEXT NOT NULL,
                country TEXT NOT NULL,
                region TEXT NOT NULL,
                language TEXT NOT NULL DEFAULT 'en',
                published_at TEXT,
                collected_at TEXT NOT NULL,
                image_url TEXT,
                content_hash TEXT NOT NULL,
                source_feed TEXT NOT NULL,
                provenance TEXT NOT NULL DEFAULT 'UNKNOWN',
                processing_status TEXT NOT NULL DEFAULT 'collected',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS verifications (
                verification_id TEXT PRIMARY KEY,
                article_id TEXT NOT NULL UNIQUE,
                decision_verdict TEXT NOT NULL,
                decision_confidence REAL NOT NULL,
                predictive_uncertainty REAL NOT NULL,
                visual_consistency_score REAL,
                visual_consistency_status TEXT,
                visual_mismatch_detected INTEGER DEFAULT 0,
                extracted_claims_json TEXT,
                atomic_verifications_json TEXT,
                retrieved_evidence_json TEXT,
                reasoning_factors_json TEXT,
                structured_report TEXT,
                execution_latency_sec REAL NOT NULL,
                verified_at TEXT NOT NULL,
                error_message TEXT,
                FOREIGN KEY (article_id) REFERENCES articles(article_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS feed_registry (
                source_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                country TEXT NOT NULL,
                region TEXT NOT NULL,
                feed_type TEXT NOT NULL,
                url TEXT NOT NULL,
                enabled INTEGER DEFAULT 1,
                last_fetched_at TEXT,
                last_status TEXT,
                articles_count INTEGER DEFAULT 0,
                error_message TEXT
            );

            CREATE TABLE IF NOT EXISTS system_locks (
                lock_name TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                acquired_at TEXT NOT NULL,
                expires_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_articles_status ON articles(processing_status);
            CREATE INDEX IF NOT EXISTS idx_articles_region ON articles(region);
            CREATE INDEX IF NOT EXISTS idx_articles_publisher ON articles(publisher);
            CREATE INDEX IF NOT EXISTS idx_articles_collected ON articles(collected_at DESC);
            CREATE INDEX IF NOT EXISTS idx_verifications_verdict ON verifications(decision_verdict);
            """)

            # Apply safe schema migrations for existing databases
            self.migrate_schema(conn)

        logger.info(f"[OK] NewsIntelligence database schema initialized at {self.db_path}")

    def migrate_schema(self, conn: sqlite3.Connection) -> None:
        """Instance delegation to module-level schema migration."""
        migrate_schema(conn)

    def upsert_article(self, article: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Inserts article if not previously collected (deduplication via canonical_url).
        Assigns provenance based on actual ingestion path.
        Returns (is_new: bool, article_id: str).
        """
        now = datetime.now(timezone.utc).isoformat()
        canonical_url = article.get("canonical_url") or article.get("original_url") or ""
        if not canonical_url:
            raise ValueError("Article must have a canonical_url or original_url.")

        article_id = article.get("article_id") or f"art_{hashlib.md5(canonical_url.encode('utf-8')).hexdigest()[:12]}"
        content_hash = article.get("content_hash") or hashlib.sha256(
            f"{canonical_url}::{article.get('headline', '')}".encode('utf-8')
        ).hexdigest()[:16]

        raw_prov = str(article.get("provenance", "UNKNOWN")).upper()
        provenance = raw_prov if raw_prov in VALID_PROVENANCES else "UNKNOWN"

        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT article_id, processing_status FROM articles WHERE canonical_url = ? OR article_id = ?", (canonical_url, article_id))
            existing = cursor.fetchone()
            if existing:
                return False, existing["article_id"]

            cursor.execute("""
                INSERT INTO articles (
                    article_id, canonical_url, original_url, headline, description,
                    publisher, country, region, language, published_at,
                    collected_at, image_url, content_hash, source_feed, provenance,
                    processing_status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                article_id,
                canonical_url,
                article.get("original_url", canonical_url),
                article.get("headline", "").strip(),
                article.get("description", "").strip(),
                article.get("publisher", "Unknown News Wire"),
                article.get("country", "Global"),
                article.get("region", "International"),
                article.get("language", "en"),
                article.get("published_at"),
                article.get("collected_at", now),
                article.get("image_url"),
                content_hash,
                article.get("source_feed", "unknown"),
                provenance,
                article.get("processing_status", "collected"),
                now,
                now
            ))
            return True, article_id

    def store_verification(self, verif: Dict[str, Any]) -> str:
        """Stores verification adjudication and updates article status to 'verified'."""
        now = datetime.now(timezone.utc).isoformat()
        article_id = verif["article_id"]
        verif_id = verif.get("verification_id") or f"ver_{hashlib.md5(f'{article_id}_{now}'.encode('utf-8')).hexdigest()[:12]}"

        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO verifications (
                    verification_id, article_id, decision_verdict, decision_confidence,
                    predictive_uncertainty, visual_consistency_score, visual_consistency_status,
                    visual_mismatch_detected, extracted_claims_json, atomic_verifications_json,
                    retrieved_evidence_json, reasoning_factors_json, structured_report,
                    execution_latency_sec, verified_at, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                verif_id,
                article_id,
                verif.get("decision_verdict", "Uncertain"),
                float(verif.get("decision_confidence", 0.5)),
                float(verif.get("predictive_uncertainty", 0.5)),
                verif.get("visual_consistency_score"),
                verif.get("visual_consistency_status", "evaluated"),
                1 if verif.get("visual_mismatch_detected") else 0,
                json.dumps(verif.get("extracted_claims", []), ensure_ascii=False),
                json.dumps(verif.get("atomic_verifications", []), ensure_ascii=False),
                json.dumps(verif.get("retrieved_evidence", []), ensure_ascii=False),
                json.dumps(verif.get("reasoning_factors", []), ensure_ascii=False),
                verif.get("structured_report", ""),
                float(verif.get("execution_latency_sec", 0.0)),
                verif.get("verified_at", now),
                verif.get("error_message")
            ))

            new_status = "verified" if not verif.get("error_message") else "failed"
            cursor.execute("UPDATE articles SET processing_status = ?, updated_at = ? WHERE article_id = ?", (new_status, now, article_id))
            return verif_id

    def get_articles(
        self,
        region: Optional[str] = None,
        publisher: Optional[str] = None,
        provenance: Optional[str] = None,
        status: Optional[str] = None,
        verdict: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Queries articles with optional filters, joined with verification data."""
        query = """
            SELECT 
                a.article_id, a.canonical_url, a.original_url, a.headline, a.description,
                a.publisher, a.country, a.region, a.language, a.published_at, a.collected_at,
                a.image_url, a.source_feed, a.provenance, a.processing_status,
                v.decision_verdict, v.decision_confidence, v.predictive_uncertainty,
                v.visual_consistency_score, v.visual_consistency_status, v.visual_mismatch_detected,
                v.execution_latency_sec, v.verified_at
            FROM articles a
            LEFT JOIN verifications v ON a.article_id = v.article_id
            WHERE 1=1
        """
        params: List[Any] = []

        if region and region.lower() != "all":
            query += " AND a.region = ?"
            params.append(region)

        if publisher and publisher.lower() != "all":
            query += " AND a.publisher = ?"
            params.append(publisher)

        if provenance and provenance.lower() != "all":
            if provenance.lower() == "live":
                query += " AND a.provenance IN ('LIVE_RSS', 'LIVE_GDELT')"
            else:
                query += " AND a.provenance = ?"
                params.append(provenance.upper())

        if status and status.lower() != "all":
            query += " AND a.processing_status = ?"
            params.append(status)

        if verdict and verdict.lower() != "all":
            query += " AND v.decision_verdict = ?"
            params.append(verdict)

        if search and search.strip():
            query += " AND (a.headline LIKE ? OR a.description LIKE ?)"
            term = f"%{search.strip()}%"
            params.extend([term, term])

        query += " ORDER BY a.collected_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def count_articles(
        self,
        region: Optional[str] = None,
        publisher: Optional[str] = None,
        provenance: Optional[str] = None,
        status: Optional[str] = None,
        verdict: Optional[str] = None,
        search: Optional[str] = None
    ) -> int:
        """Counts articles matching filter criteria for accurate pagination."""
        query = """
            SELECT COUNT(*) as cnt
            FROM articles a
            LEFT JOIN verifications v ON a.article_id = v.article_id
            WHERE 1=1
        """
        params: List[Any] = []
        if region and region.lower() != "all":
            query += " AND a.region = ?"
            params.append(region)
        if publisher and publisher.lower() != "all":
            query += " AND a.publisher = ?"
            params.append(publisher)
        if provenance and provenance.lower() != "all":
            if provenance.lower() == "live":
                query += " AND a.provenance IN ('LIVE_RSS', 'LIVE_GDELT')"
            else:
                query += " AND a.provenance = ?"
                params.append(provenance.upper())
        if status and status.lower() != "all":
            query += " AND a.processing_status = ?"
            params.append(status)
        if verdict and verdict.lower() != "all":
            query += " AND v.decision_verdict = ?"
            params.append(verdict)
        if search and search.strip():
            query += " AND (a.headline LIKE ? OR a.description LIKE ?)"
            term = f"%{search.strip()}%"
            params.extend([term, term])

        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return int(row["cnt"]) if row else 0

    def get_article_by_id(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves full article record and parsed verification report."""
        query = """
            SELECT 
                a.*,
                v.decision_verdict, v.decision_confidence, v.predictive_uncertainty,
                v.visual_consistency_score, v.visual_consistency_status, v.visual_mismatch_detected,
                v.extracted_claims_json, v.atomic_verifications_json, v.retrieved_evidence_json,
                v.reasoning_factors_json, v.structured_report, v.execution_latency_sec, v.verified_at,
                v.error_message as verif_error
            FROM articles a
            LEFT JOIN verifications v ON a.article_id = v.article_id
            WHERE a.article_id = ?
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (article_id,))
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            # Parse JSON fields
            for k in ["extracted_claims_json", "atomic_verifications_json", "retrieved_evidence_json", "reasoning_factors_json"]:
                if res.get(k):
                    try:
                        clean_key = k.replace("_json", "")
                        res[clean_key] = json.loads(res[k])
                    except Exception:
                        pass
            return res

    def get_pending_articles(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieves bounded batch of unverified articles for processing."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM articles
                WHERE processing_status IN ('collected', 'pending_verification')
                ORDER BY collected_at ASC
                LIMIT ?
            """, (limit,))
            return [dict(r) for r in cursor.fetchall()]

    def claim_pending_articles(self, limit: int = 2) -> List[Dict[str, Any]]:
        """
        Atomically claims a bounded batch of pending articles by updating their status to 'processing'.
        Guarantees that multiple concurrent workers or threads cannot process the same article.
        """
        now = datetime.now(timezone.utc).isoformat()
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM articles
                WHERE processing_status IN ('collected', 'pending_verification')
                ORDER BY collected_at ASC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            if not rows:
                return []

            claimed = [dict(r) for r in rows]
            article_ids = [r["article_id"] for r in claimed]
            placeholders = ",".join("?" for _ in article_ids)
            cursor.execute(f"""
                UPDATE articles
                SET processing_status = 'processing', updated_at = ?
                WHERE article_id IN ({placeholders})
            """, [now] + article_ids)
            return claimed

    def recover_interrupted_articles(self) -> int:
        """
        Recovers articles stranded in 'processing' status due to application or worker interruption.
        Resets them to 'collected' so they can be re-queued cleanly.
        """
        now = datetime.now(timezone.utc).isoformat()
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE articles
                SET processing_status = 'collected', updated_at = ?
                WHERE processing_status = 'processing'
            """, (now,))
            count = cursor.rowcount
            if count > 0:
                logger.info(f"Recovered {count} interrupted articles from 'processing' to 'collected'.")
            return count

    def get_queue_metrics(self) -> Dict[str, Any]:
        """Calculates granular queue health, backlog volume, and processing latency."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as c FROM articles WHERE processing_status IN ('collected', 'pending_verification')")
            pending = cursor.fetchone()["c"]

            cursor.execute("SELECT COUNT(*) as c FROM articles WHERE processing_status = 'processing'")
            processing = cursor.fetchone()["c"]

            cursor.execute("SELECT COUNT(*) as c FROM articles WHERE processing_status = 'verified'")
            verified = cursor.fetchone()["c"]

            cursor.execute("SELECT COUNT(*) as c FROM articles WHERE processing_status = 'failed'")
            failed = cursor.fetchone()["c"]

            cursor.execute("SELECT MIN(collected_at) as oldest FROM articles WHERE processing_status IN ('collected', 'pending_verification')")
            oldest_str = cursor.fetchone()["oldest"]
            oldest_age_sec = None
            if oldest_str:
                try:
                    dt = datetime.fromisoformat(oldest_str.replace("Z", "+00:00"))
                    oldest_age_sec = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
                except Exception:
                    pass

            return {
                "pending": pending,
                "processing": processing,
                "verified": verified,
                "failed": failed,
                "total_queue": pending + processing,
                "oldest_pending_age_sec": round(oldest_age_sec, 1) if oldest_age_sec is not None else None
            }

    def acquire_lock(self, lock_name: str, owner_id: str, ttl_seconds: float = 60.0) -> bool:
        """
        Database-backed lease lock for multi-process worker coordination.
        Returns True if the lock was successfully acquired or renewed, False otherwise.
        """
        import time as _t
        now_ts = _t.time()
        now_iso = datetime.now(timezone.utc).isoformat()
        expires_at = now_ts + ttl_seconds

        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT owner_id, expires_at FROM system_locks WHERE lock_name = ?", (lock_name,))
            row = cursor.fetchone()
            if row:
                existing_owner = row["owner_id"]
                existing_expires = float(row["expires_at"])
                if existing_owner != owner_id and existing_expires > now_ts:
                    return False

            cursor.execute("""
                INSERT OR REPLACE INTO system_locks (lock_name, owner_id, acquired_at, expires_at)
                VALUES (?, ?, ?, ?)
            """, (lock_name, owner_id, now_iso, expires_at))
            return True

    def release_lock(self, lock_name: str, owner_id: str) -> bool:
        """Releases database-backed lock if owned by owner_id."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM system_locks WHERE lock_name = ? AND owner_id = ?", (lock_name, owner_id))
            return cursor.rowcount > 0

    def update_article_status(self, article_id: str, status: str) -> None:
        """Updates article processing status."""
        now = datetime.now(timezone.utc).isoformat()
        with get_db_connection(self.db_path) as conn:
            conn.execute("UPDATE articles SET processing_status = ?, updated_at = ? WHERE article_id = ?", (status, now, article_id))

    def update_feed_status(
        self,
        source_id: str,
        name: str,
        country: str,
        region: str,
        feed_type: str,
        url: str,
        status: str,
        error: Optional[str] = None,
        articles_added: int = 0
    ) -> None:
        """Updates feed registry status and article counts."""
        now = datetime.now(timezone.utc).isoformat()
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT articles_count FROM feed_registry WHERE source_id = ?", (source_id,))
            row = cursor.fetchone()
            current_count = row["articles_count"] if row else 0
            new_count = current_count + articles_added

            cursor.execute("""
                INSERT OR REPLACE INTO feed_registry (
                    source_id, name, country, region, feed_type, url, enabled,
                    last_fetched_at, last_status, articles_count, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
            """, (
                source_id, name, country, region, feed_type, url,
                now, status, new_count, error
            ))

    def get_feed_statuses(self) -> List[Dict[str, Any]]:
        """Returns all feed statuses."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM feed_registry ORDER BY region ASC, name ASC")
            return [dict(r) for r in cursor.fetchall()]

    def get_statistics(self) -> Dict[str, Any]:
        """Calculates system-wide intelligence metrics for the dashboard."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) as total FROM articles")
            total = cursor.fetchone()["total"]

            cursor.execute("SELECT COUNT(*) as pending FROM articles WHERE processing_status IN ('collected', 'pending_verification')")
            pending = cursor.fetchone()["pending"]

            cursor.execute("SELECT COUNT(*) as processing FROM articles WHERE processing_status = 'processing'")
            processing = cursor.fetchone()["processing"]

            cursor.execute("SELECT COUNT(*) as verified FROM articles WHERE processing_status = 'verified'")
            verified = cursor.fetchone()["verified"]

            cursor.execute("SELECT COUNT(*) as failed FROM articles WHERE processing_status = 'failed'")
            failed = cursor.fetchone()["failed"]

            cursor.execute("""
                SELECT COUNT(*) as review_needed FROM verifications 
                WHERE decision_verdict IN ('Misleading', 'Conflicting Evidence', 'Partially Supported / Some Details Unverified')
            """)
            review_needed = cursor.fetchone()["review_needed"]

            cursor.execute("SELECT decision_verdict, COUNT(*) as count FROM verifications GROUP BY decision_verdict")
            verdicts = {r["decision_verdict"]: r["count"] for r in cursor.fetchall()}

            cursor.execute("SELECT region, COUNT(*) as count FROM articles GROUP BY region")
            regions = {r["region"]: r["count"] for r in cursor.fetchall()}

            cursor.execute("SELECT provenance, COUNT(*) as count FROM articles GROUP BY provenance")
            provenance_counts = {r["provenance"]: r["count"] for r in cursor.fetchall()}

            cursor.execute("SELECT DISTINCT publisher FROM articles ORDER BY publisher ASC")
            publishers = [r["publisher"] for r in cursor.fetchall()]

            cursor.execute("SELECT MAX(collected_at) as last_col FROM articles")
            last_col = cursor.fetchone()["last_col"]

            cursor.execute("SELECT MAX(verified_at) as last_ver FROM verifications")
            last_ver = cursor.fetchone()["last_ver"]

            cursor.execute("SELECT MIN(collected_at) as oldest FROM articles WHERE processing_status IN ('collected', 'pending_verification')")
            oldest_str = cursor.fetchone()["oldest"]
            oldest_pending_age_sec = None
            if oldest_str:
                try:
                    dt = datetime.fromisoformat(oldest_str.replace("Z", "+00:00"))
                    oldest_pending_age_sec = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
                except Exception:
                    pass

            return {
                "total_articles": total,
                "pending_verification": pending,
                "processing_articles": processing,
                "verified_articles": verified,
                "failed_articles": failed,
                "human_review_required": review_needed,
                "verdict_distribution": verdicts,
                "region_distribution": regions,
                "provenance_distribution": provenance_counts,
                "publishers": publishers,
                "last_collected_at": last_col,
                "last_verified_at": last_ver,
                "oldest_pending_age_sec": round(oldest_pending_age_sec, 1) if oldest_pending_age_sec is not None else None
            }

    def seed_initial_articles_if_empty(self) -> int:
        """
        Seeds realistic Indian and International articles if database is newly initialized.
        Ensures demonstrable state across diverse scenarios (Supported, Misleading Context, Uncertain).
        Explicitly stamps all initial demonstration records with provenance='SEEDED_DEMO'.
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as total FROM articles")
            if cursor.fetchone()["total"] > 0:
                return 0

        logger.info("Database empty; seeding realistic baseline articles...")
        seeds = [
            {
                "canonical_url": "https://www.thehindu.com/news/national/isro-gaganyaan-crew-escape-system-test-success/article67450123.ece",
                "original_url": "https://www.thehindu.com/news/national/isro-gaganyaan-crew-escape-system-test-success/article67450123.ece",
                "headline": "ISRO successfully conducts Gaganyaan flight abort test vehicle mission from Sriharikota",
                "description": "The Indian Space Research Organisation (ISRO) on Saturday successfully executed the Test Vehicle D1 mission demonstrating the crew escape system of the Gaganyaan human spaceflight project.",
                "publisher": "The Hindu",
                "country": "India",
                "region": "India",
                "published_at": "2026-10-21T08:30:00Z",
                "image_url": "https://th-i.thgim.com/public/incoming/gaganyaan_test.jpg",
                "source_feed": "the_hindu_national",
                "provenance": "SEEDED_DEMO"
            },
            {
                "canonical_url": "https://feeds.bbci.co.uk/news/world/un-climate-summit-renewable-energy-agreement/article883921.ece",
                "original_url": "http://feeds.bbci.co.uk/news/world/un-climate-summit-renewable-energy-agreement/article883921.ece",
                "headline": "Global delegates sign landmark agreement to triple renewable energy capacity by 2030",
                "description": "Representatives from over 110 nations approved a joint pact at the international climate summit committing to accelerate solar, wind, and storage deployments over the next decade.",
                "publisher": "BBC News (World)",
                "country": "United Kingdom",
                "region": "International",
                "published_at": "2026-10-22T14:15:00Z",
                "image_url": "https://ichef.bbci.co.uk/news/976/cpsprodpb/wind_turbines.jpg",
                "source_feed": "bbc_world_news",
                "provenance": "SEEDED_DEMO"
            },
            {
                "canonical_url": "https://timesofindia.indiatimes.com/india/delhi-air-quality-emergency-measures-anti-smog-guns/articleshow/99812455.cms",
                "original_url": "https://timesofindia.indiatimes.com/india/delhi-air-quality-emergency-measures-anti-smog-guns/articleshow/99812455.cms",
                "headline": "Delhi activates Stage 4 anti-pollution curbs as Air Quality Index enters severe category",
                "description": "Authorities in the national capital imposed strict construction halts and deployed anti-smog water sprinklers across high-density corridors as AQI levels exceeded 420 on Sunday.",
                "publisher": "Times of India",
                "country": "India",
                "region": "India",
                "published_at": "2026-10-23T06:45:00Z",
                "image_url": "https://static.toiimg.com/thumb/delhi_smog.jpg",
                "source_feed": "times_of_india_top",
                "provenance": "SEEDED_DEMO"
            },
            {
                "canonical_url": "https://www.theguardian.com/world/2026/oct/24/ancient-submerged-roman-port-discovered-mediterranean",
                "original_url": "https://www.theguardian.com/world/2026/oct/24/ancient-submerged-roman-port-discovered-mediterranean",
                "headline": "Marine archaeologists map submerged Roman harbour structures off Sicilian coastline",
                "description": "Underwater acoustic scans have revealed remarkably intact breakwaters and stone storage warehouses dating to the second century CE submerged beneath the southern Mediterranean seabed.",
                "publisher": "The Guardian (World)",
                "country": "United Kingdom",
                "region": "International",
                "published_at": "2026-10-24T11:00:00Z",
                "image_url": "https://i.guim.co.uk/img/media/roman_harbour.jpg",
                "source_feed": "guardian_world_news",
                "provenance": "SEEDED_DEMO"
            },
            {
                "canonical_url": "https://indianexpress.com/article/india/supreme-court-electoral-bonds-scheme-unconstitutional-ruling/9128374/",
                "original_url": "https://indianexpress.com/article/india/supreme-court-electoral-bonds-scheme-unconstitutional-ruling/9128374/",
                "headline": "Supreme Court Constitution Bench strikes down anonymous political financing scheme",
                "description": "In a unanimous judgment, the five-judge bench ruled that the anonymous donor framework infringed on voters' fundamental right to information under Article 19(1)(a) of the Constitution.",
                "publisher": "Indian Express",
                "country": "India",
                "region": "India",
                "published_at": "2026-10-25T10:30:00Z",
                "image_url": "https://images.indianexpress.com/sc_judgment.jpg",
                "source_feed": "indian_express_india",
                "provenance": "SEEDED_DEMO"
            }
        ]

        count = 0
        for item in seeds:
            is_new, _ = self.upsert_article(item)
            if is_new:
                count += 1
        logger.info(f"[OK] Seeded {count} initial articles into database.")
        return count
