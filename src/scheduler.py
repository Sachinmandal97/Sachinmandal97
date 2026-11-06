"""
Background News Intelligence & Verification Scheduler for VERITAS-MM 2.0.
Provides thread-safe, non-duplicating scheduled polling and bounded verification for the Streamlit runtime.

Operational Lifecycle & Governance:
- Ingestion polling interval: default 300 seconds (5 minutes).
- Bounded verification interval: default 120 seconds (2 minutes), configurable independently.
- Ingestion and verification operate as logically separate workflows.
- Verification is disabled by default unless explicitly enabled by configuration or UI control.
- Verification batch size: default 2 articles per cycle to bound CPU inference.
- Guarded by atomic in-process locks to prevent duplicate worker threads on Streamlit script reruns.
- Multi-process safety: coordinated via database-backed lease locks (system_locks table in SQLite).
- Gracefully terminates when the application process terminates (daemon threads).
- Discloses clearly that automated operations run only while the application process is active.
- Does NOT claim continuous 24/7 background monitoring without an active host process.
"""

from datetime import datetime, timezone
import hashlib
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from src.database import NewsIntelligenceDB
from src.news_ingestion import NewsIngestionService
from src.news_verifier import AutomaticNewsVerifier
from src.utils import setup_logger

logger = setup_logger("scheduler")

# Singleton state guards
_SCHEDULER_LOCK = threading.Lock()
_POLL_EXECUTION_LOCK = threading.Lock()
_VERIFY_EXECUTION_LOCK = threading.Lock()

_INGESTION_THREAD: Optional[threading.Thread] = None
_VERIFY_THREAD: Optional[threading.Thread] = None
_STOP_EVENT = threading.Event()

# Ingestion Telemetry
_LAST_POLL_TIME: Optional[str] = None
_LAST_POLL_RESULT: Optional[Dict[str, Any]] = None
_NEXT_POLL_TIMESTAMP: Optional[float] = None
_POLL_INTERVAL_SEC: int = 300
_IS_RUNNING: bool = False

# Verification Telemetry
_AUTO_VERIFY_ENABLED: bool = False
_VERIFY_INTERVAL_SEC: int = 120
_VERIFY_BATCH_SIZE: int = 2
_LAST_VERIFY_TIME: Optional[str] = None
_LAST_VERIFY_RESULT: Optional[Dict[str, Any]] = None
_NEXT_VERIFY_TIMESTAMP: Optional[float] = None


def _get_worker_id() -> str:
    """Returns a unique worker process/thread identifier for multi-process lease locks."""
    pid = os.getpid()
    tid = threading.get_ident()
    return f"worker_{pid}_{tid}"


def _ingestion_worker_loop(ingestion: NewsIngestionService, db: NewsIntelligenceDB, interval_sec: int):
    """Background loop executing scheduled news polling cycles."""
    global _LAST_POLL_TIME, _LAST_POLL_RESULT, _NEXT_POLL_TIMESTAMP, _IS_RUNNING
    logger.info(f"VERITAS-MM Background News Ingestion Scheduler started (interval: {interval_sec}s).")

    while not _STOP_EVENT.is_set():
        _NEXT_POLL_TIMESTAMP = time.time() + interval_sec
        if _STOP_EVENT.wait(timeout=interval_sec):
            break

        if _STOP_EVENT.is_set():
            break

        acquired = _POLL_EXECUTION_LOCK.acquire(blocking=False)
        if not acquired:
            logger.info("Scheduled poll skipped: Another ingestion cycle is currently executing.")
            continue

        worker_id = _get_worker_id()
        has_db_lock = False
        try:
            has_db_lock = db.acquire_lock("veritas_ingestion_lease", owner_id=worker_id, ttl_seconds=interval_sec * 1.5)
            if not has_db_lock:
                logger.info("Scheduled poll skipped: Another system process currently holds the ingestion lease lock.")
                continue

            logger.info("Starting scheduled news intelligence polling cycle...")
            res = ingestion.poll_all_sources(db)
            _LAST_POLL_TIME = datetime.now(timezone.utc).isoformat()
            _LAST_POLL_RESULT = res
            logger.info(f"Scheduled poll cycle complete: stored {res.get('total_new_stored', 0)} new articles.")
        except Exception as e:
            logger.error(f"Error during scheduled news polling cycle: {e}")
        finally:
            if has_db_lock:
                try:
                    db.release_lock("veritas_ingestion_lease", owner_id=worker_id)
                except Exception:
                    pass
            _POLL_EXECUTION_LOCK.release()

    _IS_RUNNING = False
    logger.info("VERITAS-MM Background News Ingestion Scheduler stopped.")


def _verification_worker_loop(
    verifier: AutomaticNewsVerifier,
    db: NewsIntelligenceDB,
    interval_sec: int,
    batch_size: int
):
    """
    Background loop executing scheduled bounded verification cycles.
    Processes up to batch_size articles per cycle without starving CPU or blocking UI.
    """
    global _LAST_VERIFY_TIME, _LAST_VERIFY_RESULT, _NEXT_VERIFY_TIMESTAMP
    logger.info(f"VERITAS-MM Bounded Verification Scheduler started (interval: {interval_sec}s, batch_size: {batch_size}).")

    # Recover any stranded articles from previous interrupted sessions on worker startup
    try:
        db.recover_interrupted_articles()
    except Exception as e:
        logger.warning(f"Could not recover interrupted articles on startup: {e}")

    while not _STOP_EVENT.is_set():
        _NEXT_VERIFY_TIMESTAMP = time.time() + interval_sec
        if _STOP_EVENT.wait(timeout=interval_sec):
            break

        if _STOP_EVENT.is_set():
            break

        if not _AUTO_VERIFY_ENABLED:
            logger.debug("Automated verification cycle skipped: auto-verify disabled by configuration.")
            continue

        acquired = _VERIFY_EXECUTION_LOCK.acquire(blocking=False)
        if not acquired:
            logger.info("Scheduled verification skipped: Another verification cycle is currently executing.")
            continue

        worker_id = _get_worker_id()
        has_db_lock = False
        try:
            has_db_lock = db.acquire_lock("veritas_verification_lease", owner_id=worker_id, ttl_seconds=interval_sec * 2.0)
            if not has_db_lock:
                logger.info("Scheduled verification skipped: Another system process currently holds the verification lease lock.")
                continue

            logger.info(f"Starting scheduled bounded verification cycle (max batch: {batch_size})...")
            t0 = time.perf_counter()
            verified = verifier.verify_pending_batch(db, batch_size=batch_size)
            elapsed = time.perf_counter() - t0

            _LAST_VERIFY_TIME = datetime.now(timezone.utc).isoformat()
            _LAST_VERIFY_RESULT = {
                "verified_count": len(verified),
                "elapsed_sec": round(elapsed, 2),
                "timestamp": _LAST_VERIFY_TIME,
                "batch_size": batch_size
            }
            if verified:
                logger.info(f"Scheduled verification cycle complete: verified {len(verified)} articles in {elapsed:.2f}s.")
            else:
                logger.debug("Scheduled verification cycle complete: queue was empty.")
        except Exception as e:
            logger.error(f"Error during scheduled verification cycle: {e}")
            _LAST_VERIFY_RESULT = {
                "verified_count": 0,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        finally:
            if has_db_lock:
                try:
                    db.release_lock("veritas_verification_lease", owner_id=worker_id)
                except Exception:
                    pass
            _VERIFY_EXECUTION_LOCK.release()

    logger.info("VERITAS-MM Bounded Verification Scheduler stopped.")


class PollResult(dict):
    """
    Polymorphic result object supporting both dict access (e.g. res["status"])
    and tuple unpacking (e.g. res, err = trigger_manual_poll()).
    """
    def __iter__(self):
        result_payload = self.get("result")
        err_msg = self.get("message") if self.get("status") in ("busy", "error") else None
        return iter((result_payload, err_msg))


def start_scheduler(
    ingestion: Optional[NewsIngestionService] = None,
    verifier: Optional[AutomaticNewsVerifier] = None,
    db: Optional[NewsIntelligenceDB] = None,
    interval_sec: int = 300,
    auto_verify: bool = False,
    verify_interval_sec: int = 120,
    verify_batch_size: int = 2
) -> Optional[threading.Thread]:
    """
    Starts background worker threads for ingestion and optional bounded verification.
    Guaranteed idempotent across Streamlit script reruns via atomic process locks.
    Returns the primary active ingestion worker thread instance for backward compatibility.
    """
    global _INGESTION_THREAD, _VERIFY_THREAD, _POLL_INTERVAL_SEC, _IS_RUNNING
    global _NEXT_POLL_TIMESTAMP, _AUTO_VERIFY_ENABLED, _VERIFY_INTERVAL_SEC, _VERIFY_BATCH_SIZE

    with _SCHEDULER_LOCK:
        _POLL_INTERVAL_SEC = interval_sec
        _AUTO_VERIFY_ENABLED = auto_verify
        _VERIFY_INTERVAL_SEC = verify_interval_sec
        _VERIFY_BATCH_SIZE = verify_batch_size

        if db is None:
            db = NewsIntelligenceDB()
        if ingestion is None:
            ingestion = NewsIngestionService()
        if verifier is None and auto_verify:
            verifier = AutomaticNewsVerifier()

        # 1. Manage Ingestion Worker Thread
        if _INGESTION_THREAD is None or not _INGESTION_THREAD.is_alive():
            _STOP_EVENT.clear()
            _IS_RUNNING = True
            _NEXT_POLL_TIMESTAMP = time.time() + interval_sec
            _INGESTION_THREAD = threading.Thread(
                target=_ingestion_worker_loop,
                args=(ingestion, db, interval_sec),
                name="VERITAS_NewsIngestionWorker",
                daemon=True
            )
            _INGESTION_THREAD.start()

        # 2. Manage Verification Worker Thread
        if verifier is not None:
            if _VERIFY_THREAD is None or not _VERIFY_THREAD.is_alive():
                _NEXT_VERIFY_TIMESTAMP = time.time() + verify_interval_sec
                _VERIFY_THREAD = threading.Thread(
                    target=_verification_worker_loop,
                    args=(verifier, db, verify_interval_sec, verify_batch_size),
                    name="VERITAS_NewsVerificationWorker",
                    daemon=True
                )
                _VERIFY_THREAD.start()

        return _INGESTION_THREAD


def stop_scheduler(timeout: float = 2.0) -> bool:
    """Signals all background polling and verification workers to stop."""
    global _INGESTION_THREAD, _VERIFY_THREAD, _IS_RUNNING
    with _SCHEDULER_LOCK:
        _STOP_EVENT.set()
        if _INGESTION_THREAD is not None and _INGESTION_THREAD.is_alive():
            _INGESTION_THREAD.join(timeout=timeout)
        if _VERIFY_THREAD is not None and _VERIFY_THREAD.is_alive():
            _VERIFY_THREAD.join(timeout=timeout)
        _IS_RUNNING = False
        _INGESTION_THREAD = None
        _VERIFY_THREAD = None
        _STOP_EVENT.clear()
        return True


def set_auto_verify(enabled: bool, interval_sec: Optional[int] = None, batch_size: Optional[int] = None) -> None:
    """Dynamically enables or disables automatic bounded verification."""
    global _AUTO_VERIFY_ENABLED, _VERIFY_INTERVAL_SEC, _VERIFY_BATCH_SIZE
    with _SCHEDULER_LOCK:
        _AUTO_VERIFY_ENABLED = enabled
        if interval_sec is not None:
            _VERIFY_INTERVAL_SEC = max(10, interval_sec)
        if batch_size is not None:
            _VERIFY_BATCH_SIZE = max(1, min(10, batch_size))
        logger.info(f"Auto-verification setting updated: enabled={_AUTO_VERIFY_ENABLED}, interval={_VERIFY_INTERVAL_SEC}s, batch_size={_VERIFY_BATCH_SIZE}")


def trigger_manual_poll(
    ingestion: Optional[NewsIngestionService] = None,
    db: Optional[NewsIntelligenceDB] = None
) -> PollResult:
    """
    Executes an immediate manual poll guarded against concurrent execution.
    Returns a PollResult supporting both dict access and tuple unpacking (res, err).
    """
    global _LAST_POLL_TIME, _LAST_POLL_RESULT, _NEXT_POLL_TIMESTAMP

    acquired = _POLL_EXECUTION_LOCK.acquire(blocking=False)
    if not acquired:
        return PollResult({
            "status": "busy",
            "message": "A polling cycle is already in progress. Please wait for it to complete.",
            "result": None
        })

    try:
        if ingestion is None:
            ingestion = NewsIngestionService()
        if db is None:
            db = NewsIntelligenceDB()

        logger.info("Executing user-triggered manual news polling cycle...")
        res = ingestion.poll_all_sources(db)
        _LAST_POLL_TIME = datetime.now(timezone.utc).isoformat()
        _LAST_POLL_RESULT = res
        _NEXT_POLL_TIMESTAMP = time.time() + _POLL_INTERVAL_SEC
        return PollResult({
            "status": "success",
            "message": None,
            "result": res
        })
    except Exception as e:
        logger.error(f"Manual news polling cycle failed: {e}")
        return PollResult({
            "status": "error",
            "message": str(e),
            "result": None
        })
    finally:
        _POLL_EXECUTION_LOCK.release()


def trigger_manual_verification(
    verifier: Optional[AutomaticNewsVerifier] = None,
    db: Optional[NewsIntelligenceDB] = None,
    batch_size: int = 2
) -> PollResult:
    """
    Executes an immediate bounded verification cycle guarded against concurrent execution.
    Returns a PollResult supporting both dict access and tuple unpacking (res, err).
    """
    global _LAST_VERIFY_TIME, _LAST_VERIFY_RESULT

    acquired = _VERIFY_EXECUTION_LOCK.acquire(blocking=False)
    if not acquired:
        return PollResult({
            "status": "busy",
            "message": "A verification cycle is already in progress. Please wait for it to complete.",
            "result": None
        })

    try:
        if db is None:
            db = NewsIntelligenceDB()
        if verifier is None:
            verifier = AutomaticNewsVerifier()

        logger.info(f"Executing user-triggered manual batch verification (batch_size={batch_size})...")
        t0 = time.perf_counter()
        verified = verifier.verify_pending_batch(db, batch_size=batch_size)
        elapsed = time.perf_counter() - t0

        _LAST_VERIFY_TIME = datetime.now(timezone.utc).isoformat()
        payload = {
            "verified_count": len(verified),
            "elapsed_sec": round(elapsed, 2),
            "records": verified,
            "timestamp": _LAST_VERIFY_TIME
        }
        _LAST_VERIFY_RESULT = payload
        return PollResult({
            "status": "success",
            "message": None,
            "result": payload
        })
    except Exception as e:
        logger.error(f"Manual verification cycle failed: {e}")
        return PollResult({
            "status": "error",
            "message": str(e),
            "result": None
        })
    finally:
        _VERIFY_EXECUTION_LOCK.release()


def get_scheduler_telemetry(db: Optional[NewsIntelligenceDB] = None) -> Dict[str, Any]:
    """Returns current scheduler state, separate ingestion and verification telemetry, and queue metrics."""
    global _LAST_POLL_TIME, _LAST_POLL_RESULT, _NEXT_POLL_TIMESTAMP, _POLL_INTERVAL_SEC, _IS_RUNNING
    global _AUTO_VERIFY_ENABLED, _VERIFY_INTERVAL_SEC, _VERIFY_BATCH_SIZE, _LAST_VERIFY_TIME, _LAST_VERIFY_RESULT, _NEXT_VERIFY_TIMESTAMP
    now = time.time()

    is_ingest_active = bool(_INGESTION_THREAD is not None and _INGESTION_THREAD.is_alive() and _IS_RUNNING)
    is_verify_active = bool(_VERIFY_THREAD is not None and _VERIFY_THREAD.is_alive())

    seconds_until_next_poll = None
    next_poll_iso = None
    if is_ingest_active and _NEXT_POLL_TIMESTAMP:
        seconds_until_next_poll = max(0, int(_NEXT_POLL_TIMESTAMP - now))
        next_poll_iso = datetime.fromtimestamp(_NEXT_POLL_TIMESTAMP, tz=timezone.utc).isoformat()

    seconds_until_next_verify = None
    next_verify_iso = None
    if is_verify_active and _NEXT_VERIFY_TIMESTAMP:
        seconds_until_next_verify = max(0, int(_NEXT_VERIFY_TIMESTAMP - now))
        next_verify_iso = datetime.fromtimestamp(_NEXT_VERIFY_TIMESTAMP, tz=timezone.utc).isoformat()

    queue_metrics = {}
    if db is not None:
        try:
            queue_metrics = db.get_queue_metrics()
        except Exception:
            pass

    return {
        # Backward-compatible fields
        "is_active": is_ingest_active,
        "is_running": is_ingest_active,
        "is_polling_now": _POLL_EXECUTION_LOCK.locked(),
        "interval_seconds": _POLL_INTERVAL_SEC,
        "interval_sec": _POLL_INTERVAL_SEC,
        "last_polled_at": _LAST_POLL_TIME,
        "next_poll_at": next_poll_iso,
        "seconds_until_next_poll": seconds_until_next_poll,
        "last_poll_result": _LAST_POLL_RESULT,

        # Separate Ingestion Telemetry
        "ingestion": {
            "is_active": is_ingest_active,
            "is_busy": _POLL_EXECUTION_LOCK.locked(),
            "interval_sec": _POLL_INTERVAL_SEC,
            "last_synced_at": _LAST_POLL_TIME,
            "next_sync_in_sec": seconds_until_next_poll,
            "last_result": _LAST_POLL_RESULT
        },

        # Separate Verification Telemetry
        "verification": {
            "is_worker_active": is_verify_active,
            "is_auto_enabled": _AUTO_VERIFY_ENABLED,
            "is_busy": _VERIFY_EXECUTION_LOCK.locked(),
            "interval_sec": _VERIFY_INTERVAL_SEC,
            "batch_size": _VERIFY_BATCH_SIZE,
            "last_verified_at": _LAST_VERIFY_TIME,
            "next_verify_in_sec": seconds_until_next_verify,
            "last_result": _LAST_VERIFY_RESULT
        },

        # Queue Metrics
        "queue": queue_metrics
    }
