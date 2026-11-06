"""
Automatic News Verification Controller for VERITAS-MM 2.0.
Processes newly ingested articles through the explainable multimodal verification pipeline:
  1. Headline claim normalization and proposition selection.
  2. Safe media fetching with image validation and text-only fallback.
  3. Retrieval-augmented NLI stance estimation and OpenCLIP consistency evaluation.
  4. Transparent evidence aggregation and uncertainty calibration.
  5. Bounded batch execution to prevent CPU exhaustion.
"""

from datetime import datetime, timezone
import io
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
from PIL import Image

from src.database import NewsIntelligenceDB
from src.pipeline import MultimodalVerificationPipeline
from src.schemas import DecisionVerdict, MultimodalSample, VerificationResult
from src.utils import setup_logger

logger = setup_logger("news_verifier")

EDITORIAL_PREFIXES = [
    r"^watch\s*:\s*", r"^live\s*:\s*", r"^breaking(\s+news)?\s*:\s*", r"^exclusive\s*:\s*",
    r"^analysis\s*:\s*", r"^opinion\s*:\s*", r"^fact\s*check\s*:\s*", r"^explained\s*:\s*",
    r"^update\s*:\s*", r"^read\s*:\s*"
]


def extract_verifiable_claim(headline: str, description: str = "") -> str:
    """
    Cleans editorial headline prefixes and formulates a clear factual assertion.
    If headline is an incomplete fragment, supplements with lead clause from description.
    """
    text = headline.strip()
    for pattern in EDITORIAL_PREFIXES:
        text = re.sub(pattern, "", text, flags=re.I).strip()

    # If headline is very short (< 4 words) and description is available, use description lead
    words = text.split()
    if len(words) < 4 and description:
        clean_desc = re.sub(r"\s+", " ", description).strip()
        sentences = [s.strip() for s in clean_desc.split(".") if len(s.strip()) > 10]
        if sentences:
            return sentences[0] + "."

    return text


def fetch_article_image(image_url: Optional[str], timeout: int = 5) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Safely retrieves and validates lead image from remote URL.
    Returns (PIL.Image or None, status_note).
    Never raises an uncaught exception; safely falls back to text-only mode.
    """
    if not image_url or not image_url.strip():
        return None, "No image URL attached to article"

    url = image_url.strip()
    if not url.startswith(("http://", "https://")):
        return None, f"Invalid image URL scheme: {url}"

    headers = {
        "User-Agent": "VERITAS-MM/2.0 ImageValidator (Academic Data Science Research)"
    }
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "").lower()
            data = resp.read(5 * 1024 * 1024)  # Max 5 MB to prevent buffer attacks

        if not data:
            return None, "Empty image response"

        # Open and validate PIL image
        img = Image.open(io.BytesIO(data))
        img.load()
        rgb_img = img.convert("RGB")

        # Basic sanity check on dimensions
        if rgb_img.width < 50 or rgb_img.height < 50:
            return None, f"Image dimensions too small ({rgb_img.width}x{rgb_img.height}) for meaningful visual analysis"

        return rgb_img, "Successfully retrieved and converted to RGB"

    except Exception as e:
        logger.debug(f"Article image fetch bypassed for '{url}': {e}")
        return None, f"Image download error: {str(e)[:100]}"


class AutomaticNewsVerifier:
    """
    Orchestrates bounded automated verification of collected news articles.
    """

    def __init__(self, pipeline: Optional[MultimodalVerificationPipeline] = None):
        self._pipeline = pipeline

    def get_pipeline(self) -> MultimodalVerificationPipeline:
        """Lazily creates or returns the shared pipeline instance."""
        if self._pipeline is None:
            self._pipeline = MultimodalVerificationPipeline()
        return self._pipeline

    def verify_article(
        self,
        article: Dict[str, Any],
        db: NewsIntelligenceDB,
        force_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verifies a single news article record through the complete multimodal pipeline.
        Saves structured adjudication into the database and updates article status.
        """
        t0 = time.perf_counter()
        article_id = article["article_id"]
        headline = article.get("headline", "")
        description = article.get("description", "")
        image_url = article.get("image_url")

        # 1. Formulate factual proposition from headline & text
        claim_text = extract_verifiable_claim(headline, description)

        # 2. Acquire and validate visual asset (if present)
        image_obj, img_status = fetch_article_image(image_url)

        # 3. Construct MultimodalSample
        sample = MultimodalSample(
            sample_id=article_id,
            claim_text=claim_text,
            image_path=None,
            image_data=image_obj,
            dataset_name="live_news_intelligence",
            split="production"
        )

        pipeline = self.get_pipeline()
        logger.info(f"Verifying article '{article_id}': \"{claim_text[:60]}...\" (has_image={image_obj is not None})")

        try:
            # 4. Execute multimodal verification pipeline
            result: VerificationResult = pipeline.verify_sample(sample, force_mode=force_mode)
            elapsed = time.perf_counter() - t0

            # 5. Package verification record for database storage
            verif_data = {
                "article_id": article_id,
                "decision_verdict": result.decision.value,
                "decision_confidence": result.confidence,
                "predictive_uncertainty": result.uncertainty_score,
                "visual_consistency_score": result.cross_modal.similarity_score,
                "visual_consistency_status": getattr(result.cross_modal, "status", "evaluated"),
                "visual_mismatch_detected": result.cross_modal.mismatch_detected,
                "extracted_claims": [c.model_dump() for c in result.decomposed_claims],
                "atomic_verifications": [a.model_dump() for a in result.atomic_verifications],
                "retrieved_evidence": [e.model_dump() for e in result.retrieved_evidence],
                "reasoning_factors": result.reasoning_factors,
                "structured_report": result.structured_explanation,
                "execution_latency_sec": round(elapsed, 2),
                "verified_at": datetime.now(timezone.utc).isoformat(),
                "error_message": None
            }

            db.store_verification(verif_data)
            logger.info(f"[OK] Article '{article_id}' verified -> {result.decision.value} (conf={result.confidence:.2f}, time={elapsed:.2f}s)")
            return verif_data

        except Exception as e:
            elapsed = time.perf_counter() - t0
            err_msg = str(e)
            logger.error(f"Verification failed for article '{article_id}': {err_msg}")

            verif_data = {
                "article_id": article_id,
                "decision_verdict": DecisionVerdict.UNCERTAIN.value,
                "decision_confidence": 0.40,
                "predictive_uncertainty": 0.60,
                "visual_consistency_score": None,
                "visual_consistency_status": "error",
                "visual_mismatch_detected": False,
                "extracted_claims": [],
                "atomic_verifications": [],
                "retrieved_evidence": [],
                "reasoning_factors": [f"Verification error occurred: {err_msg}"],
                "structured_report": f"Verification execution error: {err_msg}",
                "execution_latency_sec": round(elapsed, 2),
                "verified_at": datetime.now(timezone.utc).isoformat(),
                "error_message": err_msg[:300]
            }
            db.store_verification(verif_data)
            return verif_data

    def verify_pending_batch(
        self,
        db: NewsIntelligenceDB,
        batch_size: int = 2,
        force_mode: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes bounded verification queue: claims and processes up to batch_size pending articles.
        Protects against runaway inference loops, concurrent double-processing, and CPU saturation.
        Guarantees that a single failed article does not crash or interrupt the batch.
        """
        claimed = db.claim_pending_articles(limit=batch_size)
        if not claimed:
            logger.info("No pending articles awaiting verification in queue.")
            return []

        logger.info(f"Starting bounded batch verification on {len(claimed)} claimed articles (batch_size={batch_size})...")
        results = []
        for article in claimed:
            try:
                res = self.verify_article(article, db, force_mode=force_mode)
                results.append(res)
            except Exception as e:
                err_msg = str(e)
                logger.error(f"Unexpected error verifying article {article.get('article_id')}: {err_msg}")
                db.update_article_status(article["article_id"], "failed")
        return results
