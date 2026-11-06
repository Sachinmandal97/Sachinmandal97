"""
VERITAS-MM 2.0: Multimodal Misinformation Verification & Real-Time News Intelligence.
"""

from src.database import NewsIntelligenceDB, VALID_PROVENANCES
from src.news_ingestion import NewsIngestionService, canonicalize_url, parse_timestamp_to_iso
from src.news_verifier import AutomaticNewsVerifier, extract_verifiable_claim
from src.pipeline import MultimodalVerificationPipeline
from src.schemas import DecisionVerdict, MultimodalSample, VerificationResult
from src.scheduler import start_scheduler, stop_scheduler, trigger_manual_poll, get_scheduler_telemetry

__all__ = [
    "MultimodalVerificationPipeline",
    "MultimodalSample",
    "VerificationResult",
    "DecisionVerdict",
    "NewsIntelligenceDB",
    "VALID_PROVENANCES",
    "NewsIngestionService",
    "AutomaticNewsVerifier",
    "canonicalize_url",
    "parse_timestamp_to_iso",
    "extract_verifiable_claim",
    "start_scheduler",
    "stop_scheduler",
    "trigger_manual_poll",
    "get_scheduler_telemetry",
]
