"""
End-to-end multimodal misinformation verification pipeline.
Orchestrates:
Input (Text + Image)
  ↓
1. Preprocess
  ↓
2. OCR Extraction
  ↓
3. Cross-Modal Consistency Analysis (OpenCLIP)
  ↓
4. Claim & Entity Decomposition (spaCy)
  ↓
5. Evidence Retrieval (BM25)
  ↓
6. Verification & Relation Estimation (DeBERTa NLI)
  ↓
7. Confidence & Predictive Uncertainty
  ↓
8. Structured Explainability Report
  ↓
VerificationResult Output
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import yaml
from PIL import Image

from src.cross_modal import compute_cross_modal_consistency
from src.decomposition import decompose_claim
from src.explainability import generate_structured_explanation
from src.ocr import extract_ocr_text
from src.preprocess import clean_text, load_and_validate_image
from src.retrieval import EvidenceRetriever
from src.schemas import (
    CrossModalScore,
    DecisionVerdict,
    EvidenceSnippet,
    MultimodalSample,
    NLIResult,
    OCRResult,
    VerificationResult,
)
from src.uncertainty import compute_decision_confidence_and_uncertainty
from src.utils import HardwareProfiler, setup_logger
from src.verification import estimate_nli_relation, fuse_multimodal_decision, verify_atomic_claims

logger = setup_logger("pipeline")


class MultimodalVerificationPipeline:
    """
    Main orchestrator for multimodal verification.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("./configs/config.yaml")
        self.config = self._load_config()
        self.profiler = HardwareProfiler()

        corpus_dir = Path(self.config.get("paths", {}).get("data_corpus", "./data/corpus"))
        primary_corpus = corpus_dir / "evidence_corpus.jsonl"
        fallback_corpus = corpus_dir / "mock_evidence.jsonl"
        corpus_path = primary_corpus if primary_corpus.exists() else fallback_corpus
        enable_live = self.config.get("retrieval", {}).get("enable_live_search", True)
        rel_thresh = self.config.get("retrieval", {}).get("relevance_threshold", 0.20)
        cache_ttl = self.config.get("retrieval", {}).get("cache_ttl_seconds", 3600)
        self.retriever = EvidenceRetriever(
            corpus_path=corpus_path,
            enable_live_search=enable_live,
            relevance_threshold=rel_thresh,
            cache_ttl_seconds=cache_ttl
        )
        logger.info("[OK] Multimodal Verification Pipeline initialized.")


    def _load_config(self) -> Dict[str, Any]:
        """Loads YAML configuration file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found at {self.config_path}. Using default settings.")
            return {}
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def verify_sample(
        self,
        sample: MultimodalSample,
        force_mode: Optional[str] = None  # 'local_only', 'live_only', 'hybrid'
    ) -> VerificationResult:
        """
        Processes a single MultimodalSample through the full verification flow.
        """
        # Step 1: Preprocessing
        clean_claim = self.profiler.measure_stage("preprocess_text", clean_text, sample.claim_text)
        image_input = getattr(sample, "image_data", None) if getattr(sample, "image_data", None) is not None else sample.image_path
        image_obj, img_err = self.profiler.measure_stage("preprocess_image", load_and_validate_image, image_input)

        # Step 2: OCR Extraction
        ocr_res: OCRResult = self.profiler.measure_stage(
            "ocr",
            extract_ocr_text,
            image=image_obj,
            gpu=self.config.get("ocr", {}).get("gpu", False)
        )

        # Step 3: Cross-Modal Alignment (OpenCLIP)
        cm_threshold = self.config.get("thresholds", {}).get("cross_modal_mismatch_cutoff", 0.28)
        clip_model_name = self.config.get("models", {}).get("clip_model", "openai/clip-vit-base-patch32")
        device = self.config.get("project", {}).get("device", "cpu")

        had_image_input = bool(getattr(sample, "image_data", None) is not None or sample.image_path)
        if had_image_input and image_obj is None:
            # User provided an image input, but it failed validation or was corrupted (Case I)
            cross_modal = CrossModalScore(
                similarity_score=None,
                mismatch_detected=False,
                has_image=True,
                status="error",
                details={"error": img_err or "Corrupted or unreadable image file"}
            )
        elif not had_image_input:
            cross_modal = CrossModalScore(
                similarity_score=None,
                mismatch_detected=False,
                has_image=False,
                status="text_only",
                details={"info": "No image provided; text-only evaluation mode"}
            )
        else:
            cross_modal = self.profiler.measure_stage(
                "cross_modal",
                compute_cross_modal_consistency,
                image=image_obj,
                claim_text=clean_claim,
                ocr_text=ocr_res.extracted_text,
                model_name=clip_model_name,
                device=device,
                mismatch_threshold=cm_threshold
            )

        # Step 4: Claim Decomposition (spaCy)
        decomposed_claims = self.profiler.measure_stage(
            "decomposition",
            decompose_claim,
            claim_text=clean_claim
        )

        # Step 5: Evidence Retrieval (Universal: Hybrid / Live / Local)
        top_k = self.config.get("retrieval", {}).get("top_k", 3)
        retrieved_evidence = self.profiler.measure_stage(
            "retrieval",
            self.retriever.retrieve_for_claims,
            claims=decomposed_claims,
            top_k=top_k,
            force_mode=force_mode
        )

        # Step 6: Verification & NLI Relation Estimation
        nli_model_name = self.config.get("models", {}).get("nli_model", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
        ent_cutoff = self.config.get("thresholds", {}).get("nli_entailment_cutoff", 0.60)
        contra_cutoff = self.config.get("thresholds", {}).get("nli_contradiction_cutoff", 0.50)

        # 6a. Granular per-atomic claim verification with relevance awareness
        atomic_verifications = self.profiler.measure_stage(
            "atomic_verification",
            verify_atomic_claims,
            atomic_claims=decomposed_claims,
            retriever=self.retriever,
            model_name=nli_model_name,
            device=device,
            entailment_cutoff=ent_cutoff,
            contradiction_cutoff=contra_cutoff,
            force_mode=force_mode,
            top_k=top_k
        )

        # 6b. Global document-level relation estimation (only for relevant evidence)
        nli_results: list[NLIResult] = []
        for snip in retrieved_evidence:
            if not snip.is_relevant:
                continue
            nli_res = self.profiler.measure_stage(
                "nli",
                estimate_nli_relation,
                claim_text=clean_claim,
                evidence_text=snip.text,
                snippet_id=snip.snippet_id,
                model_name=nli_model_name,
                device=device
            )
            nli_results.append(nli_res)

        # Multimodal Decision Fusion
        decision, reasoning_factors = self.profiler.measure_stage(
            "decision_fusion",
            fuse_multimodal_decision,
            cross_modal=cross_modal,
            nli_results=nli_results,
            atomic_verifications=atomic_verifications,
            mismatch_threshold=cm_threshold,
            entailment_cutoff=ent_cutoff,
            contradiction_cutoff=contra_cutoff
        )

        # Step 7: Confidence & Uncertainty Estimation
        confidence, uncertainty_score = self.profiler.measure_stage(
            "uncertainty",
            compute_decision_confidence_and_uncertainty,
            decision=decision,
            cross_modal=cross_modal,
            nli_results=nli_results,
            mismatch_threshold=cm_threshold
        )

        # Step 8: Structured Explainability Report
        explanation = self.profiler.measure_stage(
            "explainability",
            generate_structured_explanation,
            claim_text=clean_claim,
            decision=decision,
            confidence=confidence,
            uncertainty_score=uncertainty_score,
            ocr=ocr_res,
            cross_modal=cross_modal,
            evidence=retrieved_evidence,
            nli_results=nli_results,
            reasoning_factors=reasoning_factors,
            atomic_verifications=atomic_verifications
        )

        mode_label = force_mode or ("hybrid" if self.retriever.enable_live_search else "local_corpus")

        # Complete Result Package
        return VerificationResult(
            sample_id=sample.sample_id,
            claim_text=clean_claim,
            decision=decision,
            confidence=confidence,
            uncertainty_score=uncertainty_score,
            evidence_source_mode=mode_label,
            ocr=ocr_res,
            cross_modal=cross_modal,
            decomposed_claims=decomposed_claims,
            atomic_verifications=atomic_verifications,
            retrieved_evidence=retrieved_evidence,
            nli_results=nli_results,
            reasoning_factors=reasoning_factors,
            structured_explanation=explanation
        )
