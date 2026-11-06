"""
Benchmark Dataset Adapters for NewsCLIPpings and Factify 2.
Implements decoupled adapters for Task A (Cheapfakes), Task B (Text Verification),
and Task C (Multimodal Verification) under controlled evidence grounding.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import torch

from src.cross_modal import compute_cross_modal_consistency
from src.data_loader import (
    load_factify2_samples,
    load_newsclippings,
    normalize_factify2_label,
)
from src.decomposition import decompose_claim
from src.evaluation_harness import BenchmarkPredictionRecord
from src.preprocess import clean_text, load_and_validate_image
from src.schemas import (
    AtomicClaim,
    AtomicVerificationResult,
    CrossModalScore,
    DecisionVerdict,
    EvidenceSnippet,
    Factify2Sample,
    MultimodalSample,
    NLIResult,
)
from src.uncertainty import compute_decision_confidence_and_uncertainty
from src.utils import setup_logger
from src.verification import (
    estimate_nli_relation,
    fuse_multimodal_decision,
)

logger = setup_logger("benchmark_adapters")


# ============================================================================
# 1. BASE ADAPTER INTERFACE
# ============================================================================

class BaseBenchmarkAdapter:
    """Abstract interface defining required methods for benchmark adapters."""
    name: str = "base"

    def get_target_labels(self, task_id: str) -> List[str]:
        raise NotImplementedError

    def get_sample_stratum_label(self, sample: Any) -> str:
        raise NotImplementedError

    def load_samples(
        self,
        data_path: Optional[Union[str, Path]] = None,
        split: str = "val",
        limit: Optional[int] = None
    ) -> List[Any]:
        raise NotImplementedError

    def evaluate_sample(
        self,
        sample: Any,
        pipeline: Any,
        task_id: str,
        split: str = "val",
        **kwargs
    ) -> BenchmarkPredictionRecord:
        raise NotImplementedError


# ============================================================================
# 2. NEWSCLIPPINGS ADAPTER (TASK A — CHEAPFAKE DETECTION)
# ============================================================================

class NewsCLIPpingsAdapter(BaseBenchmarkAdapter):
    """
    Adapter for the NewsCLIPpings dataset (Luo et al., EMNLP 2021).
    Evaluates out-of-context image-caption cheapfakes (Task A).
    Target: 'pristine' (0) vs 'falsified' (1).
    Evidence: None (visual-semantic correspondence only).
    """
    name: str = "newsclippings"

    def get_target_labels(self, task_id: str) -> List[str]:
        return ["pristine", "falsified"]

    def get_sample_stratum_label(self, sample: MultimodalSample) -> str:
        raw = str(sample.raw_label or "pristine").lower()
        return "falsified" if raw in ["falsified", "1", "true"] else "pristine"

    def load_samples(
        self,
        data_path: Optional[Union[str, Path]] = None,
        split: str = "test",
        limit: Optional[int] = None
    ) -> List[MultimodalSample]:
        dir_path = Path(data_path) if data_path else Path("./data/raw/news_clippings")
        return load_newsclippings(data_dir=dir_path, split=split, limit=limit)

    def evaluate_sample(
        self,
        sample: MultimodalSample,
        pipeline: Any,
        task_id: str = "TASK_A",
        split: str = "test",
        **kwargs
    ) -> BenchmarkPredictionRecord:
        """
        Evaluates a single NewsCLIPpings sample on Task A.
        Computes OpenCLIP cosine similarity and evaluates against threshold_cm.
        """
        threshold_cm = float(kwargs.get("threshold_cm", 0.22))
        clip_model_name = pipeline.config.get("models", {}).get("clip_model", "openai/clip-vit-base-patch32")
        device = pipeline.config.get("project", {}).get("device", "cpu")

        raw_label = str(sample.raw_label or "pristine").lower()
        ground_truth_mapped = "falsified" if raw_label in ["falsified", "1", "true"] else "pristine"

        # Check for precomputed embeddings in sample metadata
        s_ti: Optional[float] = None
        mismatch_detected = False

        if "clip_text_embedding" in sample.metadata and "clip_image_embedding" in sample.metadata:
            t_emb = torch.tensor(sample.metadata["clip_text_embedding"], dtype=torch.float32)
            i_emb = torch.tensor(sample.metadata["clip_image_embedding"], dtype=torch.float32)
            cos_sim = torch.cosine_similarity(t_emb.unsqueeze(0), i_emb.unsqueeze(0)).item()
            s_ti = float(cos_sim)
        elif sample.image_path:
            img_obj, _ = load_and_validate_image(sample.image_path)
            if img_obj is not None:
                cm_score: CrossModalScore = compute_cross_modal_consistency(
                    image=img_obj,
                    claim_text=clean_text(sample.claim_text),
                    model_name=clip_model_name,
                    device=device,
                    mismatch_threshold=threshold_cm
                )
                s_ti = cm_score.similarity_score
                mismatch_detected = cm_score.mismatch_detected
            else:
                raise ValueError(f"Could not load image file from {sample.image_path}")
        else:
            raise ValueError(f"NewsCLIPpings sample '{sample.sample_id}' has neither image_path nor precomputed embeddings.")

        # Decision rule for Task A:
        # Pristine if s_TI >= threshold_cm; Falsified if s_TI < threshold_cm
        predicted_verdict = "pristine" if s_ti >= threshold_cm else "falsified"
        mismatch_detected = bool(s_ti < threshold_cm)
        is_correct = (predicted_verdict == ground_truth_mapped)

        # Continuous confidence and uncertainty
        norm_score = max(0.0, min(1.0, (s_ti + 1.0) / 2.0))
        conf = norm_score if predicted_verdict == "pristine" else (1.0 - norm_score)
        conf = round(max(0.0, min(1.0, conf)), 4)
        uncertainty = round(1.0 - conf, 4)

        return BenchmarkPredictionRecord(
            sample_id=sample.sample_id,
            dataset_name=self.name,
            split=split,
            task_id=task_id.upper(),
            ground_truth_raw=raw_label,
            ground_truth_mapped=ground_truth_mapped,
            predicted_verdict=predicted_verdict,
            is_correct=is_correct,
            confidence=conf,
            uncertainty_score=uncertainty,
            cross_modal_score=round(s_ti, 4) if s_ti is not None else None,
            mismatch_detected=mismatch_detected,
            atomic_claims_count=1,
            nli_predicted_relation=None,
            nli_p_entailment=None,
            nli_p_neutral=None,
            nli_p_contradiction=None,
            evidence_source_mode="none",
            execution_time_sec=0.0,
            error_message=None
        )


# ============================================================================
# 3. FACTIFY 2 ADAPTER (TASK B & TASK C)
# ============================================================================

class Factify2Adapter(BaseBenchmarkAdapter):
    """
    Adapter for the Factify 2 dataset (Mishra et al., AAAI-23 Workshop).
    Supports:
      - Task B: Evidence-Grounded Factual Claim Verification (3-class text verification).
      - Task C: Unified Multimodal Misinformation Assessment (3-class or 5-class).
    Uses controlled paired reference document injection to isolate verification from web drift.
    """
    name: str = "factify2"

    def get_target_labels(self, task_id: str, protocol: str = "c2") -> List[str]:
        t_id = task_id.upper()
        if t_id == "TASK_B" or (t_id == "TASK_C" and protocol.lower() == "c2"):
            return ["SUPPORTED", "MISLEADING", "UNCERTAIN"]
        elif t_id == "TASK_C" and protocol.lower() == "c1":
            return ["Support_Multimodal", "Support_Text", "Refute", "Neutral", "Insufficient_Multimodal"]
        return ["SUPPORTED", "MISLEADING", "UNCERTAIN"]

    def get_sample_stratum_label(self, sample: Factify2Sample) -> str:
        return str(sample.label).strip()

    def load_samples(
        self,
        data_path: Optional[Union[str, Path]] = None,
        split: str = "val",
        limit: Optional[int] = None
    ) -> List[Factify2Sample]:
        return load_factify2_samples(path=data_path, split=split, limit=limit)

    def evaluate_sample(
        self,
        sample: Factify2Sample,
        pipeline: Any,
        task_id: str = "TASK_B",
        split: str = "val",
        **kwargs
    ) -> BenchmarkPredictionRecord:
        """
        Evaluates a single Factify 2 sample under controlled evidence grounding.
        """
        t_id = task_id.upper()
        protocol = kwargs.get("protocol", "c2").lower()
        ablation = kwargs.get("ablation", "none").lower()

        # Thresholds
        threshold_cm = float(kwargs.get("threshold_cm", pipeline.config.get("thresholds", {}).get("cross_modal_mismatch_cutoff", 0.22)))
        ent_cutoff = float(kwargs.get("entailment_cutoff", pipeline.config.get("thresholds", {}).get("nli_entailment_cutoff", 0.60)))
        contra_cutoff = float(kwargs.get("contradiction_cutoff", pipeline.config.get("thresholds", {}).get("nli_contradiction_cutoff", 0.50)))

        device = pipeline.config.get("project", {}).get("device", "cpu")
        nli_model_name = pipeline.config.get("models", {}).get("nli_model", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
        clip_model_name = pipeline.config.get("models", {}).get("clip_model", "openai/clip-vit-base-patch32")

        clean_claim_text = clean_text(sample.claim_text)
        doc_text = clean_text(sample.document_text or "")

        # 1. Map ground truth
        raw_label = sample.label.strip()
        norm_key = raw_label.lower()

        if t_id == "TASK_B" or (t_id == "TASK_C" and protocol == "c2"):
            # 3-Class Ground Truth Mapping
            if norm_key in ["support_text", "support_multimodal"]:
                ground_truth_mapped = "SUPPORTED"
            elif norm_key in ["refute"]:
                ground_truth_mapped = "MISLEADING"
            elif norm_key in ["neutral", "insufficient_multimodal"]:
                ground_truth_mapped = "UNCERTAIN"
            else:
                ground_truth_mapped = "UNCERTAIN"
        else:
            # 5-Class Native Protocol C1
            label_standard_map = {
                "support_multimodal": "Support_Multimodal",
                "support_text": "Support_Text",
                "refute": "Refute",
                "neutral": "Neutral",
                "insufficient_multimodal": "Insufficient_Multimodal"
            }
            ground_truth_mapped = label_standard_map.get(norm_key, raw_label)

        # 2. Controlled Evidence Grounding (bypasses live search)
        doc_snippet = EvidenceSnippet(
            snippet_id=f"doc_{sample.sample_id}",
            text=doc_text,
            source="Factify 2 Paired Document",
            title="Paired Reference Document",
            is_relevant=True,
            relevance_score=1.0,
            relevance_method="controlled_paired_document",
            relevance_reason="Official Factify 2 reference document",
            source_type="local_corpus",
            content_type="full_text",
            access_status="accessible" if bool(doc_text) else "inaccessible"
        )

        # 3. Proposition Decomposition (or ablation bypass)
        if ablation == "no_decomp":
            propositions = [
                AtomicClaim(
                    claim_id=f"prop_{sample.sample_id}_0",
                    text=clean_claim_text
                )
            ]
        else:
            propositions = decompose_claim(clean_claim_text)

        # 4. Cross-Modal Alignment (OpenCLIP)
        cm_score_val: Optional[float] = None
        mismatch_flag = False

        if t_id == "TASK_C" and ablation != "text_only" and sample.claim_image:
            img_obj, _ = load_and_validate_image(sample.claim_image)
            if img_obj is not None:
                cm_res = compute_cross_modal_consistency(
                    image=img_obj,
                    claim_text=clean_claim_text,
                    model_name=clip_model_name,
                    device=device,
                    mismatch_threshold=threshold_cm
                )
                cm_score_val = cm_res.similarity_score
                mismatch_flag = cm_res.mismatch_detected
                cross_modal = cm_res
            else:
                cross_modal = CrossModalScore(similarity_score=0.5, mismatch_detected=False)
        else:
            # Text-only mode (Task B or text-only ablation)
            cross_modal = CrossModalScore(similarity_score=1.0, mismatch_detected=False)

        # 5. Granular NLI Verification against Document
        atomic_verifications: List[AtomicVerificationResult] = []
        for prop in propositions:
            if not doc_text:
                atomic_verifications.append(
                    AtomicVerificationResult(
                        claim_id=prop.claim_id,
                        atomic_text=prop.text,
                        status="UNVERIFIED",
                        retrieved_evidence=None,
                        nli_result=None,
                        explanation="No paired document text provided."
                    )
                )
                continue

            nli_res = estimate_nli_relation(
                claim_text=prop.text,
                evidence_text=doc_text,
                snippet_id=doc_snippet.snippet_id,
                model_name=nli_model_name,
                device=device
            )

            if nli_res.predicted_relation == "contradiction" and nli_res.p_contradiction >= contra_cutoff:
                st = "CONTRADICTED"
            elif nli_res.predicted_relation == "entailment" and nli_res.p_entailment >= ent_cutoff:
                st = "SUPPORTED"
            else:
                st = "UNVERIFIED"

            atomic_verifications.append(
                AtomicVerificationResult(
                    claim_id=prop.claim_id,
                    atomic_text=prop.text,
                    status=st,
                    retrieved_evidence=doc_snippet,
                    nli_result=nli_res,
                    explanation=f"Evaluated against paired doc (p_ent: {nli_res.p_entailment:.2f}, p_contra: {nli_res.p_contradiction:.2f})"
                )
            )

        # Global document-level NLI
        global_nli: Optional[NLIResult] = None
        if doc_text:
            global_nli = estimate_nli_relation(
                claim_text=clean_claim_text,
                evidence_text=doc_text,
                snippet_id=doc_snippet.snippet_id,
                model_name=nli_model_name,
                device=device
            )

        nli_results_list = [global_nli] if global_nli is not None else []

        # 6. Multimodal Decision Fusion
        decision, reasoning_factors = fuse_multimodal_decision(
            cross_modal=cross_modal,
            nli_results=nli_results_list,
            atomic_verifications=atomic_verifications,
            mismatch_threshold=threshold_cm,
            entailment_cutoff=ent_cutoff,
            contradiction_cutoff=contra_cutoff
        )

        # Ablation 4: Forced Binary Classification (no abstention)
        if ablation == "forced_binary" and global_nli is not None:
            if global_nli.p_contradiction >= global_nli.p_entailment:
                decision = DecisionVerdict.MISLEADING
            else:
                decision = DecisionVerdict.SUPPORTED

        # Confidence and uncertainty
        confidence, uncertainty = compute_decision_confidence_and_uncertainty(
            decision=decision,
            cross_modal=cross_modal,
            nli_results=nli_results_list,
            mismatch_threshold=threshold_cm
        )

        # 7. Map predicted verdict to task target space
        if t_id == "TASK_B" or (t_id == "TASK_C" and protocol == "c2"):
            # 3-Class Prediction Space
            if decision in [DecisionVerdict.SUPPORTED, DecisionVerdict.PARTIALLY_SUPPORTED]:
                predicted_verdict = "SUPPORTED"
            elif decision == DecisionVerdict.MISLEADING:
                predicted_verdict = "MISLEADING"
            else:
                predicted_verdict = "UNCERTAIN"
        else:
            # 5-Class Prediction Space (Protocol C1)
            if decision == DecisionVerdict.SUPPORTED and not mismatch_flag:
                predicted_verdict = "Support_Multimodal"
            elif decision == DecisionVerdict.PARTIALLY_SUPPORTED or (decision == DecisionVerdict.SUPPORTED and mismatch_flag):
                predicted_verdict = "Support_Text"
            elif decision == DecisionVerdict.MISLEADING:
                predicted_verdict = "Refute"
            else:
                # Uncertain: differentiate Neutral vs Insufficient
                if doc_text and len(doc_text.split()) > 15:
                    predicted_verdict = "Neutral"
                else:
                    predicted_verdict = "Insufficient_Multimodal"

        is_correct = (predicted_verdict == ground_truth_mapped)

        # Extract top NLI relation and probabilities
        nli_pred_rel = global_nli.predicted_relation if global_nli else None
        p_ent = global_nli.p_entailment if global_nli else None
        p_neu = global_nli.p_neutral if global_nli else None
        p_con = global_nli.p_contradiction if global_nli else None

        return BenchmarkPredictionRecord(
            sample_id=sample.sample_id,
            dataset_name=self.name,
            split=split,
            task_id=t_id,
            ground_truth_raw=raw_label,
            ground_truth_mapped=ground_truth_mapped,
            predicted_verdict=predicted_verdict,
            is_correct=is_correct,
            confidence=round(confidence, 4),
            uncertainty_score=round(uncertainty, 4),
            cross_modal_score=round(cm_score_val, 4) if cm_score_val is not None else None,
            mismatch_detected=mismatch_flag,
            atomic_claims_count=len(propositions),
            nli_predicted_relation=nli_pred_rel,
            nli_p_entailment=round(p_ent, 4) if p_ent is not None else None,
            nli_p_neutral=round(p_neu, 4) if p_neu is not None else None,
            nli_p_contradiction=round(p_con, 4) if p_con is not None else None,
            evidence_source_mode="paired_document",
            execution_time_sec=0.0,
            error_message=None
        )
