"""
Claim-evidence verification module using pretrained DeBERTa-v3 NLI.
Estimates entailment, contradiction, and neutral relations between claims and evidence snippets,
and performs initial baseline multimodal decision fusion.
Note: NLI is an evidence-relation estimator, not an absolute truth oracle.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from src.schemas import CrossModalScore, DecisionVerdict, EvidenceSnippet, NLIResult
from src.utils import setup_logger

logger = setup_logger("verification")

# Lazy-loaded global NLI model singletons
_NLI_MODEL = None
_NLI_TOKENIZER = None
_NLI_DEVICE = "cpu"


def get_nli_model(
    model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    device: str = "cpu"
):
    """
    Returns or lazily loads the shared DeBERTa NLI model and tokenizer.
    """
    global _NLI_MODEL, _NLI_TOKENIZER, _NLI_DEVICE
    if _NLI_MODEL is not None:
        return _NLI_MODEL, _NLI_TOKENIZER, _NLI_DEVICE

    try:
        import os
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info(f"Loading NLI model: {model_name} on {device}...")
        try:
            # Check local cache first; set HF_HUB_OFFLINE temporarily for local loading to bypass network retries
            orig_offline = os.environ.get("HF_HUB_OFFLINE")
            orig_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            try:
                _NLI_TOKENIZER = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                _NLI_MODEL = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
            finally:
                if orig_offline is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = orig_offline
                if orig_tf_offline is None:
                    os.environ.pop("TRANSFORMERS_OFFLINE", None)
                else:
                    os.environ["TRANSFORMERS_OFFLINE"] = orig_tf_offline
        except Exception:
            _NLI_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            _NLI_MODEL = AutoModelForSequenceClassification.from_pretrained(model_name)

        _NLI_DEVICE = device
        _NLI_MODEL.to(device)
        _NLI_MODEL.eval()
        logger.info("[OK] NLI model loaded successfully.")
        return _NLI_MODEL, _NLI_TOKENIZER, _NLI_DEVICE
    except Exception as e:
        logger.warning(f"Failed to load NLI model ({model_name}): {e}. Deterministic relation fallback will be used.")
        return None, None, device


def estimate_nli_relation(
    claim_text: str,
    evidence_text: str,
    snippet_id: str = "ev_0",
    model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    device: str = "cpu"
) -> NLIResult:
    """
    Predicts NLI probabilities (entailment, neutral, contradiction) for a (claim, evidence) pair.
    """
    if not claim_text.strip() or not evidence_text.strip():
        return NLIResult(
            claim_text=claim_text,
            evidence_snippet_id=snippet_id,
            p_entailment=0.0,
            p_neutral=1.0,
            p_contradiction=0.0,
            predicted_relation="neutral"
        )

    model, tokenizer, dev = get_nli_model(model_name=model_name, device=device)

    if model is None or tokenizer is None:
        # Deterministic lexical heuristic fallback when model weights are not downloaded
        c_words = set(claim_text.lower().split())
        e_words = set(evidence_text.lower().split())
        overlap = len(c_words.intersection(e_words)) / max(1, len(c_words))
        
        # Check simple contradiction keywords
        contradict_keys = {"not", "false", "denies", "wrong", "different", "never", "unrelated"}
        has_negation = bool(c_words.intersection(contradict_keys) ^ e_words.intersection(contradict_keys))

        if has_negation and overlap > 0.3:
            p_e, p_n, p_c = 0.1, 0.2, 0.7
            rel = "contradiction"
        elif overlap > 0.5:
            p_e, p_n, p_c = 0.8, 0.15, 0.05
            rel = "entailment"
        else:
            p_e, p_n, p_c = 0.1, 0.8, 0.1
            rel = "neutral"

        return NLIResult(
            claim_text=claim_text,
            evidence_snippet_id=snippet_id,
            p_entailment=round(p_e, 4),
            p_neutral=round(p_n, 4),
            p_contradiction=round(p_c, 4),
            predicted_relation=rel
        )

    try:
        # Format as standard premise-hypothesis pair: [CLS] Evidence [SEP] Claim [SEP]
        inputs = tokenizer(
            evidence_text,
            claim_text,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

        # Model mapping for DeBERTa MNLI: 0=entailment, 1=neutral, 2=contradiction
        id2label = model.config.id2label if hasattr(model.config, "id2label") else {0: "entailment", 1: "neutral", 2: "contradiction"}
        
        # Normalize labels
        prob_dict = {}
        for idx, p in enumerate(probs):
            lbl = id2label.get(idx, str(idx)).lower()
            if "entail" in lbl:
                prob_dict["entailment"] = float(p)
            elif "contra" in lbl:
                prob_dict["contradiction"] = float(p)
            else:
                prob_dict["neutral"] = float(p)

        p_ent = prob_dict.get("entailment", 0.33)
        p_neu = prob_dict.get("neutral", 0.34)
        p_con = prob_dict.get("contradiction", 0.33)

        top_rel = max([("entailment", p_ent), ("neutral", p_neu), ("contradiction", p_con)], key=lambda x: x[1])[0]

        return NLIResult(
            claim_text=claim_text,
            evidence_snippet_id=snippet_id,
            p_entailment=round(p_ent, 4),
            p_neutral=round(p_neu, 4),
            p_contradiction=round(p_con, 4),
            predicted_relation=top_rel
        )
    except Exception as e:
        logger.error(f"Error in NLI prediction: {e}")
        return NLIResult(
            claim_text=claim_text,
            evidence_snippet_id=snippet_id,
            p_entailment=0.0,
            p_neutral=1.0,
            p_contradiction=0.0,
            predicted_relation="neutral"
        )


def verify_atomic_claims(
    atomic_claims: List["AtomicClaim"],
    retriever: "EvidenceRetriever",
    model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    device: str = "cpu",
    entailment_cutoff: float = 0.60,
    contradiction_cutoff: float = 0.50,
    force_mode: Optional[str] = None,
    top_k: int = 3
) -> List["AtomicVerificationResult"]:
    """
    Independently retrieves evidence and performs NLI verification for each atomic claim.
    Distinguishes:
      - SUPPORTED: Relevant evidence provides positive entailment >= cutoff
      - CONTRADICTED: Relevant evidence directly refutes the proposition >= cutoff
      - UNVERIFIED: Evidence is neutral, missing, or fails topical relevance cutoff (NOT assumed false)
    """
    from src.schemas import AtomicVerificationResult

    atomic_results: List[AtomicVerificationResult] = []

    for idx, claim in enumerate(atomic_claims):
        snippets = retriever.retrieve(claim.text, top_k=top_k, force_mode=force_mode)
        
        # Prioritize relevant snippets
        relevant_snippets = [s for s in snippets if s.is_relevant]
        best_ev = relevant_snippets[0] if relevant_snippets else (snippets[0] if snippets else None)

        # If no snippet exists OR the best snippet is irrelevant to the proposition
        if best_ev is None or not best_ev.is_relevant:
            reason = "No relevant evidence found for this claim in the retrieved sources."
            if best_ev and not best_ev.is_relevant:
                rel_str = f"{best_ev.relevance_score:.2f}" if best_ev.relevance_score is not None else "Unavailable"
                reason = f"No relevant evidence found for this claim in the retrieved sources (closest retrieved passage [{best_ev.snippet_id}] failed topical relevance threshold: score {rel_str} < {retriever.relevance_threshold:.2f})."
            
            atomic_results.append(
                AtomicVerificationResult(
                    claim_id=claim.claim_id,
                    atomic_text=claim.text,
                    status="UNVERIFIED",
                    retrieved_evidence=best_ev,
                    nli_result=None,
                    explanation=reason
                )
            )
            continue

        # Evaluate NLI across all relevant snippets for this atomic claim
        nli_evals = []
        for s in relevant_snippets:
            res = estimate_nli_relation(
                claim_text=claim.text,
                evidence_text=s.text,
                snippet_id=s.snippet_id,
                model_name=model_name,
                device=device,
            )
            nli_evals.append((s, res))

        # Check for conflicting evidence among relevant passages
        entailing_snippets = [(s, r) for s, r in nli_evals if r.p_entailment >= entailment_cutoff]
        contradicting_snippets = [(s, r) for s, r in nli_evals if r.p_contradiction >= contradiction_cutoff]

        if entailing_snippets and contradicting_snippets:
            status = "CONFLICTING"
            best_ev = entailing_snippets[0][0]
            nli_res = entailing_snippets[0][1]
            con_ev = contradicting_snippets[0][0]
            con_res = contradicting_snippets[0][1]
            explanation = (
                f"Conflicting evidence detected across retrieved sources: [{best_ev.snippet_id}] from {best_ev.source} supports the proposition "
                f"(p_entail: {nli_res.p_entailment:.2f}), but [{con_ev.snippet_id}] from {con_ev.source} refutes it "
                f"(p_contra: {con_res.p_contradiction:.2f})."
            )
        elif contradicting_snippets:
            best_ev, nli_res = max(contradicting_snippets, key=lambda x: x[1].p_contradiction)
            status = "CONTRADICTED"
            explanation = f"Contradicted by relevant evidence [{best_ev.snippet_id}] from {best_ev.source} (p_contra: {nli_res.p_contradiction:.2f})."
        elif entailing_snippets:
            best_ev, nli_res = max(entailing_snippets, key=lambda x: x[1].p_entailment)
            status = "SUPPORTED"
            explanation = f"Supported by relevant evidence [{best_ev.snippet_id}] from {best_ev.source} (p_entail: {nli_res.p_entailment:.2f})."
        else:
            best_ev, nli_res = nli_evals[0]
            status = "UNVERIFIED"
            explanation = f"Not established by evidence [{best_ev.snippet_id}] (neutral / insufficient relation, p_neutral: {nli_res.p_neutral:.2f})."

        atomic_results.append(
            AtomicVerificationResult(
                claim_id=claim.claim_id,
                atomic_text=claim.text,
                status=status,
                retrieved_evidence=best_ev,
                nli_result=nli_res,
                explanation=explanation,
            )
        )

    return atomic_results


def aggregate_atomic_verifications(
    atomic_verifications: List["AtomicVerificationResult"]
) -> Tuple[Optional[DecisionVerdict], List[str]]:
    """
    Explicit aggregation rules for atomic claim verifications:
      1. If ANY atomic claim has CONFLICTING evidence across credible sources -> CONFLICTING.
      2. If ANY atomic claim is CONTRADICTED by relevant evidence -> MISLEADING.
      3. If multiple atomic claims exist:
         - If SOME are SUPPORTED and SOME are UNVERIFIED (none contradicted) ->
           PARTIALLY_SUPPORTED ("Partially Supported / Some Details Unverified").
         - If ALL are SUPPORTED -> SUPPORTED.
         - If ALL are UNVERIFIED -> UNCERTAIN.
      4. If single atomic claim:
         - If CONFLICTING -> CONFLICTING.
         - If CONTRADICTED -> MISLEADING.
         - If SUPPORTED -> SUPPORTED.
         - If UNVERIFIED -> UNCERTAIN.
    """
    if not atomic_verifications:
        return None, []

    reasoning: List[str] = []
    num_claims = len(atomic_verifications)
    num_conflicting = sum(1 for a in atomic_verifications if a.status == "CONFLICTING")
    num_supported = sum(1 for a in atomic_verifications if a.status == "SUPPORTED")
    num_contradicted = sum(1 for a in atomic_verifications if a.status == "CONTRADICTED")
    num_unverified = sum(1 for a in atomic_verifications if a.status == "UNVERIFIED")

    # Rule 1: Conflicting evidence takes precedence when sources materially disagree
    if num_conflicting > 0:
        reasoning.append(
            f"Credible sources materially disagree on {num_conflicting}/{num_claims} proposition(s) (conflicting evidence detected)."
        )
        return DecisionVerdict.CONFLICTING, reasoning

    # Rule 2: Contradiction takes precedence for factual falsification
    if num_contradicted > 0:
        reasoning.append(f"At least one atomic proposition was refuted by relevant external evidence ({num_contradicted}/{num_claims} contradicted).")
        return DecisionVerdict.MISLEADING, reasoning

    # Rule 3: Multi-claim mixture (Supported + Unverified)
    if num_claims > 1 and num_supported > 0 and num_unverified > 0:
        reasoning.append(
            f"Core factual assertions are corroborated ({num_supported}/{num_claims} supported), "
            f"but secondary details remain unverified by available evidence ({num_unverified}/{num_claims} unverified)."
        )
        return DecisionVerdict.PARTIALLY_SUPPORTED, reasoning

    # Rule 4: All supported
    if num_supported == num_claims:
        reasoning.append(f"All atomic factual assertions are substantiated by retrieved evidence ({num_supported}/{num_claims} supported).")
        return DecisionVerdict.SUPPORTED, reasoning

    # Rule 5: All unverified
    if num_unverified == num_claims:
        reasoning.append("No relevant evidence found for this claim in the retrieved sources.")
        return DecisionVerdict.UNCERTAIN, reasoning

    return None, reasoning


def fuse_multimodal_decision(
    cross_modal: CrossModalScore,
    nli_results: List[NLIResult],
    atomic_verifications: Optional[List["AtomicVerificationResult"]] = None,
    mismatch_threshold: float = 0.28,
    entailment_cutoff: float = 0.60,
    contradiction_cutoff: float = 0.50,
    atomic_results: Optional[List["AtomicVerificationResult"]] = None
) -> Tuple[DecisionVerdict, List[str]]:
    """
    Multimodal decision fusion combining cross-modal alignment and evidence NLI relations.
    Mandatory Rules:
      - Low OpenCLIP alignment alone NEVER forces an overall MISLEADING verdict if factual claims are supported.
      - A visual discrepancy is reported as an explainable factor (e.g. out-of-context or repurposed visual), yielding PARTIALLY_SUPPORTED.
      - If claims are contradicted by relevant evidence -> MISLEADING (visual match never overrides factual falsification).
      - If credible sources materially disagree -> CONFLICTING.
      - If claims are unverified or evidence is missing -> UNCERTAIN (visual match never manufactures factual truth).
      - In text-only mode (no image), visual checks are bypassed honestly without fabricating visual scores.
    """
    if atomic_verifications is None and atomic_results is not None:
        atomic_verifications = atomic_results

    reasoning_factors: List[str] = []
    has_image = getattr(cross_modal, "has_image", True)
    cm_status = getattr(cross_modal, "status", "evaluated")

    # 1. Visual-Text Cross-Modal Consistency Signals
    if not has_image or cm_status == "text_only":
        reasoning_factors.append("Text-only evaluation mode: visual alignment verification was bypassed.")
    elif cm_status == "error":
        err_msg = cross_modal.details.get("error", "Image validation failed")
        reasoning_factors.append(f"Visual asset error: {err_msg}. Visual verification could not be completed.")
    elif cm_status == "unavailable":
        reasoning_factors.append("Visual verification model offline: visual alignment evaluation was unavailable.")
    elif cross_modal.mismatch_detected:
        score_val = f"{cross_modal.similarity_score:.2f}" if cross_modal.similarity_score is not None else "N/A"
        reasoning_factors.append(
            f"Visual-text semantic mismatch detected (OpenCLIP score: {score_val} < cutoff {mismatch_threshold:.2f}). "
            "Image appears out-of-context or repurposed."
        )
    else:
        score_val = f"{cross_modal.similarity_score:.2f}" if cross_modal.similarity_score is not None else "N/A"
        reasoning_factors.append(f"Image and text caption exhibit consistent visual semantics (OpenCLIP score: {score_val})")

    # 2. Granular Atomic Verification Aggregation (if available)
    if atomic_verifications:
        atomic_verdict, atomic_reasons = aggregate_atomic_verifications(atomic_verifications)
        reasoning_factors.extend(atomic_reasons)
        if atomic_verdict is not None:
            if atomic_verdict == DecisionVerdict.CONFLICTING:
                return DecisionVerdict.CONFLICTING, reasoning_factors
            if atomic_verdict == DecisionVerdict.MISLEADING:
                # Contradiction in evidence takes precedence over any visual match (Case C & Case F)
                if has_image and not cross_modal.mismatch_detected and cm_status == "evaluated":
                    reasoning_factors.append("Visual asset aligns with topic, but factual proposition is refuted by evidence.")
                return DecisionVerdict.MISLEADING, reasoning_factors
            if atomic_verdict == DecisionVerdict.SUPPORTED:
                # If factual claim was fully supported, but visual was mismatched, it represents an out-of-context pairing (Case B & Case E)
                if has_image and cross_modal.mismatch_detected:
                    reasoning_factors.append(
                        "Textual assertions are corroborated by verified evidence, but visual asset exhibits semantic discrepancy (flagged as out-of-context / repurposed visual)."
                    )
                    return DecisionVerdict.PARTIALLY_SUPPORTED, reasoning_factors
                return DecisionVerdict.SUPPORTED, reasoning_factors
            elif atomic_verdict == DecisionVerdict.PARTIALLY_SUPPORTED:
                if has_image and cross_modal.mismatch_detected:
                    reasoning_factors.append("Visual mismatch reinforces uncertainty in peripheral claims.")
                return DecisionVerdict.PARTIALLY_SUPPORTED, reasoning_factors
            elif atomic_verdict == DecisionVerdict.UNCERTAIN:
                # Unverified claims remain uncertain (Case H)
                return DecisionVerdict.UNCERTAIN, reasoning_factors

            return atomic_verdict, reasoning_factors

    # 3. Fallback Document-Level Aggregate NLI Signals
    max_contradiction = max([n.p_contradiction for n in nli_results], default=0.0)
    max_entailment = max([n.p_entailment for n in nli_results], default=0.0)
    has_evidence = len(nli_results) > 0

    # Conflict check across passages
    has_strong_entailment = any(n.p_entailment >= entailment_cutoff for n in nli_results)
    has_strong_contradiction = any(n.p_contradiction >= contradiction_cutoff for n in nli_results)
    if has_strong_entailment and has_strong_contradiction:
        reasoning_factors.append(
            f"Conflicting evidence detected across retrieved sources: credible reports materially disagree (entailment: {max_entailment:.2f}, contradiction: {max_contradiction:.2f})."
        )
        return DecisionVerdict.CONFLICTING, reasoning_factors

    if max_contradiction >= contradiction_cutoff:
        reasoning_factors.append(f"Retrieved evidence refutes the claim proposition (contradiction: {max_contradiction:.2f})")
        return DecisionVerdict.MISLEADING, reasoning_factors

    if max_entailment >= entailment_cutoff:
        if has_image and cross_modal.mismatch_detected:
            reasoning_factors.append(
                "Textual claim is corroborated by evidence, but accompanying image exhibits low visual alignment (flagged as out-of-context visual)."
            )
            return DecisionVerdict.PARTIALLY_SUPPORTED, reasoning_factors
        reasoning_factors.append(f"Retrieved verified evidence supports the claim (entailment: {max_entailment:.2f})")
        return DecisionVerdict.SUPPORTED, reasoning_factors

    if not has_evidence or all(n.predicted_relation == "neutral" for n in nli_results):
        reasoning_factors.append("No relevant evidence found for this claim in the retrieved sources.")
        return DecisionVerdict.UNCERTAIN, reasoning_factors

    return DecisionVerdict.UNCERTAIN, reasoning_factors
