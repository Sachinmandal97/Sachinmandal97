"""
Explainability and attribution module.
Generates deterministic structured explanations highlighting cross-modal findings,
OCR signals, evidence citations, and primary decision factors without LLM hallucinations.
"""

from typing import List, Optional
from src.schemas import (
    AtomicVerificationResult,
    CrossModalScore,
    DecisionVerdict,
    EvidenceSnippet,
    NLIResult,
    OCRResult,
)


def generate_structured_explanation(
    claim_text: str,
    decision: DecisionVerdict,
    confidence: float,
    uncertainty_score: float,
    ocr: OCRResult,
    cross_modal: CrossModalScore,
    evidence: List[EvidenceSnippet],
    nli_results: List[NLIResult],
    reasoning_factors: List[str],
    atomic_verifications: Optional[List[AtomicVerificationResult]] = None,
) -> str:
    """
    Synthesizes a transparent, formatted explanation report.
    Includes granular per-atomic claim verification status and evidence grounding.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("AI-ASSISTED VERIFICATION REPORT")
    lines.append("=" * 60)
    lines.append(f"CLAIM: \"{claim_text}\"")
    lines.append(f"DECISION: {decision.value.upper()} (Confidence: {confidence * 100:.1f}%, Uncertainty: {uncertainty_score * 100:.1f}%)")
    lines.append("")
    
    # 1. Key Factors
    lines.append("PRIMARY DECISION FACTORS:")
    for factor in reasoning_factors:
        lines.append(f"  - {factor}")
    lines.append("")

    # 2. Granular Atomic Proposition Verification
    if atomic_verifications:
        lines.append("GRANULAR ATOMIC CLAIM BREAKDOWN:")
        for idx, at in enumerate(atomic_verifications, 1):
            stat_icon = "✅" if at.status == "SUPPORTED" else ("🚨" if at.status == "CONTRADICTED" else ("⚔️" if at.status == "CONFLICTING" else "ℹ️"))
            lines.append(f"  [{idx}] {stat_icon} Proposition: \"{at.atomic_text}\"")
            lines.append(f"      Status: {at.status}")
            if at.nli_result:
                lines.append(
                    f"      NLI Scores: Entailment={at.nli_result.p_entailment:.2f}, "
                    f"Neutral={at.nli_result.p_neutral:.2f}, "
                    f"Contradiction={at.nli_result.p_contradiction:.2f}"
                )
            if at.retrieved_evidence:
                rel_val = f"{at.retrieved_evidence.relevance_score:.2f}" if at.retrieved_evidence.relevance_score is not None else "Unavailable"
                rel_note = f" (Relevance: {rel_val})"
                lines.append(f"      Retrieved Evidence [{at.retrieved_evidence.snippet_id}]: \"{at.retrieved_evidence.text}\"{rel_note}")
                if at.retrieved_evidence.search_query:
                    lines.append(f"      Search Query: \"{at.retrieved_evidence.search_query}\"")
                if at.retrieved_evidence.url:
                    lines.append(f"      Source URL: {at.retrieved_evidence.url}")
                if at.retrieved_evidence.publisher_domain:
                    lines.append(f"      Publisher Domain: {at.retrieved_evidence.publisher_domain}")
                if at.retrieved_evidence.wrapper_url:
                    lines.append(f"      Wrapper URL: {at.retrieved_evidence.wrapper_url}")
                lines.append(f"      Content Type: {at.retrieved_evidence.content_type.upper()} | Access Status: {at.retrieved_evidence.access_status.upper()}")
            else:
                lines.append("      Retrieved Evidence: None (not located in knowledge base or live search)")
            lines.append(f"      Finding: {at.explanation}")
        lines.append("")

    # 3. Multimodal & OCR Analysis
    lines.append("CROSS-MODAL ANALYSIS:")
    if not getattr(cross_modal, "has_image", True) or cross_modal.similarity_score is None:
        if getattr(cross_modal, "status", "") == "error":
            err_str = cross_modal.details.get("error", "Corrupted or invalid image")
            lines.append(f"  - Image Error: {err_str} (Visual alignment could not be computed)")
        elif getattr(cross_modal, "status", "") == "unavailable":
            lines.append("  - Visual verification model offline: alignment evaluation unavailable.")
        else:
            lines.append("  - Text-Only Mode: No image provided; visual alignment evaluation bypassed.")
    else:
        lines.append(f"  - Image-Text Similarity Score: {cross_modal.similarity_score:.4f} (Mismatch Flagged: {cross_modal.mismatch_detected})")
        if ocr.has_text:
            lines.append(f"  - OCR Detected In Image: \"{ocr.extracted_text}\"")
            if cross_modal.ocr_text_overlap > 0:
                lines.append(f"  - OCR-Caption Word Overlap: {cross_modal.ocr_text_overlap * 100:.1f}%")
        else:
            lines.append("  - OCR: No significant text detected in image.")
    lines.append("")

    # 4. Evidence Grounding & Overall NLI Relations
    lines.append("RETRIEVED EVIDENCE & RELATION ESTIMATIONS:")
    relevant_count = sum(1 for s in evidence if s.is_relevant)
    if not evidence or relevant_count == 0:
        lines.append("  - No relevant evidence found for this claim in the retrieved sources.")
    else:
        nli_map = {n.evidence_snippet_id: n for n in nli_results}
        for idx, snip in enumerate(evidence, 1):
            nli_entry = nli_map.get(snip.snippet_id)
            rel_str = f"Relation: {nli_entry.predicted_relation.upper()} (entail: {nli_entry.p_entailment:.2f}, contra: {nli_entry.p_contradiction:.2f})" if nli_entry else "Relation: Unscored / Excluded"
            relevance_flag = "RELEVANT" if snip.is_relevant else "WEAK / EXCLUDED"
            score_disp = f"{snip.relevance_score:.2f}" if snip.relevance_score is not None else "Unavailable"
            lines.append(f"  [{idx}] Source: {snip.source} [{snip.source_type.upper()}] — {relevance_flag} (Relevance: {score_disp})")
            if snip.search_query:
                lines.append(f"      Query: \"{snip.search_query}\"")
            if snip.url:
                lines.append(f"      URL: {snip.url}")
            if snip.wrapper_url:
                lines.append(f"      Wrapper URL: {snip.wrapper_url}")
            lines.append(f"      Content Type: {snip.content_type.upper()} | Access Status: {snip.access_status.upper()}")
            lines.append(f"      Text: \"{snip.text}\"")
            if snip.relevance_reason:
                lines.append(f"      Relevance Reason: {snip.relevance_reason}")
            lines.append(f"      {rel_str}")

    lines.append("=" * 60)
    return "\n".join(lines)
