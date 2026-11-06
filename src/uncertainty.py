"""
Uncertainty and confidence estimation module.
Implements a baseline predictive uncertainty score combining cross-modal margins and NLI entropy/margin.
Note: This is a normalized predictive confidence heuristic; full conformal calibration will be integrated in later phases.
"""

from typing import List, Tuple
from src.schemas import CrossModalScore, DecisionVerdict, NLIResult


def compute_decision_confidence_and_uncertainty(
    decision: DecisionVerdict,
    cross_modal: CrossModalScore,
    nli_results: List[NLIResult],
    mismatch_threshold: float = 0.28
) -> Tuple[float, float]:
    """
    Computes (confidence, uncertainty_score) in range [0.0, 1.0].
    Confidence + Uncertainty = 1.0.
    Handles text-only mode and missing visual scores honestly without fabricating visual margins.
    """
    if decision == DecisionVerdict.UNCERTAIN:
        # Default high uncertainty for ambiguous cases
        return 0.40, 0.60

    if decision == DecisionVerdict.CONFLICTING:
        # High uncertainty due to material disagreement between credible sources
        return 0.45, 0.55

    if decision == DecisionVerdict.PARTIALLY_SUPPORTED:
        # Balanced confidence representing verified core but unverified periphery or visual mismatch
        return 0.65, 0.35

    # Cross-modal margin (only if an image was actually evaluated)
    has_image = getattr(cross_modal, "has_image", True)
    cm_score = cross_modal.similarity_score
    if has_image and cm_score is not None:
        cm_margin = abs(cm_score - mismatch_threshold) / max(1e-5, max(mismatch_threshold, 1.0 - mismatch_threshold))
        cm_margin = min(1.0, cm_margin)
    else:
        cm_margin = 0.0

    # Evidence NLI margin
    if nli_results:
        top_nli = max(nli_results, key=lambda n: max(n.p_entailment, n.p_contradiction))
        nli_top_prob = max(top_nli.p_entailment, top_nli.p_contradiction)
        nli_margin = max(0.0, min(1.0, nli_top_prob - top_nli.p_neutral))
    else:
        nli_margin = 0.0

    # Blended confidence estimate [0.5 - 0.95]
    if has_image and cm_score is not None:
        raw_confidence = 0.50 + 0.25 * cm_margin + 0.25 * nli_margin
    else:
        # Text-only or visual unavailable: confidence derived directly from evidence verification margin
        raw_confidence = 0.50 + 0.40 * nli_margin

    confidence = round(max(0.0, min(1.0, raw_confidence)), 4)
    uncertainty_score = round(1.0 - confidence, 4)

    return confidence, uncertainty_score
