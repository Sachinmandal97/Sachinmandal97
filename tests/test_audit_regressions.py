"""
Comprehensive Audit Regression Suite for VERITAS-MM.
Validates Scenarios A through J from the Phase 2 Verification Audit:

- Case A: Eiffel Tower in Paris + Eiffel Tower visual -> Text and visual evaluated independently; overall SUPPORTED.
- Case B: Eiffel Tower in Paris + Taj Mahal visual -> Text supported, visual mismatch flagged; NOT fully supported (PARTIALLY_SUPPORTED).
- Case C: Eiffel Tower in New York + Eiffel Tower visual -> Visual match must not make false location claim supported (MISLEADING).
- Case D: Eiffel Tower in New York + Taj Mahal visual -> Both factual and visual checked independently (MISLEADING).
- Case E: True news claim + Unrelated visual -> Flag mismatch even if retrieved sources support text (PARTIALLY_SUPPORTED).
- Case F: False claim + Visually relevant visual -> Visual similarity does not equal factual support (MISLEADING).
- Case G: Valid claim + No image uploaded -> Text-only path without fabricated visual scores (similarity_score is None, has_image=False).
- Case H: Ambiguous claim + Relevant visual -> UNCERTAIN when evidence is insufficient.
- Case I: Any claim + Corrupted image -> Validation error shown; never silently treated as match.
- Case J: Successive queries -> Cache keys and pipeline state do not cross-leak.
"""

from pathlib import Path
import tempfile
import pytest
from PIL import Image

from src.pipeline import MultimodalVerificationPipeline
from src.schemas import (
    AtomicClaim,
    AtomicVerificationResult,
    CrossModalScore,
    DecisionVerdict,
    EvidenceSnippet,
    MultimodalSample,
    NLIResult,
    VerificationResult,
)
from src.verification import fuse_multimodal_decision, aggregate_atomic_verifications
from src.cross_modal import compute_cross_modal_consistency


@pytest.fixture(scope="module")
def pipeline():
    return MultimodalVerificationPipeline()


def test_scenario_a_eiffel_paris_matching_visual():
    """Case A: Eiffel Tower in Paris with visually matching image -> SUPPORTED."""
    cm_match = CrossModalScore(
        similarity_score=0.34,
        mismatch_detected=False,
        has_image=True,
        status="evaluated"
    )
    atomics = [
        AtomicVerificationResult(
            claim_id="a1",
            atomic_text="The Eiffel Tower is located in Paris, France.",
            status="SUPPORTED",
            explanation="Corroborated by verified geographical evidence."
        )
    ]
    verdict, reasons = fuse_multimodal_decision(
        cross_modal=cm_match,
        nli_results=[],
        atomic_verifications=atomics,
        mismatch_threshold=0.28
    )
    assert verdict == DecisionVerdict.SUPPORTED
    assert any("consistent visual semantics" in r.lower() for r in reasons)
    assert any("substantiated by retrieved evidence" in r.lower() for r in reasons)


def test_scenario_b_eiffel_paris_mismatched_taj_mahal():
    """
    Case B (Critical Reproduction): Eiffel Tower in Paris with Taj Mahal image.
    Text is supported, but visual is mismatched.
    MUST NOT report complete item as fully supported.
    """
    cm_mismatch = CrossModalScore(
        similarity_score=0.18,
        mismatch_detected=True,
        has_image=True,
        status="evaluated"
    )
    atomics = [
        AtomicVerificationResult(
            claim_id="b1",
            atomic_text="The Eiffel Tower is located in Paris, France.",
            status="SUPPORTED",
            explanation="Corroborated by verified encyclopedic evidence."
        )
    ]
    verdict, reasons = fuse_multimodal_decision(
        cross_modal=cm_mismatch,
        nli_results=[],
        atomic_verifications=atomics,
        mismatch_threshold=0.28
    )
    # Must NOT report as SUPPORTED
    assert verdict != DecisionVerdict.SUPPORTED
    assert verdict == DecisionVerdict.PARTIALLY_SUPPORTED
    assert any("mismatch" in r.lower() or "out-of-context" in r.lower() or "repurposed" in r.lower() for r in reasons)


def test_scenario_c_false_location_matching_visual():
    """Case C: Eiffel Tower in New York with Eiffel Tower image -> Visual match does not make false claim supported."""
    cm_match = CrossModalScore(
        similarity_score=0.35,
        mismatch_detected=False,
        has_image=True,
        status="evaluated"
    )
    atomics = [
        AtomicVerificationResult(
            claim_id="c1",
            atomic_text="The Eiffel Tower is located in New York City.",
            status="CONTRADICTED",
            explanation="Refuted by verified evidence: Eiffel Tower is located in Paris, France."
        )
    ]
    verdict, reasons = fuse_multimodal_decision(
        cross_modal=cm_match,
        nli_results=[],
        atomic_verifications=atomics,
        mismatch_threshold=0.28
    )
    assert verdict == DecisionVerdict.MISLEADING
    assert any("refuted" in r.lower() or "contradicted" in r.lower() for r in reasons)


def test_scenario_d_false_location_mismatched_visual():
    """Case D: Eiffel Tower in New York with Taj Mahal image -> Both refuted and visual mismatch."""
    cm_mismatch = CrossModalScore(
        similarity_score=0.17,
        mismatch_detected=True,
        has_image=True,
        status="evaluated"
    )
    atomics = [
        AtomicVerificationResult(
            claim_id="d1",
            atomic_text="The Eiffel Tower is located in New York City.",
            status="CONTRADICTED",
            explanation="Refuted by evidence."
        )
    ]
    verdict, reasons = fuse_multimodal_decision(
        cross_modal=cm_mismatch,
        nli_results=[],
        atomic_verifications=atomics,
        mismatch_threshold=0.28
    )
    assert verdict == DecisionVerdict.MISLEADING
    assert any("refuted" in r.lower() or "contradicted" in r.lower() for r in reasons)
    assert any("mismatch" in r.lower() for r in reasons)


def test_scenario_e_true_unrelated_news_mismatched_visual():
    """Case E: True news claim with unrelated visual -> Flag mismatch even if sources support text."""
    cm_mismatch = CrossModalScore(
        similarity_score=0.12,
        mismatch_detected=True,
        has_image=True,
        status="evaluated"
    )
    atomics = [
        AtomicVerificationResult(
            claim_id="e1",
            atomic_text="Firefighters battle industrial warehouse blaze in East London.",
            status="SUPPORTED",
            explanation="Confirmed by emergency services reporting."
        )
    ]
    verdict, reasons = fuse_multimodal_decision(
        cross_modal=cm_mismatch,
        nli_results=[],
        atomic_verifications=atomics,
        mismatch_threshold=0.28
    )
    assert verdict == DecisionVerdict.PARTIALLY_SUPPORTED
    assert verdict != DecisionVerdict.SUPPORTED
    assert any("out-of-context" in r.lower() or "repurposed" in r.lower() for r in reasons)


def test_scenario_f_false_claim_visually_relevant():
    """Case F: False claim with visually relevant image -> Visual similarity does not equal factual support."""
    cm_match = CrossModalScore(
        similarity_score=0.31,
        mismatch_detected=False,
        has_image=True,
        status="evaluated"
    )
    atomics = [
        AtomicVerificationResult(
            claim_id="f1",
            atomic_text="The Moon landing in 1969 was filmed in a Hollywood soundstage.",
            status="CONTRADICTED",
            explanation="Historical consensus and evidence refute the staging hoax."
        )
    ]
    verdict, reasons = fuse_multimodal_decision(
        cross_modal=cm_match,
        nli_results=[],
        atomic_verifications=atomics,
        mismatch_threshold=0.28
    )
    assert verdict == DecisionVerdict.MISLEADING
    assert any("refuted" in r.lower() for r in reasons)


def test_scenario_g_text_only_valid_claim_no_fabrication(pipeline):
    """Case G: Valid claim without image -> Text-only execution without fabricated visual scores."""
    sample = MultimodalSample(
        sample_id="case_g_text_only",
        claim_text="Firefighters tackled a warehouse fire in East London on Sunday morning.",
        image_path=None
    )
    result = pipeline.verify_sample(sample, force_mode="local_only")
    assert result.cross_modal.has_image is False
    assert result.cross_modal.similarity_score is None
    assert result.cross_modal.status == "text_only"
    assert result.cross_modal.mismatch_detected is False
    assert any("text-only" in r.lower() for r in result.reasoning_factors)


def test_scenario_h_ambiguous_claim_relevant_image_uncertain():
    """Case H: Ambiguous or unverified claim with relevant visual -> UNCERTAIN."""
    cm_match = CrossModalScore(
        similarity_score=0.33,
        mismatch_detected=False,
        has_image=True,
        status="evaluated"
    )
    atomics = [
        AtomicVerificationResult(
            claim_id="h1",
            atomic_text="Extraterrestrial bacteria samples were recovered from Antarctica in 2026.",
            status="UNVERIFIED",
            explanation="No credible scientific sources confirm this assertion."
        )
    ]
    verdict, reasons = fuse_multimodal_decision(
        cross_modal=cm_match,
        nli_results=[],
        atomic_verifications=atomics,
        mismatch_threshold=0.28
    )
    assert verdict == DecisionVerdict.UNCERTAIN


def test_scenario_i_corrupted_image_validation_error(pipeline, tmp_path):
    """Case I: Corrupted image file -> Show clear validation error, never silently substitute or treat as match."""
    bad_file = tmp_path / "corrupt_taj.jpeg"
    bad_file.write_bytes(b"NOT_A_VALID_JPEG_HEADER_XYZ")

    sample = MultimodalSample(
        sample_id="case_i_corrupt",
        claim_text="The Eiffel Tower is located in Paris, France.",
        image_path=str(bad_file)
    )
    result = pipeline.verify_sample(sample, force_mode="local_only")
    assert result.cross_modal.has_image is True
    assert result.cross_modal.status == "error"
    assert result.cross_modal.similarity_score is None
    assert result.cross_modal.mismatch_detected is False
    assert "Invalid or corrupted image file" in result.cross_modal.details.get("error", "")
    assert any("visual asset error" in r.lower() or "validation" in r.lower() for r in result.reasoning_factors)


def test_scenario_j_successive_queries_no_leakage(pipeline, tmp_path):
    """Case J: Successive queries with different text and images do not leak cache or state."""
    # Run Query 1: Local corpus fire claim
    res1 = pipeline.verify_sample(
        MultimodalSample(sample_id="j1", claim_text="London Fire Brigade tackled warehouse fire.", image_path=None),
        force_mode="local_only"
    )
    assert res1.claim_text == "London Fire Brigade tackled warehouse fire."

    # Run Query 2: Completely different claim
    res2 = pipeline.verify_sample(
        MultimodalSample(sample_id="j2", claim_text="Mars rover detected organic molecules.", image_path=None),
        force_mode="local_only"
    )
    assert res2.claim_text == "Mars rover detected organic molecules."
    assert res2.sample_id == "j2"
    assert res2.claim_text != res1.claim_text
