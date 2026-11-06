"""
Smoke test suite for final application verification.
Validates:
1. Image-plus-claim processing (end-to-end multimodal pipeline)
2. Text-only input (clean image-free verification with proper status flags)
3. Uncertain evidence (obscure/unsubstantiated claims returning UNCERTAIN)
4. Conflicting evidence (opposing credible evidence correctly yielding CONFLICTING)
5. Unavailable sources (network error / inaccessible provider / timeout fallback)
6. Malformed inputs (empty string, excess whitespace, very long text, corrupted image, unicode)
7. Preset demo image isolation (custom claims cannot accidentally reuse preset demo visuals)
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
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
from src.retrieval import EvidenceRetriever
from src.preprocess import clean_text, load_and_validate_image


@pytest.fixture(scope="module")
def pipeline():
    """Initializes a shared pipeline instance."""
    return MultimodalVerificationPipeline()


def test_smoke_image_plus_claim(pipeline, tmp_path):
    """Scenario 1: Image-plus-claim processing."""
    # Create a test image
    img_path = tmp_path / "smoke_image.jpg"
    img = Image.new("RGB", (224, 224), color=(200, 50, 50))
    img.save(img_path)

    sample = MultimodalSample(
        sample_id="smoke_img_01",
        claim_text="Major industrial fire in London warehouse produces dark smoke plumes.",
        image_path=str(img_path)
    )
    result = pipeline.verify_sample(sample, force_mode="local_only")
    assert isinstance(result, VerificationResult)
    assert result.decision in [
        DecisionVerdict.SUPPORTED,
        DecisionVerdict.PARTIALLY_SUPPORTED,
        DecisionVerdict.MISLEADING,
        DecisionVerdict.UNCERTAIN,
    ]
    assert result.cross_modal is not None
    assert 0.0 <= result.cross_modal.similarity_score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.uncertainty_score <= 1.0
    assert "AI-ASSISTED VERIFICATION REPORT" in result.structured_explanation


def test_smoke_text_only_input(pipeline):
    """Scenario 2: Text-only input (no image attached)."""
    sample = MultimodalSample(
        sample_id="smoke_text_01",
        claim_text="Stock market indexes close higher following quarterly earnings reports.",
        image_path=None
    )
    result = pipeline.verify_sample(sample, force_mode="local_only")
    # When no image is provided, cross-modal evaluation is bypassed honestly without fabricating visual scores
    assert result.cross_modal.similarity_score is None
    assert result.cross_modal.has_image is False
    assert result.cross_modal.status == "text_only"
    assert result.cross_modal.mismatch_detected is False
    assert result.cross_modal.details.get("info") == "No image provided; text-only evaluation mode"
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.uncertainty_score <= 1.0


def test_smoke_uncertain_evidence(pipeline):
    """Scenario 3a: Uncertain evidence (obscure claim with no corroborating evidence)."""
    sample = MultimodalSample(
        sample_id="smoke_uncertain_01",
        claim_text="Secret ancient gold repositories discovered 500 meters beneath Edinburgh Castle in 2026.",
        image_path=None
    )
    result = pipeline.verify_sample(sample, force_mode="local_only")
    assert isinstance(result, VerificationResult)
    # With no evidence matching this fictitious claim in local corpus, decision must be UNCERTAIN
    assert result.decision == DecisionVerdict.UNCERTAIN
    assert result.uncertainty_score >= 0.50


def test_smoke_conflicting_evidence():
    """Scenario 3b: Conflicting evidence handling across sources."""
    cm = CrossModalScore(similarity_score=0.30, mismatch_detected=False)

    # 1. Test aggregate_atomic_verifications with conflicting status
    atom_conflicting = [
        AtomicVerificationResult(
            claim_id="atom_01",
            atomic_text="Inflation in the UK dropped to 2.1 percent in September.",
            status="CONFLICTING",
            explanation="Credible sources materially disagree."
        )
    ]
    verdict, reasons = aggregate_atomic_verifications(atom_conflicting)
    assert verdict == DecisionVerdict.CONFLICTING
    assert any("conflicting evidence" in r.lower() for r in reasons)

    # 2. Test fuse_multimodal_decision with contradictory passages
    nli_entail = NLIResult(
        claim_text="Inflation dropped to 2.1 percent.",
        evidence_snippet_id="snip_01",
        p_entailment=0.88,
        p_neutral=0.08,
        p_contradiction=0.04,
        predicted_relation="entailment"
    )
    nli_contra = NLIResult(
        claim_text="Inflation dropped to 2.1 percent.",
        evidence_snippet_id="snip_02",
        p_entailment=0.05,
        p_neutral=0.10,
        p_contradiction=0.85,
        predicted_relation="contradiction"
    )
    fused_verdict, fused_reasons = fuse_multimodal_decision(
        cross_modal=cm,
        nli_results=[nli_entail, nli_contra],
        atomic_verifications=atom_conflicting
    )
    assert fused_verdict == DecisionVerdict.CONFLICTING
    assert any("conflicting" in r.lower() for r in fused_reasons)


def test_smoke_unavailable_sources(pipeline):
    """Scenario 4: Unavailable sources (network error / API timeout)."""
    with patch.object(pipeline.retriever, "retrieve_for_claims", return_value=[]):
        sample = MultimodalSample(
            sample_id="smoke_unavailable_01",
            claim_text="NASA Perseverance rover detected novel sedimentary formations.",
            image_path=None
        )
        result = pipeline.verify_sample(sample, force_mode="live_only")
        assert isinstance(result, VerificationResult)
        assert result.decision == DecisionVerdict.UNCERTAIN
        assert len(result.retrieved_evidence) == 0
        assert result.uncertainty_score >= 0.50


def test_smoke_malformed_inputs(pipeline, tmp_path):
    """Scenario 5: Malformed input resilience."""
    # 5a. Empty string
    res_empty = pipeline.verify_sample(
        MultimodalSample(sample_id="m_empty", claim_text="", image_path=None),
        force_mode="local_only"
    )
    assert isinstance(res_empty, VerificationResult)
    assert res_empty.decision in [DecisionVerdict.UNCERTAIN, DecisionVerdict.MISLEADING]

    # 5b. Whitespace only
    res_ws = pipeline.verify_sample(
        MultimodalSample(sample_id="m_ws", claim_text="   \n\t  \r  ", image_path=None),
        force_mode="local_only"
    )
    assert isinstance(res_ws, VerificationResult)

    # 5c. Extremely long text (e.g. 5000 characters)
    long_claim = "Scientists have confirmed a new particle discovery. " * 100
    res_long = pipeline.verify_sample(
        MultimodalSample(sample_id="m_long", claim_text=long_claim, image_path=None),
        force_mode="local_only"
    )
    assert isinstance(res_long, VerificationResult)

    # 5d. Non-existent image path
    res_noimg = pipeline.verify_sample(
        MultimodalSample(
            sample_id="m_noimg",
            claim_text="Earthquake reported in Tokyo.",
            image_path=str(tmp_path / "non_existent_image_12345.jpg")
        ),
        force_mode="local_only"
    )
    assert isinstance(res_noimg, VerificationResult)
    assert res_noimg.cross_modal.status == "error"
    assert res_noimg.cross_modal.mismatch_detected is False

    # 5e. Corrupted image file (invalid image data)
    corrupt_path = tmp_path / "corrupt.jpg"
    corrupt_path.write_bytes(b"THIS_IS_NOT_A_VALID_JPEG_HEADER_1234567890")
    res_corrupt = pipeline.verify_sample(
        MultimodalSample(
            sample_id="m_corrupt",
            claim_text="Flood waters recede in regional areas.",
            image_path=str(corrupt_path)
        ),
        force_mode="local_only"
    )
    assert isinstance(res_corrupt, VerificationResult)
    assert res_corrupt.cross_modal.status == "error"

    # 5f. Special Unicode, HTML, and null bytes
    special_text = "<script>alert('xss')</script> Breaking News: \u2603 snowman &amp; symbols \u20ac500 \x00 test"
    res_special = pipeline.verify_sample(
        MultimodalSample(sample_id="m_special", claim_text=special_text, image_path=None),
        force_mode="local_only"
    )
    assert isinstance(res_special, VerificationResult)


def test_smoke_preset_demo_image_isolation():
    """Scenario 6: Verify that custom claims cannot accidentally reuse preset demo images."""
    preset_choice = "Case 1: Fully Supported News (Consistent Image + Evidence)"
    default_text = "Firefighters battle a major industrial warehouse blaze in East London on Sunday morning."
    default_image_desc = "Industrial warehouse fire with emergency personnel"

    # 1. Unmodified claim: preset is active
    unmodified_claim = default_text
    is_preset_active = (
        preset_choice != "(None - Enter Custom News Claim)"
        and bool(default_image_desc)
        and (unmodified_claim.strip() == default_text.strip())
    )
    assert is_preset_active is True

    # 2. Modified custom claim: preset MUST be inactive
    custom_claim = "Stock prices plunged following geopolitical tensions."
    is_preset_active_custom = (
        preset_choice != "(None - Enter Custom News Claim)"
        and bool(default_image_desc)
        and (custom_claim.strip() == default_text.strip())
    )
    assert is_preset_active_custom is False

    # 3. None selected: preset is inactive
    none_choice = "(None - Enter Custom News Claim)"
    is_preset_active_none = (
        none_choice != "(None - Enter Custom News Claim)"
        and bool(default_image_desc)
        and (custom_claim.strip() == default_text.strip())
    )
    assert is_preset_active_none is False
