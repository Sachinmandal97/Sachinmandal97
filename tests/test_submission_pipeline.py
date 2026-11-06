"""
Comprehensive test suite for the MSc project submission application.
Verifies:
1. Valid text input end-to-end inference
2. Empty/invalid input handling
3. Image processing and cross-modal consistency
4. Evidence retrieval and ranking
5. Structured explanation generation
6. Uncertainty & confidence bounds
"""

from pathlib import Path
from PIL import Image


from src.pipeline import MultimodalVerificationPipeline
from src.schemas import DecisionVerdict, MultimodalSample, VerificationResult
from src.preprocess import clean_text, load_and_validate_image
from src.retrieval import EvidenceRetriever


def test_clean_text_normalization():
    """Tests text cleaning on messy input."""
    raw = "  “Breaking News: Major flood in Paris—”   \n"
    cleaned = clean_text(raw)
    assert cleaned == '"Breaking News: Major flood in Paris-"'


def test_empty_input_handling():
    """Tests that empty claims produce graceful fallback without crashing."""
    pipeline = MultimodalVerificationPipeline()
    sample = MultimodalSample(
        sample_id="test_empty_01",
        claim_text="",
        image_path=None
    )
    result = pipeline.verify_sample(sample)
    assert isinstance(result, VerificationResult)
    assert result.decision in [DecisionVerdict.UNCERTAIN, DecisionVerdict.MISLEADING]
    assert len(result.retrieved_evidence) == 0



def test_image_loading_and_cross_modal(tmp_path: Path):
    """Tests valid PIL image loading and OpenCLIP alignment scoring."""
    test_img_path = tmp_path / "fire_test.jpg"
    img = Image.new("RGB", (224, 224), color="red")
    img.save(test_img_path)

    loaded, err = load_and_validate_image(test_img_path)
    assert err is None
    assert loaded is not None

    pipeline = MultimodalVerificationPipeline()
    sample = MultimodalSample(
        sample_id="test_img_01",
        claim_text="Warehouse fire with heavy smoke.",
        image_path=str(test_img_path)
    )
    result = pipeline.verify_sample(sample)
    assert isinstance(result, VerificationResult)
    assert 0.0 <= result.cross_modal.similarity_score <= 1.0


def test_evidence_retrieval():
    """Tests BM25 evidence retrieval over the submission corpus."""
    pipeline = MultimodalVerificationPipeline()
    hits = pipeline.retriever.retrieve("warehouse fire London", top_k=2)
    assert len(hits) > 0
    assert "fire" in hits[0].text.lower() or "london" in hits[0].text.lower()
    assert hits[0].score > 0.0


def test_end_to_end_verification_decision():
    """Tests complete end-to-end pipeline execution with structured report synthesis."""
    pipeline = MultimodalVerificationPipeline()
    sample = MultimodalSample(
        sample_id="test_full_01",
        claim_text="London Fire Brigade tackled a warehouse blaze on Sunday morning.",
        image_path=None
    )
    result = pipeline.verify_sample(sample)
    assert result.decision in [
        DecisionVerdict.SUPPORTED,
        DecisionVerdict.PARTIALLY_SUPPORTED,
        DecisionVerdict.MISLEADING,
        DecisionVerdict.UNCERTAIN
    ]
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.uncertainty_score <= 1.0
    assert round(result.confidence + result.uncertainty_score, 2) == 1.00
    assert "AI-ASSISTED VERIFICATION REPORT" in result.structured_explanation
    assert len(result.reasoning_factors) > 0
    assert len(result.atomic_verifications) > 0

