"""
Integration and regression test suite for Multimodal Rhetorical Manipulation Classifier (VLDBench Phase 8).
Validates:
1. Headline sanitization (leakage guard)
2. GatedFusionAdapter parameterization (1.58M params) and tensor shapes
3. Model weight loading and singleton reuse (no redundant encoder loading)
4. Multimodal inference (calibrated probabilities + modality salience weights)
5. Text-only fallback behavior (graceful handling of missing images)
6. Empty/whitespace input robustness
7. Epistemic boundary constraints (framing != factual falsehood)
8. Streamlit service layer integration
"""

import os
from pathlib import Path
import pytest
from PIL import Image
import torch

from src.rhetoric_classifier import (
    GatedFusionAdapter,
    RhetoricalManipulationClassifier,
    RhetoricalManipulationResult,
    get_rhetoric_classifier,
    sanitize_headline,
    DEFAULT_WEIGHTS_PATH,
)


def test_headline_sanitization():
    """Validates right-split publisher suffix stripping across standard newsroom formats."""
    # Standard delimiter formats
    clean, removed = sanitize_headline("Fed Cuts Interest Rates as Inflation Cools - Reuters")
    assert removed is True
    assert clean == "Fed Cuts Interest Rates as Inflation Cools"

    clean, removed = sanitize_headline("Global Climate Summit Reaches Landmark Deal | CNN")
    assert removed is True
    assert clean == "Global Climate Summit Reaches Landmark Deal"

    clean, removed = sanitize_headline("Severe Storm Warning Issued for Coast – AP News")
    assert removed is True
    assert clean == "Severe Storm Warning Issued for Coast"

    # No delimiter present (preserve text unchanged)
    clean, removed = sanitize_headline("NASA Prepares for New Lunar Exploration Mission")
    assert removed is False
    assert clean == "NASA Prepares for New Lunar Exploration Mission"

    # Edge cases: short prefix or excessively long suffix (do not strip)
    clean, removed = sanitize_headline("Hello - This is a very long trailing clause that represents actual content rather than a publisher tag")
    assert removed is False

    clean, removed = sanitize_headline("A B - Reuters")
    assert removed is False


def test_adapter_architecture_parameter_count():
    """Validates Option 3 Gated Multimodal Fusion Adapter parameter count and forward shapes."""
    adapter = GatedFusionAdapter(text_dim=768, vision_dim=768, proj_dim=512, dropout=0.3)
    total_params = sum(p.numel() for p in adapter.parameters())
    assert total_params == 1582594, f"Expected 1,582,594 parameters, got {total_params}"

    # Verify forward pass tensor dimensions
    batch_size = 2
    dummy_text = torch.randn(batch_size, 768)
    dummy_vis = torch.randn(batch_size, 768)

    logits, gate = adapter(dummy_text, dummy_vis)
    assert logits.shape == (batch_size, 2)
    assert gate.shape == (batch_size, 512)
    assert (gate >= 0.0).all() and (gate <= 1.0).all()


def test_rhetoric_classifier_singleton_and_weights():
    """Validates global singleton retrieval and verified weight checkpoint."""
    assert DEFAULT_WEIGHTS_PATH.exists(), f"Weights missing at {DEFAULT_WEIGHTS_PATH}"
    assert DEFAULT_WEIGHTS_PATH.stat().st_size > 6000000, "Weights file is unexpectedly small"

    clf1 = get_rhetoric_classifier()
    clf2 = get_rhetoric_classifier()
    assert clf1 is clf2, "get_rhetoric_classifier must return the identical singleton instance"
    assert clf1.is_loaded is True, "Adapter weights failed to load"


def test_multimodal_inference_and_salience():
    """Validates end-to-end multimodal inference with text + image."""
    clf = get_rhetoric_classifier()
    test_img = Image.new("RGB", (224, 224), color=(60, 120, 180))

    result = clf.analyze("Stock Market Faces Sharp Decline Amid New Economic Projections - Bloomberg", test_img)
    assert isinstance(result, RhetoricalManipulationResult)
    assert result.status == "success"
    assert result.has_image is True
    assert result.suffix_removed is True
    assert result.sanitized_headline == "Stock Market Faces Sharp Decline Amid New Economic Projections"

    # Probabilities must sum to 1.0 within numerical tolerance
    prob_sum = result.probability_likely + result.probability_unlikely
    assert abs(prob_sum - 1.0) < 1e-4

    # Salience weights must sum to 1.0
    salience_sum = result.text_salience_weight + result.visual_salience_weight
    assert abs(salience_sum - 1.0) < 1e-4
    assert 0.0 <= result.text_salience_weight <= 1.0
    assert 0.0 <= result.visual_salience_weight <= 1.0

    # Risk level must be one of the three governed states
    assert result.risk_level in ("Low", "Uncertain", "Elevated")


def test_text_only_fallback_behavior():
    """Validates transparent fallback when no image is provided."""
    clf = get_rhetoric_classifier()

    result = clf.analyze("Mayor Announces Citywide Renewable Energy Transit Plan", image=None)
    assert isinstance(result, RhetoricalManipulationResult)
    assert result.status == "text_only_fallback"
    assert result.has_image is False
    assert result.visual_salience_weight == 0.0
    assert result.text_salience_weight == 1.0

    prob_sum = result.probability_likely + result.probability_unlikely
    assert abs(prob_sum - 1.0) < 1e-4


def test_empty_input_handling():
    """Validates graceful handling of empty or whitespace text."""
    clf = get_rhetoric_classifier()

    res_empty = clf.analyze("")
    assert res_empty.status == "error"
    assert res_empty.risk_level == "Uncertain"
    assert res_empty.probability_likely == 0.50
    assert res_empty.probability_unlikely == 0.50

    res_ws = clf.analyze("   \n\t  ")
    assert res_ws.status == "error"
    assert res_ws.risk_level == "Uncertain"


def test_epistemic_disclaimer_integrity():
    """Validates epistemic boundary notice in result payload."""
    clf = get_rhetoric_classifier()
    res = clf.analyze("Sample Headline For Evaluation")
    assert "NOT evaluate whether factual claims are true or false" in res.disclaimer


def test_streamlit_service_layer_cached_getter():
    """Validates that app.py get_rhetoric_service returns the operational classifier."""
    from app import get_rhetoric_service
    svc = get_rhetoric_service()
    assert isinstance(svc, RhetoricalManipulationClassifier)
    assert svc.is_loaded is True
