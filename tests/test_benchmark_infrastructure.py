"""
Unit and integration tests for the benchmark evaluation infrastructure.
Tests metric calculations, Wilson confidence intervals, ROC-AUC, stratified sampling,
adapters for NewsCLIPpings and Factify 2, error isolation, and schema compliance.
"""

import json
from pathlib import Path
import tempfile

from src.benchmark_adapters import Factify2Adapter, NewsCLIPpingsAdapter
from src.evaluation_harness import (
    BenchmarkHarness,
    BenchmarkPredictionRecord,
    calculate_classification_metrics,
    compute_confusion_matrix,
    compute_roc_auc_binary,
    compute_wilson_score_interval,
    stratified_sample,
)
from src.schemas import Factify2Sample, MultimodalSample


# ============================================================================
# 1. METRICS & STATISTICAL FUNCTIONS TESTS
# ============================================================================

def test_wilson_score_confidence_interval():
    """Verifies Wilson score confidence intervals for standard and boundary conditions."""
    # Standard 50/100 proportion at 95%
    lower, upper = compute_wilson_score_interval(successes=50, total=100, confidence=0.95)
    assert 0.40 <= lower <= 0.41, f"Expected lower ~0.4038, got {lower}"
    assert 0.59 <= upper <= 0.60, f"Expected upper ~0.5962, got {upper}"

    # Boundary: 0/10 successes
    l_zero, u_zero = compute_wilson_score_interval(successes=0, total=10, confidence=0.95)
    assert l_zero == 0.0
    assert 0.25 <= u_zero <= 0.32

    # Boundary: 10/10 successes
    l_perf, u_perf = compute_wilson_score_interval(successes=10, total=10, confidence=0.95)
    assert 0.68 <= l_perf <= 0.75
    assert u_perf == 1.0

    # Empty total
    l_empty, u_empty = compute_wilson_score_interval(successes=0, total=0)
    assert l_empty == 0.0 and u_empty == 0.0


def test_classification_metrics_computation():
    """Verifies accuracy, balanced accuracy, precision, recall, and F1 on known data."""
    # 4 samples: 2 cat, 2 dog. Pred: 1 cat, 3 dog.
    y_true = ["cat", "cat", "dog", "dog"]
    y_pred = ["cat", "dog", "dog", "dog"]

    metrics = calculate_classification_metrics(y_true, y_pred, labels=["cat", "dog"])

    assert metrics["total_samples"] == 4
    assert metrics["accuracy"] == 0.75
    # cat: TP=1, FP=0, FN=1 -> rec=0.5, prec=1.0, F1=0.6667
    # dog: TP=2, FP=1, FN=0 -> rec=1.0, prec=0.6667, F1=0.8000
    assert metrics["per_class"]["cat"]["precision"] == 1.0
    assert metrics["per_class"]["cat"]["recall"] == 0.5
    assert abs(metrics["per_class"]["cat"]["f1"] - 0.6667) < 1e-3

    assert abs(metrics["per_class"]["dog"]["precision"] - 0.6667) < 1e-3
    assert metrics["per_class"]["dog"]["recall"] == 1.0
    assert metrics["per_class"]["dog"]["f1"] == 0.8

    # Balanced accuracy: (0.5 + 1.0) / 2 = 0.75
    assert metrics["balanced_accuracy"] == 0.75

    # Macro F1: (0.6667 + 0.8) / 2 = 0.7333
    assert abs(metrics["macro_f1"] - 0.7333) < 1e-3

    # Check confusion matrix
    cm_raw = metrics["confusion_matrix_raw"]
    assert cm_raw["cat"]["cat"] == 1
    assert cm_raw["cat"]["dog"] == 1
    assert cm_raw["dog"]["dog"] == 2
    assert cm_raw["dog"]["cat"] == 0


def test_roc_auc_binary_mann_whitney_u():
    """Verifies binary ROC-AUC calculation, rank ties, and single-class edge case."""
    y_true = ["falsified", "falsified", "pristine", "pristine"]

    # 1. Perfect separation
    scores_perf = [0.95, 0.85, 0.15, 0.05]
    auc_perf = compute_roc_auc_binary(y_true, scores_perf, pos_label="falsified")
    assert auc_perf == 1.0

    # 2. Perfect inverse separation
    scores_inv = [0.05, 0.15, 0.85, 0.95]
    auc_inv = compute_roc_auc_binary(y_true, scores_inv, pos_label="falsified")
    assert auc_inv == 0.0

    # 3. Complete tie (all scores equal)
    scores_tie = [0.5, 0.5, 0.5, 0.5]
    auc_tie = compute_roc_auc_binary(y_true, scores_tie, pos_label="falsified")
    assert auc_tie == 0.5

    # 4. Single-class edge case (mathematically undefined)
    y_single = ["falsified", "falsified"]
    scores_single = [0.8, 0.9]
    auc_single = compute_roc_auc_binary(y_single, scores_single, pos_label="falsified")
    assert auc_single is None  # Must return None gracefully rather than throwing an unhandled exception


# ============================================================================
# 2. STRATIFIED SAMPLING TESTS
# ============================================================================

def test_deterministic_stratified_sampling():
    """Verifies stratified sampling guarantees class quota and bit-for-bit repeatability."""
    records = [
        {"id": f"c_{i}", "label": "cat"} for i in range(20)
    ] + [
        {"id": f"d_{i}", "label": "dog"} for i in range(20)
    ]

    # Sample 10 items balanced (5 cats, 5 dogs)
    s1 = stratified_sample(records, lambda r: r["label"], n_samples=10, seed=42, balanced=True)
    assert len(s1) == 10
    cats_1 = sum(1 for x in s1 if x["label"] == "cat")
    dogs_1 = sum(1 for x in s1 if x["label"] == "dog")
    assert cats_1 == 5
    assert dogs_1 == 5

    # Deterministic repeatability with same seed
    s2 = stratified_sample(records, lambda r: r["label"], n_samples=10, seed=42, balanced=True)
    assert [x["id"] for x in s1] == [x["id"] for x in s2]

    # Different seed produces different ordering
    s3 = stratified_sample(records, lambda r: r["label"], n_samples=10, seed=99, balanced=True)
    assert [x["id"] for x in s1] != [x["id"] for x in s3]


# ============================================================================
# 3. SCHEMA COMPLIANCE (20 FIELDS)
# ============================================================================

def test_benchmark_prediction_record_schema_20_fields():
    """Verifies that BenchmarkPredictionRecord implements all 20 required schema fields."""
    rec = BenchmarkPredictionRecord(
        sample_id="test_001",
        dataset_name="newsclippings",
        split="test",
        task_id="TASK_A",
        ground_truth_raw="falsified",
        ground_truth_mapped="falsified",
        predicted_verdict="falsified",
        is_correct=True,
        confidence=0.88,
        uncertainty_score=0.12,
        cross_modal_score=0.10,
        mismatch_detected=True,
        atomic_claims_count=1,
        nli_predicted_relation=None,
        nli_p_entailment=None,
        nli_p_neutral=None,
        nli_p_contradiction=None,
        evidence_source_mode="none",
        execution_time_sec=0.15,
        error_message=None
    )

    data = json.loads(rec.model_dump_json())
    required_fields = [
        "sample_id", "dataset_name", "split", "task_id",
        "ground_truth_raw", "ground_truth_mapped", "predicted_verdict", "is_correct",
        "confidence", "uncertainty_score", "cross_modal_score", "mismatch_detected",
        "atomic_claims_count", "nli_predicted_relation", "nli_p_entailment",
        "nli_p_neutral", "nli_p_contradiction", "evidence_source_mode",
        "execution_time_sec", "error_message"
    ]
    assert len(required_fields) == 20
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"


# ============================================================================
# 4. ADAPTER TESTS
# ============================================================================

def test_newsclippings_adapter_offline_embedding_evaluation():
    """Verifies NewsCLIPpingsAdapter evaluates Task A using precomputed embeddings."""
    adapter = NewsCLIPpingsAdapter()
    assert adapter.get_target_labels("TASK_A") == ["pristine", "falsified"]

    # Sample with identical embeddings (cosine sim = 1.0 -> pristine)
    sample_pristine = MultimodalSample(
        sample_id="nc_offline_01",
        claim_text="An authentic headline",
        raw_label="pristine",
        metadata={
            "clip_text_embedding": [1.0, 0.0, 0.0],
            "clip_image_embedding": [1.0, 0.0, 0.0]
        }
    )

    class DummyPipeline:
        config = {"project": {"device": "cpu"}, "models": {}}

    rec_pristine = adapter.evaluate_sample(
        sample=sample_pristine,
        pipeline=DummyPipeline(),
        task_id="TASK_A",
        threshold_cm=0.22
    )

    assert rec_pristine.cross_modal_score == 1.0
    assert rec_pristine.predicted_verdict == "pristine"
    assert rec_pristine.is_correct is True
    assert rec_pristine.mismatch_detected is False

    # Sample with orthogonal/negative embeddings (cosine sim = 0.0 < 0.22 -> falsified)
    sample_falsified = MultimodalSample(
        sample_id="nc_offline_02",
        claim_text="A mismatched headline",
        raw_label="falsified",
        metadata={
            "clip_text_embedding": [1.0, 0.0, 0.0],
            "clip_image_embedding": [0.0, 1.0, 0.0]
        }
    )

    rec_falsified = adapter.evaluate_sample(
        sample=sample_falsified,
        pipeline=DummyPipeline(),
        task_id="TASK_A",
        threshold_cm=0.22
    )

    assert rec_falsified.cross_modal_score == 0.0
    assert rec_falsified.predicted_verdict == "falsified"
    assert rec_falsified.is_correct is True
    assert rec_falsified.mismatch_detected is True


def test_factify2_adapter_controlled_document_grounding():
    """Verifies Factify 2 adapter grounds verification strictly on paired documents."""
    adapter = Factify2Adapter()
    assert adapter.get_target_labels("TASK_B") == ["SUPPORTED", "MISLEADING", "UNCERTAIN"]

    sample_support = Factify2Sample(
        sample_id="f2_test_01",
        claim_text="The central bank raised benchmark interest rates.",
        document_text="The central bank officially raised benchmark interest rates today.",
        label="Support_Text"
    )

    class DummyPipeline:
        config = {
            "project": {"device": "cpu"},
            "models": {"nli_model": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"},
            "thresholds": {"cross_modal_mismatch_cutoff": 0.22, "nli_entailment_cutoff": 0.60, "nli_contradiction_cutoff": 0.50}
        }

    rec = adapter.evaluate_sample(
        sample=sample_support,
        pipeline=DummyPipeline(),
        task_id="TASK_B"
    )

    assert rec.evidence_source_mode == "paired_document"
    assert rec.ground_truth_mapped == "SUPPORTED"
    assert rec.task_id == "TASK_B"
    assert rec.error_message is None


# ============================================================================
# 5. HARNESS ERROR ISOLATION & EXPORT TESTS
# ============================================================================

def test_harness_error_isolation_and_export():
    """Verifies that an error in one sample does not abort evaluation and exports correctly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        harness = BenchmarkHarness(output_dir=output_dir)

        # Mock adapter with 2 valid samples and 1 failing sample
        class MockFaultyAdapter:
            name = "mock_bench"

            def get_target_labels(self, task_id):
                return ["pristine", "falsified"]

            def load_samples(self, data_path=None, split="val", limit=None):
                return [
                    MultimodalSample(sample_id="s1", claim_text="Good 1", raw_label="pristine"),
                    MultimodalSample(sample_id="s2", claim_text="Bad", raw_label="falsified"),
                    MultimodalSample(sample_id="s3", claim_text="Good 2", raw_label="falsified")
                ]

            def evaluate_sample(self, sample, pipeline, task_id, split="val", **kwargs):
                if sample.sample_id == "s2":
                    raise RuntimeError("Simulated processing error on sample s2")
                return BenchmarkPredictionRecord(
                    sample_id=sample.sample_id,
                    dataset_name="mock_bench",
                    split=split,
                    task_id=task_id,
                    ground_truth_raw=sample.raw_label,
                    ground_truth_mapped=sample.raw_label,
                    predicted_verdict=sample.raw_label,
                    is_correct=True,
                    confidence=0.9,
                    uncertainty_score=0.1,
                    evidence_source_mode="none"
                )

        summary, records = harness.run_evaluation(
            adapter=MockFaultyAdapter(),
            pipeline=None,
            task_id="TASK_A",
            split="val"
        )

        # Verify error isolation
        assert len(records) == 3
        assert summary.total_evaluated == 3
        assert summary.total_failed == 1
        assert summary.coverage_rate == round(2 / 3, 4)
        assert summary.error_rate == round(1 / 3, 4)

        # The failed record has error details
        failed_rec = records[1]
        assert failed_rec.predicted_verdict == "ERROR"
        assert failed_rec.error_message == "Simulated processing error on sample s2"

        # Check export files exist and are valid JSON / JSONL
        jsonl_file = output_dir / "mock_bench_task_a_predictions.jsonl"
        json_file = output_dir / "mock_bench_task_a_metrics.json"

        assert jsonl_file.exists(), "JSONL predictions file was not created"
        assert json_file.exists(), "JSON metrics summary file was not created"

        # Verify JSONL content
        lines = jsonl_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        first_line = json.loads(lines[0])
        assert first_line["sample_id"] == "s1"

        # Verify JSON summary content
        summary_data = json.loads(json_file.read_text(encoding="utf-8"))
        assert summary_data["dataset_name"] == "mock_bench"
        assert summary_data["total_failed"] == 1


def test_newsclippings_genuine_data_loading_and_schema():
    """
    Verifies that NewsCLIPpingsAdapter correctly loads genuine benchmark data
    from merged annotations, metadata, and precomputed embeddings.
    """
    adapter = NewsCLIPpingsAdapter()
    samples = adapter.load_samples(split="val", limit=5)

    assert len(samples) == 5, f"Expected 5 samples, got {len(samples)}"
    for sample in samples:
        assert sample.sample_id.startswith("nc_val_")
        assert len(sample.claim_text.strip()) > 10
        assert sample.raw_label in ["pristine", "falsified"]
        assert adapter.get_sample_stratum_label(sample) in ["pristine", "falsified"]
        assert "clip_text_embedding" in sample.metadata
        assert "clip_image_embedding" in sample.metadata
        assert sample.metadata["clip_text_embedding"].shape == (512,)
        assert sample.metadata["clip_image_embedding"].shape == (512,)

    # Verify fallback to demo when non-existent path is requested
    fallback_samples = adapter.load_samples(data_path=Path("./non_existent_dir"), split="val", limit=5)
    assert len(fallback_samples) == 5
    assert fallback_samples[0].sample_id.startswith("nc_demo_")

