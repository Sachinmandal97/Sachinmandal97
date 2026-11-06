"""
Benchmark Evaluation Harness & Metrics Module.
Provides reproducible academic evaluation, strict schema validation,
exact metric calculations (accuracy, balanced accuracy, macro-F1, ROC-AUC, Wilson CIs),
and stratified sampling for NewsCLIPpings and Factify 2.
"""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field


# ============================================================================
# 1. STABLE BENCHMARK PREDICTION RECORD SCHEMA (20 FIELDS)
# ============================================================================

class BenchmarkPredictionRecord(BaseModel):
    """
    Standardized, privacy-preserving benchmark prediction record.
    Matches the approved Phase 2A schema exactly (20 fields).
    Excludes raw images, byte buffers, or large HTML bodies.
    """
    sample_id: str = Field(description="Unique sample identifier from benchmark dataset")
    dataset_name: str = Field(description="Benchmark name ('newsclippings' or 'factify2')")
    split: str = Field(description="Dataset split ('val', 'test', or 'demo')")
    task_id: str = Field(description="Task identifier ('TASK_A', 'TASK_B', or 'TASK_C')")
    ground_truth_raw: str = Field(description="Original unmapped ground-truth label string")
    ground_truth_mapped: str = Field(description="Normalized target class string")
    predicted_verdict: str = Field(description="Model prediction mapped to target class space")
    is_correct: bool = Field(description="True if predicted_verdict equals ground_truth_mapped")
    confidence: float = Field(default=0.0, description="Continuous model confidence [0.0 - 1.0]")
    uncertainty_score: float = Field(default=1.0, description="Predictive uncertainty index [0.0 - 1.0]")
    cross_modal_score: Optional[float] = Field(default=None, description="OpenCLIP cosine similarity score")
    mismatch_detected: bool = Field(default=False, description="True if cross-modal similarity < cutoff")
    atomic_claims_count: int = Field(default=0, description="Number of decomposed propositions")
    nli_predicted_relation: Optional[str] = Field(default=None, description="Primary NLI relation against evidence")
    nli_p_entailment: Optional[float] = Field(default=None, description="Softmax entailment probability")
    nli_p_neutral: Optional[float] = Field(default=None, description="Softmax neutral probability")
    nli_p_contradiction: Optional[float] = Field(default=None, description="Softmax contradiction probability")
    evidence_source_mode: str = Field(default="paired_document", description="Grounding source mode")
    execution_time_sec: float = Field(default=0.0, description="Wall-clock latency in seconds")
    error_message: Optional[str] = Field(default=None, description="Error trace string if sample evaluation failed")


# ============================================================================
# 2. STATISTICAL METRICS & CONFIDENCE INTERVALS
# ============================================================================

def compute_wilson_score_interval(
    successes: int,
    total: int,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Computes the Wilson score confidence interval for a binomial proportion.
    More robust than the normal approximation near boundary values (0 or 1) and for small N.
    Default: 95% confidence (z = 1.95996).
    """
    if total <= 0:
        return (0.0, 0.0)

    # Critical z-values for standard confidence levels
    z_map = {0.90: 1.64485, 0.95: 1.95996, 0.99: 2.57583}
    z = z_map.get(confidence, 1.95996)

    p_hat = successes / total
    z2 = z * z
    denom = 1.0 + (z2 / total)
    center = (p_hat + (z2 / (2.0 * total))) / denom
    spread = (z * math.sqrt((p_hat * (1.0 - p_hat) / total) + (z2 / (4.0 * total * total)))) / denom

    ci_lower = max(0.0, center - spread)
    ci_upper = min(1.0, center + spread)
    return (round(ci_lower, 4), round(ci_upper, 4))


def compute_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str]
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, float]]]:
    """
    Computes raw count confusion matrix and row-normalized confusion matrix.
    Row = True label, Column = Predicted label.
    """
    raw_cm: Dict[str, Dict[str, int]] = {
        true_lbl: {pred_lbl: 0 for pred_lbl in labels} for true_lbl in labels
    }

    for yt, yp in zip(y_true, y_pred):
        if yt in raw_cm and yp in raw_cm[yt]:
            raw_cm[yt][yp] += 1

    norm_cm: Dict[str, Dict[str, float]] = {}
    for true_lbl, preds in raw_cm.items():
        row_sum = sum(preds.values())
        norm_cm[true_lbl] = {
            pred_lbl: round(count / row_sum, 4) if row_sum > 0 else 0.0
            for pred_lbl, count in preds.items()
        }

    return raw_cm, norm_cm


def calculate_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes comprehensive classification metrics from ground-truth and prediction vectors:
      - Overall Accuracy with 95% Wilson Confidence Interval and Standard Error
      - Balanced Accuracy (mean of per-class recalls)
      - Macro-Precision, Macro-Recall, Macro-F1
      - Weighted-F1
      - Per-class breakdown: TP, FP, FN, TN, Support, Precision, Recall, F1
      - Full Confusion Matrix (raw counts and row-normalized)

    Pure Python implementation without requiring scikit-learn.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}")

    n = len(y_true)
    if n == 0:
        return {
            "total_samples": 0,
            "accuracy": 0.0,
            "accuracy_ci_95": {"lower": 0.0, "upper": 0.0},
            "balanced_accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "per_class": {},
            "confusion_matrix_raw": {},
            "confusion_matrix_normalized": {}
        }

    # Deduce unique label set
    if labels is None:
        unique_labels = sorted(list(set(y_true).union(set(y_pred))))
    else:
        unique_labels = list(labels)

    raw_cm, norm_cm = compute_confusion_matrix(y_true, y_pred, unique_labels)

    correct_total = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    acc = correct_total / n
    ci_lower, ci_upper = compute_wilson_score_interval(correct_total, n, confidence=0.95)
    se = math.sqrt((acc * (1.0 - acc)) / n) if n > 0 else 0.0

    per_class: Dict[str, Dict[str, Any]] = {}
    recalls: List[float] = []
    precisions: List[float] = []
    f1s: List[float] = []
    supports: List[int] = []

    for lbl in unique_labels:
        tp = raw_cm[lbl][lbl] if lbl in raw_cm else 0
        fp = sum(raw_cm[other][lbl] for other in unique_labels if other != lbl and other in raw_cm)
        fn = sum(raw_cm[lbl][other] for other in unique_labels if other != lbl and lbl in raw_cm)
        tn = n - (tp + fp + fn)
        support = tp + fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[lbl] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)

    num_classes = len(unique_labels)
    macro_prec = sum(precisions) / num_classes if num_classes > 0 else 0.0
    macro_rec = sum(recalls) / num_classes if num_classes > 0 else 0.0
    macro_f1 = sum(f1s) / num_classes if num_classes > 0 else 0.0

    # Balanced accuracy is the unweighted average recall across classes with support > 0
    classes_with_support = [r for r, s in zip(recalls, supports) if s > 0]
    balanced_acc = sum(classes_with_support) / len(classes_with_support) if classes_with_support else 0.0

    # Weighted F1
    weighted_f1 = sum(f * s for f, s in zip(f1s, supports)) / n if n > 0 else 0.0

    return {
        "total_samples": n,
        "accuracy": round(acc, 4),
        "accuracy_ci_95": {"lower": ci_lower, "upper": ci_upper},
        "accuracy_standard_error": round(se, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix_raw": raw_cm,
        "confusion_matrix_normalized": norm_cm
    }


def compute_roc_auc_binary(
    y_true: List[str],
    y_scores: List[float],
    pos_label: str
) -> Optional[float]:
    """
    Computes the Area Under the Receiver Operating Characteristic Curve (ROC-AUC)
    for binary classification using the rank-based Mann-Whitney U statistic.
    Handles tied scores correctly by assigning fractional average ranks.

    Returns:
      float in [0.0, 1.0] representing the exact empirical ROC-AUC.
      None if the dataset contains only one unique ground-truth class (single-class edge case).
    """
    if len(y_true) != len(y_scores):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} vs len(y_scores)={len(y_scores)}")

    if len(y_true) < 2:
        return None

    # Count positive and negative samples
    n_pos = sum(1 for yt in y_true if yt == pos_label)
    n_neg = len(y_true) - n_pos

    # If only one class is present, ROC-AUC is mathematically undefined
    if n_pos == 0 or n_neg == 0:
        return None

    # Pair scores with binary indicator: 1 for pos_label, 0 for negative
    binary_pairs = [(float(score), 1 if yt == pos_label else 0) for score, yt in zip(y_scores, y_true)]

    # Sort ascending by score
    binary_pairs.sort(key=lambda x: x[0])

    # Assign average ranks to ties
    n = len(binary_pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and binary_pairs[j + 1][0] == binary_pairs[i][0]:
            j += 1
        # Range i through j has tied scores
        # 1-based ranks
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    # Sum of ranks for positive class
    sum_ranks_pos = sum(rank for rank, (_, is_pos) in zip(ranks, binary_pairs) if is_pos == 1)

    # Mann-Whitney U statistic: U = R_pos - (n_pos * (n_pos + 1)) / 2
    u_stat = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
    auc = u_stat / (n_pos * n_neg)

    return round(max(0.0, min(1.0, auc)), 4)


# ============================================================================
# 3. DETERMINISTIC STRATIFIED SAMPLING UTILITY
# ============================================================================

def stratified_sample(
    records: List[Any],
    label_getter: Callable[[Any], str],
    n_samples: int,
    seed: int = 42,
    balanced: bool = True
) -> List[Any]:
    """
    Performs deterministic stratified sampling from a list of records.
    Guarantees bit-for-bit repeatability across executions with the same seed.

    Args:
      records: List of input sample items.
      label_getter: Callable extracting stratum label string from item.
      n_samples: Total number of samples requested.
      seed: Pseudo-random generator seed (fixed default: 42).
      balanced: If True, draws equal quota per class; if False, preserves proportions.

    Returns:
      Stratified list of sampled items of length min(n_samples, len(records)).
    """
    if not records or n_samples <= 0:
        return []

    if n_samples >= len(records):
        return list(records)

    # Group into strata
    strata: Dict[str, List[Any]] = {}
    for item in records:
        lbl = str(label_getter(item))
        strata.setdefault(lbl, []).append(item)

    num_classes = len(strata)
    if num_classes == 0:
        return []

    # Deterministic shuffle within each stratum
    rng = random.Random(seed)
    shuffled_strata = {}
    for lbl, group in strata.items():
        copied = list(group)
        rng.shuffle(copied)
        shuffled_strata[lbl] = copied

    selected: List[Any] = []

    if balanced:
        # Equal quota per stratum
        base_quota = n_samples // num_classes
        remainder = n_samples % num_classes
        strata_keys = sorted(strata.keys())

        # Distribute base quota
        quotas = {k: min(len(shuffled_strata[k]), base_quota) for k in strata_keys}
        
        # Distribute remainder
        for k in strata_keys[:remainder]:
            if len(shuffled_strata[k]) > quotas[k]:
                quotas[k] += 1

        # If some strata had fewer samples than quota, redistribute surplus
        deficit = n_samples - sum(quotas.values())
        while deficit > 0:
            added = False
            for k in strata_keys:
                if len(shuffled_strata[k]) > quotas[k] and deficit > 0:
                    quotas[k] += 1
                    deficit -= 1
                    added = True
            if not added:
                break

        for k in strata_keys:
            selected.extend(shuffled_strata[k][:quotas[k]])

    else:
        # Proportional quota per stratum
        total_rec = len(records)
        strata_keys = sorted(strata.keys())
        quotas = {}
        for k in strata_keys:
            prop = len(strata[k]) / total_rec
            quotas[k] = min(len(shuffled_strata[k]), max(1, round(prop * n_samples)))

        # Adjust total
        diff = sum(quotas.values()) - n_samples
        if diff > 0:
            for k in reversed(strata_keys):
                while diff > 0 and quotas[k] > 1:
                    quotas[k] -= 1
                    diff -= 1
        elif diff < 0:
            for k in strata_keys:
                while diff < 0 and quotas[k] < len(shuffled_strata[k]):
                    quotas[k] += 1
                    diff += 1

        for k in strata_keys:
            selected.extend(shuffled_strata[k][:quotas[k]])

    # Shuffle the combined selected set deterministically
    rng.shuffle(selected)
    return selected


# ============================================================================
# 4. BENCHMARK EVALUATION SUMMARY & HARNESS
# ============================================================================

class EvaluationSummary(BaseModel):
    """
    Comprehensive machine-readable summary of a benchmark evaluation run.
    """
    dataset_name: str
    task_id: str
    split: str
    timestamp_utc: str
    total_requested: int
    total_evaluated: int
    total_failed: int
    coverage_rate: float
    error_rate: float
    total_execution_time_sec: float
    mean_latency_per_sample_sec: float
    metrics: Dict[str, Any]
    roc_auc: Optional[float] = None
    config_parameters: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkHarness:
    """
    Execution engine for benchmark evaluations.
    Manages sample loading, adapter dispatch, timing, error isolation,
    metric computation, and serialization to JSONL and JSON.
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        self.output_dir = Path(output_dir or "./outputs/benchmark_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_evaluation(
        self,
        adapter: Any,
        pipeline: Any,
        task_id: str,
        split: str = "val",
        limit: Optional[int] = None,
        seed: int = 42,
        data_path: Optional[Union[str, Path]] = None,
        config_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[EvaluationSummary, List[BenchmarkPredictionRecord]]:
        """
        Executes a complete benchmark evaluation pass.
        
        Args:
          adapter: Dataset adapter (NewsCLIPpingsAdapter or Factify2Adapter).
          pipeline: MultimodalVerificationPipeline instance.
          task_id: Evaluation task ('TASK_A', 'TASK_B', or 'TASK_C').
          split: Dataset split ('val', 'test', or 'demo').
          limit: Max sample count to evaluate.
          seed: Random seed for stratified sampling (default 42).
          data_path: Optional path to external raw dataset file/dir.
          config_params: Diagnostic config parameters dictionary.

        Returns:
          Tuple of (EvaluationSummary, List[BenchmarkPredictionRecord]).
        """
        config_params = config_params or {}
        dataset_name = adapter.name.lower()
        t_start = time.perf_counter()

        # Step 1: Load raw samples
        raw_samples = adapter.load_samples(data_path=data_path, split=split)
        total_available = len(raw_samples)

        # Step 2: Stratified sampling if limit < total_available
        if limit is not None and limit < total_available:
            label_getter = getattr(adapter, "get_sample_stratum_label", lambda s: getattr(s, "raw_label", "default"))
            eval_samples = stratified_sample(
                raw_samples,
                label_getter=label_getter,
                n_samples=limit,
                seed=seed,
                balanced=True
            )
        else:
            eval_samples = raw_samples[:limit] if limit else raw_samples

        total_requested = len(eval_samples)
        records: List[BenchmarkPredictionRecord] = []
        failed_count = 0

        # Step 3: Evaluate each sample with error isolation and latency tracking
        for idx, sample in enumerate(eval_samples, start=1):
            t_sample_start = time.perf_counter()
            try:
                rec = adapter.evaluate_sample(
                    sample=sample,
                    pipeline=pipeline,
                    task_id=task_id,
                    split=split,
                    **config_params
                )
                rec.execution_time_sec = round(time.perf_counter() - t_sample_start, 3)
                records.append(rec)
            except Exception as exc:
                t_elapsed = round(time.perf_counter() - t_sample_start, 3)
                failed_count += 1
                fallback_id = getattr(sample, "sample_id", f"sample_{idx}")
                fallback_raw = getattr(sample, "raw_label", getattr(sample, "label", "unknown"))
                rec = BenchmarkPredictionRecord(
                    sample_id=str(fallback_id),
                    dataset_name=dataset_name,
                    split=split,
                    task_id=task_id,
                    ground_truth_raw=str(fallback_raw),
                    ground_truth_mapped=str(fallback_raw),
                    predicted_verdict="ERROR",
                    is_correct=False,
                    confidence=0.0,
                    uncertainty_score=1.0,
                    evidence_source_mode="error",
                    execution_time_sec=t_elapsed,
                    error_message=str(exc)
                )
                records.append(rec)

        t_total = time.perf_counter() - t_start
        total_eval = len(records)
        coverage_rate = round((total_eval - failed_count) / max(1, total_eval), 4)
        error_rate = round(failed_count / max(1, total_eval), 4)
        mean_lat = round(t_total / max(1, total_eval), 3)

        # Step 4: Calculate evaluation metrics across successfully evaluated records
        valid_records = [r for r in records if r.error_message is None]
        y_true = [r.ground_truth_mapped for r in valid_records]
        y_pred = [r.predicted_verdict for r in valid_records]

        # Target class labels
        target_labels = adapter.get_target_labels(task_id=task_id)
        metrics = calculate_classification_metrics(y_true, y_pred, labels=target_labels)

        # ROC-AUC calculation for binary tasks (e.g. Task A)
        roc_auc = None
        if task_id.upper() == "TASK_A":
            # For Task A, continuous distance is (1.0 - cross_modal_score)
            scores = [
                1.0 - (r.cross_modal_score if r.cross_modal_score is not None else 0.0)
                for r in valid_records
            ]
            roc_auc = compute_roc_auc_binary(y_true, scores, pos_label="falsified")

        # Step 5: Construct summary
        timestamp_str = datetime.now(timezone.utc).isoformat()
        summary = EvaluationSummary(
            dataset_name=dataset_name,
            task_id=task_id,
            split=split,
            timestamp_utc=timestamp_str,
            total_requested=total_requested,
            total_evaluated=total_eval,
            total_failed=failed_count,
            coverage_rate=coverage_rate,
            error_rate=error_rate,
            total_execution_time_sec=round(t_total, 3),
            mean_latency_per_sample_sec=mean_lat,
            metrics=metrics,
            roc_auc=roc_auc,
            config_parameters={
                **config_params,
                "seed": seed,
                "total_available_in_split": total_available
            }
        )

        # Step 6: Export results to disk
        self._export_results(dataset_name, task_id, records, summary)

        return summary, records

    def _export_results(
        self,
        dataset_name: str,
        task_id: str,
        records: List[BenchmarkPredictionRecord],
        summary: EvaluationSummary
    ) -> None:
        """
        Saves predictions to JSONL and summary metrics to JSON in output_dir.
        """
        prefix = f"{dataset_name}_{task_id.lower()}"
        jsonl_path = self.output_dir / f"{prefix}_predictions.jsonl"
        summary_path = self.output_dir / f"{prefix}_metrics.json"

        # Write predictions JSONL
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(rec.model_dump_json() + "\n")

        # Write summary JSON
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary.model_dump_json(indent=2) + "\n")
