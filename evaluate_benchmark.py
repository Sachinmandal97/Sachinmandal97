"""
CLI Entrypoint for Benchmark Evaluation.
Reproducibly evaluates NewsCLIPpings (Task A) and Factify 2 (Task B & Task C)
using the modular benchmark evaluation infrastructure.

Usage examples:
  python evaluate_benchmark.py --dataset newsclippings --limit 5
  python evaluate_benchmark.py --dataset factify2 --task task_b --limit 5
  python evaluate_benchmark.py --dataset factify2 --task task_c --limit 5
"""

import argparse
from pathlib import Path
import sys
import time

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark_adapters import Factify2Adapter, NewsCLIPpingsAdapter
from src.evaluation_harness import BenchmarkHarness, EvaluationSummary
from src.pipeline import MultimodalVerificationPipeline
from src.utils import setup_logger

logger = setup_logger("evaluate_benchmark")


def print_summary_table(summary: EvaluationSummary, output_dir: Path) -> None:
    """Prints a clean, academic-formatted evaluation report to stdout."""
    prefix = f"{summary.dataset_name}_{summary.task_id.lower()}"
    jsonl_path = output_dir / f"{prefix}_predictions.jsonl"
    metrics_path = output_dir / f"{prefix}_metrics.json"

    print("\n" + "=" * 78)
    print(f"BENCHMARK EVALUATION RESULTS: {summary.dataset_name.upper()} ({summary.task_id.upper()})")
    print("=" * 78)
    print(f"  Split:                  {summary.split}")
    print(f"  Timestamp (UTC):        {summary.timestamp_utc}")
    print(f"  Total Requested:        {summary.total_requested}")
    print(f"  Total Evaluated:        {summary.total_evaluated}")
    print(f"  Failed / Errors:        {summary.total_failed} (Error Rate: {summary.error_rate * 100:.1f}%)")
    print(f"  Coverage Rate:          {summary.coverage_rate * 100:.1f}%")
    print(f"  Total Latency:          {summary.total_execution_time_sec:.2f}s ({summary.mean_latency_per_sample_sec:.2f}s/sample)")
    print("-" * 78)

    m = summary.metrics
    acc = m.get("accuracy", 0.0) * 100
    ci = m.get("accuracy_ci_95", {})
    ci_l = ci.get("lower", 0.0) * 100
    ci_u = ci.get("upper", 0.0) * 100
    bal_acc = m.get("balanced_accuracy", 0.0) * 100
    m_f1 = m.get("macro_f1", 0.0) * 100
    m_prec = m.get("macro_precision", 0.0) * 100
    m_rec = m.get("macro_recall", 0.0) * 100

    print(f"  Overall Accuracy:       {acc:.2f}%  (95% Wilson CI: [{ci_l:.1f}%, {ci_u:.1f}%])")
    print(f"  Balanced Accuracy:      {bal_acc:.2f}%")
    print(f"  Macro-Precision:        {m_prec:.2f}%")
    print(f"  Macro-Recall:           {m_rec:.2f}%")
    print(f"  Macro-F1:               {m_f1:.2f}%")
    if summary.roc_auc is not None:
        print(f"  ROC-AUC (Binary):       {summary.roc_auc:.4f}")
    print("-" * 78)

    print("  PER-CLASS PERFORMANCE BREAKDOWN:")
    print(f"  {'Class':<26} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    print("  " + "-" * 72)
    per_class = m.get("per_class", {})
    for cls_name, p_data in per_class.items():
        p = p_data.get("precision", 0.0) * 100
        r = p_data.get("recall", 0.0) * 100
        f1 = p_data.get("f1", 0.0) * 100
        sup = p_data.get("support", 0)
        print(f"  {cls_name:<26} {p:>8.2f}%    {r:>8.2f}%    {f1:>8.2f}%    {sup:>6}")
    print("-" * 78)

    print("  CONFUSION MATRIX (Raw Counts):")
    raw_cm = m.get("confusion_matrix_raw", {})
    classes = list(per_class.keys())
    header = f"  {'True \\ Pred':<22}" + "".join(f"{c[:10]:>12}" for c in classes)
    print(header)
    for true_lbl in classes:
        row = f"  {true_lbl[:20]:<22}"
        for pred_lbl in classes:
            cnt = raw_cm.get(true_lbl, {}).get(pred_lbl, 0)
            row += f"{cnt:>12}"
        print(row)

    print("-" * 78)
    print("  ARTIFACT OUTPUTS:")
    print(f"  Predictions (JSONL):    {jsonl_path}")
    print(f"  Summary Metrics (JSON): {metrics_path}")
    print("=" * 78 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="MSc Benchmark Evaluation Harness for NewsCLIPpings & Factify 2."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["newsclippings", "factify2"],
        help="Target benchmark dataset name."
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["task_a", "task_b", "task_c", "TASK_A", "TASK_B", "TASK_C"],
        help="Evaluation task (task_a for newsclippings, task_b or task_c for factify2)."
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split (default: 'test' for newsclippings, 'val' for factify2)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (e.g. 5 for smoke test, 100 or 200 for full)."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional explicit path to external dataset directory or file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/benchmark_results",
        help="Directory to save predictions JSONL and metrics JSON."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic pseudo-random seed for stratified sampling (default: 42)."
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="c2",
        choices=["c1", "c2"],
        help="Evaluation protocol for Task C (c2 = 3-class epistemic, c1 = 5-class native)."
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        choices=["none", "no_decomp", "no_rel_filter", "text_only", "forced_binary"],
        help="Component ablation configuration to execute."
    )
    parser.add_argument(
        "--threshold-cm",
        type=float,
        default=0.22,
        help="OpenCLIP cross-modal mismatch cutoff (default: 0.22)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = args.dataset.lower()

    # Default task based on dataset
    if args.task is None:
        task_id = "TASK_A" if dataset_name == "newsclippings" else "TASK_B"
    else:
        task_id = args.task.upper()

    # Validate task compatibility
    if dataset_name == "newsclippings" and task_id != "TASK_A":
        print(f"[ERROR] NewsCLIPpings only supports TASK_A (got {task_id}). Exiting.")
        sys.exit(1)
    if dataset_name == "factify2" and task_id == "TASK_A":
        print(f"[ERROR] Factify 2 does not support TASK_A (use TASK_B or TASK_C). Exiting.")
        sys.exit(1)

    # Default split
    split = args.split or ("test" if dataset_name == "newsclippings" else "val")

    print(f"[INIT] Initializing evaluation pipeline on device CPU...")
    pipeline = MultimodalVerificationPipeline()

    if dataset_name == "newsclippings":
        adapter = NewsCLIPpingsAdapter()
    else:
        adapter = Factify2Adapter()

    harness = BenchmarkHarness(output_dir=args.output_dir)

    print(f"[RUN] Starting evaluation: dataset={dataset_name}, task={task_id}, split={split}, limit={args.limit}, seed={args.seed}")
    summary, records = harness.run_evaluation(
        adapter=adapter,
        pipeline=pipeline,
        task_id=task_id,
        split=split,
        limit=args.limit,
        seed=args.seed,
        data_path=args.data_path,
        config_params={
            "protocol": args.protocol,
            "ablation": args.ablation,
            "threshold_cm": args.threshold_cm,
        }
    )

    print_summary_table(summary, Path(args.output_dir))


if __name__ == "__main__":
    main()
