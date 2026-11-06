import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation_harness import calculate_classification_metrics

def run_sweep_for_file(pred_file_path: Path, n_label: str):
    records = [json.loads(line) for line in pred_file_path.read_text(encoding="utf-8").strip().split("\n")]
    y_true = [r["ground_truth_mapped"] for r in records]
    scores = [r["cross_modal_score"] for r in records]

    thresholds = [0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.29, 0.30, 0.32]
    print(f"\n--- Threshold Sweep: {n_label} (N={len(records)}) ---")
    print(f"{'Threshold':<10} {'Accuracy':<10} {'Balanced Acc':<14} {'Macro-F1':<10} {'Pristine Rec':<14} {'Falsified Rec':<14}")
    print("-" * 74)

    sweep_results = []
    for tau in thresholds:
        y_pred = ["pristine" if s >= tau else "falsified" for s in scores]
        m = calculate_classification_metrics(y_true, y_pred, labels=["pristine", "falsified"])
        acc = m["accuracy"]
        b_acc = m["balanced_accuracy"]
        f1 = m["macro_f1"]
        p_rec = m["per_class"]["pristine"]["recall"]
        f_rec = m["per_class"]["falsified"]["recall"]
        sweep_results.append({
            "threshold": tau,
            "accuracy": acc,
            "balanced_accuracy": b_acc,
            "macro_f1": f1,
            "pristine_recall": p_rec,
            "falsified_recall": f_rec,
            "raw_cm": m["confusion_matrix_raw"]
        })
        print(f"{tau:<10.2f} {acc*100:>6.2f}%    {b_acc*100:>8.2f}%     {f1*100:>6.2f}%    {p_rec*100:>8.2f}%      {f_rec*100:>8.2f}%")

    out_file = pred_file_path.parent / "threshold_sensitivity.json"
    out_file.write_text(json.dumps(sweep_results, indent=2), encoding="utf-8")
    print(f"Saved threshold sensitivity results to {out_file}")


def main():
    p100 = Path("outputs/benchmark_results/newsclippings_val_100/newsclippings_task_a_predictions.jsonl")
    if p100.exists():
        run_sweep_for_file(p100, "100-Sample Validation Set")

    p200 = Path("outputs/benchmark_results/newsclippings_val_200/newsclippings_task_a_predictions.jsonl")
    if p200.exists():
        run_sweep_for_file(p200, "200-Sample Validation Set")


if __name__ == "__main__":
    main()
