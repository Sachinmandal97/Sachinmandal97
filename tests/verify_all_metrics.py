import json
import math
from pathlib import Path

def compute_metrics(jsonl_path: Path):
    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").strip().split("\n")]
    n = len(records)
    classes = ["pristine", "falsified"]
    
    cm = {"pristine": {"pristine": 0, "falsified": 0}, "falsified": {"pristine": 0, "falsified": 0}}
    correct = 0
    y_true = []
    y_scores = []
    
    for r in records:
        gt = r["ground_truth_mapped"]
        pred = r["predicted_verdict"]
        score = r["cross_modal_score"]
        y_true.append(gt)
        y_scores.append(score)
        cm[gt][pred] += 1
        if gt == pred:
            correct += 1
            
    acc = correct / n
    se = math.sqrt(acc * (1 - acc) / n)
    
    # Wilson 95% Confidence Interval
    z = 1.959963984540054
    denom = 1 + (z**2) / n
    center = (acc + (z**2) / (2 * n)) / denom
    margin = (z * math.sqrt((acc * (1 - acc) / n) + (z**2) / (4 * (n**2)))) / denom
    ci_lower = max(0.0, center - margin)
    ci_upper = min(1.0, center + margin)
    
    # Per-class metrics
    per_class = {}
    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[other][c] for other in classes if other != c)
        fn = sum(cm[c][other] for other in classes if other != c)
        support = sum(cm[c].values())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": support, "tp": tp, "fp": fp, "fn": fn}
        
    macro_prec = sum(per_class[c]["precision"] for c in classes) / len(classes)
    macro_rec = sum(per_class[c]["recall"] for c in classes) / len(classes)
    macro_f1 = sum(per_class[c]["f1"] for c in classes) / len(classes)
    balanced_acc = macro_rec
    
    # ROC-AUC via rank-based Mann-Whitney U (positive = pristine)
    binary_pairs = [(s, 1 if yt == "pristine" else 0) for s, yt in zip(y_scores, y_true)]
    binary_pairs.sort(key=lambda x: x[0])
    
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and binary_pairs[j + 1][0] == binary_pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
        
    n_pos = sum(1 for yt in y_true if yt == "pristine")
    n_neg = n - n_pos
    sum_ranks_pos = sum(rank for rank, (_, is_pos) in zip(ranks, binary_pairs) if is_pos == 1)
    u_stat = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
    auc = u_stat / (n_pos * n_neg)
    
    return {
        "n": n,
        "acc": acc,
        "se": se,
        "ci": (ci_lower, ci_upper),
        "balanced_acc": balanced_acc,
        "macro_prec": macro_prec,
        "macro_rec": macro_rec,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "cm": cm,
        "roc_auc": auc
    }

def main():
    runs = [
        ("Full Test (tau=0.22)", Path("outputs/benchmark_results/newsclippings_test_full_tau22/newsclippings_task_a_predictions.jsonl")),
        ("Full Test (tau=0.28)", Path("outputs/benchmark_results/newsclippings_test_full_tau28/newsclippings_task_a_predictions.jsonl")),
        ("Test 200 (tau=0.22)", Path("outputs/benchmark_results/newsclippings_test_200_tau22/newsclippings_task_a_predictions.jsonl")),
        ("Test 200 (tau=0.28)", Path("outputs/benchmark_results/newsclippings_test_200_tau28/newsclippings_task_a_predictions.jsonl")),
        ("Val 200 (tau=0.22)", Path("outputs/benchmark_results/newsclippings_val_200/newsclippings_task_a_predictions.jsonl")),
        ("Val 100 (tau=0.22)", Path("outputs/benchmark_results/newsclippings_val_100/newsclippings_task_a_predictions.jsonl")),
    ]
    
    print("=" * 90)
    print("INDEPENDENT RECALCULATION OF METRICS DIRECTLY FROM PREDICTION JSONL FILES")
    print("=" * 90)
    
    for name, p in runs:
        m = compute_metrics(p)
        print(f"\n[{name}] -> {p}")
        print(f"  N: {m['n']} | Accuracy: {m['acc']*100:.2f}% | SE: {m['se']:.4f} | 95% Wilson CI: [{m['ci'][0]*100:.2f}%, {m['ci'][1]*100:.2f}%]")
        print(f"  Balanced Acc: {m['balanced_acc']*100:.2f}% | Macro-F1: {m['macro_f1']*100:.2f}% | Macro-Prec: {m['macro_prec']*100:.2f}% | Macro-Rec: {m['macro_rec']*100:.2f}%")
        print(f"  Pristine:  Prec={m['per_class']['pristine']['precision']*100:.2f}%, Rec={m['per_class']['pristine']['recall']*100:.2f}%, F1={m['per_class']['pristine']['f1']*100:.2f}%, Support={m['per_class']['pristine']['support']}")
        print(f"  Falsified: Prec={m['per_class']['falsified']['precision']*100:.2f}%, Rec={m['per_class']['falsified']['recall']*100:.2f}%, F1={m['per_class']['falsified']['f1']*100:.2f}%, Support={m['per_class']['falsified']['support']}")
        print(f"  Raw Confusion Matrix: True Pristine: {m['cm']['pristine']} | True Falsified: {m['cm']['falsified']}")
        print(f"  Empirical ROC-AUC: {m['roc_auc']:.4f}")

if __name__ == "__main__":
    main()
