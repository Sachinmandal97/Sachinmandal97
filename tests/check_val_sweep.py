import json
import pickle
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation_harness import calculate_classification_metrics

def main():
    # Load validation data
    val_meta = json.load(open("data/raw/newsclippings/merged_balanced/val.json", encoding="utf-8"))
    ann = val_meta["annotations"]
    print(f"Total Validation Annotations: {len(ann)}")

    text_emb = pickle.load(open("data/raw/newsclippings/embeddings/clip_text_embeddings/clip_text_embeddings_val.pkl", "rb"))
    img_emb = pickle.load(open("data/raw/newsclippings/embeddings/clip_image_embeddings/clip_image_embeddings_val.pkl", "rb"))
    print(f"Loaded Text Embeddings: {len(text_emb)}, Image Embeddings: {len(img_emb)}")

    # Compute cosine similarities for all validation samples
    y_true = []
    scores = []
    skipped = 0

    for a in ann:
        label = "pristine" if a["falsified"] is False else "falsified"
        t_id = a["id"]
        i_id = a["image_id"]
        if t_id in text_emb and i_id in img_emb:
            t_vec = text_emb[t_id]
            i_vec = img_emb[i_id]
            # normalized dot product
            t_norm = t_vec / (np.linalg.norm(t_vec) + 1e-9)
            i_norm = i_vec / (np.linalg.norm(i_vec) + 1e-9)
            sim = float(np.dot(t_norm, i_norm))
            y_true.append(label)
            scores.append(sim)
        else:
            skipped += 1

    print(f"Evaluated {len(scores)} validation samples (skipped: {skipped})")
    
    # Sweep thresholds on full validation split
    thresholds = [0.20, 0.22, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.32]
    print("\n" + "=" * 90)
    print("THRESHOLD SENSITIVITY SWEEP ON FULL VALIDATION SET (N=7,024)")
    print("=" * 90)
    print(f"{'Tau':<8} {'Accuracy':<10} {'Balanced Acc':<14} {'Macro-F1':<10} {'Pristine Rec':<14} {'Falsified Rec':<14} {'Pristine Prec':<14} {'Falsified Prec':<14}")
    print("-" * 90)

    best_bal = (0, 0)
    best_f1 = (0, 0)

    for tau in thresholds:
        y_pred = ["pristine" if s >= tau else "falsified" for s in scores]
        m = calculate_classification_metrics(y_true, y_pred, labels=["pristine", "falsified"])
        acc = m["accuracy"] * 100
        b_acc = m["balanced_accuracy"] * 100
        f1 = m["macro_f1"] * 100
        p_rec = m["per_class"]["pristine"]["recall"] * 100
        f_rec = m["per_class"]["falsified"]["recall"] * 100
        p_prec = m["per_class"]["pristine"]["precision"] * 100
        f_prec = m["per_class"]["falsified"]["precision"] * 100

        if b_acc > best_bal[1]:
            best_bal = (tau, b_acc)
        if f1 > best_f1[1]:
            best_f1 = (tau, f1)

        print(f"{tau:<8.2f} {acc:>6.2f}%    {b_acc:>8.2f}%     {f1:>6.2f}%    {p_rec:>8.2f}%      {f_rec:>8.2f}%       {p_prec:>8.2f}%        {f_prec:>8.2f}%")

    print("-" * 90)
    print(f"Optimal Threshold for Balanced Accuracy: tau = {best_bal[0]:.2f} ({best_bal[1]:.2f}%)")
    print(f"Optimal Threshold for Macro-F1:          tau = {best_f1[0]:.2f} ({best_f1[1]:.2f}%)")

if __name__ == "__main__":
    main()
