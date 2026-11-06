import json
from pathlib import Path

def main():
    test_json = json.load(open("data/raw/newsclippings/merged_balanced/test.json", encoding="utf-8"))
    gen_map = {f"nc_test_{a['id']}_{a['image_id']}": a for a in test_json["annotations"]}

    pred_file = Path("outputs/benchmark_results/newsclippings_test_full_tau28/newsclippings_task_a_predictions.jsonl")
    lines = [json.loads(l) for l in pred_file.read_text(encoding="utf-8").strip().split("\n")]

    gen0_samples = []
    for r in lines:
        meta = gen_map.get(r["sample_id"])
        if meta and meta["source_dataset"] == 0:
            gen0_samples.append((r, meta))

    print("Total Gen 0 samples:", len(gen0_samples))
    p_scores = [r["cross_modal_score"] for r, m in gen0_samples if r["ground_truth_mapped"] == "pristine"]
    f_scores = [r["cross_modal_score"] for r, m in gen0_samples if r["ground_truth_mapped"] == "falsified"]

    print(f"Pristine (N={len(p_scores)}): mean={sum(p_scores)/len(p_scores):.4f}, median={sorted(p_scores)[len(p_scores)//2]:.4f}, min={min(p_scores):.4f}, max={max(p_scores):.4f}")
    print(f"Falsified (N={len(f_scores)}): mean={sum(f_scores)/len(f_scores):.4f}, median={sorted(f_scores)[len(f_scores)//2]:.4f}, min={min(f_scores):.4f}, max={max(f_scores):.4f}")

    fp = [(r, m) for r, m in gen0_samples if r["ground_truth_mapped"] == "falsified" and r["predicted_verdict"] == "pristine"]
    fn = [(r, m) for r, m in gen0_samples if r["ground_truth_mapped"] == "pristine" and r["predicted_verdict"] == "falsified"]
    tp = [(r, m) for r, m in gen0_samples if r["ground_truth_mapped"] == "pristine" and r["predicted_verdict"] == "pristine"]
    tn = [(r, m) for r, m in gen0_samples if r["ground_truth_mapped"] == "falsified" and r["predicted_verdict"] == "falsified"]

    print(f"Gen 0 Confusion: TP={len(tp)}, FN={len(fn)}, FP={len(fp)}, TN={len(tn)}")
    print(f"False Positives (Falsified predicted Pristine): {len(fp)} / {len(f_scores)} ({len(fp)/len(f_scores)*100:.1f}%)")
    print(f"False Negatives (Pristine predicted Falsified): {len(fn)} / {len(p_scores)} ({len(fn)/len(p_scores)*100:.1f}%)")

    raw_meta = json.load(open("data/raw/newsclippings/metadata/test.json", encoding="utf-8"))

    print("\nSample 3 Misclassified Falsified Pairs (Adversarial False Positives):")
    for idx, (r, m) in enumerate(fp[:3], 1):
        cap = raw_meta.get(str(m["id"]), {}).get("caption", "N/A")
        img_cap = raw_meta.get(str(m["image_id"]), {}).get("caption", "N/A")
        print(f"[{idx}] ID: {r['sample_id']} | Score: {r['cross_modal_score']:.4f}")
        print(f"    Claim/Caption: \"{cap}\"")
        print(f"    Mismatched Image Original Caption: \"{img_cap}\"")

    print("\nSample 3 Correctly Flagged Falsified Pairs (True Negatives):")
    for idx, (r, m) in enumerate(tn[:3], 1):
        cap = raw_meta.get(str(m["id"]), {}).get("caption", "N/A")
        img_cap = raw_meta.get(str(m["image_id"]), {}).get("caption", "N/A")
        print(f"[{idx}] ID: {r['sample_id']} | Score: {r['cross_modal_score']:.4f}")
        print(f"    Claim/Caption: \"{cap}\"")
        print(f"    Mismatched Image Original Caption: \"{img_cap}\"")

if __name__ == "__main__":
    main()
