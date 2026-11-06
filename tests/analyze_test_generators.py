import json
from pathlib import Path

def main():
    test_json = json.load(open("data/raw/newsclippings/merged_balanced/test.json", encoding="utf-8"))
    gen_map = {f"nc_test_{a['id']}_{a['image_id']}": a["source_dataset"] for a in test_json["annotations"]}
    gen_names = {
        0: "semantics_clip_text_image",
        1: "semantics_clip_text_text",
        2: "scene_resnet_place",
        3: "person_sbert_text_text"
    }

    for tau_str in ["22", "28"]:
        pred_path = Path(f"outputs/benchmark_results/newsclippings_test_full_tau{tau_str}/newsclippings_task_a_predictions.jsonl")
        lines = pred_path.read_text(encoding="utf-8").strip().split("\n")
        by_gen = {i: {"correct": 0, "total": 0, "p_corr": 0, "p_tot": 0, "f_corr": 0, "f_tot": 0} for i in range(4)}
        for l in lines:
            r = json.loads(l)
            gid = gen_map.get(r["sample_id"])
            if gid is not None:
                by_gen[gid]["total"] += 1
                if r["is_correct"]:
                    by_gen[gid]["correct"] += 1
                if r["ground_truth_mapped"] == "pristine":
                    by_gen[gid]["p_tot"] += 1
                    if r["is_correct"]:
                        by_gen[gid]["p_corr"] += 1
                else:
                    by_gen[gid]["f_tot"] += 1
                    if r["is_correct"]:
                        by_gen[gid]["f_corr"] += 1

        print(f"\n=== Full Test Generator Breakdown (tau = 0.{tau_str}) ===")
        for gid in range(4):
            g = by_gen[gid]
            acc = g["correct"] / g["total"] * 100
            p_rec = g["p_corr"] / g["p_tot"] * 100
            f_rec = g["f_corr"] / g["f_tot"] * 100
            print(f"  Gen {gid} ({gen_names[gid]:<26}): Acc = {acc:>6.2f}% | Pristine Rec = {p_rec:>6.2f}% | Falsified Rec = {f_rec:>6.2f}% | N = {g['total']}")

if __name__ == "__main__":
    main()
