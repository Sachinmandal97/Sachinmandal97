import json
from pathlib import Path
import sys
from collections import Counter
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_newsclippings

def main():
    data_dir = Path("data/raw/newsclippings")
    samples = load_newsclippings(data_dir, split="test")
    total = len(samples)
    print(f"Total test samples loaded: {total}")
    
    pristine_count = sum(1 for s in samples if s.raw_label == "pristine")
    falsified_count = sum(1 for s in samples if s.raw_label == "falsified")
    print(f"Pristine: {pristine_count} ({pristine_count/total*100:.2f}%)")
    print(f"Falsified: {falsified_count} ({falsified_count/total*100:.2f}%)")

    # Embedding checks
    has_text_emb = sum(1 for s in samples if "clip_text_embedding" in s.metadata)
    has_img_emb = sum(1 for s in samples if "clip_image_embedding" in s.metadata)
    print(f"Text embeddings attached: {has_text_emb} / {total} ({has_text_emb/total*100:.2f}%)")
    print(f"Image embeddings attached: {has_img_emb} / {total} ({has_img_emb/total*100:.2f}%)")

    # Generator breakdown
    gen_counter = Counter(s.metadata.get("source_dataset") for s in samples)
    gen_names = {
        0: "semantics_clip_text_image",
        1: "semantics_clip_text_text",
        2: "scene_resnet_place",
        3: "person_sbert_text_text"
    }
    print("\nGenerator Breakdown:")
    for g_id, count in sorted(gen_counter.items()):
        name = gen_names.get(g_id, str(g_id))
        p_c = sum(1 for s in samples if s.metadata.get("source_dataset") == g_id and s.raw_label == "pristine")
        f_c = sum(1 for s in samples if s.metadata.get("source_dataset") == g_id and s.raw_label == "falsified")
        print(f"  Gen {g_id} ({name}): Total={count} ({count/total*100:.1f}%), Pristine={p_c}, Falsified={f_c}")

    # Inspect first 3 samples
    print("\nFirst 3 test samples:")
    for i, s in enumerate(samples[:3]):
        t_emb = s.metadata.get("clip_text_embedding")
        i_emb = s.metadata.get("clip_image_embedding")
        sim = float(np.dot(t_emb, i_emb) / (np.linalg.norm(t_emb) * np.linalg.norm(i_emb)))
        print(f"  [{i}] ID: {s.sample_id} | Label: {s.raw_label} | CosSim: {sim:.4f}")
        print(f"      Claim: \"{s.claim_text[:60]}...\"")
        print(f"      Title: \"{s.metadata.get('title')}\"")

if __name__ == "__main__":
    main()
