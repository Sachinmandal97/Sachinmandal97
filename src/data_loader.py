"""
Data loading and dataset ingestion module.
Provides clean loading of NewsCLIPpings samples into standardized MultimodalSample objects.
Designed so Factify 2 and MMFakeBench loaders can be added cleanly later without rewrites.
"""

import csv
import json
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Union
from PIL import Image, ImageDraw, ImageFont

from src.schemas import DecisionVerdict, Factify2Sample, MultimodalSample



def get_demo_newsclippings_samples(data_dir: Optional[Path] = None) -> List[MultimodalSample]:
    """
    Returns a small curated set of 5 realistic multimodal news samples
    with real-world text captions, synthetic image generation for offline testing,
    and ground-truth out-of-context mismatch labels.
    """
    base_dir = data_dir or Path("./data/raw/news_clippings")
    images_dir = base_dir / "sample_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 5 curated test cases (3 pristine/consistent, 2 falsified/mismatched)
    curated_data = [
        {
            "id": "nc_demo_01",
            "caption": "Firefighters battle a major industrial warehouse blaze in East London on Sunday morning.",
            "image_filename": "fire_london.jpg",
            "image_text": "LONDON FIRE BRIGADE",
            "image_desc": "Firefighters spraying water at burning building",
            "raw_label": "pristine",
            "falsification_type": "none"
        },
        {
            "id": "nc_demo_02",
            "caption": "Prime Minister addresses delegates at the annual economic summit in Geneva regarding inflation.",
            "image_filename": "summit_geneva.jpg",
            "image_text": "WORLD ECONOMIC FORUM GENEVA",
            "image_desc": "Leader speaking at podium with international flags",
            "raw_label": "pristine",
            "falsification_type": "none"
        },
        {
            "id": "nc_demo_03",
            "caption": "Massive blizzard paralyzes public transportation across downtown Chicago with over 2 feet of snow.",
            "image_filename": "beach_summer.jpg",
            "image_text": "MIAMI BEACH RESORT",
            "image_desc": "Sunny beach with palm trees and ocean waves",
            "raw_label": "falsified",
            "falsification_type": "semantics_clip_image"
        },
        {
            "id": "nc_demo_04",
            "caption": "Protesters gather outside City Hall demanding climate action and clean energy investments.",
            "image_filename": "climate_rally.jpg",
            "image_text": "ACT ON CLIMATE NOW",
            "image_desc": "Crowd holding climate protest banners",
            "raw_label": "pristine",
            "falsification_type": "none"
        },
        {
            "id": "nc_demo_05",
            "caption": "Historic floods submerge residential streets in Venice following record-high tides.",
            "image_filename": "desert_dunes.jpg",
            "image_text": "SAHARA EXPEDITION",
            "image_desc": "Dry desert sand dunes under hot sun",
            "raw_label": "falsified",
            "falsification_type": "semantics_clip_text"
        },
    ]

    samples: List[MultimodalSample] = []

    for item in curated_data:
        img_path = images_dir / item["image_filename"]
        
        # If image does not exist on disk, create a simple legible PIL test image with embedded text
        if not img_path.exists():
            img = Image.new("RGB", (380, 240), color=(40, 60, 90) if item["raw_label"] == "pristine" else (120, 50, 40))
            draw = ImageDraw.Draw(img)
            # Draw simple text overlay for OCR and visual tests
            draw.text((20, 40), f"[SCENE]: {item['image_desc'][:30]}", fill=(240, 240, 240))
            draw.text((20, 100), f"OCR OVERLAY: {item['image_text']}", fill=(255, 255, 100))
            draw.rectangle([(15, 15), (365, 225)], outline=(200, 200, 200), width=2)
            img.save(img_path)

        sample = MultimodalSample(
            sample_id=item["id"],
            claim_text=item["caption"],
            image_path=str(img_path),
            raw_label=item["raw_label"],
            dataset_name="newsclippings",
            split="demo",
            metadata={
                "falsification_type": item["falsification_type"],
                "expected_ocr": item["image_text"]
            }
        )
        samples.append(sample)

    return samples


def load_newsclippings(
    data_dir: Path,
    split: str = "test",
    limit: Optional[int] = None
) -> List[MultimodalSample]:
    """
    Loads NewsCLIPpings dataset from raw json splits or official merged/embeddings layout,
    falling back to the verified demo sample set for quick testing.

    Supported directory layouts:
      1. Official benchmark format:
         data_dir / 'merged_balanced' / f'{split}.json'
         data_dir / 'metadata' / f'{split}.json'
         data_dir / 'embeddings' / 'clip_text_embeddings' / f'clip_text_embeddings_{split}.pkl'
         data_dir / 'embeddings' / 'clip_image_embeddings' / f'clip_image_embeddings_{split}.pkl'
      2. Flat JSON format:
         data_dir / f'news_clippings_{split}.json'
      3. Curated demo fallback.
    """
    data_path = Path(data_dir)

    # Resolve candidate directory if newsclippings / news_clippings variant exists
    candidate_paths = [data_path]
    if data_path.name == "news_clippings":
        candidate_paths.append(data_path.parent / "newsclippings")
    elif data_path.name == "newsclippings":
        candidate_paths.append(data_path.parent / "news_clippings")

    target_dir = None
    for p in candidate_paths:
        if (p / "merged_balanced" / f"{split}.json").exists() or (p / f"news_clippings_{split}.json").exists():
            target_dir = p
            break
    if target_dir is None:
        target_dir = data_path

    # Check 1: Official benchmark merged_balanced structure
    merged_split_file = target_dir / "merged_balanced" / f"{split}.json"
    if merged_split_file.exists():
        with open(merged_split_file, "r", encoding="utf-8") as f:
            merged_data = json.load(f)

        raw_annotations = merged_data.get("annotations", []) if isinstance(merged_data, dict) else merged_data

        # Load metadata if present
        metadata_map: Dict[str, Dict[str, Any]] = {}
        meta_file = target_dir / "metadata" / f"{split}.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata_map = json.load(f)

        # Load precomputed CLIP embeddings if present
        text_embeddings: Dict[int, Any] = {}
        t_emb_file = target_dir / "embeddings" / "clip_text_embeddings" / f"clip_text_embeddings_{split}.pkl"
        if t_emb_file.exists():
            with open(t_emb_file, "rb") as f:
                text_embeddings = pickle.load(f)

        image_embeddings: Dict[int, Any] = {}
        i_emb_file = target_dir / "embeddings" / "clip_image_embeddings" / f"clip_image_embeddings_{split}.pkl"
        if i_emb_file.exists():
            with open(i_emb_file, "rb") as f:
                image_embeddings = pickle.load(f)

        samples: List[MultimodalSample] = []
        for idx, item in enumerate(raw_annotations):
            if limit and idx >= limit:
                break

            cap_id = item.get("id")
            img_id = item.get("image_id", cap_id)
            is_falsified = bool(item.get("falsified", False))
            raw_label = "falsified" if is_falsified else "pristine"

            # Extract caption text from metadata map or annotation
            meta_entry = metadata_map.get(str(cap_id), {}) if metadata_map else {}
            caption_text = meta_entry.get("caption", item.get("caption", item.get("text", "")))
            img_path = meta_entry.get("image_path", item.get("image_path"))

            sample_metadata: Dict[str, Any] = {
                "falsified": is_falsified,
                "source_dataset": item.get("source_dataset"),
                "similarity_score": item.get("similarity_score"),
                "caption_id": cap_id,
                "image_id": img_id,
            }
            if meta_entry:
                sample_metadata["title"] = meta_entry.get("title")
                sample_metadata["source"] = meta_entry.get("source")
                sample_metadata["timestamp"] = meta_entry.get("timestamp")
                sample_metadata["entities"] = meta_entry.get("caption_entities_spacy")

            if cap_id in text_embeddings:
                sample_metadata["clip_text_embedding"] = text_embeddings[cap_id]
            if img_id in image_embeddings:
                sample_metadata["clip_image_embedding"] = image_embeddings[img_id]

            sample = MultimodalSample(
                sample_id=f"nc_{split}_{cap_id}_{img_id}",
                claim_text=caption_text,
                image_path=img_path,
                raw_label=raw_label,
                dataset_name="newsclippings",
                split=split,
                metadata=sample_metadata
            )
            samples.append(sample)
        return samples

    # Check 2: Flat JSON format (news_clippings_{split}.json)
    split_file = target_dir / f"news_clippings_{split}.json"
    if split_file.exists():
        with open(split_file, "r", encoding="utf-8") as f:
            raw_entries = json.load(f)

        entries = raw_entries if isinstance(raw_entries, list) else raw_entries.get("annotations", [])
        samples = []

        for idx, item in enumerate(entries):
            if limit and idx >= limit:
                break

            raw_lbl = str(item.get("falsified", item.get("label", "pristine"))).lower()
            norm_label = "falsified" if raw_lbl in ["true", "1", "falsified"] else "pristine"

            sample = MultimodalSample(
                sample_id=str(item.get("id", f"nc_{split}_{idx}")),
                claim_text=item.get("caption", item.get("text", "")),
                image_path=item.get("image_path"),
                raw_label=norm_label,
                dataset_name="newsclippings",
                split=split,
                metadata=item
            )
            samples.append(sample)
        return samples

    # Fallback: Curated verified demo samples for fast local execution
    demo_samples = get_demo_newsclippings_samples(target_dir)
    return demo_samples[:limit] if limit else demo_samples


# ============================================================================
# FACTIFY 2 DATA ADAPTER & SCHEMA VALIDATOR
# ============================================================================

FACTIFY2_LABEL_MAP: Dict[str, DecisionVerdict] = {
    "support_multimodal": DecisionVerdict.SUPPORTED,
    "support_text": DecisionVerdict.SUPPORTED,
    "refute": DecisionVerdict.MISLEADING,
    "neutral": DecisionVerdict.UNCERTAIN,
    "insufficient_multimodal": DecisionVerdict.UNCERTAIN,
}


def normalize_factify2_label(raw_label: str) -> DecisionVerdict:
    """
    Maps a raw Factify 2 label into the standard internal three-state decision:
      - Support_Multimodal -> DecisionVerdict.SUPPORTED
      - Support_Text       -> DecisionVerdict.SUPPORTED
      - Refute             -> DecisionVerdict.MISLEADING
      - Neutral            -> DecisionVerdict.UNCERTAIN
      - Insufficient_Multimodal -> DecisionVerdict.UNCERTAIN

    Preserves the original label in the source record while returning the normalized verdict.
    Raises ValueError on unrecognized labels to prevent silent corruption.
    """
    if not isinstance(raw_label, str) or not raw_label.strip():
        raise ValueError(f"Factify 2 label must be a non-empty string, got: {raw_label!r}")

    normalized_key = raw_label.strip().lower()
    if normalized_key in FACTIFY2_LABEL_MAP:
        return FACTIFY2_LABEL_MAP[normalized_key]

    valid_labels = ["Support_Multimodal", "Support_Text", "Refute", "Neutral", "Insufficient_Multimodal"]
    raise ValueError(
        f"Unrecognized Factify 2 label: '{raw_label}'. Expected one of: {', '.join(valid_labels)}"
    )


def validate_factify2_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates a raw Factify 2 record dictionary against the expected schema.
    
    Required fields:
      - sample_id (or 'id' / 'Id' / 'ID')
      - claim_text (or 'claim')
      - label (or 'category' / 'Category')

    Optional fields:
      - claim_image (or 'claim_image_path')
      - claim_ocr
      - document_text (or 'document')
      - document_image (or 'document_image_path')
      - document_ocr

    Raises ValueError with a descriptive message if required fields are missing or empty.
    Does not silently invent missing values.
    """
    if not isinstance(record, dict):
        raise ValueError(f"Factify 2 record must be a dict, got: {type(record).__name__}")

    # Extract sample ID
    sample_id = (
        record.get("sample_id")
        or record.get("id")
        or record.get("Id")
        or record.get("ID")
    )
    if sample_id is None or not str(sample_id).strip():
        raise ValueError(f"Factify 2 record missing required identifier ('sample_id', 'id', or 'Id'): {record}")

    # Extract claim text
    claim_text = record.get("claim_text") or record.get("claim") or record.get("claim_caption")
    if claim_text is None or not str(claim_text).strip():
        raise ValueError(f"Factify 2 record '{sample_id}' missing required non-empty field 'claim_text'.")

    # Extract label
    label = record.get("label") or record.get("category") or record.get("Category")
    if label is None or not str(label).strip():
        raise ValueError(f"Factify 2 record '{sample_id}' missing required non-empty field 'label'/'category'.")

    # Optional fields
    claim_image = record.get("claim_image") or record.get("claim_image_path")
    claim_ocr = record.get("claim_ocr")
    document_text = record.get("document_text") or record.get("document")
    document_image = record.get("document_image") or record.get("document_image_path")
    document_ocr = record.get("document_ocr")

    # Store any extra metadata
    known_keys = {
        "sample_id", "id", "Id", "ID",
        "claim_text", "claim", "claim_caption",
        "label", "category", "Category",
        "claim_image", "claim_image_path", "claim_ocr",
        "document_text", "document", "document_image", "document_image_path", "document_ocr",
        "split"
    }
    source_metadata = {k: v for k, v in record.items() if k not in known_keys}

    return {
        "sample_id": str(sample_id).strip(),
        "claim_text": str(claim_text).strip(),
        "label": str(label).strip(),
        "claim_image": str(claim_image).strip() if claim_image else None,
        "claim_ocr": str(claim_ocr).strip() if claim_ocr else None,
        "document_text": str(document_text).strip() if document_text else None,
        "document_image": str(document_image).strip() if document_image else None,
        "document_ocr": str(document_ocr).strip() if document_ocr else None,
        "split": str(record.get("split", "val")).strip(),
        "source_metadata": source_metadata,
    }


def load_factify2_metadata(
    path: Union[str, Path],
    split: str = "val",
    limit: Optional[int] = None
) -> List[Factify2Sample]:
    """
    Loads and validates Factify 2 metadata from a local JSON, JSONL, TSV, or CSV file.
    
    Returns a list of structured Factify2Sample objects.
    Raises FileNotFoundError if the file does not exist.
    Raises ValueError if records fail schema validation.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Factify 2 metadata file not found at: {file_path}")

    raw_records: List[Dict[str, Any]] = []

    suffix = file_path.suffix.lower()
    if suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                raw_records = data
            elif isinstance(data, dict):
                raw_records = data.get("data", data.get("samples", [data]))
            else:
                raise ValueError(f"Unexpected JSON structure in {file_path}")
    elif suffix == ".jsonl":
        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if line:
                    try:
                        raw_records.append(json.loads(line))
                    except json.JSONDecodeError as err:
                        raise ValueError(f"Malformed JSONL at line {line_no} in {file_path}: {err}")
    elif suffix in [".tsv", ".csv"]:
        delimiter = "\t" if suffix == ".tsv" else ","
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            raw_records = list(reader)
    else:
        raise ValueError(f"Unsupported Factify 2 file extension '{suffix}'. Expected .json, .jsonl, .tsv, or .csv")

    validated_samples: List[Factify2Sample] = []
    for idx, raw in enumerate(raw_records):
        if limit and idx >= limit:
            break
        
        # Merge split if not present
        if "split" not in raw:
            raw["split"] = split

        validated_dict = validate_factify2_record(raw)
        sample = Factify2Sample(**validated_dict)
        validated_samples.append(sample)

    return validated_samples


def get_demo_factify2_samples(data_dir: Optional[Path] = None) -> List[Factify2Sample]:
    """
    Returns a curated set of 5 realistic Factify 2 multimodal verification samples
    covering all 5 benchmark ground-truth classes with paired documents and synthetic images.
    """
    base_dir = data_dir or Path("./data/raw/factify2")
    images_dir = base_dir / "sample_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    curated_data = [
        {
            "sample_id": "f2_demo_01",
            "claim_text": "James Webb Space Telescope captures detailed infrared view of the Pillars of Creation.",
            "document_text": "NASA's James Webb Space Telescope has captured a lush, highly detailed view of the Pillars of Creation in near-infrared light, revealing new stars forming inside dense gas clouds.",
            "label": "Support_Multimodal",
            "claim_image_name": "jwst_pillars.jpg",
            "doc_image_name": "jwst_reference.jpg",
            "image_desc": "Infrared space observatory view of cosmic dust pillars"
        },
        {
            "sample_id": "f2_demo_02",
            "claim_text": "Bank of England raises benchmark interest rates by 50 basis points to curb inflation.",
            "document_text": "The Monetary Policy Committee of the Bank of England voted to raise the benchmark interest rate by 50 basis points as inflation remained above target.",
            "label": "Support_Text",
            "claim_image_name": "stadium_crowd.jpg",
            "doc_image_name": "bank_england_hq.jpg",
            "image_desc": "Crowd of sports spectators at a football stadium"
        },
        {
            "sample_id": "f2_demo_03",
            "claim_text": "World Health Organization announces that drinking hot lemon juice completely cures COVID-19 within 24 hours.",
            "document_text": "The World Health Organization confirmed there is no medical evidence that drinking hot lemon juice or herbal beverages cures COVID-19, warning against unverified remedies.",
            "label": "Refute",
            "claim_image_name": "lemon_juice.jpg",
            "doc_image_name": "who_statement.jpg",
            "image_desc": "Glass of lemon juice on a wooden table"
        },
        {
            "sample_id": "f2_demo_04",
            "claim_text": "Electric vehicle market share in Norway exceeded eighty percent of all new passenger cars.",
            "document_text": "European automotive regulators met to discuss general clean energy transportation targets and regional lithium battery supply chains across Nordic economies.",
            "label": "Neutral",
            "claim_image_name": "ev_charging.jpg",
            "doc_image_name": "nordic_conference.jpg",
            "image_desc": "Electric car plugged into an urban charging station"
        },
        {
            "sample_id": "f2_demo_05",
            "claim_text": "Secret diplomatic ceasefire negotiations were conducted at an undisclosed airfield in Zurich on Tuesday.",
            "document_text": "Zurich International Airport reported routine runway repaving operations during the week, with minor adjustments to commercial flight scheduling.",
            "label": "Insufficient_Multimodal",
            "claim_image_name": "private_hangar.jpg",
            "doc_image_name": "commercial_runway.jpg",
            "image_desc": "Private aviation hangar at a misty airfield"
        },
    ]

    samples: List[Factify2Sample] = []
    for item in curated_data:
        c_img_path = images_dir / item["claim_image_name"]
        d_img_path = images_dir / item["doc_image_name"]

        # Ensure synthetic images exist for multimodal tests
        if not c_img_path.exists():
            img = Image.new("RGB", (380, 240), color=(50, 70, 100))
            draw = ImageDraw.Draw(img)
            draw.text((20, 50), f"[CLAIM IMG]: {item['image_desc'][:30]}", fill=(240, 240, 240))
            draw.rectangle([(15, 15), (365, 225)], outline=(200, 200, 200), width=2)
            img.save(c_img_path)

        if not d_img_path.exists():
            img = Image.new("RGB", (380, 240), color=(70, 90, 60))
            draw = ImageDraw.Draw(img)
            draw.text((20, 50), f"[DOC IMG]: Reference asset", fill=(240, 240, 240))
            draw.rectangle([(15, 15), (365, 225)], outline=(180, 180, 180), width=2)
            img.save(d_img_path)

        sample = Factify2Sample(
            sample_id=item["sample_id"],
            claim_text=item["claim_text"],
            claim_image=str(c_img_path),
            document_text=item["document_text"],
            document_image=str(d_img_path),
            label=item["label"],
            split="demo",
            source_metadata={"curated_demo": True}
        )
        samples.append(sample)

    return samples


def load_factify2_samples(
    path: Optional[Union[str, Path]] = None,
    split: str = "val",
    limit: Optional[int] = None
) -> List[Factify2Sample]:
    """
    Loads Factify 2 samples from an external file if provided and existing,
    otherwise gracefully falls back to the curated 5-sample demo set.
    """
    if path is not None and Path(path).exists():
        return load_factify2_metadata(path=path, split=split, limit=limit)

    demo_samples = get_demo_factify2_samples()
    if limit:
        return demo_samples[:limit]
    return demo_samples


