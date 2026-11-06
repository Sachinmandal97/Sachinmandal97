"""
Cross-modal image-text consistency module using OpenCLIP / HuggingFace CLIP.
Measures cosine alignment between image features and caption text representations.
Note: Low similarity indicates potential image-text discrepancy, not proof of misinformation.
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from PIL import Image

from src.schemas import CrossModalScore
from src.utils import setup_logger

logger = setup_logger("cross_modal")

# Lazy-loaded global CLIP model & processor singletons
_CLIP_MODEL = None
_CLIP_PROCESSOR = None
_CLIP_DEVICE = "cpu"


def get_clip_model(model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
    """
    Returns or lazily loads the shared CLIP model and processor.
    """
    global _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_DEVICE
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_DEVICE

    try:
        from transformers import CLIPModel, CLIPProcessor

        logger.info(f"Loading CLIP model: {model_name} on {device}...")
        try:
            # Prefer local cache to prevent proxy/socket timeouts in sandboxed or offline environments
            _CLIP_MODEL = CLIPModel.from_pretrained(model_name, local_files_only=True)
            _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
        except Exception:
            _CLIP_MODEL = CLIPModel.from_pretrained(model_name)
            _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_name)

        _CLIP_DEVICE = device
        _CLIP_MODEL.to(device)
        _CLIP_MODEL.eval()
        logger.info("[OK] CLIP model loaded successfully.")
        return _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_DEVICE
    except Exception as e:
        logger.warning(f"Failed to load CLIP model ({model_name}): {e}. Fallback scoring will be used.")
        return None, None, device


def compute_cross_modal_consistency(
    image: Optional[Image.Image],
    claim_text: str,
    ocr_text: str = "",
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = "cpu",
    mismatch_threshold: float = 0.28
) -> CrossModalScore:
    """
    Computes image-text cosine similarity using CLIP.
    Distinguishes evaluated images, text-only mode (image is None), and unavailable models.
    Never silently fabricates a matching score when visual input is missing or failed.
    """
    if not claim_text:
        return CrossModalScore(
            similarity_score=None,
            mismatch_detected=False,
            has_image=(image is not None),
            status="error",
            details={"error": "Empty claim text"}
        )

    if image is None:
        return CrossModalScore(
            similarity_score=None,
            mismatch_detected=False,
            has_image=False,
            status="text_only",
            details={"info": "No image provided; text-only evaluation mode"}
        )

    model, processor, dev = get_clip_model(model_name=model_name, device=device)

    if model is None or processor is None:
        # Fallback when model weights are not locally available or network is restricted
        return CrossModalScore(
            similarity_score=None,
            mismatch_detected=False,
            has_image=True,
            status="unavailable",
            details={"warning": "CLIP model offline; visual alignment unavailable"}
        )

    try:
        with torch.no_grad():
            inputs = processor(
                text=[claim_text],
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            )
            inputs = {k: v.to(dev) for k, v in inputs.items()}

            # Obtain normalized embeddings
            img_embeds = model.get_image_features(pixel_values=inputs["pixel_values"])
            txt_embeds = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))

            img_embeds = F.normalize(img_embeds, p=2, dim=-1)
            txt_embeds = F.normalize(txt_embeds, p=2, dim=-1)

            # Cosine similarity in range [-1.0, 1.0]
            cosine_sim = float((img_embeds * txt_embeds).sum().cpu().item())
            raw_score = round(cosine_sim, 4)

            # Simple lexical overlap between OCR text and claim text
            ocr_overlap = 0.0
            if ocr_text:
                claim_words = set(claim_text.lower().split())
                ocr_words = set(ocr_text.lower().split())
                if claim_words:
                    ocr_overlap = round(len(claim_words.intersection(ocr_words)) / len(claim_words), 4)

            is_mismatch = raw_score < mismatch_threshold

            return CrossModalScore(
                similarity_score=raw_score,
                mismatch_detected=is_mismatch,
                has_image=True,
                status="evaluated",
                ocr_text_overlap=ocr_overlap,
                details={
                    "raw_cosine": raw_score,
                    "model": model_name,
                    "threshold": mismatch_threshold
                }
            )

    except Exception as e:
        logger.error(f"Error in CLIP cross-modal scoring: {e}")
        return CrossModalScore(
            similarity_score=None,
            mismatch_detected=False,
            has_image=True,
            status="error",
            details={"error": str(e)}
        )
