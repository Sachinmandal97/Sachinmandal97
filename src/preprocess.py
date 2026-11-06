"""
Preprocessing module for text and image inputs.
Performs deterministic string normalization, whitespace stripping, and PIL image validation.
"""

import re
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image


def clean_text(text: Optional[str]) -> str:
    """
    Normalizes input claim/caption text:
    - Strips leading/trailing whitespace
    - Collapses multiple whitespace characters
    - Normalizes common unicode quotation marks and dashes
    """
    if not text:
        return ""
    
    # Normalize unicode quotes and dashes
    cleaned = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    cleaned = cleaned.replace("—", "-").replace("–", "-")
    
    # Collapse multiple whitespace/tabs/newlines
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def load_and_validate_image(image_input: Optional[str | Path | Image.Image]) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Loads and validates an image from a filepath or PIL Image object.
    Returns:
        (PIL.Image in RGB format or None, error_message or None)
    """
    if image_input is None:
        return None, "No image provided"

    # If already a PIL Image
    if isinstance(image_input, Image.Image):
        try:
            return image_input.convert("RGB"), None
        except Exception as e:
            return None, f"Failed to convert PIL Image to RGB: {e}"

    # If filepath
    path = Path(image_input)
    if not path.exists():
        return None, f"Image file does not exist at path: {path}"

    try:
        with Image.open(path) as img:
            # Fully load image data to prevent truncated file issues
            img.load()
            rgb_img = img.convert("RGB")
            return rgb_img, None
    except Exception as e:
        return None, f"Invalid or corrupted image file: {e}"
