"""
OCR extraction module using EasyOCR with graceful fallback handling.
Extracts overlay text and spatial bounding boxes from images without fabricating output.
"""

from typing import List, Optional
import numpy as np
from PIL import Image

from src.schemas import OCRResult
from src.utils import setup_logger

logger = setup_logger("ocr")

# Lazy-loaded global reader singleton to avoid reloading weights for every sample
_EASYOCR_READER = None


def get_ocr_reader(languages: Optional[List[str]] = None, gpu: bool = False):
    """
    Returns or initializes the shared EasyOCR Reader instance.
    """
    global _EASYOCR_READER
    if _EASYOCR_READER is not None:
        return _EASYOCR_READER

    try:
        import easyocr
        langs = languages or ["en"]
        logger.info(f"Initializing EasyOCR reader for languages: {langs} (gpu={gpu})")
        _EASYOCR_READER = easyocr.Reader(langs, gpu=gpu, verbose=False)
        return _EASYOCR_READER
    except ImportError:
        logger.warning("easyocr package is not installed. OCR will be disabled.")
        return None
    except Exception as e:
        logger.warning(f"EasyOCR initialization failed: {e}. OCR will be disabled.")
        return None


def extract_ocr_text(
    image: Optional[Image.Image],
    languages: Optional[List[str]] = None,
    gpu: bool = False
) -> OCRResult:
    """
    Extracts text and bounding boxes from an input PIL Image.
    Attempts EasyOCR first, falling back to pytesseract, and finally returns empty OCR gracefully.
    """
    if image is None:
        return OCRResult(extracted_text="", boxes=[], has_text=False)

    # 1. Try EasyOCR
    reader = get_ocr_reader(languages=languages, gpu=gpu)
    if reader is not None:
        try:
            img_array = np.array(image)
            results = reader.readtext(img_array)

            extracted_lines = []
            boxes = []
            for bbox, text, conf in results:
                clean_str = text.strip()
                if clean_str:
                    extracted_lines.append(clean_str)
                    box_coords = [[int(pt[0]), int(pt[1])] for pt in bbox]
                    boxes.append({
                        "text": clean_str,
                        "box": box_coords,
                        "confidence": round(float(conf), 4)
                    })

            full_text = " ".join(extracted_lines)
            if full_text:
                return OCRResult(
                    extracted_text=full_text,
                    boxes=boxes,
                    has_text=True
                )
        except Exception as e:
            logger.debug(f"EasyOCR extraction attempt failed: {e}")

    # 2. Try pytesseract as lightweight fallback
    try:
        import pytesseract
        text = pytesseract.image_to_string(image).strip()
        if text:
            return OCRResult(
                extracted_text=text,
                boxes=[],
                has_text=True
            )
    except Exception as e:
        logger.debug(f"Pytesseract fallback skipped/unavailable: {e}")

    # 3. Default empty OCR result when OCR engines are unconfigured
    return OCRResult(extracted_text="", boxes=[], has_text=False)

