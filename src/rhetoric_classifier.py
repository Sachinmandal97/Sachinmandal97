"""
VERITAS-MM: Multimodal Rhetorical Manipulation Classifier (VLDBench Phase 8)
=============================================================================
Option 3 Gated Multimodal Fusion Adapter over frozen DeBERTa-v3 and OpenCLIP ViT-B/32.
Detects multimodal framing, rhetorical spin, and manipulative presentation.

Governance & Epistemic Boundaries:
- Evaluates rhetorical presentation and visual spin; NOT an oracle of factual truth.
- High rhetorical probability does NOT prove an assertion is factually false.
- Low rhetorical probability does NOT prove an assertion is factually true.
- Gating weights represent "Modality Salience Weight" (learned mixture weights),
  NOT causal attribution.
"""

from dataclasses import dataclass, field
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.utils import setup_logger

logger = setup_logger("rhetoric_classifier")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "models" / "rhetoric_adapter_gated.pt"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "models" / "rhetoric_adapter_gated.json"

# Local snapshot fallbacks for guaranteed offline / sandboxed execution
DEBERTA_SNAPSHOT = r"C:\Users\94in\.cache\huggingface\hub\models--MoritzLaurer--DeBERTa-v3-base-mnli-fever-anli\snapshots\6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
CLIP_SNAPSHOT = r"C:\Users\94in\.cache\huggingface\hub\models--openai--clip-vit-base-patch32\snapshots\3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"

# ─────────────────────────────────────────────────────────────────────────────
# 1. HEADLINE SANITIZATION (LEAKAGE PREVENTION)
# ─────────────────────────────────────────────────────────────────────────────
KNOWN_DELIMITERS = [" - ", " | ", ": ", " – ", " — "]

def sanitize_headline(raw_text: str) -> Tuple[str, bool]:
    """
    Strips trailing publisher attribution suffixes from headlines.
    Uses anchor-bounded right-split matching verified across 88 newsrooms in VLDBench.
    Returns: (sanitized_headline, suffix_was_removed)
    """
    if not raw_text or not isinstance(raw_text, str):
        return "", False

    text = raw_text.strip()
    if not text:
        return "", False

    best_cand = text
    removed = False

    for delim in KNOWN_DELIMITERS:
        if delim in text:
            parts = text.rsplit(delim, 1)
            candidate_prefix = parts[0].strip()
            suffix = parts[1].strip()
            if len(suffix) <= 40 and len(suffix.split()) <= 6 and len(candidate_prefix.split()) >= 3:
                best_cand = candidate_prefix
                removed = True
                break

    return best_cand, removed


# ─────────────────────────────────────────────────────────────────────────────
# 2. GATED MULTIMODAL FUSION ARCHITECTURE (OPTION 3: 1.58M PARAMETERS)
# ─────────────────────────────────────────────────────────────────────────────
class GatedFusionAdapter(nn.Module):
    """
    Approved Option 3 Gated Multimodal Fusion Adapter.
    Total Trainable Parameters: 1,582,594 (~1.58M)
    Dynamically gates text and visual modalities via learned sigmoid channel weights.
    """
    def __init__(self, text_dim: int = 768, vision_dim: int = 768, proj_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.vis_proj = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(text_dim + vision_dim),
            nn.Linear(text_dim + vision_dim, proj_dim),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 2)
        )

    def forward(self, h_text: torch.Tensor, h_vis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u_text = self.text_proj(h_text)
        u_vis = self.vis_proj(h_vis)
        cat_features = torch.cat([h_text, h_vis], dim=-1)
        g = self.gate(cat_features)
        fused = g * u_text + (1.0 - g) * u_vis
        logits = self.classifier(fused)
        return logits, g


# ─────────────────────────────────────────────────────────────────────────────
# 3. RESULT DATA SCHEMA
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RhetoricalManipulationResult:
    """Structured output from the Rhetorical Manipulation Analysis component."""
    predicted_label: str  # "Likely Manipulation" vs "Unlikely Manipulation"
    probability_likely: float  # [0.0, 1.0] calibrated probability of rhetorical manipulation
    probability_unlikely: float  # [0.0, 1.0]
    risk_level: str  # "Low", "Uncertain", "Elevated"
    text_salience_weight: float  # Modality Salience Weight: text component [0.0, 1.0]
    visual_salience_weight: float  # Modality Salience Weight: visual component [0.0, 1.0]
    has_image: bool
    sanitized_headline: str
    suffix_removed: bool
    status: str = "success"  # "success", "text_only_fallback", "error"
    decision_threshold: float = 0.50
    disclaimer: str = (
        "Rhetorical manipulation classification assesses persuasive framing, editorial spin, "
        "and sensationalist presentation. It does NOT evaluate whether factual claims are true or false."
    )
    details: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLASSIFIER SERVICE SINGLETON
# ─────────────────────────────────────────────────────────────────────────────
class RhetoricalManipulationClassifier:
    """
    Manages loading, feature extraction, and inference for the Gated Multimodal Adapter.
    Shares underlying DeBERTa and OpenCLIP encoders from VERITAS-MM pipeline without reloading.
    """
    def __init__(self, weights_path: Optional[Path] = None, config_path: Optional[Path] = None):
        self.weights_path = Path(weights_path or DEFAULT_WEIGHTS_PATH)
        self.config_path = Path(config_path or DEFAULT_CONFIG_PATH)
        self.device = torch.device("cpu")
        self.adapter = GatedFusionAdapter(text_dim=768, vision_dim=768, proj_dim=512)
        self.is_loaded = False
        self._load_adapter_weights()

    def _load_adapter_weights(self):
        if self.weights_path.exists():
            try:
                state_dict = torch.load(self.weights_path, map_location=self.device, weights_only=True)
                self.adapter.load_state_dict(state_dict)
                self.adapter.eval()
                self.is_loaded = True
                logger.info(f"[OK] Loaded GatedFusionAdapter weights from {self.weights_path}")
            except Exception as e:
                logger.warning(f"Failed to load adapter weights from {self.weights_path}: {e}")
                self.is_loaded = False
        else:
            logger.warning(f"Adapter weights file not found at {self.weights_path}")
            self.is_loaded = False

    def _get_shared_nli(self):
        """Retrieves or loads shared DeBERTa model and tokenizer with strict offline handling."""
        import src.verification as v_mod
        if v_mod._NLI_MODEL is not None and v_mod._NLI_TOKENIZER is not None:
            return v_mod._NLI_MODEL, v_mod._NLI_TOKENIZER

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_target = DEBERTA_SNAPSHOT if os.path.exists(DEBERTA_SNAPSHOT) else "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        tokenizer = AutoTokenizer.from_pretrained(model_target, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_target, local_files_only=True)
        model.eval()
        v_mod._NLI_MODEL = model
        v_mod._NLI_TOKENIZER = tokenizer
        return model, tokenizer

    def _get_shared_clip(self):
        """Retrieves or loads shared CLIP model and processor with strict offline handling."""
        import src.cross_modal as c_mod
        if c_mod._CLIP_MODEL is not None and c_mod._CLIP_PROCESSOR is not None:
            return c_mod._CLIP_MODEL, c_mod._CLIP_PROCESSOR

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import CLIPModel, CLIPProcessor

        clip_target = CLIP_SNAPSHOT if os.path.exists(CLIP_SNAPSHOT) else "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(clip_target, local_files_only=True)
        processor = CLIPProcessor.from_pretrained(clip_target, local_files_only=True)
        model.eval()
        c_mod._CLIP_MODEL = model
        c_mod._CLIP_PROCESSOR = processor
        return model, processor

    def _extract_text_representation(self, sanitized_text: str) -> Optional[torch.Tensor]:
        """Extracts DeBERTa-v3 mean-pooled 768-d representation using the pipeline's shared NLI model."""
        try:
            nli_model, tokenizer = self._get_shared_nli()
            if nli_model is None or tokenizer is None:
                return None

            inputs = tokenizer(
                sanitized_text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )

            with torch.no_grad():
                out = nli_model.deberta(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                token_embeds = out.last_hidden_state  # [1, L, 768]
                mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeds.size()).float()
                sum_embeds = torch.sum(token_embeds * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                mean_pooled = sum_embeds / sum_mask  # [1, 768]

            return mean_pooled
        except Exception as e:
            logger.warning(f"Error extracting text representation: {e}")
            return None

    def _extract_vision_representation(self, image: Optional[Image.Image]) -> Optional[torch.Tensor]:
        """Extracts OpenCLIP ViT-B/32 pooled 768-d representation using the pipeline's shared CLIP model."""
        if image is None:
            return None
        try:
            clip_model, clip_processor = self._get_shared_clip()
            if clip_model is None or clip_processor is None:
                return None

            if not isinstance(image, Image.Image):
                return None
            img_rgb = image.convert("RGB")

            inputs = clip_processor(images=[img_rgb], return_tensors="pt")
            with torch.inference_mode():
                vision_out = clip_model.vision_model(inputs["pixel_values"])
                pooler_output = vision_out.pooler_output  # [1, 768]

            return pooler_output
        except Exception as e:
            logger.warning(f"Error extracting vision representation: {e}")
            return None

    def analyze(
        self,
        claim_text: str,
        image: Optional[Image.Image] = None
    ) -> RhetoricalManipulationResult:
        """
        Executes multimodal rhetorical manipulation classification.
        Handles missing images gracefully with transparent text-only fallback.
        """
        if not claim_text or not claim_text.strip():
            return RhetoricalManipulationResult(
                predicted_label="Uncertain",
                probability_likely=0.50,
                probability_unlikely=0.50,
                risk_level="Uncertain",
                text_salience_weight=0.50,
                visual_salience_weight=0.50,
                has_image=bool(image),
                sanitized_headline="",
                suffix_removed=False,
                status="error",
                details={"error": "Empty or whitespace-only text input."}
            )

        sanitized_text, suffix_removed = sanitize_headline(claim_text)
        h_text = self._extract_text_representation(sanitized_text)

        if h_text is None:
            return RhetoricalManipulationResult(
                predicted_label="Uncertain",
                probability_likely=0.50,
                probability_unlikely=0.50,
                risk_level="Uncertain",
                text_salience_weight=0.50,
                visual_salience_weight=0.50,
                has_image=bool(image),
                sanitized_headline=sanitized_text,
                suffix_removed=suffix_removed,
                status="error",
                details={"error": "Text encoder unavailable."}
            )

        h_vis = self._extract_vision_representation(image)
        has_image = h_vis is not None

        if not has_image:
            h_vis = torch.zeros((1, 768), dtype=torch.float32)

        if not self.is_loaded:
            self._load_adapter_weights()

        try:
            self.adapter.eval()
            with torch.no_grad():
                logits, g = self.adapter(h_text, h_vis)
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                prob_unlikely = float(probs[0])
                prob_likely = float(probs[1])

                if g is not None:
                    mean_gate = float(g.mean().item())
                else:
                    mean_gate = 0.50

            if prob_likely >= 0.60:
                risk_level = "Elevated"
                label = "Likely Manipulation"
            elif prob_likely <= 0.40:
                risk_level = "Low"
                label = "Unlikely Manipulation"
            else:
                risk_level = "Uncertain"
                label = "Ambiguous / Moderate"

            if not has_image:
                salience_text = 1.0
                salience_vis = 0.0
                status_str = "text_only_fallback"
            else:
                salience_text = mean_gate
                salience_vis = 1.0 - mean_gate
                status_str = "success"

            return RhetoricalManipulationResult(
                predicted_label=label,
                probability_likely=prob_likely,
                probability_unlikely=prob_unlikely,
                risk_level=risk_level,
                text_salience_weight=salience_text,
                visual_salience_weight=salience_vis,
                has_image=has_image,
                sanitized_headline=sanitized_text,
                suffix_removed=suffix_removed,
                status=status_str,
                details={
                    "logits": [float(logits[0, 0].item()), float(logits[0, 1].item())],
                    "raw_gate_scalar": mean_gate
                }
            )
        except Exception as e:
            logger.warning(f"Inference error in RhetoricalManipulationClassifier: {e}")
            return RhetoricalManipulationResult(
                predicted_label="Uncertain",
                probability_likely=0.50,
                probability_unlikely=0.50,
                risk_level="Uncertain",
                text_salience_weight=0.50,
                visual_salience_weight=0.50,
                has_image=has_image,
                sanitized_headline=sanitized_text,
                suffix_removed=suffix_removed,
                status="error",
                details={"error": str(e)}
            )


# Global singleton instance
_GLOBAL_RHETORIC_CLASSIFIER: Optional[RhetoricalManipulationClassifier] = None

def get_rhetoric_classifier() -> RhetoricalManipulationClassifier:
    """Returns or lazily creates the global singleton RhetoricalManipulationClassifier."""
    global _GLOBAL_RHETORIC_CLASSIFIER
    if _GLOBAL_RHETORIC_CLASSIFIER is None:
        _GLOBAL_RHETORIC_CLASSIFIER = RhetoricalManipulationClassifier()
    return _GLOBAL_RHETORIC_CLASSIFIER
