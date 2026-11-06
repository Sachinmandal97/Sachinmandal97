"""
Optional LLM Provider Module.
Provides a decoupled hook for optional Gemini or external LLM reasoning.
Note: The core framework functions 100% locally with open-weight models without this module.
"""

from typing import Any, Dict, List, Optional
from src.schemas import AtomicClaim


class OptionalLLMProvider:
    """
    Decoupled interface for optional LLM enhancements (e.g. Google Gemini).
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self.is_enabled = bool(api_key)

    def is_available(self) -> bool:
        """Checks whether external API credentials are provided."""
        return self.is_enabled

    def decompose_claim(self, text: str) -> List[AtomicClaim]:
        """Optional LLM-assisted multi-hop claim decomposition."""
        if not self.is_available():
            raise RuntimeError("Optional LLM provider is not enabled or missing API key.")
        # Future Gemini integration point
        return []

    def synthesize_narrative_explanation(self, payload: Dict[str, Any]) -> str:
        """Optional LLM-assisted multi-evidence natural narrative generation."""
        if not self.is_available():
            raise RuntimeError("Optional LLM provider is not enabled or missing API key.")
        return ""
