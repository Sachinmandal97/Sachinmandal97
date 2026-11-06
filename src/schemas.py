"""
Structured data models passed between pipeline stages.
Using Pydantic ensures type safety, clear data flow, and easy viva explanation.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DecisionVerdict(str, Enum):
    """
    AI-assisted adjudication outcome.
    Not an absolute declaration of truth, but an evidence-grounded recommendation.
    """
    SUPPORTED = "Supported"
    PARTIALLY_SUPPORTED = "Partially Supported / Some Details Unverified"
    MISLEADING = "Misleading"
    UNCERTAIN = "Uncertain"
    CONFLICTING = "Conflicting Evidence"


class MultimodalSample(BaseModel):
    """
    Standardized input representation for a news/social post.
    """
    model_config = {"arbitrary_types_allowed": True}

    sample_id: str
    claim_text: str
    image_path: Optional[str] = None
    image_data: Optional[Any] = None          # Optional in-memory PIL.Image to avoid filesystem races
    raw_label: Optional[str] = None           # Dataset original label (e.g. 'falsified', 'pristine')
    dataset_name: str = "newsclippings"
    split: str = "test"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Factify2Sample(BaseModel):
    """
    Structured representation of a Factify 2 multimodal verification sample.
    Encapsulates claim, visual asset, pre-extracted OCR, and reference document context.
    """
    sample_id: str
    claim_text: str
    claim_image: Optional[str] = None
    claim_ocr: Optional[str] = None
    document_text: Optional[str] = None
    document_image: Optional[str] = None
    document_ocr: Optional[str] = None
    label: str                                # Original dataset label preserved
    split: str = "val"
    source_metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_multimodal_sample(self) -> MultimodalSample:
        """Converts Factify2Sample to standard pipeline MultimodalSample."""
        return MultimodalSample(
            sample_id=self.sample_id,
            claim_text=self.claim_text,
            image_path=self.claim_image,
            raw_label=self.label,
            dataset_name="factify2",
            split=self.split,
            metadata={
                "claim_ocr": self.claim_ocr,
                "document_text": self.document_text,
                "document_image": self.document_image,
                "document_ocr": self.document_ocr,
                **self.source_metadata,
            }
        )



class OCRResult(BaseModel):
    """
    Extracted text and spatial detections from the input image.
    """
    extracted_text: str = ""
    boxes: List[Dict[str, Any]] = Field(default_factory=list)  # [{"text": str, "box": list, "confidence": float}]
    has_text: bool = False


class CrossModalScore(BaseModel):
    """
    Visual-text alignment and discrepancy score (OpenCLIP).
    """
    similarity_score: Optional[float] = None  # Raw cosine similarity between image and claim text (None for text-only/error)
    mismatch_detected: bool = False           # Flagged if similarity falls below configured threshold
    has_image: bool = True                    # False if executed in text-only mode
    status: str = "evaluated"                 # 'evaluated' | 'text_only' | 'unavailable' | 'error'
    ocr_text_overlap: float = 0.0             # Overlap ratio between caption entities and OCR text
    details: Dict[str, Any] = Field(default_factory=dict)


class AtomicClaim(BaseModel):
    """
    A single verifiable factual proposition extracted from the main claim text.
    """
    claim_id: str
    text: str
    entities: List[str] = Field(default_factory=list)
    temporal_anchor: Optional[str] = None
    location_anchor: Optional[str] = None


class EvidenceSnippet(BaseModel):
    """
    Retrieved reference document or fact-check snippet with full provenance tracking.
    """
    snippet_id: str
    text: str
    source: str = "Local Knowledge Base"
    title: Optional[str] = None               # Article headline or document title
    search_query: Optional[str] = None        # Exact query used to retrieve this snippet
    score: Optional[float] = None             # BM25 or raw retrieval ranking score (None if unranked)
    relevance_score: Optional[float] = None   # Explicit topic/lexical relevance score [0.0 - 1.0] (None if unavailable)
    is_relevant: bool = False                 # True only if snippet meets topic relevance cutoff
    relevance_method: str = "uncalculated"    # Algorithm used: 'entity_keyword_overlap', 'bm25_lexical', etc.
    relevance_reason: str = ""                # Diagnostic reason for inclusion or exclusion
    url: Optional[str] = None                 # Resolved original publisher article URL (never root homepage)
    wrapper_url: Optional[str] = None         # Google News RSS wrapper URL (if distinct from publisher URL)
    publisher_domain: Optional[str] = None    # Publisher root domain (e.g. 'https://www.reuters.com')
    publisher: Optional[str] = None           # Publisher name / agency (e.g., 'Reuters', 'NASA Science')
    published_date: Optional[str] = None      # Publication timestamp / date string
    retrieval_timestamp: Optional[str] = None # Exact retrieval time
    source_type: str = "local_corpus"         # 'live_web' | 'wikipedia' | 'local_corpus' | 'synthetic_demo'
    content_type: str = "snippet_only"        # 'full_text' | 'excerpt' | 'snippet_only' | 'headline_only'
    access_status: str = "accessible"         # 'accessible' | 'snippet_only' | 'inaccessible' | 'resolved' | 'google_wrapper_unresolved'
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NLIResult(BaseModel):
    """
    Natural Language Inference relation between a claim and retrieved evidence.
    """
    claim_text: str
    evidence_snippet_id: str
    p_entailment: float                       # Probability that evidence supports claim
    p_neutral: float                          # Probability that evidence is insufficient/neutral
    p_contradiction: float                    # Probability that evidence refutes claim
    predicted_relation: str                   # 'entailment' | 'neutral' | 'contradiction'


class AtomicVerificationResult(BaseModel):
    """
    Granular verification outcome for an individual atomic claim proposition.
    """
    claim_id: str
    atomic_text: str
    status: str                               # 'SUPPORTED' | 'UNVERIFIED' | 'CONTRADICTED'
    retrieved_evidence: Optional[EvidenceSnippet] = None
    nli_result: Optional[NLIResult] = None
    explanation: str = ""


class VerificationResult(BaseModel):
    """
    Complete output package produced by the verification pipeline.
    """
    sample_id: str
    claim_text: str
    decision: DecisionVerdict
    confidence: float                         # Confidence in the predicted decision [0.0 - 1.0]
    uncertainty_score: float                  # Predictive uncertainty [0.0 - 1.0]
    evidence_source_mode: str = "hybrid"      # 'live_search' | 'local_corpus' | 'hybrid'
    
    # Stage breakdown
    ocr: OCRResult
    cross_modal: CrossModalScore
    decomposed_claims: List[AtomicClaim] = Field(default_factory=list)
    atomic_verifications: List[AtomicVerificationResult] = Field(default_factory=list)
    retrieved_evidence: List[EvidenceSnippet] = Field(default_factory=list)
    nli_results: List[NLIResult] = Field(default_factory=list)
    
    # Reasoning & Explainability
    reasoning_factors: List[str] = Field(default_factory=list)
    structured_explanation: str = ""
