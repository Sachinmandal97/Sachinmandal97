"""
Claim and entity decomposition module.
Baseline deterministic method using spaCy and rule-based linguistic patterns.
Note: This is a deterministic linguistic baseline for atomic claim/entity parsing,
not a complex LLM-based decomposition system.
"""

from typing import List, Optional
from src.schemas import AtomicClaim
from src.utils import setup_logger

logger = setup_logger("decomposition")

_SPACY_NLP = None


def get_spacy_model():
    """
    Returns or lazily loads the English spaCy model.
    """
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP

    try:
        import spacy
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm")
        except Exception:
            # Fallback blank English tokenizer
            logger.warning("en_core_web_sm not found. Initializing blank English model.")
            _SPACY_NLP = spacy.blank("en")
        return _SPACY_NLP
    except Exception as e:
        logger.warning(f"Failed to load spaCy: {e}")
        return None


def split_into_clauses(sentence_text: str) -> List[str]:
    """
    Splits compound sentences into atomic propositional clauses using
    linguistic boundaries (adverbial/participial markers, coordinating conjunctions,
    semicolons, em-dashes).
    """
    import re

    # Pattern matches:
    # 1. semicolons or dashes (; or — or --)
    # 2. comma followed by clause-introducing words: with, while, as, but, and, although, despite, amid, where, which
    pattern = r"\s*;\s*|\s*[—–]\s*|,\s*(?:with|while|as|but|and|although|despite|amid|where|which)\s+"
    parts = re.split(pattern, sentence_text, flags=re.IGNORECASE)

    clean_parts = []
    for p in parts:
        cleaned = p.strip().rstrip(".,;")
        # Avoid tiny fragments that aren't verifiable propositions
        if len(cleaned.split()) >= 3 and len(cleaned) >= 12:
            clean_parts.append(cleaned)

    return clean_parts if len(clean_parts) > 1 else [sentence_text.strip().rstrip(".,")]


def decompose_claim(claim_text: str, max_claims: int = 5) -> List[AtomicClaim]:
    """
    Decomposes an input narrative/caption into atomic proposition statements
    with extracted entities, location anchors, and temporal references.
    Splits multi-sentence inputs and compound sentences into verifiable atomic units.
    """
    if not claim_text or not claim_text.strip():
        return []

    nlp = get_spacy_model()
    if nlp is None:
        # Fallback simple split on periods/commas
        return [
            AtomicClaim(
                claim_id="claim_0",
                text=claim_text.strip(),
                entities=[],
                temporal_anchor=None,
                location_anchor=None
            )
        ]

    doc = nlp(claim_text)

    # 1. Global Named Entities
    entities = [ent.text for ent in doc.ents]
    
    # Extract locations (GPE, LOC, FAC)
    locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC", "FAC"]]
    primary_location = locations[0] if locations else None

    # Extract temporal references (DATE, TIME)
    dates = [ent.text for ent in doc.ents if ent.label_ in ["DATE", "TIME"]]
    primary_temporal = dates[0] if dates else None

    # 2. Sentence segmentation and sub-clause proposition extraction
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if not sentences:
        sentences = [claim_text.strip()]

    propositions: List[str] = []
    for sent in sentences:
        clauses = split_into_clauses(sent)
        propositions.extend(clauses)

    if not propositions:
        propositions = [claim_text.strip()]

    atomic_claims: List[AtomicClaim] = []

    for idx, prop_text in enumerate(propositions[:max_claims]):
        prop_doc = nlp(prop_text)
        prop_entities = [ent.text for ent in prop_doc.ents]
        prop_locations = [ent.text for ent in prop_doc.ents if ent.label_ in ["GPE", "LOC", "FAC"]]
        prop_dates = [ent.text for ent in prop_doc.ents if ent.label_ in ["DATE", "TIME"]]

        atomic = AtomicClaim(
            claim_id=f"claim_{idx}",
            text=prop_text,
            entities=prop_entities or entities,
            temporal_anchor=prop_dates[0] if prop_dates else primary_temporal,
            location_anchor=prop_locations[0] if prop_locations else primary_location
        )
        atomic_claims.append(atomic)

    return atomic_claims
