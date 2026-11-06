"""
Universal real-news and fake-news verification test suite.

Evaluates:
1. No assumptions about truth:
   - Claims are not assumed true because of famous entities or authoritative phrasing.
   - Claims are not assumed fake because evidence is missing or CLIP score is low.
2. Five-state verdict capability:
   - DecisionVerdict.SUPPORTED (real/verified claims with corroborating evidence)
   - DecisionVerdict.MISLEADING (fabricated/debunked claims with refuting evidence)
   - DecisionVerdict.PARTIALLY_SUPPORTED (claims with verified core but unverified details)
   - DecisionVerdict.UNCERTAIN (unverified claims / absence of evidence)
   - DecisionVerdict.CONFLICTING (claims where credible sources materially disagree)
3. Granular proposition verification and contradiction precedence.
4. Cheapfake / out-of-context image handling decoupled from factual text verification.
5. Multi-domain coverage (politics, science, business, sports, public health, hoaxes).
"""

import unittest
from unittest.mock import MagicMock

from src.pipeline import MultimodalVerificationPipeline
from src.retrieval import EvidenceRetriever
from src.schemas import (
    AtomicClaim,
    AtomicVerificationResult,
    CrossModalScore,
    DecisionVerdict,
    EvidenceSnippet,
    MultimodalSample,
    NLIResult,
    VerificationResult,
)
from src.verification import (
    aggregate_atomic_verifications,
    fuse_multimodal_decision,
    verify_atomic_claims,
)
from src.uncertainty import compute_decision_confidence_and_uncertainty


def test_real_claim_supported_with_genuine_evidence():
    """
    Verifies that a genuine news assertion corroborated by relevant evidence
    yields DecisionVerdict.SUPPORTED with high confidence.
    """
    atomic = [
        AtomicVerificationResult(
            claim_id="prop_01",
            atomic_text="The central bank lowered benchmark interest rates by 25 basis points.",
            status="SUPPORTED",
            retrieved_evidence=EvidenceSnippet(
                snippet_id="ev_fed",
                text="The central bank lowered benchmark interest rates by 25 basis points on Wednesday.",
                source="Financial Times",
                is_relevant=True,
                relevance_score=0.88,
            ),
            nli_result=NLIResult(
                claim_text="The central bank lowered benchmark interest rates by 25 basis points.",
                evidence_snippet_id="ev_fed",
                p_entailment=0.92,
                p_neutral=0.06,
                p_contradiction=0.02,
                predicted_relation="entailment",
            ),
            explanation="Supported by relevant evidence [ev_fed].",
        )
    ]
    cm = CrossModalScore(similarity_score=0.28, mismatch_detected=False)
    verdict, reasons = fuse_multimodal_decision(cross_modal=cm, nli_results=[], atomic_results=atomic)

    assert verdict == DecisionVerdict.SUPPORTED
    conf, unc = compute_decision_confidence_and_uncertainty(verdict, cm, [atomic[0].nli_result])
    assert conf > 0.65
    assert unc < 0.35
    assert any("substantiated" in r.lower() or "corroborated" in r.lower() for r in reasons)


def test_fabricated_claim_refuted_by_fact_check():
    """
    Verifies that a deliberately fabricated claim with explicit debunking evidence
    yields DecisionVerdict.MISLEADING.
    """
    atomic = [
        AtomicVerificationResult(
            claim_id="prop_who",
            atomic_text="The World Health Organization officially banned coffee worldwide due to toxicity.",
            status="CONTRADICTED",
            retrieved_evidence=EvidenceSnippet(
                snippet_id="ev_factcheck",
                text="Fact-check: Claims that the World Health Organization banned coffee worldwide are false. The WHO issued no such ban.",
                source="Reuters Fact Check",
                is_relevant=True,
                relevance_score=0.85,
            ),
            nli_result=NLIResult(
                claim_text="The World Health Organization officially banned coffee worldwide due to toxicity.",
                evidence_snippet_id="ev_factcheck",
                p_entailment=0.02,
                p_neutral=0.04,
                p_contradiction=0.94,
                predicted_relation="contradiction",
            ),
            explanation="Contradicted by relevant evidence [ev_factcheck] (p_contra: 0.94).",
        )
    ]
    cm = CrossModalScore(similarity_score=0.25, mismatch_detected=False)
    verdict, reasons = fuse_multimodal_decision(cross_modal=cm, nli_results=[], atomic_results=atomic)

    assert verdict == DecisionVerdict.MISLEADING
    assert any("refuted" in r.lower() or "contradicted" in r.lower() for r in reasons)


def test_partially_supported_mixed_propositions():
    """
    Verifies that a claim with a corroborated core assertion but an unverified secondary detail
    yields DecisionVerdict.PARTIALLY_SUPPORTED.
    """
    atomics = [
        AtomicVerificationResult(
            claim_id="prop_01",
            atomic_text="London Fire Brigade deployed 70 firefighters to an industrial warehouse fire.",
            status="SUPPORTED",
            explanation="Supported by relevant evidence.",
        ),
        AtomicVerificationResult(
            claim_id="prop_02",
            atomic_text="Thick black smoke was visible across the entire city skyline.",
            status="UNVERIFIED",
            explanation="Not established by evidence (neutral relation).",
        ),
    ]
    verdict, reasons = aggregate_atomic_verifications(atomics)
    assert verdict == DecisionVerdict.PARTIALLY_SUPPORTED
    assert any("corroborated" in r.lower() and "unverified" in r.lower() for r in reasons)


def test_unverified_hoax_no_evidence_yields_uncertain():
    """
    MANDATORY SAFETY RULE:
    A pure fabrication with no relevant evidence must yield DecisionVerdict.UNCERTAIN.
    The system must NEVER assume falsehood merely because evidence was not found.
    """
    atomics = [
        AtomicVerificationResult(
            claim_id="prop_antarctica",
            atomic_text="Explorers discovered an ancient subterranean neon city buried beneath Antarctic ice sheets.",
            status="UNVERIFIED",
            retrieved_evidence=None,
            nli_result=None,
            explanation="No relevant evidence found for this claim in the retrieved sources.",
        )
    ]
    cm = CrossModalScore(similarity_score=0.15, mismatch_detected=True)
    verdict, reasons = fuse_multimodal_decision(cross_modal=cm, nli_results=[], atomic_results=atomics)

    # Strictly UNCERTAIN, NOT MISLEADING (absence of evidence != evidence of falsity)
    assert verdict == DecisionVerdict.UNCERTAIN
    conf, unc = compute_decision_confidence_and_uncertainty(verdict, cm, [])
    assert unc == 0.60
    assert conf == 0.40


def test_conflicting_evidence_material_disagreement():
    """
    Verifies that when credible retrieved sources materially disagree
    (one strong entailment and one strong contradiction), the system yields DecisionVerdict.CONFLICTING.
    """
    # Test at atomic verification level
    atomics = [
        AtomicVerificationResult(
            claim_id="prop_election",
            atomic_text="Audit board confirmed Candidate A won the contested district election.",
            status="CONFLICTING",
            explanation="Conflicting evidence detected across retrieved sources: [ev_01] supports (p_entail: 0.88), but [ev_02] refutes (p_contra: 0.91).",
        )
    ]
    verdict, reasons = aggregate_atomic_verifications(atomics)
    assert verdict == DecisionVerdict.CONFLICTING
    assert any("materially disagree" in r.lower() or "conflicting" in r.lower() for r in reasons)

    # Test uncertainty heuristic for conflicting verdicts
    cm = CrossModalScore(similarity_score=0.25, mismatch_detected=False)
    conf, unc = compute_decision_confidence_and_uncertainty(verdict, cm, [])
    assert unc == 0.55
    assert conf == 0.45

    # Test at document-level NLI fallback
    nli_results = [
        NLIResult(
            claim_text="Candidate A won the district",
            evidence_snippet_id="ev_01",
            p_entailment=0.88,
            p_neutral=0.08,
            p_contradiction=0.04,
            predicted_relation="entailment",
        ),
        NLIResult(
            claim_text="Candidate A won the district",
            evidence_snippet_id="ev_02",
            p_entailment=0.03,
            p_neutral=0.05,
            p_contradiction=0.92,
            predicted_relation="contradiction",
        ),
    ]
    doc_verdict, doc_reasons = fuse_multimodal_decision(
        cross_modal=cm, nli_results=nli_results, atomic_results=None
    )
    assert doc_verdict == DecisionVerdict.CONFLICTING
    assert any("conflicting" in r.lower() for r in doc_reasons)


def test_visual_mismatch_on_supported_text_cheapfake_prevention():
    """
    Verifies that low visual-text alignment alone does NOT force MISLEADING
    when the factual claim is substantiated by verified text evidence.
    Instead, it flags PARTIALLY_SUPPORTED with an out-of-context visual notice.
    """
    atomic = [
        AtomicVerificationResult(
            claim_id="prop_real",
            atomic_text="NASA Perseverance rover detected organic molecules on Mars.",
            status="SUPPORTED",
            explanation="Supported by verified evidence.",
        )
    ]
    # Out-of-context image (e.g. stock photo of a desert)
    cm_mismatch = CrossModalScore(similarity_score=0.10, mismatch_detected=True)
    verdict, reasons = fuse_multimodal_decision(
        cross_modal=cm_mismatch, nli_results=[], atomic_results=atomic
    )

    assert verdict == DecisionVerdict.PARTIALLY_SUPPORTED
    assert any("repurposed" in r.lower() or "out-of-context" in r.lower() for r in reasons)


def test_visual_mismatch_on_unverified_text_yields_uncertain():
    """
    Verifies that an unverified claim accompanied by a mismatched visual
    remains UNCERTAIN (not assumed to be intentionally false).
    """
    atomic = [
        AtomicVerificationResult(
            claim_id="prop_unv",
            atomic_text="A new species of giant squid was captured off the coast of Greenland.",
            status="UNVERIFIED",
            explanation="No relevant evidence found.",
        )
    ]
    cm_mismatch = CrossModalScore(similarity_score=0.08, mismatch_detected=True)
    verdict, reasons = fuse_multimodal_decision(
        cross_modal=cm_mismatch, nli_results=[], atomic_results=atomic
    )

    assert verdict == DecisionVerdict.UNCERTAIN


def test_contradiction_precedence_in_compound_claim():
    """
    Verifies that if a compound claim contains both a supported proposition
    and a refuted proposition (e.g., real event + fabricated detail),
    contradiction strictly takes precedence yielding DecisionVerdict.MISLEADING.
    """
    atomics = [
        AtomicVerificationResult(
            claim_id="prop_real",
            atomic_text="TechCorp published quarterly financial results on Monday.",
            status="SUPPORTED",
            explanation="Corroborated by earnings press release.",
        ),
        AtomicVerificationResult(
            claim_id="prop_fake",
            atomic_text="TechCorp Chief Executive resigned following an embezzlement probe.",
            status="CONTRADICTED",
            explanation="Company spokesperson and official filings confirmed the CEO remains in office.",
        ),
    ]
    verdict, reasons = aggregate_atomic_verifications(atomics)
    assert verdict == DecisionVerdict.MISLEADING
    assert any("refuted" in r.lower() or "contradicted" in r.lower() for r in reasons)


def test_arbitrary_news_categories_pipeline_execution():
    """
    Verifies that the verification pipeline can dynamically process claims
    across arbitrary categories without hardcoding or domain bias.
    """
    pipeline = MultimodalVerificationPipeline()

    categories = [
        ("Public Health", "The health ministry initiated a nationwide booster vaccination drive on Monday."),
        ("Economy", "Unemployment rate fell to 3.8% according to the latest labour statistics."),
        ("Environment", "Satellite observations recorded record seasonal meltwater across Arctic glaciers."),
        ("Sports", "The national team qualified for the continental tournament following a 2-1 victory."),
        ("Technology", "Researchers unveiled a fault-tolerant quantum computing processor with 100 logical qubits."),
    ]

    for cat_name, claim_text in categories:
        sample = MultimodalSample(
            sample_id=f"test_cat_{cat_name.lower()}",
            claim_text=claim_text,
            image_path=None,
        )
        result = pipeline.verify_sample(sample, force_mode="local_only")
        assert isinstance(result, VerificationResult)
        assert result.decision in [
            DecisionVerdict.SUPPORTED,
            DecisionVerdict.PARTIALLY_SUPPORTED,
            DecisionVerdict.MISLEADING,
            DecisionVerdict.UNCERTAIN,
            DecisionVerdict.CONFLICTING,
        ]
        assert len(result.decomposed_claims) >= 1
        assert len(result.atomic_verifications) >= 1
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.uncertainty_score <= 1.0


def test_truth_assumption_neutrality():
    """
    Verifies that authority words and famous names do NOT artificially tilt the verdict
    in the absence of relevant evidence.
    """
    authoritative_hoax = "NASA and the White House officially confirmed the immediate cancellation of all space exploration programs."
    sample = MultimodalSample(
        sample_id="test_auth_hoax",
        claim_text=authoritative_hoax,
        image_path=None,
    )
    pipeline = MultimodalVerificationPipeline()
    result = pipeline.verify_sample(sample, force_mode="local_only")

    # The local corpus does NOT contain evidence corroborating this wild claim.
    # Therefore it must NOT be marked SUPPORTED just because 'NASA' and 'White House' appear in the text.
    assert result.decision != DecisionVerdict.SUPPORTED
    assert result.decision in [DecisionVerdict.UNCERTAIN, DecisionVerdict.MISLEADING]
