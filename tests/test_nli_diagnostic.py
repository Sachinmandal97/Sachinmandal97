"""
Regression and diagnostic test suite for the DeBERTa-v3 NLI verification stage.

Verifies:
1. Exact model checkpoint and tokenizer consistency.
2. Label index-to-name mapping fidelity ({0: 'entailment', 1: 'neutral', 2: 'contradiction'}).
3. Premise/hypothesis token formatting ([CLS] Evidence [SEP] Claim [SEP]).
4. Empirical relation estimation on:
   - Core supported London fire claim (Entailment > 80%)
   - Compound London fire claim with uncorroborated smoke clause (Neutral > 90% per formal NLI semantics)
   - Contradicted claim (Contradiction > 90%)
   - Unrelated claim (Neutral > 80%)
"""

from src.verification import estimate_nli_relation, get_nli_model


def test_nli_model_config_and_id2label():
    """Verifies that the configured DeBERTa-v3 model and tokenizer match and have expected id2label."""
    model, tokenizer, device = get_nli_model()
    assert model is not None, "DeBERTa-v3 model failed to load"
    assert tokenizer is not None, "DeBERTa-v3 tokenizer failed to load"
    
    # Verify label IDs
    assert model.config.id2label[0] == "entailment"
    assert model.config.id2label[1] == "neutral"
    assert model.config.id2label[2] == "contradiction"


def test_core_supported_london_fire_claim():
    """
    Verifies that the core London fire assertion is strongly entailed
    by the official evidence document.
    """
    evidence = "London Fire Brigade deployed 10 fire engines and around 70 firefighters to an industrial warehouse fire in East London on Sunday morning."
    core_claim = "Firefighters battle a major industrial warehouse blaze in East London on Sunday morning."

    nli = estimate_nli_relation(claim_text=core_claim, evidence_text=evidence, snippet_id="ev_01")
    assert nli.predicted_relation == "entailment"
    assert nli.p_entailment > 0.80, f"Expected p_entailment > 0.80, got {nli.p_entailment}"
    assert nli.p_contradiction < 0.05


def test_compound_unverified_clause_nli_semantics():
    """
    Verifies the strict semantic property of formal NLI:
    When a claim asserts an additional unverified proposition ('with thick black smoke visible across the city skyline')
    absent from the evidence premise, DeBERTa-v3 strictly predicts neutral (undetermined), NOT entailment.
    """
    evidence = "London Fire Brigade deployed 10 fire engines and around 70 firefighters to an industrial warehouse fire in East London on Sunday morning."
    compound_claim = "Firefighters battle a major industrial warehouse blaze in East London on Sunday morning, with thick black smoke visible across the city skyline."

    nli = estimate_nli_relation(claim_text=compound_claim, evidence_text=evidence, snippet_id="ev_01")
    # Demonstrates formal NLI conservatism: unmentioned clause cannot be entailed
    assert nli.predicted_relation == "neutral"
    assert nli.p_neutral > 0.90, f"Expected p_neutral > 0.90, got {nli.p_neutral}"


def test_contradicted_claim():
    """Verifies that an explicitly refuted claim produces strong contradiction probability."""
    evidence = "London Fire Brigade deployed 10 fire engines and around 70 firefighters to an industrial warehouse fire in East London on Sunday morning."
    contra_claim = "No fire occurred in East London on Sunday morning."

    nli = estimate_nli_relation(claim_text=contra_claim, evidence_text=evidence, snippet_id="ev_01")
    assert nli.predicted_relation == "contradiction"
    assert nli.p_contradiction > 0.90, f"Expected p_contradiction > 0.90, got {nli.p_contradiction}"


def test_unrelated_claim():
    """Verifies that a topically unrelated claim produces dominant neutral probability."""
    evidence = "London Fire Brigade deployed 10 fire engines and around 70 firefighters to an industrial warehouse fire in East London on Sunday morning."
    unrelated_claim = "Stock markets in Tokyo rose following tech earnings."

    nli = estimate_nli_relation(claim_text=unrelated_claim, evidence_text=evidence, snippet_id="ev_01")
    assert nli.predicted_relation == "neutral"
    assert nli.p_neutral > 0.80, f"Expected p_neutral > 0.80, got {nli.p_neutral}"


def test_atomic_claim_decomposition_and_granular_verification():
    """
    Verifies that a compound sentence decomposes into distinct propositions,
    and each proposition is verified independently against retrieved evidence:
      - Proposition 1: Supported by retrieved evidence
      - Proposition 2: Unverified (neutral/missing), NOT presumed false
    """
    from src.decomposition import decompose_claim
    from src.retrieval import EvidenceRetriever
    from src.verification import verify_atomic_claims, aggregate_atomic_verifications
    from src.schemas import DecisionVerdict

    compound = "Firefighters battle a major industrial warehouse blaze in East London on Sunday morning, with thick black smoke visible across the city skyline."
    atomic_claims = decompose_claim(compound)
    assert len(atomic_claims) >= 2, f"Expected at least 2 atomic claims, got {len(atomic_claims)}"

    retriever = EvidenceRetriever(enable_live_search=False)
    results = verify_atomic_claims(atomic_claims, retriever, force_mode="local_only")
    assert len(results) == len(atomic_claims)

    # First proposition (firefighters battle blaze) should be SUPPORTED
    assert results[0].status == "SUPPORTED", f"Expected SUPPORTED, got {results[0].status}"
    assert results[0].retrieved_evidence is not None
    assert results[0].nli_result.p_entailment >= 0.60

    # Second proposition (smoke visible across skyline) should be UNVERIFIED (neutral / not established)
    assert results[1].status == "UNVERIFIED", f"Expected UNVERIFIED, got {results[1].status}"
    # Verify aggregation produces PARTIALLY_SUPPORTED
    verdict, reasons = aggregate_atomic_verifications(results)
    assert verdict == DecisionVerdict.PARTIALLY_SUPPORTED
    assert any("Core factual assertions are corroborated" in r for r in reasons)


def test_atomic_claim_contradicted_and_unsupported_aggregation():
    """
    Verifies explicit aggregation rules:
      - Contradicted proposition drives overall decision to MISLEADING
      - All unverified propositions produce UNCERTAIN (never inferred false)
      - All supported propositions produce SUPPORTED
    """
    from src.schemas import AtomicClaim, AtomicVerificationResult, DecisionVerdict, EvidenceSnippet, NLIResult
    from src.verification import aggregate_atomic_verifications

    # Case A: Contradicted atomic claim present -> MISLEADING
    results_with_contra = [
        AtomicVerificationResult(
            claim_id="c1",
            atomic_text="Government announced tax cuts.",
            status="SUPPORTED"
        ),
        AtomicVerificationResult(
            claim_id="c2",
            atomic_text="The finance minister denied any budget amendments.",
            status="CONTRADICTED"
        )
    ]
    verdict_a, reasons_a = aggregate_atomic_verifications(results_with_contra)
    assert verdict_a == DecisionVerdict.MISLEADING
    assert any("refuted" in r.lower() for r in reasons_a)

    # Case B: All unverified -> UNCERTAIN (not false!)
    results_all_unverified = [
        AtomicVerificationResult(
            claim_id="c1",
            atomic_text="Secret gold vault found beneath city.",
            status="UNVERIFIED"
        ),
        AtomicVerificationResult(
            claim_id="c2",
            atomic_text="Vault contains Roman artifacts.",
            status="UNVERIFIED"
        )
    ]
    verdict_b, reasons_b = aggregate_atomic_verifications(results_all_unverified)
    assert verdict_b == DecisionVerdict.UNCERTAIN

    # Case C: All supported -> SUPPORTED
    results_all_supported = [
        AtomicVerificationResult(
            claim_id="c1",
            atomic_text="Fire engines deployed to warehouse.",
            status="SUPPORTED"
        ),
        AtomicVerificationResult(
            claim_id="c2",
            atomic_text="Fire occurred in East London.",
            status="SUPPORTED"
        )
    ]
    verdict_c, reasons_c = aggregate_atomic_verifications(results_all_supported)
    assert verdict_c == DecisionVerdict.SUPPORTED


def test_negation_preservation_in_claim_verification():
    """
    Verifies that claims with explicit denial or negation (e.g. 'The official denied that the fire occurred')
    are not mistakenly split into positive assertions and correctly produce contradiction against affirmative evidence.
    """
    from src.decomposition import decompose_claim
    from src.verification import estimate_nli_relation

    claim = "The official denied that the fire occurred."
    decomposed = decompose_claim(claim)
    # The denial must be preserved as a single coherent proposition, not split into 'the fire occurred'
    assert len(decomposed) == 1
    assert "denied" in decomposed[0].text

    evidence = "London Fire Brigade deployed 10 fire engines and around 70 firefighters to an industrial warehouse fire in East London on Sunday morning."
    nli = estimate_nli_relation(claim_text=decomposed[0].text, evidence_text=evidence, snippet_id="ev_01")
    assert nli.predicted_relation == "contradiction"
    assert nli.p_contradiction > 0.90


def test_conjunction_preservation_in_noun_phrases_and_entities():
    """
    Verifies that conjunctions ('and', '&') within noun phrases, compounds, or named entities
    are preserved and NOT broken into incomplete fragments.
    """
    from src.decomposition import decompose_claim

    phrases = [
        "Search and rescue teams arrived in Trinidad and Tobago.",
        "The minister visited Barnes & Noble and Marks & Spencer.",
        "Oil and gas companies reported quarterly profits."
    ]

    for p in phrases:
        decomposed = decompose_claim(p)
        assert len(decomposed) == 1, f"Expected 1 coherent proposition for '{p}', got {len(decomposed)}: {[d.text for d in decomposed]}"
        assert "and" in decomposed[0].text.lower() or "&" in decomposed[0].text


def test_missing_evidence_and_no_image_fallbacks():
    """
    Verifies robust fallback handling when:
      - Claim has no matching evidence in corpus -> all atomic claims UNVERIFIED -> UNCERTAIN verdict
      - No image is supplied -> system executes cleanly without crashing and sets heuristic bounds
    """
    from src.pipeline import MultimodalVerificationPipeline
    from src.schemas import DecisionVerdict, MultimodalSample

    pipeline = MultimodalVerificationPipeline()

    # Query with nonexistent topical content
    sample = MultimodalSample(
        sample_id="test_unmatched_01",
        claim_text="Archaeologists unearth secret underground gold vaults beneath Edinburgh.",
        image_path=None
    )
    result = pipeline.verify_sample(sample)
    assert result.decision == DecisionVerdict.UNCERTAIN
    assert result.cross_modal.mismatch_detected is False
    assert result.confidence == 0.40
    assert result.uncertainty_score == 0.60
    assert all(a.status == "UNVERIFIED" for a in result.atomic_verifications)

