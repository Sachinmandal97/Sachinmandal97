"""
Manual live multi-domain verification script.
Tests:
1. Science (NASA Broom Point)
2. Sports (Champions League)
3. Business (Federal Reserve interest rates)
4. Public Safety (London Fire Brigade)
5. Fabricated / Unattested (Antarctica undersea city)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import MultimodalVerificationPipeline
from src.schemas import MultimodalSample


def run_manual_audit():
    pipeline = MultimodalVerificationPipeline()

    test_cases = [
        ("Science", "NASA's Perseverance rover captured a view of the Broom Point rock formation on Mars on May 15, 2025."),
        ("Sports", "Real Madrid secured victory in the European Champions League final."),
        ("Business", "The Federal Reserve lowered benchmark interest rates by 25 basis points."),
        ("Public Safety", "London Fire Brigade deployed 70 firefighters to an industrial warehouse fire."),
        ("Fabricated", "Secret undersea city discovered off the coast of Antarctica by deep sea explorers.")
    ]

    print("======================================================================")
    print("MANUAL MULTI-DOMAIN LIVE VERIFICATION AUDIT")
    print("======================================================================")

    for domain, claim in test_cases:
        print(f"\n[DOMAIN: {domain.upper()}]")
        print(f"Claim: {claim}")

        sample = MultimodalSample(
            sample_id=f"audit_{domain.lower()}",
            claim_text=claim,
            image_path=None
        )

        result = pipeline.verify_sample(sample, force_mode="live_only")

        print(f"Decision: {result.decision.value}")
        print(f"Confidence: {result.confidence:.1%} | Predictive Uncertainty: {result.uncertainty_score:.1%}")
        print(f"Retrieved Evidence Count: {len(result.retrieved_evidence)}")

        for idx, ev in enumerate(result.retrieved_evidence, 1):
            rel_str = f"{ev.relevance_score:.2f}" if ev.relevance_score is not None else "N/A"
            print(f"  [{idx}] Title: {ev.title or ev.source[:45]}")
            print(f"      Source: {ev.publisher or ev.source} | Format: {ev.content_type} | Access: {ev.access_status}")
            print(f"      URL: {ev.url}")
            print(f"      Relevance: {rel_str} | Is Relevant: {ev.is_relevant}")
            print(f"      Reason: {ev.relevance_reason}")

        print("Reasoning Factors:")
        for r in result.reasoning_factors:
            print(f"  - {r}")


if __name__ == "__main__":
    run_manual_audit()
