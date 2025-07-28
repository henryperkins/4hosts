#!/usr/bin/env python3
"""Test the LLM classification integration"""

import asyncio
import sys

sys.path.append("/home/azureuser/4hosts/four-hosts-app/backend")

from services.classification_engine import ClassificationEngine, HostParadigm


async def test_llm_classification():
    """Test classification with LLM scoring enabled"""

    # Create engines with and without LLM
    engine_with_llm = ClassificationEngine(use_llm=True, cache_enabled=False)
    engine_without_llm = ClassificationEngine(use_llm=False, cache_enabled=False)

    test_queries = [
        "How can we expose corporate corruption in the pharmaceutical industry?",
        "What resources exist to help homeless veterans find housing?",
        "Analyze the statistical correlation between social media use and depression",
        "What strategies can small businesses use to compete with Amazon?",
    ]

    print("Testing LLM Classification Integration")
    print("=" * 80)

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)

        # Test without LLM (rule-based only)
        result_rules = await engine_without_llm.classify_query(query)
        print(f"\nRule-Based Only:")
        print(f"  Primary: {result_rules.primary_paradigm.value}")
        print(f"  Confidence: {result_rules.confidence:.2%}")
        print(
            f"  Distribution: {', '.join([f'{p.value}: {s:.1%}' for p, s in result_rules.distribution.items()])}"
        )

        # Test with LLM (hybrid)
        try:
            result_hybrid = await engine_with_llm.classify_query(query)
            print(f"\nHybrid (Rules + LLM):")
            print(f"  Primary: {result_hybrid.primary_paradigm.value}")
            print(f"  Confidence: {result_hybrid.confidence:.2%}")
            print(
                f"  Distribution: {', '.join([f'{p.value}: {s:.1%}' for p, s in result_hybrid.distribution.items()])}"
            )

            # Show LLM reasoning if available
            if result_hybrid.reasoning.get(result_hybrid.primary_paradigm):
                llm_reasoning = [
                    r
                    for r in result_hybrid.reasoning[result_hybrid.primary_paradigm]
                    if r.startswith("LLM:")
                ]
                if llm_reasoning:
                    print(f"  LLM Reasoning: {llm_reasoning[0]}")

        except Exception as e:
            print(f"\nHybrid classification failed: {e}")
            print("(This is expected if LLM API is not configured)")

    print("\n" + "=" * 80)
    print("LLM Integration Test Complete")

    # Compare scores to show the effect of LLM integration
    print(
        "\nNote: With LLM integration enabled, classifications should be more nuanced"
    )
    print("and confidence scores may differ based on semantic understanding.")


if __name__ == "__main__":
    asyncio.run(test_llm_classification())
