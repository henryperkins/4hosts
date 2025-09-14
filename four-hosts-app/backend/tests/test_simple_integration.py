#!/usr/bin/env python3
"""Simple test to verify classification engine integration"""

import sys

sys.path.append("/home/azureuser/4hosts/four-hosts-app/backend")

import asyncio


async def test_integration():
    # Import required modules
    from services.classification_engine import classification_engine, HostParadigm
    from models import Paradigm, ParadigmClassification

    test_query = "How can we expose corporate corruption in the tech industry?"

    # Test direct classification engine
    result = await classification_engine.classify_query(test_query)

    # Test the conversion logic (this is what main.py does)
    classification = ParadigmClassification(
        primary=Paradigm(result.primary_paradigm.value),
        secondary=(
            Paradigm(result.secondary_paradigm.value)
            if result.secondary_paradigm
            else None
        ),
        distribution={
            "revolutionary": result.distribution.get(HostParadigm.DOLORES, 0),
            "devotion": result.distribution.get(HostParadigm.TEDDY, 0),
            "analytical": result.distribution.get(HostParadigm.BERNARD, 0),
            "strategic": result.distribution.get(HostParadigm.MAEVE, 0),
        },
        confidence=result.confidence,
        explanation={
            result.primary_paradigm.value: "; ".join(
                result.reasoning.get(result.primary_paradigm, [])[:2]
            )
        },
    )

    print(f"Query: {test_query}")
    print(f"Primary Paradigm: {classification.primary.value}")
    print(f"Confidence: {classification.confidence:.2%}")
    print(f"Distribution: {classification.distribution}")
    print(f"Explanation: {classification.explanation}")
    print("\n✅ Classification engine integration successful!")


if __name__ == "__main__":
    asyncio.run(test_integration())
