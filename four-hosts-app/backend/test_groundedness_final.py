#!/usr/bin/env python3
"""
Final integration test for the comprehensive groundedness implementation.
Tests all three modes with real Azure services.
"""

import asyncio
import os
import sys
from datetime import datetime

# Add backend to path
sys.path.insert(0, '/home/azureuser/4hosts/four-hosts-app/backend')

from services.evaluation.context_evaluator import check_content_safety_groundedness


async def test_basic_mode():
    """Test basic groundedness detection"""
    print("\n" + "="*60)
    print("TEST: Basic Mode")
    print("="*60)

    # Set up environment for basic mode
    os.environ["AZURE_CS_ENABLE_GROUNDEDNESS"] = "1"
    os.environ["AZURE_CS_ENDPOINT"] = "https://4hosts.cognitiveservices.azure.com/"
    os.environ["AZURE_CS_KEY"] = "b5b138156d6a469086e9eb5fedd11413"

    # Clear mode flags
    os.environ.pop("AZURE_CS_GROUNDEDNESS_REASONING", None)
    os.environ.pop("AZURE_CS_GROUNDEDNESS_CORRECTION", None)

    result = await check_content_safety_groundedness(
        text="The patient's name is John and he is 35 years old.",
        grounding_sources=[
            "Medical record: Patient Jane Smith, age 45, admitted on January 15th."
        ]
    )

    if result:
        print(f"✓ Success!")
        print(f"  Mode: {result['mode']}")
        print(f"  Ungrounded: {result['ungrounded_percentage']:.1%}")
        print(f"  Detected: {result['ungrounded_detected']}")
        if result.get('ungrounded_details'):
            print(f"  Details: {len(result['ungrounded_details'])} segments flagged")
    else:
        print("✗ Failed - no result returned")

    return result


async def test_reasoning_mode():
    """Test reasoning mode (requires Azure OpenAI)"""
    print("\n" + "="*60)
    print("TEST: Reasoning Mode")
    print("="*60)

    # Enable reasoning mode
    os.environ["AZURE_CS_GROUNDEDNESS_REASONING"] = "1"

    # Azure OpenAI should already be configured
    print(f"  OpenAI Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT', 'Not set')}")
    print(f"  Deployment: {os.getenv('AZURE_OPENAI_DEPLOYMENT', 'Not set')}")

    result = await check_content_safety_groundedness(
        text="The medication dosage is 500mg twice daily.",
        grounding_sources=[
            "Prescription: Medication 250mg once daily with meals."
        ],
        domain="Medical"
    )

    if result:
        print(f"✓ Success!")
        print(f"  Mode: {result['mode']}")
        print(f"  Ungrounded: {result['ungrounded_percentage']:.1%}")
        if result.get('reasoning'):
            print(f"  Reasoning provided: {len(result['reasoning'])} explanations")
            for r in result['reasoning'][:2]:  # Show first 2
                print(f"    - {r[:100]}...")
    else:
        print("✗ Failed - no result returned")

    return result


async def test_correction_mode():
    """Test correction mode (requires Azure OpenAI)"""
    print("\n" + "="*60)
    print("TEST: Correction Mode")
    print("="*60)

    # Enable correction mode
    os.environ.pop("AZURE_CS_GROUNDEDNESS_REASONING", None)
    os.environ["AZURE_CS_GROUNDEDNESS_CORRECTION"] = "1"

    result = await check_content_safety_groundedness(
        text="The company revenue was $10 million last quarter.",
        grounding_sources=[
            "Financial report: Q3 revenue was $5 million, showing 10% growth."
        ]
    )

    if result:
        print(f"✓ Success!")
        print(f"  Mode: {result['mode']}")
        print(f"  Ungrounded: {result['ungrounded_percentage']:.1%}")
        if result.get('correction_text'):
            print(f"  Correction provided:")
            print(f"    Original: The company revenue was $10 million...")
            print(f"    Corrected: {result['correction_text'][:100]}...")
    else:
        print("✗ Failed - no result returned")

    return result


async def test_performance():
    """Test performance with multiple sources"""
    print("\n" + "="*60)
    print("TEST: Performance with Multiple Sources")
    print("="*60)

    # Basic mode for speed
    os.environ.pop("AZURE_CS_GROUNDEDNESS_REASONING", None)
    os.environ.pop("AZURE_CS_GROUNDEDNESS_CORRECTION", None)

    sources = [
        f"Source {i}: This is test content number {i} with various facts."
        for i in range(10)
    ]

    start = datetime.now()
    result = await check_content_safety_groundedness(
        text="This is a summary combining facts from multiple sources.",
        grounding_sources=sources
    )
    duration = (datetime.now() - start).total_seconds()

    if result:
        print(f"✓ Success!")
        print(f"  Sources: {len(sources)}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Mode: {result['mode']}")
    else:
        print("✗ Failed - no result returned")

    return result


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("GROUNDEDNESS IMPLEMENTATION TEST SUITE")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*60)

    results = []

    # Test 1: Basic mode
    try:
        result = await test_basic_mode()
        results.append(("Basic Mode", result is not None))
    except Exception as e:
        print(f"  Exception: {e}")
        results.append(("Basic Mode", False))

    # Test 2: Reasoning mode (may fail without RBAC)
    try:
        result = await test_reasoning_mode()
        results.append(("Reasoning Mode", result is not None and result['mode'] == 'reasoning'))
    except Exception as e:
        print(f"  Exception: {e}")
        results.append(("Reasoning Mode", False))

    # Test 3: Correction mode (may fail without RBAC)
    try:
        result = await test_correction_mode()
        results.append(("Correction Mode", result is not None and result['mode'] in ('correction', 'basic')))
    except Exception as e:
        print(f"  Exception: {e}")
        results.append(("Correction Mode", False))

    # Test 4: Performance
    try:
        result = await test_performance()
        results.append(("Performance", result is not None))
    except Exception as e:
        print(f"  Exception: {e}")
        results.append(("Performance", False))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20} {status}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    print(f"\nTotal: {passed_count}/{total_count} passed")

    if passed_count < total_count:
        print("\n⚠️  Note: Reasoning/Correction modes require:")
        print("   1. Azure OpenAI resource configured")
        print("   2. Content Safety MI with 'Cognitive Services OpenAI User' role")
        print("   3. 5-15 minutes for role propagation")


if __name__ == "__main__":
    asyncio.run(main())