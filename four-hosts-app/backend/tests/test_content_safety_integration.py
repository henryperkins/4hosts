#!/usr/bin/env python3
"""
Integration test for Azure Content Safety groundedness detection.

This script tests the actual Content Safety API integration using the
configured credentials in the environment.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the function to test
from services.evaluation.context_evaluator import check_content_safety_groundedness


async def test_grounded_text():
    """Test text that is well grounded in sources"""
    print("\n=== Testing GROUNDED text ===")

    result = await check_content_safety_groundedness(
        text="The patient's name is Jane and she is 45 years old.",
        grounding_sources=[
            "Medical Record: Patient Jane Smith, age 45, admitted on January 15th."
        ],
        task_type="Summarization"
    )

    if result:
        print(f"✓ API call successful")
        print(f"  Ungrounded detected: {result['ungrounded_detected']}")
        print(f"  Ungrounded percentage: {result['ungrounded_percentage']:.2%}")
        if result['ungrounded_details']:
            print(f"  Details: {result['ungrounded_details']}")
    else:
        print("✗ No result returned - check configuration")

    return result


async def test_ungrounded_text():
    """Test text that is NOT grounded in sources"""
    print("\n=== Testing UNGROUNDED text ===")

    result = await check_content_safety_groundedness(
        text="The patient's name is John and he is 35 years old.",
        grounding_sources=[
            "Medical Record: Patient Jane Smith, age 45, admitted on January 15th."
        ],
        task_type="Summarization"
    )

    if result:
        print(f"✓ API call successful")
        print(f"  Ungrounded detected: {result['ungrounded_detected']}")
        print(f"  Ungrounded percentage: {result['ungrounded_percentage']:.2%}")
        if result['ungrounded_details']:
            print(f"  Details: {result['ungrounded_details']}")
    else:
        print("✗ No result returned - check configuration")

    return result


async def test_qna_task():
    """Test Q&A task type"""
    print("\n=== Testing Q&A task ===")

    result = await check_content_safety_groundedness(
        text="She currently gets paid $12/hour at the bank.",
        grounding_sources=[
            "I currently work for a bank that pays me $10/hour and it's not unheard of to get a raise in 6 months."
        ],
        task_type="QnA"
    )

    if result:
        print(f"✓ API call successful")
        print(f"  Ungrounded detected: {result['ungrounded_detected']}")
        print(f"  Ungrounded percentage: {result['ungrounded_percentage']:.2%}")
        if result['ungrounded_details']:
            print(f"  Details: {result['ungrounded_details']}")
    else:
        print("✗ No result returned - check configuration")

    return result


async def main():
    """Run all integration tests"""

    # Check if credentials are configured
    endpoint = os.getenv("CONTENT_SAFETY_ENDPOINT")
    api_key = os.getenv("CONTENT_SAFETY_API_KEY")

    if not endpoint or not api_key:
        print("⚠️  Content Safety credentials not configured in environment")
        print("   Set CONTENT_SAFETY_ENDPOINT and CONTENT_SAFETY_API_KEY")
        return

    print(f"Using endpoint: {endpoint}")
    print(f"API key configured: {'✓' if api_key else '✗'}")

    # Run tests
    results = []

    result1 = await test_grounded_text()
    results.append(("Grounded text", result1))

    result2 = await test_ungrounded_text()
    results.append(("Ungrounded text", result2))

    result3 = await test_qna_task()
    results.append(("Q&A task", result3))

    # Summary
    print("\n=== SUMMARY ===")
    for test_name, result in results:
        if result:
            status = "✓ PASS"
            if test_name == "Grounded text":
                # Should detect as grounded (low ungrounded percentage)
                if result['ungrounded_percentage'] < 0.3:
                    status = "✓ PASS (correctly grounded)"
                else:
                    status = "✗ FAIL (should be grounded)"
            elif test_name == "Ungrounded text":
                # Should detect as ungrounded (high ungrounded percentage)
                if result['ungrounded_percentage'] > 0.5:
                    status = "✓ PASS (correctly ungrounded)"
                else:
                    status = "✗ FAIL (should be ungrounded)"
        else:
            status = "✗ FAIL (no response)"

        print(f"{test_name}: {status}")


if __name__ == "__main__":
    asyncio.run(main())