#!/usr/bin/env python3
"""Test script to verify API fixes"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()


async def test_search_apis():
    """Test search API initialization and basic functionality"""
    print("Testing Search APIs...")
    print("=" * 60)

    from services.search_apis import create_search_manager, SearchConfig

    # Create and initialize search manager
    manager = create_search_manager()
    await manager.initialize()

    print(f"✓ Search manager initialized")
    print(f"  Available APIs: {list(manager.apis.keys())}")
    print(f"  Fallback order: {manager.fallback_order}")

    # Test a simple search
    config = SearchConfig(max_results=5, language="en", region="us")
    query = "artificial intelligence"

    print(f"\nTesting search for: '{query}'")
    try:
        results = await manager.search_with_fallback(query, config)
        print(f"✓ Search successful! Got {len(results)} results")

        if results:
            print(f"\nFirst result:")
            print(f"  Title: {results[0].title}")
            print(f"  URL: {results[0].url}")
            print(f"  Source: {results[0].source}")
    except Exception as e:
        print(f"✗ Search failed: {str(e)}")

    # Cleanup
    await manager.cleanup()
    print("✓ Search manager cleaned up")


async def test_llm_client():
    """Test LLM client with Azure OpenAI"""
    print("\n\nTesting LLM Client...")
    print("=" * 60)

    from services.llm_client import llm_client

    try:
        # Test a simple completion
        prompt = "What is 2+2?"
        response = await llm_client.generate_completion(
            prompt=prompt, model="gpt-4o-mini", max_tokens=50, temperature=0.1
        )

        print(f"✓ LLM client working!")
        print(f"  Prompt: {prompt}")
        print(f"  Response: {response}")

        # Test with o3 model if configured
        if os.getenv("AZURE_OPENAI_DEPLOYMENT") == "o3":
            print("\nTesting o3 model...")
            response = await llm_client.generate_completion(
                prompt="Say 'Hello from o3'", model="o3", max_tokens=20, temperature=0.1
            )
            print(f"✓ o3 model working!")
            print(f"  Response: {response}")

    except Exception as e:
        print(f"✗ LLM client failed: {str(e)}")


async def test_research_orchestrator():
    """Test research orchestrator initialization"""
    print("\n\nTesting Research Orchestrator...")
    print("=" * 60)

    from services.research_orchestrator import (
        research_orchestrator,
        initialize_research_system,
    )

    try:
        await initialize_research_system()
        print("✓ Research orchestrator initialized")

        # Get stats
        stats = await research_orchestrator.get_execution_stats()
        print(f"  Stats: {stats}")

        # Cleanup
        await research_orchestrator.cleanup()
        print("✓ Research orchestrator cleaned up")

    except Exception as e:
        print(f"✗ Research orchestrator failed: {str(e)}")


async def main():
    """Run all tests"""
    print("Four Hosts Research API - Test Suite")
    print("=" * 60)

    # Check environment variables
    print("Environment Check:")
    print(
        f"  GOOGLE_SEARCH_API_KEY: {'✓' if os.getenv('GOOGLE_SEARCH_API_KEY') else '✗'}"
    )
    print(
        f"  GOOGLE_SEARCH_ENGINE_ID: {'✓' if os.getenv('GOOGLE_SEARCH_ENGINE_ID') else '✗'}"
    )
    print(
        f"  BRAVE_SEARCH_API_KEY: {'✓' if os.getenv('BRAVE_SEARCH_API_KEY') else '✗'}"
    )
    print(
        f"  AZURE_OPENAI_ENDPOINT: {'✓' if os.getenv('AZURE_OPENAI_ENDPOINT') else '✗'}"
    )
    print(
        f"  AZURE_OPENAI_API_KEY: {'✓' if os.getenv('AZURE_OPENAI_API_KEY') else '✗'}"
    )

    # Run tests
    await test_search_apis()
    await test_llm_client()
    await test_research_orchestrator()

    print("\n" + "=" * 60)
    print("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
