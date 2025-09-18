#!/usr/bin/env python3
"""
Test script for Brave Search API integration
Run this script to verify your Brave Search API implementation
"""

import asyncio
import os
import sys
import json
from dotenv import load_dotenv
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.search_apis import BraveSearchAPI, SearchConfig


def print_result(result, index):
    """Pretty print a search result"""
    print(f"\n{index}. {result.title}")
    print(f"   URL: {result.url}")
    print(f"   Domain: {result.domain}")
    print(f"   Type: {result.result_type}")
    if result.published_date:
        print(f"   Published: {result.published_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"   Snippet: {result.snippet[:150]}...")
    if result.bias_rating:
        print(f"   Source: {result.bias_rating}")


async def test_basic_search():
    """Test basic web search functionality"""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Web Search")
    print("=" * 60)

    # Load environment variables
    load_dotenv()

    # Get API key
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        print("Error: BRAVE_SEARCH_API_KEY not found in environment variables")
        print("Please set it in your .env file")
        print("Get your API key from: https://api-dashboard.search.brave.com/")
        return False

    # Create API instance
    brave_api = BraveSearchAPI(api_key=api_key)

    query = "artificial intelligence ethics 2024"
    config = SearchConfig(max_results=5, language="en", region="us")

    print(f"\nSearching for: '{query}'")
    print(
        f"Config: max_results={config.max_results}, language={config.language}, region={config.region}"
    )

    async with brave_api:
        try:
            results = await brave_api.search(query, config)

            if results:
                print(f"\nFound {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print_result(result, i)
                return True
            else:
                print("No results found")
                return False

        except Exception as e:
            print(f"Error: {str(e)}")
            return False


async def test_news_search():
    """Test news-specific search"""
    print("\n" + "=" * 60)
    print("TEST 2: News Search with Date Filter")
    print("=" * 60)

    load_dotenv()
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        return False

    brave_api = BraveSearchAPI(api_key=api_key)

    query = "technology breakthroughs"
    config = SearchConfig(
        max_results=5, date_range="w", source_types=["news"]  # Last week
    )

    print(f"\nSearching for: '{query}'")
    print(f"Config: date_range=last week, source_types=news only")

    async with brave_api:
        try:
            results = await brave_api.search(query, config)

            news_results = [r for r in results if r.result_type == "news"]
            if news_results:
                print(f"\nFound {len(news_results)} news results:")
                for i, result in enumerate(news_results[:5], 1):
                    print_result(result, i)
                return True
            else:
                print("No news results found")
                return False

        except Exception as e:
            print(f"Error: {str(e)}")
            return False


async def test_academic_search():
    """Test academic/FAQ/discussion search"""
    print("\n" + "=" * 60)
    print("TEST 3: Academic-style Search (FAQ & Discussions)")
    print("=" * 60)

    load_dotenv()
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        return False

    brave_api = BraveSearchAPI(api_key=api_key)

    query = "quantum computing explained"
    config = SearchConfig(max_results=10, source_types=["academic", "web"])

    print(f"\nSearching for: '{query}'")
    print(f"Config: source_types=academic+web")

    async with brave_api:
        try:
            results = await brave_api.search(query, config)

            # Categorize results by type
            by_type = {}
            for result in results:
                if result.result_type not in by_type:
                    by_type[result.result_type] = []
                by_type[result.result_type].append(result)

            print(f"\nFound {len(results)} total results:")
            for result_type, type_results in by_type.items():
                print(f"\n{result_type.upper()} Results ({len(type_results)}):")
                for i, result in enumerate(type_results[:3], 1):
                    print_result(result, i)

            return True

        except Exception as e:
            print(f"Error: {str(e)}")
            return False


async def test_safe_search():
    """Test safe search filtering"""
    print("\n" + "=" * 60)
    print("TEST 4: Safe Search Settings")
    print("=" * 60)

    load_dotenv()
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        return False

    brave_api = BraveSearchAPI(api_key=api_key)

    query = "adult content test"
    config = SearchConfig(max_results=3, safe_search="strict")

    print(f"\nSearching for: '{query}'")
    print(f"Config: safe_search=strict")

    async with brave_api:
        try:
            results = await brave_api.search(query, config)

            if results:
                print(f"\nFound {len(results)} filtered results:")
                for i, result in enumerate(results, 1):
                    print_result(result, i)
                return True
            else:
                print("No results found (expected with strict filtering)")
                return True

        except Exception as e:
            print(f"Error: {str(e)}")
            return False


async def test_search_manager_integration():
    """Test Brave Search integration with SearchAPIManager"""
    print("\n" + "=" * 60)
    print("TEST 5: SearchAPIManager Integration")
    print("=" * 60)

    from services.search_apis import create_search_manager

    manager = create_search_manager()

    # Check if Brave is configured
    if "brave" in manager.apis:
        print("✓ Brave Search API is configured in the manager")
        print(f"  Available APIs: {list(manager.apis.keys())}")
        print(f"  Fallback order: {manager.fallback_order}")

        # Test search with fallback
        config = SearchConfig(max_results=5)
        results = await manager.search_with_fallback(
            "latest technology trends 2024", config
        )

        if results:
            print(f"\n✓ Search successful! Found {len(results)} results")
            print(f"  First result: {results[0].title}")
            print(f"  Source: {results[0].source}")
            return True
        else:
            print("\n✗ No results found")
            return False
    else:
        print("✗ Brave Search API is not configured")
        print("  Make sure BRAVE_SEARCH_API_KEY is set in your .env file")
        return False


async def test_rate_limiting():
    """Test rate limit handling"""
    print("\n" + "=" * 60)
    print("TEST 6: Rate Limit Monitoring")
    print("=" * 60)

    load_dotenv()
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        return False

    brave_api = BraveSearchAPI(api_key=api_key)

    query = "rate limit test"
    config = SearchConfig(max_results=1)

    print(f"\nPerforming quick search to check rate limits...")

    async with brave_api:
        try:
            # Make a single request to check headers
            results = await brave_api.search(query, config)
            print("✓ Search completed successfully")
            print("  Check logs for rate limit warnings if any")
            return True

        except Exception as e:
            if "Rate limit exceeded" in str(e):
                print(f"⚠️  Rate limit hit: {str(e)}")
                return True  # Expected behavior
            else:
                print(f"Error: {str(e)}")
                return False


async def main():
    """Run all tests"""
    print("\nBrave Search API Test Suite")
    print("===========================")
    print("Make sure you have set BRAVE_SEARCH_API_KEY in your .env file")
    print("Get your API key from: https://api-dashboard.search.brave.com/\n")

    # Track test results
    tests = [
        ("Basic Web Search", test_basic_search),
        ("News Search", test_news_search),
        ("Academic Search", test_academic_search),
        ("Safe Search", test_safe_search),
        ("Manager Integration", test_search_manager_integration),
        ("Rate Limiting", test_rate_limiting),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\nTest '{test_name}' failed with unexpected error: {str(e)}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Brave Search API is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    asyncio.run(main())
