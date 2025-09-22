#!/usr/bin/env python3
"""
Diagnostic script to debug search API issues.
Tests each search API individually with detailed logging and timeout tracking.
"""
import asyncio
import os
import sys
import time
import traceback
from datetime import datetime
from dotenv import load_dotenv
import aiohttp
import json

# Load environment variables
load_dotenv()

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_header(text):
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}")

def print_result(label, status, detail="", duration=None):
    status_icon = "✅" if status == "SUCCESS" else "❌" if status == "FAILED" else "⚠️"
    duration_str = f" ({duration:.2f}s)" if duration else ""
    print(f"{status_icon} {label}: {status}{duration_str}")
    if detail:
        print(f"   └─ {detail}")

async def test_brave_api():
    """Test Brave Search API directly"""
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        return "FAILED", "No BRAVE_SEARCH_API_KEY in environment"

    # Brave API keys can be various lengths (typically 31-32 chars)
    print(f"   API key length: {len(api_key)} chars")
    if len(api_key) < 20:
        return "FAILED", f"API key appears invalid (length: {len(api_key)})"

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key
    }
    params = {
        "q": "test search query",
        "count": 5
    }

    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results_count = len(data.get("web", {}).get("results", []))
                    return "SUCCESS", f"Got {results_count} results"
                elif response.status == 401:
                    return "FAILED", "Invalid API key (401 Unauthorized)"
                elif response.status == 429:
                    return "FAILED", "Rate limited (429 Too Many Requests)"
                else:
                    text = await response.text()
                    return "FAILED", f"HTTP {response.status}: {text[:100]}"
    except asyncio.TimeoutError:
        return "FAILED", "Request timed out after 10 seconds"
    except Exception as e:
        return "FAILED", f"Exception: {str(e)}"

async def test_google_cse_api():
    """Test Google Custom Search Engine API directly"""
    api_key = os.getenv("GOOGLE_CSE_API_KEY")
    cx = os.getenv("GOOGLE_CSE_CX")

    if not api_key:
        return "FAILED", "No GOOGLE_CSE_API_KEY in environment"
    if not cx:
        return "FAILED", "No GOOGLE_CSE_CX in environment"

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx,
        "q": "test search query",
        "num": 5
    }

    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results_count = len(data.get("items", []))
                    queries_used = data.get("queries", {}).get("request", [{}])[0].get("totalResults", 0)
                    return "SUCCESS", f"Got {results_count} results (total available: {queries_used})"
                elif response.status == 403:
                    data = await response.json()
                    error = data.get("error", {})
                    if "quotaExceeded" in str(error):
                        return "FAILED", "Daily quota exceeded"
                    return "FAILED", f"Forbidden: {error.get('message', 'Unknown error')}"
                elif response.status == 400:
                    return "FAILED", "Invalid API key or CX ID"
                else:
                    text = await response.text()
                    return "FAILED", f"HTTP {response.status}: {text[:100]}"
    except asyncio.TimeoutError:
        return "FAILED", "Request timed out after 10 seconds"
    except Exception as e:
        return "FAILED", f"Exception: {str(e)}"

async def test_search_manager():
    """Test the actual SearchAPIManager initialization and basic search"""
    try:
        from services.search_apis import create_search_manager, SearchConfig

        # Create manager
        manager = create_search_manager()

        # Check which APIs were loaded
        api_count = len(manager.apis)
        api_names = list(manager.apis.keys())

        if api_count == 0:
            return "FAILED", "No search APIs were initialized"

        # Try a simple search using search_with_priority
        config = SearchConfig(max_results=5)

        # Create a simple query candidate for testing
        from services.query_planning.types import QueryCandidate
        candidate = QueryCandidate(
            query="test query",
            label="test",
            stage="rule_based"
        )

        try:
            results = await asyncio.wait_for(
                manager.search_with_priority([candidate], config),
                timeout=20.0
            )
            result_count = sum(len(r) for r in results.values()) if results else 0
            return "SUCCESS", f"Manager initialized with {api_count} APIs ({', '.join(api_names)}), search returned {result_count} results"
        except asyncio.TimeoutError:
            return "FAILED", f"Search timed out (APIs: {', '.join(api_names)})"
        except Exception as e:
            return "WARNING", f"Manager created but search failed: {str(e)[:100]}"

    except ImportError as e:
        return "FAILED", f"Import error: {str(e)}"
    except Exception as e:
        return "FAILED", f"Exception: {str(e)[:200]}"

async def test_network_connectivity():
    """Test basic network connectivity"""
    test_urls = [
        ("Google", "https://www.google.com"),
        ("Brave API", "https://api.search.brave.com"),
        ("Google API", "https://www.googleapis.com")
    ]

    results = []
    for name, url in test_urls:
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.head(url) as response:
                    if response.status < 400:
                        results.append(f"{name}: OK")
                    else:
                        results.append(f"{name}: HTTP {response.status}")
        except Exception as e:
            results.append(f"{name}: Failed ({str(e)[:30]})")

    if all("OK" in r for r in results):
        return "SUCCESS", "All endpoints reachable"
    else:
        return "WARNING", " | ".join(results)

async def check_environment_vars():
    """Check all search-related environment variables"""
    required_vars = {
        "BRAVE_SEARCH_API_KEY": "Brave Search API key",
        "GOOGLE_CSE_API_KEY": "Google Custom Search API key",
        "GOOGLE_CSE_CX": "Google Custom Search Engine ID",
    }

    optional_vars = {
        "SEARCH_DISABLE_BRAVE": "Brave disable flag",
        "SEARCH_DISABLE_GOOGLE": "Google disable flag",
        "SEARCH_USE_PRIORITY_MODE": "Priority mode flag",
        "SEARCH_TASK_TIMEOUT_SEC": "Search task timeout",
        "SEARCH_PROVIDER_TIMEOUT_SEC": "Provider timeout",
    }

    missing = []
    configured = []
    flags = []

    for var, desc in required_vars.items():
        value = os.getenv(var)
        if value:
            # Mask sensitive data
            if "KEY" in var:
                masked = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
                configured.append(f"{var}={masked} (len={len(value)})")
            else:
                configured.append(f"{var}={value}")
        else:
            missing.append(desc)

    for var, desc in optional_vars.items():
        value = os.getenv(var)
        if value:
            flags.append(f"{var}={value}")

    details = []
    if configured:
        details.append(f"Found: {len(configured)} vars")
    if missing:
        details.append(f"Missing: {', '.join(missing)}")
    if flags:
        details.append(f"Flags: {', '.join(flags)}")

    if missing:
        return "WARNING", " | ".join(details)
    return "SUCCESS", " | ".join(details)

async def main():
    print_header("SEARCH API DIAGNOSTIC TEST")
    print(f"Started at: {datetime.now().isoformat()}")

    # Track all test results
    all_tests_passed = True

    # Test 1: Environment variables
    print_header("1. Environment Variables Check")
    start = time.time()
    status, detail = await check_environment_vars()
    print_result("Environment vars", status, detail, time.time() - start)
    if status == "FAILED":
        all_tests_passed = False

    # Test 2: Network connectivity
    print_header("2. Network Connectivity Test")
    start = time.time()
    status, detail = await test_network_connectivity()
    print_result("Network connectivity", status, detail, time.time() - start)
    if status == "FAILED":
        all_tests_passed = False

    # Test 3: Brave API
    print_header("3. Brave Search API Test")
    start = time.time()
    status, detail = await test_brave_api()
    print_result("Brave Search API", status, detail, time.time() - start)
    if status == "FAILED":
        all_tests_passed = False

    # Test 4: Google CSE API
    print_header("4. Google Custom Search API Test")
    start = time.time()
    status, detail = await test_google_cse_api()
    print_result("Google CSE API", status, detail, time.time() - start)
    if status == "FAILED":
        all_tests_passed = False

    # Test 5: Search Manager
    print_header("5. SearchAPIManager Integration Test")
    start = time.time()
    status, detail = await test_search_manager()
    print_result("Search Manager", status, detail, time.time() - start)
    if status == "FAILED":
        all_tests_passed = False

    # Summary
    print_header("TEST SUMMARY")
    if all_tests_passed:
        print("✅ All critical tests passed!")
        print("\nNext steps:")
        print("1. Check if the backend server is running correctly")
        print("2. Review the backend logs for any error messages during search")
        print("3. Try increasing timeout values in .env if searches are slow")
    else:
        print("❌ Some tests failed. Please review the results above.")
        print("\nCommon fixes:")
        print("1. Verify API keys are complete and valid")
        print("2. Check API quotas haven't been exceeded")
        print("3. Ensure network allows outbound HTTPS connections")
        print("4. Try running with increased timeouts")

if __name__ == "__main__":
    asyncio.run(main())