#!/usr/bin/env python3
"""
Fix for the search timeout issue in the research flow.
This script adds better error handling and logging to diagnose and fix the hanging search phase.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def add_search_timeout_logging():
    """Add comprehensive logging to search execution"""

    # Patch 1: Add detailed logging to search_with_priority
    search_apis_patch = """
# Add this at the beginning of search_with_priority method (line ~1764)
logger.info(
    "search_with_priority called",
    planned_count=len(planned) if planned else 0,
    primary_api=self.primary_api,
    fallback_apis=self.fallback_apis,
    research_id=research_id
)

# Add timeout logging in _search_single_provider (line ~1960)
logger.info(
    f"Searching {name} with timeout {timeout}s for {len(planned)} candidates",
    research_id=research_id
)
"""

    # Patch 2: Add timeout handling to orchestrator
    orchestrator_patch = """
# In execute_research method, wrap search execution with explicit timeout (line ~1929)
try:
    logger.info(f"Calling search_with_plan with {len(planned_candidates)} candidates", research_id=research_id)

    # Add explicit timeout wrapper
    search_timeout = float(os.getenv("SEARCH_TASK_TIMEOUT_SEC", "30"))
    results_by_label = await asyncio.wait_for(
        self.search_manager.search_with_plan(planned_candidates, search_config),
        timeout=search_timeout
    )

    logger.info(f"Search completed successfully with {len(results_by_label)} labels", research_id=research_id)

except asyncio.TimeoutError:
    logger.error(f"Search timed out after {search_timeout}s", research_id=research_id)
    # Return empty results instead of hanging
    results_by_label = {}

except Exception as e:
    logger.error(f"Search failed: {e}", research_id=research_id, exc_info=True)
    results_by_label = {}
"""

    print("Suggested patches to fix the search timeout issue:")
    print("=" * 60)
    print("\n1. SEARCH APIS PATCH (services/search_apis.py):")
    print(search_apis_patch)
    print("\n2. ORCHESTRATOR PATCH (services/research_orchestrator.py):")
    print(orchestrator_patch)
    print("=" * 60)

def verify_api_keys():
    """Verify all required API keys are present"""
    print("\nChecking API Keys:")
    print("-" * 40)

    keys_status = {
        "BRAVE_SEARCH_API_KEY": os.getenv("BRAVE_SEARCH_API_KEY"),
        "GOOGLE_CSE_API_KEY": os.getenv("GOOGLE_CSE_API_KEY"),
        "GOOGLE_CSE_CX": os.getenv("GOOGLE_CSE_CX"),
        "EXA_API_KEY": os.getenv("EXA_API_KEY"),
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
    }

    for key, value in keys_status.items():
        if value:
            masked = f"{value[:8]}...{value[-4:]}" if len(value) > 12 and "KEY" in key else value
            print(f"✅ {key}: {masked}")
        else:
            print(f"❌ {key}: Not set")

    # Check search disable flags
    print("\nSearch Provider Flags:")
    print("-" * 40)
    flags = {
        "SEARCH_DISABLE_BRAVE": os.getenv("SEARCH_DISABLE_BRAVE", "0"),
        "SEARCH_DISABLE_GOOGLE": os.getenv("SEARCH_DISABLE_GOOGLE", "0"),
        "SEARCH_DISABLE_EXA": os.getenv("SEARCH_DISABLE_EXA", "0"),
        "SEARCH_USE_PRIORITY_MODE": os.getenv("SEARCH_USE_PRIORITY_MODE", "0"),
    }

    for key, value in flags.items():
        status = "Disabled" if value in ["1", "true"] else "Enabled"
        print(f"  {key}: {value} ({status})")

def suggest_env_fixes():
    """Suggest environment variable fixes"""
    print("\nRecommended .env Configuration:")
    print("-" * 40)

    fixes = []

    # Check if Google CSE is missing but not disabled
    if not os.getenv("GOOGLE_CSE_API_KEY") and os.getenv("SEARCH_DISABLE_GOOGLE", "0") == "0":
        fixes.append("Either set GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX, or set SEARCH_DISABLE_GOOGLE=1")

    # Suggest timeout increases if defaults are too low
    task_timeout = int(os.getenv("SEARCH_TASK_TIMEOUT_SEC", "30"))
    if task_timeout < 60:
        fixes.append(f"Consider increasing SEARCH_TASK_TIMEOUT_SEC from {task_timeout} to 60")

    provider_timeout = int(os.getenv("SEARCH_PROVIDER_TIMEOUT_SEC", "25"))
    if provider_timeout < 30:
        fixes.append(f"Consider increasing SEARCH_PROVIDER_TIMEOUT_SEC from {provider_timeout} to 30")

    if fixes:
        for fix in fixes:
            print(f"• {fix}")
    else:
        print("✅ Configuration looks good!")

async def test_search_flow():
    """Test the search flow with timeout handling"""
    from services.search_apis import create_search_manager, SearchConfig
    from services.query_planning.types import QueryCandidate

    print("\nTesting Search Flow:")
    print("-" * 40)

    try:
        manager = create_search_manager()
        print(f"✅ Search manager created with APIs: {list(manager.apis.keys())}")

        # Test with a simple query
        config = SearchConfig(max_results=5)
        candidate = QueryCandidate(
            query="test search query",
            label="test",
            stage="rule_based"
        )

        print("Testing search with 10s timeout...")
        try:
            results = await asyncio.wait_for(
                manager.search_with_priority([candidate], config),
                timeout=10.0
            )
            print(f"✅ Search succeeded, got {len(results)} results")
        except asyncio.TimeoutError:
            print("❌ Search timed out - this is the issue!")
            print("   The search is hanging and not returning within timeout")
        except Exception as e:
            print(f"❌ Search failed with error: {e}")

    except Exception as e:
        print(f"❌ Failed to create search manager: {e}")

def main():
    print("=" * 60)
    print("SEARCH TIMEOUT FIX DIAGNOSTIC")
    print("=" * 60)

    # Step 1: Verify API keys
    verify_api_keys()

    # Step 2: Suggest env fixes
    suggest_env_fixes()

    # Step 3: Test search flow
    asyncio.run(test_search_flow())

    # Step 4: Show patch suggestions
    add_search_timeout_logging()

    print("\n" + "=" * 60)
    print("IMMEDIATE ACTIONS TO FIX THE ISSUE:")
    print("=" * 60)
    print("""
1. QUICK FIX - Increase timeouts in .env:
   SEARCH_TASK_TIMEOUT_SEC=60
   SEARCH_PROVIDER_TIMEOUT_SEC=30
   SEARCH_PER_PROVIDER_TIMEOUT_SEC=20

2. DISABLE UNUSED PROVIDERS - Add to .env:
   SEARCH_DISABLE_GOOGLE=1  (if you don't have Google CSE keys)

3. RESTART THE BACKEND:
   Kill the current backend process and restart it to reload environment variables

4. APPLY THE PATCHES:
   The patches above add better timeout handling and logging
   """)

if __name__ == "__main__":
    main()