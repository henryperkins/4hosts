#!/usr/bin/env python3
"""
Test to reproduce the research timeout issue.
Simulates the full authenticated research flow.
"""
import asyncio
import os
import sys
import time
import json
from datetime import datetime
from dotenv import load_dotenv
import structlog

# Load environment variables
load_dotenv()

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger(__name__)

async def test_research_flow():
    """Test the full research orchestrator flow"""
    try:
        from services.research_orchestrator import ResearchOrchestrator
        from models.context_models import ResearchQuery
        from models.synthesis_models import ResearchResponse
        from services.search_apis import create_search_manager

        # Create research query
        query = ResearchQuery(
            query="What are the latest developments in AI safety?",
            depth="standard",
            include_academic=True,
            max_results=10
        )

        # Initialize search manager first
        logger.info("Creating search manager...")
        search_manager = create_search_manager()
        logger.info(f"Search manager initialized with APIs: {list(search_manager.apis.keys())}")

        # Initialize orchestrator with the search manager
        logger.info("Creating research orchestrator...")
        orchestrator = ResearchOrchestrator(search_manager=search_manager)
        await orchestrator.initialize()

        # Set a research ID
        research_id = f"test-{int(time.time())}"
        logger.info(f"Starting research with ID: {research_id}")

        # Execute research with timeout monitoring
        phases = [
            "classification",
            "context_engineering",
            "search",
            "synthesis"
        ]

        start_time = time.time()

        try:
            # Run with a reasonable timeout
            logger.info("Starting research execution...")
            response = await asyncio.wait_for(
                orchestrator.execute_research(query, research_id=research_id),
                timeout=60.0  # 60 second total timeout
            )

            elapsed = time.time() - start_time
            logger.info(f"✅ Research completed successfully in {elapsed:.1f} seconds")

            if response:
                logger.info(f"Response type: {type(response)}")
                if hasattr(response, 'paradigm'):
                    logger.info(f"Paradigm: {response.paradigm}")
                if hasattr(response, 'search_results_count'):
                    logger.info(f"Search results: {response.search_results_count}")
                if hasattr(response, 'answer'):
                    logger.info(f"Answer length: {len(response.answer) if response.answer else 0} chars")

            return True

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"❌ Research timed out after {elapsed:.1f} seconds")

            # Try to diagnose where it got stuck
            if hasattr(orchestrator, 'current_phase'):
                logger.error(f"Stuck in phase: {orchestrator.current_phase}")

            return False

    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_search_only():
    """Test just the search phase independently"""
    try:
        from services.search_apis import create_search_manager, SearchConfig
        from services.query_planning.types import QueryCandidate

        logger.info("Testing search phase independently...")

        # Create search manager
        manager = create_search_manager()
        config = SearchConfig(max_results=10)

        # Create test queries
        candidates = [
            QueryCandidate(
                query="AI safety developments 2024",
                label="main",
                stage="rule_based"
            ),
            QueryCandidate(
                query="artificial intelligence alignment research",
                label="academic",
                stage="paradigm"
            )
        ]

        start_time = time.time()

        try:
            results = await asyncio.wait_for(
                manager.search_with_priority(candidates, config),
                timeout=30.0
            )

            elapsed = time.time() - start_time
            total_results = sum(len(r) for r in results.values())

            logger.info(f"✅ Search completed in {elapsed:.1f}s, got {total_results} results")

            for label, res_list in results.items():
                logger.info(f"  {label}: {len(res_list)} results")

            return True

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"❌ Search timed out after {elapsed:.1f}s")
            return False

    except Exception as e:
        logger.error(f"Search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("="*60)
    print(" RESEARCH FLOW TIMEOUT DIAGNOSTIC")
    print("="*60)
    print(f"Started at: {datetime.now().isoformat()}")
    print()

    # Test 1: Search only
    print("TEST 1: Search Phase Only")
    print("-"*40)
    search_ok = await test_search_only()
    print()

    # Test 2: Full research flow
    print("TEST 2: Full Research Flow")
    print("-"*40)
    research_ok = await test_research_flow()
    print()

    # Summary
    print("="*60)
    print(" SUMMARY")
    print("="*60)

    if search_ok and research_ok:
        print("✅ All tests passed - research flow is working!")
    elif search_ok and not research_ok:
        print("⚠️ Search works but full research fails")
        print("Issue is likely in context engineering or synthesis phases")
    elif not search_ok:
        print("❌ Search phase is failing")
        print("Check API keys and network connectivity")

    print("\nRecommendations:")
    if not research_ok:
        print("1. Check logs for timeout messages")
        print("2. Verify Azure OpenAI credentials for synthesis")
        print("3. Check if context engineering LLM calls are working")
        print("4. Review timeout settings in .env file")

if __name__ == "__main__":
    asyncio.run(main())