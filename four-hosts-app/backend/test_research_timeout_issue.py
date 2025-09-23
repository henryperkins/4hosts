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
from logging_config import configure_logging

# Load environment variables
load_dotenv()

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force pretty logs for interactive diagnostic
# Enable pretty console logs when not running under CI but avoid re-initialising
os.environ.setdefault("LOG_PRETTY", "1")
configure_logging()

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
    logger.info("Research flow timeout diagnostic start",
                started_at=datetime.now().isoformat())

    # Test 1: Search only
    logger.info("TEST 1: Search Phase Only")
    search_ok = await test_search_only()

    # Test 2: Full research flow
    logger.info("TEST 2: Full Research Flow")
    research_ok = await test_research_flow()

    # Summary
    logger.info("Summary start")

    if search_ok and research_ok:
        logger.info("All diagnostics passed")
    elif search_ok and not research_ok:
        logger.warning("Search OK, full flow failed",
                       hint="Check context engineering or synthesis phases")
    elif not search_ok:
        logger.error("Search phase failing",
                     hint="Verify API keys / network connectivity")

    if not research_ok:
        logger.info("Recommendations",
                    steps=[
                        "Check logs for timeout messages",
                        "Verify Azure OpenAI credentials",
                        "Check context engineering LLM calls",
                        "Review timeout settings in .env",
                    ])

if __name__ == "__main__":
    asyncio.run(main())
