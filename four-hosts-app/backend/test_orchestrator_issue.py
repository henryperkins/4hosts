#!/usr/bin/env python3
"""
Diagnostic test to identify where the research orchestrator gets stuck.
"""
import asyncio
import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv
import structlog
from logging_config import configure_logging

# Load environment variables
load_dotenv()

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force pretty console logs for this diagnostic if LOG_PRETTY not set
# Ensure pretty logs for developer diagnostics but avoid double configuration
os.environ.setdefault("LOG_PRETTY", "1")
configure_logging()

logger = structlog.get_logger(__name__)

async def test_orchestrator_search():
    """Test the orchestrator's search execution specifically"""
    try:
        from services.research_orchestrator import ResearchOrchestrator
        from services.enhanced_integration import enhanced_classification_engine
        from services.context_engineering import context_pipeline
        from services.search_apis import create_search_manager
        from models.research import ResearchQuery, ResearchOptions
        from models.base import ResearchDepth

        query_text = "What are the latest developments in artificial intelligence?"

        logger.info("=" * 60)
        logger.info("Testing Research Orchestrator Search Phase")
        logger.info("=" * 60)

        # Step 1: Classification
        logger.info("Step 1: Classifying query...")
        start = time.time()
        try:
            classification = await enhanced_classification_engine.classify_query(query_text)
            logger.info(f"✅ Classification completed in {time.time() - start:.2f}s")
            logger.info(f"   Primary paradigm: {classification.primary_paradigm}")
        except Exception as e:
            logger.error(f"❌ Classification failed: {e}")
            return False

        # Step 2: Context Engineering
        logger.info("\nStep 2: Context engineering...")
        start = time.time()
        try:
            context_engineered = await context_pipeline.process_query(
                classification,
                research_id="test"
            )
            logger.info(f"✅ Context engineering completed in {time.time() - start:.2f}s")

            # Check what queries were generated
            if hasattr(context_engineered, 'queries'):
                logger.info(f"   Generated {len(context_engineered.queries)} search queries")
                for i, q in enumerate(context_engineered.queries[:3]):
                    logger.info(f"   Query {i+1}: {q[:50]}...")
        except Exception as e:
            logger.error(f"❌ Context engineering failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Step 3: Initialize Orchestrator
        logger.info("\nStep 3: Initializing orchestrator...")
        try:
            # The orchestrator creates its own search manager internally
            orchestrator = ResearchOrchestrator()
            await orchestrator.initialize()

            # Check what search manager it has
            if orchestrator.search_manager:
                logger.info(f"   Search APIs available: {list(orchestrator.search_manager.apis.keys())}")
            else:
                logger.warning("   No search manager initialized!")

            logger.info("✅ Orchestrator initialized")
        except Exception as e:
            logger.error(f"❌ Orchestrator initialization failed: {e}")
            return False

        # Step 4: Execute search phase only
        logger.info("\nStep 4: Executing search phase...")
        research_id = f"test-{int(time.time())}"
        start = time.time()

        try:
            # Try to call the search method directly
            logger.info("   Calling orchestrator._execute_search...")

            # Use a shorter timeout for testing
            search_results = await asyncio.wait_for(
                orchestrator._execute_search(
                    classification=classification,
                    context_engineered=context_engineered,
                    research_id=research_id
                ),
                timeout=30.0
            )

            elapsed = time.time() - start
            logger.info(f"✅ Search completed in {elapsed:.2f}s")

            if search_results:
                logger.info(f"   Retrieved {len(search_results)} results")
                # Show first few results
                for i, result in enumerate(search_results[:3]):
                    if hasattr(result, 'title'):
                        logger.info(f"   Result {i+1}: {result.title[:50]}...")
            else:
                logger.warning("   No results returned")

            return True

        except asyncio.TimeoutError:
            elapsed = time.time() - start
            logger.error(f"❌ Search phase timed out after {elapsed:.2f}s")
            logger.error("   This is where the issue occurs!")
            return False
        except AttributeError as e:
            logger.error(f"❌ Method not found: {e}")
            logger.info("   Trying alternative approach...")

            # Try the full execute_research method with minimal options
            try:
                result = await asyncio.wait_for(
                    orchestrator.execute_research(
                        classification=classification,
                        context_engineered=context_engineered,
                        user_context={"query": query_text},
                        research_id=research_id,
                        synthesize_answer=False  # Skip synthesis to isolate search
                    ),
                    timeout=30.0
                )

                elapsed = time.time() - start
                logger.info(f"✅ Research completed in {elapsed:.2f}s")

                if result and 'results' in result:
                    logger.info(f"   Retrieved {len(result['results'])} results")

                return True

            except asyncio.TimeoutError:
                elapsed = time.time() - start
                logger.error(f"❌ Full research timed out after {elapsed:.2f}s")
                return False

    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    success = await test_orchestrator_search()

    banner = "=" * 60
    logger.info(banner)
    if success:
        logger.info("Search phase diagnostic passed")
    else:
        logger.error("Search phase diagnostic failed",
                     possible_causes=[
                        "Invalid generated queries",
                        "Search manager not initialized",
                        "Timeout too restrictive",
                        "Async deadlock",
                     ])
    logger.info(banner)

if __name__ == "__main__":
    asyncio.run(main())
