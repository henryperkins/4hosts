#!/usr/bin/env python3
"""
Diagnostic script to test the research pipeline
"""

import asyncio
import sys
import os
import json
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_classification():
    """Test paradigm classification"""
    logger.info("Testing paradigm classification...")
    try:
        from services.classification_engine import ClassificationEngine
        from models.context_models import QueryContext

        engine = ClassificationEngine()

        test_queries = [
            "How does climate change affect the economy?",
            "What are the best practices for software development?",
            "Tell me about quantum computing",
        ]

        for query in test_queries:
            context = QueryContext(query=query)
            result = await engine.classify(context)
            logger.info(f"Query: {query}")
            logger.info(f"  Primary: {result.primary_paradigm}")
            logger.info(f"  Secondary: {result.secondary_paradigms}")
            logger.info(f"  Confidence: {result.confidence}")

        return True
    except Exception as e:
        logger.error(f"Classification test failed: {e}")
        return False

async def test_search_apis():
    """Test search API availability"""
    logger.info("Testing search APIs...")
    try:
        from services.search_apis import get_available_search_apis

        apis = await get_available_search_apis()
        logger.info(f"Available search APIs: {apis}")

        if not apis:
            logger.warning("No search APIs available!")
            return False

        return True
    except Exception as e:
        logger.error(f"Search API test failed: {e}")
        return False

async def test_research_orchestrator():
    """Test research orchestrator initialization"""
    logger.info("Testing research orchestrator...")
    try:
        from services.research_orchestrator import ResearchOrchestrator

        orchestrator = ResearchOrchestrator()
        logger.info("Research orchestrator initialized successfully")

        return True
    except Exception as e:
        logger.error(f"Research orchestrator test failed: {e}")
        return False

async def test_llm_client():
    """Test LLM client"""
    logger.info("Testing LLM client...")
    try:
        from services.llm_client import get_llm_client

        client = get_llm_client()
        if not client:
            logger.warning("LLM client not available")
            return False

        # Test simple completion
        messages = [{"role": "user", "content": "Say 'test successful' and nothing else"}]
        response = await client.generate_completion(messages, max_tokens=10)
        logger.info(f"LLM response: {response}")

        return True
    except Exception as e:
        logger.error(f"LLM client test failed: {e}")
        return False

async def test_full_research_pipeline():
    """Test the full research pipeline"""
    logger.info("Testing full research pipeline...")
    try:
        from services.research_orchestrator import ResearchOrchestrator
        from models.context_models import QueryContext, ResearchQuery
        from models.paradigms import HostParadigm

        orchestrator = ResearchOrchestrator()

        # Create a simple research query
        query = ResearchQuery(
            query="What is artificial intelligence?",
            use_llm=False,  # Start without LLM
            max_results=5,
            include_pdfs=False
        )

        logger.info(f"Executing research for: {query.query}")

        # Track progress
        progress_updates = []

        async def progress_callback(update):
            progress_updates.append(update)
            logger.info(f"Progress: {update}")

        # Execute research
        result = await orchestrator.orchestrate_research(
            query=query,
            progress_callback=progress_callback
        )

        if result:
            logger.info("Research completed successfully!")
            logger.info(f"  Total sources: {result.metadata.get('total_sources', 0)}")
            logger.info(f"  Paradigm: {result.paradigm}")
            logger.info(f"  Answer length: {len(result.answer) if result.answer else 0}")
        else:
            logger.warning("Research returned no results")
            return False

        return True
    except Exception as e:
        logger.error(f"Full pipeline test failed: {e}", exc_info=True)
        return False

async def test_database():
    """Test database connection"""
    logger.info("Testing database connection...")
    try:
        from database.connection import get_db_session
        from sqlalchemy import text

        async with get_db_session() as session:
            result = await session.execute(text("SELECT 1"))
            logger.info("Database connection successful")
            return True
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        return False

async def test_cache():
    """Test cache system"""
    logger.info("Testing cache system...")
    try:
        from services.cache import get_cache

        cache = await get_cache()
        if not cache:
            logger.warning("Cache not available (Redis might not be running)")
            return False

        # Test set/get
        await cache.set("test_key", "test_value", ttl=10)
        value = await cache.get("test_key")

        if value == "test_value":
            logger.info("Cache system working correctly")
            return True
        else:
            logger.error(f"Cache test failed: expected 'test_value', got {value}")
            return False
    except Exception as e:
        logger.error(f"Cache test failed: {e}")
        return False

async def check_environment():
    """Check environment variables"""
    logger.info("Checking environment variables...")

    required_vars = [
        "DATABASE_URL",
        "JWT_SECRET_KEY",
    ]

    optional_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT",
        "GOOGLE_CSE_API_KEY",
        "BRAVE_SEARCH_API_KEY",
        "REDIS_URL",
    ]

    missing_required = []
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
            logger.error(f"Missing required: {var}")
        else:
            logger.info(f"✓ {var} is set")

    missing_optional = []
    for var in optional_vars:
        if not os.getenv(var):
            missing_optional.append(var)
            logger.warning(f"Missing optional: {var}")
        else:
            logger.info(f"✓ {var} is set")

    if missing_required:
        logger.error(f"Missing required environment variables: {missing_required}")
        return False

    if missing_optional:
        logger.warning(f"Missing optional environment variables (some features may not work): {missing_optional}")

    return True

async def main():
    """Run all diagnostic tests"""
    logger.info("=" * 60)
    logger.info("Starting Four Hosts Research Diagnostic")
    logger.info("=" * 60)

    # Load environment variables
    from dotenv import load_dotenv
    env_path = backend_path / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")
    else:
        logger.warning(f"No .env file found at {env_path}")

    results = {}

    # Run tests in order
    tests = [
        ("Environment", check_environment),
        ("Database", test_database),
        ("Cache", test_cache),
        ("Classification", test_classification),
        ("Search APIs", test_search_apis),
        ("LLM Client", test_llm_client),
        ("Research Orchestrator", test_research_orchestrator),
        ("Full Pipeline", test_full_research_pipeline),
    ]

    for name, test_func in tests:
        logger.info("-" * 40)
        logger.info(f"Running: {name}")
        logger.info("-" * 40)
        try:
            result = await test_func()
            results[name] = result
            logger.info(f"{name}: {'✓ PASSED' if result else '✗ FAILED'}")
        except Exception as e:
            logger.error(f"{name}: ✗ FAILED with exception: {e}")
            results[name] = False

    # Summary
    logger.info("=" * 60)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 60)

    for name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{name:20} {status}")

    passed = sum(1 for r in results.values() if r)
    total = len(results)
    logger.info("-" * 40)
    logger.info(f"Results: {passed}/{total} tests passed")

    if passed < total:
        logger.error("Some tests failed. Please check the logs above for details.")
        return 1
    else:
        logger.info("All tests passed!")
        return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)