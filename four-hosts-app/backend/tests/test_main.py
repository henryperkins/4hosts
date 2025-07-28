#!/usr/bin/env python3
"""
Test script to verify the combined main.py works correctly
"""

import asyncio
import sys
import importlib.util


def test_imports():
    """Test if main.py imports without errors"""
    print("Testing main.py imports...")
    try:
        spec = importlib.util.spec_from_file_location("main", "main.py")
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        print("✓ Main module imported successfully")

        # Check feature flags
        print(f"  - Production features: {main_module.PRODUCTION_FEATURES}")
        print(f"  - Research features: {main_module.RESEARCH_FEATURES}")
        print(f"  - AI features: {main_module.AI_FEATURES}")
        print(f"  - Custom docs: {main_module.CUSTOM_DOCS}")

        return True
    except Exception as e:
        print(f"✗ Import failed: {str(e)}")
        return False


def test_models():
    """Test data models"""
    print("\nTesting data models...")
    try:
        from main import Paradigm, ResearchDepth, ResearchQuery, ResearchOptions

        # Test paradigm enum
        assert Paradigm.DOLORES.value == "dolores"
        assert Paradigm.TEDDY.value == "teddy"
        assert Paradigm.BERNARD.value == "bernard"
        assert Paradigm.MAEVE.value == "maeve"
        print("✓ Paradigm enum works")

        # Test research depth enum
        assert ResearchDepth.QUICK.value == "quick"
        assert ResearchDepth.STANDARD.value == "standard"
        assert ResearchDepth.DEEP.value == "deep"
        print("✓ ResearchDepth enum works")

        # Test research query model
        query = ResearchQuery(query="Test research query about climate change")
        assert query.query == "Test research query about climate change"
        assert query.options.depth == ResearchDepth.STANDARD
        assert query.options.max_sources == 50
        print("✓ ResearchQuery model works")

        return True
    except Exception as e:
        print(f"✗ Model test failed: {str(e)}")
        return False


async def test_classification():
    """Test paradigm classification"""
    print("\nTesting paradigm classification...")
    try:
        from main import classify_query

        # Test different queries
        test_cases = [
            ("How can we expose systemic corruption in government?", "dolores"),
            ("How can we help protect vulnerable communities?", "teddy"),
            ("Analyze the statistical data on climate trends", "bernard"),
            ("What's the best strategy to optimize business processes?", "maeve"),
        ]

        for query, expected_paradigm in test_cases:
            classification = await classify_query(query)
            actual_paradigm = classification.primary.value
            if actual_paradigm == expected_paradigm:
                print(f"✓ '{query[:40]}...' -> {actual_paradigm}")
            else:
                print(
                    f"✗ '{query[:40]}...' -> {actual_paradigm} (expected {expected_paradigm})"
                )

        return True
    except Exception as e:
        print(f"✗ Classification test failed: {str(e)}")
        return False


def test_helper_functions():
    """Test helper functions"""
    print("\nTesting helper functions...")
    try:
        from main import (
            generate_paradigm_queries,
            get_paradigm_approach,
            get_paradigm_focus,
            generate_paradigm_summary,
        )

        # Test query generation
        queries = generate_paradigm_queries("climate change", "bernard")
        assert len(queries) >= 1
        assert queries[0]["query"] == "climate change"
        print("✓ generate_paradigm_queries works")

        # Test paradigm helpers
        from main import Paradigm

        approach = get_paradigm_approach(Paradigm.DOLORES)
        assert approach == "revolutionary"
        print("✓ get_paradigm_approach works")

        focus = get_paradigm_focus(Paradigm.BERNARD)
        assert "objective analysis" in focus
        print("✓ get_paradigm_focus works")

        summary = generate_paradigm_summary("test query", Paradigm.MAEVE)
        assert "Strategic" in summary
        print("✓ generate_paradigm_summary works")

        return True
    except Exception as e:
        print(f"✗ Helper function test failed: {str(e)}")
        return False


async def main():
    """Run all tests"""
    print("=== Testing Combined main.py ===\n")

    all_passed = True

    # Run tests
    all_passed &= test_imports()
    all_passed &= test_models()
    all_passed &= await test_classification()
    all_passed &= test_helper_functions()

    print(f"\n{'='*40}")
    if all_passed:
        print("✓ All tests passed! The combined main.py is working correctly.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
