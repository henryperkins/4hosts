#!/usr/bin/env python3
"""
Test script for Four Hosts Answer Generation System
Phase 4: Tests paradigm-specific synthesis and presentation
"""

import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.answer_generator import (
    SynthesisContext,
    DoloresAnswerGenerator,
    TeddyAnswerGenerator,
)
from services.answer_generator_continued import (
    BernardAnswerGenerator,
    MaeveAnswerGenerator,
    AnswerGenerationOrchestrator,
    answer_orchestrator,
)


# Mock search results for testing
def get_mock_search_results(query: str) -> list:
    """Generate mock search results for testing"""
    return [
        {
            "title": f"Revolutionary Analysis: {query}",
            "url": "https://example.com/article1",
            "snippet": "This article exposes systemic issues and power imbalances in the current system. Evidence shows repeated patterns of exploitation and corruption at the highest levels.",
            "domain": "investigativereport.com",
            "credibility_score": 0.85,
            "published_date": "2024-01-15",
            "result_type": "web",
        },
        {
            "title": f"Support Resources for {query}",
            "url": "https://example.com/article2",
            "snippet": "Comprehensive guide to available resources and support systems. Multiple organizations offer help including counseling, material support, and community connections.",
            "domain": "communitysupport.org",
            "credibility_score": 0.92,
            "published_date": "2024-01-10",
            "result_type": "web",
        },
        {
            "title": f"Statistical Analysis of {query}",
            "url": "https://arxiv.org/article3",
            "snippet": "Meta-analysis of 45 studies (n=12,500) reveals significant correlations. Linear regression models show R²=0.67 with p<0.001. Effect sizes range from d=0.5 to d=0.8.",
            "domain": "arxiv.org",
            "credibility_score": 0.95,
            "published_date": "2024-01-05",
            "result_type": "academic",
        },
        {
            "title": f"Strategic Framework for {query}",
            "url": "https://example.com/article4",
            "snippet": "Three-phase implementation strategy with clear KPIs. Phase 1: Quick wins through rapid prototyping. Phase 2: Scale successful experiments. Phase 3: Operational excellence.",
            "domain": "strategyconsulting.com",
            "credibility_score": 0.88,
            "published_date": "2024-01-08",
            "result_type": "web",
        },
        {
            "title": f"Hidden Patterns in {query}",
            "url": "https://example.com/article5",
            "snippet": "Investigation reveals previously unknown connections and systemic failures. Documents obtained through FOIA requests show deliberate misrepresentation of facts.",
            "domain": "whistleblower.net",
            "credibility_score": 0.82,
            "published_date": "2024-01-12",
            "result_type": "web",
        },
    ]


# Mock context engineering output
def get_mock_context_engineering() -> dict:
    """Generate mock context engineering output"""
    return {
        "write_output": {
            "documentation_focus": "Expose systemic issues and empower resistance",
            "key_themes": ["injustice", "power", "resistance", "accountability"],
            "narrative_frame": "victim-oppressor dynamics",
        },
        "select_output": {
            "search_queries": [
                {"query": "expose corruption", "weight": 0.8},
                {"query": "systemic failure", "weight": 0.7},
            ],
            "source_preferences": ["investigative", "alternative"],
            "max_sources": 100,
        },
        "compress_output": {
            "compression_ratio": 0.7,
            "priority_elements": ["evidence", "impact", "accountability"],
        },
        "isolate_output": {
            "isolation_strategy": "pattern_of_injustice",
            "focus_areas": ["root causes", "responsible parties"],
        },
    }


async def test_individual_generators():
    """Test each paradigm generator individually"""
    print("🧪 Testing Individual Answer Generators")
    print("=" * 60)

    test_query = "How can small businesses compete with Amazon?"
    search_results = get_mock_search_results(test_query)
    context_engineering = get_mock_context_engineering()

    generators = [
        ("Dolores (Revolutionary)", DoloresAnswerGenerator()),
        ("Teddy (Devotion)", TeddyAnswerGenerator()),
        ("Bernard (Analytical)", BernardAnswerGenerator()),
        ("Maeve (Strategic)", MaeveAnswerGenerator()),
    ]

    for name, generator in generators:
        print(f"\n{name}:")
        try:
            # Create synthesis context
            context = SynthesisContext(
                query=test_query,
                paradigm=generator.paradigm,
                search_results=search_results,
                context_engineering=context_engineering,
                max_length=1000,
                metadata={"research_id": f"test_{generator.paradigm}_001"},
            )

            # Generate answer
            answer = await generator.generate_answer(context)

            print(f"  ✓ Generated answer with {len(answer.sections)} sections")
            print(f"  ✓ Summary: {answer.summary[:100]}...")
            print(f"  ✓ Confidence: {answer.confidence_score:.2f}")
            print(f"  ✓ Synthesis Quality: {answer.synthesis_quality:.2f}")
            print(f"  ✓ Citations: {len(answer.citations)}")
            print(f"  ✓ Action Items: {len(answer.action_items)}")

            # Show section titles
            print("  Sections:")
            for section in answer.sections:
                print(f"    - {section.title} ({section.word_count} words)")

        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            import traceback

            traceback.print_exc()


async def test_orchestrator():
    """Test the answer generation orchestrator"""
    print("\n🎯 Testing Answer Generation Orchestrator")
    print("=" * 60)

    test_queries = [
        "What are the ethical implications of AI?",
        "How to support homeless veterans?",
        "Climate change impact on agriculture",
        "Strategies for market expansion",
    ]

    paradigm_mappings = [
        ("dolores", "Exposing AI ethics violations"),
        ("teddy", "Supporting vulnerable populations"),
        ("bernard", "Analyzing empirical data"),
        ("maeve", "Strategic market positioning"),
    ]

    for i, (query, (paradigm, description)) in enumerate(
        zip(test_queries, paradigm_mappings)
    ):
        print(f"\nTest {i+1}: {description}")
        print(f"Query: {query}")
        print(f"Paradigm: {paradigm}")

        try:
            # Generate answer
            answer = await answer_orchestrator.generate_answer(
                paradigm=paradigm,
                query=query,
                search_results=get_mock_search_results(query),
                context_engineering=get_mock_context_engineering(),
                options={"research_id": f"orch_test_{i+1}", "max_length": 1500},
            )

            print(f"  ✓ Answer generated successfully")
            print(f"  ✓ Generation time: {answer.generation_time:.2f}s")
            print(f"  ✓ First action item: {answer.action_items[0]['action']}")

        except Exception as e:
            print(f"  ❌ Error: {str(e)}")


async def test_multi_paradigm():
    """Test multi-paradigm answer generation"""
    print("\n🔄 Testing Multi-Paradigm Answer Generation")
    print("=" * 60)

    query = "Impact of social media on mental health"

    print(f"Query: {query}")
    print("Primary: Bernard (Analytical)")
    print("Secondary: Teddy (Devotion)")

    try:
        combined = await answer_orchestrator.generate_multi_paradigm_answer(
            primary_paradigm="bernard",
            secondary_paradigm="teddy",
            query=query,
            search_results=get_mock_search_results(query),
            context_engineering=get_mock_context_engineering(),
            options={"research_id": "multi_test_001"},
        )

        print("\n  ✓ Multi-paradigm answer generated")
        print(
            f"  ✓ Primary paradigm sections: {len(combined['primary_paradigm']['answer'].sections)}"
        )
        if combined["secondary_paradigm"]:
            print(
                f"  ✓ Secondary paradigm sections: {len(combined['secondary_paradigm']['answer'].sections)}"
            )
        print(f"  ✓ Combined synthesis quality: {combined['synthesis_quality']:.2f}")

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()


async def test_citation_system():
    """Test the citation management system"""
    print("\n📚 Testing Citation System")
    print("=" * 60)

    generator = BernardAnswerGenerator()
    query = "Effectiveness of renewable energy"

    context = SynthesisContext(
        query=query,
        paradigm="bernard",
        search_results=get_mock_search_results(query),
        context_engineering=get_mock_context_engineering(),
        metadata={"research_id": "cite_test_001"},
    )

    try:
        answer = await generator.generate_answer(context)

        print(f"Citations created: {len(answer.citations)}")
        for cite_id, citation in list(answer.citations.items())[:3]:
            print(f"\n{cite_id}:")
            print(f"  Title: {citation.source_title}")
            print(f"  Domain: {citation.domain}")
            print(f"  Credibility: {citation.credibility_score:.2f}")
            print(f"  Type: {citation.fact_type}")
            print(f"  Alignment: {citation.paradigm_alignment:.2f}")

        print("\n  ✓ Citation system working correctly")

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")


async def run_all_tests():
    """Run all answer generation tests"""
    print("🚀 Four Hosts Answer Generation System - Phase 4 Tests")
    print("=" * 70)

    # Initialize system
    print("\nInitializing Answer Generation System...")
    from services.answer_generator_continued import initialize_answer_generation

    await initialize_answer_generation()
    print("✓ System initialized")

    # Run test suites
    await test_individual_generators()
    await test_orchestrator()
    await test_multi_paradigm()
    await test_citation_system()

    # Show final stats
    print("\n📊 GENERATION STATISTICS")
    print("=" * 60)
    stats = answer_orchestrator.get_generation_stats()
    print(f"Total answers generated: {stats['total_generated']}")
    if stats["total_generated"] > 0:
        print("Paradigm distribution:")
        for paradigm, count in stats["paradigm_distribution"].items():
            print(f"  {paradigm}: {count}")
        print(f"Last generation: {stats['last_generation']}")

    print("\n🎉 All tests completed!")
    print("\n📋 Next Steps:")
    print("   1. Integrate with actual LLM APIs (OpenAI/Anthropic)")
    print("   2. Connect to Research Execution Layer")
    print("   3. Implement fact verification pipeline")
    print("   4. Add real-time progress tracking")
    print("   5. Deploy to production environment")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
