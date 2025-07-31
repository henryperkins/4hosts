import asyncio
from services.research_orchestrator_v2 import research_orchestrator_v2
from services.research_orchestrator_compat import research_orchestrator_compat
from models.context_models import (
    ClassificationResultSchema, UserContextSchema,
    ContextEngineeredQuerySchema, HostParadigm,
    QueryFeaturesSchema
)

async def test_v2_orchestrator():
    """Test V2 orchestrator directly"""

    # Create test data
    features = QueryFeaturesSchema(
        text="How can small businesses compete with Amazon?",
        tokens=["small", "businesses", "compete", "Amazon"],
        entities=["Amazon"],
        intent_signals=["compete", "strategy"],
        domain="business",
        urgency_score=0.5,
        complexity_score=0.7,
        emotional_valence=0.3
    )

    classification = ClassificationResultSchema(
        query="How can small businesses compete with Amazon?",
        primary_paradigm=HostParadigm.MAEVE,
        secondary_paradigm=None,
        distribution={HostParadigm.MAEVE: 0.8, HostParadigm.DOLORES: 0.2},
        confidence=0.85,
        features=features,
        reasoning={HostParadigm.MAEVE: ["Strategic business question"]}
    )

    user_context = UserContextSchema(
        user_id="test_user",
        role="PRO",
        max_sources=20
    )

    context_engineered = ContextEngineeredQuerySchema(
        original_query=classification.query,
        classification=classification,
        write_layer_output={},
        select_layer_output={},
        compress_layer_output={},
        isolate_layer_output={},
        refined_queries=[
            "small business compete Amazon strategy",
            "Amazon competitive disadvantages",
            "local business advantages over Amazon"
        ],
        search_strategy={"paradigm": "strategic"},
        context_metadata={}
    )

    # Execute research
    result = await research_orchestrator_v2.execute_research(
        classification,
        context_engineered,
        user_context
    )

    print(f"V2 Results: {len(result['results'])} sources found")
    print(f"Metadata: {result['metadata']}")

    # Test compatibility layer
    # Create mock old-style query
    from types import SimpleNamespace
    old_style_query = SimpleNamespace(
        original_query="How can small businesses compete with Amazon?",
        classification=SimpleNamespace(
            primary_paradigm=SimpleNamespace(value="strategic"),
            secondary_paradigm=None
        ),
        select_output=SimpleNamespace(
            search_queries=[
                {"query": "small business Amazon", "type": "main", "weight": 1.0}
            ]
        )
    )

    compat_result = await research_orchestrator_compat.execute_paradigm_research(
        old_style_query,
        max_results=20
    )

    print(f"Compatibility Results: {len(compat_result.filtered_results)} sources")

if __name__ == "__main__":
    asyncio.run(test_v2_orchestrator())
