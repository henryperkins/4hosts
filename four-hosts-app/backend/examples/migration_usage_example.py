"""
Migration Usage Examples
Shows how to use the V1→V2 migration in practice
"""

import asyncio
import logging
from datetime import datetime

# Import configuration
from config.migration_config import (
    migration_config, get_research_store, get_context_pipeline,
    apply_environment_config, use_v2_research_store, use_v2_context_engineering
)

# Import models
from models.context_models import (
    ResearchRequestSchema, ClassificationResultSchema, 
    QueryFeaturesSchema, HostParadigm, ResearchStatus
)
from services.classification_engine import ClassificationResult, QueryFeatures

logger = logging.getLogger(__name__)


async def example_1_basic_research_workflow():
    """Example 1: Basic research workflow with migration support"""
    print("📝 Example 1: Basic Research Workflow")
    
    # Configuration automatically chooses V1 or V2 based on environment
    research_store = get_research_store()
    context_pipeline = get_context_pipeline()
    
    print(f"Using V2 Research Store: {use_v2_research_store()}")
    print(f"Using V2 Context Engineering: {use_v2_context_engineering()}")
    
    # Initialize services
    await research_store.initialize()
    
    # Create research request (works with both V1 and V2)
    if use_v2_research_store():
        # V2 approach with schema
        request = ResearchRequestSchema(
            id=f"example-1-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            query="climate change impacts on coastal cities",
            user_context={
                "user_id": "example-user",
                "role": "RESEARCHER"
            },
            status=ResearchStatus.PROCESSING
        )
        research_id = await research_store.store_research_request(request)
    else:
        # V1 approach with dictionary
        research_id = f"example-1-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        request_data = {
            "id": research_id,
            "query": "climate change impacts on coastal cities",
            "user_id": "example-user",
            "status": "processing",
            "created_at": datetime.utcnow().isoformat()
        }
        await research_store.set(research_id, request_data)
    
    print(f"✓ Research request stored with ID: {research_id}")
    
    # Create sample classification
    features = QueryFeatures(
        text="climate change impacts on coastal cities",
        tokens=["climate", "change", "impacts", "coastal", "cities"],
        entities=["climate", "cities"],
        intent_signals=["research", "analysis"],
        domain="environmental",
        urgency_score=0.8,
        complexity_score=0.7,
        emotional_valence=0.0
    )
    
    classification = ClassificationResult(
        query="climate change impacts on coastal cities",
        primary_paradigm=HostParadigm.BERNARD,
        secondary_paradigm=HostParadigm.DOLORES,
        distribution={
            HostParadigm.BERNARD: 0.7,
            HostParadigm.DOLORES: 0.6,
            HostParadigm.MAEVE: 0.3,
            HostParadigm.TEDDY: 0.4
        },
        confidence=0.8,
        features=features,
        reasoning={
            HostParadigm.BERNARD: ["scientific research", "data analysis"],
            HostParadigm.DOLORES: ["environmental justice", "community impact"]
        }
    )
    
    # Process through context engineering (works with both V1 and V2)
    if use_v2_context_engineering():
        # V2 approach via bridge
        context_result = await context_pipeline.process_query(
            classification,
            include_debug=migration_config.include_debug_info
        )
        queries = context_result.refined_queries
        print(f"✓ V2 Context engineering generated {len(queries)} queries")
    else:
        # V1 approach
        context_result = await context_pipeline.process_query(classification)
        queries = []
        if hasattr(context_result.select_output, 'search_queries'):
            for query_data in context_result.select_output.search_queries:
                if isinstance(query_data, dict):
                    queries.append(query_data.get('query', ''))
                else:
                    queries.append(str(query_data))
        print(f"✓ V1 Context engineering generated {len(queries)} queries")
    
    # Update status
    if use_v2_research_store():
        await research_store.update_status(research_id, ResearchStatus.COMPLETED)
    else:
        await research_store.update_field(research_id, "status", "completed")
    
    print(f"✓ Research completed: {research_id}")
    print(f"Generated queries: {queries[:3]}")  # Show first 3 queries
    
    return research_id, queries


async def example_2_gradual_migration():
    """Example 2: Gradual migration from V1 to V2"""
    print("\n🔄 Example 2: Gradual Migration")
    
    # Start with V1 configuration
    print("Starting with V1 configuration...")
    apply_environment_config("development")  # V1 by default
    
    research_store = get_research_store()
    await research_store.initialize()
    
    # Store some data with V1
    research_id = f"migration-example-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    v1_data = {
        "id": research_id,
        "query": "renewable energy storage solutions",
        "user_id": "migration-user",
        "status": "processing",
        "created_at": datetime.utcnow().isoformat()
    }
    await research_store.set(research_id, v1_data)
    print(f"✓ Data stored with V1: {research_id}")
    
    # Migrate to V2 configuration
    print("Migrating to V2 configuration...")
    apply_environment_config("staging")  # V2 enabled
    
    # Get new store instance (V2)
    research_store_v2 = get_research_store()
    await research_store_v2.initialize()
    
    # Verify we can still access the data (if using Redis)
    if hasattr(research_store_v2, 'use_redis') and research_store_v2.use_redis:
        retrieved_data = await research_store_v2.get_research(research_id)
        if retrieved_data:
            print(f"✓ V2 store can access V1 data: {retrieved_data['query']}")
        else:
            print("⚠️  V2 store cannot access V1 data (expected with in-memory fallback)")
    
    # Test V2 functionality
    from models.context_models import ResearchRequestSchema
    v2_request = ResearchRequestSchema(
        id=f"v2-example-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        query="renewable energy storage solutions",
        user_context={"user_id": "migration-user", "role": "RESEARCHER"},
        status=ResearchStatus.PROCESSING
    )
    
    v2_research_id = await research_store_v2.store_research_request(v2_request)
    print(f"✓ Data stored with V2: {v2_research_id}")
    
    return research_id, v2_research_id


async def example_3_performance_comparison():
    """Example 3: Performance comparison between V1 and V2"""
    print("\n⚡ Example 3: Performance Comparison")
    
    import time
    
    # Test data
    test_query = "artificial intelligence ethics in healthcare"
    
    # Create test classification
    features = QueryFeatures(
        text=test_query,
        tokens=test_query.split(),
        entities=["AI", "healthcare"],
        intent_signals=["research"],
        domain="technology",
        urgency_score=0.6,
        complexity_score=0.8,
        emotional_valence=0.0
    )
    
    classification = ClassificationResult(
        query=test_query,
        primary_paradigm=HostParadigm.BERNARD,
        secondary_paradigm=HostParadigm.TEDDY,
        distribution={HostParadigm.BERNARD: 0.8, HostParadigm.TEDDY: 0.6},
        confidence=0.8,
        features=features,
        reasoning={HostParadigm.BERNARD: ["research needed"]}
    )
    
    # Test V1 performance
    apply_environment_config("development")  # V1
    v1_pipeline = get_context_pipeline()
    
    start_time = time.time()
    v1_result = await v1_pipeline.process_query(classification)
    v1_time = time.time() - start_time
    
    v1_queries = []
    if hasattr(v1_result.select_output, 'search_queries'):
        for q in v1_result.select_output.search_queries:
            if isinstance(q, dict):
                v1_queries.append(q.get('query', ''))
    
    print(f"V1 Performance: {v1_time:.3f}s, {len(v1_queries)} queries")
    
    # Test V2 performance
    apply_environment_config("staging")  # V2
    v2_pipeline = get_context_pipeline()
    
    start_time = time.time()
    v2_result = await v2_pipeline.process_query(classification, include_debug=False)
    v2_time = time.time() - start_time
    
    print(f"V2 Performance: {v2_time:.3f}s, {len(v2_result.refined_queries)} queries")
    
    # Performance analysis
    improvement = ((v1_time - v2_time) / v1_time) * 100 if v1_time > 0 else 0
    print(f"Performance improvement: {improvement:.1f}%")
    
    # Feature comparison
    print(f"V1 features: Basic pipeline, {len(v1_queries)} queries")
    print(f"V2 features: Debug info, metadata, {len(v2_result.refined_queries)} queries")
    
    if hasattr(v2_result, 'context_metadata'):
        metadata = v2_result.context_metadata
        print(f"V2 metadata: {metadata.get('total_transformations', 0)} transformations")
    
    return v1_time, v2_time


async def example_4_error_handling_and_fallback():
    """Example 4: Error handling and fallback mechanisms"""
    print("\n🛡️  Example 4: Error Handling and Fallback")
    
    # Enable V2 with fallback
    apply_environment_config("staging")
    migration_config.enable_fallback_to_v1 = True
    
    from services.context_engineering_bridge import context_engineering_bridge
    
    # Test fallback mechanism
    print("Testing fallback mechanism...")
    
    # Create test classification
    features = QueryFeatures(
        text="test error handling",
        tokens=["test", "error"],
        entities=[],
        intent_signals=["test"],
        domain="testing",
        urgency_score=0.5,
        complexity_score=0.5,
        emotional_valence=0.0
    )
    
    classification = ClassificationResult(
        query="test error handling",
        primary_paradigm=HostParadigm.BERNARD,
        secondary_paradigm=None,
        distribution={HostParadigm.BERNARD: 0.9},
        confidence=0.9,
        features=features,
        reasoning={HostParadigm.BERNARD: ["test"]}
    )
    
    # Test V2 with potential fallback
    try:
        context_engineering_bridge.use_v2 = True
        result = await context_engineering_bridge.process_query(classification)
        print(f"✓ V2 processing successful: {len(result.refined_queries)} queries")
    except Exception as e:
        print(f"⚠️  V2 processing failed: {e}")
        
        # Fallback to V1
        print("Falling back to V1...")
        context_engineering_bridge.use_v2 = False
        result = await context_engineering_bridge.process_query(classification)
        print(f"✓ V1 fallback successful")
    
    # Test research store error handling
    research_store = get_research_store()
    
    try:
        await research_store.initialize()
        
        if use_v2_research_store():
            # Test V2 store with error handling
            metrics = research_store.get_metrics()
            print(f"✓ V2 store metrics: {metrics['redis_enabled']} Redis, {metrics['cache_enabled']} cache")
        else:
            print("✓ V1 store initialized successfully")
            
    except Exception as e:
        print(f"⚠️  Store initialization error: {e}")
    
    return True


async def main():
    """Run all migration examples"""
    print("🚀 Migration Usage Examples")
    print("=" * 50)
    
    try:
        # Example 1: Basic workflow
        research_id_1, queries_1 = await example_1_basic_research_workflow()
        
        # Example 2: Gradual migration
        old_id, new_id = await example_2_gradual_migration()
        
        # Example 3: Performance comparison
        v1_time, v2_time = await example_3_performance_comparison()
        
        # Example 4: Error handling
        await example_4_error_handling_and_fallback()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        print(f"📊 Performance: V1={v1_time:.3f}s, V2={v2_time:.3f}s")
        print(f"📝 Research IDs: {research_id_1}, {old_id}, {new_id}")
        
        # Final configuration status
        print(f"🔧 Final config: V2 Store={use_v2_research_store()}, V2 Context={use_v2_context_engineering()}")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run examples
    asyncio.run(main())
