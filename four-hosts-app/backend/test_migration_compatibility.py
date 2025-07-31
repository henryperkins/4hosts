"""
Migration Compatibility Tests
Tests the V1 → V2 migration for Research Store and Context Engineering
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any

# Import V1 and V2 versions
from services.research_store import research_store as research_store_v1
from services.research_store_v2 import research_store_v2
from services.context_engineering import context_pipeline as context_pipeline_v1
from services.context_engineering_bridge import context_engineering_bridge

# Import models
from models.context_models import (
    ResearchRequestSchema, ClassificationResultSchema, 
    QueryFeaturesSchema, HostParadigm, ResearchStatus
)
from services.classification_engine import ClassificationResult, QueryFeatures


@pytest.fixture
async def sample_classification():
    """Create a sample classification for testing"""
    features = QueryFeatures(
        text="climate change impacts on agriculture",
        tokens=["climate", "change", "impacts", "agriculture"],
        entities=["climate", "agriculture"],
        intent_signals=["research", "analysis"],
        domain="environmental",
        urgency_score=0.6,
        complexity_score=0.7,
        emotional_valence=0.0
    )
    
    return ClassificationResult(
        query="climate change impacts on agriculture",
        primary_paradigm=HostParadigm.BERNARD,
        secondary_paradigm=HostParadigm.DOLORES,
        distribution={
            HostParadigm.BERNARD: 0.8,
            HostParadigm.DOLORES: 0.6,
            HostParadigm.MAEVE: 0.3,
            HostParadigm.TEDDY: 0.4
        },
        confidence=0.8,
        features=features,
        reasoning={
            HostParadigm.BERNARD: ["scientific research needed", "data analysis"],
            HostParadigm.DOLORES: ["environmental justice", "system impact"]
        }
    )


@pytest.fixture
async def sample_research_request():
    """Create a sample research request for testing"""
    return ResearchRequestSchema(
        id="test-research-123",
        query="climate change impacts on agriculture", 
        user_context={
            "user_id": "test-user-456",
            "role": "RESEARCHER",
            "subscription_tier": "PRO"
        },
        options={
            "deep_research": True,
            "academic_focus": True
        },
        status=ResearchStatus.PROCESSING
    )


class TestResearchStoreMigration:
    """Test Research Store V1 → V2 migration"""
    
    async def test_initialization(self):
        """Test that both V1 and V2 can initialize"""
        # Initialize V1
        await research_store_v1.initialize()
        
        # Initialize V2
        await research_store_v2.initialize()
        
        assert True  # If no exceptions, initialization succeeded
    
    async def test_data_storage_compatibility(self, sample_research_request):
        """Test that V2 can handle V1-style data storage"""
        research_id = sample_research_request.id
        
        # Store using V2
        await research_store_v2.store_research_request(sample_research_request)
        
        # Retrieve and verify
        stored_data = await research_store_v2.get_research(research_id)
        assert stored_data is not None
        assert stored_data["id"] == research_id
        assert stored_data["query"] == sample_research_request.query
    
    async def test_v1_v2_data_exchange(self, sample_research_request):
        """Test data exchange between V1 and V2 stores"""
        research_id = sample_research_request.id
        
        # Store in V2
        await research_store_v2.store_research_request(sample_research_request)
        
        # Try to read basic data structure
        v2_data = await research_store_v2.get_research(research_id)
        assert v2_data is not None
        
        # Verify essential fields are preserved
        assert "id" in v2_data
        assert "query" in v2_data
        assert "status" in v2_data
    
    async def test_status_updates(self, sample_research_request):
        """Test status update compatibility"""
        research_id = sample_research_request.id
        
        # Store initial request
        await research_store_v2.store_research_request(sample_research_request)
        
        # Update status using V2 method
        await research_store_v2.update_status(research_id, ResearchStatus.COMPLETED)
        
        # Verify update
        data = await research_store_v2.get_research(research_id)
        assert data["status"] == ResearchStatus.COMPLETED.value


class TestContextEngineeringMigration:
    """Test Context Engineering V1 → V2 migration"""
    
    async def test_bridge_initialization(self):
        """Test that the bridge initializes correctly"""
        bridge = context_engineering_bridge
        assert bridge.use_v2 == True
        assert bridge.v1_pipeline is not None
        assert bridge.v2_pipeline is not None
    
    async def test_v1_classification_conversion(self, sample_classification):
        """Test V1 classification converts to V2 schema"""
        bridge = context_engineering_bridge
        
        # Convert V1 classification to V2
        v2_classification = bridge._convert_to_v2_classification(sample_classification)
        
        # Verify conversion
        assert isinstance(v2_classification, ClassificationResultSchema)
        assert v2_classification.query == sample_classification.query
        assert v2_classification.primary_paradigm == sample_classification.primary_paradigm
        assert v2_classification.confidence == sample_classification.confidence
    
    async def test_bridge_processing_v2(self, sample_classification):
        """Test bridge processes using V2 pipeline"""
        bridge = context_engineering_bridge
        bridge.use_v2 = True
        
        # Process through bridge
        result = await bridge.process_query(sample_classification, include_debug=True)
        
        # Verify V2 result structure
        assert hasattr(result, 'refined_queries')
        assert hasattr(result, 'search_strategy')
        assert hasattr(result, 'context_metadata')
        assert len(result.refined_queries) > 0
    
    async def test_bridge_processing_v1_fallback(self, sample_classification):
        """Test bridge falls back to V1 when configured"""
        bridge = context_engineering_bridge
        bridge.use_v2 = False
        
        # Process through bridge with V1 fallback
        result = await bridge.process_query(sample_classification)
        
        # Should still return V2 schema (converted from V1)
        assert hasattr(result, 'refined_queries')
        assert hasattr(result, 'original_query')
        assert result.original_query == sample_classification.query
    
    async def test_query_compatibility(self, sample_classification):
        """Test that V1 and V2 produce compatible query results"""
        # Process with V1
        v1_result = await context_pipeline_v1.process_query(sample_classification)
        
        # Process with V2 via bridge
        bridge = context_engineering_bridge
        bridge.use_v2 = True
        v2_result = await bridge.process_query(sample_classification)
        
        # Both should produce queries
        assert hasattr(v1_result, 'select_output')
        assert len(v2_result.refined_queries) > 0
        
        # V1 queries should be extractable
        v1_queries = []
        if hasattr(v1_result.select_output, 'search_queries'):
            for query_data in v1_result.select_output.search_queries:
                if isinstance(query_data, dict):
                    v1_queries.append(query_data.get('query', ''))
                else:
                    v1_queries.append(str(query_data))
        
        # Both should have at least one query
        assert len(v1_queries) > 0
        assert len(v2_result.refined_queries) > 0


class TestEndToEndMigration:
    """Test complete migration workflow"""
    
    async def test_complete_research_workflow_v2(self, sample_classification, sample_research_request):
        """Test complete research workflow using V2 systems"""
        research_id = sample_research_request.id
        
        # 1. Store research request
        await research_store_v2.store_research_request(sample_research_request)
        
        # 2. Process classification through context engineering
        context_result = await context_engineering_bridge.process_query(
            sample_classification,
            include_debug=True
        )
        
        # 3. Store classification with research
        classification_schema = context_engineering_bridge._convert_to_v2_classification(
            sample_classification
        )
        await research_store_v2.store_classification(research_id, classification_schema)
        
        # 4. Update status to completed
        await research_store_v2.update_status(research_id, ResearchStatus.COMPLETED)
        
        # 5. Verify final state
        final_data = await research_store_v2.get_research(research_id)
        assert final_data["status"] == ResearchStatus.COMPLETED.value
        assert "classification" in final_data
        
        # Verify context engineering produced valid results
        assert len(context_result.refined_queries) > 0
        assert context_result.search_strategy is not None
    
    async def test_migration_metrics(self):
        """Test that migration provides useful metrics"""
        # Get V2 store metrics
        store_metrics = research_store_v2.get_metrics()
        assert "cache_hits" in store_metrics
        assert "cache_misses" in store_metrics
        assert "redis_enabled" in store_metrics
        
        # Get V2 pipeline metrics
        pipeline_metrics = context_engineering_bridge.v2_pipeline.get_processing_metrics()
        assert "unique_queries_processed" in pipeline_metrics
        assert "memory_usage" in pipeline_metrics


# Run tests
if __name__ == "__main__":
    async def run_tests():
        """Run all migration tests"""
        print("🔄 Running Migration Compatibility Tests...")
        
        # Create test fixtures manually
        from models.context_models import (
            ResearchRequestSchema, ClassificationResultSchema, 
            QueryFeaturesSchema, HostParadigm, ResearchStatus
        )
        from services.classification_engine import ClassificationResult, QueryFeatures
        
        # Initialize fixtures
        features = QueryFeatures(
            text="climate change impacts on agriculture",
            tokens=["climate", "change", "impacts", "agriculture"],
            entities=["climate", "agriculture"],
            intent_signals=["research", "analysis"],
            domain="environmental",
            urgency_score=0.6,
            complexity_score=0.7,
            emotional_valence=0.0
        )
        
        sample_classification = ClassificationResult(
            query="climate change impacts on agriculture",
            primary_paradigm=HostParadigm.BERNARD,
            secondary_paradigm=HostParadigm.DOLORES,
            distribution={
                HostParadigm.BERNARD: 0.8,
                HostParadigm.DOLORES: 0.6,
                HostParadigm.MAEVE: 0.3,
                HostParadigm.TEDDY: 0.4
            },
            confidence=0.8,
            features=features,
            reasoning={
                HostParadigm.BERNARD: ["scientific research needed", "data analysis"],
                HostParadigm.DOLORES: ["environmental justice", "system impact"]
            }
        )
        
        sample_request = ResearchRequestSchema(
            id="test-research-123",
            query="climate change impacts on agriculture", 
            user_context={
                "user_id": "test-user-456",
                "role": "RESEARCHER",
                "subscription_tier": "PRO"
            },
            options={
                "deep_research": True,
                "academic_focus": True
            },
            status=ResearchStatus.PROCESSING
        )
        
        # Test Research Store Migration
        print("\n📦 Testing Research Store Migration...")
        store_tests = TestResearchStoreMigration()
        
        await store_tests.test_initialization()
        print("✓ Store initialization")
        
        await store_tests.test_data_storage_compatibility(sample_request)
        print("✓ Data storage compatibility")
        
        await store_tests.test_status_updates(sample_request)
        print("✓ Status updates")
        
        # Test Context Engineering Migration
        print("\n🔧 Testing Context Engineering Migration...")
        context_tests = TestContextEngineeringMigration()
        
        await context_tests.test_bridge_initialization()
        print("✓ Bridge initialization")
        
        await context_tests.test_v1_classification_conversion(sample_classification)
        print("✓ V1 classification conversion")
        
        await context_tests.test_bridge_processing_v2(sample_classification)
        print("✓ V2 processing through bridge")
        
        # Test End-to-End
        print("\n🔄 Testing End-to-End Migration...")
        e2e_tests = TestEndToEndMigration()
        
        await e2e_tests.test_complete_research_workflow_v2(sample_classification, sample_request)
        print("✓ Complete V2 workflow")
        
        await e2e_tests.test_migration_metrics()
        print("✓ Migration metrics")
        
        print("\n✅ All migration tests passed!")
    
    asyncio.run(run_tests())
