Here’s a cleaner, consistently formatted version of [[Untitled 1]].

Research Orchestrator Replacement Guide

Overview
You have 4 existing orchestrator files that need to be replaced/updated with the new research_orchestrator_v2.py:
1) research_orchestrator.py — Original implementation (950 lines)
2) enhanced_research_orchestrator.py — Enhanced version with Brave MCP (403 lines)
3) unified_research_orchestrator.py — Unified interface (110 lines)
4) research_orchestrator_v2.py — New implementation with all fixes (413 lines)

Key Differences in V2
- Deterministic result merging: Results are sorted consistently regardless of async execution order
- Origin tracking: Each result tracks which query and API produced it
- Dynamic query optimization: Replaces hard-coded [:8] limit with paradigm-aware optimization
- Full context preservation: Passes complete context through the pipeline
- Proper deduplication: Groups by URL and selects best content
- User context utilization: Respects user tier and preferences

Step-by-Step Migration

Step 1: Backup Existing Files
cd /home/azureuser/4hosts/four-hosts-app/backend/services
mkdir backup_orchestrators
cp research_orchestrator.py backup_orchestrators/
cp enhanced_research_orchestrator.py backup_orchestrators/
cp unified_research_orchestrator.py backup_orchestrators/

Step 2: Update Imports in V2
Add these imports to the top of research_orchestrator_v2.py:
from services.search_apis import SearchAPIManager, SearchResult, SearchConfig
from services.credibility import CredibilityChecker, get_source_credibility
from services.paradigm_search import get_search_strategy, SearchContext
from services.cache import cache_manager, get_cached_search_results, cache_search_results

Step 3: Create Compatibility Wrapper
Create services/research_orchestrator_compat.py:
```python
"""
Compatibility wrapper for gradual migration to V2
"""
import logging
from typing import Dict, Any, Optional

from services.research_orchestrator_v2 import research_orchestrator_v2
from services.classification_engine import HostParadigm
from models.context_models import (
    ClassificationResultSchema, ContextEngineeredQuerySchema, UserContextSchema
)

logger = logging.getLogger(__name__)


class ResearchOrchestratorCompat:
    """Provides backward compatibility for existing code"""

    def __init__(self):
        self.v2_orchestrator = research_orchestrator_v2

    async def execute_paradigm_research(
        self,
        context_engineered_query,
        max_results: int = 100,
        progress_tracker=None,
        research_id: str = None
    ):
        """Legacy method signature compatibility"""

        # Convert old context_engineered_query to V2 schemas
        classification = ClassificationResultSchema(
            query=context_engineered_query.original_query,
            primary_paradigm=HostParadigm(context_engineered_query.classification.primary_paradigm.value),
            secondary_paradigm=HostParadigm(context_engineered_query.classification.secondary_paradigm.value)
                if context_engineered_query.classification.secondary_paradigm else None,
            distribution={},  # Will be populated from classification
            confidence=0.8,
            features=None,  # Will need to be populated
            reasoning={}
        )

        # Create minimal user context for backward compatibility
        user_context = UserContextSchema(
            user_id="legacy_user",
            role="BASIC",
            max_sources=max_results
        )

        # Create context engineered schema
        context_engineered = ContextEngineeredQuerySchema(
            original_query=context_engineered_query.original_query,
            classification=classification,
            write_layer_output={},
            select_layer_output={
                "search_queries": context_engineered_query.select_output.search_queries
            },
            compress_layer_output={},
            isolate_layer_output={},
            refined_queries=[q["query"] for q in context_engineered_query.select_output.search_queries],
            search_strategy={},
            context_metadata={}
        )

        # Execute V2 research
        result = await self.v2_orchestrator.execute_research(
            classification,
            context_engineered,
            user_context,
            progress_callback=progress_tracker.update_progress if progress_tracker else None
        )

        # Convert V2 result to legacy format
        return self._convert_to_legacy_result(result, context_engineered_query)

    def _convert_to_legacy_result(self, v2_result: Dict[str, Any], original_query):
        """Convert V2 result to legacy ResearchExecutionResult format"""
        from services.research_orchestrator import ResearchExecutionResult

        return ResearchExecutionResult(
            original_query=original_query.original_query,
            paradigm=v2_result["metadata"]["paradigm"] if "paradigm" in v2_result.get("metadata", {}) else "unknown",
            secondary_paradigm=None,
            search_queries_executed=[],
            raw_results={},
            filtered_results=v2_result["results"],
            credibility_scores={},
            execution_metrics=v2_result["metadata"],
            cost_breakdown={},
            secondary_results=[]
        )


# Create compatibility instance
research_orchestrator_compat = ResearchOrchestratorCompat()
```

Step 4: Update main.py Integration Points
Replace imports:
```python
# Old
from services.research_orchestrator import research_orchestrator, initialize_research_system
# New
from services.research_orchestrator_v2 import research_orchestrator_v2
from services.research_orchestrator_compat import research_orchestrator_compat as research_orchestrator
from models.context_models import UserContextSchema, ContextEngineeredQuerySchema

Update the background task function:

async def execute_real_research(
    research_id: str,
    research: ResearchRequest,
    user_id: str,
):
    """Execute research with V2 orchestrator"""
    try:
        # Get stored classification
        research_data = await research_store.get(research_id)
        classification = ClassificationResultSchema.from_redis_dict(
            research_data["classification"]
        )

        # Create user context
        user = await get_user_by_id(user_id)
        user_context = UserContextSchema(
            user_id=user_id,
            role=user.role.value,
            preferences=user.preferences or {},
            location=user.location,
            language=user.language or "en",
            max_sources=research.options.max_results or 20
        )

        # Get context engineered query
        context_engineered = await context_pipeline_v2.process_query(
            classification,
            include_debug=user_context.is_pro_user
        )

        # Execute V2 research
        search_results = await research_orchestrator_v2.execute_research(
            classification,
            context_engineered,
            user_context,
            progress_callback=lambda msg: progress_tracker.update_progress(
                research_id, msg, 50
            )
        )

        # Rest of the processing...
```
Step 5: Update Enhanced Orchestrator
Replace enhanced_research_orchestrator.py with a V2-compatible version:
```python
"""
Enhanced Research Orchestrator V2 Wrapper
Maintains compatibility while using V2 internally
"""
import logging
from typing import Dict, Any, Optional

from services.research_orchestrator_v2 import research_orchestrator_v2
from services.brave_mcp_integration import brave_mcp, BraveSearchType
from models.context_models import (
    ClassificationResultSchema, UserContextSchema,
    ContextEngineeredQuerySchema, HostParadigm
)

logger = logging.getLogger(__name__)


class EnhancedResearchOrchestratorV2:
    """Enhanced orchestrator using V2 implementation"""

    def __init__(self):
        self.v2_orchestrator = research_orchestrator_v2
        self.brave_enabled = False

    async def initialize(self):
        """Initialize enhanced orchestrator"""
        try:
            from services.brave_mcp_integration import initialize_brave_mcp
            self.brave_enabled = await initialize_brave_mcp()
            logger.info(f"Brave MCP enabled: {self.brave_enabled}")
        except Exception as e:
            logger.warning(f"Brave MCP initialization failed: {e}")

    async def execute_paradigm_research(
        self,
        query: str,
        primary_paradigm: HostParadigm,
        secondary_paradigm: Optional[HostParadigm] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute research with enhanced features"""

        # Create classification
        classification = ClassificationResultSchema(
            query=query,
            primary_paradigm=primary_paradigm,
            secondary_paradigm=secondary_paradigm,
            distribution={primary_paradigm: 0.8},
            confidence=0.8,
            features=None,  # Would be populated by classification engine
            reasoning={primary_paradigm: ["Direct paradigm selection"]}
        )

        # Create user context from options
        user_context = UserContextSchema(
            user_id=options.get("user_id", "anonymous"),
            role=options.get("user_role", "BASIC"),
            preferences=options.get("preferences", {}),
            max_sources=options.get("max_results", 20)
        )

        # Simple context engineering
        context_engineered = ContextEngineeredQuerySchema(
            original_query=query,
            classification=classification,
            write_layer_output={},
            select_layer_output={},
            compress_layer_output={},
            isolate_layer_output={},
            refined_queries=[query],  # Simple pass-through
            search_strategy={"paradigm": primary_paradigm.value},
            context_metadata={"enhanced": True, "brave_enabled": self.brave_enabled}
        )

        # Execute with V2
        return await self.v2_orchestrator.execute_research(
            classification,
            context_engineered,
            user_context
        )


# Global instance
enhanced_orchestrator = EnhancedResearchOrchestratorV2()
```

Step 6: Update Unified Orchestrator
Update unified_research_orchestrator.py to use V2:
```python
""""
Unified Research Orchestrator V2
Single interface for all research operations
"""
import logging
from typing import Dict, Any, Optional

from services.research_orchestrator_v2 import research_orchestrator_v2
from services.enhanced_research_orchestrator import enhanced_orchestrator
from services.classification_engine import HostParadigm
from models.context_models import (
    ClassificationResultSchema, UserContextSchema,
    ContextEngineeredQuerySchema
)

logger = logging.getLogger(__name__)


class UnifiedResearchOrchestratorV2:
    """Unified interface using V2 implementation"""

    def __init__(self):
        self.orchestrator = research_orchestrator_v2
        self.enhanced = enhanced_orchestrator

    async def initialize(self):
        """Initialize unified orchestrator"""
        await self.enhanced.initialize()
        logger.info("✓ Unified orchestrator V2 initialized")

    async def execute_research(
        self,
        query: str,
        primary_paradigm: HostParadigm,
        secondary_paradigm: Optional[HostParadigm] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute research using V2 implementation"""

        # Use enhanced orchestrator which wraps V2
        return await self.enhanced.execute_paradigm_research(
            query=query,
            primary_paradigm=primary_paradigm,
            secondary_paradigm=secondary_paradigm,
            options=options
        )

    def get_capabilities(self) -> Dict[str, Any]:
        """Get current orchestrator capabilities"""
        return {
            "version": "2.0",
            "features": {
                "deterministic_results": True,
                "origin_tracking": True,
                "dynamic_compression": True,
                "full_context_preservation": True,
                "user_context_aware": True,
                "paradigm_optimization": True,
                "brave_mcp": self.enhanced.brave_enabled
            }
        }


# Global instance
unified_orchestrator = UnifiedResearchOrchestratorV2()

# Backward compatibility
research_orchestrator = unified_orchestrator
```
Step 7: Testing the Migration
Create test_orchestrator_migration.py:
```python
import asyncio
from services.research_orchestrator_v2 import research_orchestrator_v2
from services.research_orchestrator_compat import research_orchestrator_compat
from models.context_models import (
    ClassificationResultSchema, UserContextSchema,
    ContextEngineeredQuerySchema, HostParadigm
)

async def test_v2_orchestrator():
    """Test V2 orchestrator directly"""

    # Create test data
    classification = ClassificationResultSchema(
        query="How can small businesses compete with Amazon?",
        primary_paradigm=HostParadigm.MAEVE,
        secondary_paradigm=None,
        distribution={HostParadigm.MAEVE: 0.8, HostParadigm.DOLORES: 0.2},
        confidence=0.85,
        features=None,
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
```
Important Notes
1) Gradual Migration: The compatibility wrapper allows existing code to work while you migrate
2) API Changes: V2 uses Pydantic schemas instead of dataclasses/SimpleNamespace
3) Context Required: V2 requires full context (classification, user context, context engineering)
4) Progress Tracking: V2 uses a simple callback instead of the full progress tracker object
5) Result Format: V2 returns a simplified dict structure focused on results and metadata

Rollback Plan
If issues occur:
# Restore original files
`cd /home/azureuser/4hosts/four-hosts-app/backend/services`
`cp backup_orchestrators/*.py` .
# Restart the application

The V2 implementation provides significant improvements while maintaining the ability to work with existing code through the compatibility layer.
