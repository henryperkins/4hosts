"""
Compatibility wrapper for gradual migration to V2
"""
import logging
from typing import Dict, Any, Optional

from services.research_orchestrator_v2 import research_orchestrator_v2
from services.classification_engine import HostParadigm
from models.context_models import (
    ClassificationResultSchema, ContextEngineeredQuerySchema, UserContextSchema,
    QueryFeaturesSchema
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
        # Create minimal features for compatibility
        features = QueryFeaturesSchema(
            text=context_engineered_query.original_query,
            tokens=context_engineered_query.original_query.split(),
            entities=[],
            intent_signals=[],
            domain=None,
            urgency_score=0.5,
            complexity_score=0.5,
            emotional_valence=0.5
        )

        primary_paradigm = HostParadigm(
            context_engineered_query.classification.primary_paradigm.value
        )
        secondary_paradigm = None
        if context_engineered_query.classification.secondary_paradigm:
            secondary_paradigm = HostParadigm(
                context_engineered_query.classification.secondary_paradigm.value
            )

        classification = ClassificationResultSchema(
            query=context_engineered_query.original_query,
            primary_paradigm=primary_paradigm,
            secondary_paradigm=secondary_paradigm,
            distribution={primary_paradigm: 0.8},
            confidence=0.8,
            features=features,
            reasoning={primary_paradigm: ["Legacy compatibility"]}
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
