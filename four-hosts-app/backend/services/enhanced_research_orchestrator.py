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
            context_metadata={
                "enhanced": True, 
                "brave_enabled": self.brave_enabled
            }
        )

        # Execute with V2
        return await self.v2_orchestrator.execute_research(
            classification,
            context_engineered,
            user_context
        )


# Global instance
enhanced_orchestrator = EnhancedResearchOrchestratorV2()
