"""
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
execute_research = unified_orchestrator.execute_research


async def initialize_unified_orchestrator():
    """Initialize the unified research orchestrator"""
    await unified_orchestrator.initialize()
    return True
