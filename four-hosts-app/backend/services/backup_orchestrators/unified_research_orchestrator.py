"""
Unified Research Orchestrator
Combines all research capabilities with Brave MCP as the primary search provider
"""

import logging
from typing import Dict, Any, Optional

from services.enhanced_research_orchestrator import enhanced_orchestrator
from services.research_orchestrator import research_orchestrator as legacy_orchestrator
from services.classification_engine import HostParadigm

logger = logging.getLogger(__name__)


class UnifiedResearchOrchestrator:
    """Unified interface for all research operations"""
    
    def __init__(self):
        self.enhanced = enhanced_orchestrator
        self.legacy = legacy_orchestrator
        self.use_enhanced = True
    
    async def initialize(self):
        """Initialize the unified orchestrator"""
        # Initialize enhanced orchestrator (includes Brave MCP)
        try:
            await self.enhanced.initialize()
            self.use_enhanced = True
            logger.info("✓ Unified orchestrator using enhanced mode with Brave MCP")
        except Exception as e:
            logger.warning(f"Enhanced orchestrator initialization failed: {e}")
            logger.info("Falling back to legacy orchestrator")
            self.use_enhanced = False
    
    async def execute_research(
        self,
        query: str,
        primary_paradigm: HostParadigm,
        secondary_paradigm: Optional[HostParadigm] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute research using the best available method"""
        
        if self.use_enhanced:
            try:
                # Use enhanced orchestrator with Brave MCP
                return await self.enhanced.execute_paradigm_research(
                    query=query,
                    primary_paradigm=primary_paradigm,
                    secondary_paradigm=secondary_paradigm,
                    options=options
                )
            except Exception as e:
                logger.error(f"Enhanced research failed: {e}")
                # Fall back to legacy if enhanced fails
                if hasattr(self.legacy, 'execute_research'):
                    logger.info("Falling back to legacy research")
                    return await self.legacy.execute_research(
                        query=query,
                        paradigm=primary_paradigm.value,
                        options=options
                    )
                raise
        else:
            # Use legacy orchestrator
            if hasattr(self.legacy, 'execute_research'):
                return await self.legacy.execute_research(
                    query=query,
                    paradigm=primary_paradigm.value,
                    options=options
                )
            else:
                raise NotImplementedError("No research orchestrator available")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get current orchestrator capabilities"""
        return {
            "mode": "enhanced" if self.use_enhanced else "legacy",
            "brave_mcp": self.enhanced.brave_enabled if self.use_enhanced else False,
            "search_apis": [
                "google",
                "arxiv",
                "pubmed",
                "brave_web" if self.use_enhanced and self.enhanced.brave_enabled else None,
                "brave_news" if self.use_enhanced and self.enhanced.brave_enabled else None,
                "brave_summarizer" if self.use_enhanced and self.enhanced.brave_enabled else None,
            ],
            "features": {
                "paradigm_optimization": True,
                "multi_source_synthesis": True,
                "mcp_tools": self.use_enhanced,
                "background_processing": True,
            }
        }


# Global instance
unified_orchestrator = UnifiedResearchOrchestrator()


# Backward compatibility exports
research_orchestrator = unified_orchestrator
execute_research = unified_orchestrator.execute_research


async def initialize_unified_orchestrator():
    """Initialize the unified research orchestrator"""
    await unified_orchestrator.initialize()
    return True