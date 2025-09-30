"""
Brave Search MCP Server Integration
Provides specialized search capabilities through Brave's MCP server
"""

import os
import structlog
from typing import Dict, List, Any, Optional
from enum import Enum

from services.mcp.mcp_integration import MCPServer, MCPCapability, mcp_integration

from logging_config import configure_logging

configure_logging()
logger = structlog.get_logger(__name__)


class BraveSearchType(str, Enum):
    """Available Brave search types"""
    WEB = "web"
    LOCAL = "local"
    VIDEO = "video"
    IMAGE = "image"
    NEWS = "news"
    SUMMARIZER = "summarizer"


class BraveMCPConfig:
    """Configuration for Brave Search MCP server"""
    
    def __init__(self):
        # Get Brave API key from environment
        self.api_key = os.getenv("BRAVE_API_KEY") or os.getenv("BRAVE_SEARCH_API_KEY")
        self.mcp_url = os.getenv("BRAVE_MCP_URL", "http://localhost:8080/mcp")
        self.mcp_transport = os.getenv("BRAVE_MCP_TRANSPORT", "HTTP")
        self.mcp_host = os.getenv("BRAVE_MCP_HOST", "localhost")
        self.mcp_port = int(os.getenv("BRAVE_MCP_PORT", "8080"))
        
        # Search configuration
        self.default_country = os.getenv("BRAVE_DEFAULT_COUNTRY", "US")
        self.default_language = os.getenv("BRAVE_DEFAULT_LANGUAGE", "en")
        self.safe_search = os.getenv("BRAVE_SAFE_SEARCH", "moderate")
        
    def is_configured(self) -> bool:
        """Check if Brave MCP is properly configured"""
        return bool(self.api_key) and self.api_key not in ["", "your_api_key_here"]


class BraveMCPIntegration:
    """Manages Brave Search MCP server integration"""
    
    def __init__(self, config: BraveMCPConfig):
        self.config = config
        self.server_registered = False
        
    async def initialize(self) -> bool:
        """Initialize Brave MCP server connection"""
        if not self.config.is_configured():
            logger.warning("Brave API key not configured - skipping MCP server registration")
            return False
        
        try:
            # Register Brave MCP server
            brave_server = MCPServer(
                name="brave_search",
                url=self.config.mcp_url,
                capabilities=[MCPCapability.SEARCH],
                auth_token=self.config.api_key
            )
            
            mcp_integration.register_server(brave_server)
            
            # Try to discover available tools
            try:
                tools = await mcp_integration.discover_tools("brave_search")
                logger.info(f"Discovered {len(tools)} Brave search tools")
                self.server_registered = True
                return True
            except Exception as discover_error:
                logger.warning(f"Could not discover Brave MCP tools (server may not be running): {discover_error}")
                # Still mark as registered since we can use direct Brave API
                self.server_registered = False
                return False
            
        except Exception as e:
            logger.error(f"Failed to initialize Brave MCP server: {e}")
            return False
    
    def get_paradigm_search_config(self, paradigm: str) -> Dict[str, Any]:
        """Get search configuration optimized for each paradigm"""
        
        # Base configuration
        config = {
            "country": self.config.default_country,
            "language": self.config.default_language,
            "safesearch": self.config.safe_search,
        }
        
        # Paradigm-specific optimizations
        paradigm_configs = {
            "dolores": {
                # Revolutionary - focus on uncovering truth
                "freshness": "recent",  # Recent events and exposés
                "search_types": [BraveSearchType.WEB, BraveSearchType.NEWS],
                "extra_params": {
                    "include_controversial": True,
                    "prioritize_independent_sources": True
                }
            },
            "teddy": {
                # Devotion - focus on helping and community
                "search_types": [BraveSearchType.WEB, BraveSearchType.LOCAL],
                "safesearch": "strict",  # Protect vulnerable users
                "extra_params": {
                    "prioritize_official_sources": True,
                    "include_community_resources": True
                }
            },
            "bernard": {
                # Analytical - focus on data and research
                "search_types": [BraveSearchType.WEB, BraveSearchType.SUMMARIZER],
                "extra_params": {
                    "prioritize_academic_sources": True,
                    "include_statistics": True,
                    "summarizer_enabled": True
                }
            },
            "maeve": {
                # Strategic - focus on business and competition
                "search_types": [BraveSearchType.WEB, BraveSearchType.NEWS],
                "freshness": "recent",
                "extra_params": {
                    "prioritize_business_sources": True,
                    "include_market_data": True
                }
            }
        }
        
        # Merge paradigm-specific config
        if paradigm.lower() in paradigm_configs:
            config.update(paradigm_configs[paradigm.lower()])
        
        return config
    
    async def search_with_paradigm(
        self,
        query: str,
        paradigm: str,
        search_type: BraveSearchType = BraveSearchType.WEB
    ) -> Dict[str, Any]:
        """Execute a paradigm-aware search using Brave MCP"""
        
        if not self.server_registered:
            raise RuntimeError("Brave MCP server not initialized")
        
        # Get paradigm-specific configuration
        search_config = self.get_paradigm_search_config(paradigm)
        
        # Build tool name based on search type
        tool_name = f"brave_search_brave_{search_type}_search"
        
        # Prepare search parameters
        search_params = {
            "query": query,
            "country": search_config.get("country"),
            "language": search_config.get("language"),
            "safesearch": search_config.get("safesearch"),
        }
        
        # Add freshness if specified
        if "freshness" in search_config:
            search_params["freshness"] = search_config["freshness"]
        
        try:
            # Execute search through MCP
            result = await mcp_integration.execute_tool_call(
                tool_name,
                search_params
            )
            
            # Post-process results based on paradigm
            processed_result = self._process_results_for_paradigm(
                result,
                paradigm,
                search_config.get("extra_params", {})
            )
            
            return processed_result
            
        except Exception as e:
            logger.error(f"Brave search error for {paradigm}: {e}")
            raise
    
    def _process_results_for_paradigm(
        self,
        raw_results: Any,
        paradigm: str,
        extra_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process search results according to paradigm preferences"""
        
        # Extract results based on expected format
        if isinstance(raw_results, dict):
            results = raw_results.get("results", [])
        else:
            results = []
        
        # Paradigm-specific filtering and ranking
        if paradigm.lower() == "dolores":
            # Prioritize results that expose issues
            if extra_params.get("prioritize_independent_sources"):
                results = self._prioritize_independent_sources(results)
                
        elif paradigm.lower() == "teddy":
            # Filter out potentially harmful content
            if extra_params.get("include_community_resources"):
                results = self._highlight_community_resources(results)
                
        elif paradigm.lower() == "bernard":
            # Prioritize academic and data-rich sources
            if extra_params.get("prioritize_academic_sources"):
                results = self._prioritize_academic_sources(results)
                
        elif paradigm.lower() == "maeve":
            # Focus on business and strategic insights
            if extra_params.get("prioritize_business_sources"):
                results = self._prioritize_business_sources(results)
        
        return {
            "paradigm": paradigm,
            "results": results,
            "result_count": len(results),
            "search_params": extra_params
        }
    
    def _prioritize_independent_sources(self, results: List[Any]) -> List[Any]:
        """Prioritize independent and alternative media sources"""
        # Implementation would analyze domain authority and independence
        return results
    
    def _highlight_community_resources(self, results: List[Any]) -> List[Any]:
        """Highlight community resources and support organizations"""
        # Implementation would identify .org, .gov, and community sites
        return results
    
    def _prioritize_academic_sources(self, results: List[Any]) -> List[Any]:
        """Prioritize academic and research sources"""
        # Implementation would identify .edu domains and research papers
        return results
    
    def _prioritize_business_sources(self, results: List[Any]) -> List[Any]:
        """Prioritize business and market analysis sources"""
        # Implementation would identify business publications and analysis
        return results


# Global instance
brave_config = BraveMCPConfig()
brave_mcp = BraveMCPIntegration(brave_config)


async def initialize_brave_mcp():
    """Initialize Brave MCP integration during startup"""
    if await brave_mcp.initialize():
        logger.info("✓ Brave Search MCP server initialized")
        return True
    else:
        logger.warning("Brave Search MCP server not initialized")
        return False
