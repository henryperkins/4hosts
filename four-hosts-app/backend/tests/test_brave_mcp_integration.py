"""
Test Brave MCP Integration with Four Hosts Paradigms
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock

from services.mcp.brave_mcp_integration import (
    BraveMCPConfig,
    BraveMCPIntegration,
    BraveSearchType,
    brave_mcp
)
from services.classification_engine import HostParadigm
from services.enhanced_research_orchestrator import EnhancedResearchOrchestrator


class TestBraveMCPConfig:
    """Test Brave MCP configuration"""
    
    def test_config_default_values(self):
        """Test default configuration values"""
        config = BraveMCPConfig()
        assert config.mcp_url == "http://localhost:8080/mcp"
        assert config.mcp_transport == "HTTP"
        assert config.mcp_host == "localhost"
        assert config.mcp_port == 8080
        assert config.default_country == "US"
        assert config.default_language == "en"
        assert config.safe_search == "moderate"
    
    @patch.dict('os.environ', {'BRAVE_API_KEY': 'test_api_key'})
    def test_config_is_configured(self):
        """Test configuration validation"""
        config = BraveMCPConfig()
        assert config.is_configured() is True
        assert config.api_key == 'test_api_key'
    
    @patch.dict('os.environ', {'BRAVE_API_KEY': ''})
    def test_config_not_configured(self):
        """Test configuration validation with empty key"""
        config = BraveMCPConfig()
        assert config.is_configured() is False


class TestBraveMCPIntegration:
    """Test Brave MCP integration functionality"""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
        config = Mock(spec=BraveMCPConfig)
        config.is_configured.return_value = True
        config.api_key = "test_key"
        config.mcp_url = "http://localhost:8080/mcp"
        config.default_country = "US"
        config.default_language = "en"
        config.safe_search = "moderate"
        return config
    
    @pytest.fixture
    def brave_integration(self, mock_config):
        """Create Brave MCP integration instance"""
        return BraveMCPIntegration(mock_config)
    
    def test_paradigm_search_config_dolores(self, brave_integration):
        """Test Dolores paradigm search configuration"""
        config = brave_integration.get_paradigm_search_config("dolores")
        
        assert config["freshness"] == "recent"
        assert BraveSearchType.WEB in config["search_types"]
        assert BraveSearchType.NEWS in config["search_types"]
        assert config["extra_params"]["include_controversial"] is True
        assert config["extra_params"]["prioritize_independent_sources"] is True
    
    def test_paradigm_search_config_teddy(self, brave_integration):
        """Test Teddy paradigm search configuration"""
        config = brave_integration.get_paradigm_search_config("teddy")
        
        assert config["safesearch"] == "strict"
        assert BraveSearchType.WEB in config["search_types"]
        assert BraveSearchType.LOCAL in config["search_types"]
        assert config["extra_params"]["prioritize_official_sources"] is True
        assert config["extra_params"]["include_community_resources"] is True
    
    def test_paradigm_search_config_bernard(self, brave_integration):
        """Test Bernard paradigm search configuration"""
        config = brave_integration.get_paradigm_search_config("bernard")
        
        assert BraveSearchType.WEB in config["search_types"]
        assert BraveSearchType.SUMMARIZER in config["search_types"]
        assert config["extra_params"]["prioritize_academic_sources"] is True
        assert config["extra_params"]["summarizer_enabled"] is True
    
    def test_paradigm_search_config_maeve(self, brave_integration):
        """Test Maeve paradigm search configuration"""
        config = brave_integration.get_paradigm_search_config("maeve")
        
        assert config["freshness"] == "recent"
        assert BraveSearchType.WEB in config["search_types"]
        assert BraveSearchType.NEWS in config["search_types"]
        assert config["extra_params"]["prioritize_business_sources"] is True
        assert config["extra_params"]["include_market_data"] is True
    
    @pytest.mark.asyncio
    async def test_search_with_paradigm_not_initialized(self, brave_integration):
        """Test search fails when not initialized"""
        brave_integration.server_registered = False
        
        with pytest.raises(RuntimeError, match="Brave MCP server not initialized"):
            await brave_integration.search_with_paradigm(
                "test query",
                "dolores",
                BraveSearchType.WEB
            )
    
    def test_process_results_for_paradigm(self, brave_integration):
        """Test result processing for different paradigms"""
        raw_results = {
            "results": [
                {"title": "Test 1", "url": "http://example1.com"},
                {"title": "Test 2", "url": "http://example2.com"}
            ]
        }
        
        # Test Dolores processing
        processed = brave_integration._process_results_for_paradigm(
            raw_results,
            "dolores",
            {"prioritize_independent_sources": True}
        )
        
        assert processed["paradigm"] == "dolores"
        assert processed["result_count"] == 2
        assert "prioritize_independent_sources" in processed["search_params"]


class TestEnhancedResearchOrchestrator:
    """Test enhanced research orchestrator with Brave MCP"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance"""
        return EnhancedResearchOrchestrator()
    
    def test_enhance_query_for_paradigm(self, orchestrator):
        """Test query enhancement for different paradigms"""
        # Test Dolores
        enhanced = orchestrator._enhance_query_for_paradigm(
            "corporate misconduct",
            HostParadigm.DOLORES
        )
        assert "expose" in enhanced.lower()
        
        # Test Teddy
        enhanced = orchestrator._enhance_query_for_paradigm(
            "elderly care",
            HostParadigm.TEDDY
        )
        assert "help" in enhanced.lower()
        
        # Test Bernard
        enhanced = orchestrator._enhance_query_for_paradigm(
            "climate change",
            HostParadigm.BERNARD
        )
        assert "research" in enhanced.lower()
        
        # Test Maeve
        enhanced = orchestrator._enhance_query_for_paradigm(
            "market analysis",
            HostParadigm.MAEVE
        )
        assert "strategy" in enhanced.lower()
    
    def test_calculate_paradigm_alignment(self, orchestrator):
        """Test paradigm alignment calculation"""
        # High alignment text for Dolores
        text = "We must expose the corruption and reveal the truth about injustice"
        alignment = orchestrator._calculate_paradigm_alignment(
            text,
            HostParadigm.DOLORES
        )
        assert alignment > 0.5
        
        # Low alignment text for Bernard
        text = "Just my personal opinion without any evidence"
        alignment = orchestrator._calculate_paradigm_alignment(
            text,
            HostParadigm.BERNARD
        )
        assert alignment < 0.5
    
    def test_extract_insights(self, orchestrator):
        """Test insight extraction from synthesis"""
        synthesis = """
        Here are the key findings:
        • First important insight about the topic
        • Second key discovery from the research
        - Third notable pattern identified
        1. Fourth numbered insight
        2) Fifth insight with different formatting
        
        Some regular text that shouldn't be extracted.
        """
        
        insights = orchestrator._extract_insights(synthesis, HostParadigm.BERNARD)
        
        assert len(insights) == 5
        assert "First important insight" in insights[0]
        assert "Fifth insight" in insights[4]


@pytest.mark.integration
class TestBraveMCPIntegrationE2E:
    """End-to-end integration tests (requires Brave MCP server running)"""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not BraveMCPConfig().is_configured(),
        reason="Brave API key not configured"
    )
    async def test_initialize_brave_mcp(self):
        """Test actual initialization of Brave MCP"""
        from services.mcp.brave_mcp_integration import initialize_brave_mcp
        
        result = await initialize_brave_mcp()
        assert result is True
        assert brave_mcp.server_registered is True
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not BraveMCPConfig().is_configured(),
        reason="Brave API key not configured"
    )
    async def test_paradigm_search_e2e(self):
        """Test actual paradigm search (requires server)"""
        await brave_mcp.initialize()
        
        # Test Dolores search
        result = await brave_mcp.search_with_paradigm(
            "corporate whistleblower protection",
            "dolores",
            BraveSearchType.WEB
        )
        
        assert result["paradigm"] == "dolores"
        assert "results" in result
        assert result["result_count"] > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])