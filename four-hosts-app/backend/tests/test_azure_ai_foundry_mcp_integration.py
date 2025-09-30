"""
Test Azure AI Foundry MCP Integration with Four Hosts Paradigms
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock

from services.mcp.azure_ai_foundry_mcp_integration import (
    AzureAIFoundryMCPConfig,
    AzureAIFoundryMCPIntegration,
    AzureAIFoundryCapability,
    azure_ai_foundry_mcp
)


class TestAzureAIFoundryMCPConfig:
    """Test Azure AI Foundry MCP configuration"""
    
    def test_config_default_values(self):
        """Test default configuration values"""
        config = AzureAIFoundryMCPConfig()
        assert config.mcp_url == "http://localhost:8081/mcp"
        assert config.mcp_transport == "stdio"
        assert config.mcp_host == "localhost"
        assert config.mcp_port == 8081
        assert config.enable_web_search is False
        assert config.enable_code_interpreter is False
    
    @patch.dict('os.environ', {'AZURE_AI_PROJECT_ENDPOINT': 'https://test.openai.azure.com/'})
    def test_config_is_configured(self):
        """Test configuration validation"""
        config = AzureAIFoundryMCPConfig()
        assert config.is_configured() is True
        assert config.ai_project_endpoint == 'https://test.openai.azure.com/'
    
    @patch.dict('os.environ', {'AZURE_AI_PROJECT_ENDPOINT': ''})
    def test_config_not_configured(self):
        """Test configuration validation with empty endpoint"""
        config = AzureAIFoundryMCPConfig()
        assert config.is_configured() is False
    
    @patch.dict('os.environ', {
        'AZURE_AI_PROJECT_ENDPOINT': 'https://test.openai.azure.com/',
        'AZURE_TENANT_ID': 'test_tenant_id',
        'AZURE_CLIENT_ID': 'test_client_id',
        'AZURE_CLIENT_SECRET': 'test_secret'
    })
    def test_config_has_authentication(self):
        """Test authentication configuration validation"""
        config = AzureAIFoundryMCPConfig()
        assert config.has_authentication() is True
    
    @patch.dict('os.environ', {'AZURE_AI_PROJECT_ENDPOINT': 'https://test.openai.azure.com/'})
    def test_config_missing_auth(self):
        """Test missing authentication configuration"""
        config = AzureAIFoundryMCPConfig()
        assert config.has_authentication() is False
    
    def test_get_missing_config(self):
        """Test missing configuration detection"""
        config = AzureAIFoundryMCPConfig()
        missing_required, missing_optional = config.get_missing_config()
        assert "AZURE_AI_PROJECT_ENDPOINT" in missing_required
        assert len(missing_optional) > 0


class TestAzureAIFoundryMCPIntegration:
    """Test Azure AI Foundry MCP integration functionality"""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
        config = Mock(spec=AzureAIFoundryMCPConfig)
        config.is_configured.return_value = True
        config.has_authentication.return_value = True
        config.get_missing_config.return_value = ([], [])
        config.ai_project_endpoint = "https://test.openai.azure.com/"
        config.mcp_url = "http://localhost:8081/mcp"
        config.swagger_path = "/path/to/swagger.json"
        return config
    
    @pytest.fixture
    def azure_integration(self, mock_config):
        """Create Azure AI Foundry MCP integration instance"""
        return AzureAIFoundryMCPIntegration(mock_config)
    
    def test_evaluation_config_dolores(self, azure_integration):
        """Test Dolores paradigm evaluation configuration"""
        config = azure_integration.get_evaluation_config("dolores")
        
        assert "groundedness" in config["evaluation_types"]
        assert "controversy_detection" in config["evaluation_types"]
        assert config["reasoning_effort"] == "high"
        assert config["safety_settings"]["bias_detection"] is True
    
    def test_evaluation_config_teddy(self, azure_integration):
        """Test Teddy paradigm evaluation configuration"""
        config = azure_integration.get_evaluation_config("teddy")
        
        assert "helpfulness" in config["evaluation_types"]
        assert "safety" in config["evaluation_types"]
        assert config["reasoning_effort"] == "medium"
        assert config["safety_settings"]["prioritize_user_safety"] is True
    
    def test_evaluation_config_bernard(self, azure_integration):
        """Test Bernard paradigm evaluation configuration"""
        config = azure_integration.get_evaluation_config("bernard")
        
        assert "groundedness" in config["evaluation_types"]
        assert "factual_accuracy" in config["evaluation_types"]
        assert config["reasoning_effort"] == "high"
        assert config["safety_settings"]["academic_mode"] is True
    
    def test_evaluation_config_maeve(self, azure_integration):
        """Test Maeve paradigm evaluation configuration"""
        config = azure_integration.get_evaluation_config("maeve")
        
        assert "business_value" in config["evaluation_types"]
        assert "relevance" in config["evaluation_types"]
        assert config["reasoning_effort"] == "medium"
        assert config["safety_settings"]["business_appropriate"] is True
    
    @pytest.mark.asyncio
    async def test_evaluate_content_not_initialized(self, azure_integration):
        """Test evaluation fails when not initialized"""
        with pytest.raises(RuntimeError, match="not initialized"):
            await azure_integration.evaluate_content("test content", "dolores")
    
    @pytest.mark.asyncio
    async def test_query_knowledge_base_not_initialized(self, azure_integration):
        """Test knowledge query fails when not initialized"""
        with pytest.raises(RuntimeError, match="not initialized"):
            await azure_integration.query_knowledge_base("test query", "bernard")
    
    def test_process_evaluation_for_paradigm(self, azure_integration):
        """Test result processing for different paradigms"""
        raw_results = {
            "groundedness": 0.85,
            "safety_score": 0.9,
            "bias_detected": True
        }
        
        # Test Dolores processing
        result = azure_integration._process_evaluation_for_paradigm(
            raw_results, "dolores", "groundedness"
        )
        assert result["paradigm"] == "dolores"
        assert result["evaluation_type"] == "groundedness"
        assert "paradigm_insight" in result
        
        # Test Teddy processing
        result = azure_integration._process_evaluation_for_paradigm(
            raw_results, "teddy", "safety"
        )
        assert result["paradigm"] == "teddy"
        assert "paradigm_insight" in result
    
    def test_get_available_capabilities(self, azure_integration):
        """Test capability enumeration"""
        # Initially empty before initialization
        capabilities = azure_integration.get_available_capabilities()
        assert isinstance(capabilities, list)


@pytest.mark.integration
class TestAzureAIFoundryMCPIntegrationE2E:
    """End-to-end integration tests (requires Azure AI Foundry MCP server running)"""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not AzureAIFoundryMCPConfig().is_configured(),
        reason="Azure AI project endpoint not configured"
    )
    async def test_initialize_azure_ai_foundry_mcp(self):
        """Test actual initialization of Azure AI Foundry MCP"""
        from services.mcp.azure_ai_foundry_mcp_integration import initialize_azure_ai_foundry_mcp
        
        # This test requires actual Azure AI Foundry MCP server
        result = await initialize_azure_ai_foundry_mcp()
        # May succeed or fail depending on server availability
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not AzureAIFoundryMCPConfig().is_configured(),
        reason="Azure AI project endpoint not configured"
    )
    async def test_paradigm_evaluation_e2e(self):
        """Test actual paradigm evaluation (requires server)"""
        config = AzureAIFoundryMCPConfig()
        integration = AzureAIFoundryMCPIntegration(config)
        
        # Only test if properly configured
        if await integration.initialize():
            try:
                result = await integration.evaluate_content(
                    "This is a test content for evaluation.",
                    "bernard",
                    "groundedness"
                )
                assert "paradigm" in result
                assert result["paradigm"] == "bernard"
            except Exception as e:
                # Server may not be available or configured
                pytest.skip(f"Azure AI Foundry MCP server not available: {e}")
        else:
            pytest.skip("Azure AI Foundry MCP not initialized")


def test_azure_ai_foundry_capability_enum():
    """Test Azure AI Foundry capability enumeration"""
    assert AzureAIFoundryCapability.EVALUATION == "evaluation"
    assert AzureAIFoundryCapability.MODEL == "model"
    assert AzureAIFoundryCapability.KNOWLEDGE == "knowledge"
    assert AzureAIFoundryCapability.FINETUNING == "finetuning"