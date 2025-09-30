"""
Azure AI Foundry MCP Server Integration
Provides AI evaluation and model capabilities through Azure AI Foundry's MCP server
"""

import os
import structlog
from typing import Dict, List, Any, Optional
from enum import Enum

from services.mcp_integration import MCPServer, MCPCapability, mcp_integration

from logging_config import configure_logging

configure_logging()
logger = structlog.get_logger(__name__)


class AzureAIFoundryCapability(str, Enum):
    """Available Azure AI Foundry capabilities"""
    EVALUATION = "evaluation"
    MODEL = "model"
    KNOWLEDGE = "knowledge"
    FINETUNING = "finetuning"


class AzureAIFoundryMCPConfig:
    """Configuration for Azure AI Foundry MCP server"""
    
    def __init__(self):
        # Azure AI project configuration
        self.ai_project_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
        self.subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        self.resource_group = os.getenv("AZURE_RESOURCE_GROUP_NAME")
        self.project_name = os.getenv("AZURE_AI_PROJECT_NAME")
        
        # Azure authentication
        self.client_id = os.getenv("AZURE_CLIENT_ID")
        self.client_secret = os.getenv("AZURE_CLIENT_SECRET")
        self.tenant_id = os.getenv("AZURE_TENANT_ID")
        
        # MCP server configuration
        self.mcp_url = os.getenv("AZURE_AI_FOUNDRY_MCP_URL", "http://localhost:8081/mcp")
        self.mcp_transport = os.getenv("AZURE_AI_FOUNDRY_MCP_TRANSPORT", "stdio")
        self.mcp_host = os.getenv("AZURE_AI_FOUNDRY_MCP_HOST", "localhost")
        self.mcp_port = int(os.getenv("AZURE_AI_FOUNDRY_MCP_PORT", "8081"))
        
        # Additional Azure configuration
        self.swagger_path = os.getenv("SWAGGER_PATH")
        self.enable_web_search = os.getenv("AZURE_AI_ENABLE_WEB_SEARCH", "false").lower() == "true"
        self.enable_code_interpreter = os.getenv("AZURE_AI_ENABLE_CODE_INTERPRETER", "false").lower() == "true"
        
    def is_configured(self) -> bool:
        """Check if Azure AI Foundry MCP is properly configured"""
        # At minimum, we need the AI project endpoint
        return bool(self.ai_project_endpoint)
    
    def has_authentication(self) -> bool:
        """Check if Azure authentication is configured"""
        return bool(self.tenant_id and (self.client_id or self.client_secret))
    
    def get_missing_config(self) -> List[str]:
        """Get list of missing required configuration variables"""
        missing = []
        
        if not self.ai_project_endpoint:
            missing.append("AZURE_AI_PROJECT_ENDPOINT")
        
        # Optional but recommended
        optional_missing = []
        if not self.subscription_id:
            optional_missing.append("AZURE_SUBSCRIPTION_ID")
        if not self.resource_group:
            optional_missing.append("AZURE_RESOURCE_GROUP_NAME")
        if not self.project_name:
            optional_missing.append("AZURE_AI_PROJECT_NAME")
        if not self.tenant_id:
            optional_missing.append("AZURE_TENANT_ID")
        
        return missing, optional_missing


class AzureAIFoundryMCPIntegration:
    """Manages Azure AI Foundry MCP server integration"""
    
    def __init__(self, config: AzureAIFoundryMCPConfig):
        self.config = config
        self.server_registered = False
        self._capabilities = []
        
    async def initialize(self) -> bool:
        """Initialize Azure AI Foundry MCP server connection"""
        if not self.config.is_configured():
            missing_required, missing_optional = self.config.get_missing_config()
            
            logger.warning(
                "Azure AI Foundry MCP not fully configured",
                missing_required=missing_required,
                missing_optional=missing_optional
            )
            
            if missing_required:
                logger.error(
                    "Required Azure AI Foundry configuration missing",
                    required=missing_required
                )
                return False
        
        if not self.config.has_authentication():
            logger.warning(
                "Azure AI Foundry authentication not configured - some features may not work",
                hint="Set AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET"
            )
        
        try:
            # Determine available capabilities based on configuration
            capabilities = [MCPCapability.CUSTOM]
            self._capabilities = []
            
            # Always available if endpoint is configured
            self._capabilities.append(AzureAIFoundryCapability.EVALUATION)
            self._capabilities.append(AzureAIFoundryCapability.MODEL)
            
            # Check for optional capabilities
            if self.config.swagger_path:
                self._capabilities.append(AzureAIFoundryCapability.FINETUNING)
                logger.info("Azure AI Foundry finetuning capabilities available")
            
            # Knowledge capabilities typically available
            self._capabilities.append(AzureAIFoundryCapability.KNOWLEDGE)
            
            # Register Azure AI Foundry MCP server
            azure_server = MCPServer(
                name="azure_ai_foundry",
                url=self.config.mcp_url,
                capabilities=capabilities,
                auth_token=None,  # Azure uses different auth mechanism
                timeout=60  # AI operations may take longer
            )
            
            mcp_integration.register_server(azure_server)
            
            # Try to discover available tools
            try:
                tools = await mcp_integration.discover_tools("azure_ai_foundry")
                logger.info(
                    "Discovered Azure AI Foundry tools",
                    tool_count=len(tools),
                    capabilities=self._capabilities
                )
                self.server_registered = True
                return True
            except Exception as discover_error:
                logger.warning(
                    "Could not discover Azure AI Foundry MCP tools (server may not be running)",
                    error=str(discover_error),
                    hint="Check if azure-ai-foundry/mcp-foundry server is running"
                )
                # Still mark as registered since configuration is valid
                self.server_registered = False
                return False
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure AI Foundry MCP server: {e}")
            return False
    
    def get_evaluation_config(self, paradigm: str) -> Dict[str, Any]:
        """Get evaluation configuration optimized for each paradigm"""
        
        # Base configuration
        config = {
            "project_endpoint": self.config.ai_project_endpoint,
            "enable_reasoning": True,  # Use reasoning for better evaluations
            "max_retries": 3,
        }
        
        # Paradigm-specific evaluation preferences
        paradigm_configs = {
            "dolores": {
                # Revolutionary - focus on uncovering biases and controversial topics
                "evaluation_types": ["groundedness", "coherence", "controversy_detection"],
                "reasoning_effort": "high",  # Deep analysis for truth-seeking
                "safety_settings": {
                    "enable_content_safety": True,
                    "detect_harmful_content": True,
                    "bias_detection": True
                }
            },
            "teddy": {
                # Devotion - focus on helpfulness and safety
                "evaluation_types": ["helpfulness", "safety", "groundedness"],
                "reasoning_effort": "medium",
                "safety_settings": {
                    "enable_content_safety": True,
                    "prioritize_user_safety": True,
                    "filter_sensitive_content": True
                }
            },
            "bernard": {
                # Analytical - focus on accuracy and coherence
                "evaluation_types": ["groundedness", "coherence", "relevance", "factual_accuracy"],
                "reasoning_effort": "high",  # Thorough analysis
                "safety_settings": {
                    "enable_content_safety": False,  # Focus on accuracy over safety
                    "academic_mode": True
                }
            },
            "maeve": {
                # Strategic - focus on business value and practicality
                "evaluation_types": ["relevance", "coherence", "business_value"],
                "reasoning_effort": "medium",
                "safety_settings": {
                    "enable_content_safety": True,
                    "business_appropriate": True
                }
            }
        }
        
        # Merge paradigm-specific config
        if paradigm.lower() in paradigm_configs:
            config.update(paradigm_configs[paradigm.lower()])
        
        return config
    
    async def evaluate_content(
        self,
        content: str,
        paradigm: str,
        evaluation_type: str = "groundedness",
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate content using Azure AI Foundry evaluation tools"""
        
        if not self.server_registered:
            raise RuntimeError("Azure AI Foundry MCP server not initialized")
        
        # Get paradigm-specific configuration
        eval_config = self.get_evaluation_config(paradigm)
        
        # Build tool name for evaluation
        tool_name = f"azure_ai_foundry_evaluate_{evaluation_type}"
        
        # Prepare evaluation parameters
        eval_params = {
            "content": content,
            "context": context,
            "paradigm": paradigm,
            "reasoning_enabled": eval_config.get("reasoning_effort") == "high",
            "project_endpoint": eval_config.get("project_endpoint"),
        }
        
        # Add safety settings
        if "safety_settings" in eval_config:
            eval_params["safety_settings"] = eval_config["safety_settings"]
        
        try:
            # Execute evaluation through MCP
            result = await mcp_integration.execute_tool_call(
                tool_name,
                eval_params
            )
            
            # Post-process results based on paradigm
            processed_result = self._process_evaluation_for_paradigm(
                result,
                paradigm,
                evaluation_type
            )
            
            return processed_result
            
        except Exception as e:
            logger.error(f"Azure AI Foundry evaluation error for {paradigm}: {e}")
            raise
    
    async def query_knowledge_base(
        self,
        query: str,
        paradigm: str,
        knowledge_type: str = "general"
    ) -> Dict[str, Any]:
        """Query Azure AI Foundry knowledge base"""
        
        if not self.server_registered:
            raise RuntimeError("Azure AI Foundry MCP server not initialized")
        
        # Build tool name for knowledge query
        tool_name = f"azure_ai_foundry_knowledge_query"
        
        # Prepare query parameters
        query_params = {
            "query": query,
            "knowledge_type": knowledge_type,
            "paradigm": paradigm,
            "project_endpoint": self.config.ai_project_endpoint,
        }
        
        try:
            # Execute query through MCP
            result = await mcp_integration.execute_tool_call(
                tool_name,
                query_params
            )
            
            return {
                "paradigm": paradigm,
                "query": query,
                "results": result,
                "knowledge_type": knowledge_type
            }
            
        except Exception as e:
            logger.error(f"Azure AI Foundry knowledge query error for {paradigm}: {e}")
            raise
    
    def _process_evaluation_for_paradigm(
        self,
        raw_results: Any,
        paradigm: str,
        evaluation_type: str
    ) -> Dict[str, Any]:
        """Process evaluation results according to paradigm preferences"""
        
        # Extract results based on expected format
        if isinstance(raw_results, dict):
            results = raw_results
        else:
            results = {"evaluation": raw_results}
        
        # Add paradigm context
        results["paradigm"] = paradigm
        results["evaluation_type"] = evaluation_type
        
        # Paradigm-specific result interpretation
        if paradigm.lower() == "dolores":
            # Focus on controversial or biased content
            if "bias_detected" in results:
                results["paradigm_insight"] = "Bias detection aligns with revolutionary truth-seeking"
        
        elif paradigm.lower() == "teddy":
            # Focus on safety and helpfulness
            if "safety_score" in results:
                results["paradigm_insight"] = "Safety evaluation supports protective devotion"
        
        elif paradigm.lower() == "bernard":
            # Focus on analytical accuracy
            if "groundedness" in results:
                results["paradigm_insight"] = "Groundedness analysis supports analytical approach"
        
        elif paradigm.lower() == "maeve":
            # Focus on strategic value
            if "relevance" in results:
                results["paradigm_insight"] = "Relevance assessment supports strategic planning"
        
        return results
    
    def get_available_capabilities(self) -> List[str]:
        """Get list of available Azure AI Foundry capabilities"""
        return [cap.value for cap in self._capabilities]


# Global instance
azure_ai_foundry_config = AzureAIFoundryMCPConfig()
azure_ai_foundry_mcp = AzureAIFoundryMCPIntegration(azure_ai_foundry_config)


async def initialize_azure_ai_foundry_mcp():
    """Initialize Azure AI Foundry MCP integration during startup"""
    if await azure_ai_foundry_mcp.initialize():
        logger.info("✓ Azure AI Foundry MCP server initialized")
        return True
    else:
        logger.warning("Azure AI Foundry MCP server not initialized")
        return False