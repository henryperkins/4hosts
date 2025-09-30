"""
MCP (Model Context Protocol) Integration Module
Provides MCP server integration, telemetry, and authentication
"""

# Core MCP integration
from .mcp_integration import (
    MCPCapability,
    MCPServer,
    MCPToolDefinition,
    MCPIntegration,
    mcp_integration,
    configure_default_servers,
)

# Telemetry and monitoring
from .mcp_telemetry import (
    MCPToolCall,
    MCPServerMetrics,
    MCPTelemetry,
    mcp_telemetry,
    track_mcp_call,
    complete_mcp_call,
)

# Azure AI Foundry integration
from .azure_ai_foundry_mcp_integration import (
    AzureAIFoundryCapability,
    AzureAIFoundryMCPConfig,
    AzureAIFoundryMCPIntegration,
    azure_ai_foundry_config,
    azure_ai_foundry_mcp,
    initialize_azure_ai_foundry_mcp,
)

# Azure AI Foundry authentication
from .azure_ai_foundry_auth import (
    AzureAIFoundryAuth,
    azure_foundry_auth,
)

# Brave MCP integration
from .brave_mcp_integration import (
    BraveMCPIntegration,
    brave_mcp,
    initialize_brave_mcp,
)

__all__ = [
    # Core MCP
    "MCPCapability",
    "MCPServer",
    "MCPToolDefinition",
    "MCPIntegration",
    "mcp_integration",
    "configure_default_servers",
    # Telemetry
    "MCPToolCall",
    "MCPServerMetrics",
    "MCPTelemetry",
    "mcp_telemetry",
    "track_mcp_call",
    "complete_mcp_call",
    # Azure AI Foundry
    "AzureAIFoundryCapability",
    "AzureAIFoundryMCPConfig",
    "AzureAIFoundryMCPIntegration",
    "azure_ai_foundry_config",
    "azure_ai_foundry_mcp",
    "initialize_azure_ai_foundry_mcp",
    "AzureAIFoundryAuth",
    "azure_foundry_auth",
    # Brave MCP
    "BraveMCPIntegration",
    "brave_mcp",
    "initialize_brave_mcp",
]