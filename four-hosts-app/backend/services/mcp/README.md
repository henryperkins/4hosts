# MCP (Model Context Protocol) Module

This module provides comprehensive MCP server integration, telemetry, and authentication for the Four Hosts research system.

## Overview

The MCP module enables the Four Hosts system to connect to remote MCP servers for extended capabilities like AI evaluation, knowledge queries, and model access. It includes built-in support for:

- **Azure AI Foundry** - AI evaluation and model capabilities
- **Brave Search** - Web search via MCP
- **Custom MCP Servers** - Extensible framework for adding new servers

## Module Structure

```
services/mcp/
├── __init__.py                            # Unified exports
├── mcp_integration.py                     # Core MCP framework
├── mcp_telemetry.py                      # Metrics and monitoring
├── azure_ai_foundry_mcp_integration.py   # Azure AI Foundry integration
├── azure_ai_foundry_auth.py              # Azure authentication
└── brave_mcp_integration.py              # Brave search integration
```

## Quick Start

### Basic Usage

```python
from services.mcp import (
    mcp_integration,
    configure_default_servers,
    mcp_telemetry,
)

# Configure MCP servers
configure_default_servers()

# Get available MCP tools for LLM
mcp_tools = mcp_integration.get_responses_mcp_tools()

# Check metrics
summary = mcp_telemetry.get_summary()
print(f"Total MCP calls: {summary['total_calls']}")
```

### Azure AI Foundry

```python
from services.mcp import azure_ai_foundry_mcp, azure_foundry_auth

# Check authentication
auth_info = azure_foundry_auth.get_auth_info()
if auth_info["configured"]:
    # Evaluate content with paradigm-specific config
    result = await azure_ai_foundry_mcp.evaluate_content(
        content="Example text",
        paradigm="bernard",
        evaluation_type="groundedness"
    )
```

## Components

### 1. MCP Integration (`mcp_integration.py`)

Core MCP framework that manages connections to remote MCP servers.

**Key Classes:**
- `MCPIntegration` - Main integration class
- `MCPServer` - Server configuration
- `MCPCapability` - Server capability types

**Key Functions:**
- `configure_default_servers()` - Auto-configure MCP servers from environment
- `register_server(server)` - Register a custom MCP server
- `discover_tools(server_name)` - Discover available tools
- `execute_tool_call(tool_name, params)` - Execute a tool
- `get_responses_mcp_tools()` - Get tools in Responses API format

**Example:**

```python
from services.mcp import mcp_integration, MCPServer, MCPCapability

# Register a custom MCP server
server = MCPServer(
    name="my_server",
    url="http://localhost:8080/mcp",
    capabilities=[MCPCapability.CUSTOM],
    auth_token="optional-token",
    timeout=30
)
mcp_integration.register_server(server)

# Discover tools
tools = await mcp_integration.discover_tools("my_server")

# Execute a tool
result = await mcp_integration.execute_tool_call(
    "my_server_tool_name",
    {"param1": "value1"}
)
```

### 2. Telemetry (`mcp_telemetry.py`)

Comprehensive monitoring and metrics for all MCP operations.

**Key Classes:**
- `MCPTelemetry` - Main telemetry collector
- `MCPToolCall` - Individual tool call record
- `MCPServerMetrics` - Per-server metrics

**Tracked Metrics:**
- Total calls, success/failure counts, success rate
- Average/min/max duration
- Tool usage frequency
- Paradigm usage patterns
- Recent errors

**Example:**

```python
from services.mcp import mcp_telemetry

# Get server-specific metrics
metrics = mcp_telemetry.get_server_metrics("azure_ai_foundry")
print(f"Success rate: {metrics['success_rate']:.1%}")
print(f"Avg duration: {metrics['avg_duration_ms']}ms")

# Get paradigm usage
paradigm_usage = mcp_telemetry.get_paradigm_usage()
# {"bernard": 150, "dolores": 100, "teddy": 80, "maeve": 70}

# Export for monitoring systems
export = mcp_telemetry.export_metrics()
```

### 3. Azure AI Foundry (`azure_ai_foundry_mcp_integration.py`)

Azure AI Foundry MCP server integration with paradigm-aware evaluation.

**Key Classes:**
- `AzureAIFoundryMCPIntegration` - Main integration
- `AzureAIFoundryMCPConfig` - Configuration
- `AzureAIFoundryCapability` - Available capabilities

**Features:**
- Paradigm-specific evaluation configurations
- Content evaluation (groundedness, coherence, safety, etc.)
- Knowledge base queries
- AI model access

**Example:**

```python
from services.mcp import azure_ai_foundry_mcp

# Get paradigm-specific configuration
config = azure_ai_foundry_mcp.get_evaluation_config("bernard")
# Returns config optimized for analytical paradigm

# Evaluate content
result = await azure_ai_foundry_mcp.evaluate_content(
    content="This is a test statement.",
    paradigm="bernard",
    evaluation_type="groundedness",
    context="Supporting evidence here..."
)
```

**Paradigm Configurations:**

- **Dolores (Revolutionary)**: High reasoning, bias detection, controversy detection
- **Teddy (Devotion)**: Medium reasoning, safety prioritization, helpfulness
- **Bernard (Analytical)**: High reasoning, factual accuracy, academic mode
- **Maeve (Strategic)**: Medium reasoning, business value, practical relevance

### 4. Azure Authentication (`azure_ai_foundry_auth.py`)

Azure authentication using DefaultAzureCredential.

**Key Classes:**
- `AzureAIFoundryAuth` - Authentication manager

**Supported Methods** (in priority order):
1. Service Principal (AZURE_CLIENT_ID + AZURE_CLIENT_SECRET)
2. Azure CLI (`az login`)
3. Managed Identity (for Azure deployments)

**Example:**

```python
from services.mcp import azure_foundry_auth

# Get credential (auto-selects best method)
credential = azure_foundry_auth.get_credential()

# Get auth headers for API calls
headers = azure_foundry_auth.get_auth_headers(
    scope="https://cognitiveservices.azure.com/.default"
)

# Check authentication status
auth_info = azure_foundry_auth.get_auth_info()
# {
#     "configured": True,
#     "method": "azure_cli",
#     "tenant_id": "...",
#     "subscription_id": "...",
#     "has_service_principal": False
# }
```

### 5. Brave Search (`brave_mcp_integration.py`)

Brave search MCP server integration for web search capabilities.

**Key Classes:**
- `BraveMCPIntegration` - Main integration
- `BraveMCPConfig` - Configuration
- `BraveSearchType` - Search types (web, news, images, videos)

**Example:**

```python
from services.mcp import brave_mcp

# Initialize (if not auto-configured)
await brave_mcp.initialize()

# Execute search
results = await brave_mcp.execute_search(
    query="AI research latest",
    search_type="web",
    paradigm="bernard"
)
```

## Configuration

### Environment Variables

```bash
# Enable MCP tools in deep research
ENABLE_MCP_DEFAULT=1

# Azure AI Foundry
AZURE_AI_PROJECT_ENDPOINT=https://your-project.services.ai.azure.com/api/projects/your-project
AZURE_AI_PROJECT_NAME=your-project
AZURE_TENANT_ID=your-tenant-id
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP_NAME=your-resource-group

# Optional: Service Principal
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret

# Brave Search
BRAVE_API_KEY=your-brave-api-key
BRAVE_MCP_URL=http://localhost:8080/mcp

# Custom MCP Servers
MCP_FILESYSTEM_URL=http://localhost:8082/mcp
MCP_DATABASE_URL=http://localhost:8083/mcp
```

### Automatic Configuration

MCP servers are automatically configured on application startup:

```python
# In core/app.py
from services.mcp import configure_default_servers

configure_default_servers()
# Registers all configured MCP servers
```

## Integration with Research Pipeline

### Deep Research Service

MCP tools are automatically included in deep research queries when `ENABLE_MCP_DEFAULT=1`:

```python
# In deep_research_service.py
from services.mcp import mcp_integration

# Get MCP tools for Responses API
mcp_tools = mcp_integration.get_responses_mcp_tools()

# Pass to LLM
await responses_deep_research(
    query="What is AI safety?",
    mcp_servers=mcp_tools,
    ...
)
```

### LLM Access

MCP tools are passed to Azure OpenAI Responses API:

```json
{
  "model": "o3",
  "input": [...],
  "tools": [
    {
      "type": "web_search_preview",
      "search_context_size": "medium"
    },
    {
      "type": "mcp",
      "server_label": "azure_ai_foundry",
      "server_url": "stdio://azure-ai-foundry/mcp-foundry",
      "require_approval": "never"
    }
  ]
}
```

## Monitoring & Observability

### Metrics Collection

All MCP operations are automatically tracked:

```python
from services.mcp import mcp_telemetry

# Real-time metrics
summary = mcp_telemetry.get_summary()
# {
#     "total_calls": 450,
#     "successful_calls": 440,
#     "failed_calls": 10,
#     "success_rate": 0.978,
#     "active_servers": 2,
#     "servers": ["azure_ai_foundry", "brave"]
# }

# Server-specific metrics
metrics = mcp_telemetry.get_server_metrics("azure_ai_foundry")
# {
#     "total_calls": 300,
#     "avg_duration_ms": 245.5,
#     "tools_used": {"evaluate_content": 200, "query_knowledge": 100},
#     "paradigms_used": {"bernard": 150, "dolores": 100, ...}
# }
```

### Logging

Structured logging for all MCP operations:

```python
# Automatic logs
2025-09-30 11:00:00 [info] MCP tool call started
  tool=evaluate_content
  server=azure_ai_foundry
  paradigm=bernard

2025-09-30 11:00:01 [info] MCP tool call completed
  duration_ms=245.5
  success=True
```

### WebSocket Updates

Real-time progress updates sent to connected clients:

```javascript
// Frontend receives
{
  "type": "mcp_tool_executing",
  "data": {
    "research_id": "abc123",
    "server": "azure_ai_foundry",
    "tool": "evaluate_content"
  }
}
```

## Testing

### Run Tests

```bash
# All MCP tests
pytest tests/test_*mcp*.py -v

# Azure AI Foundry tests
pytest tests/test_azure_ai_foundry_mcp_integration.py -v

# Integration tests
python test_azure_mcp_integration.py
python test_llm_mcp_accessibility.py
python test_pipeline_integration.py
```

### Test Coverage

- Configuration validation
- Authentication detection
- Server registration
- Tool discovery
- Tool execution
- Telemetry tracking
- Pipeline integration
- Import verification

## Extending the Module

### Adding a New MCP Server

```python
from services.mcp import mcp_integration, MCPServer, MCPCapability

# 1. Create server configuration
my_server = MCPServer(
    name="my_server",
    url="http://localhost:8084/mcp",
    capabilities=[MCPCapability.CUSTOM],
    auth_token=None,
    timeout=30
)

# 2. Register server
mcp_integration.register_server(my_server)

# 3. Discover tools
tools = await mcp_integration.discover_tools("my_server")

# 4. Tools are now available to LLM
mcp_tools = mcp_integration.get_responses_mcp_tools()
# Includes your new server
```

### Creating a Custom Integration

```python
# 1. Create integration class
class MyMCPIntegration:
    def __init__(self, config):
        self.config = config
        self.server_registered = False

    async def initialize(self) -> bool:
        # Register server
        server = MCPServer(...)
        mcp_integration.register_server(server)
        self.server_registered = True
        return True

    async def my_custom_operation(self, params):
        # Execute custom tool
        result = await mcp_integration.execute_tool_call(
            "my_server_tool",
            params
        )
        return result

# 2. Create global instance
my_mcp = MyMCPIntegration(config)

# 3. Initialize on startup
async def initialize_my_mcp():
    return await my_mcp.initialize()
```

## Best Practices

1. **Always use `configure_default_servers()`** on startup
2. **Check `ENABLE_MCP_DEFAULT`** before attaching tools
3. **Use telemetry** for monitoring production usage
4. **Handle errors gracefully** - MCP servers may be unavailable
5. **Include paradigm context** in tool parameters for better results
6. **Monitor metrics** regularly for performance issues
7. **Reset telemetry** periodically in long-running instances

## Troubleshooting

### Common Issues

**Problem:** MCP tools not appearing in LLM calls
**Solution:** Check `ENABLE_MCP_DEFAULT=1` in .env

**Problem:** Authentication errors
**Solution:** Run `az login` or set service principal credentials

**Problem:** Tool discovery fails
**Solution:** Verify MCP server is running and accessible

**Problem:** High latency
**Solution:** Check `mcp_telemetry.get_server_metrics()` for slow tools

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
```

View MCP communication details in logs.

## API Reference

See individual module docstrings for detailed API documentation:

- `mcp_integration.py` - Core MCP framework
- `mcp_telemetry.py` - Telemetry system
- `azure_ai_foundry_mcp_integration.py` - Azure AI Foundry
- `azure_ai_foundry_auth.py` - Azure authentication
- `brave_mcp_integration.py` - Brave search

## Related Documentation

- [Azure AI Foundry MCP Integration Guide](../../docs/azure_ai_foundry_mcp_integration.md)
- [MCP Integration Guide](../../docs/mcp_integration.md)
- [Four Hosts Paradigm System](../../docs/paradigms.md)

---

**Version:** 2.0
**Last Updated:** 2025-09-30
**Status:** Production Ready