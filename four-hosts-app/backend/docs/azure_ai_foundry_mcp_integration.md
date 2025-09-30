# Azure AI Foundry MCP Integration

This documentation describes the integration of Azure AI Foundry's MCP (Model Context Protocol) server with the Four Hosts research system.

## Overview

The Azure AI Foundry MCP integration provides advanced AI evaluation and model capabilities through Microsoft's Azure AI Foundry platform. This integration enables paradigm-aware content evaluation, knowledge base queries, and AI model interactions.

## Features

- **Content Evaluation**: Evaluate content for groundedness, coherence, safety, and business value
- **Paradigm-Aware Processing**: Different evaluation strategies for each Four Hosts paradigm
- **Knowledge Base Integration**: Query Azure AI knowledge bases with paradigm-specific filters
- **Model Access**: Access to Azure AI Foundry's models and capabilities
- **Graceful Degradation**: Handles missing configuration and server unavailability

## Configuration

### Required Environment Variables

The minimum required configuration is:

```bash
AZURE_AI_PROJECT_ENDPOINT=https://your-project.openai.azure.com/
```

### Recommended Environment Variables

For full functionality, configure these additional variables:

```bash
# Azure Authentication
AZURE_TENANT_ID=your-tenant-id
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret

# Azure Project Details
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP_NAME=your-resource-group
AZURE_AI_PROJECT_NAME=your-project-name

# MCP Server Configuration (optional)
AZURE_AI_FOUNDRY_MCP_URL=http://localhost:8081/mcp
AZURE_AI_FOUNDRY_MCP_TRANSPORT=stdio
AZURE_AI_FOUNDRY_MCP_HOST=localhost
AZURE_AI_FOUNDRY_MCP_PORT=8081

# Feature Toggles (optional)
SWAGGER_PATH=/path/to/swagger.json
AZURE_AI_ENABLE_WEB_SEARCH=false
AZURE_AI_ENABLE_CODE_INTERPRETER=false
```

### Environment File Setup

The `start-app.sh` script automatically creates a `.env` file with commented-out Azure AI Foundry variables. Uncomment and configure the ones you need:

```bash
# Uncomment and set your values:
# AZURE_AI_PROJECT_ENDPOINT=https://your-project.openai.azure.com/
# AZURE_TENANT_ID=your-tenant-id
# AZURE_CLIENT_ID=your-client-id
# AZURE_CLIENT_SECRET=your-client-secret
```

## Paradigm-Specific Behavior

The Azure AI Foundry MCP integration adapts its evaluation and processing based on the Four Hosts paradigm:

### Dolores (Revolutionary)
- **Focus**: Uncovering biases, controversial content detection, truth-seeking
- **Evaluation Types**: Groundedness, coherence, controversy detection
- **Reasoning Effort**: High (deep analysis)
- **Safety Settings**: Bias detection enabled, harmful content detection

### Teddy (Devotion)
- **Focus**: Helpfulness, user safety, protective filtering
- **Evaluation Types**: Helpfulness, safety, groundedness
- **Reasoning Effort**: Medium
- **Safety Settings**: Prioritize user safety, filter sensitive content

### Bernard (Analytical)
- **Focus**: Factual accuracy, coherence, academic rigor
- **Evaluation Types**: Groundedness, coherence, relevance, factual accuracy
- **Reasoning Effort**: High (thorough analysis)
- **Safety Settings**: Academic mode, focus on accuracy over safety

### Maeve (Strategic)
- **Focus**: Business value, practical relevance, strategic insights
- **Evaluation Types**: Relevance, coherence, business value
- **Reasoning Effort**: Medium
- **Safety Settings**: Business-appropriate content filtering

## Usage Examples

### Basic Content Evaluation

```python
from services.azure_ai_foundry_mcp_integration import azure_ai_foundry_mcp

# Evaluate content with Bernard's analytical approach
result = await azure_ai_foundry_mcp.evaluate_content(
    content="This is the content to evaluate",
    paradigm="bernard",
    evaluation_type="groundedness",
    context="Supporting context for the evaluation"
)

print(f"Paradigm: {result['paradigm']}")
print(f"Evaluation: {result['evaluation_type']}")
print(f"Results: {result}")
```

### Knowledge Base Query

```python
# Query knowledge base with Maeve's strategic focus
result = await azure_ai_foundry_mcp.query_knowledge_base(
    query="What are the latest market trends?",
    paradigm="maeve",
    knowledge_type="business"
)

print(f"Query results: {result['results']}")
```

### Check Configuration

```python
from services.azure_ai_foundry_mcp_integration import AzureAIFoundryMCPConfig

config = AzureAIFoundryMCPConfig()

if config.is_configured():
    print("✓ Azure AI Foundry is configured")
    if config.has_authentication():
        print("✓ Authentication is configured")
    else:
        print("⚠️ Authentication not configured - some features may not work")
else:
    missing_required, missing_optional = config.get_missing_config()
    print(f"❌ Missing required config: {missing_required}")
    print(f"ℹ️ Missing optional config: {missing_optional}")
```

## Integration Points

### Research Orchestrator

The Azure AI Foundry MCP is automatically initialized in the research orchestrator:

```python
# This happens automatically during startup
self.azure_ai_foundry_enabled = await initialize_azure_ai_foundry_mcp()
```

### MCP Server Registration

The integration registers with the global MCP system:

```python
# Automatic registration in configure_default_servers()
if os.getenv("AZURE_AI_PROJECT_ENDPOINT"):
    mcp_integration.register_server(MCPServer(
        name="azure_ai_foundry",
        url=os.getenv("AZURE_AI_FOUNDRY_MCP_URL", "stdio://azure-ai-foundry/mcp-foundry"),
        capabilities=[MCPCapability.EVALUATION, MCPCapability.AI_MODEL],
        auth_token=None,  # Uses Azure authentication
        timeout=120
    ))
```

## Error Handling

The integration provides graceful error handling:

- **Missing Configuration**: Logs warnings and continues without Azure AI Foundry features
- **Server Unavailable**: Falls back to other evaluation methods
- **Authentication Errors**: Clear error messages with configuration hints
- **Network Issues**: Proper timeout handling and retries

## Troubleshooting

### Common Issues

1. **"Azure AI project endpoint not configured"**
   - Set `AZURE_AI_PROJECT_ENDPOINT` in your `.env` file

2. **"Authentication not configured"**
   - Set `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, and `AZURE_CLIENT_SECRET`

3. **"MCP server not running"**
   - Ensure the Azure AI Foundry MCP server is running
   - Check the `AZURE_AI_FOUNDRY_MCP_URL` configuration

4. **"Tool execution failed"**
   - Verify your Azure AI project has the required permissions
   - Check the Azure AI Foundry service status

### Debug Mode

Enable debug logging to see detailed MCP communication:

```bash
export LOG_LEVEL=DEBUG
```

This will show MCP tool discovery, execution, and any communication errors.

## Related Documentation

- [Azure AI Foundry Documentation](https://docs.microsoft.com/azure/ai-foundry/)
- [Model Context Protocol (MCP)](https://github.com/microsoft/mcp)
- [Four Hosts Paradigm System](./paradigms.md)
- [MCP Integration Guide](./mcp_integration.md)