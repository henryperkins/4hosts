# Azure AI Foundry MCP Integration - Complete Flow

## Overview

This document describes how the Four Hosts application integrates with Azure AI Foundry MCP (Model Context Protocol) servers to enhance research capabilities with Azure AI services.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         APPLICATION STARTUP                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  core/app.py:149  →  configure_default_servers()                        │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ Reads Environment Variables:                                    │   │
│  │  - AZURE_AI_PROJECT_ENDPOINT                                    │   │
│  │  - AZURE_CLIENT_ID / AZURE_CLIENT_SECRET                        │   │
│  │  - AZURE_AI_FOUNDRY_MCP_URL                                     │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                           │                                             │
│                           ▼                                             │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ MCP Server Registered:                                          │   │
│  │  Name: azure_ai_foundry                                         │   │
│  │  URL:  http://localhost:8081/mcp                                │   │
│  │  Capabilities: [EVALUATION, AI_MODEL]                           │   │
│  │  Auth: Service Principal (1abaa595-...)                         │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         RESEARCH QUERY FLOW                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. User Request                                                         │
│     │                                                                    │
│     ├─→ POST /api/research                                              │
│     │   Body: { "query": "Evaluate AI safety claims" }                  │
│     │                                                                    │
│     ▼                                                                    │
│                                                                          │
│  2. Deep Research Service (services/deep_research_service.py:372)       │
│     │                                                                    │
│     ├─→ Check ENABLE_MCP_DEFAULT=1 in .env                              │
│     │                                                                    │
│     ├─→ Call mcp_integration.get_responses_mcp_tools()                  │
│     │                                                                    │
│     │   Returns:                                                         │
│     │   [                                                                │
│     │     {                                                              │
│     │       "type": "mcp",                                               │
│     │       "server_label": "azure_ai_foundry",                         │
│     │       "server_url": "http://localhost:8081/mcp",                  │
│     │       "require_approval": "never"                                 │
│     │     }                                                              │
│     │   ]                                                                │
│     │                                                                    │
│     ▼                                                                    │
│                                                                          │
│  3. Query Classification & Paradigm Routing                             │
│     │                                                                    │
│     ├─→ Classify query → ["analytical", "critical"]                     │
│     │                                                                    │
│     ▼                                                                    │
│                                                                          │
│  4. Paradigm Evaluation with MCP Tools                                  │
│     │                                                                    │
│     ├─→ ANALYTICAL Paradigm                                             │
│     │   │                                                                │
│     │   └─→ Receives MCP tools: [evaluation, grounding, verification]  │
│     │                                                                    │
│     ├─→ CRITICAL Paradigm                                               │
│     │   │                                                                │
│     │   └─→ Receives MCP tools: [credibility_assessment, bias_detection]│
│     │                                                                    │
│     ▼                                                                    │
│                                                                          │
│  5. Azure OpenAI Responses API Call                                     │
│     │                                                                    │
│     ├─→ Endpoint: AZURE_OPENAI_ENDPOINT/openai/deployments/o3/...      │
│     │                                                                    │
│     ├─→ Request Includes:                                               │
│     │   {                                                                │
│     │     "messages": [...paradigm context...],                         │
│     │     "tools": [                                                     │
│     │       {                                                            │
│     │         "type": "mcp",                                             │
│     │         "server_label": "azure_ai_foundry",                       │
│     │         "server_url": "http://localhost:8081/mcp",                │
│     │         "require_approval": "never"                               │
│     │       }                                                            │
│     │     ],                                                             │
│     │     "model": "o3"                                                  │
│     │   }                                                                │
│     │                                                                    │
│     ▼                                                                    │
│                                                                          │
│  6. LLM Reasoning with Tool Access                                      │
│     │                                                                    │
│     ├─→ LLM analyzes query                                              │
│     │                                                                    │
│     ├─→ LLM decides to call: evaluate_groundedness                      │
│     │   Parameters:                                                      │
│     │     - query: "Recent AI safety claims"                            │
│     │     - response: "..."                                             │
│     │     - context: "..."                                              │
│     │                                                                    │
│     ▼                                                                    │
│                                                                          │
│  7. MCP Tool Execution                                                   │
│     │                                                                    │
│     ├─→ Azure OpenAI → MCP Server @ localhost:8081                      │
│     │                                                                    │
│     ├─→ MCP Server → Azure AI Foundry API                               │
│     │   Authentication: Service Principal                               │
│     │   Project: oaisubresource                                         │
│     │   Endpoint: https://oaisubresource.services.ai.azure.com/...     │
│     │                                                                    │
│     ├─→ Azure AI Foundry executes groundedness evaluation               │
│     │                                                                    │
│     ├─→ Results: { "score": 0.92, "grounded": true, ... }               │
│     │                                                                    │
│     ▼                                                                    │
│                                                                          │
│  8. Tool Results Integration                                            │
│     │                                                                    │
│     ├─→ Azure OpenAI receives tool results                              │
│     │                                                                    │
│     ├─→ LLM integrates results into reasoning                           │
│     │                                                                    │
│     ├─→ Paradigm response includes tool evidence                        │
│     │   "The claims are well-grounded (score: 0.92)..."                 │
│     │                                                                    │
│     ▼                                                                    │
│                                                                          │
│  9. Mesh Synthesis & Response                                           │
│     │                                                                    │
│     ├─→ Combine paradigm responses                                      │
│     │   - Analytical + MCP evidence                                     │
│     │   - Critical + MCP evidence                                       │
│     │                                                                    │
│     ├─→ Generate final synthesized answer                               │
│     │                                                                    │
│     └─→ Return to user with provenance                                  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

## Key Files and Their Roles

### 1. Configuration & Startup

**File**: `core/app.py`
- **Line 149**: `configure_default_servers()` called during app initialization
- **Purpose**: Registers MCP servers based on environment variables

**File**: `services/mcp/mcp_integration.py`
- **Function**: `configure_default_servers()` (line 308)
- **Purpose**: Reads `AZURE_AI_PROJECT_ENDPOINT` and registers Azure AI Foundry MCP server

### 2. MCP Server Registration

**File**: `services/mcp/mcp_integration.py`
- **Lines 340-347**: Azure AI Foundry server registration
```python
if os.getenv("AZURE_AI_PROJECT_ENDPOINT"):
    mcp_integration.register_server(MCPServer(
        name="azure_ai_foundry",
        url=os.getenv("AZURE_AI_FOUNDRY_MCP_URL", "stdio://azure-ai-foundry/mcp-foundry"),
        capabilities=[MCPCapability.EVALUATION, MCPCapability.AI_MODEL],
        auth_token=None,  # Uses Azure authentication
        timeout=120
    ))
```

### 3. Tool Discovery

**File**: `services/mcp/mcp_integration.py`
- **Method**: `discover_tools(server_name: str)` (line 89)
- **Purpose**: Queries MCP server for available tools
- **Method**: `get_all_tools()` (line 265)
- **Purpose**: Returns cached tools from all registered servers

### 4. Research Query Integration

**File**: `services/deep_research_service.py`
- **Lines 372-374**: MCP tools attachment
```python
if os.getenv("ENABLE_MCP_DEFAULT", "0").lower() in {"1", "true", "yes"}:
    from .mcp.mcp_integration import mcp_integration
    mcp_tools = mcp_integration.get_responses_mcp_tools()
```

### 5. Azure AI Foundry Integration

**File**: `services/mcp/azure_ai_foundry_mcp_integration.py`
- **Class**: `AzureAIFoundryMCPConfig` (line 27)
- **Purpose**: Manages Azure AI Foundry configuration
- **Class**: `AzureAIFoundryMCPIntegration` (line 83)
- **Purpose**: Handles connection to Azure AI Foundry MCP server

### 6. Authentication

**File**: `services/mcp/azure_ai_foundry_auth.py`
- **Class**: `AzureAIFoundryAuth` (line 13)
- **Purpose**: Manages Azure authentication using DefaultAzureCredential
- **Supports**: Service Principal, Azure CLI, Managed Identity

## Environment Variables

### Required Variables
```bash
# Azure AI Foundry Project
AZURE_AI_PROJECT_ENDPOINT=https://oaisubresource.services.ai.azure.com/api/projects/oaisubresource
AZURE_AI_PROJECT_NAME=oaisubresource
AZURE_RESOURCE_GROUP_NAME=rg-hperkin4-8776
AZURE_SUBSCRIPTION_ID=fe000daf-8df4-49e4-99d8-6e789060f760

# Azure Authentication
AZURE_TENANT_ID=41f88ecb-ca63-404d-97dd-ab0a169fd138
AZURE_CLIENT_ID=1abaa595-8b78-4cf2-b4f1-b5cfdaf30ded
AZURE_CLIENT_SECRET=<your-azure-client-secret>

# MCP Server Configuration
AZURE_AI_FOUNDRY_MCP_URL=http://localhost:8081/mcp
AZURE_AI_FOUNDRY_MCP_TRANSPORT=http
AZURE_AI_FOUNDRY_MCP_HOST=localhost
AZURE_AI_FOUNDRY_MCP_PORT=8081

# Enable MCP Tools in Research
ENABLE_MCP_DEFAULT=1
```

## Azure Resources

### Service Principal
- **Application ID**: 1abaa595-8b78-4cf2-b4f1-b5cfdaf30ded
- **Display Name**: four-hosts-mcp
- **Roles**:
  - Cognitive Services User
  - Contributor (Resource Group scope)
  - Azure AI Developer (Resource Group scope)
  - Cognitive Services OpenAI User (oaisubresource scope)

### AI Foundry Project
- **Name**: oaisubresource
- **Endpoint**: https://oaisubresource.services.ai.azure.com
- **Location**: East US 2
- **Model Deployments**: 17 models including o3, gpt-5, gpt-4.1, DeepSeek-V3.1, grok-4-fast-reasoning

## Example MCP Tools from Azure AI Foundry

### 1. Groundedness Evaluation
```json
{
  "name": "evaluate_groundedness",
  "description": "Evaluate if a response is grounded in the provided context",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "response": {"type": "string"},
      "context": {"type": "string"}
    }
  }
}
```

### 2. Credibility Assessment
```json
{
  "name": "assess_credibility",
  "description": "Assess the credibility of sources and claims",
  "inputSchema": {
    "type": "object",
    "properties": {
      "claim": {"type": "string"},
      "sources": {"type": "array"}
    }
  }
}
```

### 3. AI Model Invocation
```json
{
  "name": "invoke_ai_model",
  "description": "Invoke an Azure AI model for analysis",
  "inputSchema": {
    "type": "object",
    "properties": {
      "model_name": {"type": "string"},
      "prompt": {"type": "string"}
    }
  }
}
```

## Paradigm-Specific Tool Usage

### Analytical Paradigm
**Tools**: Evaluation, grounding, verification
**Use Case**: Verify factual claims, assess evidence quality

### Critical Paradigm
**Tools**: Credibility assessment, bias detection
**Use Case**: Identify weaknesses, evaluate source reliability

### Creative Paradigm
**Tools**: Model access, generation
**Use Case**: Explore alternatives, generate scenarios

### Systems Paradigm
**Tools**: Relationship mapping, integration
**Use Case**: Understand interconnections, holistic analysis

## Testing

### Run Integration Tests
```bash
./venv/bin/python -m pytest test_azure_mcp_integration.py -v
./venv/bin/python -m pytest test_llm_mcp_accessibility.py -v
./venv/bin/python -m pytest test_pipeline_integration.py -v
```

### Run Demonstration
```bash
./venv/bin/python demo_mcp_integration.py
```

## Docker Deployment

### MCP Server Container
```yaml
azure-ai-foundry-mcp:
  image: azure-ai-foundry-mcp:latest
  container_name: azure-ai-foundry-mcp
  environment:
    - AZURE_AI_PROJECT_ENDPOINT=${AZURE_AI_PROJECT_ENDPOINT}
    - AZURE_CLIENT_ID=${AZURE_CLIENT_ID}
    - AZURE_CLIENT_SECRET=${AZURE_CLIENT_SECRET}
    # ... additional env vars
  ports:
    - "127.0.0.1:8081:8081"
  networks:
    - fourhosts-network
```

## Benefits of MCP Integration

1. **Enhanced Credibility**: Tool-assisted verification improves answer quality
2. **Transparency**: Tool invocations are logged and traceable
3. **Flexibility**: Easy to add new tools without code changes
4. **Separation of Concerns**: MCP server handles Azure API details
5. **Model Capabilities**: Access to Azure's latest AI models and evaluation services
6. **Scalability**: MCP servers can be deployed independently and scaled

## Troubleshooting

### MCP Server Not Running
- Check if container is running: `docker ps | grep azure-ai-foundry-mcp`
- Check logs: `docker logs azure-ai-foundry-mcp`
- Verify port 8081 is accessible: `curl http://localhost:8081/health`

### Authentication Failures
- Verify service principal credentials in `.env`
- Check role assignments: `az role assignment list --assignee <CLIENT_ID>`
- Test authentication: `./venv/bin/python -c "from services.mcp.azure_ai_foundry_auth import AzureAIFoundryAuth; auth = AzureAIFoundryAuth(); print(auth.get_credential())"`

### No Tools Discovered
- MCP server must be running and accessible
- Check MCP URL in `.env` matches server address
- Verify network connectivity between app and MCP server