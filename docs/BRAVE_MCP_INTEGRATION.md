# Brave Search MCP Server Integration

This document describes how the Four Hosts application integrates with the Brave Search MCP (Model Context Protocol) server to enhance research capabilities.

## Overview

The Brave Search MCP integration provides:
- **Multi-modal search** (web, local, video, image, news)
- **AI-powered summarization**
- **Paradigm-optimized search configurations**
- **Tool calling support for Azure OpenAI o3 model**

## Architecture

```
Four Hosts App
    ↓
Enhanced Research Orchestrator
    ↓
MCP Integration Layer
    ↓
Brave MCP Server (HTTP/STDIO)
    ↓
Brave Search API
```

## Setup

### 1. Install Brave MCP Server

Using Docker (recommended):
```bash
docker-compose -f docker-compose.mcp.yml up -d
```

Using NPX:
```bash
npx @modelcontextprotocol/server-brave-search
```

### 2. Configure Environment Variables

Add to your `.env` file:
```env
# Required
BRAVE_SEARCH_API_KEY=your_brave_api_key_here

# Optional (defaults shown)
BRAVE_MCP_URL=http://localhost:8080/mcp
BRAVE_MCP_TRANSPORT=HTTP
BRAVE_MCP_HOST=localhost
BRAVE_MCP_PORT=8080
BRAVE_DEFAULT_COUNTRY=US
BRAVE_DEFAULT_LANGUAGE=en
BRAVE_SAFE_SEARCH=moderate
```

### 3. Get Brave API Key

1. Visit [Brave Search API](https://brave.com/search/api/)
2. Sign up for an account
3. Choose a plan:
   - **Free**: 2,000 queries/month, basic web search
   - **Pro**: Enhanced features, local search, AI summaries

## Paradigm-Specific Optimizations

### Dolores (Revolutionary)
- **Focus**: Exposing truth, uncovering injustices
- **Search Types**: Web, News
- **Features**:
  - Recent content prioritized
  - Independent sources emphasized
  - Controversial topics included

### Teddy (Devotion)
- **Focus**: Helping communities, protecting vulnerable
- **Search Types**: Web, Local
- **Features**:
  - Strict safe search
  - Official sources prioritized
  - Community resources highlighted

### Bernard (Analytical)
- **Focus**: Data-driven research, academic rigor
- **Search Types**: Web, Summarizer
- **Features**:
  - Academic sources prioritized
  - AI summarization enabled
  - Statistical data included

### Maeve (Strategic)
- **Focus**: Business intelligence, competitive advantage
- **Search Types**: Web, News
- **Features**:
  - Recent market data
  - Business sources prioritized
  - Strategic insights emphasized

## Usage Examples

### Basic Search
```python
from services.brave_mcp_integration import brave_mcp, BraveSearchType

# Initialize (done automatically during startup)
await brave_mcp.initialize()

# Perform paradigm-aware search
result = await brave_mcp.search_with_paradigm(
    query="renewable energy innovations",
    paradigm="bernard",  # Analytical perspective
    search_type=BraveSearchType.WEB
)
```

### Enhanced Research
```python
from services.enhanced_research_orchestrator import enhanced_orchestrator
from services.classification_engine import HostParadigm

# Execute comprehensive research
result = await enhanced_orchestrator.execute_paradigm_research(
    query="impact of AI on employment",
    primary_paradigm=HostParadigm.DOLORES,  # Revolutionary perspective
    options={
        "use_brave": True,  # Enable Brave MCP
        "use_traditional": True,  # Also use Google, ArXiv, etc.
        "depth": "deep"  # Comprehensive analysis
    }
)
```

### Tool Calling with o3
```python
# The system automatically provides Brave search tools to o3
tools = await mcp_integration.discover_tools("brave_search")
# Tools include:
# - brave_web_search
# - brave_local_search
# - brave_video_search
# - brave_image_search
# - brave_news_search
# - brave_summarizer
```

## API Endpoints

### Research with Brave MCP
```
POST /research/query
{
    "query": "your search query",
    "options": {
        "use_brave": true,
        "depth": "standard"
    }
}
```

The system will automatically:
1. Classify the query into paradigms
2. Configure Brave search for the paradigm
3. Execute parallel searches
4. Synthesize results with LLM

## Monitoring and Debugging

### Check MCP Server Status
```bash
curl http://localhost:8080/health
```

### View Logs
```bash
docker logs brave-mcp-server
```

### Debug Configuration
```python
from services.brave_mcp_integration import brave_config

print(f"API Key Configured: {brave_config.is_configured()}")
print(f"MCP URL: {brave_config.mcp_url}")
```

## Rate Limits

- **Free Plan**: 2,000 queries/month
- **Pro Plan**: Based on subscription level
- The system automatically handles rate limiting and caching

## Best Practices

1. **Cache Results**: Search results are automatically cached to reduce API calls
2. **Paradigm Alignment**: Let the system choose search parameters based on paradigm
3. **Error Handling**: The system gracefully falls back to traditional search if Brave fails
4. **Tool Usage**: The o3 model can autonomously decide when to use Brave tools

## Troubleshooting

### MCP Server Not Connecting
1. Check if server is running: `docker ps`
2. Verify port is not in use: `lsof -i :8080`
3. Check API key is valid

### No Search Results
1. Verify API key has remaining quota
2. Check search parameters in logs
3. Try a simpler query

### Integration Not Working
1. Ensure environment variables are set
2. Check initialization logs during startup
3. Verify network connectivity to MCP server

## Future Enhancements

1. **Additional MCP Servers**: Support for filesystem, database, and custom MCP servers
2. **Advanced Filtering**: More sophisticated result filtering based on paradigm
3. **Multi-Language**: Paradigm-aware search in multiple languages
4. **Custom Tools**: Ability to add custom search tools through MCP