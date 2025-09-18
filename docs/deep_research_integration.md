# Deep Research Integration

## Overview

This document describes the integration of OpenAI's Responses API and o3-deep-research model into the Four Hosts Research Application.

## Features

### 1. OpenAI Responses API Client (`openai_responses_client.py`)
- Complete implementation of the Responses API
- Support for web search, code interpreter, and MCP tools
- Background mode for long-running tasks
- Streaming support for real-time updates

### 2. Deep Research Service (`deep_research_service.py`)
- Integration with the paradigm-based research system
- Paradigm-specific system prompts for focused research
- Support for multiple research modes:
  - `PARADIGM_FOCUSED`: Uses paradigm-specific prompts
  - `COMPREHENSIVE`: Multi-perspective analysis
  - `QUICK_FACTS`: Fast fact-finding
  - `ANALYTICAL`: Data-driven analysis
  - `STRATEGIC`: Business/strategic focus

### 3. Research Orchestrator Integration
- New `execute_deep_research` method that combines standard search with deep research
- Seamless integration with existing context engineering pipeline
- Cost tracking and progress reporting

### 4. API Endpoints
- **POST /research/query**: Updated to support `deep_research` depth option
- **POST /research/deep**: Dedicated endpoint for deep research (PRO users only)
- **GET /research/deep/status**: Get status of deep research queries
- **GET /test/deep-research**: Admin-only test endpoint

### 5. Frontend Updates
- Added "Deep AI" option in research depth selector (PRO badge)
- Deep research indicator in results display
- Support for deep research in API service

## Usage

### Backend
```python
# Using deep research in a request
research = ResearchQuery(
    query="Your research question",
    options=ResearchOptions(
        depth=ResearchDepth.DEEP_RESEARCH,
        max_sources=100
    )
)
```

### Frontend
```typescript
// Submit deep research
const options: ResearchOptions = {
  depth: 'deep_research',
  max_sources: 100
}
await api.submitResearch(query, options)
```

## Access Control
- Deep research requires PRO subscription or higher
- Free users see the option but cannot select it
- Admin users have access to test endpoints

## Technical Details

### Deep Research Flow
1. User submits query with `deep_research` depth
2. System classifies query and creates context engineering
3. Deep research service is invoked with paradigm context
4. o3-deep-research model performs multi-source analysis
5. Results are combined with standard search (if enabled)
6. Final synthesis includes deep research content

### Cost Management
- Deep research tracks web search calls, code interpreter usage, and MCP calls
- Costs are aggregated and reported in the response
- Budget alerts can be configured

### Background Processing
- Deep research runs in background mode by default
- Progress is tracked and reported via WebSocket
- Timeout is configurable (default: 30 minutes)

## Future Enhancements
- [ ] Full background mode support with webhooks
- [ ] MCP server integration for private data
- [ ] Enhanced web search tool configuration
- [ ] Prompt engineering UI for custom research instructions