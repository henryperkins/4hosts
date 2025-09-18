# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Four Hosts application is a paradigm-aware research system that classifies queries into four consciousness paradigms (based on Westworld hosts) and executes paradigm-aligned research with AI-powered answer generation.

### The Four Paradigms
- **Dolores (Revolutionary)**: Investigative, exposing systemic issues, challenging status quo
- **Teddy (Devotion)**: Supportive, community-focused, empathetic care
- **Bernard (Analytical)**: Data-driven, empirical research, academic rigor
- **Maeve (Strategic)**: Business intelligence, actionable strategies, optimization

## Common Development Commands

### Backend
```bash
# Start backend server
cd four-hosts-app/backend
source venv/bin/activate  # On Windows: venv\Scripts\activate (Python 3.12 runtime)
uvicorn main_new:app --reload

# Run tests
pytest                              # Run all tests
pytest tests/test_main.py          # Run specific test file
pytest -k "test_classification"    # Run tests matching pattern
pytest -m integration              # Run integration tests only

# Database migrations
alembic upgrade head               # Apply migrations
alembic revision --autogenerate -m "description"  # Create new migration
```

### Frontend
```bash
# Start frontend dev server
cd four-hosts-app/frontend
npm run dev

# Build for production
npm run build

# Run linting
npm run lint
```

### Full Stack
```bash
# Start both frontend and backend (from repo root)
./start-app.sh  # Ensures dockerized Postgres on 127.0.0.1:5433 and runs Alembic migrations
```

## Architecture Overview

### Backend Pipeline
1. **Query Classification** (`services/classification_engine.py`)
   - Analyzes query to identify primary/secondary paradigms
   - Uses keyword matching, patterns, and optional LLM classification

2. **Context Engineering** (`services/context_engineering.py`)
   - W-S-C-I (Write-Select-Compress-Isolate) pipeline
   - Refines queries for optimal search results

3. **Research Orchestration** (`services/research_orchestrator.py`)
   - Coordinates entire research flow
   - Manages search execution across multiple APIs
   - Handles caching, rate limiting, and result aggregation

4. **Search Execution** (`services/search_apis.py`, `services/paradigm_search.py`)
   - Multi-API search (Google, Brave, ArXiv, PubMed, Semantic Scholar)
   - Paradigm-specific search strategies
   - Result deduplication and credibility scoring

5. **Answer Generation** (`services/answer_generator.py`)
   - LLM-based synthesis using Azure OpenAI GPT-4
   - Paradigm-aligned tone and structure
   - Evidence building and citation management

### Key Service Components
- **Authentication**: JWT-based with refresh tokens (`services/auth_service.py`)
- **Deep Research**: Enhanced multi-round research (`services/deep_research_service.py`)
- **Export**: Multiple format export (`services/export_service.py`)
- **Caching**: Redis-based result caching (`services/cache.py`)
- **Background Processing**: Async task execution (`services/background_llm.py`)

### Data Models
- **Paradigm Models**: `models/paradigms.py` - Core paradigm definitions
- **Context Models**: `models/context_models.py` - Query and classification schemas
- **Synthesis Models**: `models/synthesis_models.py` - Answer generation structures
- **Auth Models**: `models/auth.py` - User and authentication

### Frontend Architecture
- **React 19** with TypeScript
- **Vite** build tooling
- **Tailwind CSS v4** for styling
- **Context API** for state management
- **WebSocket** support for real-time updates
- **Key Components**:
  - `ResearchFormEnhanced.tsx` - Main research input
  - `ResultsDisplayEnhanced.tsx` - Result presentation
  - `ResearchProgress.tsx` - Real-time progress tracking

## Important Configuration

### Required Environment Variables
```bash
# Azure OpenAI (Required for LLM features)
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_DEPLOYMENT=
AZURE_OPENAI_API_VERSION=2024-10-01-preview

# Search APIs (at least one required)
GOOGLE_CSE_API_KEY=
GOOGLE_CSE_CX=
BRAVE_SEARCH_API_KEY=

# Database
DATABASE_URL=postgresql+asyncpg://user:password@127.0.0.1:5433/fourhosts

# Redis (Optional, for caching)
REDIS_URL=redis://localhost:6379

# Auth
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### API Rate Limits
- Google Custom Search: 100 queries/day (free tier)
- ArXiv: 3 requests/second
- Brave Search: 2000 queries/month (free tier)
- PubMed: 3 requests/second
- Semantic Scholar: 100 requests/5 minutes

## Testing Strategy

### Backend Testing
- Unit tests for individual services
- Integration tests requiring external services (marked with `@pytest.mark.integration`)
- Mock external APIs for consistent testing
- Test paradigm classification accuracy
- Validate answer generation quality

### Key Test Files
- `tests/test_classification_engine.py` - Paradigm classification
- `tests/test_azure_openai.py` - LLM integration
- `tests/test_brave_mcp_integration.py` - Brave search
- `tests/test_system.py` - End-to-end system tests

## Development Guidelines

1. **Paradigm Consistency**: Ensure paradigm classification flows correctly through the entire pipeline
2. **Enum Mapping**: Be careful with HostParadigm vs Paradigm enum conversions
3. **Error Handling**: All external API calls should have proper error handling and fallbacks
4. **Rate Limiting**: Respect API rate limits; use caching where possible
5. **Type Safety**: Use TypeScript/Pydantic models for type validation
6. **Async Patterns**: Backend uses async/await throughout for performance

## Troubleshooting

### Common Issues
- **LLM not working**: Check Azure OpenAI credentials and deployment name
- **Search returns no results**: Verify API keys and rate limits
- **Database errors**: Ensure the compose Postgres service is running (port 5433) and rerun `alembic upgrade head`
- **Frontend build fails**: Ensure Node.js 18+ and run `npm install`
- **Test failures**: Some tests require environment variables or external services
