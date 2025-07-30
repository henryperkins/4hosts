# Four Hosts Research Application - Claude Code Context

This file contains essential information about the Four Hosts paradigm-aware research application for Claude Code and custom subagents.

## Project Overview

The Four Hosts application is a sophisticated research system that classifies queries into four distinct "consciousness paradigms" and executes paradigm-aligned research with AI-powered answer generation.

### The Four Paradigms:
- **Dolores (Revolutionary)**: Investigative, exposing systemic issues, challenging status quo
- **Teddy (Devotion)**: Supportive, community-focused, empathetic care
- **Bernard (Analytical)**: Data-driven, empirical research, academic rigor
- **Maeve (Strategic)**: Business intelligence, actionable strategies, optimization

## System Architecture

### Backend Pipeline Flow:
1. **Query Classification** → Identifies primary/secondary paradigms
2. **Context Engineering** → W-S-C-I pipeline refines query
3. **Search Execution** → Paradigm-aware multi-API search
4. **Answer Generation** → LLM synthesis with paradigm tone

### Key Backend Components:
- `main.py`: FastAPI application with WebSocket support
- `classification_engine.py`: Query analysis and paradigm classification
- `context_engineering.py`: W-S-C-I (Write-Select-Compress-Isolate) pipeline
- `paradigm_search.py`: Paradigm-specific search strategies
- `search_apis.py`: Google, ArXiv, Brave, PubMed API integrations
- `answer_generator*.py`: Paradigm-aligned content generation
- `research_orchestrator.py`: Coordinates the research flow
- `llm_client.py`: Azure OpenAI GPT-4 integration

### Frontend Stack:
- React 18 with TypeScript
- Vite build tool
- Tailwind CSS
- Context API for state
- WebSocket for real-time updates

## Custom Subagents

Located in `.claude/agents/`:

1. **paradigm-analyzer**: Reviews code for paradigm alignment
2. **research-optimizer**: Optimizes search and answer quality
3. **test-engineer**: Creates paradigm-aware tests
4. **api-integrator**: Adds new search data sources
5. **llm-prompt-engineer**: Optimizes GPT-4 prompts
6. **react-component-builder**: Creates paradigm-aware UI components

## Important Technical Details

### API Limits:
- Google Custom Search: 100 queries/day (free tier)
- ArXiv: 3 requests/second
- Brave Search: 2000 queries/month
- PubMed: 3 requests/second

### Authentication:
- JWT-based auth with refresh tokens
- Role-based access (FREE, BASIC, PRO, ENTERPRISE, ADMIN)
- Deep research requires PRO+ subscription

### Testing:
- pytest for backend tests
- Test files in `backend/tests/`
- Mock external APIs for consistent testing

### Environment Variables:
- `AZURE_OPENAI_API_KEY`: Required for LLM
- `GOOGLE_API_KEY` + `GOOGLE_SEARCH_ENGINE_ID`: For Google search
- `BRAVE_API_KEY`: For Brave search
- Database and Redis configuration

## Development Commands

```bash
# Backend
cd four-hosts-app/backend
source venv/bin/activate
python main.py

# Frontend
cd four-hosts-app/frontend
npm run dev

# Tests
pytest backend/tests/
```

## Paradigm-Specific Considerations

When working on this codebase:

1. **Maintain paradigm consistency** across the pipeline
2. **Check enum mappings** between HostParadigm and Paradigm
3. **Test with paradigm-specific queries** for each component
4. **Optimize API usage** due to rate limits
5. **Follow established patterns** in existing components

## Recent Updates

- Custom subagents created for specialized development tasks
- Enhanced with project-specific implementation details
- Paradigm-aware component patterns documented
- API integration patterns clarified
- Testing strategies outlined

---

*This context file helps Claude Code and subagents understand the system architecture and make paradigm-aligned contributions.*