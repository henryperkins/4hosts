# Four Hosts Research System MVP

An AI-powered research application that classifies queries into consciousness paradigms based on Westworld hosts and provides paradigm-specific research results.

## Features

- **Paradigm Classification**: Automatically classifies research queries into four paradigms:
  - 🔴 **Dolores (Revolutionary)**: Exposes systemic injustices and power imbalances
  - 🟠 **Teddy (Devotion)**: Protects and supports vulnerable communities
  - 🔵 **Bernard (Analytical)**: Provides objective analysis and empirical evidence
  - 🟢 **Maeve (Strategic)**: Delivers actionable strategies and competitive advantage

- **Real-time Research**: Conducts paradigm-aware research with simulated results
- **Interactive UI**: Clean, responsive interface built with React and Tailwind CSS
- **RESTful API**: FastAPI backend with comprehensive endpoints

## Tech Stack

- **Backend**: FastAPI 0.116.1, Python 3.12, Pydantic 2.10
- **Frontend**: React 19.1, Vite 7, TypeScript, Tailwind CSS 4.1
- **Architecture**: RESTful API with CORS support

## Quick Start

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the FastAPI server:
```bash
uvicorn main_new:app --reload
```

The API will be available at `http://localhost:8000`
API documentation at `http://localhost:8000/docs`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:5173`

## Enable Exa (Optional)

- Add your Exa API key and toggles:
  - Edit `four-hosts-app/backend/.env` (or copy from `.env.example`) and set:
    ```
    EXA_API_KEY=your-key
    SEARCH_DISABLE_EXA=0
    EXA_INCLUDE_TEXT=0
    EXA_SEARCH_AS_PRIMARY=0
    EXA_BASE_URL=https://api.exa.ai
    EXA_TIMEOUT_SEC=15
    ```
  - The top-level `start-app.sh` seeds these keys in `backend/.env` if missing.
  - Docker users: `four-hosts-app/docker-compose.yml` injects the same `EXA_*` env vars into the backend container.

- Notes:
  - Set `EXA_SEARCH_AS_PRIMARY=1` to make Exa the primary provider; otherwise it is added as a fallback.
  - `EXA_INCLUDE_TEXT=1` returns page text directly from Exa (useful for faster credibility checks); it may increase payload size and cost.

### Docker Compose Deployment

Use the bundled Docker setup when you want the full stack (PostgreSQL, Redis, backend, frontend) running with one command.

1. Ensure the shared Docker network exists (required because several compose files join the same network):
   ```bash
   cd four-hosts-app
   ./scripts/setup-docker-network.sh
   ```

2. Start the full stack from the repository root:
   ```bash
   docker compose up -d --build
   ```

   - Backend API: http://localhost:8001
   - API docs:    http://localhost:8001/docs
   - Frontend UI: http://localhost:5173
   - PostgreSQL:  localhost:5433 (user/password/fourhosts)

3. To tear the stack down:
   ```bash
   docker compose down
   ```

Optional compose files:

- `backend/docker-compose.yml` starts only PostgreSQL and Redis for local backend development.
- `backend/docker-compose.mcp.yml` runs the Brave MCP server; run the network script above first so it can attach to `fourhosts-network`.

## Documentation

- Technical flow and implementation map: `docs/agentic-research-technical-flow.md`
- Concept overview (SSOTA): `docs/ssota-concept.md`

## API Endpoints

- `POST /research/query` - Submit a research query
- `GET /research/status/{research_id}` - Check research status
- `GET /research/results/{research_id}` - Get research results
- `POST /paradigms/classify` - Classify a query without full research

## Usage

1. Enter a research question (minimum 10 characters)
2. The system will classify your query into one of the four paradigms
3. Research is conducted based on the paradigm classification
4. Results are displayed with:
   - Summary
   - Paradigm-specific sections
   - Action items
   - Sources and citations
   - Metadata about the research process

## Development

This is an MVP implementation with simulated research results. The paradigm classification uses keyword matching and heuristics. In a production version, this would integrate with:
- Real search APIs (Google, Bing)
- LLM APIs for advanced classification
- Database for result storage
- Caching layer for performance
- User authentication and preferences

## Azure OpenAI Integration

This implementation includes support for Azure OpenAI services. See [backend/README_AZURE_OPENAI.md](backend/README_AZURE_OPENAI.md) for configuration details.

## License

This project is part of the Four Hosts Agentic Research System.
