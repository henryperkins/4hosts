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
uvicorn main:app --reload
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
