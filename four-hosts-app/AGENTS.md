# Repository Guidelines

Welcome to the **Four Hosts Research System**. This guide captures the house-rules for a smooth, predictable contribution workflow. Keep PRs focused, write clear tests, and leave the codebase cleaner than you found it.

## Project Structure

- `backend/` – FastAPI 0.116 (Python 3.12).  Core business logic lives in `services/`, DB layer in `database/`, REST entry-points in `routes/`, and tests under `backend/tests/`.
- `frontend/` – React 19 + Vite.  Feature-oriented folders inside `src/`; UI primitives in `components/ui/`.
- `alembic/` – Database migrations; new models must ship a revision.
- Root scripts (`start_backend.sh`, `test-integration.sh`) provide local orchestration.

## Local Development

Backend
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt      # one-time
uvicorn main:app --reload            # http://localhost:8000
alembic upgrade head                 # apply migrations (Postgres)
pytest -q                            # run unit + async tests
```

Frontend
```bash
cd frontend && npm install           # Node >=20
npm run dev                          # http://localhost:5173
npm run lint && npm test             # lint / (future) tests
```

### Docker Compose
`docker compose -f backend/docker-compose.mcp.yml up` launches the Brave-MCP mock plus supporting network.

### One-shot Stack Runner

Run `./start-app.sh` to spin up backend (FastAPI hot-reload) and frontend (Vite) in one terminal window.  Stop with `Ctrl-C`.

## Coding Style

- **Python**: PEP 8, 4-space indent, 79-char lines.  Auto-format with `black` & `isort`; static-check via `flake8`.
- **TypeScript**: ESLint ruleset in `eslint.config.js`; prefer functional components, `camelCase` filenames, and explicit return types.
- Docstrings follow Google style.  Public APIs must include type hints.

## Testing & Coverage

- `pytest` with `pytest-asyncio`; aim for ≥ 85 % backend coverage (`pytest --cov=backend`).
- Shell flows (`test-auth-flow.sh`, `test-integration.sh`) serve as smoke tests—keep them green.

## Commits & Pull Requests

- **Conventional Commits** (`feat`, `fix`, `docs`, `refactor`, …) gate the changelog.
- Rebase before opening a PR; keep the diff under 400 LoC when possible.
- PR checklist: description, linked issue, test output, and screenshots for UI changes.

## Secrets & Configuration

- Create `.env` (see comments in `backend/main.py`) with `DATABASE_URL`, `SECRET_KEY`, `BRAVE_SEARCH_API_KEY`, etc.
- Never commit credentials; CI will block if `.env` is detected.
