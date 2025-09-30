# Repository Guidelines

## Project Structure & Module Organization
The application lives in `four-hosts-app/`. The FastAPI backend (`backend/`) splits responsibilities across `core/` for orchestration, `services/` for external adapters, `schemas/` for Pydantic payloads, `routes/` for HTTP endpoints, and database material in `alembic/` plus `database/`. The React + Vite frontend is in `frontend/` with components in `src/`. Repository-level helpers (`models/`, `utils/`, `source_credibility.py`) support the small root test suite in `tests/`, while broader research docs are stored under `docs/`.

## Build, Test, and Development Commands
Use `./start-app.sh` from the repo root to spin up Redis, seed `backend/.env`, and run backend and frontend together. Backend iteration happens in `four-hosts-app/backend`; activate the virtualenv and run `python -m uvicorn main_new:app --reload` or `./start_backend.sh`. Frontend development occurs in `four-hosts-app/frontend`; install with `npm install`, serve locally via `npm run dev`, and create production builds with `npm run build`. Run backend tests using `cd four-hosts-app/backend && pytest -q`. The root `pytest.ini` pins `-k "credibility"`; override it when needed (`pytest -q tests -k ""`) to exercise additional root tests. Frontend checks rely on `npm run lint` and the design token guard `npm run design:lint`.

## Coding Style & Naming Conventions
Python follows PEP 8 with four-space indentation, snake_case modules, and type annotations mirroring existing files under `core/` and `services/`. Use the shared logging setup in `logging_config.py`/`structlog`, keep configuration in `.env` (generated locally by `start-app.sh`), and avoid committing secrets. Frontend code should respect `frontend/eslint.config.js`, keeping components PascalCase, hooks camelCase, and Tailwind classes grouped consistently with existing UI files.

## Testing Guidelines
Backend tests reside in `four-hosts-app/backend/tests/` and follow pytest discovery on `test_*.py`. Mark scenarios requiring external systems with `@pytest.mark.integration` and invoke them via `pytest -m integration` before merging relevant changes. Regenerate OpenAPI artefacts through `python tests/generate_openapi.py` when response models shift, and extend the root credibility tests only when shared helpers change.

## Commit & Pull Request Guidelines
Adopt conventional subjects (`feat:`, `fix:`, `chore:`) with concise imperative descriptions and affected scope. Pull requests should describe the behaviour change, include proof of executed tests or npm scripts, reference issues, and add UI screenshots when interfaces move; flag any configuration or secret adjustments explicitly and loop in the owners of touched areas.

## Environment & Security Notes
`start-app.sh`, `start_backend.sh`, and `start-docker.sh` encapsulate local setup; run `four-hosts-app/scripts/setup-docker-network.sh` before composing multi-service stacks. Treat certificates in `frontend/certs/` and scripts such as `generate-ssl-cert.sh` or `setup-ssl-certbot.sh` as sensitive, and confirm generated `.env` files remain ignored before pushing.
