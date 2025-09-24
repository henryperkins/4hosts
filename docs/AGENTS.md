# Repository Guidelines

## Project Structure & Module Organization
- `four-hosts-app/backend`: FastAPI service (Python 3.12) with contracts, services, migrations, and tests under `tests/`.
- `four-hosts-app/frontend`: Vite + React/TypeScript UI; shared pieces sit in `src/components`, hooks in `src/hooks`, state logic in `src/store`.
- `docs/` plus `four-hosts-app/docs/`: architecture notes and runbooks—consult them before touching APIs or orchestration flows.
- Root scripts such as `start-app.sh` and `stop-app.sh` coordinate the local stack; keep them executable and cross-shell friendly.

- `./start-app.sh` / `./stop-app.sh`: launch or halt backend (uvicorn) and frontend (Vite); the starter script now seeds `backend/.env`, ensures Docker Postgres on `localhost:5433`, runs Alembic migrations, and then starts both services with hot reload.
- Backend setup: `cd four-hosts-app/backend && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`.
- Backend-only run: `four-hosts-app/backend/start_backend.sh`; migrations via `alembic upgrade head` (uses the same async config as startup entrypoints).
- Frontend setup: `cd four-hosts-app/frontend && npm install && npm run dev -- --port 5173`; production build via `npm run build`.
- Quality gates: `pytest`, `npm run lint`, and `npm run design:lint` must pass before submitting changes.

## Coding Style & Naming Conventions
- Python follows PEP 8, 4-space indentation, and explicit type hints; modules/functions use `snake_case`, classes use `PascalCase`.
- Prefer `structlog.get_logger(__name__)` for logging (see `services/monitoring.py`) and sanitize inputs with helpers in `utils/security`.
- React/TypeScript components remain `PascalCase`, hooks start with `use`, and Tailwind utility classes come from `tailwind.config.ts`; avoid inline styles.
- Configuration stays in gitignored `.env` files; never commit secrets or tokens.

## Testing Guidelines
- Run `cd four-hosts-app/backend && source venv/bin/activate && pytest`; tag slow/external checks with `@pytest.mark.integration` so `pytest -m "not integration"` stays quick.
- Name backend regression files `test_<feature>.py` and reuse fixtures from `tests/conftest.py`; favor async tests for coroutine endpoints.
- Frontend changes should still run lint/design scripts and document manual verification if no automated test exists.

## Commit & Pull Request Guidelines
- Commit messages mirror history: concise imperatives with optional Conventional Commit prefixes (e.g., `feat(security): tighten token audit`).
- PRs include a change summary, linked issue, verification notes (`pytest`, `npm run build`, etc.), and UI screenshots or GIFs for visual work.
- Keep scope focused; flag schema or config migrations early and stage unrelated refactors separately.

- `start-app.sh` seeds `backend/.env` with a development JWT secret and exports a default `DATABASE_URL` targeting the compose Postgres on port 5433—rotate secrets and override environment variables for shared or deployed environments.
- Strip API keys, cookies, and dumps from the tree; extend `.gitignore` when introducing new generated artifacts or logs.
