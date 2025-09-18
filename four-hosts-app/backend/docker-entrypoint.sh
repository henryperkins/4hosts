#!/bin/bash
set -euo pipefail

# Run database migrations before starting the app
if command -v alembic >/dev/null 2>&1; then
  echo "📦 Running Alembic migrations..."
  alembic upgrade head
  echo "✅ Alembic migrations applied"
else
  echo "⚠️ Alembic not found on PATH; skipping migrations"
fi

# Execute the provided command
exec "$@"
