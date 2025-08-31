#!/usr/bin/env bash
#
# Convenience script: boot full Four Hosts stack (backend + frontend)
#
# Prerequisites:
#   - Python 3.12 available on PATH
#   - Node.js >= 20 installed
#   - Postgres running and DATABASE_URL exported (or configured in .env)
#
# Usage:
#   chmod +x start-app.sh
#   ./start-app.sh

set -euo pipefail

BACKEND_DIR="$(dirname "$0")/backend"
FRONTEND_DIR="$(dirname "$0")/frontend"

echo "🚀  Starting Four Hosts stack..."

# --- Backend --------------------------------------------------------------
echo "\n🟦  Booting FastAPI backend (hot-reload)"
(
  cd "$BACKEND_DIR"
  # Activate venv if it exists; otherwise run python directly.
  if [ -d "venv" ]; then
    source venv/bin/activate
  fi
  exec uvicorn main_new:app --reload --port 8000 --log-level info
) &
BACKEND_PID=$!

# --- Frontend -------------------------------------------------------------
echo "\n🟩  Launching Vite dev server"
(
  cd "$FRONTEND_DIR"
  exec npm run dev -- --port 5173
) &
FRONTEND_PID=$!

echo "\n✔️  Backend: http://localhost:8000   |   Frontend: http://localhost:5173"
echo "Press Ctrl+C to stop both."

trap 'echo "\n🛑  Shutting down..."; kill $BACKEND_PID $FRONTEND_PID; wait' INT TERM
wait
