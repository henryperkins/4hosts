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

# Port configuration with defaults
BACKEND_PORT=${BACKEND_PORT:-8000}
FRONTEND_PORT=${FRONTEND_PORT:-5173}

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 1  # Port is in use
    else
        return 0  # Port is available
    fi
}

# Function to find next available port
find_available_port() {
    local base_port=$1
    local port=$base_port
    while ! check_port $port; do
        echo "⚠️  Port $port is in use, trying next..." >&2
        port=$((port + 1))
    done
    echo $port
}

echo "🚀  Starting Four Hosts stack..."

# Seed backend/.env with EXA_* toggles if missing (dev convenience)
ENV_FILE="$BACKEND_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
  echo "Creating $ENV_FILE with default development values..."
  touch "$ENV_FILE"
fi

ensure_env() {
  local key="$1"; shift
  local default="$1"
  if ! grep -qE "^${key}=" "$ENV_FILE"; then
    echo "${key}=${default}" >> "$ENV_FILE"
  fi
}

# Exa provider toggles (safe defaults)
ensure_env EXA_API_KEY ""
ensure_env SEARCH_DISABLE_EXA "0"
ensure_env EXA_INCLUDE_TEXT "0"
ensure_env EXA_SEARCH_AS_PRIMARY "0"
ensure_env EXA_BASE_URL "https://api.exa.ai"
ensure_env EXA_TIMEOUT_SEC "15"

# Check for Docker containers that might conflict
if docker ps --format '{{.Names}}' | grep -q '^fourhosts-'; then
    echo "⚠️  Found running Four Hosts Docker containers."
    echo "   To use Docker stack: http://localhost:5173 (frontend) and http://localhost:8001 (backend)"
    echo "   To stop Docker and run dev mode: docker-compose down"
    read -p "   Continue with dev mode anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Check and adjust ports if needed
if ! check_port $BACKEND_PORT; then
    echo "⚠️  Backend port $BACKEND_PORT is in use"
    BACKEND_PORT=$(find_available_port $BACKEND_PORT)
    echo "✅  Using backend port $BACKEND_PORT instead"
fi

if ! check_port $FRONTEND_PORT; then
    echo "⚠️  Frontend port $FRONTEND_PORT is in use"
    FRONTEND_PORT=$(find_available_port $FRONTEND_PORT)
    echo "✅  Using frontend port $FRONTEND_PORT instead"
fi

# --- Backend --------------------------------------------------------------
echo -e "\n🟦  Booting FastAPI backend (hot-reload)"
(
  cd "$BACKEND_DIR"
  # Activate venv if it exists; otherwise run python directly.
  if [ -d "venv" ]; then
    source venv/bin/activate
  fi
  exec uvicorn main_new:app --reload --port $BACKEND_PORT --log-level info
) &
BACKEND_PID=$!

# --- Frontend -------------------------------------------------------------
echo -e "\n🟩  Launching Vite dev server"
(
  cd "$FRONTEND_DIR"
  exec npm run dev -- --port $FRONTEND_PORT
) &
FRONTEND_PID=$!

echo -e "\n✔️  Backend: http://localhost:$BACKEND_PORT   |   Frontend: http://localhost:$FRONTEND_PORT"
echo "Press Ctrl+C to stop both."

trap 'echo -e "\n🛑  Shutting down..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; wait' INT TERM
wait
