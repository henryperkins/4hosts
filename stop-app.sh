#!/bin/bash

echo "🛑 Stopping Four Hosts Application"
echo "=================================="

# Function to kill processes by port
kill_by_port() {
    local port=$1
    local service_name=$2

    if command -v lsof >/dev/null 2>&1; then
        local pids=$(lsof -ti:$port 2>/dev/null)
        if [ -n "$pids" ]; then
            echo "🔍 Stopping $service_name on port $port..."
            echo "$pids" | xargs kill -TERM 2>/dev/null
            sleep 2
            # Force kill if still running
            echo "$pids" | xargs kill -KILL 2>/dev/null || true
            echo "✅ $service_name stopped"
        else
            echo "ℹ️  No $service_name process found on port $port"
        fi
    else
        echo "⚠️  lsof not available, trying alternative method..."
        pkill -f "$service_name" 2>/dev/null && echo "✅ $service_name processes stopped" || echo "ℹ️  No $service_name processes found"
    fi
}

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
BACKEND_DIR="$PROJECT_ROOT/four-hosts-app/backend"

# Stop services by port
kill_by_port 8000 "Backend"
kill_by_port 5173 "Frontend"
kill_by_port 5174 "Frontend (alt port)"
kill_by_port 8080 "MCP Server"

# Stop MCP server if PID file exists
if [ -f "$BACKEND_DIR/.mcp_server.pid" ]; then
    MCP_PID=$(cat "$BACKEND_DIR/.mcp_server.pid")
    echo "Stopping MCP server (PID: $MCP_PID)..."
    kill $MCP_PID 2>/dev/null
    rm "$BACKEND_DIR/.mcp_server.pid"
    echo "✅ MCP server stopped"
fi

# Stop Docker containers if running
if command -v docker &> /dev/null; then
    if docker ps | grep -q "brave-mcp-server"; then
        echo "Stopping Docker containers..."
        cd "$BACKEND_DIR"
        docker compose -f docker-compose.mcp.yml down
        echo "✅ Docker containers stopped"
    fi
fi

# Also try to kill by process name
echo -e "\n🔍 Cleaning up any remaining processes..."
pkill -f "uvicorn.*main:app" 2>/dev/null && echo "✅ Uvicorn processes stopped" || true
pkill -f "vite.*--port" 2>/dev/null && echo "✅ Vite processes stopped" || true
pkill -f "server-brave-search" 2>/dev/null && echo "✅ Brave MCP processes stopped" || true

echo -e "\n✨ Four Hosts Application stopped!"
