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

# Stop services by port
kill_by_port 8000 "Backend"
kill_by_port 5173 "Frontend"
kill_by_port 5174 "Frontend (alt port)"

# Also try to kill by process name
echo -e "\n🔍 Cleaning up any remaining processes..."
pkill -f "uvicorn.*main:app" 2>/dev/null && echo "✅ Uvicorn processes stopped" || true
pkill -f "vite.*--port" 2>/dev/null && echo "✅ Vite processes stopped" || true

echo -e "\n✨ Four Hosts Application stopped!"
