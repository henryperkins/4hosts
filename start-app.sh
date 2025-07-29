#!/bin/bash

echo "🚀 Starting Four Hosts Application"
echo "=================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo "📁 Project root: $PROJECT_ROOT"
echo "🔄 Running in background mode"

# Function to cleanup on exit
cleanup() {
    echo -e "\n\n🛑 Shutting down services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Increase file watcher limit to prevent ENOSPC error
echo "📊 Configuring system file watcher limits..."
if [ -w /proc/sys/fs/inotify/max_user_watches ]; then
    echo 524288 | sudo tee /proc/sys/fs/inotify/max_user_watches > /dev/null
    echo "✅ File watcher limit increased"
else
    echo "⚠️  Could not increase file watcher limit (may need sudo)"
fi

# Kill any existing processes on our ports
echo -e "\n🔍 Checking for existing processes..."
if command -v lsof >/dev/null 2>&1; then
    lsof -ti:8000 | xargs -r kill -9 2>/dev/null
    lsof -ti:5173 | xargs -r kill -9 2>/dev/null
    lsof -ti:5174 | xargs -r kill -9 2>/dev/null
    echo "✅ Ports cleared"
else
    echo "⚠️  lsof not available, skipping port cleanup"
fi

# Start Backend
echo -e "\n📡 Starting Backend Service..."
BACKEND_DIR="$PROJECT_ROOT/four-hosts-app/backend"

if [ ! -d "$BACKEND_DIR" ]; then
    echo "❌ Error: Backend directory not found at $BACKEND_DIR"
    echo "   Please ensure the project structure is correct"
    exit 1
fi

cd "$BACKEND_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install dependencies
source venv/bin/activate
pip install -r requirements.txt > /dev/null 2>&1

# Set environment variable
export ENVIRONMENT=development

# Start backend without watching venv directory
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --reload --reload-dir . --reload-exclude venv --reload-exclude __pycache__ --reload-exclude "*.pyc" --reload-exclude "*.log" --reload-exclude "test_*" &
BACKEND_PID=$!
echo "✅ Backend started (PID: $BACKEND_PID)"
echo "   Available at: http://localhost:8000"
echo "   API Docs at: http://localhost:8000/docs"

# Wait a moment for backend to start
sleep 2

# Start Frontend
echo -e "\n🎨 Starting Frontend Service..."
FRONTEND_DIR="$PROJECT_ROOT/four-hosts-app/frontend"

if [ ! -d "$FRONTEND_DIR" ]; then
    echo "❌ Error: Frontend directory not found at $FRONTEND_DIR"
    echo "   Please ensure the project structure is correct"
    exit 1
fi

cd "$FRONTEND_DIR"

# Verify package.json exists
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found in $FRONTEND_DIR"
    echo "   Please ensure the frontend project is properly set up"
    exit 1
fi

echo "✅ Frontend package.json verified"

# Check if .env exists, if not create from example
if [ ! -f .env ] && [ -f .env.example ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
fi

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

# Check if default port is available, otherwise use 5174
PORT=5173
if command -v lsof >/dev/null 2>&1; then
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        PORT=5174
        echo "⚠️  Port 5173 is in use, using port $PORT instead"
    fi
else
    echo "⚠️  lsof not available, using default port $PORT"
fi

# Start frontend with explicit port
VITE_PORT=$PORT npm run dev -- --port $PORT &
FRONTEND_PID=$!
echo "✅ Frontend started (PID: $FRONTEND_PID)"
echo "   Available at: http://localhost:$PORT"

# Display status
echo -e "\n✨ Four Hosts Application is running!"
echo "=================================="
echo "Backend:  http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo "Frontend: http://localhost:$PORT"

echo -e "\n🔧 Services started in background mode"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo -e "\nTo stop services, run:"
echo "  ./stop-app.sh"
echo "  or: kill $BACKEND_PID $FRONTEND_PID"
echo "  or: pkill -f 'uvicorn\|vite'"
