#!/bin/bash

echo "🚀 Starting Four Hosts Application"
echo "=================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo "📁 Project root: $PROJECT_ROOT"
echo "🔄 Running in background mode"

# Initialize variables
MCP_STARTED=false

# Source nvm to ensure npm is available when running with sudo
if [ -f "/home/azureuser/.nvm/nvm.sh" ]; then
    export NVM_DIR="/home/azureuser/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
    echo "✅ NVM loaded, using node $(node --version) and npm $(npm --version)"
else
    echo "⚠️  NVM not found, checking for system npm..."
    if ! command -v npm >/dev/null 2>&1; then
        echo "❌ npm is not installed or not in PATH"
        echo "   Please install Node.js and npm first"
        exit 1
    fi
fi

# Function to cleanup on exit
cleanup() {
    echo -e "\n\n🛑 Shutting down services..."
    
    # Stop backend and frontend
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    
    # Stop MCP server if running
    if [ -f "$BACKEND_DIR/.mcp_server.pid" ]; then
        MCP_PID=$(cat "$BACKEND_DIR/.mcp_server.pid")
        echo "Stopping MCP server (PID: $MCP_PID)..."
        kill $MCP_PID 2>/dev/null
        rm "$BACKEND_DIR/.mcp_server.pid"
    fi
    
    # Stop Docker containers if running
    if [ "$MCP_STARTED" = "true" ] && command -v docker &> /dev/null; then
        echo "Stopping Docker containers..."
        cd "$BACKEND_DIR"
        docker compose -f docker-compose.mcp.yml down
    fi
    
    exit
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Increase file watcher limits to prevent ENOSPC error
echo "📊 Configuring system file watcher limits..."
current_watches=$(cat /proc/sys/fs/inotify/max_user_watches 2>/dev/null || echo "unknown")
current_instances=$(cat /proc/sys/fs/inotify/max_user_instances 2>/dev/null || echo "unknown")
current_events=$(cat /proc/sys/fs/inotify/max_queued_events 2>/dev/null || echo "unknown")

echo "Current limits - watches: $current_watches, instances: $current_instances, events: $current_events"

# Try to increase all inotify limits
echo "Attempting to increase inotify limits..."
if sudo sysctl -w fs.inotify.max_user_watches=1048576 >/dev/null 2>&1 && \
   sudo sysctl -w fs.inotify.max_user_instances=2048 >/dev/null 2>&1 && \
   sudo sysctl -w fs.inotify.max_queued_events=32768 >/dev/null 2>&1; then
    echo "✅ All inotify limits increased successfully"
    echo "   watches: 1048576, instances: 2048, events: 32768"
else
    echo "⚠️  Could not increase inotify limits automatically."
    echo "   If you get ENOSPC errors, run these commands manually:"
    echo "   sudo sysctl -w fs.inotify.max_user_watches=1048576"
    echo "   sudo sysctl -w fs.inotify.max_user_instances=2048"
    echo "   sudo sysctl -w fs.inotify.max_queued_events=32768"
fi

# Kill any existing processes on our ports
echo -e "\n🔍 Checking for existing processes..."
if command -v lsof >/dev/null 2>&1; then
    lsof -ti:8000 | xargs -r kill -9 2>/dev/null
    lsof -ti:5173 | xargs -r kill -9 2>/dev/null
    lsof -ti:5174 | xargs -r kill -9 2>/dev/null
    lsof -ti:8080 | xargs -r kill -9 2>/dev/null  # MCP server port
    echo "✅ Ports cleared"
else
    echo "⚠️  lsof not available, skipping port cleanup"
fi

# Start MCP Server if Brave API key is configured
echo -e "\n🌐 Checking MCP Server configuration..."
BACKEND_DIR="$PROJECT_ROOT/four-hosts-app/backend"

if [ -f "$BACKEND_DIR/.env" ]; then
    if grep -q "BRAVE_SEARCH_API_KEY=" "$BACKEND_DIR/.env" && ! grep -q "BRAVE_SEARCH_API_KEY=your_brave_search_api_key_here" "$BACKEND_DIR/.env"; then
        echo "✓ Brave API key detected"
        
        # Check if Docker is available
        if command -v docker &> /dev/null; then
            echo "✓ Docker detected"
            
            # Check if docker network exists, create if not
            if ! docker network inspect fourhosts-network &> /dev/null; then
                echo "Creating Docker network..."
                docker network create fourhosts-network
            fi
            
            # Start Brave MCP server using Docker Compose
            echo "Starting Brave MCP server..."
            cd "$BACKEND_DIR"
            docker compose -f docker-compose.mcp.yml up -d
            MCP_STARTED=true
            
            # Wait for MCP server to be ready
            echo "Waiting for MCP server to be ready..."
            for i in {1..30}; do
                if curl -s http://localhost:8080/health > /dev/null 2>&1; then
                    echo "✓ Brave MCP server is ready"
                    break
                fi
                echo -n "."
                sleep 1
            done
            echo ""
        else
            echo "⚠️  Docker not found. Starting MCP server with NPX..."
            
            # Check if NPX is available
            if command -v npx &> /dev/null; then
                # Start MCP server in background
                cd "$BACKEND_DIR"
                npx @modelcontextprotocol/server-brave-search &
                MCP_PID=$!
                echo "✓ Brave MCP server started (PID: $MCP_PID)"
                
                # Save PID for cleanup
                echo $MCP_PID > .mcp_server.pid
            else
                echo "❌ Neither Docker nor NPX found. Skipping MCP server."
                echo "   Install Docker or Node.js to enable Brave search."
            fi
        fi
    else
        echo "⚠️  Brave API key not configured. Skipping MCP server."
        echo "   To enable, add your API key to $BACKEND_DIR/.env"
    fi
else
    echo "⚠️  Backend .env file not found. Skipping MCP server."
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

# Start frontend with explicit port (try file watching first, fallback to polling)
echo "Starting frontend with file watching..."
VITE_PORT=$PORT npm run dev -- --port $PORT &
FRONTEND_PID=$!

# Wait a moment to see if frontend starts successfully
sleep 3

# Check if frontend process is still running
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "⚠️  Frontend failed with file watching, trying polling mode..."

    # Kill any remaining processes on the port
    if command -v lsof >/dev/null 2>&1; then
        lsof -ti:$PORT | xargs -r kill -9 2>/dev/null
    fi

    # Start with polling fallback
    CHOKIDAR_USEPOLLING=true VITE_PORT=$PORT npm run dev -- --port $PORT &
    FRONTEND_PID=$!
    sleep 2

    if kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "✅ Frontend started with polling mode (PID: $FRONTEND_PID)"
        echo "   Available at: http://localhost:$PORT"
        echo "   Note: Using polling mode (slower file watching)"
    else
        echo "❌ Frontend failed to start even with polling mode"
        exit 1
    fi
else
    echo "✅ Frontend started with file watching (PID: $FRONTEND_PID)"
    echo "   Available at: http://localhost:$PORT"
fi

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

echo -e "\n⏳ Waiting for services to run..."
echo "Press Ctrl+C to stop all services"

# Keep the script running to maintain the background processes
# This prevents the EXIT trap from immediately killing the services
wait $BACKEND_PID $FRONTEND_PID
