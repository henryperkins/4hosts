#!/bin/bash

# Start script for Four Hosts with Brave MCP Server

echo "🚀 Starting Four Hosts Research Application with Brave MCP Server..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found. Please copy .env.example to .env and configure."
    exit 1
fi

# Check if Brave API key is configured
if ! grep -q "BRAVE_SEARCH_API_KEY=" .env || grep -q "BRAVE_SEARCH_API_KEY=your_brave_search_api_key_here" .env; then
    echo "⚠️  Warning: Brave API key not configured. MCP server will not start."
    echo "   To enable Brave search, add your API key to .env file."
else
    echo "✓ Brave API key detected"
    
    # Check if Docker is available
    if command -v docker &> /dev/null; then
        echo "✓ Docker detected"
        
        # Ensure Docker network exists
        if ! docker network inspect fourhosts-network &> /dev/null; then
            echo "Creating Docker network..."
            docker network create fourhosts-network
        fi

        # Prepare or build Brave MCP server image (clone repo if missing)
        USE_NPX_MCP=false
        BACKEND_DIR="$(pwd)"
        MCP_DIR="$BACKEND_DIR/brave-search-mcp-server"

        if ! docker image inspect brave-mcp-server:latest >/dev/null 2>&1; then
            echo "Preparing Brave MCP server source..."
            if [ ! -d "$MCP_DIR" ]; then
                echo "Cloning Brave MCP server repository..."
                if command -v git >/dev/null 2>&1; then
                    git clone --depth 1 https://github.com/brave/brave-search-mcp-server.git "$MCP_DIR" || {
                        echo "❌ Failed to clone Brave MCP server repository"
                        USE_NPX_MCP=true
                    }
                else
                    echo "❌ git not found; cannot clone Brave MCP server"
                    USE_NPX_MCP=true
                fi
            else
                if command -v git >/dev/null 2>&1 && [ -d "$MCP_DIR/.git" ]; then
                    echo "Updating Brave MCP server repository..."
                    git -C "$MCP_DIR" pull --ff-only || true
                fi
            fi

            if [ "$USE_NPX_MCP" = "false" ]; then
                echo "Building Brave MCP server Docker image..."
                cd "$MCP_DIR"
                docker build -t brave-mcp-server:latest . || {
                    echo "❌ Docker build failed"
                    USE_NPX_MCP=true
                }
                cd "$BACKEND_DIR"
            fi
        fi

        # Start Brave MCP server using Docker Compose or NPX fallback
        echo "Starting Brave MCP server..."
        if [ "$USE_NPX_MCP" = "true" ]; then
            if command -v npx &> /dev/null; then
                # Load API key from .env and start MCP server in background
                BRAVE_API_KEY_VAL="$(grep -E '^(BRAVE_API_KEY|BRAVE_SEARCH_API_KEY)=' .env | head -n1 | cut -d= -f2-)"
                if [ -z "$BRAVE_API_KEY_VAL" ]; then
                    echo "❌ BRAVE_API_KEY not found in .env; cannot start NPX MCP server"
                else
                    BRAVE_API_KEY="$BRAVE_API_KEY_VAL" npx -y @brave/brave-search-mcp-server --transport http --host 0.0.0.0 --port 8080 &
                    MCP_PID=$!
                    echo "✓ Brave MCP server started via NPX (PID: $MCP_PID)"
                    # Save PID for cleanup
                    echo $MCP_PID > .mcp_server.pid
                fi
            else
                echo "❌ Could not start MCP server: Docker build failed and NPX not available"
            fi
        else
            docker compose -f docker-compose.mcp.yml up -d

            # Wait for MCP server to be ready
            echo "Waiting for MCP server to be ready..."
            for i in {1..30}; do
                if curl -s http://localhost:8080/ping > /dev/null 2>&1; then
                    echo "✓ Brave MCP server is ready"
                    break
                fi
                echo -n "."
                sleep 1
            done
            echo ""
        fi
    else
        echo "⚠️  Docker not found. Starting MCP server with NPX..."
        
        # Check if NPX is available
        if command -v npx &> /dev/null; then
            # Start MCP server in background
            BRAVE_API_KEY_VAL="$(grep -E '^(BRAVE_API_KEY|BRAVE_SEARCH_API_KEY)=' .env | head -n1 | cut -d= -f2-)"
            if [ -z "$BRAVE_API_KEY_VAL" ]; then
                echo "❌ BRAVE_API_KEY not found in .env; cannot start NPX MCP server"
            else
                BRAVE_API_KEY="$BRAVE_API_KEY_VAL" npx -y @brave/brave-search-mcp-server --transport http --host 0.0.0.0 --port 8080 &
                MCP_PID=$!
                echo "✓ Brave MCP server started (PID: $MCP_PID)"
                # Save PID for cleanup
                echo $MCP_PID > .mcp_server.pid
            fi
        else
            echo "❌ Neither Docker nor NPX found. Cannot start MCP server."
            echo "   Install Docker or Node.js to enable Brave search."
        fi
    fi
fi

# Start the main application
echo "Starting Four Hosts backend..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start the application
python main_new.py

# Cleanup on exit
if [ -f .mcp_server.pid ]; then
    MCP_PID=$(cat .mcp_server.pid)
    echo "Stopping MCP server (PID: $MCP_PID)..."
    kill $MCP_PID 2> /dev/null
    rm .mcp_server.pid
fi

if command -v docker &> /dev/null; then
    echo "Stopping Docker containers..."
    docker compose -f docker-compose.mcp.yml down
fi
