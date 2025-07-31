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
        
        # Start Brave MCP server using Docker Compose
        echo "Starting Brave MCP server..."
        docker-compose -f docker-compose.mcp.yml up -d
        
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
            npx @modelcontextprotocol/server-brave-search &
            MCP_PID=$!
            echo "✓ Brave MCP server started (PID: $MCP_PID)"
            
            # Save PID for cleanup
            echo $MCP_PID > .mcp_server.pid
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
python main.py

# Cleanup on exit
if [ -f .mcp_server.pid ]; then
    MCP_PID=$(cat .mcp_server.pid)
    echo "Stopping MCP server (PID: $MCP_PID)..."
    kill $MCP_PID 2> /dev/null
    rm .mcp_server.pid
fi

if command -v docker &> /dev/null; then
    echo "Stopping Docker containers..."
    docker-compose -f docker-compose.mcp.yml down
fi