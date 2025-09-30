# MCP HTTP Bridge

HTTP-to-stdio bridge for Azure AI Foundry MCP server, allowing HTTP clients to communicate with stdio-based MCP servers.

## Architecture

```
HTTP Client (Four Hosts Backend)
    ↓ POST /mcp
HTTP Bridge Server (this container)
    ↓ stdin/stdout pipes
Azure AI Foundry MCP Server (stdio subprocess)
    ↓
Azure AI Foundry APIs
```

## Features

- **HTTP to stdio translation**: Accepts HTTP POST requests and communicates with stdio-based MCP servers
- **Subprocess management**: Spawns and manages the MCP server as a child process
- **Request/response matching**: Handles concurrent requests with proper ID tracking
- **Health monitoring**: Health and metrics endpoints for monitoring
- **Logging**: Structured logging of all operations

## Endpoints

### POST /mcp
Forward MCP JSON-RPC requests to the stdio server

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools.list",
  "params": {},
  "id": "req_1"
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": "req_1",
  "result": {
    "tools": [...]
  }
}
```

### GET /health
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "mcp_server_running": true,
  "uptime_seconds": 3600,
  "requests_handled": 150
}
```

### GET /metrics
Metrics endpoint

**Response:**
```json
{
  "uptime_seconds": 3600,
  "total_requests": 150,
  "mcp_server_pid": 42,
  "active_requests": 2
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BRIDGE_HOST` | `0.0.0.0` | Host to bind HTTP server |
| `BRIDGE_PORT` | `8081` | Port to bind HTTP server |
| `MCP_COMMAND` | `uvx --prerelease=allow ...` | Command to spawn MCP server |
| `GITHUB_TOKEN` | - | GitHub token for MCP server |
| `AZURE_AI_PROJECT_ENDPOINT` | - | Azure AI project endpoint |
| `AZURE_SUBSCRIPTION_ID` | - | Azure subscription ID |
| `AZURE_TENANT_ID` | - | Azure tenant ID |
| `AZURE_CLIENT_ID` | - | Azure service principal client ID |
| `AZURE_CLIENT_SECRET` | - | Azure service principal secret |

All Azure environment variables are passed through to the MCP server subprocess.

## Building

```bash
docker build -t azure-ai-foundry-mcp:latest .
```

## Running Standalone

```bash
docker run -p 8081:8081 \
  -e GITHUB_TOKEN=your_token \
  -e AZURE_AI_PROJECT_ENDPOINT=your_endpoint \
  -e AZURE_SUBSCRIPTION_ID=your_sub_id \
  -e AZURE_TENANT_ID=your_tenant_id \
  -e AZURE_CLIENT_ID=your_client_id \
  -e AZURE_CLIENT_SECRET=your_secret \
  azure-ai-foundry-mcp:latest
```

## Running with Docker Compose

See `../docker-compose.mcp.yml`

```bash
cd /home/azureuser/4hosts/four-hosts-app/backend
docker compose -f docker-compose.mcp.yml up -d
```

## Testing

```bash
# Health check
curl http://localhost:8081/health

# List tools
curl -X POST http://localhost:8081/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools.list",
    "params": {},
    "id": "test_1"
  }'
```

## Logs

```bash
docker logs azure-ai-foundry-mcp -f
```

## Troubleshooting

### Bridge starts but MCP server fails
Check that all Azure credentials are set correctly:
```bash
docker exec azure-ai-foundry-mcp env | grep AZURE
```

### Connection refused
Ensure the container is running and healthy:
```bash
docker ps | grep azure-ai-foundry-mcp
curl http://localhost:8081/health
```

### Slow responses
Check metrics to see if requests are timing out:
```bash
curl http://localhost:8081/metrics
```

## Development

To run locally without Docker:

```bash
cd mcp-http-bridge
pip install -r requirements.txt

# Set environment variables
export AZURE_AI_PROJECT_ENDPOINT=...
export AZURE_CLIENT_ID=...
# etc.

# Run bridge
python bridge_server.py
```

## Implementation Notes

- Uses asyncio subprocess management for the MCP server
- Matches responses to requests using JSON-RPC id field
- Handles concurrent requests safely
- Logs stderr from MCP server for debugging
- Graceful shutdown with SIGTERM handling