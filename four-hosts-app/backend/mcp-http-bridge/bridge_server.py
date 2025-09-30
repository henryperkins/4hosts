#!/usr/bin/env python3
"""
HTTP-to-Stdio Bridge for Azure AI Foundry MCP Server

This bridge allows HTTP clients to communicate with stdio-based MCP servers.
It spawns the MCP server as a subprocess and translates between HTTP and stdio.
"""

import asyncio
import json
import logging
import os
import sys
from typing import Optional, Dict, Any
from datetime import datetime

from aiohttp import web
import structlog

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = structlog.get_logger(__name__)


class StdioMCPBridge:
    """Bridge between HTTP requests and stdio MCP server"""

    def __init__(self, command: list[str]):
        """
        Initialize the bridge with MCP server command

        Args:
            command: Command to spawn MCP server (e.g., ['uvx', 'run-azure-ai-foundry-mcp'])
        """
        self.command = command
        self.process: Optional[asyncio.subprocess.Process] = None
        self.read_task: Optional[asyncio.Task] = None
        self.response_queue: Dict[str, asyncio.Queue] = {}
        self.running = False
        self.request_counter = 0
        self.initialized = False
        self.server_info: Optional[Dict[str, Any]] = None

    async def start(self):
        """Start the MCP server subprocess"""
        try:
            logger.info("Starting MCP server subprocess", command=self.command)

            self.process = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy()
            )

            self.running = True

            # Start reading stdout in background
            self.read_task = asyncio.create_task(self._read_stdout())

            # Start stderr logger
            asyncio.create_task(self._log_stderr())

            logger.info("MCP server subprocess started", pid=self.process.pid)

            # Auto-initialize MCP session
            await self._initialize()

        except Exception as e:
            logger.error("Failed to start MCP server", error=str(e))
            raise

    async def stop(self):
        """Stop the MCP server subprocess"""
        self.running = False

        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()

            logger.info("MCP server subprocess stopped")

        if self.read_task:
            self.read_task.cancel()
            try:
                await self.read_task
            except asyncio.CancelledError:
                pass

    async def _read_stdout(self):
        """Read responses from MCP server stdout"""
        if not self.process or not self.process.stdout:
            return

        while self.running:
            try:
                line = await self.process.stdout.readline()
                if not line:
                    break

                # Parse JSON response
                try:
                    response = json.loads(line.decode('utf-8'))
                    request_id = response.get('id')

                    if request_id and request_id in self.response_queue:
                        await self.response_queue[request_id].put(response)
                        logger.debug("Received response", request_id=request_id)
                    else:
                        logger.warning("Received response for unknown request", request_id=request_id)

                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse MCP response", error=str(e), line=line)

            except Exception as e:
                logger.error("Error reading stdout", error=str(e))
                if not self.running:
                    break

    async def _log_stderr(self):
        """Log stderr output from MCP server"""
        if not self.process or not self.process.stderr:
            return

        while self.running:
            try:
                line = await self.process.stderr.readline()
                if not line:
                    break

                log_line = line.decode('utf-8').strip()
                if log_line:
                    logger.info("MCP stderr", message=log_line)

            except Exception as e:
                logger.error("Error reading stderr", error=str(e))
                if not self.running:
                    break

    async def _initialize(self):
        """Initialize MCP session with handshake"""
        try:
            logger.info("Initializing MCP session")

            # Send initialize request
            init_request = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "roots": {
                            "listChanged": True
                        },
                        "sampling": {}
                    },
                    "clientInfo": {
                        "name": "four-hosts-mcp-bridge",
                        "version": "1.0.0"
                    }
                },
                "id": "init"
            }

            # Send initialization
            response = await self.send_request(init_request, timeout=10.0)

            if 'error' in response:
                logger.error("MCP initialization failed", error=response['error'])
                raise RuntimeError(f"MCP initialization failed: {response['error']}")

            if 'result' in response:
                self.server_info = response['result']
                logger.info("MCP initialization successful",
                           server_name=self.server_info.get('serverInfo', {}).get('name'),
                           protocol_version=self.server_info.get('protocolVersion'))

                # Send initialized notification to complete handshake
                initialized_notification = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {}
                }

                # Notifications don't have responses, just send it
                if self.process and self.process.stdin:
                    notification_json = json.dumps(initialized_notification) + '\n'
                    self.process.stdin.write(notification_json.encode('utf-8'))
                    await self.process.stdin.drain()
                    logger.info("Sent initialized notification")

                self.initialized = True
                logger.info("MCP session fully initialized and ready")
            else:
                raise RuntimeError("MCP initialization returned no result")

        except Exception as e:
            logger.error("Failed to initialize MCP session", error=str(e))
            raise

    async def send_request(self, request: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        """
        Send request to MCP server and wait for response

        Args:
            request: MCP request dict
            timeout: Timeout in seconds

        Returns:
            MCP response dict
        """
        if not self.process or not self.process.stdin:
            raise RuntimeError("MCP server not running")

        # Ensure request has an ID
        if 'id' not in request:
            self.request_counter += 1
            request['id'] = f"req_{self.request_counter}"

        request_id = request['id']

        # Create response queue for this request
        self.response_queue[request_id] = asyncio.Queue()

        try:
            # Send request to stdin
            request_json = json.dumps(request) + '\n'
            self.process.stdin.write(request_json.encode('utf-8'))
            await self.process.stdin.drain()

            logger.debug("Sent request to MCP server", request_id=request_id, method=request.get('method'))

            # Wait for response
            try:
                response = await asyncio.wait_for(
                    self.response_queue[request_id].get(),
                    timeout=timeout
                )
                return response
            except asyncio.TimeoutError:
                logger.error("Request timeout", request_id=request_id)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32000,
                        "message": "Request timeout"
                    }
                }

        finally:
            # Clean up response queue
            if request_id in self.response_queue:
                del self.response_queue[request_id]


class MCPBridgeServer:
    """HTTP server that bridges to stdio MCP server"""

    def __init__(self, bridge: StdioMCPBridge, host: str = '0.0.0.0', port: int = 8081):
        self.bridge = bridge
        self.host = host
        self.port = port
        self.app = web.Application()
        self.setup_routes()
        self.start_time = datetime.utcnow()
        self.request_count = 0

    def setup_routes(self):
        """Setup HTTP routes"""
        self.app.router.add_post('/mcp', self.handle_mcp_request)
        self.app.router.add_get('/health', self.handle_health)
        self.app.router.add_get('/metrics', self.handle_metrics)

    async def handle_mcp_request(self, request: web.Request) -> web.Response:
        """Handle MCP request via HTTP POST"""
        try:
            self.request_count += 1

            # Parse request body
            try:
                mcp_request = await request.json()
            except json.JSONDecodeError:
                return web.json_response({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32700,
                        "message": "Parse error"
                    }
                }, status=400)

            # Check if MCP session is initialized (unless this is an initialize request)
            if mcp_request.get('method') != 'initialize' and not self.bridge.initialized:
                return web.json_response({
                    "jsonrpc": "2.0",
                    "id": mcp_request.get('id'),
                    "error": {
                        "code": -32002,
                        "message": "MCP session not initialized. Bridge initializes automatically on startup."
                    }
                }, status=503)

            logger.info("Received MCP request",
                       method=mcp_request.get('method'),
                       request_id=mcp_request.get('id'))

            # Forward to MCP server
            mcp_response = await self.bridge.send_request(mcp_request)

            # Return response
            status = 200 if 'result' in mcp_response else 500
            return web.json_response(mcp_response, status=status)

        except Exception as e:
            logger.error("Error handling MCP request", error=str(e))
            return web.json_response({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }, status=500)

    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint"""
        is_running = self.bridge.running and self.bridge.process and self.bridge.process.returncode is None
        is_healthy = is_running and self.bridge.initialized

        return web.json_response({
            "status": "healthy" if is_healthy else "unhealthy",
            "mcp_server_running": is_running,
            "mcp_initialized": self.bridge.initialized,
            "server_info": self.bridge.server_info.get('serverInfo') if self.bridge.server_info else None,
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "requests_handled": self.request_count
        }, status=200 if is_healthy else 503)

    async def handle_metrics(self, request: web.Request) -> web.Response:
        """Metrics endpoint"""
        return web.json_response({
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "total_requests": self.request_count,
            "mcp_server_pid": self.bridge.process.pid if self.bridge.process else None,
            "active_requests": len(self.bridge.response_queue)
        })

    async def start_background_tasks(self, app):
        """Start bridge when app starts"""
        await self.bridge.start()

    async def cleanup_background_tasks(self, app):
        """Stop bridge when app stops"""
        await self.bridge.stop()

    def run(self):
        """Run the HTTP server"""
        self.app.on_startup.append(self.start_background_tasks)
        self.app.on_cleanup.append(self.cleanup_background_tasks)

        logger.info("Starting HTTP bridge server", host=self.host, port=self.port)
        web.run_app(self.app, host=self.host, port=self.port)


def main():
    """Main entry point"""
    # Get MCP server command from environment or use default
    mcp_command_str = os.getenv('MCP_COMMAND', 'uvx --prerelease=allow --from git+https://github.com/azure-ai-foundry/mcp-foundry.git run-azure-ai-foundry-mcp')
    mcp_command = mcp_command_str.split()

    host = os.getenv('BRIDGE_HOST', '0.0.0.0')
    port = int(os.getenv('BRIDGE_PORT', '8081'))

    logger.info("Initializing MCP HTTP Bridge",
                command=mcp_command,
                host=host,
                port=port)

    # Create bridge and server
    bridge = StdioMCPBridge(mcp_command)
    server = MCPBridgeServer(bridge, host=host, port=port)

    # Run server
    server.run()


if __name__ == '__main__':
    main()