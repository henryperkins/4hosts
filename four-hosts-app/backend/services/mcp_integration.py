"""
MCP (Model Context Protocol) Server Integration
Enables the Four Hosts system to connect to remote MCP servers for extended capabilities
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import aiohttp
from pydantic import BaseModel, HttpUrl

logger = logging.getLogger(__name__)


class MCPCapability(str, Enum):
    """Available MCP server capabilities"""
    SEARCH = "search"
    DATABASE = "database"
    FILESYSTEM = "filesystem"
    COMPUTATION = "computation"
    CUSTOM = "custom"


@dataclass
class MCPServer:
    """Remote MCP server configuration"""
    name: str
    url: str
    capabilities: List[MCPCapability]
    auth_token: Optional[str] = None
    timeout: int = 30


class MCPToolDefinition(BaseModel):
    """Tool definition compatible with Azure OpenAI"""
    type: str = "function"
    function: Dict[str, Any]


class MCPRequest(BaseModel):
    """Request to MCP server"""
    method: str
    params: Dict[str, Any]
    id: Optional[str] = None


class MCPResponse(BaseModel):
    """Response from MCP server"""
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[str] = None


class MCPIntegration:
    """Manages connections to remote MCP servers"""
    
    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self._tool_handlers: Dict[str, Callable] = {}
        self._server_health: Dict[str, bool] = {}
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def register_server(self, server: MCPServer):
        """Register a remote MCP server"""
        self.servers[server.name] = server
        self._server_health[server.name] = False  # Initially mark as unhealthy
        logger.info(f"Registered MCP server: {server.name} with capabilities: {server.capabilities}")
    
    async def discover_tools(self, server_name: str) -> List[MCPToolDefinition]:
        """Discover available tools from an MCP server"""
        server = self.servers.get(server_name)
        if not server:
            raise ValueError(f"Unknown MCP server: {server_name}")
        
        request = MCPRequest(
            method="tools.list",
            params={},
            id=f"discover_{server_name}"
        )
        
        response = await self._send_request(server, request)
        
        if response.error:
            logger.error(f"Error discovering tools from {server_name}: {response.error}")
            return []
        
        # Convert MCP tools to Azure OpenAI format
        tools = []
        for tool_data in response.result.get("tools", []):
            tool_def = MCPToolDefinition(
                function={
                    "name": f"{server_name}_{tool_data['name']}",
                    "description": tool_data.get("description", ""),
                    "parameters": tool_data.get("parameters", {})
                }
            )
            tools.append(tool_def)
            
            # Register handler for this tool
            self._tool_handlers[tool_def.function["name"]] = lambda params, t=tool_data, s=server: self._execute_tool(s, t["name"], params)
        
        return tools
    
    async def execute_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute a tool call through the appropriate MCP server"""
        handler = self._tool_handlers.get(tool_name)
        if not handler:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        return await handler(parameters)
    
    async def _execute_tool(self, server: MCPServer, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute a tool on a remote MCP server"""
        request = MCPRequest(
            method="tools.execute",
            params={
                "name": tool_name,
                "parameters": parameters
            },
            id=f"exec_{tool_name}"
        )
        
        response = await self._send_request(server, request)
        
        if response.error:
            logger.error(f"Tool execution error: {response.error}")
            raise Exception(f"MCP tool execution failed: {response.error}")
        
        return response.result
    
    async def _send_request(self, server: MCPServer, request: MCPRequest) -> MCPResponse:
        """Send a request to an MCP server"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if server.auth_token:
            headers["Authorization"] = f"Bearer {server.auth_token}"
        
        try:
            async with self.session.post(
                server.url,
                json=request.dict(),
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=server.timeout)
            ) as response:
                # Check content type before trying to parse JSON
                content_type = response.headers.get('content-type', '')
                if 'application/json' not in content_type:
                    text_content = await response.text()
                    logger.error(f"Unexpected content type from {server.name}: {content_type}. Content: {text_content[:200]}")
                    return MCPResponse(error={"code": -32002, "message": f"Unexpected content type: {content_type}. Expected JSON but got {content_type}"})
                
                try:
                    data = await response.json()
                    return MCPResponse(**data)
                except json.JSONDecodeError as e:
                    text_content = await response.text()
                    logger.error(f"Invalid JSON from {server.name}: {e}. Content: {text_content[:200]}")
                    return MCPResponse(error={"code": -32003, "message": f"Invalid JSON response: {str(e)}"})
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout connecting to MCP server {server.name}")
            return MCPResponse(error={"code": -32000, "message": "Connection timeout"})
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error connecting to MCP server {server.name}: {e}")
            return MCPResponse(error={"code": -32001, "message": f"HTTP error: {str(e)}"})
        except Exception as e:
            logger.error(f"Error connecting to MCP server {server.name}: {e}")
            return MCPResponse(error={"code": -32001, "message": str(e)})
    
    def get_all_tools(self) -> List[MCPToolDefinition]:
        """Get all registered tools from all servers"""
        all_tools = []
        for server_name in self.servers:
            # Note: This is synchronous, you'd need to call discover_tools first
            # In practice, you'd cache discovered tools
            pass
        return all_tools
    
    async def check_server_health(self, server_name: str) -> bool:
        """Check if an MCP server is healthy and responding"""
        server = self.servers.get(server_name)
        if not server:
            return False
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            # Try a simple health check endpoint first
            health_url = f"{server.url}/health"
            headers = {}
            if server.auth_token:
                headers["Authorization"] = f"Bearer {server.auth_token}"
            
            async with self.session.get(
                health_url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    self._server_health[server_name] = True
                    return True
        except:
            # If health endpoint doesn't exist, try a simple ping/echo request
            pass
        
        # Try a simple MCP request as fallback
        try:
            request = MCPRequest(
                method="ping",
                params={},
                id="health_check"
            )
            response = await self._send_request(server, request)
            
            # If we get any response (even an error for unknown method), server is alive
            self._server_health[server_name] = not (response.error and response.error.get("code") == -32001)
            return self._server_health[server_name]
            
        except Exception as e:
            logger.warning(f"Health check failed for {server_name}: {e}")
            self._server_health[server_name] = False
            return False
    
    def is_server_healthy(self, server_name: str) -> bool:
        """Get cached health status of a server"""
        return self._server_health.get(server_name, False)


# Global MCP integration instance
mcp_integration = MCPIntegration()


# Example configuration for common MCP servers
def configure_default_servers():
    """Configure default MCP servers based on environment"""
    import os
    
    # Example: Filesystem MCP server
    if os.getenv("MCP_FILESYSTEM_URL"):
        mcp_integration.register_server(MCPServer(
            name="filesystem",
            url=os.getenv("MCP_FILESYSTEM_URL"),
            capabilities=[MCPCapability.FILESYSTEM],
            auth_token=os.getenv("MCP_FILESYSTEM_TOKEN")
        ))
    
    # Example: Search MCP server
    if os.getenv("MCP_SEARCH_URL"):
        mcp_integration.register_server(MCPServer(
            name="search",
            url=os.getenv("MCP_SEARCH_URL"),
            capabilities=[MCPCapability.SEARCH],
            auth_token=os.getenv("MCP_SEARCH_TOKEN")
        ))
    
    # Example: Database MCP server
    if os.getenv("MCP_DATABASE_URL"):
        mcp_integration.register_server(MCPServer(
            name="database",
            url=os.getenv("MCP_DATABASE_URL"),
            capabilities=[MCPCapability.DATABASE],
            auth_token=os.getenv("MCP_DATABASE_TOKEN")
        ))