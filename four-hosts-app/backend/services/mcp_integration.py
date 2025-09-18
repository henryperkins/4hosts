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
from datetime import datetime, timezone
from services.websocket_service import WSMessage, WSEventType, progress_tracker  # type: ignore

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
        # Cache of discovered tools per server so callers can aggregate without
        # re-querying remote endpoints.
        self._discovered_tools: Dict[str, List[MCPToolDefinition]] = {}
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def register_server(self, server: MCPServer):
        """Register a remote MCP server"""
        self.servers[server.name] = server
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
            logger.debug(f"Error discovering tools from {server_name}: {response.error}")
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
        # Cache discovered tools for aggregation later
        self._discovered_tools[server_name] = tools
        
        return tools
    
    async def execute_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute a tool call through the appropriate MCP server"""
        handler = self._tool_handlers.get(tool_name)
        if not handler:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        return await handler(parameters)
    
    async def _execute_tool(self, server: MCPServer, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute a tool on a remote MCP server"""
        # Extract research_id for progress broadcasting if provided
        research_id = None
        try:
            if isinstance(parameters, dict):
                research_id = parameters.get("research_id")
        except Exception:
            research_id = None

        # Notify start of MCP tool execution
        if research_id:
            try:
                await progress_tracker.connection_manager.broadcast_to_research(
                    str(research_id),
                    WSMessage(
                        type=WSEventType.MCP_TOOL_EXECUTING,
                        data={
                            "research_id": str(research_id),
                            "server": server.name,
                            "tool": tool_name,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    ),
                )
            except Exception:
                pass

        request = MCPRequest(
            method="tools.execute",
            params={
                "name": tool_name,
                "parameters": parameters
            },
            id=f"exec_{tool_name}"
        )
        
        response = await self._send_request(server, request)
        # Notify completion of MCP tool execution
        if research_id:
            try:
                await progress_tracker.connection_manager.broadcast_to_research(
                    str(research_id),
                    WSMessage(
                        type=WSEventType.MCP_TOOL_COMPLETED,
                        data={
                            "research_id": str(research_id),
                            "server": server.name,
                            "tool": tool_name,
                            "status": "error" if response.error else "ok",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    ),
                )
            except Exception:
                pass
        
        if research_id and not response.error:
            research_key = str(research_id)
            try:
                state = progress_tracker.research_progress.get(research_key)
                current = int(state.get("mcp_tools_used", 0) or 0) + 1 if state else None
            except Exception:
                current = None
            if current is not None:
                try:
                    await progress_tracker.update_progress(
                        research_key,
                        mcp_tools_used=current,
                    )
                except Exception:
                    pass

        if response.error:
            logger.warning(f"Tool execution error: {response.error}")
            raise Exception(f"MCP tool execution failed: {response.error}")
        
        return response.result
    
    async def _send_request(self, server: MCPServer, request: MCPRequest) -> MCPResponse:
        """Send a request to an MCP server"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
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
                data = await response.json()
                return MCPResponse(**data)
        except asyncio.TimeoutError:
            logger.debug(f"Timeout connecting to MCP server {server.name}")
            return MCPResponse(error={"code": -32000, "message": "Connection timeout"})
        except Exception as e:
            logger.debug(f"Error connecting to MCP server {server.name}: {e}")
            return MCPResponse(error={"code": -32001, "message": str(e)})
    
    def get_all_tools(self) -> List[MCPToolDefinition]:
        """Aggregate all cached tool definitions discovered so far.

        This does not perform network discovery; callers should invoke
        ``discover_tools(server_name)`` at least once per server to populate
        the cache. The returned list is a flattened snapshot across servers.
        """
        tools: List[MCPToolDefinition] = []
        # Prefer cached discoveries
        for server_name, items in self._discovered_tools.items():
            if not items:
                continue
            tools.extend(items)

        # If nothing has been discovered yet, fall back to constructing
        # empty function descriptors for known servers so callers have a
        # deterministic (though empty) response without raising.
        if not tools and self.servers:
            return []
        return tools

    def get_responses_mcp_tools(self, require_approval: str = "never") -> List[Dict[str, Any]]:
        """Return a list of Responses API MCP tool descriptors for all registered servers.

        Shape matches OpenAI Responses MCP tool objects:
          {"type":"mcp", "server_label": name, "server_url": url, "require_approval": "never"}
        """
        tools: List[Dict[str, Any]] = []
        for name, srv in self.servers.items():
            tools.append({
                "type": "mcp",
                "server_label": name,
                "server_url": srv.url,
                "require_approval": require_approval,
            })
        return tools


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
