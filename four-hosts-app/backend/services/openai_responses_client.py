"""
OpenAI Responses API Client
---------------------------
Provides support for the OpenAI Responses API, including deep research models,
web search, and background mode for long-running tasks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from datetime import datetime

import httpx
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Logging
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
#  Enums / Constants
# ────────────────────────────────────────────────────────────
class ResponseStatus(Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SearchContextSize(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class WebSearchAction(Enum):
    SEARCH = "search"
    OPEN_PAGE = "open_page"
    FIND_IN_PAGE = "find_in_page"


@dataclass
class WebSearchTool:
    """Configuration for web search tool"""
    type: str = "web_search_preview"
    search_context_size: SearchContextSize = SearchContextSize.MEDIUM
    user_location: Optional[Dict[str, str]] = None


@dataclass
class CodeInterpreterTool:
    """Configuration for code interpreter tool"""
    type: str = "code_interpreter"
    container: Dict[str, Any] = None

    def __post_init__(self):
        if self.container is None:
            self.container = {"type": "auto"}


@dataclass
class MCPTool:
    """Configuration for Model Context Protocol tool"""
    type: str = "mcp"
    server_label: str = ""
    server_url: str = ""
    require_approval: str = "never"


@dataclass
class ResponsesAPIRequest:
    """Request structure for Responses API"""
    model: str
    input: Union[str, List[Dict[str, Any]]]
    tools: Optional[List[Dict[str, Any]]] = None
    background: bool = False
    stream: bool = False
    reasoning: Optional[Dict[str, str]] = None
    max_tool_calls: Optional[int] = None
    instructions: Optional[str] = None
    store: bool = True


@dataclass
class Citation:
    """Citation from web search results"""
    url: str
    title: str
    start_index: int
    end_index: int


@dataclass
class ResponseOutput:
    """Output from Responses API"""
    id: str
    type: str
    status: Optional[str] = None
    content: Optional[Any] = None
    role: Optional[str] = None
    action: Optional[Dict[str, Any]] = None
    sequence_number: Optional[int] = None


# ────────────────────────────────────────────────────────────
#  OpenAI Responses Client
# ────────────────────────────────────────────────────────────
class OpenAIResponsesClient:
    """Client for OpenAI Responses API with deep research support"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = AsyncOpenAI(api_key=self.api_key, timeout=3600)
        self.base_url = "https://api.openai.com/v1"
        
    # ─────────── Core API Methods ───────────
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    )
    async def create_response(
        self,
        model: str,
        input: Union[str, List[Dict[str, Any]]],
        *,
        tools: Optional[List[Union[WebSearchTool, CodeInterpreterTool, MCPTool, Dict]]] = None,
        background: bool = False,
        stream: bool = False,
        reasoning: Optional[Dict[str, str]] = None,
        max_tool_calls: Optional[int] = None,
        instructions: Optional[str] = None,
        store: bool = True,
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        Create a response using the Responses API.
        
        Args:
            model: Model to use (e.g., "o3-deep-research", "gpt-4.1")
            input: Text input or conversation messages
            tools: List of tools to enable
            background: Whether to run in background mode
            stream: Whether to stream the response
            reasoning: Reasoning configuration (e.g., {"summary": "auto"})
            max_tool_calls: Maximum number of tool calls to make
            instructions: System instructions
            store: Whether to store the response
            
        Returns:
            Response object or async iterator for streaming
        """
        # Convert tool objects to dicts
        tool_dicts = []
        if tools:
            for tool in tools:
                if isinstance(tool, (WebSearchTool, CodeInterpreterTool, MCPTool)):
                    tool_dict = {"type": tool.type}
                    if isinstance(tool, WebSearchTool):
                        if tool.search_context_size != SearchContextSize.MEDIUM:
                            tool_dict["search_context_size"] = tool.search_context_size.value
                        if tool.user_location:
                            tool_dict["user_location"] = tool.user_location
                    elif isinstance(tool, CodeInterpreterTool):
                        tool_dict["container"] = tool.container
                    elif isinstance(tool, MCPTool):
                        tool_dict.update({
                            "server_label": tool.server_label,
                            "server_url": tool.server_url,
                            "require_approval": tool.require_approval
                        })
                    tool_dicts.append(tool_dict)
                else:
                    tool_dicts.append(tool)
        
        # Build request
        request_data = {
            "model": model,
            "input": input,
            "store": store,
        }
        
        if tool_dicts:
            request_data["tools"] = tool_dicts
        if background:
            request_data["background"] = background
        if stream:
            request_data["stream"] = stream
        if reasoning:
            request_data["reasoning"] = reasoning
        if max_tool_calls is not None:
            request_data["max_tool_calls"] = max_tool_calls
        if instructions:
            request_data["instructions"] = instructions
        
        # Make request
        async with httpx.AsyncClient(timeout=3600) as client:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            if stream:
                response = await client.post(
                    f"{self.base_url}/responses",
                    headers=headers,
                    json=request_data,
                    timeout=3600,
                )
                response.raise_for_status()
                return self._stream_response(response)
            else:
                response = await client.post(
                    f"{self.base_url}/responses",
                    headers=headers,
                    json=request_data,
                    timeout=3600,
                )
                response.raise_for_status()
                return response.json()
    
    async def retrieve_response(self, response_id: str) -> Dict[str, Any]:
        """Retrieve a response by ID (for background mode)"""
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }
            response = await client.get(
                f"{self.base_url}/responses/{response_id}",
                headers=headers,
            )
            response.raise_for_status()
            return response.json()
    
    async def cancel_response(self, response_id: str) -> Dict[str, Any]:
        """Cancel a background response"""
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }
            response = await client.post(
                f"{self.base_url}/responses/{response_id}/cancel",
                headers=headers,
            )
            response.raise_for_status()
            return response.json()
    
    async def stream_response(
        self, response_id: str, starting_after: Optional[int] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream events from a background response"""
        url = f"{self.base_url}/responses/{response_id}?stream=true"
        if starting_after is not None:
            url += f"&starting_after={starting_after}"
        
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }
            async with client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        yield json.loads(data)
    
    # ─────────── Deep Research Methods ───────────
    async def deep_research(
        self,
        query: str,
        *,
        use_web_search: bool = True,
        web_search_config: Optional[WebSearchTool] = None,
        use_code_interpreter: bool = False,
        mcp_servers: Optional[List[MCPTool]] = None,
        system_prompt: Optional[str] = None,
        max_tool_calls: Optional[int] = None,
        background: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a deep research task using o3-deep-research model.
        
        Args:
            query: Research query
            use_web_search: Whether to enable web search
            web_search_config: Custom web search configuration
            use_code_interpreter: Whether to enable code interpreter
            mcp_servers: List of MCP servers to connect
            system_prompt: Custom system prompt
            max_tool_calls: Maximum number of tool calls
            background: Whether to run in background mode
            
        Returns:
            Response object with research results
        """
        tools = []
        
        if use_web_search:
            tools.append(web_search_config or WebSearchTool())
        
        if use_code_interpreter:
            tools.append(CodeInterpreterTool())
        
        if mcp_servers:
            tools.extend(mcp_servers)
        
        # Build input
        input_messages = []
        
        if system_prompt:
            input_messages.append({
                "role": "developer",
                "content": [{"type": "input_text", "text": system_prompt}]
            })
        
        input_messages.append({
            "role": "user",
            "content": [{"type": "input_text", "text": query}]
        })
        
        return await self.create_response(
            model="o3-deep-research",
            input=input_messages,
            tools=tools,
            background=background,
            reasoning={"summary": "auto"},
            max_tool_calls=max_tool_calls,
        )
    
    async def wait_for_response(
        self,
        response_id: str,
        poll_interval: int = 2,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Wait for a background response to complete.
        
        Args:
            response_id: Response ID to wait for
            poll_interval: Seconds between polls
            timeout: Maximum seconds to wait
            
        Returns:
            Completed response object
        """
        start_time = asyncio.get_event_loop().time()
        
        while True:
            response = await self.retrieve_response(response_id)
            
            if response["status"] not in ["queued", "in_progress"]:
                return response
            
            if timeout and (asyncio.get_event_loop().time() - start_time) > timeout:
                raise TimeoutError(f"Response {response_id} did not complete within {timeout} seconds")
            
            await asyncio.sleep(poll_interval)
    
    # ─────────── Helper Methods ───────────
    def extract_final_text(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract the final text output from a response"""
        if "output" not in response:
            return None
        
        for item in reversed(response["output"]):
            if item["type"] == "message" and item.get("content"):
                for content in item["content"]:
                    if content["type"] == "output_text":
                        return content["text"]
        
        return None
    
    def extract_citations(self, response: Dict[str, Any]) -> List[Citation]:
        """Extract citations from a response"""
        citations = []
        
        if "output" not in response:
            return citations
        
        for item in response["output"]:
            if item["type"] == "message" and item.get("content"):
                for content in item["content"]:
                    if content["type"] == "output_text" and "annotations" in content:
                        for annotation in content["annotations"]:
                            if annotation.get("type") == "url_citation":
                                citations.append(Citation(
                                    url=annotation["url"],
                                    title=annotation["title"],
                                    start_index=annotation["start_index"],
                                    end_index=annotation["end_index"],
                                ))
        
        return citations
    
    def extract_web_search_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract web search tool calls from a response"""
        web_searches = []
        
        if "output" not in response:
            return web_searches
        
        for item in response["output"]:
            if item["type"] == "web_search_call":
                web_search = {
                    "id": item.get("id"),
                    "status": item.get("status"),
                }
                
                # Extract action details if available
                if "action" in item:
                    action = item["action"]
                    web_search["action_type"] = action.get("type")
                    web_search["query"] = action.get("query")
                    web_search["domains"] = action.get("domains", [])
                
                web_searches.append(web_search)
        
        return web_searches
    
    def extract_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool calls from a response"""
        tool_calls = []
        
        if "output" not in response:
            return tool_calls
        
        for item in response["output"]:
            if item["type"] in ["web_search_call", "code_interpreter_call", "mcp_call"]:
                tool_calls.append(item)
        
        return tool_calls
    
    async def _stream_response(self, response: httpx.Response) -> AsyncIterator[Dict[str, Any]]:
        """Stream response events"""
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                yield json.loads(data)


# ────────────────────────────────────────────────────────────
#  Singleton Instance
# ────────────────────────────────────────────────────────────
responses_client: Optional[OpenAIResponsesClient] = None


def get_responses_client() -> OpenAIResponsesClient:
    """Get or create the OpenAI Responses client singleton"""
    global responses_client
    if responses_client is None:
        responses_client = OpenAIResponsesClient()
    return responses_client