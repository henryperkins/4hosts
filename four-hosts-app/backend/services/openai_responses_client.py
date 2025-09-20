"""
OpenAI Responses API Client
---------------------------
Simplified client for OpenAI Responses API, providing only the methods
actually used by the application.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx
from openai import AsyncOpenAI

# Logging
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
#  OpenAI Responses Client
# ────────────────────────────────────────────────────────────
class OpenAIResponsesClient:
    """Client for OpenAI Responses API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize for OpenAI Cloud or Azure OpenAI.

        Preference order:
        1) Azure OpenAI when AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are set
        2) OpenAI Cloud when OPENAI_API_KEY is set
        """
        # Check for Azure configuration
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "preview")

        self.is_azure = bool(azure_key and azure_endpoint)

        if self.is_azure:
            self.api_key = azure_key
            endpoint = azure_endpoint.rstrip("/")
            self.base_url = f"{endpoint}/openai/v1"
            self.azure_api_version = azure_api_version
            # Keep an SDK client for compatibility (not actively used)
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=3600,
            )
            logger.info(f"✓ Azure Responses API client initialized (endpoint: {self.base_url})")
        else:
            # OpenAI Cloud
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "OpenAI API key is required (or set AZURE_OPENAI_API_KEY/ENDPOINT)"
                )
            self.api_key = key
            self.client = AsyncOpenAI(api_key=self.api_key, timeout=3600)
            self.base_url = "https://api.openai.com/v1"
            logger.info("✓ OpenAI Responses API client initialized")

    # ────────────────────────────────────────────────────────────
    #  Core API Methods (Actually Used)
    # ────────────────────────────────────────────────────────────

    async def create_response(
        self,
        model: str,
        input: Union[str, List[Dict[str, Any]]],
        *,
        tools: Optional[List[Any]] = None,
        background: bool = False,
        stream: bool = False,
        reasoning: Optional[Dict[str, str]] = None,
        max_tool_calls: Optional[int] = None,
        instructions: Optional[str] = None,
        store: bool = True,
        previous_response_id: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        text: Optional[Dict[str, Any]] = None,
        **kwargs  # Accept additional kwargs for forward compatibility
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        Create a response using the Responses API.

        This is the primary method used by routes/responses.py, background_llm.py,
        and indirectly via llm_client.py wrappers.
        """
        # Build request data
        request_data: Dict[str, Any] = {
            "model": model or os.getenv("AZURE_OPENAI_DEPLOYMENT", "o3"),
            "input": input,
            "store": store,
        }

        # Background mode requires stored responses
        if background and not store:
            raise ValueError("Background mode requires store=True")

        # Add optional parameters
        if tools:
            # Filter out web_search tools on Azure (not supported)
            if self.is_azure:
                filtered_tools = [
                    t for t in tools
                    if not (isinstance(t, dict) and str(t.get("type", "")).startswith("web_search"))
                ]
                if len(filtered_tools) < len(tools):
                    logger.warning("Removed web_search tools on Azure (not supported)")
                request_data["tools"] = filtered_tools
            else:
                request_data["tools"] = tools

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
        if previous_response_id:
            request_data["previous_response_id"] = previous_response_id
        if max_output_tokens is not None:
            request_data["max_output_tokens"] = max_output_tokens

        # Azure doesn't support response_format in preview
        if response_format and not self.is_azure:
            request_data["response_format"] = response_format
        if text:
            request_data["text"] = text

        # Make the HTTP request
        async with httpx.AsyncClient(timeout=3600) as client:
            if self.is_azure:
                headers = {
                    "api-key": self.api_key,
                    "Content-Type": "application/json",
                }
                url = f"{self.base_url}/responses?api-version={self.azure_api_version}"
            else:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                url = f"{self.base_url}/responses"

            if stream:
                # Return async generator for streaming
                return self._stream_response_sse(url, headers, request_data)
            else:
                response = await client.post(
                    url,
                    headers=headers,
                    json=request_data,
                    timeout=3600,
                )
                response.raise_for_status()
                return response.json()

    async def retrieve_response(
        self,
        response_id: str,
        *,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve a response by ID.

        Used by routes/responses.py and background_llm.py for polling.
        """
        async with httpx.AsyncClient() as client:
            base = f"{self.base_url}/responses/{response_id}"
            if self.is_azure:
                headers = {"api-key": self.api_key}
                qp = [("api-version", self.azure_api_version)]
            else:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                qp = []

            if include:
                qp.extend([("include[]", item) for item in include])

            url = f"{base}?{httpx.QueryParams(qp)}" if qp else base

            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()

    async def cancel_response(self, response_id: str) -> Dict[str, Any]:
        """
        Cancel a background response.

        Used by routes/responses.py and background_llm.py.
        """
        async with httpx.AsyncClient() as client:
            if self.is_azure:
                headers = {"api-key": self.api_key}
                url = f"{self.base_url}/responses/{response_id}/cancel?api-version={self.azure_api_version}"
            else:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                url = f"{self.base_url}/responses/{response_id}/cancel"

            response = await client.post(url, headers=headers)
            response.raise_for_status()
            return response.json()

    async def stream_response(
        self,
        response_id: str,
        starting_after: Optional[int] = None,
        include: Optional[List[str]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream events from a background response.

        Used by routes/responses.py for SSE streaming.
        """
        if self.is_azure:
            qp: List[tuple[str, str]] = [("stream", "true"), ("api-version", self.azure_api_version)]
            if starting_after is not None:
                qp.append(("starting_after", str(starting_after)))
            if include:
                qp.extend(("include[]", item) for item in include)
            url = f"{self.base_url}/responses/{response_id}?{httpx.QueryParams(qp)}"
            headers = {
                "api-key": self.api_key,
                "Accept": "text/event-stream",
            }
        else:
            qp = [("stream", "true")]
            if starting_after is not None:
                qp.append(("starting_after", str(starting_after)))
            if include:
                qp.extend(("include[]", item) for item in include)
            url = f"{self.base_url}/responses/{response_id}?{httpx.QueryParams(qp)}"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "text/event-stream",
            }

        async with httpx.AsyncClient() as client:
            async with client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        yield json.loads(data)

    # ────────────────────────────────────────────────────────────
    #  Private Helper Methods
    # ────────────────────────────────────────────────────────────

    async def _stream_response_sse(
        self,
        url: str,
        headers: Dict[str, str],
        request_data: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """Generate SSE stream for create_response with stream=True."""
        stream_headers = dict(headers)
        stream_headers["Accept"] = "text/event-stream"

        async with httpx.AsyncClient(timeout=3600) as client:
            async with client.stream(
                "POST",
                url,
                headers=stream_headers,
                json=request_data,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        yield json.loads(data)

# ────────────────────────────────────────────────────────────
#  Singleton Instance
# ────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_responses_client() -> OpenAIResponsesClient:
    """Return a cached singleton instance of the responses client."""
    return OpenAIResponsesClient()

# ────────────────────────────────────────────────────────────
#  Tool Classes (for type hints only - kept for compatibility)
# ────────────────────────────────────────────────────────────
# These are imported by llm_client.py for type hints, but never instantiated
# in production code. Keeping minimal versions for backward compatibility.

class WebSearchTool:
    """Web search tool configuration (for type hints)."""
    def __init__(self, search_context_size: str = "medium", user_location: Optional[Dict] = None):
        self.type = "web_search_preview"
        self.search_context_size = search_context_size
        self.user_location = user_location

class CodeInterpreterTool:
    """Code interpreter tool configuration (for type hints)."""
    def __init__(self, container: Optional[Dict] = None):
        self.type = "code_interpreter"
        self.container = container or {"type": "auto"}

class MCPTool:
    """MCP tool configuration (for type hints)."""
    def __init__(self, server_label: str = "", server_url: str = "", require_approval: str = "never"):
        self.type = "mcp"
        self.server_label = server_label
        self.server_url = server_url
        self.require_approval = require_approval

# Note: Citation class is imported by test but not used in production
class Citation:
    """Citation from web search results (for compatibility)."""
    def __init__(self, url: str, title: str, start_index: int, end_index: int):
        self.url = url
        self.title = title
        self.start_index = start_index
        self.end_index = end_index
