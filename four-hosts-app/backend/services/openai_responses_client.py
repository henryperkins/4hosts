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
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, cast, Dict, List, Optional, Union

import httpx
from openai import AsyncOpenAI
try:
    # Central default for output token limits (align with llm_client/config)
    from core.config import SYNTHESIS_BASE_TOKENS as DEFAULT_MAX_TOKENS
except Exception:  # pragma: no cover - fallback for safety
    DEFAULT_MAX_TOKENS = 8000
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

# Logging
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
#  Enums / Constants
# ────────────────────────────────────────────────────────────
class ResponseStatus(Enum):
    """Lifecycle states for a Responses API task."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SearchContextSize(Enum):
    """Web search breadth preset for the preview tool."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class WebSearchAction(Enum):
    """Supported actions for the web search tool."""

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
    container: Optional[Dict[str, Any]] = None

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
    response_format: Optional[Dict[str, Any]] = None
    text: Optional[Dict[str, Any]] = None


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
        """Initialise for OpenAI Cloud or Azure OpenAI.

        Preference order:
        1) Azure OpenAI when AZURE_OPENAI_API_KEY and
           AZURE_OPENAI_ENDPOINT are set
        2) OpenAI Cloud when OPENAI_API_KEY is set
        """

        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        # Prefer Azure preview by default for Responses API
        azure_api_version_env = os.getenv(
            "AZURE_OPENAI_API_VERSION", "preview"
        )

        self.is_azure = bool(azure_key and azure_endpoint)
        if self.is_azure:
            # These env vars are guaranteed when is_azure is True.
            self.api_key: str = cast(str, azure_key)
            endpoint = cast(str, azure_endpoint).rstrip("/")
            # Azure Responses API base path mirrors OpenAI's /openai/v1
            self.base_url = f"{endpoint}/openai/v1"
            # Use configured API version (default to env or preview)
            self.azure_api_version = azure_api_version_env
            # SDK client configured to hit Azure base_url as well
            # (not strictly required here)
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=f"{self.base_url}/",
                timeout=3600,
            )
        else:
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "OpenAI API key is required "
                    "(or set AZURE_OPENAI_API_KEY/ENDPOINT)"
                )
            self.api_key = key
            self.client = AsyncOpenAI(api_key=self.api_key, timeout=3600)
            self.base_url = "https://api.openai.com/v1"

    # ─────────── Core API Methods ───────────
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=(
            retry_if_exception_type(
                (httpx.TimeoutException, httpx.ConnectError)
            )
            | retry_if_exception(
                lambda e: isinstance(e, httpx.HTTPStatusError)
                and (e.response is not None)
                and (
                    e.response.status_code == 429
                    or 500 <= e.response.status_code <= 599
                )
            )
        ),
        reraise=True,
    )
    async def create_response(  # pylint: disable=redefined-builtin
        self,
        model: str,
        input: Union[str, List[Dict[str, Any]]],
        *,
        tools: Optional[
            List[Union[WebSearchTool, CodeInterpreterTool, MCPTool, Dict]]
        ] = None,
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
        tool_dicts: List[Dict[str, Any]] = []
        if tools:
            for tool in tools:
                if isinstance(
                    tool, (WebSearchTool, CodeInterpreterTool, MCPTool)
                ):
                    tool_dict: Dict[str, Any] = {"type": tool.type}
                    if isinstance(tool, WebSearchTool):
                        if (
                            tool.search_context_size
                            != SearchContextSize.MEDIUM
                        ):
                            tool_dict["search_context_size"] = (
                                tool.search_context_size.value
                            )
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

        # Azure limitation: Responses API currently does not support
        # web_search tools
        if self.is_azure and tool_dicts:
            filtered = [
                t
                for t in tool_dicts
                if not str(t.get("type", "")).startswith("web_search")
            ]
            if len(filtered) < len(tool_dicts):
                logger.warning(
                    "Removed web_search tool(s) on Azure "
                    "(Responses API unsupported)."
                )
            tool_dicts = filtered

        # Build request
        # Azure expects the deployment name in the "model" field.
        # Your deployments are named the same as models
        # (o3, gpt-4.1, gpt-4.1-mini),
        # so we should pass through the requested model directly.
        # Fall back to AZURE_OPENAI_DEPLOYMENT only when model is not provided.
        request_data: Dict[str, Any] = {
            "model": (model or os.getenv("AZURE_OPENAI_DEPLOYMENT", "o3")),
            "input": input,
            "store": store,
        }

        # Background mode requires stored responses for later
        # retrieval/streaming
        if background and not store:
            raise ValueError(
                "Background mode requires store=True for Responses API."
            )

        if tool_dicts:
            request_data["tools"] = tool_dicts
        if background:
            request_data["background"] = background
        if stream:
            request_data["stream"] = stream
        # Default reasoning behavior for o-series when not explicitly provided
        if reasoning is None:
            try:
                m = str(model or "").lower()
            except Exception:
                m = ""
            if m.startswith("o"):
                reasoning = {"summary": "auto"}
        if reasoning:
            request_data["reasoning"] = reasoning
        if max_tool_calls is not None:
            request_data["max_tool_calls"] = max_tool_calls
        if instructions:
            request_data["instructions"] = instructions
        if previous_response_id:
            request_data["previous_response_id"] = previous_response_id
        # Centralize max output tokens default
        if max_output_tokens is None:
            max_output_tokens = DEFAULT_MAX_TOKENS
        if max_output_tokens is not None:
            request_data["max_output_tokens"] = max_output_tokens
        if response_format:
            if self.is_azure:
                # Azure Responses preview currently rejects response_format.
                logger.debug(
                    "Azure Responses API does not support response_format – relying on instructions fallback"
                )
            else:
                request_data["response_format"] = response_format
        if text:
            request_data["text"] = text

        # Make request
        async with httpx.AsyncClient(timeout=3600) as client:
            if self.is_azure:
                headers: Dict[str, str] = {
                    "api-key": cast(str, self.api_key),
                    "Content-Type": "application/json",
                }
                base = f"{self.base_url}/responses"
                sep = "?" if "?" not in base else "&"
                url = f"{base}{sep}api-version={self.azure_api_version}"
            else:
                headers: Dict[str, str] = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                url = f"{self.base_url}/responses"

            if stream:
                # Ensure server-sent events (SSE) for streaming
                stream_headers: Dict[str, str] = dict(headers)
                stream_headers["Accept"] = "text/event-stream"

                async def _generator():
                    async with httpx.AsyncClient(timeout=3600) as _c:
                        async with _c.stream(
                            "POST",
                            url,
                            headers=stream_headers,
                            json=request_data,
                        ) as resp:
                            resp.raise_for_status()
                            async for line in resp.aiter_lines():
                                if line.startswith("data: "):
                                    data = line[6:]
                                    if data == "[DONE]":
                                        break
                                    yield json.loads(data)
                return _generator()
            else:
                response = await client.post(
                    url,
                    headers=headers,
                    json=request_data,
                    timeout=3600,
                )
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    try:
                        body = exc.response.text
                    except Exception:
                        body = "<unavailable>"
                    logger.error(
                        "Responses API request failed (%s): %s",
                        exc.response.status_code,
                        body,
                    )
                    raise
                return response.json()

    async def retrieve_response(
        self,
        response_id: str,
        *,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Retrieve a response by ID (for background mode).
        Optionally include extra fields (e.g., include=["reasoning"])."""
        async with httpx.AsyncClient() as client:
            if self.is_azure:
                headers: Dict[str, str] = {"api-key": cast(str, self.api_key)}
                base = f"{self.base_url}/responses/{response_id}"
                sep = "?" if "?" not in base else "&"
                url = f"{base}{sep}api-version={self.azure_api_version}"
                if include:
                    url += "".join(f"&include={item}" for item in include)
            else:
                headers: Dict[str, str] = {
                    "Authorization": f"Bearer {self.api_key}"
                }
                url = f"{self.base_url}/responses/{response_id}"
                if include:
                    url += "".join(
                        ("&" if "?" in url else "?") + f"include={item}"
                        for item in include
                    )
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()

    async def cancel_response(self, response_id: str) -> Dict[str, Any]:
        """Cancel a background response"""
        async with httpx.AsyncClient() as client:
            if self.is_azure:
                headers: Dict[str, str] = {"api-key": cast(str, self.api_key)}
                base = f"{self.base_url}/responses/{response_id}/cancel"
                sep = "?" if "?" not in base else "&"
                url = f"{base}{sep}api-version={self.azure_api_version}"
            else:
                headers: Dict[str, str] = {
                    "Authorization": f"Bearer {self.api_key}"
                }
                url = f"{self.base_url}/responses/{response_id}/cancel"
            response = await client.post(url, headers=headers)
            response.raise_for_status()
            return response.json()

    async def stream_response(
        self, response_id: str, starting_after: Optional[int] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream events from a background response"""
        if self.is_azure:
            base = f"{self.base_url}/responses/{response_id}"
            sep = "?" if "?" not in base else "&"
            url = (
                f"{base}{sep}stream=true&api-version={self.azure_api_version}"
            )
            if starting_after is not None:
                url += f"&starting_after={starting_after}"
            headers: Dict[str, str] = {
                "api-key": cast(str, self.api_key),
                "Accept": "text/event-stream",
            }
        else:
            url = f"{self.base_url}/responses/{response_id}?stream=true"
            if starting_after is not None:
                url += f"&starting_after={starting_after}"
            headers: Dict[str, str] = {
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

    async def delete_response(self, response_id: str) -> Dict[str, Any]:
        """Delete a stored response object."""
        async with httpx.AsyncClient() as client:
            if self.is_azure:
                headers: Dict[str, str] = {"api-key": cast(str, self.api_key)}
                base = f"{self.base_url}/responses/{response_id}"
                sep = "?" if "?" not in base else "&"
                url = f"{base}{sep}api-version={self.azure_api_version}"
            else:
                headers: Dict[str, str] = {
                    "Authorization": f"Bearer {self.api_key}"
                }
                url = f"{self.base_url}/responses/{response_id}"
            resp = await client.delete(url, headers=headers)
            resp.raise_for_status()
            try:
                return resp.json()
            except json.JSONDecodeError:
                return {"deleted": True, "status_code": resp.status_code}

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

        return cast(Dict[str, Any], await self.create_response(
            model="o3-deep-research",
            input=input_messages,
            tools=tools,
            background=background,
            reasoning={"summary": "auto"},
            max_tool_calls=max_tool_calls,
        ))

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
        # Exponential backoff with jitter, capped to 15s
        interval = max(1.0, float(poll_interval))
        max_interval = 15.0

        while True:
            response = await self.retrieve_response(response_id)

            status = str(response.get("status", "")).lower()
            if status not in ["queued", "in_progress"]:
                return response

            if timeout and (asyncio.get_event_loop().time() - start_time) > timeout:
                raise TimeoutError(
                    f"Response {response_id} did not complete within {timeout} seconds"
                )

            # Backoff with small jitter to reduce churn
            await asyncio.sleep(interval)
            interval = min(max_interval, interval * 1.5 + (0.25 if interval < 10 else 0.0))

    # ─────────── Helper Methods ───────────
    def extract_final_text(self, response: Dict[str, Any]) -> Optional[str]:
        """Delegates to shared extractor in llm_client for consistency."""
        from .llm_client import extract_responses_final_text
        return extract_responses_final_text(response)

    def extract_citations(self, response: Dict[str, Any]) -> List[Citation]:
        """Extract citations via llm_client and return a typed list."""
        from .llm_client import extract_responses_citations
        items = extract_responses_citations(response)
        out: List[Citation] = []
        for i in items:
            url = str(i.get("url") or "")
            title = str(i.get("title") or "")
            try:
                start_index = int(i.get("start_index") or 0)
            except (TypeError, ValueError):
                start_index = 0
            try:
                end_index = int(i.get("end_index") or 0)
            except (TypeError, ValueError):
                end_index = 0
            out.append(
                Citation(
                    url=url,
                    title=title,
                    start_index=start_index,
                    end_index=end_index,
                )
            )
        return out

    def extract_web_search_calls(
        self, response: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Return normalised web search tool calls from a response."""
        from .llm_client import extract_responses_web_search_calls
        return extract_responses_web_search_calls(response)

    def extract_tool_calls(
        self, response: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Return normalised tool calls from a response."""
        from .llm_client import extract_responses_tool_calls
        return extract_responses_tool_calls(response)

    async def _stream_response(
        self, response: httpx.Response
    ) -> AsyncIterator[Dict[str, Any]]:
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
@lru_cache(maxsize=1)
def get_responses_client() -> OpenAIResponsesClient:
    """Return a cached singleton instance of the responses client."""
    return OpenAIResponsesClient()
