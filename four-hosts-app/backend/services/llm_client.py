"""
LLM Client for Four-Hosts Research Application
---------------------------------------------
Simplified, paradigm-aware LLM interface supporting Azure OpenAI and OpenAI.
Provides only the interfaces actually used by the application.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union, TYPE_CHECKING

import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from utils.otel import otel_span as _otel_span

# Type checking imports
if TYPE_CHECKING:
    from services.classification_engine import HostParadigm

# ────────────────────────────────────────────────────────────
#  Logging
# ────────────────────────────────────────────────────────────
logger = structlog.get_logger(__name__)


# ────────────────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────────────────
try:
    from core.config import SYNTHESIS_BASE_TOKENS as DEFAULT_MAX_TOKENS
except Exception:
    DEFAULT_MAX_TOKENS = 8000

# Paradigm -> Model mapping (all use o3 for consistency)
_PARADIGM_MODEL_MAP: Dict[str, str] = {
    "dolores": "o3",
    "teddy": "o3",
    "bernard": "o3",
    "maeve": "o3",
}

# Paradigm-specific temperatures
_PARADIGM_TEMPERATURE: Dict[str, float] = {
    "bernard": 0.2,   # Analytical - low creativity
    "maeve": 0.4,     # Strategic - balanced
    "dolores": 0.6,   # Revolutionary - higher creativity
    "teddy": 0.5,     # Supportive - moderate
}

# Paradigm-specific reasoning effort
_PARADIGM_REASONING: Dict[str, str] = {
    "bernard": "low",     # Analytical - fast, precise
    "maeve": "medium",    # Strategic - balanced depth
    "dolores": "medium",  # Revolutionary - thoughtful
    "teddy": "low",       # Supportive - quick responses
}

# ────────────────────────────────────────────────────────────
#  Helper Functions
# ────────────────────────────────────────────────────────────
def _normalize_paradigm_code(value: Union[str, HostParadigm]) -> str:
    """Normalize paradigm to internal code (e.g., 'analytical' -> 'bernard')."""
    try:
        from models.paradigms import normalize_to_internal_code
        return normalize_to_internal_code(value)
    except Exception:
        # Fallback if imports fail
        if hasattr(value, 'value'):
            value = value.value
        mapping = {
            'analytical': 'bernard',
            'strategic': 'maeve',
            'revolutionary': 'dolores',
            'devotion': 'teddy'
        }
        return mapping.get(str(value).lower(), 'bernard')

def _get_system_prompt(paradigm_key: str) -> str:
    """Get system prompt for paradigm."""
    try:
        from models.paradigms_prompts import SYSTEM_PROMPTS
        return SYSTEM_PROMPTS.get(paradigm_key, "")
    except Exception:
        return ""

def _is_o_series(model: str) -> bool:
    """Check if model is o-series (o1/o3/o4) or gpt-5."""
    return str(model).startswith(("o", "gpt-5"))

# ────────────────────────────────────────────────────────────
#  Main LLM Client
# ────────────────────────────────────────────────────────────
class LLMClient:
    """Simplified LLM client with paradigm awareness."""

    def __init__(self):
        self.azure_client: Optional[AsyncOpenAI] = None
        self.openai_client: Optional[AsyncOpenAI] = None
        self._initialized = False
        self._azure_endpoint: Optional[str] = None
        self._azure_api_version: str = "preview"

        # Try to initialize clients
        try:
            self._init_clients()
            self._initialized = True
        except Exception as e:
            logger.warning(f"LLM client initialization deferred: {e}")

    def _init_clients(self) -> None:
        """Initialize Azure and/or OpenAI clients based on environment."""
        # Azure OpenAI
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self._azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "preview")

        if azure_key and azure_endpoint and azure_deployment:
            self._azure_endpoint = azure_endpoint.rstrip("/")
            # Use OpenAI v1-compatible path with explicit api-version for Azure
            azure_base = f"{self._azure_endpoint}/openai/v1"
            if self._azure_api_version:
                azure_base = f"{azure_base}?api-version={self._azure_api_version}"
            self.azure_client = AsyncOpenAI(
                api_key=azure_key,
                base_url=azure_base,
                timeout=120
            )
            logger.info(
                "✓ Azure OpenAI client initialized",
                _type="azure",
                endpoint=self._azure_endpoint,
                api_version=self._azure_api_version,
            )

        # OpenAI Cloud
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.openai_client = AsyncOpenAI(api_key=openai_key, timeout=120)
            logger.info("✓ OpenAI client initialized")

        if not self.azure_client and not self.openai_client:
            raise RuntimeError("Neither AZURE_OPENAI_* nor OPENAI_API_KEY environment variables are set")

    def _ensure_initialized(self) -> None:
        """Ensure client is initialized before use."""
        if not self._initialized:
            self._init_clients()
            self._initialized = True

    # ────────────────────────────────────────────────────────────
    #  Public Interfaces (Actually Used)
    # ────────────────────────────────────────────────────────────

    async def generate_completion(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        paradigm: Union[str, HostParadigm] = "bernard",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: Optional[float] = None,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        response_format: Optional[Dict[str, Any]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        stream: bool = False,
        reasoning_effort: Optional[str] = None,
    ) -> Union[str, AsyncIterator[str]]:
        """
        Generate completion with paradigm awareness.
        This is the primary interface used throughout the application.
        """
        self._ensure_initialized()

        # Determine model and parameters based on paradigm
        paradigm_key = _normalize_paradigm_code(paradigm)
        if not model:
            model = os.getenv("LLM_MODEL_OVERRIDE") or _PARADIGM_MODEL_MAP.get(paradigm_key, "o3")

        if temperature is None:
            temperature = _PARADIGM_TEMPERATURE.get(paradigm_key, 0.5)
            # Lower temperature for structured outputs
            if response_format or json_schema:
                temperature = min(temperature, 0.3)

        if reasoning_effort is None:
            reasoning_effort = _PARADIGM_REASONING.get(paradigm_key, "medium")

        # For o-series models, delegate to Responses API
        if _is_o_series(model) or response_format or json_schema:
            # Instrument Responses API generation
            _attrs = {
                "paradigm": paradigm_key,
                "model": model,
                "stream": bool(stream),
                "structured": bool(response_format or json_schema),
            }
            with _otel_span("llm.generate", _attrs) as _sp:
                _t0 = time.time()
                _result = await self._generate_via_responses(
                    prompt=prompt,
                    model=model,
                    paradigm_key=paradigm_key,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    response_format=response_format,
                    json_schema=json_schema,
                    stream=stream
                )
                try:
                    if _sp:
                        _sp.set_attribute("latency_ms", int((time.time() - _t0) * 1000))
                        _sp.set_attribute("success", True)
                except Exception:
                    pass
                return _result

        # For non-o models, use Chat Completions
        logger.info(
            "Using Chat Completions API",
            model=model,
            paradigm_key=paradigm_key
        )
        messages = [
            {"role": "system", "content": _get_system_prompt(paradigm_key)},
            {"role": "user", "content": prompt}
        ]

        # Select client (prefer Azure)
        client = self.azure_client or self.openai_client
        if not client:
            logger.error("No LLM client available")
            raise RuntimeError("No LLM client available")

        # Make the request
        try:
            logger.info(
                "Sending request to LLM",
                client_type="azure" if client == self.azure_client else "openai",
                model=self._get_deployment_name(model)
            )
            with _otel_span(
                "llm.chat_completions",
                {
                    "model": self._get_deployment_name(model),
                    "paradigm": paradigm_key,
                    "stream": bool(stream),
                },
            ) as _sp:
                _t0 = time.time()
                response = await client.chat.completions.create(
                    model=self._get_deployment_name(model),
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    response_format=response_format if response_format else None,
                    stream=stream
                )
                try:
                    if _sp:
                        _sp.set_attribute("latency_ms", int((time.time() - _t0) * 1000))
                        _sp.set_attribute("success", True)
                except Exception:
                    pass

            if stream:
                logger.info("Streaming response initiated")
                return self._stream_chat_response(response)

            # Extract content from response
            if hasattr(response, 'choices') and response.choices:
                result = response.choices[0].message.content or ""
                logger.info(
                    "Completion received successfully",
                    response_length=len(result),
                    model=model
                )
                return result
            logger.warning("Empty response received from LLM")
            return ""

        except Exception as e:
            logger.error(
                "LLM request failed",
                error=str(e),
                model=model,
                paradigm_key=paradigm_key,
                exc_info=True
            )
            raise

    async def generate_structured_output(
        self,
        prompt: str,
        schema: Dict[str, Any],
        *,
        model: Optional[str] = None,
        paradigm: Union[str, HostParadigm] = "bernard",
    ) -> Dict[str, Any]:
        """Generate JSON output matching the provided schema."""
        logger.info(
            "Generating structured output",
            paradigm=paradigm,
            model=model,
            schema_keys=list(schema.keys()) if isinstance(schema, dict) else None
        )
        raw = await self.generate_completion(
            prompt,
            model=model,
            paradigm=paradigm,
            json_schema=schema,
            temperature=0.3,
            stream=False,
        )

        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse structured output: {e}")
                raise ValueError("Structured generation failed")

        raise ValueError("Unexpected response type from generate_completion")

    async def generate_paradigm_content(
        self,
        prompt: str,
        *,
        paradigm: Union[str, HostParadigm],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.7,
        model: Optional[str] = None,
    ) -> str:
        """Generate content with specific paradigm perspective."""
        result = await self.generate_completion(
            prompt=prompt,
            paradigm=paradigm,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
            stream=False,
        )

        # Ensure string return
        if isinstance(result, str):
            return result
        return ""

    def get_active_backend_info(self) -> Dict[str, Any]:
        """Return diagnostic info about configured backends."""
        return {
            "azure_enabled": bool(self.azure_client),
            "openai_enabled": bool(self.openai_client),
            "default_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", "o3"),
            "api_version": self._azure_api_version,
        }

    def is_initialized(self) -> bool:
        """Check if client is initialized."""
        return self._initialized

    def initialize(self) -> None:
        """Initialize the client if not already done."""
        self._ensure_initialized()

    # ────────────────────────────────────────────────────────────
    #  Private Helper Methods
    # ────────────────────────────────────────────────────────────

    def _get_deployment_name(self, model: str) -> str:
        """Get deployment name for Azure or model name for OpenAI."""
        if self.azure_client:
            return os.getenv("AZURE_OPENAI_DEPLOYMENT", model)
        return model

    async def _generate_via_responses(
        self,
        prompt: str,
        model: str,
        paradigm_key: str,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str,
        response_format: Optional[Dict[str, Any]],
        json_schema: Optional[Dict[str, Any]],
        stream: bool
    ) -> Union[str, AsyncIterator[str]]:
        """Generate using Responses API for o-series models."""
        # Build input messages
        messages = [
            {
                "role": "developer" if _is_o_series(model) else "system",
                "content": _get_system_prompt(paradigm_key)
            },
            {"role": "user", "content": prompt}
        ]

        # Build request
        request_data = {
            "model": self._get_deployment_name(model),
            "input": messages,
            "max_output_tokens": max_tokens,
            "stream": stream,
            "store": True,
        }

        # Add reasoning for o-series
        if _is_o_series(model):
            request_data["reasoning"] = {"effort": reasoning_effort}

        # Handle structured output
        if json_schema:
            request_data["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": json_schema.get("name", "JSONSchema"),
                    "schema": json_schema.get("schema", json_schema)
                }
            }
            request_data["instructions"] = "You must respond with valid JSON that matches the required schema."
        elif response_format and response_format.get("type") == "json_object":
            request_data["text"] = {"format": {"type": "json_object"}}
            request_data["instructions"] = "You must respond with valid JSON."

        # Delegate to Responses API
        result = await responses_create(**request_data)

        if stream:
            return self._stream_responses(result)

        # Extract text from response
        return extract_text_from_any(result)

    async def _stream_chat_response(self, response: AsyncIterator[Any]) -> AsyncIterator[str]:
        """Stream tokens from Chat Completions response."""
        async for chunk in response:
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    yield delta.content

    async def _stream_responses(self, response: AsyncIterator[Any]) -> AsyncIterator[str]:
        """Stream tokens from Responses API."""
        async for event in response:
            if isinstance(event, dict) and event.get("type") == "response.output_text.delta":
                if "delta" in event:
                    yield event["delta"]

# ────────────────────────────────────────────────────────────
#  Singleton Instance
# ────────────────────────────────────────────────────────────
llm_client = LLMClient()

async def initialise_llm_client() -> bool:
    """Initialize LLM client on FastAPI startup."""
    global llm_client
    try:
        if not llm_client.is_initialized():
            llm_client.initialize()

            # Initialize background manager if available
            if llm_client.azure_client:
                try:
                    from services.background_llm import initialize_background_manager
                    initialize_background_manager(llm_client.azure_client)
                except ImportError:
                    pass

            logger.info("✓ LLM client initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        return False

# ────────────────────────────────────────────────────────────
#  Response Extraction Utilities
# ────────────────────────────────────────────────────────────
def extract_text_from_any(response: Any) -> str:
    """Extract text from various response formats."""
    if isinstance(response, str):
        return response.strip()

    if isinstance(response, dict):
        # Try Responses API format
        text = extract_responses_final_text(response)
        if text:
            return text.strip()

        # Try Chat Completions format
        choices = response.get("choices", [])
        if choices and isinstance(choices, list):
            msg = choices[0].get("message", {})
            content = msg.get("content", "")
            if content:
                return str(content).strip()

    # Handle object responses
    if hasattr(response, 'choices') and response.choices:
        choice = response.choices[0]
        if hasattr(choice, 'message'):
            content = choice.message.content
            if content:
                return str(content).strip()

    return ""

def extract_responses_final_text(payload: Dict[str, Any]) -> Optional[str]:
    """Extract final text from Responses API payload."""
    if not isinstance(payload, dict) or "output" not in payload:
        return None

    for item in reversed(payload.get("output") or []):
        if isinstance(item, dict) and item.get("type") == "message":
            for content in item.get("content", []):
                if isinstance(content, dict) and content.get("type") == "output_text":
                    return content.get("text")
    return None

def extract_responses_citations(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract citations from Responses API payload."""
    citations = []
    if not isinstance(payload, dict) or "output" not in payload:
        return citations

    for item in payload.get("output", []):
        if isinstance(item, dict) and item.get("type") == "message":
            for content in item.get("content", []):
                if isinstance(content, dict) and content.get("type") == "output_text":
                    for ann in content.get("annotations", []):
                        if isinstance(ann, dict) and ann.get("type") == "url_citation":
                            citations.append({
                                "url": ann.get("url"),
                                "title": ann.get("title"),
                                "start_index": ann.get("start_index"),
                                "end_index": ann.get("end_index"),
                            })
    return citations

def extract_responses_tool_calls(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tool calls from Responses API payload."""
    tool_calls = []
    if not isinstance(payload, dict) or "output" not in payload:
        return tool_calls

    for item in payload.get("output", []):
        if isinstance(item, dict) and item.get("type") in {
            "web_search_call", "code_interpreter_call", "mcp_call"
        }:
            tool_calls.append(item)
    return tool_calls

def extract_responses_web_search_calls(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract web search calls from Responses API payload."""
    return [
        c for c in extract_responses_tool_calls(payload)
        if isinstance(c, dict) and c.get("type") == "web_search_call"
    ]

class ResponsesNormalized(BaseModel):
    """Normalized Responses API content."""
    text: Optional[str] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    web_search_calls: List[Dict[str, Any]] = Field(default_factory=list)

def normalize_responses_payload(payload: Dict[str, Any]) -> ResponsesNormalized:
    """Normalize a Responses API payload."""
    return ResponsesNormalized(
        text=extract_responses_final_text(payload),
        citations=extract_responses_citations(payload),
        tool_calls=extract_responses_tool_calls(payload),
        web_search_calls=extract_responses_web_search_calls(payload),
    )

# ────────────────────────────────────────────────────────────
#  Responses API Delegation
# ────────────────────────────────────────────────────────────
async def responses_create(
    *,
    model: str,
    input: Union[str, List[Dict[str, Any]]],
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
    **kwargs
) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
    """Create a response using Responses API."""
    from services.openai_responses_client import get_responses_client

    client = get_responses_client()
    return await client.create_response(
        model=model,
        input=input,
        tools=tools,
        background=background,
        stream=stream,
        reasoning=reasoning,
        max_tool_calls=max_tool_calls,
        instructions=instructions,
        store=store,
        previous_response_id=previous_response_id,
        max_output_tokens=max_output_tokens,
        response_format=response_format,
        text=text,
        **kwargs
    )

async def responses_retrieve(response_id: str) -> Dict[str, Any]:
    """Retrieve a response by ID."""
    from services.openai_responses_client import get_responses_client
    return await get_responses_client().retrieve_response(response_id)

async def responses_stream(response_id: str, starting_after: Optional[int] = None) -> AsyncIterator[Dict[str, Any]]:
    """Stream response events."""
    from services.openai_responses_client import get_responses_client
    return get_responses_client().stream_response(response_id, starting_after)

async def responses_deep_research(
    query: str,
    *,
    use_web_search: bool = True,
    web_search_config: Optional[Dict[str, Any]] = None,
    use_code_interpreter: bool = False,
    mcp_servers: Optional[List[Dict[str, Any]]] = None,
    system_prompt: Optional[str] = None,
    max_tool_calls: Optional[int] = None,
    background: bool = True,
) -> Dict[str, Any]:
    """Execute deep research with tools."""
    from services.openai_responses_client import get_responses_client

    # Build tools list
    tools = []
    if use_web_search:
        if web_search_config:
            tools.append(web_search_config)
        else:
            tools.append({"type": "web_search_preview", "search_context_size": "medium"})

    if use_code_interpreter:
        tools.append({"type": "code_interpreter", "container": {"type": "auto"}})

    if mcp_servers:
        tools.extend(mcp_servers)

    # Build messages
    messages = []
    if system_prompt:
        messages.append({
            "role": "developer",
            "content": [{"type": "input_text", "text": system_prompt}]
        })
    messages.append({
        "role": "user",
        "content": [{"type": "input_text", "text": query}]
    })

    # Determine model
    model = os.getenv("DEEP_RESEARCH_MODEL", "o3")
    if llm_client.azure_client:
        model = os.getenv("AZURE_OPENAI_DEPLOYMENT", model)

    client = get_responses_client()
    return await client.create_response(
        model=model,
        input=messages,
        tools=tools,
        background=background,
        reasoning={"summary": "auto"},
        max_tool_calls=max_tool_calls,
    )
