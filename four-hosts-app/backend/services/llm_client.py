"""
LLM Client for Four-Hosts Research Application
---------------------------------------------
• Supports Azure OpenAI (Responses API preview) and OpenAI
• Provides chat completions, structured outputs, tool calling, and
  multi-turn conversations with optional SSE streaming + usage metrics
"""

from __future__ import annotations

import json
import logging
import os
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Union, cast

import httpx
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ────────────────────────────────────────────────────────────
#  Logging
# ────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
#  Enums / Constants
# ────────────────────────────────────────────────────────────
class ResponseFormat(Enum):
    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


class TruncationStrategy(Enum):
    AUTO = "auto"
    DISABLED = "disabled"


# Internal mapping keyed by internal code names
_PARADIGM_MODEL_MAP: Dict[str, str] = {
    "dolores": "gpt-4o",
    "teddy": "gpt-4o-mini",
    "bernard": "gpt-4o",
    "maeve": "gpt-4o-mini",
}

_SYSTEM_PROMPTS: Dict[str, str] = {
    "dolores": (
        "You are a revolutionary truth-seeker exposing systemic injustices. "
        "Focus on revealing hidden power structures and systemic failures. "
        "Use emotional, impactful language that moves people to action. "
        "Cite specific examples and evidence of wrongdoing."
    ),
    "teddy": (
        "You are a compassionate caregiver focused on helping and protecting others. "
        "Show deep understanding and empathy. Provide comprehensive resources and "
        "support options with uplifting, supportive language."
    ),
    "bernard": (
        "You are an analytical researcher focused on empirical evidence. "
        "Present statistical findings, identify patterns, and maintain "
        "scientific objectivity."
    ),
    "maeve": (
        "You are a strategic advisor focused on competitive advantage. "
        "Provide specific tactical recommendations and define clear success metrics."
    ),
}


# NOTE: Avoid importing models.paradigms at module import time to prevent
# circular imports with services.classification_engine (which imports this module
# during global initialization). We'll import lazily inside helper functions.


def _norm_code(value: Union[str, "HostParadigm"]) -> str:
    # Lazy import to break circular dependency
    from models.paradigms import normalize_to_internal_code as _norm
    return _norm(value)


def _select_model(
    paradigm: Union[str, "HostParadigm"],
    explicit_model: str | None = None,
) -> str:
    """Return the model to use, preferring an explicit value when provided.

    Accepts either a string (legacy "bernard" or enum value like "analytical")
    or a HostParadigm enum instance.
    """
    if explicit_model:
        return explicit_model
    key = _norm_code(paradigm)
    return _PARADIGM_MODEL_MAP.get(key, "gpt-4o-mini")


def _system_role_for(model: str) -> str:
    """Azure ‘o’ series models expect the special ‘developer’ role for system msgs."""
    return "developer" if model.startswith(("o1", "o3", "o4")) else "system"


# ────────────────────────────────────────────────────────────
#  LLM Client
# ────────────────────────────────────────────────────────────
class LLMClient:
    """Asynchronous client wrapper for OpenAI and Azure OpenAI."""

    openai_client: Optional[AsyncOpenAI]
    azure_client: Optional[AsyncOpenAI]  # Azure also uses AsyncOpenAI with different base_url

    def __init__(self) -> None:
        self.openai_client = None
        self.azure_client = None
        self._initialized = False
        # Try to initialize, but don't fail at import time
        try:
            self._init_clients()
            self._initialized = True
        except Exception as e:
            logger.warning(f"LLM client initialization deferred: {e}")

    # ─────────── client initialisation ───────────
    def _init_clients(self) -> None:
        # Azure OpenAI
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "preview")
        
        if azure_key and endpoint and deployment:
            # Use AsyncOpenAI with Azure endpoint (per documentation)
            endpoint = endpoint.rstrip("/")
            if not endpoint.endswith("/"):
                endpoint = f"{endpoint}/"
                
            self.azure_client = AsyncOpenAI(
                api_key=azure_key,
                base_url=endpoint,
            )
            logger.info(f"✓ Azure OpenAI client initialised (endpoint: {endpoint})")
        else:
            missing = []
            if not azure_key:
                missing.append("AZURE_OPENAI_API_KEY")
            if not endpoint:
                missing.append("AZURE_OPENAI_ENDPOINT")
            if not deployment:
                missing.append("AZURE_OPENAI_DEPLOYMENT")
            if missing:
                logger.debug(f"Azure OpenAI not configured - missing: {', '.join(missing)}")
            self.azure_client = None
            
        # OpenAI cloud
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.openai_client = AsyncOpenAI(api_key=openai_key)
            logger.info("✓ OpenAI client initialised")
        else:
            self.openai_client = None
            
        if not self.azure_client and not self.openai_client:
            raise RuntimeError("Neither AZURE_OPENAI_* nor OPENAI_API_KEY environment variables are set")
    
    def _ensure_initialized(self) -> None:
        """Ensure client is initialized before use."""
        if not self._initialized:
            self._init_clients()
            self._initialized = True

    # ─────────── utility helpers ───────────
    @staticmethod
    def _wrap_tool_choice(
        choice: Union[str, Dict[str, Any], None]
    ) -> Union[str, Dict[str, Any], None]:
        """Normalize `tool_choice` for both Azure & OpenAI.

        • OpenAI  → "auto" | "none" | {"type":"function","function": {"name": ...}}
        • Azure   → {"type":"auto"} | {"type":"none"} | {"type":"function","function": {"name": ...}}
        """
        if choice is None or choice == "auto":
            return "auto"          # ← string for OpenAI
        if choice == "none":
            return "none"
        if isinstance(choice, str):
            # explicit function name
            return {"type": "function", "function": {"name": choice}}
        return choice

    async def aclose(self) -> None:
        """Close underlying httpx client(s) to release sockets."""
        # No longer needed as we create httpx clients on demand

    # ─────────── core API methods ───────────
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    )
    async def generate_completion(
        self,
        prompt: str,
        *,
        model: str | None = None,
        paradigm: Union[str, "HostParadigm"] = "bernard",
        max_tokens: int = 2_000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        response_format: Optional[Dict[str, Any]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        stream: bool = False,
    ) -> Union[str, AsyncIterator[str]]:
        """
        Chat-style completion with optional SSE stream support.
        Returns either the full response string or an async iterator of tokens.
        """
        self._ensure_initialized()
        model_name = _select_model(paradigm, model)
        paradigm_key = _norm_code(paradigm)

        # Build shared message list
        messages = [
            {
                "role": _system_role_for(model_name),
                "content": _SYSTEM_PROMPTS.get(paradigm_key, ""),
            },
            {"role": "user", "content": prompt},
        ]

        # Prepare request kwargs
        kw: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": stream,
        }

        # Token param - use max_completion_tokens for o3
        if model_name in {"o3", "azure"}:
            kw["max_completion_tokens"] = max_tokens
        else:
            kw["max_tokens"] = max_tokens

        # Response formatting
        if response_format:
            kw["response_format"] = response_format
        elif json_schema:
            kw["response_format"] = {"type": "json_schema", "json_schema": json_schema}

        # Tools
        if tools:
            kw["tools"] = tools
            tc = self._wrap_tool_choice(tool_choice)
            if tc:
                kw["tool_choice"] = tc                # keep only when tools exist
        else:
            tool_choice = None                        # ensure we don't forward it

        # ─── Azure OpenAI path (for o3/o1 models) ───
        if model_name in {"o3", "o1", "azure", "gpt-5-mini"} and self.azure_client:
            try:
                deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "o3-synthesis")
                # Remove Azure-incompatible params
                azure_kwargs = {k: v for k, v in kw.items() if k not in ["stream", "tool_choice", "model"]}
                if tools and tool_choice:
                    azure_kwargs["tool_choice"] = self._wrap_tool_choice(tool_choice)
                    
                # Use chat completions API with Azure client
                # For o1/o3 models, adjust message format if needed
                azure_messages = []
                for msg in messages:
                    role = msg.get("role", "user")
                    # Convert "system" to "developer" for reasoning models
                    if role == "system" and model_name in {"o3", "o1", "gpt-5-mini"}:
                        role = "developer"
                    azure_messages.append({"role": role, "content": msg.get("content", "")})
                
                # Build request for Azure
                azure_request = {
                    "model": deployment,
                    "messages": azure_messages,
                }
                
                # For reasoning models (o1/o3), use max_completion_tokens and reasoning_effort
                if model_name in {"o3", "o1", "gpt-5-mini"}:
                    max_tokens_val = kw.get("max_completion_tokens", kw.get("max_tokens", 2000))
                    azure_request["max_completion_tokens"] = max_tokens_val
                    azure_request["reasoning_effort"] = kw.get("reasoning_effort", "medium")
                else:
                    # Standard models use max_tokens and temperature
                    azure_request["max_tokens"] = kw.get("max_tokens", 2000)
                    azure_request["temperature"] = temperature
                    azure_request["top_p"] = top_p
                    azure_request["frequency_penalty"] = frequency_penalty
                    azure_request["presence_penalty"] = presence_penalty
                
                # Add response format if specified
                if response_format:
                    azure_request["response_format"] = response_format
                elif json_schema:
                    azure_request["response_format"] = {"type": "json_schema", "json_schema": json_schema}
                
                # Add tools if provided
                if tools:
                    azure_request["tools"] = tools
                    if tool_choice:
                        azure_request["tool_choice"] = self._wrap_tool_choice(tool_choice)
                
                # Add stream if requested
                if stream:
                    azure_request["stream"] = stream
                
                op_res = await self.azure_client.chat.completions.create(**azure_request)
                
                if stream:
                    return self._iter_openai_stream(cast(AsyncIterator[Any], op_res))
                
                # Extract content from response
                return self._extract_content_safely(op_res)
            except Exception as exc:
                logger.error(f"Azure OpenAI request failed • {exc}")
                raise
                
        # ─── OpenAI path ───
        if self.openai_client:
            try:
                # Only include tool_choice if tools were provided
                op_kwargs = {k: v for k, v in kw.items() if k not in ["stream", "tool_choice"]}
                if tools and tool_choice:
                    op_kwargs["tool_choice"] = self._wrap_tool_choice(tool_choice)
                    
                op_res = await self.openai_client.chat.completions.create(
                    **op_kwargs,
                    stream=stream,
                )
                if stream:
                    return self._iter_openai_stream(cast(AsyncIterator[Any], op_res))
                return self._extract_content_safely(op_res)
            except Exception as exc:
                logger.error(f"OpenAI request failed • {exc}")
                raise

        raise RuntimeError(
            "No LLM back-ends configured – set AZURE_OPENAI_* or OPENAI_API_KEY env vars."
        )

    # ─────────── higher-level helpers ───────────
    async def generate_structured_output(
        self,
        prompt: str,
        schema: Dict[str, Any],
        *,
        model: str | None = None,
        paradigm: Union[str, "HostParadigm"] = "bernard",
    ) -> Dict[str, Any]:
        """Return JSON matching the provided schema."""
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
            except json.JSONDecodeError as exc:
                logger.error("Structured output parse error", exc_info=exc)
        raise ValueError("Structured generation failed")

    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        *,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
        model: str | None = None,
        paradigm: Union[str, "HostParadigm"] = "bernard",
    ) -> Dict[str, Any]:
        """Invoke model with tool-calling enabled and return content + tool_calls."""
        self._ensure_initialized()
        model_name = _select_model(paradigm, model)
        paradigm_key = normalize_to_internal_code(paradigm)
        result: Dict[str, Any] | None = None

        wrapped_choice = self._wrap_tool_choice(tool_choice)
        
        # Use Azure for o3/o1 models
        if model_name in {"o3", "o1", "azure", "gpt-5-mini"} and self.azure_client:
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "o3")
            azure_msgs = [
                {
                    "role": _system_role_for(model_name),
                    "content": _SYSTEM_PROMPTS.get(paradigm_key, ""),
                },
                {"role": "user", "content": prompt},
            ]
            
            azure_req = {
                "model": deployment,
                "messages": cast(Any, azure_msgs),
                "tools": cast(Any, tools),
                "tool_choice": cast(Any, wrapped_choice),
            }
            
            # Use appropriate token parameter for model type
            if model_name in {"o3", "o1", "gpt-5-mini"}:
                azure_req["max_completion_tokens"] = 2_000
                azure_req["reasoning_effort"] = "medium"
            else:
                azure_req["max_tokens"] = 2_000
            
            op = await self.azure_client.chat.completions.create(**azure_req)
            result = {
                "content": self._extract_content_safely(op) if not (op.choices and op.choices[0].message.tool_calls) else (op.choices[0].message.content or ""),
                "tool_calls": op.choices[0].message.tool_calls or [] if op.choices else [],
            }
        # Use OpenAI
        elif self.openai_client:
            op = await self.openai_client.chat.completions.create(
                model=model_name,
                messages=cast(Any, [
                    {
                        "role": _system_role_for(model_name),
                        "content": _SYSTEM_PROMPTS.get(paradigm_key, ""),
                    },
                    {"role": "user", "content": prompt},
                ]),
                tools=cast(Any, tools),
                tool_choice=cast(Any, wrapped_choice),       # ← use string/dict directly for OpenAI
                max_tokens=2_000,
            )
            result = {
                "content": self._extract_content_safely(op) if not (op.choices and op.choices[0].message.tool_calls) else (op.choices[0].message.content or ""),
                "tool_calls": op.choices[0].message.tool_calls or [] if op.choices else [],
            }
        else:
            raise RuntimeError("No LLM back-ends configured for tool calling.")

        # Normalise return shape
        return {
            "content": result.get("content", ""),
            "tool_calls": result.get("tool_calls", []),
        }

    async def create_conversation(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str | None = None,
        paradigm: str = "bernard",
        max_tokens: int = 2_000,
        temperature: float = 0.7,
    ) -> str:
        """Multi-turn chat conversation helper (non-streaming)."""
        model_name = _select_model(paradigm, model)
        paradigm_key = normalize_to_internal_code(paradigm)
        full_msgs = [
            {
                "role": _system_role_for(model_name),
                "content": _SYSTEM_PROMPTS.get(paradigm_key, ""),
            },
            *messages,
        ]


        # Prefer Azure when available. For o1/o3 family use max_completion_tokens,
        # for gpt-4o style deployments use standard chat params.
        if self.azure_client:
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "o3")
            azure_req = {
                "model": deployment,
                "messages": cast(Any, full_msgs),
            }
            
            # Use appropriate parameters for model type
            if model_name.startswith("o") or model_name in {"gpt-5-mini"}:
                azure_req["max_completion_tokens"] = max_tokens
                azure_req["reasoning_effort"] = "medium"
            else:
                azure_req["max_tokens"] = max_tokens
                azure_req["temperature"] = temperature
            
            op = await self.azure_client.chat.completions.create(**azure_req)
            return self._extract_content_safely(op)
        # Use OpenAI (cloud) if Azure not configured
        elif self.openai_client:
            op = await self.openai_client.chat.completions.create(
                model=model_name,
                messages=cast(Any, full_msgs),
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return self._extract_content_safely(op)

        raise RuntimeError("No LLM back-ends configured for conversation.")

    async def generate_paradigm_content(
        self,
        prompt: str,
        *,
        paradigm: Union[str, "HostParadigm"],
        max_tokens: int = 2_000,
        temperature: float = 0.7,
        model: str | None = None,
    ) -> str:
        """Generate content based on a specific paradigm's perspective."""
        result = await self.generate_completion(
            prompt=prompt,
            paradigm=paradigm,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
            stream=False,
        )
        # Ensure we return a string
        if isinstance(result, str):
            return result
        # Should not happen with stream=False, but handle it defensively
        return ""
    
    async def generate_background(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[Any] = None,
        **kwargs
    ) -> str:
        """Submit a long-running task to background processing"""
        if not self.azure_client or not hasattr(self.azure_client, 'responses'):
            raise NotImplementedError("Background mode requires Azure OpenAI Responses API")
        
        from services.background_llm import background_llm_manager
        if not background_llm_manager:
            raise RuntimeError("Background LLM manager not initialized")
        
        # Submit task to background processing
        task_id = await background_llm_manager.submit_background_task(
            messages=messages,
            tools=tools,
            callback=callback,
            **kwargs
        )
        
        return task_id


    def _extract_content_safely(self, response) -> str:
        """Safely extract text content from various response types"""
        if isinstance(response, str):
            return response.strip()
        
        # Handle Azure Responses API response
        if hasattr(response, 'output_text') and response.output_text:
            return str(response.output_text).strip()
        
        # Handle output array
        if hasattr(response, 'output') and response.output:
            text_content = ""
            for output in response.output:
                if hasattr(output, 'content') and output.content:
                    for content_item in output.content:
                        if hasattr(content_item, 'text') and content_item.type == 'output_text':
                            text_content += content_item.text
            if text_content:
                return text_content.strip()
        
        # Handle ChatCompletion-like response
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                content = choice.message.content
                if content:
                    return str(content).strip()
        
        # Handle direct content attribute
        if hasattr(response, 'content') and response.content:
            return str(response.content).strip()
        
        # Handle text attribute
        if hasattr(response, 'text') and response.text:
            return str(response.text).strip()
        
        # Default fallback
        logger.warning(f"Could not extract text from response type: {type(response)}")
        return ""

    # ─────────── streaming iterators ───────────
    @staticmethod
    async def _iter_openai_stream(raw: AsyncIterator[Any]) -> AsyncIterator[str]:
        """Yield token strings from OpenAI chat stream."""
        async for chunk in raw:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    @staticmethod
    async def _iter_responses_stream(raw: AsyncIterator[Any]) -> AsyncIterator[str]:
        """Yield token strings from Azure Responses API stream."""
        async for event in raw:
            if hasattr(event, 'type') and event.type == 'response.output_text.delta':
                if hasattr(event, 'delta'):
                    yield event.delta



# ────────────────────────────────────────────────────────────
#  Singleton instance + startup helper
# ────────────────────────────────────────────────────────────
llm_client = LLMClient()


async def initialise_llm_client() -> bool:
    """FastAPI startup hook convenience."""
    global llm_client
    try:
        if not llm_client._initialized:
            llm_client._init_clients()
            llm_client._initialized = True
            
            # Initialize background LLM manager if Azure client is available
            if llm_client.azure_client:
                from services.background_llm import initialize_background_manager
                initialize_background_manager(llm_client.azure_client)
            
            # Initialize MCP integration
            from services.mcp_integration import configure_default_servers
            configure_default_servers()
            
            logger.info("✓ LLM client initialized successfully")
        else:
            logger.info("LLM client already initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        return False
