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
from pydantic import BaseModel
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

# Central default for output tokens
try:
    from core.config import SYNTHESIS_BASE_TOKENS as DEFAULT_MAX_TOKENS
except Exception:
    DEFAULT_MAX_TOKENS = 8000


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
    # Prefer Azure OpenAI o3 across paradigms for consistent reasoning quality.
    # The Azure path is selected when model_name is one of {"o3","o1","azure","gpt-5-mini"}
    # and AZURE_OPENAI_* is configured.
    "dolores": "o3",
    "teddy": "o3",
    "bernard": "o3",
    "maeve": "o3",
}

# Default temperature and reasoning-effort per paradigm
_PARADIGM_TEMPERATURE: Dict[str, float] = {
    "bernard": 0.2,
    "maeve": 0.4,
    "dolores": 0.6,
    "teddy": 0.5,
}

_PARADIGM_REASONING: Dict[str, str] = {
    "bernard": "low",
    "maeve": "medium",
    "dolores": "medium",
    "teddy": "low",
}

def _system_prompt(paradigm_key: str) -> str:
    try:
        from models.paradigms_prompts import SYSTEM_PROMPTS as _SP
        return _SP.get(paradigm_key, "")
    except Exception:
        return ""


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
    # Allow env override to force a specific model
    override = os.getenv("LLM_MODEL_OVERRIDE")
    if override:
        return override
    return _PARADIGM_MODEL_MAP.get(key, "o3")


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
        # Prefer the Azure Responses API by default (align with v1 preview guidance)
        self._azure_use_responses = (
            os.getenv("AZURE_OPENAI_USE_RESPONSES", "1").lower() in {"1", "true", "yes"}
        )
        # Try to initialize, but don't fail at import time
        try:
            self._init_clients()
            self._initialized = True
        except Exception as e:
            logger.warning(f"LLM client initialization deferred: {e}")

    # Internal helper: map requested model to Azure deployment name
    def _azure_model_for(self, requested: str) -> str:
        """Return the Azure deployment name to use for 'model'.

        Azure's SDK expects the deployment name as the model identifier.
        Fall back to the requested name (e.g., 'o3') if no env override is set.
        """
        return os.getenv("AZURE_OPENAI_DEPLOYMENT", requested)

    # ─────────── client initialisation ───────────
    def _init_clients(self) -> None:
        # Azure OpenAI
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "preview")

        if azure_key and endpoint and deployment:
            # Use AsyncOpenAI with Azure endpoint for Responses API.
            # For SDK calls we rely on the resource's v1 routing (no query string here).
            endpoint = endpoint.rstrip("/")
            base_url = f"{endpoint}/openai/v1/"

            self.azure_client = AsyncOpenAI(api_key=azure_key, base_url=base_url)
            logger.info(f"✓ Azure OpenAI client initialised (endpoint: {base_url})")
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

    # ─────────── metrics helpers ───────────
    def _record_stage_metrics(self, stage: str, model: str | None, start_ts: float, success: bool = True, tokens_in: Optional[int] = None, tokens_out: Optional[int] = None) -> None:
        try:
            from services.metrics import metrics
            dur_ms = (time.perf_counter() - start_ts) * 1000.0
            metrics.record_stage(
                stage=stage,
                duration_ms=dur_ms,
                paradigm=None,
                success=success,
                fallback=False,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                model=model,
            )
        except Exception:
            pass

    @staticmethod
    def _extract_usage_tokens_from_chat(op: Any) -> tuple[Optional[int], Optional[int]]:
        # Try OpenAI/azure ChatCompletion-like usage fields
        try:
            usage = getattr(op, 'usage', None)
            if usage and hasattr(usage, 'prompt_tokens'):
                return int(getattr(usage, 'prompt_tokens', 0) or 0), int(getattr(usage, 'completion_tokens', 0) or 0)
        except Exception:
            pass
        # Pydantic payload path
        try:
            payload = op.model_dump() if hasattr(op, 'model_dump') else None
            if isinstance(payload, dict) and 'usage' in payload:
                u = payload['usage'] or {}
                return int(u.get('prompt_tokens') or 0), int(u.get('completion_tokens') or 0)
        except Exception:
            pass
        return None, None

    # ─────────── diagnostics / info ───────────
    def get_active_backend_info(self) -> Dict[str, Any]:
        """Return a small diagnostic snapshot of the configured LLM backend.

        Includes whether Azure is enabled, the base URL used by the SDK client,
        the preferred deployment, and if the Responses API path is active.
        """
        info: Dict[str, Any] = {
            "azure_enabled": bool(self.azure_client is not None),
            "openai_enabled": bool(self.openai_client is not None),
            "use_responses_api": bool(self._azure_use_responses),
            "default_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", "o3"),
            # Report configured Azure API version (default used if unset)
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-01-preview"),
        }
        try:
            # AsyncOpenAI stores base_url on ._client.base_url
            if self.azure_client and hasattr(self.azure_client, "_client"):
                info["azure_base_url"] = str(self.azure_client._client.base_url)
        except Exception:
            pass
        return info

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
        Chat-style completion with optional SSE stream support.
        Returns either the full response string or an async iterator of tokens.
        """
        self._ensure_initialized()
        import time as time
        _start = time.perf_counter()
        model_name = _select_model(paradigm, model)
        paradigm_key = _norm_code(paradigm)
        # Apply sensible defaults if not provided
        if temperature is None:
            temperature = _PARADIGM_TEMPERATURE.get(paradigm_key, 0.5)
            if response_format or json_schema:
                temperature = min(temperature, 0.3)
        if reasoning_effort is None:
            reasoning_effort = _PARADIGM_REASONING.get(paradigm_key, "medium")

        # Build shared message list
        messages = [
            {
                "role": _system_role_for(model_name),
                "content": _system_prompt(paradigm_key),
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
            "reasoning_effort": reasoning_effort,
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

        # ─── Azure OpenAI path (prefer Responses API when enabled) ───
        if model_name in {"o3", "o1", "azure", "gpt-5-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-5-mini", "gpt-5"} and self.azure_client:
            try:
                # Use Responses API when enabled or when structured output requested on o-series
                force_responses = bool((response_format or json_schema) and model_name.startswith("o"))
                if self._azure_use_responses or force_responses:
                    # Build Responses API payload via HTTP client for consistent api-version
                    input_msgs: List[Dict[str, str]] = []
                    for msg in messages:
                        role = msg.get("role", "user")
                        if role == "system" and model_name.startswith("o"):
                            role = "developer"
                        input_msgs.append({"role": role, "content": msg.get("content", "")})

                    resp_req: Dict[str, Any] = {
                        "model": self._azure_model_for(model_name),
                        "input": input_msgs,
                        "max_output_tokens": kw.get("max_completion_tokens", kw.get("max_tokens", max_tokens)),
                        "reasoning": ({"effort": reasoning_effort} if model_name.startswith("o") else None),
                        "tools": tools or None,
                        "background": False,
                        "stream": stream,
                        "store": True,
                    }

                    # Responses API doesn't support response_format parameter
                    # Instead, add JSON instructions to guide structured output
                    if json_schema or response_format:
                        json_instruction = "You must respond with valid JSON that matches the required schema."
                        if response_format and response_format.get("type") == "json_object":
                            json_instruction = "You must respond with valid JSON."
                        resp_req["instructions"] = json_instruction

                    if tool_choice and tools:
                        resp_req["tool_choice"] = self._wrap_tool_choice(tool_choice)

                    from services.llm_client import responses_create  # self-module helper
                    if stream:
                        stream_iter = await responses_create(**{k: v for k, v in resp_req.items() if v is not None})
                        # Streaming – can't compute tokens; return iterator
                        return self._iter_responses_stream(cast(AsyncIterator[Any], stream_iter))
                    else:
                        op_res = await responses_create(**{k: v for k, v in resp_req.items() if v is not None})
                        out = self._extract_content_safely(op_res)
                        # Metrics (Responses JSON has no standard token usage yet)
                        self._record_stage_metrics("llm_generate", model_name, _start, success=True)
                        return out

                # Fallback: Chat Completions path (legacy / when Responses disabled)
                # Remove Azure-incompatible params
                azure_messages = []
                for msg in messages:
                    role = msg.get("role", "user")
                    if role == "system" and model_name.startswith("o"):
                        role = "developer"
                    azure_messages.append({"role": role, "content": msg.get("content", "")})

                cc_req = {
                    "model": self._azure_model_for(model_name),
                    "messages": azure_messages,
                }
                if model_name.startswith("o"):
                    cc_req["max_completion_tokens"] = kw.get("max_completion_tokens", kw.get("max_tokens", DEFAULT_MAX_TOKENS))
                    cc_req["reasoning_effort"] = kw.get("reasoning_effort", "medium")
                else:
                    cc_req["max_tokens"] = kw.get("max_tokens", DEFAULT_MAX_TOKENS)
                    cc_req["temperature"] = temperature
                    cc_req["top_p"] = top_p
                    cc_req["frequency_penalty"] = frequency_penalty
                    cc_req["presence_penalty"] = presence_penalty
                if not str(model_name).startswith("o"):
                    if response_format:
                        cc_req["response_format"] = response_format
                    elif json_schema:
                        cc_req["response_format"] = {"type": "json_schema", "json_schema": json_schema}
                if tools:
                    cc_req["tools"] = tools
                    if tool_choice:
                        cc_req["tool_choice"] = self._wrap_tool_choice(tool_choice)
                if stream:
                    cc_req["stream"] = stream

                op_res = await self.azure_client.chat.completions.create(**cc_req)
                if stream:
                    return self._iter_openai_stream(cast(AsyncIterator[Any], op_res))
                return self._extract_content_safely(op_res)
            except Exception as exc:
                logger.error(f"Azure OpenAI request failed • {exc}")
                raise

        # ─── OpenAI path ───
        if self.openai_client:
            try:
                # Only include tool_choice if tools were provided
                op_kwargs = {k: v for k, v in kw.items() if k not in ["stream", "tool_choice"]}
                # For o-series on OpenAI Chat Completions, strip unsupported params
                if str(model_name).startswith("o") or model_name in {"gpt-5", "gpt-5-mini", "gpt-5-nano"}:
                    for p in ["temperature", "top_p", "presence_penalty", "frequency_penalty", "max_tokens"]:
                        op_kwargs.pop(p, None)
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
        import time as time
        _start = time.perf_counter()
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
                data = json.loads(raw)
                self._record_stage_metrics("llm_structured", model or _select_model(paradigm, model), _start, success=True)
                return data
            except json.JSONDecodeError as exc:
                logger.error("Structured output parse error", exc_info=exc)
        self._record_stage_metrics("llm_structured", model or _select_model(paradigm, model), _start, success=False)
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
        paradigm_key = _norm_code(paradigm)
        import time as time
        _start = time.perf_counter()
        result: Dict[str, Any] | None = None

        wrapped_choice = self._wrap_tool_choice(tool_choice)

        # Use Azure for o3/o1 models
        if model_name in {"o3", "o1", "azure", "gpt-5-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4o"} and self.azure_client:
            # Prefer Responses API when enabled
            if self._azure_use_responses:
                input_msgs: List[Dict[str, Any]] = [
                    {"role": _system_role_for(model_name), "content": _system_prompt(paradigm_key)},
                    {"role": "user", "content": prompt},
                ]
                resp_req: Dict[str, Any] = {
                    "model": self._azure_model_for(model_name),
                    "input": input_msgs,
                    "tools": cast(Any, tools),
                    "background": False,
                    "stream": False,
                    "store": True,
                    "max_output_tokens": DEFAULT_MAX_TOKENS,
                }
                if wrapped_choice:
                    resp_req["tool_choice"] = wrapped_choice
                from services.llm_client import responses_create as _resp_create
                op_res = await _resp_create(**{k: v for k, v in resp_req.items() if v is not None})
                # Extract content and tool_calls from Responses payload
                from services.llm_client import extract_responses_final_text as _resp_text, extract_responses_tool_calls as _resp_tools
                result = {
                    "content": _resp_text(op_res) or "",
                    "tool_calls": _resp_tools(op_res) or [],
                }
                out = {"content": result.get("content", ""), "tool_calls": result.get("tool_calls", [])}
                self._record_stage_metrics("llm_tools", model_name, _start, success=True)
                return out

            azure_msgs = [
                {
                    "role": _system_role_for(model_name),
                    "content": _system_prompt(paradigm_key),
                },
                {"role": "user", "content": prompt},
            ]

            azure_req = {
                # Use Azure deployment name for 'model'
                "model": self._azure_model_for(model_name),
                "messages": cast(Any, azure_msgs),
                "tools": cast(Any, tools),
                "tool_choice": cast(Any, wrapped_choice),
            }

            # Use appropriate token parameter for model type
            if model_name in {"o3", "o1", "gpt-5-mini"}:
                azure_req["max_completion_tokens"] = DEFAULT_MAX_TOKENS
                azure_req["reasoning_effort"] = "medium"
            else:
                azure_req["max_tokens"] = DEFAULT_MAX_TOKENS

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
                        "content": _system_prompt(paradigm_key),
                    },
                    {"role": "user", "content": prompt},
                ]),
                tools=cast(Any, tools),
                tool_choice=cast(Any, wrapped_choice),       # ← use string/dict directly for OpenAI
                max_tokens=DEFAULT_MAX_TOKENS,
            )
            result = {
                "content": self._extract_content_safely(op) if not (op.choices and op.choices[0].message.tool_calls) else (op.choices[0].message.content or ""),
                "tool_calls": op.choices[0].message.tool_calls or [] if op.choices else [],
            }
        else:
            raise RuntimeError("No LLM back-ends configured for tool calling.")

        # Normalise return shape
        out = {
            "content": result.get("content", ""),
            "tool_calls": result.get("tool_calls", []),
        }
        self._record_stage_metrics("llm_tools", model_name, _start, success=True)
        return out

    async def create_conversation(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str | None = None,
        paradigm: str = "bernard",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.7,
    ) -> str:
        """Multi-turn chat conversation helper (non-streaming)."""
        model_name = _select_model(paradigm, model)
        paradigm_key = _norm_code(paradigm)
        full_msgs = [
            {
                "role": _system_role_for(model_name),
                "content": _system_prompt(paradigm_key),
            },
            *messages,
        ]


        # Prefer Azure when available. For o1/o3 family use max_completion_tokens,
        # for gpt-4o style deployments use standard chat params.
        if self.azure_client:
            # Prefer Azure Responses API when enabled
            if self._azure_use_responses:
                input_msgs: List[Dict[str, Any]] = []
                for m in full_msgs:
                    role = m.get("role", "user")
                    if role == "system" and model_name.startswith("o"):
                        role = "developer"
                    input_msgs.append({"role": role, "content": m.get("content", "")})
                from services.llm_client import responses_create as _resp_create, extract_responses_final_text as _resp_text
                op_res = await _resp_create(
                    model=self._azure_model_for(model_name),
                    input=input_msgs,
                    background=False,
                    stream=False,
                    store=True,
                    max_output_tokens=max_tokens,
                    reasoning={"effort": "medium"} if model_name.startswith("o") else None,
                )
                text = _resp_text(op_res) or ""
                self._record_stage_metrics("llm_generate", model_name, _start, success=True)
                return text

            azure_req = {
                # Use Azure deployment name for 'model'
                "model": self._azure_model_for(model_name),
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
            text = self._extract_content_safely(op)
            pin, pout = self._extract_usage_tokens_from_chat(op)
            self._record_stage_metrics("llm_generate", model_name, _start, success=True, tokens_in=pin, tokens_out=pout)
            return text
        # Use OpenAI (cloud) if Azure not configured
        elif self.openai_client:
            op = await self.openai_client.chat.completions.create(
                model=model_name,
                messages=cast(Any, full_msgs),
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = self._extract_content_safely(op)
            pin, pout = self._extract_usage_tokens_from_chat(op)
            self._record_stage_metrics("llm_generate", model_name, _start, success=True, tokens_in=pin, tokens_out=pout)
            return text

        self._record_stage_metrics("llm_generate", model_name, _start, success=False)
        raise RuntimeError("No LLM back-ends configured for conversation.")

    async def generate_paradigm_content(
        self,
        prompt: str,
        *,
        paradigm: Union[str, "HostParadigm"],
        max_tokens: int = DEFAULT_MAX_TOKENS,
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

        # Handle raw dict payloads from Responses API
        if isinstance(response, dict):
            try:
                text = extract_responses_final_text(response)
                if text:
                    return text.strip()
            except Exception:
                pass

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

        # Handle ChatCompletion-like response (OpenAI 1.x & Azure)
        # Try dataclass/dict access patterns robustly.
        try:
            # Pydantic models expose model_dump(); fall back to __dict__ if missing
            payload = response.model_dump() if hasattr(response, 'model_dump') else None
        except Exception:
            payload = None

        if payload and isinstance(payload, dict):
            try:
                choices = payload.get('choices') or []
                if choices:
                    msg = (choices[0].get('message') or {})
                    content = msg.get('content') or msg.get('refusal') or ''
                    if isinstance(content, list):
                        # Some SDKs may return content as array of parts
                        parts = []
                        for part in content:
                            if isinstance(part, dict) and 'text' in part:
                                parts.append(str(part['text']))
                        content = ''.join(parts)
                    if content:
                        return str(content).strip()
            except Exception:
                pass

        # Attribute-style ChatCompletion handling
        if hasattr(response, 'choices') and getattr(response, 'choices'):
            try:
                choice = response.choices[0]
                # Newer SDKs: choice.message.content may be None when tool_calls are present
                if hasattr(choice, 'message'):
                    msg = choice.message
                    # Prefer content; fall back to text-like fields if any
                    content = getattr(msg, 'content', None)
                    if not content and hasattr(msg, 'refusal'):
                        content = getattr(msg, 'refusal')
                    if isinstance(content, list):
                        content = ''.join([str(getattr(p, 'text', p)) for p in content])
                    if content:
                        return str(content).strip()
                # Some responses may have .text at top level per choice (legacy)
                if hasattr(choice, 'text') and choice.text:
                    return str(choice.text).strip()
            except Exception:
                pass

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
            # Handle SDK-style objects
            if hasattr(event, 'type') and event.type == 'response.output_text.delta':
                if hasattr(event, 'delta'):
                    yield event.delta
                continue
            # Handle HTTP SSE dicts
            if isinstance(event, dict):
                et = event.get('type')
                if et == 'response.output_text.delta' and 'delta' in event:
                    yield str(event.get('delta') or '')



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


# ────────────────────────────────────────────────────────────
#  Public helpers for Responses API extraction (shared)
# ────────────────────────────────────────────────────────────

def extract_text_from_any(response: Any) -> str:
    """Extract best-effort text from any OpenAI/Azure response object.

    Delegates to the singleton client's extractor to keep behavior
    consistent across modules.
    """
    try:
        return llm_client._extract_content_safely(response)
    except Exception:
        return ""


def extract_responses_final_text(payload: Dict[str, Any]) -> Optional[str]:
    """Extract the final output_text from a Responses API JSON payload."""
    if not isinstance(payload, dict) or "output" not in payload:
        return None
    for item in reversed(payload.get("output") or []):
        if item.get("type") == "message" and item.get("content"):
            for content in item["content"]:
                if content.get("type") == "output_text":
                    return content.get("text")
    return None


def extract_responses_citations(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract URL citations from a Responses API JSON payload.

    Returns a list of dicts with keys: url, title, start_index, end_index.
    """
    citations: List[Dict[str, Any]] = []
    if not isinstance(payload, dict) or "output" not in payload:
        return citations
    for item in payload.get("output") or []:
        if item.get("type") == "message" and item.get("content"):
            for content in item["content"]:
                if content.get("type") == "output_text" and "annotations" in content:
                    for ann in content.get("annotations") or []:
                        if ann.get("type") == "url_citation":
                            citations.append(
                                {
                                    "url": ann.get("url"),
                                    "title": ann.get("title"),
                                    "start_index": ann.get("start_index"),
                                    "end_index": ann.get("end_index"),
                                }
                            )
    return citations


def extract_responses_tool_calls(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tool call records from a Responses API JSON payload."""
    tool_calls: List[Dict[str, Any]] = []
    if not isinstance(payload, dict) or "output" not in payload:
        return tool_calls
    for item in payload.get("output") or []:
        if item.get("type") in {"web_search_call", "code_interpreter_call", "mcp_call"}:
            tool_calls.append(item)
    return tool_calls


def extract_responses_web_search_calls(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract only web_search_call entries from a Responses payload."""
    return [c for c in extract_responses_tool_calls(payload) if c.get("type") == "web_search_call"]


class ResponsesNormalized(BaseModel):
    """Normalized view of Responses API content we care about."""
    text: Optional[str] = None
    citations: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []
    web_search_calls: List[Dict[str, Any]] = []


def normalize_responses_payload(payload: Dict[str, Any]) -> ResponsesNormalized:
    """Produce a normalized structure from a Responses API payload."""
    return ResponsesNormalized(
        text=extract_responses_final_text(payload),
        citations=extract_responses_citations(payload),
        tool_calls=extract_responses_tool_calls(payload),
        web_search_calls=extract_responses_web_search_calls(payload),
    )


# ────────────────────────────────────────────────────────────
#  Convenience façade: Responses API via OpenAI
#  (Delegates to OpenAIResponsesClient for now)
# ────────────────────────────────────────────────────────────

async def responses_create(
    *,
    model: str,
    input: Union[str, List[Dict[str, Any]]],
    tools: Optional[List[Dict[str, Any]]] = None,
    background: bool = False,
    stream: bool = False,
    reasoning: Optional[Dict[str, str]] = None,
    max_tool_calls: Optional[int] = None,
    instructions: Optional[str] = None,
    store: bool = True,
    previous_response_id: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
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
    )


async def responses_retrieve(response_id: str) -> Dict[str, Any]:
    from services.openai_responses_client import get_responses_client
    return await get_responses_client().retrieve_response(response_id)


async def responses_stream(response_id: str, starting_after: Optional[int] = None) -> AsyncIterator[Dict[str, Any]]:
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
    from services.openai_responses_client import get_responses_client, WebSearchTool, CodeInterpreterTool, MCPTool
    tools: List[Any] = []
    if use_web_search:
        tools.append(web_search_config or WebSearchTool())
    if use_code_interpreter:
        tools.append(CodeInterpreterTool())
    if mcp_servers:
        tools.extend(mcp_servers)

    input_messages: List[Dict[str, Any]] = []
    if system_prompt:
        input_messages.append({
            "role": "developer",
            "content": [{"type": "input_text", "text": system_prompt}],
        })
    input_messages.append({
        "role": "user",
        "content": [{"type": "input_text", "text": query}],
    })

    client = get_responses_client()
    # Resolve model: allow env override; default to o3 for Azure (no deep-research deployment),
    # otherwise prefer o3-deep-research when available
    try:
        deep_model_env = os.getenv("DEEP_RESEARCH_MODEL")
    except Exception:
        deep_model_env = None
    try:
        _rc = get_responses_client()
        is_azure = getattr(_rc, "is_azure", False)
    except Exception:
        is_azure = True
    deep_model = deep_model_env or ("o3" if is_azure else "o3-deep-research")
    if is_azure:
        try:
            # Map to Azure deployment name
            from services.llm_client import llm_client as _lc
            deep_model = _lc._azure_model_for(deep_model)
        except Exception:
            deep_model = os.getenv("AZURE_OPENAI_DEPLOYMENT", deep_model)

    return await client.create_response(
        model=deep_model,
        input=input_messages,
        tools=tools,
        background=background,
        reasoning={"summary": "auto"},
        max_tool_calls=max_tool_calls,
    )
