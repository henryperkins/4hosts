"""
LLM Client for Four-Hosts Research Application
---------------------------------------------
• Supports Azure OpenAI (Responses API preview) and OpenAI
• Provides chat completions, structured outputs, tool calling, and
  multi-turn conversations with optional SSE streaming + usage metrics
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx
from openai import AsyncOpenAI, AsyncAzureOpenAI
import anthropic
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


_PARADIGM_MODEL_MAP: Dict[str, str] = {
    "dolores": "gpt-4o",
    "teddy": "gpt-4o-mini", 
    "bernard": "gpt-4o",
    "maeve": "gpt-4o-mini",
}

# Add Anthropic Claude model mappings for each paradigm
_PARADIGM_ANTHROPIC_MODEL_MAP: Dict[str, str] = {
    "dolores": "claude-3-5-sonnet-20250123",  # Sonnet 4 for revolutionary analysis
    "teddy": "claude-3-5-sonnet-20250123",    # Sonnet 4 for empathetic care
    "bernard": "claude-3-opus-20250123",      # Opus 4 for analytical rigor
    "maeve": "claude-3-5-sonnet-20250123",   # Sonnet 4 for strategic insights
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


def _select_model(paradigm: str, explicit_model: str | None = None, provider: str = "openai") -> str:
    """Return the model to use, preferring an explicit value when provided."""
    if explicit_model:
        return explicit_model
    
    if provider == "anthropic":
        return _PARADIGM_ANTHROPIC_MODEL_MAP.get(paradigm, "claude-3-5-sonnet-20250123")
    else:
        return _PARADIGM_MODEL_MAP.get(paradigm, "gpt-4o-mini")


def _is_anthropic_model(model: str) -> bool:
    """Check if a model name belongs to Anthropic."""
    return model.startswith("claude-")


def _system_role_for(model: str) -> str:
    """Azure ‘o’ series models expect the special ‘developer’ role for system msgs."""
    return "developer" if model.startswith(("o1", "o3", "o4")) else "system"


# ────────────────────────────────────────────────────────────
#  LLM Client
# ────────────────────────────────────────────────────────────
class LLMClient:
    """Asynchronous client wrapper for OpenAI, Azure OpenAI, and Anthropic."""

    openai_client: Optional[AsyncOpenAI]
    azure_client: Optional[AsyncAzureOpenAI]
    anthropic_client: Optional[anthropic.AsyncAnthropic]

    def __init__(self) -> None:
        self.openai_client = None
        self.azure_client = None
        self.anthropic_client = None
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
            # Ensure endpoint has trailing slash
            if not endpoint.endswith("/"):
                endpoint += "/"
                
            self.azure_client = AsyncAzureOpenAI(
                api_key=azure_key,
                base_url=f"{endpoint}openai/v1/",
                api_version=api_version,
            )
            logger.info("✓ Azure OpenAI client initialised for Responses API")
        else:
            self.azure_client = None
            
        # OpenAI cloud
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.openai_client = AsyncOpenAI(api_key=openai_key)
            logger.info("✓ OpenAI client initialised")
        else:
            self.openai_client = None
            
        # Anthropic Claude
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            self.anthropic_client = anthropic.AsyncAnthropic(api_key=anthropic_key)
            logger.info("✓ Anthropic client initialised")
        else:
            self.anthropic_client = None
            
        if not self.azure_client and not self.openai_client and not self.anthropic_client:
            raise RuntimeError("No LLM client configured - set AZURE_OPENAI_*, OPENAI_API_KEY, or ANTHROPIC_API_KEY environment variables")
    
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
        paradigm: str = "bernard",
        provider: str = "openai",  # "openai" or "anthropic"
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
        stream_include_usage: bool = False,
    ) -> Union[str, AsyncIterator[str]]:
        """
        Chat-style completion with optional SSE stream support.
        Returns either the full response string or an async iterator of tokens.
        """
        self._ensure_initialized()
        
        # Auto-detect provider from model name if not explicitly set
        if model and _is_anthropic_model(model):
            provider = "anthropic"
        
        model_name = _select_model(paradigm, model, provider)

        # Check if we detected an Anthropic model that wasn't explicitly requested
        if _is_anthropic_model(model_name) and provider == "openai":
            provider = "anthropic"
            
        # Handle Anthropic models
        if _is_anthropic_model(model_name):
            return await self._generate_anthropic_completion(
                prompt=prompt,
                model=model_name,
                paradigm=paradigm,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
                stream=stream,
            )

        # Build shared message list
        messages = [
            {
                "role": _system_role_for(model_name),
                "content": _SYSTEM_PROMPTS.get(paradigm, ""),
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

        # ─── Azure OpenAI path (for o3 model) ───
        if model_name in {"o3", "azure"} and self.azure_client:
            try:
                deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "o3-synthesis")
                # Remove Azure-incompatible params
                azure_kwargs = {k: v for k, v in kw.items() if k not in ["stream", "tool_choice", "model"]}
                if tools and tool_choice:
                    azure_kwargs["tool_choice"] = self._wrap_tool_choice(tool_choice)
                    
                # For o3, use the Responses API instead of Chat Completions
                if hasattr(self.azure_client, 'responses'):
                    # Convert messages to input format for Responses API
                    input_content = []
                    for msg in messages:
                        input_content.append({
                            "type": "message",
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                    
                    # Use Responses API for o3
                    # Note: o3 doesn't support temperature/top_p parameters
                    response_kwargs = {
                        "model": deployment,
                        "input": input_content,
                        "max_output_tokens": azure_kwargs.get("max_completion_tokens", azure_kwargs.get("max_tokens", 2000)),
                    }
                    
                    if stream:
                        response_kwargs["stream"] = stream
                    
                    op_res = await self.azure_client.responses.create(**response_kwargs)
                    
                    if stream:
                        return self._iter_responses_stream(op_res)
                    
                    # Extract text from response - Responses API provides output_text directly
                    if hasattr(op_res, 'output_text') and op_res.output_text:
                        return op_res.output_text.strip()
                    
                    # Fallback: extract from output array if output_text not available
                    if hasattr(op_res, 'output') and op_res.output:
                        text_content = ""
                        for output in op_res.output:
                            if hasattr(output, 'content') and output.content:
                                for content_item in output.content:
                                    if hasattr(content_item, 'text') and content_item.type == 'output_text':
                                        text_content += content_item.text
                        if text_content:
                            return text_content.strip()
                    
                    # Log for debugging if no text found
                    logger.warning(f"No text content found in response. Response type: {type(op_res)}")
                    return ""
                else:
                    # Fallback to chat completions if responses API not available
                    op_res = await self.azure_client.chat.completions.create(
                        model=deployment,
                        **azure_kwargs,
                        stream=stream,
                    )
                if stream:
                    return self._iter_openai_stream(op_res)
                content = op_res.choices[0].message.content
                return content.strip() if content else ""
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
                    return self._iter_openai_stream(op_res)
                content = op_res.choices[0].message.content
                return content.strip() if content else ""
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
        paradigm: str = "bernard",
        provider: str = "openai",
    ) -> Dict[str, Any]:
        """Return JSON matching the provided schema."""
        raw = await self.generate_completion(
            prompt,
            model=model,
            paradigm=paradigm,
            provider=provider,
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
        paradigm: str = "bernard",
        provider: str = "openai",
    ) -> Dict[str, Any]:
        """Invoke model with tool-calling enabled and return content + tool_calls."""
        self._ensure_initialized()
        
        # Auto-detect provider from model name if not explicitly set
        if model and _is_anthropic_model(model):
            provider = "anthropic"
            
        model_name = _select_model(paradigm, model, provider)
        
        # Check if we detected an Anthropic model
        if _is_anthropic_model(model_name):
            provider = "anthropic"
            
        # Handle Anthropic tool calling
        if provider == "anthropic" and self.anthropic_client:
            return await self._generate_anthropic_with_tools(
                prompt=prompt,
                tools=tools,
                model=model_name,
                paradigm=paradigm,
            )
        
        result: Dict[str, Any] | None = None

        wrapped_choice = self._wrap_tool_choice(tool_choice)
        
        # Use Azure for o3 model
        if model_name in {"o3", "azure"} and self.azure_client:
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "o3-synthesis")
            op = await self.azure_client.chat.completions.create(
                model=deployment,
                messages=[
                    {
                        "role": _system_role_for(model_name),
                        "content": _SYSTEM_PROMPTS.get(paradigm, ""),
                    },
                    {"role": "user", "content": prompt},
                ],
                tools=tools,
                tool_choice=wrapped_choice,
                max_tokens=2_000,
            )
            result = {
                "content": op.choices[0].message.content or "",
                "tool_calls": op.choices[0].message.tool_calls or [],
            }
        # Use OpenAI
        elif self.openai_client:
            op = await self.openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": _system_role_for(model_name),
                        "content": _SYSTEM_PROMPTS.get(paradigm, ""),
                    },
                    {"role": "user", "content": prompt},
                ],
                tools=tools,
                tool_choice=wrapped_choice,       # ← use string/dict directly for OpenAI
                max_tokens=2_000,
            )
            result = {
                "content": op.choices[0].message.content or "",
                "tool_calls": op.choices[0].message.tool_calls or [],
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
        provider: str = "openai",
        max_tokens: int = 2_000,
        temperature: float = 0.7,
    ) -> str:
        """Multi-turn chat conversation helper (non-streaming)."""
        # Auto-detect provider from model name if not explicitly set
        if model and _is_anthropic_model(model):
            provider = "anthropic"
            
        model_name = _select_model(paradigm, model, provider)
        
        # Check if we detected an Anthropic model
        if _is_anthropic_model(model_name):
            provider = "anthropic"
            
        # Handle Anthropic conversation
        if provider == "anthropic" and self.anthropic_client:
            system_prompt = _SYSTEM_PROMPTS.get(paradigm, "")
            # Convert messages format for Anthropic
            anthropic_messages = []
            for msg in messages:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            try:
                response = await self.anthropic_client.messages.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=anthropic_messages,
                )
                
                content = ""
                for block in response.content:
                    if block.type == "text":
                        content += block.text
                return content.strip()
                
            except Exception as exc:
                logger.error(f"Anthropic conversation request failed • {exc}")
                raise
                
        # Build OpenAI-style messages for Azure/OpenAI  
        full_msgs = [
            {
                "role": _system_role_for(model_name),
                "content": _SYSTEM_PROMPTS.get(paradigm, ""),
            },
            *messages,
        ]

        # Use Azure for o3 model
        if model_name in {"o3", "azure"} and self.azure_client:
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "o3-synthesis")
            op = await self.azure_client.chat.completions.create(
                model=deployment,
                messages=full_msgs,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (op.choices[0].message.content or "").strip()
        
        # Use OpenAI
        elif self.openai_client:
            op = await self.openai_client.chat.completions.create(
                model=model_name,
                messages=full_msgs,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (op.choices[0].message.content or "").strip()

        raise RuntimeError("No LLM back-ends configured for conversation.")

    async def generate_paradigm_content(
        self,
        prompt: str,
        *,
        paradigm: str,
        max_tokens: int = 2_000,
        temperature: float = 0.7,
        model: str | None = None,
        provider: str = "openai",
    ) -> str:
        """Generate content based on a specific paradigm's perspective."""
        return await self.generate_completion(
            prompt=prompt,
            paradigm=paradigm,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
            provider=provider,
            stream=False,
        )


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

    @staticmethod
    async def _iter_anthropic_stream(raw: AsyncIterator[Any]) -> AsyncIterator[str]:
        """Yield token strings from Anthropic stream."""
        async for event in raw:
            if event.type == "content_block_delta":
                if hasattr(event.delta, 'text'):
                    yield event.delta.text

    # ─────────── Anthropic-specific methods ───────────
    async def _generate_anthropic_completion(
        self,
        prompt: str,
        model: str,
        paradigm: str,
        max_tokens: int = 2_000,
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
    ) -> Union[str, AsyncIterator[str]]:
        """Generate completion using Anthropic Claude models."""
        if not self.anthropic_client:
            raise RuntimeError("Anthropic client not configured - set ANTHROPIC_API_KEY environment variable")

        # Build system message and user prompt
        system_prompt = _SYSTEM_PROMPTS.get(paradigm, "")
        
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Add tools if provided
        if tools:
            # Convert OpenAI-style tools to Anthropic format
            anthropic_tools = []
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool.get("function", {})
                    anthropic_tools.append({
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {})
                    })
            kwargs["tools"] = anthropic_tools

        try:
            if stream:
                kwargs["stream"] = True
                response = await self.anthropic_client.messages.create(**kwargs)
                return self._iter_anthropic_stream(response)
            else:
                response = await self.anthropic_client.messages.create(**kwargs)
                # Extract text content from response
                content = ""
                for block in response.content:
                    if block.type == "text":
                        content += block.text
                return content.strip()
                
        except Exception as exc:
            logger.error(f"Anthropic request failed • {exc}")
            raise

    async def _generate_anthropic_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        model: str,
        paradigm: str,
    ) -> Dict[str, Any]:
        """Generate completion with tools using Anthropic Claude models."""
        if not self.anthropic_client:
            raise RuntimeError("Anthropic client not configured")

        # Convert OpenAI-style tools to Anthropic format
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                anthropic_tools.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {})
                })

        system_prompt = _SYSTEM_PROMPTS.get(paradigm, "")
        
        try:
            response = await self.anthropic_client.messages.create(
                model=model,
                max_tokens=2000,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                tools=anthropic_tools,
            )
            
            # Extract content and tool calls
            content = ""
            tool_calls = []
            
            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    # Convert Anthropic tool call to OpenAI format for compatibility
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input)
                        }
                    })
            
            return {
                "content": content.strip(),
                "tool_calls": tool_calls,
            }
            
        except Exception as exc:
            logger.error(f"Anthropic tool calling request failed • {exc}")
            raise



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
            logger.info("✓ LLM client initialized successfully")
        else:
            logger.info("LLM client already initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        return False
