"""Exa Research API integration.

Provides a lightweight client for invoking Exa's research-focused Responses
API models. The client is intentionally decoupled from the main Search API
manager so we can augment Brave search output with higher-quality research
summaries without altering the existing provider orchestration.
"""

from __future__ import annotations

import json
import os
import structlog
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


logger = structlog.get_logger(__name__)


DEFAULT_BASE_URL = "https://api.exa.ai"
DEFAULT_MODEL = "exa-research"


@dataclass
class ExaResearchOutput:
    """Structured payload returned from Exa research model."""

    summary: str
    key_findings: List[str]
    supplemental_sources: List[Dict[str, Any]]


class ExaResearchClient:
    """Minimal async client for Exa research Responses API."""

    def __init__(self) -> None:
        self.api_key = os.getenv("EXA_API_KEY", "").strip()
        self.base_url = os.getenv("EXA_RESEARCH_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        self.model = os.getenv("EXA_RESEARCH_MODEL", DEFAULT_MODEL)
        self.timeout = float(os.getenv("EXA_RESEARCH_TIMEOUT", "45") or 45)

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def supplement_with_research(
        self,
        query: str,
        brave_highlights: List[Dict[str, str]],
        *,
        focus: Optional[str] = None,
    ) -> Optional[ExaResearchOutput]:
        """Run Exa research to augment Brave search findings.

        Args:
            query: Original user question.
            brave_highlights: Representative Brave results (title/url/snippet).
            focus: Optional extra guidance for Exa (e.g., paradigm-specific).
        Returns:
            ExaResearchOutput on success, or None when disabled/failed.
        """

        if not self.is_configured():
            logger.debug("Exa research client skipped", reason="API key not configured")
            return None

        if not query:
            return None

        instructions = self._build_prompt(query, brave_highlights, focus)
        payload = {
            "model": self.model,
            "input": instructions,
        }

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/responses"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
        except Exception as exc:
            logger.warning("Exa research request failed", error=str(exc))
            return None

        data: Dict[str, Any]
        try:
            data = resp.json()
        except Exception as exc:
            logger.warning("Failed to decode Exa response JSON", error=str(exc))
            return None

        text = self._extract_output_text(data)
        if not text:
            logger.debug("Empty Exa research output_text")
            return None

        parsed = self._parse_json_payload(text)
        if not parsed:
            logger.debug("Exa research output was not valid JSON",
                        raw_text_preview=text[:200])
            return None

        return ExaResearchOutput(
            summary=str(parsed.get("summary", "")).strip(),
            key_findings=[str(item).strip() for item in parsed.get("key_findings", []) if str(item).strip()],
            supplemental_sources=[
                {
                    "title": str(src.get("title", "")).strip(),
                    "url": str(src.get("url", "")).strip(),
                    "snippet": str(src.get("snippet", "")).strip(),
                }
                for src in parsed.get("supplemental_sources", [])
                if isinstance(src, dict)
            ],
        )

    def _build_prompt(
        self,
        query: str,
        highlights: List[Dict[str, str]],
        focus: Optional[str],
    ) -> str:
        """Compose prompt guiding Exa research to complement Brave results."""

        highlight_lines = []
        for idx, item in enumerate(highlights[:5], start=1):
            title = item.get("title") or ""
            url = item.get("url") or ""
            snippet = item.get("snippet") or ""
            highlight_lines.append(
                f"{idx}. Title: {title}\n   URL: {url}\n   Summary: {snippet}"
            )

        context_block = "\n".join(highlight_lines) if highlight_lines else "(No Brave highlights available)"
        focus_line = f"Focus Areas: {focus}" if focus else "Focus Areas: expand with unique insights, data-driven context, and recent developments."

        return (
            "You are an Exa research specialist providing high-precision analysis to complement "
            "general web search findings. Review the Brave search highlights and produce a concise "
            "augmentation that introduces additional perspectives, data points, or sources.\n\n"
            f"Primary Query: {query}\n"
            f"{focus_line}\n\n"
            "Brave Highlights:\n"
            f"{context_block}\n\n"
            "Respond with strict JSON using this schema:\n"
            "{\n"
            "  \"summary\": string,\n"
            "  \"key_findings\": [string, ...],\n"
            "  \"supplemental_sources\": [\n"
            "    {\n"
            "      \"title\": string,\n"
            "      \"url\": string,\n"
            "      \"snippet\": string\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "Do not include commentary outside the JSON object."
        )

    @staticmethod
    def _extract_output_text(payload: Dict[str, Any]) -> str:
        """Best-effort extraction of text from Exa Responses API payload."""

        if not isinstance(payload, dict):
            return ""

        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        output_items = payload.get("output")
        if isinstance(output_items, list):
            buffer: List[str] = []
            for item in output_items:
                try:
                    contents = item.get("content", []) if isinstance(item, dict) else []
                except AttributeError:
                    continue
                for part in contents:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        txt = part.get("text")
                        if isinstance(txt, str):
                            buffer.append(txt)
            if buffer:
                return "\n".join(buffer).strip()

        return ""

    @staticmethod
    def _parse_json_payload(text: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Attempt to salvage JSON by locating first brace pair
            first = text.find("{")
            last = text.rfind("}")
            if first != -1 and last != -1 and last > first:
                snippet = text[first : last + 1]
                try:
                    return json.loads(snippet)
                except Exception:
                    return None
            return None


exa_research_client = ExaResearchClient()

