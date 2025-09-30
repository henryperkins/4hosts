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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx


logger = structlog.get_logger(__name__)


DEFAULT_BASE_URL = "https://api.exa.ai"
DEFAULT_MODEL = "exa-research"
DEFAULT_TIMEOUT_SECONDS = 120.0


@dataclass
class ExaResearchOutput:
    """Structured payload returned from Exa research model."""

    summary: str
    key_findings: List[str] = field(default_factory=list)
    supplemental_sources: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)


class ExaResearchClient:
    """Minimal async client for Exa research Responses API."""

    def __init__(self) -> None:
        self.api_key = os.getenv("EXA_API_KEY", "").strip()
        self.base_url = os.getenv("EXA_RESEARCH_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        self.model = os.getenv("EXA_RESEARCH_MODEL", DEFAULT_MODEL)
        # Default to 120s so we stay within the 90–180s completion window noted in docs
        timeout_raw = os.getenv("EXA_RESEARCH_TIMEOUT", str(int(DEFAULT_TIMEOUT_SECONDS)))
        try:
            self.timeout = float(timeout_raw)
        except (TypeError, ValueError):
            self.timeout = DEFAULT_TIMEOUT_SECONDS

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

        # Build enhanced query with context
        enhanced_query = self._build_query(query, brave_highlights, focus)

        # Exa API /answer endpoint format
        payload = {
            "query": enhanced_query,
            "text": True,  # Include full text content in citations
        }

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        # Use correct Exa endpoint: /answer
        url = f"{self.base_url}/answer"
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

        # Parse Exa's /answer response format
        answer = data.get("answer", "").strip()
        if not answer:
            logger.debug("Empty Exa answer")
            return None

        # Extract citations from Exa response
        citations_raw = data.get("citations", [])
        citations = []
        supplemental_sources = []

        for cite in citations_raw:
            if not isinstance(cite, dict):
                continue

            citation_obj = {
                "title": str(cite.get("title", "")).strip(),
                "url": str(cite.get("url", "")).strip(),
                "note": str(cite.get("text", "")[:200]).strip(),  # Use snippet from text
            }

            if citation_obj["url"]:
                citations.append(citation_obj)

                # Also add to supplemental sources
                supplemental_sources.append({
                    "title": citation_obj["title"],
                    "url": citation_obj["url"],
                    "snippet": citation_obj["note"],
                })

        return ExaResearchOutput(
            summary=answer,
            key_findings=[],  # Exa /answer doesn't provide structured findings
            supplemental_sources=supplemental_sources,
            citations=citations,
        )

    def _build_query(
        self,
        query: str,
        highlights: List[Dict[str, str]],
        focus: Optional[str],
    ) -> str:
        """Build enhanced query for Exa /answer endpoint.

        Exa's /answer endpoint expects a simple query string, not a long prompt.
        We enhance the query with focus context if provided.
        """
        if focus:
            return f"{query} (Focus: {focus})"
        return query


exa_research_client = ExaResearchClient()
