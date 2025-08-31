"""
Shim module providing an `EnhancedResearchOrchestrator` expected by tests.

This is a lightweight, test-focused façade that exposes utility methods for
query enhancement, alignment scoring, and insight extraction used in the
Brave MCP integration tests. It does not replace the production orchestrator
(`services.enhanced_integration`) but complements test coverage.
"""

from __future__ import annotations

import re
from typing import List

from .classification_engine import HostParadigm


class EnhancedResearchOrchestrator:
    """Utility helpers exercised by tests for paradigm-aware behavior."""

    def _enhance_query_for_paradigm(self, query: str, paradigm: HostParadigm) -> str:
        suffix_map = {
            HostParadigm.DOLORES: "expose systemic corruption and reveal injustice",
            HostParadigm.TEDDY: "help people find resources and support",
            HostParadigm.BERNARD: "research evidence with statistical analysis",
            HostParadigm.MAEVE: "define strategy with market and KPI focus",
        }
        return f"{query} {suffix_map.get(paradigm, '').strip()}".strip()

    def _calculate_paradigm_alignment(self, text: str, paradigm: HostParadigm) -> float:
        text_l = (text or "").lower()
        keywords = {
            HostParadigm.DOLORES: [
                "expose", "reveal", "corruption", "injustice", "accountability",
            ],
            HostParadigm.TEDDY: [
                "help", "support", "care", "resources", "community",
            ],
            HostParadigm.BERNARD: [
                "research", "evidence", "statistical", "data", "analysis",
            ],
            HostParadigm.MAEVE: [
                "strategy", "market", "kpi", "roi", "competitive",
            ],
        }.get(paradigm, [])

        if not keywords:
            return 0.0

        hits = sum(1 for k in keywords if k in text_l)
        return min(1.0, hits / len(keywords))

    def _extract_insights(self, synthesis: str, paradigm: HostParadigm) -> List[str]:
        """Extract up to five bullet/numbered insights from a block of text."""
        lines = (synthesis or "").splitlines()
        bullets: List[str] = []
        bullet_re = re.compile(r"^(?:\s*[•\-\*]|\s*\d+[.)])\s+(.*)")
        for ln in lines:
            m = re.match(bullet_re, ln)
            if m:
                val = m.group(1).strip()
                if val:
                    bullets.append(val)
        return bullets[:5]

