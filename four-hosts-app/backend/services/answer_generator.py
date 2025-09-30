"""
Answer Generation System - Consolidated and Deduped
Combines all answer generation functionality into a single, clean module
"""

import asyncio
import re
import json
import traceback
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
# (no collections needed)

import os
from utils.url_utils import extract_domain
from utils.date_utils import safe_parse_date, iso_or_none
from models.context_models import (
    HostParadigm,
)
from models.synthesis_models import SynthesisContext
from models.paradigms import PARADIGM_KEYWORDS as CANON_PARADIGM_KEYWORDS
from models.paradigms import normalize_to_enum
from services.llm_client import llm_client
from core.config import (
    SYNTHESIS_BASE_WORDS,
    SYNTHESIS_BASE_TOKENS,
    EVIDENCE_MAX_QUOTES_DEFAULT,
    EVIDENCE_MAX_DOCS_DEFAULT,
    EVIDENCE_BUDGET_TOKENS_DEFAULT,
    EVIDENCE_INCLUDE_SUMMARIES,
)
from utils.token_budget import (
    select_items_within_budget,
)
from utils.injection_hygiene import (
    sanitize_snippet,
    guardrail_instruction,
)
import structlog
from functools import lru_cache  # added

logger = structlog.get_logger(__name__)


# ============================================================================
# SHARED PATTERNS AND CONSTANTS
# ============================================================================

STATISTICAL_PATTERNS_OPERATORS = {
    "correlation": r"(?:\br\s*[=:]\s*|correlat\w+\s+(?:of\s+)?)([+-]?\d*\.?\d+)",
    "percentage": r"(\d+(?:\.\d+)?)\s*%",
    "p_value": r"\bp\s*[=<>]\s*([+-]?\d*\.?\d+)",
    "sample_size": r"n\s*=\s*(\d+)",
    "confidence": r"(\d+(?:\.\d+)?)\s*%\s*(?:CI|confidence)",
    "effect_size": r"(?:Cohen's\s*d|effect\s*size)\s*=\s*([+-]?\d*\.?\d+)",
}

STRATEGIC_PATTERN_OPERATIONS = {
    "market_size": r"\$(\d+(?:\.\d+)?)\s*(?:billion|million|B|M)",
    "growth_rate": r"(\d+(?:\.\d+)?)\s*%\s*(?:growth|CAGR|increase)",
    "market_share": r"(\d+(?:\.\d+)?)\s*%\s*(?:market\s*share|of\s*the\s*market)",
    "roi": r"(?:ROI|return)\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%",
    "cost_savings": r"(?:save|reduce\s*costs?)\s*(?:by\s*)?\$?(\d+(?:\.\d+)?)\s*(?:million|thousand|K|m)?",
    "timeline": r"(\d+)\s*(?:months?|years?|quarters?|weeks?)",
}
# Back-compat aliases for existing code paths
STATISTICAL_PATTERNS = STATISTICAL_PATTERNS_OPERATORS
STRATEGIC_PATTERNS = STRATEGIC_PATTERN_OPERATIONS

PARADIGM_KEYWORDS: Dict[HostParadigm, List[str]] = CANON_PARADIGM_KEYWORDS

PARADIGM_THEMES: Dict[str, List[str]] = {
    "bernard": [
        "empirical evidence",
        "methodology",
        "statistical significance",
        "data quality",
        "replication",
        "limitations",
    ],
    "maeve": [
        "competitive advantage",
        "market positioning",
        "roi potential",
        "implementation roadmap",
        "risk mitigation",
        "success metrics",
    ],
    "dolores": [
        "systemic patterns",
        "power structures",
        "documented injustice",
        "accountability",
        "whistleblower evidence",
        "impacted communities",
    ],
    "teddy": [
        "resource availability",
        "accessibility",
        "eligibility criteria",
        "support networks",
        "cost considerations",
        "crisis options",
    ],
}

PARADIGM_CITATION_INSTRUCTIONS: Dict[str, str] = {
    "bernard": (
        "Prioritize peer-reviewed or government sources. Include study design details and sample sizes when available. "
        "Flag methodological limitations explicitly."
    ),
    "maeve": (
        "Prefer industry reports, case studies, and recent analyst briefings. Tie each citation to measurable ROI or timeline data when possible."
    ),
    "dolores": (
        "Cite investigative journalism, primary documents, or whistleblower accounts. Cross-check serious allegations with at least two independent sources."
    ),
    "teddy": (
        "Reference nonprofit and government resources. Provide contact details or application steps when cited resources offer direct support."
    ),
}

PARADIGM_GUARDRAILS: Dict[str, str] = {
    "bernard": "Ensure every claim is supported by quantifiable evidence from the quotes provided.",
    "maeve": "Tie recommendations to actionable metrics and avoid speculative promises without evidence.",
    "dolores": "Corroborate systemic claims with multiple sources and avoid inflammatory language without proof.",
    "teddy": "Verify resource availability dates and include access instructions when suggesting support options.",
}


# ============================================================================
# V1 COMPATIBILITY DATACLASSES (Citation/AnswerSection/GeneratedAnswer)
# Note: SynthesisContext is now unified under models.synthesis_models.SynthesisContext
# ============================================================================


@dataclass
class Citation:
    """V1 citation for backwards compatibility"""
    id: str
    source_title: str
    source_url: str
    domain: str
    snippet: str
    credibility_score: float
    fact_type: str = "reference"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "source_title": self.source_title,
            "source_url": self.source_url,
            "domain": self.domain,
            "snippet": self.snippet,
            "credibility_score": self.credibility_score,
            "fact_type": self.fact_type,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else None,
        }
        return d


@dataclass
class AnswerSection:
    """V1 answer section for backwards compatibility"""
    title: str
    paradigm: str
    content: str
    confidence: float
    citations: List[str]
    word_count: int
    key_insights: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedAnswer:
    """V1 generated answer for backwards compatibility"""
    research_id: str
    query: str
    paradigm: str
    summary: str
    sections: List[AnswerSection]
    action_items: List[Dict[str, Any]]
    citations: Dict[str, Citation]
    confidence_score: float
    synthesis_quality: float
    generation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# CORE DATACLASSES

@dataclass
class StatisticalInsight:
    """Statistical finding for analytical paradigm"""
    metric: str
    value: float
    unit: str
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    sample_size: Optional[int] = None
    context: str = ""


@dataclass
class StrategicRecommendation:
    """Strategic recommendation for strategic paradigm"""
    title: str
    description: str
    impact: str  # high, medium, low
    effort: str  # high, medium, low
    timeline: str
    dependencies: List[str]
    success_metrics: List[str]
    risks: List[str]
    roi_potential: Optional[float] = None


# ----------------------------------------------------------------------------
# BASE GENERATOR CLASS

class BaseAnswerGenerator:
    """Base class for all paradigm-specific generators"""

    def __init__(self, paradigm: str):
        self.paradigm = paradigm
        self.citation_counter = 0
        self.citations = {}

    # Small utility to safely read attributes/keys from mixed objects
    def _qval(self, obj: Any, name: str, default: Any = "") -> Any:
        """Best‑effort getter supporting dicts, attrs, and Pydantic v1/v2.

        This replaces several duplicated local helpers across evidence/context
        formatting blocks. It intentionally swallows lookup errors and returns
        the provided default.
        """
        try:
            if isinstance(obj, dict):
                return obj.get(name, default)
            if hasattr(obj, name):
                return getattr(obj, name)
            if hasattr(obj, "model_dump"):
                # Pydantic v2
                return obj.model_dump().get(name, default)
            if hasattr(obj, "dict"):
                # Pydantic v1
                return obj.dict().get(name, default)
        except Exception:
            return default
        return default

    def _create_citations(self, results: List[Dict[str, Any]], label: str) -> List[str]:
        """Create citations for results and return their IDs."""
        ids: List[str] = []
        for result in results:
            citation = self.create_citation(result, label)
            ids.append(citation.id)
        return ids

    # Centralized progress-tracker resolver (works with either backend)
    def _get_progress_tracker(self):
        try:
            from services.progress import progress as _pt  # type: ignore
            return _pt
        except Exception:
            try:
                from services.websocket_service import progress_tracker as _pt  # type: ignore
                return _pt
            except Exception:
                return None

    def _get_llm_backend_info(self) -> Dict[str, Any]:
        """Return active LLM backend information, if available."""
        try:
            return llm_client.get_active_backend_info()
        except Exception:
            return {}

    # ─────────── Experiments (A/B) helpers ───────────
    def _resolve_prompt_variant(self, context: SynthesisContext, experiment_name: str = "answer_generation_prompt") -> str:
        # Respect explicit option first
        try:
            pv = (getattr(context, "metadata", {}) or {}).get("prompt_variant")
            if isinstance(pv, str) and pv in {"v1", "v2"}:
                return pv
        except Exception:
            pass
        # Deterministic assignment when experiments are enabled
        try:
            from . import experiments  # lazy import
            unit_id = (getattr(getattr(context, "metadata", {}), "get", lambda *_: None)("research_id") or context.query or "").strip() or "anon"
            return experiments.variant_or_default(experiment_name, unit_id, default="v1")
        except Exception:
            return "v1"

    def _variant_addendum(self, context: SynthesisContext, paradigm: str) -> str:
        variant = self._resolve_prompt_variant(context)
        if variant != "v2":
            return ""
        if paradigm == "bernard":
            return (
                "Variant v2 directives:\n"
                "- STRICT: include inline citations for every statistic.\n"
                "- Prefer meta-analyses, RCTs; note sample sizes & CIs.\n"
                "- Avoid speculation; state limitations explicitly.\n"
            )
        if paradigm == "maeve":
            return (
                "Variant v2 directives:\n"
                "- Provide a numbered 30/60/90-day plan with KPIs.\n"
                "- Include risks, mitigations, and resource needs.\n"
                "- Quantify ROI where possible (ranges acceptable).\n"
            )
        if paradigm == "dolores":
            return (
                "Variant v2 directives:\n"
                "- Map power structures and conflicts of interest.\n"
                "- Cite primary sources; flag uncertain claims.\n"
                "- End with 2–3 concrete calls to action.\n"
            )
        if paradigm == "teddy":
            return (
                "Variant v2 directives:\n"
                "- List inclusive resources with eligibility and cost.\n"
                "- Add crisis options and accessibility notes.\n"
                "- Use stigma-free, empowering language.\n"
            )
        return ""

    def _top_relevant_results(self, context: SynthesisContext, k: int = 5) -> List[Dict[str, Any]]:
        """Select top-k sources using credibility, evidence density, and recency with domain diversity.

        Score = 0.6*credibility + 0.25*evidence_density + 0.15*recency, then diversify by domain.
        """
        logger.info(
            "Starting source ranking and selection",
            stage="source_ranking",
            paradigm=self.paradigm,
            requested_k=k,
            total_results=len(context.search_results or []),
        )

        results = list(context.search_results or [])
        # Evidence density from evidence bundle
        ev_counts_by_url: Dict[str, int] = {}
        ev_counts_by_domain: Dict[str, int] = {}
        try:
            eb = getattr(context, "evidence_bundle", None)
            if eb is not None and getattr(eb, "quotes", None):
                quotes = list(getattr(eb, "quotes", []) or [])
                for q in quotes:
                    u = (self._qval(q, "url", "") or "").strip()
                    d = (self._qval(q, "domain", "") or "").lower()
                    if u:
                        ev_counts_by_url[u] = ev_counts_by_url.get(u, 0) + 1
                    if d:
                        ev_counts_by_domain[d] = ev_counts_by_domain.get(d, 0) + 1
        except Exception:
            pass

        from datetime import datetime, timezone
        def _recency_score(v) -> float:
            try:
                dt = safe_parse_date(v)
                if not dt:
                    return 0.0
                now = datetime.now(timezone.utc)
                # 0..1 over ~5 years
                age_days = max(0.0, (now - dt).days)
                return max(0.0, 1.0 - min(1.0, age_days / (365.0 * 5)))
            except Exception:
                return 0.0

        def _domain_of(row: Dict[str, Any]) -> str:
            dom = (row.get("domain") or "").strip().lower()
            if dom:
                return dom
            u2 = (row.get("url") or "").strip()
            return extract_domain(u2)

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for r in results:
            cred = float(r.get("credibility_score", 0.0) or 0.0)
            u = (r.get("url") or "").strip()
            d = _domain_of(r)
            evid = ev_counts_by_url.get(u, 0) or ev_counts_by_domain.get(d, 0)
            evid_norm = min(1.0, evid / 3.0)  # 0..1 for 0..3+ quotes
            rec = _recency_score(r.get("published_date"))
            score = 0.6 * cred + 0.25 * evid_norm + 0.15 * rec
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        picked: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for _score, r in scored:
            dom = _domain_of(r)
            if dom in seen and len(picked) < k // 2:
                continue
            picked.append(r)
            seen.add(dom)
            if len(picked) >= k:
                break

        logger.info(
            "Completed source ranking",
            stage="source_ranking",
            paradigm=self.paradigm,
            record_count=len(picked[:k]),
            unique_domains=len(seen),
            metrics={
                "total_scored": len(scored),
                "selected": len(picked[:k]),
                "domains_seen": list(seen)[:10],
            },
        )
        return picked[:k]

    def _format_evidence_block(self, context: SynthesisContext, max_quotes: int = EVIDENCE_MAX_QUOTES_DEFAULT) -> str:
        quotes = []
        try:
            eb = getattr(context, "evidence_bundle", None)
            if eb is not None and getattr(eb, "quotes", None):
                quotes = list(getattr(eb, "quotes", []) or [])[:max_quotes]
        except Exception as e:
            logger.debug(f"_format_evidence_block: evidence access failed: {e}")
            quotes = []
        lines: List[str] = []
        for q in quotes:
            qid = self._qval(q, "id", "")
            dom = (self._qval(q, "domain", "") or "").lower()
            qt = self._qval(q, "quote", "")
            lines.append(f"- [{qid}][{dom}] {qt}")
        return "\n".join(lines) if lines else "(no evidence quotes)"

    def _safe_evidence_block(
        self,
        context: SynthesisContext,
        max_quotes: int = EVIDENCE_MAX_QUOTES_DEFAULT,
        budget_tokens: int = EVIDENCE_BUDGET_TOKENS_DEFAULT,
        inline_ctx: bool = False,
    ) -> str:
        """Injection-safe, token-budgeted evidence list for prompts.

        When inline_ctx is True, appends a short context window after each quote.
        """
        logger.info(
            "Starting evidence extraction and validation",
            stage="evidence_extraction",
            paradigm=self.paradigm,
            config={
                "max_quotes": max_quotes,
                "budget_tokens": budget_tokens,
                "inline_ctx": inline_ctx,
            },
        )

        # Gather quotes (ensure local is always defined to avoid UnboundLocalError)
        quotes: List[Any] = []
        try:
            eb = getattr(context, "evidence_bundle", None)
            if eb is not None and getattr(eb, "quotes", None):
                quotes = list(getattr(eb, "quotes", []) or [])
        except Exception as e:
            logger.error(
                "Evidence access failed",
                stage="evidence_extraction",
                paradigm=self.paradigm,
                error_type=type(e).__name__,
                stack_trace=traceback.format_exc(),
            )
            quotes = []
        if not quotes:
            return "(no evidence quotes)"
        # Build items compatible with select_items_within_budget (uses estimate_tokens_for_result)
        items_for_budget: List[Dict[str, Any]] = []
        include_ctx = inline_ctx and (os.getenv("EVIDENCE_INCLUDE_CONTEXT", "1") in {"1", "true", "yes"})
        ctx_max = int(os.getenv("EVIDENCE_CONTEXT_MAX_CHARS", "220"))
        for q in quotes:
            items_for_budget.append({
                "id": self._qval(q, "id", ""),
                "domain": self._qval(q, "domain", ""),
                "title": "",           # no title for quotes
                "snippet": "",         # no snippet for quotes
                "content": (self._qval(q, "quote", "") or "") + (f" | ctx: {sanitize_snippet(self._qval(q, 'context_window', '') or '', max_len=ctx_max)}" if include_ctx and self._qval(q, 'context_window', '') else ""),
            })
        selected, _used, _dropped = select_items_within_budget(items_for_budget, max_tokens=budget_tokens)
        selected = selected[:max_quotes]

        logger.info(
            "Evidence validation and selection completed",
            stage="evidence_validation",
            paradigm=self.paradigm,
            record_count=len(selected),
            metrics={
                "total_quotes": len(quotes),
                "selected_quotes": len(selected),
                "tokens_used": _used,
                "items_dropped": _dropped,
            },
        )

        # Sanitize and format
        lines: List[str] = []
        for q in selected:
            qid = q.get("id", "")
            dom = (q.get("domain") or "").lower()
            qt = sanitize_snippet(q.get("content", "") or "")
            lines.append(f"- [{qid}][{dom}] {qt}")
        return "\n".join(lines) if lines else "(no evidence quotes)"

    def _context_windows_block(self, context: SynthesisContext, max_quotes: int = EVIDENCE_MAX_QUOTES_DEFAULT) -> str:
        """Emit a separate block of short windows around evidence quotes.

        Uses EvidenceQuote.context_window built by the evidence builder.
        """
        quotes = []
        try:
            eb = getattr(context, "evidence_bundle", None)
            if eb is not None and getattr(eb, "quotes", None):
                quotes = list(getattr(eb, "quotes", []) or [])[:max_quotes]
        except Exception:
            quotes = []
        if not quotes:
            return "(no context windows)"
        ctx_max = int(os.getenv("EVIDENCE_CONTEXT_MAX_CHARS", "220"))
        lines: List[str] = []
        for q in quotes:
            qid = self._qval(q, "id", "")
            dom = (self._qval(q, "domain", "") or "").lower()
            ctx = self._qval(q, "context_window", "") or ""
            if not ctx:
                continue
            ctx = sanitize_snippet(ctx, max_len=ctx_max)
            lines.append(f"- [{qid}][{dom}] {ctx}")
        return "\n".join(lines) if lines else "(no context windows)"

    def _source_cards_block(self, context: SynthesisContext, k: int = 5) -> str:
        """Source metadata highlighting credibility and paradigm alignment."""
        try:
            results = self._top_relevant_results(context, k)
        except Exception:
            results = list(context.search_results or [])[:k]
        lines: List[str] = []
        from datetime import datetime
        def _fmt_date(v) -> str:
            try:
                if isinstance(v, str):
                    # Trim to date component when iso
                    if 'T' in v:
                        return v.split('T', 1)[0]
                    return v
                if isinstance(v, datetime):
                    return v.date().isoformat()
            except Exception:
                return ""
            return ""
        paradigm_code = (getattr(context, "paradigm", None) or self.paradigm or "bernard").lower()
        for r in results:
            md = r.get("metadata", {}) or {}
            ext = (md.get("extracted_meta") or {}) if isinstance(md, dict) else {}
            title = (r.get("title") or ext.get("title") or "").strip()
            # Prefer explicit domain, else derive from URL; avoid provider name
            domain = (r.get("domain") or "").strip()
            if not domain:
                domain = extract_domain((r.get("url") or "").strip())
            date = (r.get("published_date") or ext.get("published_date") or "")
            date_s = _fmt_date(date)
            authors = ext.get("authors") if isinstance(ext, dict) else None
            if isinstance(authors, list):
                author_s = ", ".join([a for a in authors if a])[:120]
            else:
                author_s = (authors or "")[:120]
            parts = [p for p in [title, domain, date_s, author_s] if p]
            if not parts:
                continue
            try:
                cred = r.get("credibility_score")
            except Exception:
                cred = None
            if cred is None and isinstance(md, dict):
                try:
                    cred = md.get("credibility_score")
                except Exception:
                    cred = None
            try:
                cred_val = float(cred) if cred is not None else None
            except Exception:
                cred_val = None

            align_val = None
            if isinstance(md, dict):
                alignments = md.get("paradigm_alignment")
                if isinstance(alignments, dict):
                    try:
                        align_val = alignments.get(paradigm_code)
                    except Exception:
                        align_val = None
            try:
                align_val = float(align_val) if align_val is not None else None
            except Exception:
                align_val = None

            info_bits: List[str] = []
            if cred_val is not None:
                info_bits.append(f"Credibility: {cred_val:.2f}")
            if align_val is not None:
                info_bits.append(f"{paradigm_code.title()} Alignment: {align_val:.2f}")

            base = " | ".join(parts)
            suffix = f" | {'; '.join(info_bits)}" if info_bits else ""
            lines.append(f"- {base}{suffix}")
        return "\n".join(lines) if lines else "(no source metadata)"

    def _source_summaries_block(
        self,
        context: SynthesisContext,
        max_items: int = EVIDENCE_MAX_DOCS_DEFAULT,
        budget_tokens: int = EVIDENCE_BUDGET_TOKENS_DEFAULT,
    ) -> str:
        """Optional block of per‑source summaries (deduped by URL/domain).

        Summaries can be attached by the evidence builder on each quote as
        "doc_summary". We dedupe by URL, falling back to domain, and then
        select within a token budget to avoid overflow.
        """
        try:
            eb = getattr(context, "evidence_bundle", None)
            documents = list(getattr(eb, "documents", []) or []) if eb is not None else []
        except Exception as e:
            logger.debug(f"_source_summaries_block: documents access failed: {e}")
            documents = []

        def _dval(doc: Any, name: str, default: Any = "") -> Any:
            try:
                if isinstance(doc, dict):
                    return doc.get(name, default)
                if hasattr(doc, name):
                    return getattr(doc, name)
                if hasattr(doc, "model_dump"):
                    return doc.model_dump().get(name, default)
            except Exception:
                return default
            return default

        if documents:
            items_for_budget: List[Dict[str, Any]] = []
            for doc in documents:
                doc_id = _dval(doc, "id", "") or f"d{len(items_for_budget)+1:03d}"
                title = _dval(doc, "title", "")
                domain = _dval(doc, "domain", "")
                token_count = int(_dval(doc, "token_count", 0) or 0)
                content = _dval(doc, "content", "")
                if not content:
                    continue
                header_bits = [f"[{doc_id}]"]
                if title:
                    header_bits.append(title)
                if domain:
                    header_bits.append(f"({domain})")
                if token_count:
                    header_bits.append(f"≈{token_count} tokens")
                header = " ".join(header_bits)
                items_for_budget.append({
                    "title": header,
                    "snippet": content,
                    "_meta": {
                        "id": doc_id,
                        "header": header,
                        "token_count": token_count,
                    },
                })

            if items_for_budget:
                try:
                    env_budget = os.getenv("EVIDENCE_DOCUMENT_BUDGET_TOKENS")
                    if env_budget is not None:
                        budget_tokens = int(env_budget or budget_tokens)
                except Exception:
                    pass
                selected, _used, _dropped = select_items_within_budget(
                    items_for_budget,
                    max_tokens=budget_tokens,
                    per_item_min_tokens=200,
                )
                selected = selected[:max_items]
                blocks: List[str] = []
                for it in selected:
                    meta = it.get("_meta", {})
                    header = meta.get("header") or it.get("title") or ""
                    content = it.get("snippet") or ""
                    if not content:
                        continue
                    blocks.append(f"{header}\n{content}")
                if blocks:
                    return "\n---\n".join(blocks)

        # Fallback to legacy summary mode using quote-attached summaries
        quotes = []
        try:
            eb = getattr(context, "evidence_bundle", None)
            quotes = list(getattr(eb, "quotes", []) or []) if eb is not None else []
        except Exception:
            quotes = []
        if not quotes:
            return "(no source summaries)"

        # Collate first summary per URL
        seen: set[str] = set()
        items_for_budget: List[Dict[str, Any]] = []
        for q in quotes:
            url = (self._qval(q, "url", "") or "").strip()
            dom = (self._qval(q, "domain", "") or "").lower()
            sid = url or dom
            if not sid or sid in seen:
                continue
            seen.add(sid)
            summary = self._qval(q, "doc_summary", "") or ""
            if not summary:
                continue
            items_for_budget.append({
                "id": sid,
                "domain": dom,
                "title": "",
                "snippet": "",
                "content": f"{dom}: {summary}",
            })

        if not items_for_budget:
            return "(no source summaries)"
        # Allow env override for budget tokens
        try:
            budget_tokens = int(os.getenv("EVIDENCE_SUMMARY_BUDGET_TOKENS", str(budget_tokens)))
        except Exception:
            pass
        selected, _u, _d = select_items_within_budget(items_for_budget, max_tokens=budget_tokens)
        selected = selected[:max_items]
        lines: List[str] = []
        for it in selected:
            # Format as: - domain (YYYY-MM-DD): summary
            content = it.get('content','') or ''
            lines.append(f"- {sanitize_snippet(content)}")
        return "\n".join(lines) if lines else "(no source summaries)"

    def _coverage_table(self, context: SynthesisContext, max_rows: int = 6) -> str:
        # Use focus areas from EvidenceBundle and supplement with paradigm defaults
        focus: List[str] = []
        try:
            eb = getattr(context, "evidence_bundle", None)
            if eb is not None and getattr(eb, "focus_areas", None):
                focus = [str(f) for f in (getattr(eb, "focus_areas", []) or []) if f]
        except Exception:
            focus = []

        paradigm_code = (getattr(context, "paradigm", None) or self.paradigm or "bernard").lower()
        default_themes = PARADIGM_THEMES.get(paradigm_code, PARADIGM_THEMES.get("bernard", []))

        combined: List[str] = []
        seen = set()
        for theme in focus:
            norm = theme.strip()
            if not norm:
                continue
            lowered = norm.lower()
            if lowered in seen:
                continue
            combined.append(norm)
            seen.add(lowered)
        for theme in default_themes:
            lowered = theme.lower()
            if lowered in seen:
                continue
            combined.append(theme)
            seen.add(lowered)

        if not combined:
            return "(no coverage targets)"

        focus_limited = combined[:max_rows]
        # Pick best URL per theme by token overlap
        rows: List[str] = ["Theme | Covered? | Best Domain"]
        def _tok(t: str) -> set:
            import re
            return set([w for w in re.findall(r"[A-Za-z0-9]+", (t or "").lower()) if len(w) > 2])
        results_for_scan = list(context.search_results or [])
        for theme in focus_limited:
            tt = _tok(theme)
            best = None
            best_score = 0.0
            for r in results_for_scan[:20]:
                text = f"{r.get('title','')} {r.get('snippet','')}"
                st = _tok(text)
                if not st:
                    continue
                j = len(tt & st) / float(len(tt | st)) if (tt or st) else 0.0
                if j > best_score:
                    best_score = j
                    best = r
            covered = "yes" if best_score >= 0.5 else ("partial" if best_score >= 0.25 else "no")
            dom = ""
            if isinstance(best, dict) and best:
                dom = (best.get("domain") or "")
                if not dom:
                    dom = extract_domain((best.get("url") or "").strip())
                    if not dom:  # Handle empty domain case
                        dom = ""
            rows.append(f"{theme} | {covered} | {dom}")
        return "\n".join(rows)

    def _get_isolated_findings(self, context: SynthesisContext) -> Dict[str, Any]:
        """Return isolation findings from canonical EvidenceBundle when available."""
        try:
            if getattr(context, "evidence_bundle", None):
                eb = context.evidence_bundle
                return {
                    "matches": list(getattr(eb, "matches", []) or []),
                    "by_domain": dict(getattr(eb, "by_domain", {}) or {}),
                    "focus_areas": list(getattr(eb, "focus_areas", []) or []),
                }
        except Exception:
            return {"matches": [], "by_domain": {}, "focus_areas": []}

    def create_citation(self, source: Dict[str, Any], fact_type: str = "reference") -> Citation:
        """Create a citation from a source"""
        self.citation_counter += 1
        citation_id = f"cite_{self.citation_counter:03d}"

        logger.debug(
            "Creating citation",
            stage="citation_creation",
            paradigm=self.paradigm,
            citation_id=citation_id,
            fact_type=fact_type,
            source_domain=source.get("domain", ""),
        )

        # Normalize timestamp and preserve raw if needed
        meta = dict(source.get("metadata", {}) or {})
        ts_val = source.get("published_date")
        ts_norm: Optional[datetime] = None
        if isinstance(ts_val, datetime):
            ts_norm = ts_val
        elif isinstance(ts_val, str):
            try:
                # Try simple ISO formats
                ts_norm = safe_parse_date(ts_val)
            except Exception:
                meta["published_date_raw"] = ts_val

        citation = Citation(
            id=citation_id,
            source_title=source.get("title", ""),
            source_url=source.get("url", ""),
            domain=source.get("domain", ""),
            snippet=source.get("snippet", ""),
            credibility_score=float(source.get("credibility_score", 0.5)),
            fact_type=fact_type,
            metadata=meta,
            timestamp=ts_norm
        )

        self.citations[citation_id] = citation
        return citation

    async def generate_answer(self, context: SynthesisContext) -> GeneratedAnswer:
        """Generate answer - must be implemented by subclasses"""
        logger.info(
            "Answer generation requested (base class)",
            stage="answer_generation",
            paradigm=self.paradigm,
            query=(context.query[:100] if context.query else "N/A"),
        )
        raise NotImplementedError("Subclasses must implement generate_answer")

    def get_section_structure(self) -> List[Dict[str, Any]]:
        """Get section structure - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement get_section_structure")

    def _get_alignment_keywords(self) -> List[str]:
        """Get paradigm alignment keywords"""
        paradigm_enum = normalize_to_enum(self.paradigm)
        if paradigm_enum is None:
            return []
        # Avoid passing Optional[HostParadigm] into dict.get (Pylance typing)
        keywords = PARADIGM_KEYWORDS.get(paradigm_enum)
        return keywords if keywords is not None else []

    # ───────────────────────────────────────────────────────────────────
    #  Claim↔Source Mapping and Top‑line Recommendation helpers
    # ───────────────────────────────────────────────────────────────────
    def _build_claim_source_map(
        self,
        sections: List["AnswerSection"],
        citations: Dict[str, "Citation"],
        max_claims: int = 12,
    ) -> List[Dict[str, Any]]:
        """Create a simple mapping of each section's key insights to its citations.

        The UI can render this as a compact table. Claims are de‑duplicated and
        truncated for readability.
        """
        seen: set[str] = set()
        rows: List[Dict[str, Any]] = []
        for sec in sections:
            insight_list = list(getattr(sec, "key_insights", []) or [])
            cite_ids = list(getattr(sec, "citations", []) or [])
            mapped_cites: List[Dict[str, Any]] = []
            for cid in cite_ids:
                c = citations.get(cid)
                if not c:
                    continue
                mapped_cites.append({
                    "id": c.id,
                    "title": c.source_title,
                    "url": c.source_url,
                    "domain": c.domain,
                })
            for claim in insight_list:
                key = claim.strip().lower()[:300]
                if not key or key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "claim": claim.strip()[:500],
                    "citations": mapped_cites,
                    "section": getattr(sec, "title", "")
                })
                if len(rows) >= max_claims:
                    return rows
        return rows

    def _compose_topline_recommendation(self, context: SynthesisContext) -> str:
        """Produce a concise (2–3 sentences) recommendation block without extra LLM calls."""
        results = list(context.search_results or [])
        def _dom(row: Dict[str, Any]) -> str:
            d = (row.get("domain") or "").lower()
            if d:
                return d
            return extract_domain((row.get("url") or "").strip())
        domains = { _dom(r) for r in results }
        n = len(results)
        d = len([x for x in domains if x])
        if n >= 12 and d >= 6:
            evidence = "moderately strong and diverse"
        elif n >= 6 and d >= 3:
            evidence = "adequate but mixed"
        else:
            evidence = "limited"

        recs = []
        # Baseline recommendation informed by paradigm
        if self.paradigm == "bernard":
            recs.append(
                f"Evidence coverage appears {evidence}. Prioritize broader query variants and lower relevance thresholds when recall is low; add synonyms and related concepts to increase yield."
            )
            recs.append(
                "Use citation-backed claims only; link each key statement to at least one source and resolve conflicts via cross-domain agreement."
            )
        elif self.paradigm == "maeve":
            recs.append(
                f"Evidence coverage appears {evidence}. Focus on high-ROI tactics first and timebox deeper investigation to de-risk decisions."
            )
            recs.append("Track success with 2–3 measurable KPIs and revisit quarterly.")
        elif self.paradigm == "dolores":
            recs.append(
                f"Evidence coverage appears {evidence}. Center verified primary sources and corroborate patterns across independent domains."
            )
            recs.append("Document systemic patterns with clear, sourced examples to drive accountability.")
        else:  # teddy
            recs.append(
                f"Evidence coverage appears {evidence}. Curate practical resources from trusted domains and prioritize accessibility."
            )
            recs.append("Offer next-step guidance and local options where available.")
        return " " .join(recs)

    async def _maybe_dynamic_actions(
        self,
        context: SynthesisContext,
        fallback_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Optionally generate dynamic, LLM-powered action items (behind ENABLE_DYNAMIC_ACTIONS).
        Returns fallback_items when disabled or on failure."""
        try:
            from services.action_items import (  # type: ignore
                enabled as _actions_enabled,
                generate_action_items as _gen_actions,
            )
        except Exception:
            return fallback_items

        try:
            if not _actions_enabled():
                return fallback_items

            # Grounding: prefer evidence bundle quotes, fall back to context.evidence_quotes
            search_results = list(getattr(context, "search_results", []) or [])
            try:
                eb = getattr(context, "evidence_bundle", None)
                raw_quotes = list(getattr(eb, "quotes", []) or []) if eb is not None else []
                # Normalize to plain dicts for downstream action generator
                evidence_quotes = []
                for q in raw_quotes:
                    if isinstance(q, dict):
                        evidence_quotes.append(q)
                    elif hasattr(q, "model_dump"):
                        evidence_quotes.append(q.model_dump())
                    elif hasattr(q, "dict"):
                        evidence_quotes.append(q.dict())

            except Exception:
                evidence_quotes = []

            items = await _gen_actions(
                query=getattr(context, "query", ""),
                paradigm=self.paradigm,
                search_results=search_results,
                evidence_quotes=evidence_quotes,
            )
            return items or fallback_items
        except Exception as e:
            logger.debug(f"Dynamic action items skipped: {e}")
            return fallback_items

    # ============================= Dedup Helpers =============================
    def build_iso_block(self, context: SynthesisContext, max_matches: int = 5) -> str:
        """Unified isolated findings formatter (replaces repeated iso_lines blocks)."""
        try:
            iso = self._get_isolated_findings(context) or {}
            matches = list((iso.get("matches") or [])[:max_matches])
        except Exception:
            matches = []
        lines: List[str] = []
        for m in matches:
            try:
                dom = (m.get("domain") if isinstance(m, dict) else getattr(m, "domain", "")) or ""
                frags = (m.get("fragments") if isinstance(m, dict) else getattr(m, "fragments", [])) or []
                if not isinstance(frags, (list, tuple)):
                    frags = [frags]
                for frag in list(frags)[:1]:
                    lines.append(f"- [{dom}] {sanitize_snippet(frag or '')}")
            except Exception:
                continue
        return "\n".join(lines) if lines else "(no isolated findings)"

    def decide_context_mode(self, section_weight: float) -> Tuple[int, bool, bool]:
        """Return (section_tokens, inline_ctx, use_windows) based on env & weight."""
        section_tokens = int(SYNTHESIS_BASE_TOKENS * section_weight)
        mode = (os.getenv("EVIDENCE_CONTEXT_MODE", "auto") or "auto").lower()
        threshold = int(os.getenv("EVIDENCE_CONTEXT_SECTION_TOKENS_MIN", "1500"))
        inline_ctx = False
        use_windows = False
        if mode == "inline":
            inline_ctx = True
        elif mode == "window":
            use_windows = True
        elif mode == "auto":
            if section_tokens >= threshold:
                inline_ctx = True
        return section_tokens, inline_ctx, use_windows

    async def update_progress_step(
        self,
        progress_tracker,
        research_id: Optional[str],
        section_name: str,
        step_idx: int,
        total_steps: int,
        label: str,
    ) -> None:
        if not (progress_tracker and research_id):
            return
        try:
            await progress_tracker.update_progress(
                research_id,
                phase="synthesis",
                message=f"{section_name}: {label}",
                items_done=step_idx,
                items_total=total_steps,
            )
        except Exception:
            logger.debug("progress_update_failed", section=section_name, step=step_idx)

    def render_generic_fallback(
        self,
        section_def: Dict[str, Any],
        results: List[Dict[str, Any]],
        intro: Optional[str] = None,
    ) -> str:
        """Generic fallback text summarising a few sources."""
        title = section_def.get("title", "Section")
        focus = section_def.get("focus", "")
        content = f"# {title}\n\n" + (intro or f"This section focuses on: {focus}") + "\n\n"
        if results:
            content += "## Source Highlights\n\n"
            for r in results[:3]:
                dom = r.get("domain") or extract_domain(r.get("url", "") or "") or "source"
                snippet = r.get("snippet") or r.get("title") or "(no snippet)"
                content += f"- {dom}: {sanitize_snippet(snippet)[:220]}\n"
        else:
            content += "(no sources available)\n"
        return content

    def get_relevant_results_and_citations(self, context: SynthesisContext, k: int, label: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        results = self._top_relevant_results(context, k)
        citation_ids = self._create_citations(results, label)
        return results, citation_ids

    @lru_cache(maxsize=1)
    def _get_jinja_env(self):
        try:
            import jinja2  # type: ignore
            return jinja2.Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)
        except Exception:
            return None

    def build_prompt(
        self,
        *,
        paradigm: str,
        section_title: str,
        section_focus: str,
        query: str,
        source_cards: str,
        evidence_block: str,
        ctx_windows_block: str,
        iso_block: str,
        coverage_tbl: str,
        summaries_block: str,
        target_words: int,
        variant_extra: str,
        paradigm_directives: List[str],
        extra_requirements: Optional[List[str]] = None,
    ) -> str:
        """Shared prompt scaffolding with optional Jinja2 (keeps old formatting)."""
        guard = guardrail_instruction
        paradigm_code = (paradigm or self.paradigm or "bernard").lower()
        guard_extra = PARADIGM_GUARDRAILS.get(paradigm_code)
        if guard_extra:
            guard = f"{guard}\n{guard_extra}"
        citation_guidelines = PARADIGM_CITATION_INSTRUCTIONS.get(paradigm_code, "")
        extra_requirements = extra_requirements or []
        ctx_windows_line = (
            f"Context Windows (for quotes):\n{ctx_windows_block}" if ctx_windows_block and "disabled" not in ctx_windows_block else ""
        )
        tmpl = """{{ guard }}\nWrite the "{{ section_title }}" section focusing on: {{ section_focus }}\nQuery: {{ query }}\nSource Metadata (assess credibility before citing):\n{{ source_cards }}\nEvidence Quotes (primary evidence; cite by [qid]):\n{{ evidence_block }}\n{% if ctx_windows_line %}{{ ctx_windows_line }}\n{% endif %}Isolated Findings:\n{{ iso_block }}\nCoverage Table (Theme | Covered? | Best Domain):\n{{ coverage_tbl }}\nFull Document Context (cite using [d###]):\n{{ summaries_block }}\n\nParadigm Directives:\n{% for d in paradigm_directives %}- {{ d }}\n{% endfor %}Additional Requirements:\n{% for r in extra_requirements %}- {{ r }}\n{% endfor %}{% if citation_guidelines %}Citation Guidelines:\n{{ citation_guidelines }}\n{% endif %}STRICT: Do not fabricate claims beyond the evidence quotes above.\n{{ variant_extra }}\nLength: {{ target_words }} words\n"""
        env = self._get_jinja_env()
        if env:
            return env.from_string(tmpl).render(
                guard=guard,
                section_title=section_title,
                section_focus=section_focus,
                query=query,
                source_cards=source_cards,
                evidence_block=evidence_block,
                ctx_windows_line=ctx_windows_line,
                iso_block=iso_block,
                coverage_tbl=coverage_tbl,
                summaries_block=summaries_block,
                target_words=target_words,
                variant_extra=variant_extra,
                paradigm_directives=paradigm_directives,
                extra_requirements=extra_requirements,
                citation_guidelines=citation_guidelines,
            )
        # Fallback f-string assembly
        directives_txt = "\n".join(f"- {d}" for d in paradigm_directives)
        reqs_txt = "\n".join(f"- {r}" for r in extra_requirements)
        citation_block = (
            f"Citation Guidelines:\n{citation_guidelines}\n" if citation_guidelines else ""
        )
        return (
            f"{guard}\nWrite the \"{section_title}\" section focusing on: {section_focus}\nQuery: {query}\n"
            f"Source Metadata (assess credibility before citing):\n{source_cards}\nEvidence Quotes (primary evidence; cite by [qid]):\n{evidence_block}\n"
            f"{ctx_windows_line + '\n' if ctx_windows_line else ''}Isolated Findings:\n{iso_block}\nCoverage Table (Theme | Covered? | Best Domain):\n{coverage_tbl}\n"
            f"Full Document Context (cite using [d###]):\n{summaries_block}\n\nParadigm Directives:\n{directives_txt}\n\nAdditional Requirements:\n{reqs_txt}\n"
            f"{citation_block}"
            f"STRICT: Do not fabricate claims beyond the evidence quotes above.\n{variant_extra}\nLength: {target_words} words\n"
        )

# ============================================================================
# PARADIGM-SPECIFIC GENERATORS

class DoloresAnswerGenerator(BaseAnswerGenerator):
    """Revolutionary paradigm answer generator"""

    def __init__(self):
        super().__init__("dolores")

    def get_section_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "title": "Exposing the System",
                "focus": "Reveal systemic issues and power structures",
                "weight": 0.3,
            },
            {
                "title": "Voices of the Oppressed",
                "focus": "Highlight victim testimonies and impacts",
                "weight": 0.25,
            },
            {
                "title": "Pattern of Injustice",
                "focus": "Document recurring patterns and systemic failures",
                "weight": 0.25,
            },
            {
                "title": "Path to Revolution",
                "focus": "Outline resistance strategies and calls to action",
                "weight": 0.2,
            },
        ]

    async def generate_answer(self, context: SynthesisContext) -> GeneratedAnswer:
        """Generate revolutionary paradigm answer"""
        start_time = datetime.now()
        research_id = context.metadata.get('research_id') if hasattr(context, 'metadata') else 'unknown'

        logger.info(
            "Starting answer generation",
            stage="answer_generation_start",
            paradigm="dolores",
            research_id=research_id,
            query=(context.query[:100] if context.query else "N/A"),
            config={
                "evidence_quotes": (
                    len(getattr(context.evidence_bundle, "quotes", []))
                    if hasattr(context, "evidence_bundle")
                    else 0
                ),
                "search_results": len(context.search_results) if context.search_results else 0,
            },
        )

        self.citation_counter = 0
        self.citations = {}

        sections = []
        section_metrics = []
        sections_total = len(self.get_section_structure())

        # Report synthesis started
        progress_tracker = self._get_progress_tracker()
        if progress_tracker and context.research_id:
            await progress_tracker.update_progress(
                context.research_id, phase="synthesis",
                message="Starting answer synthesis",
                items_done=0, items_total=sections_total
            )

        for idx, section_def in enumerate(self.get_section_structure()):
            section_start = datetime.now()

            # Report section being drafted
            if progress_tracker and context.research_id:
                section_title = section_def.get('title', 'Section')[:40]
                await progress_tracker.update_progress(
                    context.research_id, phase="synthesis",
                    message=f"Drafting {section_title}",
                    items_done=idx, items_total=sections_total
                )

            try:
                section = await self._generate_section(context, section_def)
                sections.append(section)

                # Report section completed
                if progress_tracker and context.research_id:
                    await progress_tracker.update_progress(
                        context.research_id, phase="synthesis",
                        items_done=idx+1, items_total=sections_total
                    )

                duration = (datetime.now() - section_start).total_seconds() * 1000

                logger.info(
                    "Section generation completed",
                    stage="section_generation",
                    paradigm="dolores",
                    section_title=section_def.get('title'),
                    section_index=idx,
                    duration_ms=duration,
                    metrics={
                        "word_count": section.word_count,
                        "citations": len(section.citations),
                        "insights": len(section.key_insights),
                    },
                )
                section_metrics.append({"title": section_def.get('title'), "duration_ms": duration})
            except Exception as e:
                logger.error(
                    "Section generation failed",
                    stage="section_generation_error",
                    paradigm="dolores",
                    section_title=section_def.get('title'),
                    error_type=type(e).__name__,
                    stack_trace=traceback.format_exc(),
                )
                if os.getenv("LLM_STRICT", "0") == "1":
                    raise

        # Top‑line recommendation (2–3 sentences) then short summary
        topline = self._compose_topline_recommendation(context)
        base_summary = sections[0].content[:300] + "..." if sections else ""
        summary = (topline + "\n\n" + base_summary).strip()

        total_duration = (datetime.now() - start_time).total_seconds()

        logger.info(
            "Answer generation completed",
            stage="answer_generation_complete",
            paradigm="dolores",
            research_id=research_id,
            duration_ms=total_duration * 1000,
            metrics={
                "total_sections": len(sections),
                "total_citations": len(self.citations),
                "confidence_score": 0.8,
                "synthesis_quality": 0.85,
                "section_metrics": section_metrics,
            },
        )

        return GeneratedAnswer(
            research_id=context.metadata.get("research_id", "unknown"),
            query=context.query,
            paradigm=self.paradigm,
            summary=summary,
            sections=sections,
            action_items=await self._maybe_dynamic_actions(context, self._generate_action_items(context)),
            citations=self.citations,
            confidence_score=0.8,
            synthesis_quality=0.85,
            generation_time=(datetime.now() - start_time).total_seconds(),
            metadata={
                "tone": "investigative",
                "focus": "exposing injustice",
                "topline_recommendation": topline,
                "claim_source_map": self._build_claim_source_map(sections, self.citations),
            }
        )

    async def _generate_section(self, context: SynthesisContext, section_def: Dict[str, Any]) -> AnswerSection:
        research_id = context.metadata.get("research_id") if hasattr(context, "metadata") else None
        progress_tracker = self._get_progress_tracker()
        section_name = section_def.get('title', 'Section')
        total_sub_ops = 4
        await self.update_progress_step(progress_tracker, research_id, section_name, 0, total_sub_ops, "Filtering relevant sources")
        relevant_results, citation_ids = self.get_relevant_results_and_citations(context, 5, "evidence")
        await self.update_progress_step(progress_tracker, research_id, section_name, 1, total_sub_ops, "Creating citations")
        await self.update_progress_step(progress_tracker, research_id, section_name, 2, total_sub_ops, "Generating content")
        try:
            iso_block = self.build_iso_block(context)
            _section_tokens, inline_ctx, use_windows = self.decide_context_mode(section_def['weight'])
            evidence_block = self._safe_evidence_block(context, inline_ctx=inline_ctx)
            ctx_windows_block = self._context_windows_block(context) if use_windows else "(context windows disabled)"
            summaries_block = self._source_summaries_block(context) if EVIDENCE_INCLUDE_SUMMARIES else "(summaries disabled)"
            coverage_tbl = self._coverage_table(context)
            source_cards = self._source_cards_block(context) if os.getenv("SOURCE_CARDS_ENABLE", "1") in {"1", "true", "yes"} else "(source cards disabled)"
            variant_extra = self._variant_addendum(context, "dolores")
            prompt = self.build_prompt(
                paradigm="dolores",
                section_title=section_def['title'],
                section_focus=section_def['focus'],
                query=context.query,
                source_cards=source_cards,
                evidence_block=evidence_block,
                ctx_windows_block=ctx_windows_block,
                iso_block=iso_block,
                coverage_tbl=coverage_tbl,
                summaries_block=summaries_block,
                target_words=int(SYNTHESIS_BASE_WORDS * section_def['weight']),
                variant_extra=variant_extra,
                paradigm_directives=[
                    "Use passionate, urgent language exposing injustice",
                    "Ground every factual claim in evidence quotes",
                    "Highlight structural power dynamics",
                ],
                extra_requirements=[],
            )
            content = await llm_client.generate_paradigm_content(prompt=prompt, paradigm="dolores")
        except Exception as e:
            logger.error("LLM generation failed", stage="llm_generation_error", paradigm="dolores", section=section_def['title'], error_type=type(e).__name__, stack_trace=traceback.format_exc())
            if os.getenv("LLM_STRICT", "0") == "1":
                raise
            logger.warning(f"LLM generation failed: {e}, using fallback")
            content = self._generate_fallback_content(section_def, relevant_results)
        await self.update_progress_step(progress_tracker, research_id, section_name, 3, total_sub_ops, "Extracting insights")
        section = AnswerSection(
            title=section_def['title'],
            paradigm=self.paradigm,
            content=content,
            confidence=0.75,
            citations=citation_ids,
            word_count=len(content.split()),
            key_insights=self._extract_insights(content),
            metadata={"section_weight": section_def['weight']}
        )
        await self.update_progress_step(progress_tracker, research_id, section_name, total_sub_ops, total_sub_ops, "Section complete")
        return section

    def _generate_fallback_content(self, section_def: Dict[str, Any], results: List[Dict[str, Any]]) -> str:
        return self.render_generic_fallback(section_def, results)

class BernardAnswerGenerator(BaseAnswerGenerator):
    """Analytical paradigm answer generator with enhanced statistical analysis"""

    def __init__(self):
        super().__init__("bernard")

    def get_section_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "title": "Executive Summary",
                "focus": "Key findings, statistical overview, and evidence quality assessment",
                "weight": 0.15,
            },
            {
                "title": "Quantitative Analysis",
                "focus": "Statistical data, trends, correlations, and empirical patterns",
                "weight": 0.25,
            },
            {
                "title": "Causal Mechanisms",
                "focus": "Identified causal relationships, mediating variables, and effect sizes",
                "weight": 0.20,
            },
            {
                "title": "Methodological Assessment",
                "focus": "Research design evaluation, bias analysis, and validity threats",
                "weight": 0.15,
            },
            {
                "title": "Evidence Synthesis",
                "focus": "Meta-analytical insights and cross-study comparisons",
                "weight": 0.15,
            },
            {
                "title": "Research Implications",
                "focus": "Knowledge gaps, future directions, and practical applications",
                "weight": 0.10,
            },
        ]

    async def generate_answer(self, context: SynthesisContext) -> GeneratedAnswer:
        """Generate analytical answer with statistical insights"""
        logger.info(
            "Starting Bernard answer generation",
            research_id=context.metadata.get('research_id'),
            stage="bernard_answer_generation_start",
        )
        start_time = datetime.now()
        self.citation_counter = 0
        self.citations = {}

        # Get progress tracker if available
        research_id = context.metadata.get("research_id") if hasattr(context, "metadata") else None
        progress_tracker = self._get_progress_tracker()

        # Extract statistical insights from search results
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="synthesis",
                message="Bernard: Extracting statistical patterns"
            )
        statistical_insights = self._extract_statistical_insights(context.search_results)

        # Perform meta-analysis if possible
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="synthesis",
                message="Bernard: Performing meta-analysis"
            )
        meta_analysis = self._perform_meta_analysis(context.search_results)

        # Generate sections
        sections = []
        section_metrics = []
        for idx, section_def in enumerate(self.get_section_structure()):
            section_start = datetime.now()
            section = await self._generate_analytical_section(
                context, section_def, statistical_insights, meta_analysis
            )
            sections.append(section)
            duration = (datetime.now() - section_start).total_seconds() * 1000
            section_metrics.append({"title": section_def.get('title'), "duration_ms": duration})

        topline = self._compose_topline_recommendation(context)
        summary_core = self._generate_analytical_summary(sections, statistical_insights)
        summary = (topline + "\n\n" + summary_core).strip()

        total_duration = (datetime.now() - start_time).total_seconds()

        logger.info(
            "Answer generation completed",
            stage="answer_generation_complete",
            paradigm="bernard",
            research_id=research_id,
            duration_ms=total_duration * 1000,
            metrics={
                "total_sections": len(sections),
                "total_citations": len(self.citations),
                "statistical_insights": len(statistical_insights),
                "meta_analysis_results": len(meta_analysis) if meta_analysis else 0,
                "confidence_score": 0.85,
                "synthesis_quality": 0.87,
                "section_metrics": section_metrics,
            },
        )
        logger.info(
            "Finished Bernard answer generation",
            research_id=context.metadata.get('research_id'),
            stage="bernard_answer_generation_end",
            generation_time=(datetime.now() - start_time).total_seconds(),
        )
        return GeneratedAnswer(
            research_id=context.metadata.get("research_id", "unknown"),
            query=context.query,
            paradigm=self.paradigm,
            summary=summary,
            sections=sections,
            action_items=await self._maybe_dynamic_actions(context, self._generate_research_action_items(statistical_insights)),
            citations=self.citations,
            confidence_score=self._calculate_analytical_confidence(statistical_insights),
            synthesis_quality=0.9,
            generation_time=(datetime.now() - start_time).total_seconds(),
            metadata={
                "statistical_insights": len(statistical_insights),
                "meta_analysis_performed": meta_analysis is not None,
                "peer_reviewed_sources": self._count_peer_reviewed(context.search_results),
                "topline_recommendation": topline,
                "claim_source_map": self._build_claim_source_map(sections, self.citations),
                "llm_backend": self._get_llm_backend_info(),
            }
        )

    async def _generate_analytical_section(
        self,
        context: SynthesisContext,
        section_def: Dict[str, Any],
        statistical_insights: List[StatisticalInsight],
        meta_analysis: Optional[Dict[str, Any]]
    ) -> AnswerSection:
        """Generate analytical section with statistical context"""
        # Get progress tracker if available
        research_id = context.metadata.get("research_id") if hasattr(context, "metadata") else None
        progress_tracker = self._get_progress_tracker()

        section_name = section_def.get('title', 'Section')

        # Filter relevant results
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="synthesis",
                message=f"{section_name}: Analyzing sources"
            )
        relevant_results = self._top_relevant_results(context, 5)

        # Create citations
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="synthesis",
                message=f"{section_name}: Processing empirical citations"
            )
        citation_ids = self._create_citations(relevant_results, "empirical")

        # Generate content
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="synthesis",
                message=f"{section_name}: Extracting statistical insights"
            )
        try:
            insights_summary = self._format_statistical_insights(statistical_insights[:5])
            iso_block = self.build_iso_block(context)
            _section_tokens, inline_ctx, use_windows = self.decide_context_mode(section_def['weight'])
            evidence_block = self._safe_evidence_block(context, inline_ctx=inline_ctx)
            ctx_windows_block = self._context_windows_block(context) if use_windows else "(context windows disabled)"
            summaries_block = self._source_summaries_block(context) if EVIDENCE_INCLUDE_SUMMARIES else "(summaries disabled)"
            coverage_tbl = self._coverage_table(context)
            source_cards = self._source_cards_block(context) if os.getenv("SOURCE_CARDS_ENABLE", "1") in {"1", "true", "yes"} else "(source cards disabled)"
            variant_extra = self._variant_addendum(context, "bernard")
            meta_line = f"Meta-analysis results: {meta_analysis}\n" if meta_analysis else ""
            prompt = self.build_prompt(
                paradigm="bernard",
                section_title=section_def['title'],
                section_focus=section_def['focus'],
                query=context.query,
                source_cards=source_cards,
                evidence_block=evidence_block,
                ctx_windows_block=ctx_windows_block,
                iso_block=iso_block,
                coverage_tbl=coverage_tbl,
                summaries_block=summaries_block,
                target_words=int(SYNTHESIS_BASE_WORDS * section_def['weight']),
                variant_extra=variant_extra,
                paradigm_directives=[
                    "Use precise scientific language",
                    "Differentiate correlation vs causation",
                    "Include effect sizes / p-values when present",
                    "Acknowledge limitations",
                ],
                extra_requirements=[meta_line.strip()] if meta_line else [],
            )
            content = await llm_client.generate_paradigm_content(prompt=prompt, paradigm="bernard")
        except Exception as e:
            logger.error(
                "Error generating section",
                exc_info=True,
                research_id=research_id,
                stage="bernard_section_generation_error",
                section=section_def['title'],
                error=str(e),
            )
            if os.getenv("LLM_STRICT", "0") == "1":
                raise
            logger.warning(f"LLM generation failed: {e}, using fallback")
            content = self._generate_analytical_fallback(section_def, relevant_results, statistical_insights)
        # Extract quantitative insights
        key_insights = self._extract_quantitative_insights(content, statistical_insights)

        return AnswerSection(
            title=section_def['title'],
            paradigm=self.paradigm,
            content=content,
            confidence=0.85,
            citations=citation_ids,
            word_count=len(content.split()),
            key_insights=key_insights,
            metadata={
                "section_weight": section_def['weight'],
                "statistical_evidence": len([i for i in statistical_insights if i.metric in content])
            }
        )

    def _generate_analytical_fallback(
        self,
        section_def: Dict[str, Any],
        results: List[Dict[str, Any]],
        insights: List[StatisticalInsight]
    ) -> str:
        content = self.render_generic_fallback(section_def, results, intro=f"This section provides {section_def.get('focus','analytical focus')}.")
        if insights:
            content += "\n## Key Statistical Findings\n\n"
            for insight in insights[:5]:
                content += f"- {insight.metric}: {insight.value}{insight.unit}\n"
        return content

    def _extract_statistical_insights(self, search_results: List[Dict[str, Any]]) -> List[StatisticalInsight]:
        """Extract statistical insights from search results"""
        insights: List[StatisticalInsight] = []

        for result in search_results:
            snippet = result.get("snippet", "")
            title = result.get("title", "")

            # Look for statistical patterns in snippets and titles
            text = f"{title} {snippet}".lower()

            # Extract percentages
            import re
            percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
            for pct in percentages:
                insights.append(StatisticalInsight(
                    metric="Percentage",
                    value=float(pct),
                    unit="%",
                    context=f"Found in {result.get('domain', 'unknown source')}"
                ))

            # Extract numbers with units (basic pattern)
            number_patterns = [
                (r'(\d+(?:\.\d+)?)\s*(?:million|billion|trillion)', "Count"),
                (r'\$?(\d+(?:\.\d+)?)\s*(?:dollars?|USD)', "Cost"),
                (r'(\d+(?:\.\d+)?)\s*(?:years?|months?|days?)', "Time Period"),
            ]

            for pattern, metric_name in number_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    try:
                        value = float(match[0] if isinstance(match, tuple) else match)
                        insights.append(StatisticalInsight(
                            metric=metric_name,
                            value=value,
                            unit="units",
                            context=f"Found in {result.get('domain', 'unknown source')}"
                        ))
                    except (ValueError, IndexError):
                        continue

        return insights[:10]  # Limit to top 10 insights

    def _perform_meta_analysis(self, search_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Perform basic meta-analysis on search results"""
        if not search_results:
            return None

        # Simple meta-analysis: aggregate credibility scores and find patterns
        credibility_scores = [r.get("credibility_score", 0.5) for r in search_results]
        avg_credibility = sum(credibility_scores) / len(credibility_scores)

        # Count sources by domain for diversity analysis
        domains = {}
        for result in search_results:
            domain = result.get("domain", "unknown")
            domains[domain] = domains.get(domain, 0) + 1

        # Find most common domains
        top_domains = sorted(domains.items(), key=lambda x: x[1], reverse=True)[:3]

        return {
            "average_credibility": avg_credibility,
            "total_sources": len(search_results),
            "unique_domains": len(domains),
            "top_domains": top_domains,
            "credibility_range": {
                "min": min(credibility_scores),
                "max": max(credibility_scores)
            }
        }

    def _format_statistical_insights(self, insights: List[StatisticalInsight]) -> str:
        """Format statistical insights for prompt inclusion"""
        if not insights:
            return "(no statistical insights)"

        lines = []
        for insight in insights:
            lines.append(f"- {insight.metric}: {insight.value}{insight.unit}")
            if insight.context:
                lines.append(f"  Context: {insight.context}")

        return "\n".join(lines)

    def _extract_quantitative_insights(self, content: str, statistical_insights: List[StatisticalInsight]) -> List[str]:
        """Extract quantitative insights from generated content"""
        insights = []

        # Look for numbers and patterns in the content
        import re

        # Extract percentages
        percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', content)
        for pct in percentages:
            insights.append(f"Found {pct}% statistic in analysis")

        # Extract dollar amounts
        dollar_amounts = re.findall(r'\$([0-9,]+(?:\.[0-9]+)?)', content)
        for amount in dollar_amounts:
            insights.append(f"Found ${amount} cost/economic impact")

        # Extract time periods
        time_periods = re.findall(r'(\d+)\s*(years?|months?|days?)', content)
        for time, unit in time_periods:
            insights.append(f"Found {time} {unit} time period")

        # Add insights from statistical insights if they appear in content
        for insight in statistical_insights:
            if insight.metric.lower() in content.lower():
                insights.append(f"Statistical finding: {insight.metric} = {insight.value}{insight.unit}")

        return insights[:5]  # Limit to top 5 insights

    def _generate_analytical_summary(self, sections: List[AnswerSection], statistical_insights: List[StatisticalInsight]) -> str:
        """Generate analytical summary from sections"""
        if not sections:
            return "No analytical sections generated."

        # Take key insights from first few sections
        key_insights = []
        for section in sections[:2]:  # First 2 sections
            key_insights.extend(section.key_insights[:2])  # First 2 insights per section

        # Add statistical insights
        if statistical_insights:
            key_insights.append(f"Key statistical findings: {len(statistical_insights)} insights identified")

        summary = " | ".join(key_insights[:3])  # Limit to 3 key points
        return summary if summary else "Analysis completed with multiple sections."

    def _calculate_analytical_confidence(self, statistical_insights: List[StatisticalInsight]) -> float:
        """Calculate confidence score based on statistical evidence"""
        base_confidence = 0.7  # Base confidence

        if not statistical_insights:
            return base_confidence

        # Increase confidence based on number and quality of statistical insights
        insight_bonus = min(0.2, len(statistical_insights) * 0.05)  # Up to 0.2 bonus
        diversity_bonus = min(0.1, len(set(i.metric for i in statistical_insights)) * 0.02)  # Up to 0.1 bonus

        return min(0.95, base_confidence + insight_bonus + diversity_bonus)

    def _count_peer_reviewed(self, search_results: List[Dict[str, Any]]) -> int:
        """Count peer-reviewed sources"""
        peer_reviewed_domains = {
            'nature.com', 'science.org', 'cell.com', 'thelancet.com',
            'nejm.org', 'jamanetwork.com', 'bmj.com', 'pnas.org'
        }

        count = 0
        for result in search_results:
            domain = result.get('domain', '').lower()
            if any(pr_domain in domain for pr_domain in peer_reviewed_domains):
                count += 1

        return count

    def _generate_research_action_items(self, statistical_insights: List[StatisticalInsight]) -> List[Dict[str, Any]]:
        """Generate research-focused action items based on statistical insights"""
        action_items = []

        if not statistical_insights:
            action_items.append({
                "action": "Conduct comprehensive statistical analysis",
                "priority": "high",
                "description": "Gather more quantitative data to support findings",
                "timeline": "2-4 weeks"
            })
        else:
            # Generate action items based on available insights
            action_items.append({
                "action": "Validate statistical findings with additional sources",
                "priority": "medium",
                "description": f"Cross-reference {len(statistical_insights)} statistical insights with peer-reviewed literature",
                "timeline": "1-2 weeks"
            })

        action_items.append({
            "action": "Perform meta-analysis on conflicting data",
            "priority": "medium",
            "description": "Resolve any contradictory findings through systematic review",
            "timeline": "2-3 weeks"
        })

        return action_items


class MaeveAnswerGenerator(BaseAnswerGenerator):
    """Strategic paradigm answer generator with business analysis"""

    def __init__(self):
        super().__init__("maeve")

    def get_section_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "title": "Strategic Overview",
                "focus": "Market landscape, competitive positioning, and opportunity assessment",
                "weight": 0.20,
            },
            {
                "title": "Tactical Approaches",
                "focus": "Specific strategies, implementation methods, and quick wins",
                "weight": 0.25,
            },
            {
                "title": "Resource Optimization",
                "focus": "Cost-benefit analysis, resource allocation, and efficiency gains",
                "weight": 0.20,
            },
            {
                "title": "Success Metrics",
                "focus": "KPIs, measurement frameworks, and performance tracking",
                "weight": 0.15,
            },
            {
                "title": "Implementation Roadmap",
                "focus": "Timeline, milestones, dependencies, and risk mitigation",
                "weight": 0.20,
            },
        ]

    async def generate_answer(self, context: SynthesisContext) -> GeneratedAnswer:
        """Generate strategic answer with business insights"""
        logger.info(
            "Starting Maeve answer generation",
            research_id=context.metadata.get('research_id'),
            stage="maeve_answer_generation_start",
        )
        start_time = datetime.now()
        self.citation_counter = 0
        self.citations = {}

        # Get progress tracker if available
        research_id = context.metadata.get("research_id") if hasattr(context, "metadata") else None
        progress_tracker = self._get_progress_tracker()

        # Extract strategic insights
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="synthesis",
                message="Maeve: Extracting strategic insights"
            )
        strategic_insights = self._extract_strategic_insights(context.search_results)

        # Generate SWOT analysis
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="synthesis",
                message="Maeve: Generating SWOT analysis"
            )
        swot_analysis = self._generate_swot_analysis(context.query, context.search_results)

        # Generate sections
        sections = []
        for section_def in self.get_section_structure():
            section = await self._generate_strategic_section(
                context, section_def, strategic_insights, swot_analysis
            )
            sections.append(section)

        # Generate strategic recommendations
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="synthesis",
                message="Maeve: Formulating strategic recommendations"
            )
        recommendations = self._generate_strategic_recommendations(
            context.query, strategic_insights, swot_analysis
        )

        topline = self._compose_topline_recommendation(context)
        summary_core = self._generate_strategic_summary(sections, strategic_insights)
        summary = (topline + "\n\n" + summary_core).strip()

        return GeneratedAnswer(
            research_id=context.metadata.get("research_id", "unknown"),
            query=context.query,
            paradigm=self.paradigm,
            summary=summary,
            sections=sections,
            action_items=await self._maybe_dynamic_actions(context, self._format_recommendations_as_actions(recommendations)),
            citations=self.citations,
            confidence_score=0.85,
            synthesis_quality=0.88,
            generation_time=(datetime.now() - start_time).total_seconds(),
            metadata={
                "strategic_insights": len(strategic_insights),
                "swot_completed": swot_analysis is not None,
                "recommendations": len(recommendations),
                "topline_recommendation": topline,
                "claim_source_map": self._build_claim_source_map(sections, self.citations),
            }
        )

    async def _generate_strategic_section(
        self,
        context: SynthesisContext,
        section_def: Dict[str, Any],
        strategic_insights: List[StrategicRecommendation],
        swot_analysis: Dict[str, List[str]]
    ) -> AnswerSection:
        """Generate strategic section"""
        # Get progress tracker if available
        research_id = context.metadata.get("research_id") if hasattr(context, "metadata") else None
        progress_tracker = self._get_progress_tracker()

        section_name = section_def.get('title', 'Section')

        # Filter relevant results
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="synthesis",
                message=f"{section_name}: Analyzing strategic landscape"
            )
        relevant_results = self._top_relevant_results(context, 5)

        # Create citations
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="synthesis",
                message=f"{section_name}: Building strategic citations"
            )
        citation_ids = self._create_citations(relevant_results, "strategic")

        # Generate content
        try:
            swot_summary = self._format_swot_for_prompt(swot_analysis)
            iso_block = self.build_iso_block(context)
            _section_tokens, inline_ctx, use_windows = self.decide_context_mode(section_def['weight'])
            evidence_block = self._safe_evidence_block(context, inline_ctx=inline_ctx)
            ctx_windows_block = self._context_windows_block(context) if use_windows else "(context windows disabled)"
            summaries_block = self._source_summaries_block(context) if EVIDENCE_INCLUDE_SUMMARIES else "(summaries disabled)"
            coverage_tbl = self._coverage_table(context)
            source_cards = self._source_cards_block(context) if os.getenv("SOURCE_CARDS_ENABLE", "1") in {"1", "true", "yes"} else "(source cards disabled)"
            variant_extra = self._variant_addendum(context, "maeve")
            prompt = self.build_prompt(
                paradigm="maeve",
                section_title=section_def['title'],
                section_focus=section_def['focus'],
                query=context.query,
                source_cards=source_cards,
                evidence_block=evidence_block,
                ctx_windows_block=ctx_windows_block,
                iso_block=iso_block,
                coverage_tbl=coverage_tbl,
                summaries_block=summaries_block,
                target_words=int(SYNTHESIS_BASE_WORDS * section_def['weight']),
                variant_extra=variant_extra,
                paradigm_directives=[
                    "Focus on actionable strategies",
                    "Include ROI & resource implications",
                    "Emphasize competitive advantages",
                ],
                extra_requirements=["Incorporate SWOT signals where relevant", f"SWOT Summary inline: {swot_summary[:120]}"]
            )
            content = await llm_client.generate_paradigm_content(prompt=prompt, paradigm="maeve")
        except Exception as e:
            if os.getenv("LLM_STRICT", "0") == "1":
                raise
            logger.warning(f"LLM generation failed: {e}, using fallback")
            content = self._generate_strategic_fallback(section_def, relevant_results, swot_analysis)
        # Extract strategic insights
        key_insights = self._extract_strategic_insights_from_content(content)

        return AnswerSection(
            title=section_def['title'],
            paradigm=self.paradigm,
            content=content,
            confidence=0.82,
            citations=citation_ids,
            word_count=len(content.split()),
            key_insights=key_insights,
            metadata={
                "section_weight": section_def['weight'],
                "strategic_focus": section_def['focus']
            }
        )

    def _generate_strategic_fallback(
        self,
        section_def: Dict[str, Any],
        results: List[Dict[str, Any]],
        swot: Dict[str, List[str]]
    ) -> str:
        content = self.render_generic_fallback(section_def, results, intro=f"This section addresses: {section_def.get('focus','strategic focus')}")
        content += "\n## Strategic Analysis\n\n"
        for category, items in swot.items():
            if items:
                content += f"**{category.title()}:**\n"
                for item in items[:2]:
                    content += f"- {item}\n"
                content += "\n"
        return content

    def _extract_strategic_insights(self, search_results: List[Dict[str, Any]]) -> List[StrategicRecommendation]:
        """Derive structured strategic levers from search evidence without extra LLM calls."""
        insights: List[StrategicRecommendation] = []
        seen_titles: set[str] = set()
        theme_catalog = [
            {
                "title": "Accelerate market expansion",
                "keywords": ["market", "expansion", "go-to-market", "launch", "cagr", "demand"],
                "impact": "high",
                "effort": "medium",
                "timeline": "3-6 months",
                "dependencies": ["Market sizing", "Localized value proposition"],
                "metrics": ["Net-new pipeline", "Market share delta"],
                "risks": ["Insufficient localization or regulatory friction"],
                "roi": 0.22,
            },
            {
                "title": "Differentiate the product offering",
                "keywords": ["differentiation", "unique", "innovation", "feature", "roadmap"],
                "impact": "high",
                "effort": "medium",
                "timeline": "2-4 months",
                "dependencies": ["Voice-of-customer synthesis", "Competitive teardown"],
                "metrics": ["Win-rate lift", "Feature adoption"],
                "risks": ["Feature creep diluting core value"],
                "roi": 0.18,
            },
            {
                "title": "Optimize cost structure",
                "keywords": ["efficiency", "margin", "cost", "productivity", "automation"],
                "impact": "medium",
                "effort": "medium",
                "timeline": "6-9 months",
                "dependencies": ["Process mapping", "Automation backlog"],
                "metrics": ["Unit economics", "Operating margin"],
                "risks": ["Disruption to core workflows"],
                "roi": 0.16,
            },
            {
                "title": "Build strategic partnerships",
                "keywords": ["partnership", "alliance", "ecosystem", "collaborat"],
                "impact": "medium",
                "effort": "low",
                "timeline": "1-2 quarters",
                "dependencies": ["Partner due diligence", "Joint value proposition"],
                "metrics": ["Co-marketing leads", "Partner-attributed revenue"],
                "risks": ["Misaligned incentives"],
                "roi": 0.14,
            },
            {
                "title": "Strengthen customer lifecycle",
                "keywords": ["retention", "customer experience", "lifecycle", "churn", "loyalty"],
                "impact": "medium",
                "effort": "medium",
                "timeline": "2-3 quarters",
                "dependencies": ["Journey mapping", "Service playbooks"],
                "metrics": ["Net revenue retention", "CSAT"],
                "risks": ["Under-resourced success teams"],
                "roi": 0.17,
            },
        ]

        for result in search_results or []:
            text_parts = [result.get("title", ""), result.get("snippet", ""), result.get("summary", "")]
            raw_text = " ".join(part for part in text_parts if part)
            lower_text = raw_text.lower()
            if not lower_text.strip():
                continue
            for theme in theme_catalog:
                if theme["title"] in seen_titles:
                    continue
                if any(keyword in lower_text for keyword in theme["keywords"]):
                    description = raw_text.strip()[:320] or theme["title"]
                    insights.append(
                        StrategicRecommendation(
                            title=theme["title"],
                            description=description,
                            impact=theme["impact"],
                            effort=theme["effort"],
                            timeline=theme["timeline"],
                            dependencies=list(theme["dependencies"]),
                            success_metrics=list(theme["metrics"]),
                            risks=list(theme["risks"]),
                            roi_potential=theme["roi"],
                        )
                    )
                    seen_titles.add(theme["title"])
                    break
            if len(insights) >= 5:
                break

        if not insights:
            insights.append(
                StrategicRecommendation(
                    title="Define near-term strategic roadmap",
                    description="Evidence is thematic rather than prescriptive; establish a focused roadmap that sequences quick wins with longer-term bets.",
                    impact="medium",
                    effort="medium",
                    timeline="6-8 weeks",
                    dependencies=["Executive alignment", "Resource planning"],
                    success_metrics=["Roadmap milestones achieved", "Stakeholder buy-in"],
                    risks=["Diffuse ownership"],
                    roi_potential=0.1,
                )
            )

        return insights

    def _generate_swot_analysis(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        """Create a lightweight SWOT view from titles/snippets."""
        categories = {
            "strengths": ["leading", "advantage", "strength", "success", "momentum", "leadership", "innovative"],
            "weaknesses": ["challenge", "gap", "limitation", "weak", "struggle", "pain point", "shortcoming"],
            "opportunities": ["opportunity", "growth", "emerging", "demand", "expansion", "trend", "white space"],
            "threats": ["risk", "threat", "competition", "downturn", "regulation", "pressure", "headwind"],
        }
        swot: Dict[str, List[str]] = {k: [] for k in categories}

        for result in search_results or []:
            text = " ".join(
                part for part in [result.get("title", ""), result.get("snippet", ""), result.get("summary", "")] if part
            )
            if not text:
                continue
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for sentence in sentences:
                lowered = sentence.lower()
                for cat, keywords in categories.items():
                    if any(keyword in lowered for keyword in keywords):
                        cleaned = sentence.strip()
                        if cleaned and cleaned not in swot[cat]:
                            swot[cat].append(cleaned)
                        break

        # Provide purposeful fallbacks if a bucket is empty
        defaults = {
            "strengths": [f"Growing interest in {query} suggests momentum for strategic investment."],
            "weaknesses": [f"Limited differentiated positioning around {query} could slow velocity."],
            "opportunities": [f"Emerging demand signals room to create quick wins tied to {query}."],
            "threats": [f"Active competitors and fast-moving technology shifts call for risk mitigation."],
        }
        for cat, fallback in defaults.items():
            if not swot[cat]:
                swot[cat] = fallback
            else:
                swot[cat] = swot[cat][:3]

        return swot

    def _format_swot_for_prompt(self, swot: Dict[str, List[str]]) -> str:
        """Compress SWOT items into a prompt-friendly summary."""
        parts: List[str] = []
        for cat in ("strengths", "weaknesses", "opportunities", "threats"):
            items = swot.get(cat) or []
            if items:
                parts.append(f"{cat.title()}: { '; '.join(item[:120] for item in items[:2])}")
        return " | ".join(parts) if parts else "No explicit SWOT signals detected."

    def _extract_strategic_insights_from_content(self, content: str) -> List[str]:
        """Pull headline-worthy strategic points from generated content."""
        if not content:
            return []
        insights: List[str] = []
        sentences = re.split(r"(?<=[.!?])\s+", content)
        keywords = ["strategy", "growth", "increase", "roi", "revenue", "customer", "risk", "opportunity", "metric"]
        for sentence in sentences:
            lowered = sentence.lower().strip()
            if not lowered:
                continue
            if any(keyword in lowered for keyword in keywords):
                insights.append(sentence.strip())
            if len(insights) >= 5:
                break

        if len(insights) < 3:
            bullets = [line.strip("-• \t") for line in content.splitlines() if line.strip().startswith(('-', '•'))]
            for bullet in bullets:
                if bullet and bullet not in insights:
                    insights.append(bullet[:240])
                if len(insights) >= 5:
                    break

        return insights[:5]

    def _generate_strategic_recommendations(
        self,
        query: str,
        strategic_insights: List[StrategicRecommendation],
        swot: Dict[str, List[str]],
    ) -> List[StrategicRecommendation]:
        """Blend heuristic insights with SWOT cues into prioritized recommendations."""
        recommendations: List[StrategicRecommendation] = []
        seen_titles: set[str] = set()

        for rec in strategic_insights:
            if rec.title in seen_titles:
                continue
            recommendations.append(rec)
            seen_titles.add(rec.title)
            if len(recommendations) >= 3:
                break

        if swot.get("opportunities") and "Pursue opportunity backlog" not in seen_titles:
            recommendations.append(
                StrategicRecommendation(
                    title="Pursue opportunity backlog",
                    description=f"Convert top opportunities around {query} into sequenced initiatives with ROI guardrails.",
                    impact="high",
                    effort="medium",
                    timeline="1-2 quarters",
                    dependencies=["Growth experimentation team", "Budget envelope"],
                    success_metrics=["Opportunity-to-project conversion", "Incremental revenue"],
                    risks=["Chasing too many bets simultaneously"],
                    roi_potential=0.2,
                )
            )
            seen_titles.add("Pursue opportunity backlog")

        if swot.get("weaknesses") and "Remediate capability gaps" not in seen_titles:
            recommendations.append(
                StrategicRecommendation(
                    title="Remediate capability gaps",
                    description="Address the most material weaknesses with capability investments and enablement plans.",
                    impact="medium",
                    effort="medium",
                    timeline="1 quarter",
                    dependencies=["Capability assessment", "Change management plan"],
                    success_metrics=["Time-to-value for new capabilities", "Adoption score"],
                    risks=["Change fatigue or talent constraints"],
                    roi_potential=0.12,
                )
            )
            seen_titles.add("Remediate capability gaps")

        if swot.get("threats") and "Mitigate strategic risks" not in seen_titles:
            recommendations.append(
                StrategicRecommendation(
                    title="Mitigate strategic risks",
                    description="Stand up a risk playbook to track external threats and trigger predefined responses.",
                    impact="medium",
                    effort="low",
                    timeline="6-8 weeks",
                    dependencies=["Risk owner", "Monitoring cadence"],
                    success_metrics=["Risk incident response time", "Mitigation readiness"],
                    risks=["Underestimating regulatory or competitive shifts"],
                    roi_potential=0.1,
                )
            )
            seen_titles.add("Mitigate strategic risks")

        if not recommendations:
            recommendations.append(
                StrategicRecommendation(
                    title="Prioritize quick wins",
                    description=f"Select two quick-win initiatives informed by current research on {query} to demonstrate tangible progress.",
                    impact="medium",
                    effort="low",
                    timeline="6-8 weeks",
                    dependencies=["Dedicated owner", "Success criteria"],
                    success_metrics=["Quick-win completion", "Stakeholder confidence"],
                    risks=["Under-scoped validation"],
                    roi_potential=0.08,
                )
            )

        return recommendations[:5]

    def _format_recommendations_as_actions(
        self,
        recommendations: List[StrategicRecommendation]
    ) -> List[Dict[str, Any]]:
        """Convert structured recommendations into UI-friendly action items."""
        actions: List[Dict[str, Any]] = []
        priority_map = {"high": "high", "medium": "medium", "low": "low"}
        for rec in recommendations:
            actions.append(
                {
                    "action": rec.title,
                    "priority": priority_map.get(rec.impact.lower(), "medium") if isinstance(rec.impact, str) else "medium",
                    "description": rec.description,
                    "timeline": rec.timeline,
                    "dependencies": list(rec.dependencies or []),
                    "metrics": list(rec.success_metrics or []),
                    "risks": list(rec.risks or []),
                }
            )
        return actions

    def _generate_strategic_summary(
        self,
        sections: List[AnswerSection],
        strategic_insights: List[StrategicRecommendation],
    ) -> str:
        """Assemble a concise executive summary for Maeve outputs."""
        key_points: List[str] = []
        for section in sections[:2]:
            key_points.extend(section.key_insights[:2])

        if strategic_insights:
            key_points.append(f"Identified {len(strategic_insights)} strategic levers across market, offering, and execution.")

        deduped = []
        seen: set[str] = set()
        for point in key_points:
            norm = point.strip()
            if not norm:
                continue
            if norm.lower() in seen:
                continue
            seen.add(norm.lower())
            deduped.append(norm)
            if len(deduped) >= 4:
                break

        if not deduped:
            return "Strategic synthesis completed; see sections for detailed direction."  # fallback copy

        return " | ".join(deduped)


class TeddyAnswerGenerator(BaseAnswerGenerator):
    """Supportive paradigm answer generator"""

    def __init__(self):
        super().__init__("teddy")

    def get_section_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "title": "Understanding the Need",
                "focus": "Empathetic assessment of who needs help and why",
                "weight": 0.25,
            },
            {
                "title": "Available Support Resources",
                "focus": "Comprehensive listing of help and resources",
                "weight": 0.3,
            },
            {
                "title": "Success Stories",
                "focus": "Inspiring examples of care and recovery",
                "weight": 0.25,
            },
            {
                "title": "How to Help",
                "focus": "Practical steps for providing support",
                "weight": 0.2,
            },
        ]

    async def generate_answer(self, context: SynthesisContext) -> GeneratedAnswer:
        """Generate supportive answer"""
        start_time = datetime.now()
        self.citation_counter = 0
        self.citations = {}

        sections = []
        for section_def in self.get_section_structure():
            section = await self._generate_supportive_section(context, section_def)
            sections.append(section)

        topline = self._compose_topline_recommendation(context)
        summary_core = self._generate_supportive_summary(sections)
        summary = (topline + "\n\n" + summary_core).strip()

        return GeneratedAnswer(
            research_id=context.metadata.get("research_id", "unknown"),
            query=context.query,
            paradigm=self.paradigm,
            summary=summary,
            sections=sections,
            action_items=await self._maybe_dynamic_actions(context, self._generate_supportive_actions(context)),
            citations=self.citations,
            confidence_score=0.82,
            synthesis_quality=0.86,
            generation_time=(datetime.now() - start_time).total_seconds(),
            metadata={
                "tone": "supportive",
                "focus": "community care",
                "topline_recommendation": topline,
                "claim_source_map": self._build_claim_source_map(sections, self.citations),
            }
        )

    async def _generate_supportive_section(
        self,
        context: SynthesisContext,
        section_def: Dict[str, Any]
    ) -> AnswerSection:
        """Generate supportive section"""
        # Get progress tracker if available
        research_id = context.metadata.get("research_id") if hasattr(context, "metadata") else None
        progress_tracker = self._get_progress_tracker()

        section_name = section_def.get('title', 'Section')

        # Filter relevant results
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="synthesis",
                message=f"{section_name}: Finding support resources"
            )
        relevant_results = [r for r in context.search_results[:5]]

        # Create citations
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="synthesis",
                message=f"{section_name}: Gathering support citations"
            )
        citation_ids = self._create_citations(relevant_results, "support")

        # Generate content
        try:
            # Isolation-only support
            iso_lines = []
            try:
                if isinstance(context.context_engineering, dict):
                    for m in (context.context_engineering.get("isolated_findings", {}).get("matches", []) or [])[:5]:
                        dom = m.get("domain", "")
                        for frag in (m.get("fragments", []) or [])[:1]:
                            iso_lines.append(f"- [{dom}] {sanitize_snippet(frag or '')}")
            except Exception:
                pass
            iso_block = "\n".join(iso_lines) if iso_lines else "(no isolated findings)"
            variant_extra = self._variant_addendum(context, "teddy")
            prompt = f"""{guardrail_instruction}
            Write the "{section_def['title']}" section focusing on: {section_def['focus']}
            Query: {context.query}

            Requirements:
            - Use warm, supportive language that builds hope and connection
            - Focus on human dignity and the power of community care
            - Emphasize resources, solutions, and paths forward
            - Include specific resources and support options
            - STRICT: Ground all examples in the Isolated Findings below
            Isolated Findings:
            {iso_block}

            {variant_extra}
            Length: {int(SYNTHESIS_BASE_WORDS * section_def['weight'])} words
            """

            content = await llm_client.generate_paradigm_content(
                prompt=prompt,
                paradigm="teddy",

            )
        except Exception as e:
            if os.getenv("LLM_STRICT", "0") == "1":
                raise
            logger.warning(f"LLM generation failed: {e}, using fallback")
            content = self._generate_supportive_fallback(section_def, relevant_results)

        return AnswerSection(
            title=section_def['title'],
            paradigm=self.paradigm,
            content=content,
            confidence=0.8,
            citations=citation_ids,
            word_count=len(content.split()),
            key_insights=self._extract_supportive_insights(content),
            metadata={"section_weight": section_def['weight']}
        )


    def _generate_supportive_fallback(
        self,
        section_def: Dict[str, Any],
        results: List[Dict[str, Any]]
    ) -> str:
        content = self.render_generic_fallback(section_def, results, intro=f"This section provides: {section_def.get('focus','supportive focus')}")
        content += "\nRemember: You are not alone. Help is available, and together we can make a difference.\n"
        return content


# ============================================================================
# ORCHESTRATOR

class AnswerGenerationOrchestrator:
    """Main orchestrator for answer generation"""

    def __init__(self):
        logger.info("Answer Generation Orchestrator initialized")

    def _make_generator(self, paradigm: str) -> BaseAnswerGenerator:
        if paradigm == "dolores":
            return DoloresAnswerGenerator()
        if paradigm == "bernard":
            return BernardAnswerGenerator()
        if paradigm == "maeve":
            return MaeveAnswerGenerator()
        if paradigm == "teddy":
            return TeddyAnswerGenerator()
        return BernardAnswerGenerator()

    async def generate_answer(
        self,
        paradigm: str,
        query: str,
        search_results: List[Dict[str, Any]],
        context_engineering: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> GeneratedAnswer:
        """Generate answer using specified paradigm"""

        # Validate paradigm
        if paradigm not in {"dolores", "bernard", "maeve", "teddy"}:
            logger.error(f"Unknown paradigm: {paradigm}")
            paradigm = "bernard"  # Default to analytical

        # Create synthesis context
        from core.config import SYNTHESIS_MAX_LENGTH_DEFAULT
        context = SynthesisContext(
            query=query,
            paradigm=paradigm,
            search_results=search_results,
            context_engineering=context_engineering or {},
            max_length=(options or {}).get("max_length", SYNTHESIS_MAX_LENGTH_DEFAULT),
            include_citations=(options or {}).get("include_citations", True),
            tone=(options or {}).get("tone", "professional"),
            metadata=options or {},
            evidence_quotes=(options or {}).get("evidence_quotes", []) or [],
            evidence_bundle=(options or {}).get("evidence_bundle"),
        )

        # Instantiate a fresh generator per request to avoid shared state issues
        from time import perf_counter
        start = perf_counter()
        generator = self._make_generator(paradigm)

        # Emit synthesis context snapshot for observability
        try:
            unique_sources = len({(r.get("domain") or extract_domain(r.get("url", "") or "")) for r in (search_results or []) if (r.get("url") or r.get("domain"))})
            # Attempt to resolve active prompt variant/strategy from generator
            try:
                strategy = generator._resolve_prompt_variant(context)  # type: ignore[attr-defined]
            except Exception:
                strategy = "v1"
            logger.info(
                "Synthesis context prepared",
                stage="synthesis_context",
                research_id=(options or {}).get("research_id"),
                paradigm=paradigm,
                evidence_quotes=len(context.evidence_quotes or []),
                unique_sources=unique_sources,
                token_budget=context.max_length,
                synthesis_strategy=strategy,
            )
        except Exception:
            pass

        answer = await generator.generate_answer(context)
        duration_ms = (perf_counter() - start) * 1000.0
        try:
            from .metrics import metrics
            metrics.record_stage(
                stage="answer_synthesis",
                duration_ms=duration_ms,
                paradigm=paradigm,
                success=True,
                fallback=False,
                model="llm"
            )
        except Exception:
            pass
        # Emit quality assessment summary for UI/debugging
        try:
            total_words = sum(int(getattr(sec, "word_count", 0) or 0) for sec in (answer.sections or []))
            unique_cites = len(set((answer.citations or {}).keys())) if isinstance(answer.citations, dict) else 0
            logger.info(
                "Answer quality metrics",
                stage="quality_assessment",
                research_id=(options or {}).get("research_id"),
                total_word_count=total_words,
                sections_generated=len(answer.sections or []),
                unique_citations=unique_cites,
                paradigm_alignment_score=None,
                coherence_score=None,
                factual_density=None,
            )
        except Exception:
            pass

        return answer

    async def generate_multi_paradigm_answer(
        self,
        primary_paradigm: str,
        secondary_paradigm: str,
        query: str,
        search_results: List[Dict[str, Any]],
        context_engineering: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate answer using multiple paradigms"""

        # Generate primary answer
        primary_answer = await self.generate_answer(
            primary_paradigm, query, search_results, context_engineering, options
        )

        # Generate secondary answer
        secondary_answer = await self.generate_answer(
            secondary_paradigm, query, search_results, context_engineering, options
        )

        # Combine insights
        combined_synthesis_quality = (
            primary_answer.synthesis_quality * 0.7 +
            secondary_answer.synthesis_quality * 0.3
        )

        return {
            "primary_paradigm": {
                "paradigm": primary_paradigm,
                "answer": primary_answer
            },
            "secondary_paradigm": {
                "paradigm": secondary_paradigm,
                "answer": secondary_answer
            },
            "synthesis_quality": combined_synthesis_quality,
            "generation_time": primary_answer.generation_time + secondary_answer.generation_time
        }


# ============================================================================
# LIGHTWEIGHT MARKDOWN RENDERER FOR RESEARCH REPORT
# SYNTAX FOR USE WITH SHORT CLUI BLOCK: no need, paragraphs and sentences are placed over a single line DO NOT use extra spaces to break, use: 1,2,1,3,4,1,2,3,1,2,3
# ============================================================================

# Lightweight renderer for Research Report markdown
def _fmt_sources_for_claim(claim_row: Dict[str, Any]) -> str:
    cites = claim_row.get("citations") or []
    if not cites:
        return "(no citations)"
    domains = [c.get("domain") or c.get("url") or "" for c in cites]
    seen = set()
    uniq: List[str] = []
    for d in domains:
        if d and d not in seen:
            seen.add(d)
            uniq.append(d)
    return ", ".join(uniq) if uniq else "(no citations)"

def _auto_bibliography(citations: Dict[str, Citation]) -> List[str]:
    from datetime import datetime as _dt
    def _sort_key(kv: Tuple[str, Citation]) -> int:
        cid = kv[0]
        try:
            return int(str(cid).split("_")[-1])
        except Exception:
            return 999999
    lines: List[str] = []
    for cid, c in sorted(citations.items(), key=_sort_key):
        acc = c.timestamp.isoformat() if isinstance(c.timestamp, _dt) else ""
        title = c.source_title or c.domain or c.source_url
        entry = f"[{cid}] {title}. {c.domain}. {c.source_url} {('Accessed ' + acc) if acc else ''}"
        lines.append(entry)
    return lines

def render_research_report_md(answer: GeneratedAnswer, intake: Dict[str, str]) -> str:
    meta = answer.metadata or {}
    claim_map = meta.get("claim_source_map") or []
    topline = meta.get("topline_recommendation") or (answer.summary.split("\n", 1)[0] if answer.summary else "")
    objective = intake.get("objective", "")
    scope = intake.get("scope", "")
    constraints = intake.get("constraints", "")
    key_findings: List[str] = []
    for sec in answer.sections:
        for ki in (sec.key_insights or []):
            key_findings.append(f"- {ki}")
            if len(key_findings) >= 5:
                break
        if len(key_findings) >= 5:
            break
    key_findings_md = "\n".join(key_findings) if key_findings else "- See sections below."

    claim_lines: List[str] = []
    for row in claim_map:
        claim = row.get("claim", "")
        sources_str = _fmt_sources_for_claim(row)
        claim_lines.append(f"- Claim: {claim}\n  - Sources: {sources_str}")
    claim_map_md = "\n".join(claim_lines) if claim_lines else "- (no mapped claims)"

    refs = _auto_bibliography(answer.citations or {})
    refs_md = "\n".join(f"- {r}" for r in refs) if refs else "- (none)"

    parts: List[str] = []
    parts.append("# Research Report")
    parts.append("")
    parts.append("## Executive Summary")
    parts.append(f"- Objective: {objective or '<fill from intake.objective>'}")
    parts.append(f"- Top‑line Recommendation: {topline}")
    parts.append(f"- Key Findings:\n{key_findings_md}")
    parts.append("- Confidence/Uncertainty: pending synthesis notes")
    parts.append(f"- Recommendations: {', '.join([a.get('action','') for a in (answer.action_items or [])][:3]) or '(see Recommendations section)'}")
    parts.append("")
    parts.append("## Methods")
    parts.append(f"- Scope and Constraints: {scope or '<from intake.scope>'} {constraints or ''}".strip())
    parts.append("- Data Sources and Discovery:\n  - Search engines/APIs used\n  - Query strategies and time windows\n  - Inclusion/exclusion criteria")
    parts.append("- Retrieval and Processing:\n  - Parsers, chunking strategy, embeddings model\n  - Deduplication approach (MD5/SimHash)\n  - Metadata collected")
    parts.append("- Credibility Assessment:\n  - Features: domain reputation, recency, citations, cross-source agreement\n  - Scoring function and thresholds")
    parts.append("- Orchestration:\n  - Budget caps: tokens/cost/time\n  - Tools and rate limits\n  - Failure handling/backoff")
    parts.append("- Evaluation:\n  - Rubric criteria and weighting\n  - Evaluator isolation and challenge sets")
    parts.append("")
    parts.append("## Results")
    parts.append("- Claim→Source Map (Top Claims with Citations):")
    parts.append(claim_map_md)
    parts.append("")
    parts.append("- Thematic Findings with Per-sentence Citations:")
    for i, sec in enumerate(answer.sections, 1):
        snippet = (sec.content or "").split("\n", 1)[0]
        parts.append(f"  {i}. {snippet[:200]}...")
    parts.append("- Evidence Summaries:")
    for cid, c in (answer.citations or {}).items():
        cred = getattr(c, "credibility_score", None)
        parts.append(f"  - {c.domain or c.source_url} (credibility: {cred if cred is not None else 'n/a'}): {(c.snippet or '')[:140]} [CIT:{cid}]")
    parts.append("- Conflict Analysis:\n  - (pending)")
    parts.append("- Figures and Tables\n  - (none)")
    parts.append("")
    parts.append("## Discussion\n- Interpretation of findings\n- Limitations and potential biases\n- Uncertainty quantification per claim\n- Implications and alternatives")
    parts.append("")
    parts.append("## Recommendations")
    if answer.action_items:
        for a in answer.action_items:
            parts.append(f"- {a.get('action','')} (priority: {a.get('priority','')})")
    else:
        parts.append("- (none)")
    parts.append("")
    parts.append("## References")
    parts.append(refs_md)
    parts.append("")
    parts.append("## Appendix\n- Intake Scope Card snapshot\n- Tool usage and costs\n- Reproduction guide (env, seeds, versions)\n- Full lineage and artifact IDs")
    return "\n".join(parts)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

# Create singleton orchestrator instance
answer_orchestrator = AnswerGenerationOrchestrator()

# Legacy compatibility function
async def initialize_answer_generation() -> bool:
    """Initialize answer generation system (no-op for compatibility)"""
    return True

# Export all necessary classes and functions
__all__ = [
    # V1 Compatibility
    'SynthesisContext',
    'Citation',
    'AnswerSection',
    'GeneratedAnswer',
    'BaseAnswerGenerator',

    # Generators
    'DoloresAnswerGenerator',
    'BernardAnswerGenerator',
    'MaeveAnswerGenerator',
    'TeddyAnswerGenerator',

    # Enhanced Generators (aliases for compatibility)
    'EnhancedBernardAnswerGenerator',
    'EnhancedMaeveAnswerGenerator',

    # Orchestrator
    'AnswerGenerationOrchestrator',
    'answer_orchestrator',

    # Functions
    'initialize_answer_generation',
    'render_research_report_md',

    # Core types
    'StatisticalInsight',
    'StrategicRecommendation',
]

# Create aliases for enhanced generators (for backward compatibility)
EnhancedBernardAnswerGenerator = BernardAnswerGenerator
EnhancedMaeveAnswerGenerator = MaeveAnswerGenerator
