"""
Answer Generation System - Consolidated and Deduped
Combines all answer generation functionality into a single, clean module
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import Counter, defaultdict

import os
from models.context_models import (
    ClassificationResultSchema,
    ContextEngineeredQuerySchema,
    UserContextSchema,
    HostParadigm,
    SearchResultSchema,
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

logger = logging.getLogger(__name__)


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
        results = list(context.search_results or [])
        # Evidence density from evidence bundle
        ev_counts_by_url: Dict[str, int] = {}
        ev_counts_by_domain: Dict[str, int] = {}
        try:
            eb = getattr(context, "evidence_bundle", None)
            if eb is not None and getattr(eb, "quotes", None):
                quotes = list(getattr(eb, "quotes", []) or [])
                def _qval(q, name, default=""):
                    try:
                        if isinstance(q, dict):
                            return q.get(name, default)
                        if hasattr(q, name):
                            return getattr(q, name)
                        if hasattr(q, "model_dump"):
                            return q.model_dump().get(name, default)
                        if hasattr(q, "dict"):
                            return q.dict().get(name, default)
                    except Exception:
                        return default
                    return default
                for q in quotes:
                    u = (_qval(q, "url", "") or "").strip()
                    d = (_qval(q, "domain", "") or "").lower()
                    if u:
                        ev_counts_by_url[u] = ev_counts_by_url.get(u, 0) + 1
                    if d:
                        ev_counts_by_domain[d] = ev_counts_by_domain.get(d, 0) + 1
        except Exception:
            pass

        from datetime import datetime
        def _recency_score(v) -> float:
            try:
                dt = v
                if isinstance(v, str):
                    dt = datetime.fromisoformat(v.replace('Z', '+00:00'))
                if not dt:
                    return 0.0
                # 0..1 over ~5 years
                age_days = max(0.0, (datetime.utcnow() - dt).days)
                return max(0.0, 1.0 - min(1.0, age_days / (365.0 * 5)))
            except Exception:
                return 0.0

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for r in results:
            cred = float(r.get("credibility_score", 0.0) or 0.0)
            u = (r.get("url") or "").strip()
            d = (r.get("domain") or r.get("source") or "").lower()
            evid = ev_counts_by_url.get(u, 0) or ev_counts_by_domain.get(d, 0)
            evid_norm = min(1.0, evid / 3.0)  # 0..1 for 0..3+ quotes
            rec = _recency_score(r.get("published_date"))
            score = 0.6 * cred + 0.25 * evid_norm + 0.15 * rec
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        picked: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for _score, r in scored:
            dom = (r.get("domain") or r.get("source") or "").lower()
            if dom in seen and len(picked) < k // 2:
                continue
            picked.append(r)
            seen.add(dom)
            if len(picked) >= k:
                break
        return picked[:k]

    def _format_evidence_block(self, context: SynthesisContext, max_quotes: int = EVIDENCE_MAX_QUOTES_DEFAULT) -> str:
        quotes = []
        try:
            eb = getattr(context, "evidence_bundle", None)
            if eb is not None and getattr(eb, "quotes", None):
                quotes = list(getattr(eb, "quotes", []) or [])[:max_quotes]
        except Exception:
            quotes = []
        def _qval(q, name, default=""):
            try:
                if isinstance(q, dict):
                    return q.get(name, default)
                # pydantic v2
                if hasattr(q, name):
                    return getattr(q, name)
                if hasattr(q, "model_dump"):
                    return q.model_dump().get(name, default)
                if hasattr(q, "dict"):
                    return q.dict().get(name, default)
            except Exception:
                return default
            return default
        lines: List[str] = []
        for q in quotes:
            qid = _qval(q, "id", "")
            dom = (_qval(q, "domain", "") or "").lower()
            qt = _qval(q, "quote", "")
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
        # Gather quotes (ensure local is always defined to avoid UnboundLocalError)
        quotes: List[Any] = []
        try:
            eb = getattr(context, "evidence_bundle", None)
            if eb is not None and getattr(eb, "quotes", None):
                quotes = list(getattr(eb, "quotes", []) or [])
        except Exception:
            quotes = []
        if not quotes:
            return "(no evidence quotes)"
        # Build items compatible with select_items_within_budget (uses estimate_tokens_for_result)
        items_for_budget: List[Dict[str, Any]] = []
        def _qval(q, name, default=""):
            try:
                if isinstance(q, dict):
                    return q.get(name, default)
                if hasattr(q, name):
                    return getattr(q, name)
                if hasattr(q, "model_dump"):
                    return q.model_dump().get(name, default)
                if hasattr(q, "dict"):
                    return q.dict().get(name, default)
            except Exception:
                return default
            return default
        include_ctx = inline_ctx and (os.getenv("EVIDENCE_INCLUDE_CONTEXT", "1") in {"1", "true", "yes"})
        ctx_max = int(os.getenv("EVIDENCE_CONTEXT_MAX_CHARS", "220"))
        for q in quotes:
            items_for_budget.append({
                "id": _qval(q, "id", ""),
                "domain": _qval(q, "domain", ""),
                "title": "",           # no title for quotes
                "snippet": "",         # no snippet for quotes
                "content": (_qval(q, "quote", "") or "") + (f" | ctx: {sanitize_snippet(_qval(q, 'context_window', '') or '', max_len=ctx_max)}" if include_ctx and _qval(q, 'context_window', '') else ""),
            })
        selected, _used, _dropped = select_items_within_budget(items_for_budget, max_tokens=budget_tokens)
        selected = selected[:max_quotes]
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
        def _qval(q, name, default=""):
            try:
                if isinstance(q, dict):
                    return q.get(name, default)
                if hasattr(q, name):
                    return getattr(q, name)
                if hasattr(q, "model_dump"):
                    return q.model_dump().get(name, default)
                if hasattr(q, "dict"):
                    return q.dict().get(name, default)
            except Exception:
                return default
            return default
        for q in quotes:
            qid = _qval(q, "id", "")
            dom = (_qval(q, "domain", "") or "").lower()
            ctx = _qval(q, "context_window", "") or ""
            if not ctx:
                continue
            ctx = sanitize_snippet(ctx, max_len=ctx_max)
            lines.append(f"- [{qid}][{dom}] {ctx}")
        return "\n".join(lines) if lines else "(no context windows)"

    def _source_cards_block(self, context: SynthesisContext, k: int = 5) -> str:
        """Compact source cards: title | domain | date | author (context only)."""
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
        for r in results:
            md = r.get("metadata", {}) or {}
            ext = (md.get("extracted_meta") or {}) if isinstance(md, dict) else {}
            title = (r.get("title") or ext.get("title") or "").strip()
            domain = (r.get("domain") or r.get("source") or "").strip()
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
            lines.append(f"- {' | '.join(parts)}")
        return "\n".join(lines) if lines else "(no source cards)"

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
        except Exception:
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
        def _qval(q, name, default=""):
            try:
                if isinstance(q, dict):
                    return q.get(name, default)
                if hasattr(q, name):
                    return getattr(q, name)
                if hasattr(q, "model_dump"):
                    return q.model_dump().get(name, default)
                if hasattr(q, "dict"):
                    return q.dict().get(name, default)
            except Exception:
                return default
            return default
        for q in quotes:
            url = (_qval(q, "url", "") or "").strip()
            dom = (_qval(q, "domain", "") or "").lower()
            sid = url or dom
            if not sid or sid in seen:
                continue
            seen.add(sid)
            summary = _qval(q, "doc_summary", "") or ""
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
        # Use focus areas from EvidenceBundle
        focus: List[str] = []
        try:
            eb = getattr(context, "evidence_bundle", None)
            if eb is not None and getattr(eb, "focus_areas", None):
                focus = list(getattr(eb, "focus_areas", []) or [])
        except Exception:
            focus = []
        if not focus:
            return "(no coverage targets)"
        focus = focus[:max_rows]
        # Pick best URL per theme by token overlap
        rows: List[str] = ["Theme | Covered? | Best Domain"]
        def _tok(t: str) -> set:
            import re
            return set([w for w in re.findall(r"[A-Za-z0-9]+", (t or "").lower()) if len(w) > 2])
        results_for_scan = list(context.search_results or [])
        for theme in focus:
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
                dom = (best.get("domain") or best.get("source") or "")  # type: ignore[assignment]
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

        # Normalize timestamp and preserve raw if needed
        meta = dict(source.get("metadata", {}) or {})
        ts_val = source.get("published_date")
        ts_norm: Optional[datetime] = None
        if isinstance(ts_val, datetime):
            ts_norm = ts_val
        elif isinstance(ts_val, str):
            try:
                # Try simple ISO formats
                ts_norm = datetime.fromisoformat(ts_val)
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
        domains = { (r.get("domain") or r.get("source") or "").lower() for r in results }
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
        self.citation_counter = 0
        self.citations = {}

        sections = []
        for section_def in self.get_section_structure():
            section = await self._generate_section(context, section_def)
            sections.append(section)

        # Top‑line recommendation (2–3 sentences) then short summary
        topline = self._compose_topline_recommendation(context)
        base_summary = sections[0].content[:300] + "..." if sections else ""
        summary = (topline + "\n\n" + base_summary).strip()

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
        """Generate a single section"""
        # Get progress tracker if available
        research_id = context.metadata.get("research_id") if hasattr(context, "metadata") else None
        progress_tracker = self._get_progress_tracker()

        # Track section operations
        section_name = section_def.get('title', 'Section')

        # Establish how many granular operations this helper performs so we
        # can provide determinate progress. The four key operations are:
        # 1. filter sources, 2. create citations, 3. generate content,
        # 4. extract insights.
        total_sub_ops = 4

        # Filter relevant results (sub-op 1/4)
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="synthesis",
                message=f"{section_name}: Filtering relevant sources",
                items_done=0,
                items_total=total_sub_ops,
            )
        relevant_results = self._top_relevant_results(context, 5)

        # Create citations (sub-op 2/4)
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="synthesis",
                message=f"{section_name}: Creating citations",
                items_done=1,
                items_total=total_sub_ops,
            )
        citation_ids = []
        for result in relevant_results:
            citation = self.create_citation(result, "evidence")
            citation_ids.append(citation.id)

        # Generate content with LLM or fallback (sub-op 3/4)
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="synthesis",
                message=f"{section_name}: Generating content",
                items_done=2,
                items_total=total_sub_ops,
            )
        try:
            # Isolation-only support: summarize findings
            iso_lines = []
            try:
                iso = self._get_isolated_findings(context)
                for m in (iso.get("matches", []) or [])[:5]:
                    dom = m.get("domain", "")
                    for frag in (m.get("fragments", []) or [])[:1]:
                        iso_lines.append(f"- [{dom}] {sanitize_snippet(frag or '')}")
            except Exception:
                pass
            iso_block = "\n".join(iso_lines) if iso_lines else "(no isolated findings)"
            # Decide evidence context mode (inline vs window vs auto)
            section_tokens = int(SYNTHESIS_BASE_TOKENS * section_def['weight'])
            mode = (os.getenv("EVIDENCE_CONTEXT_MODE", "auto") or "auto").lower()
            threshold = int(os.getenv("EVIDENCE_CONTEXT_SECTION_TOKENS_MIN", "1500"))
            inline_ctx = True if mode == "inline" else False
            use_windows = False
            if mode == "window":
                use_windows = True
                inline_ctx = False
            elif mode == "auto":
                if section_tokens >= threshold:
                    use_windows = True
                    inline_ctx = False
                else:
                    inline_ctx = True
            evidence_block = self._safe_evidence_block(context, inline_ctx=inline_ctx)
            ctx_windows_block = self._context_windows_block(context) if use_windows else "(context windows disabled)"
            summaries_block = self._source_summaries_block(context) if EVIDENCE_INCLUDE_SUMMARIES else "(summaries disabled)"
            coverage_tbl = self._coverage_table(context)
            # Optional source cards
            source_cards = self._source_cards_block(context) if os.getenv("SOURCE_CARDS_ENABLE", "1") in {"1", "true", "yes"} else "(source cards disabled)"

            guard = guardrail_instruction
            variant_extra = self._variant_addendum(context, "dolores")
            newline = "\n"
            ctx_windows_line = f"Context Windows (for quotes):{newline}{ctx_windows_block}" if use_windows else ""
            prompt = f"""{guard}
            Write the "{section_def['title']}" section focusing on: {section_def['focus']}
            Query: {context.query}
            Use passionate, urgent language that exposes injustice and calls for change.
            Source Cards (context only):
            {source_cards}
            Evidence Quotes (primary evidence; cite by [qid]):
            {evidence_block}
            {ctx_windows_line}
            Isolated Findings (secondary evidence):
            {iso_block}
            Coverage Table (Theme | Covered? | Best Domain):
            {coverage_tbl}
            STRICT: Do not invent facts; ground claims in the Evidence Quotes above.
            Full Document Context (cite using [d###]):
            {summaries_block}
            {variant_extra}
            Length: {int(SYNTHESIS_BASE_WORDS * section_def['weight'])} words
            """

            content = await llm_client.generate_paradigm_content(
                prompt=prompt,
                paradigm="dolores",
                max_tokens=int(SYNTHESIS_BASE_TOKENS * section_def['weight']),
                temperature=0.7
            )
        except Exception as e:
            if os.getenv("LLM_STRICT", "0") == "1":
                raise
            logger.warning(f"LLM generation failed: {e}, using fallback")
            content = self._generate_fallback_content(section_def, relevant_results)

        # Extract insights (sub-op 4/4)
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="synthesis",
                message=f"{section_name}: Extracting insights",
                items_done=3,
                items_total=total_sub_ops,
            )

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

        # Mark section complete (optional UI hook)
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="synthesis",
                message=f"{section_name}: Section complete",
                items_done=total_sub_ops,
                items_total=total_sub_ops,
            )

        return section

    def _generate_fallback_content(self, section_def: Dict[str, Any], results: List[Dict[str, Any]]) -> str:
        """Generate fallback content when LLM fails"""
        content = f"# {section_def['title']}\n\n"
        content += f"This section focuses on: {section_def['focus']}\n\n"

        for result in results[:3]:
            content += f"According to {result.get('domain', 'sources')}, "
            content += f"{result.get('snippet', 'No snippet available')}\n\n"

        return content

    def _extract_insights(self, content: str) -> List[str]:
        """Extract key insights from content"""
        sentences = re.split(r'[.!?]+', content)
        insights = [s.strip() for s in sentences if len(s.strip()) > 50][:3]
        return insights

    def _generate_action_items(self, context: SynthesisContext) -> List[Dict[str, Any]]:
        """Generate paradigm-specific action items"""
        return [
            {"action": "Organize grassroots resistance", "priority": "high"},
            {"action": "Document and expose systemic failures", "priority": "high"},
            {"action": "Build coalition of affected communities", "priority": "medium"}
        ]


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
        for section_def in self.get_section_structure():
            section = await self._generate_analytical_section(
                context, section_def, statistical_insights, meta_analysis
            )
            sections.append(section)

        topline = self._compose_topline_recommendation(context)
        summary_core = self._generate_analytical_summary(sections, statistical_insights)
        summary = (topline + "\n\n" + summary_core).strip()

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
                # Surface active backend info for verification
                "llm_backend": self._get_llm_backend_info(),
            }
        )

    def _extract_statistical_insights(self, search_results: List[Dict[str, Any]]) -> List[StatisticalInsight]:
        """Extract statistical insights from search results"""
        insights = []

        for result in search_results:
            text = f"{result.get('title', '')} {result.get('snippet', '')}"

            # Extract correlations
            for match in re.finditer(STATISTICAL_PATTERNS_OPERATORS["correlation"], text):
                insights.append(StatisticalInsight(
                    metric="correlation",
                    value=float(match.group(1)),
                    unit="r",
                    context=text[max(0, match.start()-50):match.end()+50]
                ))

            # Extract p-values
            for match in re.finditer(STATISTICAL_PATTERNS_OPERATORS["p_value"], text):
                insights.append(StatisticalInsight(
                    metric="p_value",
                    value=float(match.group(1)),
                    unit="",
                    context=text[max(0, match.start()-50):match.end()+50]
                ))

            # Extract sample sizes
            for match in re.finditer(STATISTICAL_PATTERNS_OPERATORS["sample_size"], text):
                insights.append(StatisticalInsight(
                    metric="sample_size",
                    value=float(match.group(1)),
                    unit="n",
                    context=text[max(0, match.start()-50):match.end()+50]
                ))

            # Extract effect sizes
            for match in re.finditer(STATISTICAL_PATTERNS_OPERATORS["effect_size"], text):
                insights.append(StatisticalInsight(
                    metric="effect_size",
                    value=float(match.group(1)),
                    unit="d",
                    context=text[max(0, match.start()-50):match.end()+50]
                ))

        return insights

    def _perform_meta_analysis(self, search_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Perform basic meta-analysis if multiple studies found"""
        effect_sizes = []
        sample_sizes = []

        for result in search_results:
            text = f"{result.get('title', '')} {result.get('snippet', '')}"

            # Extract effect sizes
            effect_matches = re.findall(STATISTICAL_PATTERNS_OPERATORS["effect_size"], text)
            if effect_matches:
                effect_sizes.extend([float(m) for m in effect_matches])

            # Extract sample sizes
            sample_matches = re.findall(STATISTICAL_PATTERNS_OPERATORS["sample_size"], text)
            if sample_matches:
                sample_sizes.extend([int(m) for m in sample_matches])

        if len(effect_sizes) >= 3:
            return {
                "pooled_effect_size": sum(effect_sizes) / len(effect_sizes),
                "effect_size_range": (min(effect_sizes), max(effect_sizes)),
                "total_sample_size": sum(sample_sizes) if sample_sizes else None,
                "studies_analyzed": len(effect_sizes)
            }

        return None

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
        citation_ids = []
        for result in relevant_results:
            citation = self.create_citation(result, "empirical")
            citation_ids.append(citation.id)

        # Generate content
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="synthesis",
                message=f"{section_name}: Extracting statistical insights"
            )
        try:
            insights_summary = self._format_statistical_insights(statistical_insights[:5])
            # Isolation-only support: include extracted findings if present
            isolated = self._get_isolated_findings(context)
            iso_lines = []
            try:
                for m in (isolated.get("matches", []) or [])[:5]:
                    dom = m.get("domain", "")
                    for frag in (m.get("fragments", []) or [])[:1]:
                        iso_lines.append(f"- [{dom}] {sanitize_snippet(frag or '')}")
            except Exception:
                pass
            iso_block = "\n".join(iso_lines) if iso_lines else "(no isolated findings)"
            section_tokens = int(SYNTHESIS_BASE_TOKENS * section_def['weight'])
            mode = (os.getenv("EVIDENCE_CONTEXT_MODE", "auto") or "auto").lower()
            threshold = int(os.getenv("EVIDENCE_CONTEXT_SECTION_TOKENS_MIN", "1500"))
            inline_ctx = True if mode == "inline" else False
            use_windows = False
            if mode == "window":
                use_windows = True
                inline_ctx = False
            elif mode == "auto":
                if section_tokens >= threshold:
                    use_windows = True
                    inline_ctx = False
                else:
                    inline_ctx = True
            evidence_block = self._safe_evidence_block(context, inline_ctx=inline_ctx)
            ctx_windows_block = self._context_windows_block(context) if use_windows else "(context windows disabled)"
            summaries_block = self._source_summaries_block(context) if EVIDENCE_INCLUDE_SUMMARIES else "(summaries disabled)"
            coverage_tbl = self._coverage_table(context)
            source_cards = self._source_cards_block(context) if os.getenv("SOURCE_CARDS_ENABLE", "1") in {"1", "true", "yes"} else "(source cards disabled)"

            guard = guardrail_instruction
            variant_extra = self._variant_addendum(context, "bernard")
            newline = "\n"
            ctx_windows_line = f"Context Windows (for quotes):{newline}{ctx_windows_block}" if use_windows else ""
            prompt = f"""{guard}
            Write the "{section_def['title']}" section focusing on: {section_def['focus']}
            Query: {context.query}

            Source Cards (context only):
            {source_cards}

            Statistical insights available:
            {insights_summary}

            Evidence Quotes (primary evidence; cite by [qid]):
            {evidence_block}
            {ctx_windows_line}

            Isolated Findings (SSOTA Isolate Layer - secondary evidence):
            {iso_block}

            Coverage Table (Theme | Covered? | Best Domain):
            {coverage_tbl}

            {f"Meta-analysis results: {meta_analysis}" if meta_analysis else ""}

            Requirements:
            - Use precise scientific language
            - Include effect sizes, confidence intervals, and p-values where available
            - Distinguish correlation from causation
            - Acknowledge limitations
            - STRICT: Do not introduce claims not supported by the Evidence Quotes above
            Full Document Context (cite using [d###]):
            {summaries_block}
            {variant_extra}
            Length: {int(SYNTHESIS_BASE_WORDS * section_def['weight'])} words
            """

            content = await llm_client.generate_paradigm_content(
                prompt=prompt,
                paradigm="bernard",
                max_tokens=int(SYNTHESIS_BASE_TOKENS * section_def['weight']),
                temperature=0.3
            )
        except Exception as e:
            if os.getenv("LLM_STRICT", "0") == "1":
                raise
            logger.warning(f"LLM generation failed: {e}, using fallback")
            content = self._generate_analytical_fallback(section_def, relevant_results, statistical_insights)

        # Extract quantitative insights
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="synthesis",
                message=f"{section_name}: Extracting quantitative insights"
            )
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

    def _format_statistical_insights(self, insights: List[StatisticalInsight]) -> str:
        """Format statistical insights for prompt"""
        formatted = []
        for insight in insights:
            if insight.p_value:
                formatted.append(f"- {insight.metric}: {insight.value}{insight.unit} (p={insight.p_value})")
            else:
                formatted.append(f"- {insight.metric}: {insight.value}{insight.unit}")
        return "\n".join(formatted)

    def _generate_analytical_fallback(
        self,
        section_def: Dict[str, Any],
        results: List[Dict[str, Any]],
        insights: List[StatisticalInsight]
    ) -> str:
        """Generate fallback analytical content"""
        content = f"# {section_def['title']}\n\n"
        content += f"This section provides {section_def['focus']}.\n\n"

        if insights:
            content += "## Key Statistical Findings\n\n"
            for insight in insights[:5]:
                content += f"- {insight.metric}: {insight.value}{insight.unit}\n"
            content += "\n"

        content += "## Evidence Summary\n\n"
        for result in results[:3]:
            content += f"According to research from {result.get('domain', 'sources')}, "
            content += f"{result.get('snippet', 'No data available')}\n\n"

        return content

    def _extract_quantitative_insights(
        self,
        content: str,
        statistical_insights: List[StatisticalInsight]
    ) -> List[str]:
        """Extract quantitative insights from generated content"""
        insights = []

        # Extract sentences with statistical terms
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            if any(pattern in sentence.lower() for pattern in ["correlation", "p=", "n=", "effect size", "%"]):
                if len(sentence.strip()) > 30:
                    insights.append(sentence.strip())

        # Add top statistical insights if not already in content
        for stat_insight in statistical_insights[:3]:
            insight_text = f"{stat_insight.metric.replace('_', ' ').title()}: {stat_insight.value}{stat_insight.unit}"
            if insight_text not in content:
                insights.append(insight_text)

        return insights[:5]

    def _generate_analytical_summary(
        self,
        sections: List[AnswerSection],
        statistical_insights: List[StatisticalInsight]
    ) -> str:
        """Generate analytical summary"""
        if not sections:
            return "No analysis available."

        summary = sections[0].content[:200] if sections[0].content else ""

        if statistical_insights:
            summary += f" Analysis identified {len(statistical_insights)} statistical findings"

            # Add key statistics
            effect_sizes = [i for i in statistical_insights if i.metric == "effect_size"]
            if effect_sizes:
                avg_effect = sum(i.value for i in effect_sizes) / len(effect_sizes)
                summary += f" with average effect size d={avg_effect:.2f}"

        return summary

    def _generate_research_action_items(
        self,
        statistical_insights: List[StatisticalInsight]
    ) -> List[Dict[str, Any]]:
        """Generate research-oriented action items"""
        items = []

        # Check for significant findings
        significant_findings = [i for i in statistical_insights if i.p_value and i.p_value < 0.05]
        if significant_findings:
            items.append({
                "action": f"Investigate {len(significant_findings)} statistically significant findings",
                "priority": "high"
            })

        # Check for large effect sizes
        large_effects = [i for i in statistical_insights if i.metric == "effect_size" and abs(i.value) > 0.8]
        if large_effects:
            items.append({
                "action": f"Examine {len(large_effects)} large effect sizes for practical significance",
                "priority": "high"
            })

        # Always add meta-analysis recommendation
        items.append({
            "action": "Conduct systematic review and meta-analysis",
            "priority": "medium"
        })

        return items

    def _calculate_analytical_confidence(
        self,
        statistical_insights: List[StatisticalInsight]
    ) -> float:
        """Calculate confidence based on statistical evidence"""
        base_confidence = 0.5

        # Boost for significant findings
        significant_findings = [i for i in statistical_insights if i.p_value and i.p_value < 0.05]
        base_confidence += min(0.2, len(significant_findings) * 0.05)

        # Boost for large sample sizes
        large_samples = [i for i in statistical_insights if i.metric == "sample_size" and i.value > 1000]
        base_confidence += min(0.15, len(large_samples) * 0.05)

        # Boost for consistent effect sizes
        effect_sizes = [i.value for i in statistical_insights if i.metric == "effect_size"]
        if len(effect_sizes) > 2:
            variance = sum((e - sum(effect_sizes)/len(effect_sizes))**2 for e in effect_sizes) / len(effect_sizes)
            if variance < 0.1:  # Low variance indicates consistency
                base_confidence += 0.1

        return min(0.95, base_confidence)

    def _count_peer_reviewed(self, search_results: List[Dict[str, Any]]) -> int:
        """Count peer-reviewed sources"""
        peer_reviewed_domains = ["pubmed", "arxiv", "nature", "science", "elsevier", "springer"]
        count = 0
        for result in search_results:
            domain = result.get("domain", "").lower()
            if any(pr in domain for pr in peer_reviewed_domains):
                count += 1
        return count


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

    def _extract_strategic_insights(self, search_results: List[Dict[str, Any]]) -> List[StrategicRecommendation]:
        """Extract strategic insights from search results"""
        insights = []

        for result in search_results[:5]:
            text = f"{result.get('title', '')} {result.get('snippet', '')}"

            # Look for strategic patterns
            if "strategy" in text.lower() or "competitive" in text.lower():
                insights.append(StrategicRecommendation(
                    title="Competitive Strategy",
                    description=result.get('snippet', '')[:200],
                    impact="high",
                    effort="medium",
                    timeline="3-6 months",
                    dependencies=["market analysis", "resource allocation"],
                    success_metrics=["market share", "revenue growth"],
                    risks=["competitor response", "execution challenges"]
                ))

        return insights

    def _generate_swot_analysis(
        self,
        query: str,
        search_results: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Generate SWOT analysis from search results"""
        swot = {
            "strengths": [],
            "weaknesses": [],
            "opportunities": [],
            "threats": []
        }

        for result in search_results[:10]:
            text = f"{result.get('title', '')} {result.get('snippet', '')}".lower()

            if "opportunity" in text or "growth" in text:
                swot["opportunities"].append(result.get('snippet', '')[:100])
            elif "threat" in text or "risk" in text:
                swot["threats"].append(result.get('snippet', '')[:100])
            elif "strength" in text or "advantage" in text:
                swot["strengths"].append(result.get('snippet', '')[:100])
            elif "weakness" in text or "challenge" in text:
                swot["weaknesses"].append(result.get('snippet', '')[:100])

        # Ensure each category has at least one item
        for category in swot:
            if not swot[category]:
                swot[category].append(f"Further analysis needed for {category}")

        return swot

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
        citation_ids = []
        for result in relevant_results:
            citation = self.create_citation(result, "strategic")
            citation_ids.append(citation.id)

        # Generate content
        try:
            swot_summary = self._format_swot_for_prompt(swot_analysis)

            # Isolation-only support
            iso_lines = []
            try:
                iso = self._get_isolated_findings(context)
                for m in (iso.get("matches", []) or [])[:5]:
                    dom = m.get("domain", "")
                    for frag in (m.get("fragments", []) or [])[:1]:
                        iso_lines.append(f"- [{dom}] {sanitize_snippet(frag or '')}")
            except Exception:
                pass
            iso_block = "\n".join(iso_lines) if iso_lines else "(no isolated findings)"
            section_tokens = int(SYNTHESIS_BASE_TOKENS * section_def['weight'])
            mode = (os.getenv("EVIDENCE_CONTEXT_MODE", "auto") or "auto").lower()
            threshold = int(os.getenv("EVIDENCE_CONTEXT_SECTION_TOKENS_MIN", "1500"))
            inline_ctx = True if mode == "inline" else False
            use_windows = False
            if mode == "window":
                use_windows = True
                inline_ctx = False
            elif mode == "auto":
                if section_tokens >= threshold:
                    use_windows = True
                    inline_ctx = False
                else:
                    inline_ctx = True
            evidence_block = self._safe_evidence_block(context, inline_ctx=inline_ctx)
            ctx_windows_block = self._context_windows_block(context) if use_windows else "(context windows disabled)"
            summaries_block = self._source_summaries_block(context) if EVIDENCE_INCLUDE_SUMMARIES else "(summaries disabled)"
            coverage_tbl = self._coverage_table(context)
            source_cards = self._source_cards_block(context) if os.getenv("SOURCE_CARDS_ENABLE", "1") in {"1", "true", "yes"} else "(source cards disabled)"

            guard = guardrail_instruction
            variant_extra = self._variant_addendum(context, "maeve")
            newline = "\n"
            ctx_windows_line = f"Context Windows (for quotes):{newline}{ctx_windows_block}" if use_windows else ""
            prompt = f"""{guard}
            Write the "{section_def['title']}" section focusing on: {section_def['focus']}
            Query: {context.query}

            Source Cards (context only):
            {source_cards}

            SWOT Analysis:
            {swot_summary}

            Evidence Quotes (primary evidence; cite by [qid]):
            {evidence_block}
            {ctx_windows_line}
            Isolated Findings (secondary evidence):
            {iso_block}

            Coverage Table (Theme | Covered? | Best Domain):
            {coverage_tbl}

            Requirements:
            - Focus on actionable strategies and concrete recommendations
            - Include ROI considerations and resource implications
            - Emphasize competitive advantages and market opportunities

            Full Document Context (cite using [d###]):
            {summaries_block}
            {variant_extra}
            Length: {int(SYNTHESIS_BASE_WORDS * section_def['weight'])} words
            """

            content = await llm_client.generate_paradigm_content(
                prompt=prompt,
                paradigm="maeve",
                max_tokens=int(SYNTHESIS_BASE_TOKENS * section_def['weight']),
                temperature=0.5
            )
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

    def _format_swot_for_prompt(self, swot: Dict[str, List[str]]) -> str:
        """Format SWOT analysis for prompt"""
        formatted = []
        for category, items in swot.items():
            formatted.append(f"{category.upper()}:")
            for item in items[:3]:
                formatted.append(f"  - {item}")
        return "\n".join(formatted)

    def _generate_strategic_fallback(
        self,
        section_def: Dict[str, Any],
        results: List[Dict[str, Any]],
        swot: Dict[str, List[str]]
    ) -> str:
        """Generate fallback strategic content"""
        content = f"# {section_def['title']}\n\n"
        content += f"This section addresses: {section_def['focus']}\n\n"

        # Add SWOT summary
        content += "## Strategic Analysis\n\n"
        for category, items in swot.items():
            if items:
                content += f"**{category.title()}:**\n"
                for item in items[:2]:
                    content += f"- {item}\n"
                content += "\n"

        # Add source insights
        content += "## Market Intelligence\n\n"
        for result in results[:3]:
            content += f"According to {result.get('domain', 'market analysis')}, "
            content += f"{result.get('snippet', 'No data available')}\n\n"

        return content

    def _extract_strategic_insights_from_content(self, content: str) -> List[str]:
        """Extract strategic insights from generated content"""
        insights = []

        # Look for sentences with strategic keywords
        sentences = re.split(r'[.!?]+', content)
        strategic_keywords = ["roi", "market", "competitive", "strategy", "growth", "opportunity"]

        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in strategic_keywords):
                if len(sentence.strip()) > 40:
                    insights.append(sentence.strip())

        return insights[:5]

    def _generate_strategic_recommendations(
        self,
        query: str,
        strategic_insights: List[StrategicRecommendation],
        swot: Dict[str, List[str]]
    ) -> List[StrategicRecommendation]:
        """Generate strategic recommendations"""
        recommendations = []

        # Quick win based on opportunities
        if swot["opportunities"]:
            recommendations.append(StrategicRecommendation(
                title="Quick Win Opportunity",
                description=swot["opportunities"][0],
                impact="medium",
                effort="low",
                timeline="1-3 months",
                dependencies=["minimal resources"],
                success_metrics=["early adoption", "proof of concept"],
                risks=["limited scope"],
                roi_potential=2.5
            ))

        # Strategic initiative based on strengths
        if swot["strengths"]:
            recommendations.append(StrategicRecommendation(
                title="Leverage Core Strength",
                description=f"Build on {swot['strengths'][0]}",
                impact="high",
                effort="medium",
                timeline="6-12 months",
                dependencies=["strategic alignment", "resource commitment"],
                success_metrics=["market differentiation", "competitive advantage"],
                risks=["resource intensity"],
                roi_potential=4.0
            ))

        # Risk mitigation based on threats
        if swot["threats"]:
            recommendations.append(StrategicRecommendation(
                title="Risk Mitigation Strategy",
                description=f"Address threat: {swot['threats'][0]}",
                impact="high",
                effort="high",
                timeline="3-6 months",
                dependencies=["risk assessment", "contingency planning"],
                success_metrics=["risk reduction", "resilience"],
                risks=["opportunity cost"],
                roi_potential=1.5
            ))

        return recommendations

    def _generate_strategic_summary(
        self,
        sections: List[AnswerSection],
        strategic_insights: List[StrategicRecommendation]
    ) -> str:
        """Generate strategic summary"""
        if not sections:
            return "Strategic analysis pending."

        summary = sections[0].content[:200] if sections[0].content else ""

        if strategic_insights:
            high_impact = [i for i in strategic_insights if i.impact == "high"]
            summary += f" Identified {len(high_impact)} high-impact strategic opportunities."

        return summary

    def _format_recommendations_as_actions(
        self,
        recommendations: List[StrategicRecommendation]
    ) -> List[Dict[str, Any]]:
        """Format recommendations as action items"""
        actions = []
        for rec in recommendations:
            actions.append({
                "action": rec.title,
                "description": rec.description,
                "priority": rec.impact,
                "timeline": rec.timeline,
                "roi_potential": rec.roi_potential
            })
        return actions


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
        citation_ids = []
        for result in relevant_results:
            citation = self.create_citation(result, "support")
            citation_ids.append(citation.id)

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
                max_tokens=int(SYNTHESIS_BASE_TOKENS * section_def['weight']),
                temperature=0.6
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
        """Generate fallback supportive content"""
        content = f"# {section_def['title']}\n\n"
        content += f"This section provides: {section_def['focus']}\n\n"

        content += "## Available Resources\n\n"
        for result in results[:3]:
            content += f"Support is available through {result.get('domain', 'various organizations')}. "
            content += f"{result.get('snippet', 'Help and resources are available.')}\n\n"

        content += "\nRemember: You are not alone. Help is available, and together we can make a difference.\n"

        return content

    def _extract_supportive_insights(self, content: str) -> List[str]:
        """Extract supportive insights"""
        insights = []
        sentences = re.split(r'[.!?]+', content)

        supportive_keywords = ["help", "support", "care", "resource", "available", "together"]

        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in supportive_keywords):
                if len(sentence.strip()) > 40:
                    insights.append(sentence.strip())

        return insights[:3]

    def _generate_supportive_summary(self, sections: List[AnswerSection]) -> str:
        """Generate supportive summary"""
        if not sections:
            return "Support and resources are available."

        summary = sections[0].content[:200] if sections[0].content else ""
        summary += " Help is available, and no one has to face this alone."

        return summary

    def _generate_supportive_actions(self, context: SynthesisContext) -> List[Dict[str, Any]]:
        """Generate supportive action items"""
        return [
            {"action": "Connect with local support groups", "priority": "high"},
            {"action": "Access available resources and assistance programs", "priority": "high"},
            {"action": "Build community support network", "priority": "medium"},
            {"action": "Share resources with those in need", "priority": "medium"}
        ]


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
