"""
Unified Research Orchestrator V2
Combines all the best features from multiple orchestrator implementations
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, Tuple, Set, cast, TypedDict
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, field

# Contracts
from contracts import ResearchStatus as ContractResearchStatus  # type: ignore
# json import removed (unused)
import re
from services.research_store import research_store
from models.base import ResearchStatus as RuntimeResearchStatus

from models.context_models import (
    SearchResultSchema, ClassificationResultSchema,
    ContextEngineeredQuerySchema,
    HostParadigm,
    QueryFeaturesSchema
)
from models.paradigms import normalize_to_internal_code
from services.search_apis import (
    SearchResult,
    SearchConfig,
    create_search_manager,
)
from services.credibility import get_source_credibility
from services.paradigm_search import get_search_strategy, SearchContext
from services.cache import (
    cache_manager,
)
from services.text_compression import query_compressor
from core.config import (
    EVIDENCE_MAX_DOCS_DEFAULT,
    EVIDENCE_QUOTES_PER_DOC_DEFAULT,
    EVIDENCE_BUDGET_TOKENS_DEFAULT,
    SYNTHESIS_MAX_LENGTH_DEFAULT,
)
from services.deep_research_service import (
    deep_research_service,
    DeepResearchConfig,
    DeepResearchMode,
    initialize_deep_research,
)
from services.agentic_process import (
    evaluate_coverage_from_sources,
    summarize_domain_gaps,
    propose_queries_enriched,
)
from services.result_adapter import ResultAdapter
from services.metrics import metrics
# SearchContextSize import removed (unused)

# Optional: answer generation integration
try:
    from services.answer_generator import answer_orchestrator
    from models.synthesis_models import SynthesisContext as SynthesisContextModel
    _ANSWER_GEN_AVAILABLE = True
except Exception:
    _ANSWER_GEN_AVAILABLE = False
    answer_orchestrator = None  # type: ignore
    SynthesisContextModel = None  # type: ignore

logger = logging.getLogger(__name__)


def bernard_min_relevance() -> float:
    try:
        return float(os.getenv("SEARCH_MIN_RELEVANCE_BERNARD", "0.15"))
    except Exception:
        return 0.15


def default_source_limit() -> int:
    """Resolve the default source limit from env or use a sensible default.
    UPDATED FOR O3: Increased from 50 to 200 to analyze 4x more sources.
    This value is used anywhere a `source_limit` is missing from user context.
    """
    try:
        # UPDATED FOR O3: Changed default from 50 to 200
        v = int(os.getenv("DEFAULT_SOURCE_LIMIT", "200") or 200)
        return max(1, v)
    except Exception:
        return 200  # UPDATED FOR O3: Changed from 50 to 200


@dataclass
class ResearchExecutionResult:
    """Complete research execution result"""
    original_query: str
    paradigm: str
    secondary_paradigm: Optional[str]
    search_queries_executed: List[Dict[str, Any]]
    raw_results: Dict[str, List[SearchResult]]  # Results by API
    # Back-compat: keep original attribute and add a safe alias
    filtered_results: List[SearchResult]
    # Alias to avoid AttributeError in any consumer expecting 'results'
    # Note: property defined below to mirror filtered_results
    credibility_scores: Dict[str, float]  # Domain -> score
    # Non-default fields first
    execution_metrics: Dict[str, Any]
    cost_breakdown: Dict[str, float]
    # Defaults after non-defaults to satisfy dataclass rules
    status: ContractResearchStatus = ContractResearchStatus.OK
    secondary_results: List[SearchResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    deep_research_content: Optional[str] = None

    # Provide a read-only alias so obj.results returns filtered_results
    @property
    def results(self) -> List[SearchResult]:
        return self.filtered_results





class ResultDeduplicator:
    """Removes duplicate search results using various similarity measures"""

    def __init__(self, similarity_threshold: float = 0.8):
        # Allow tuning via env; default to a less aggressive 0.9 to reduce
        # accidental drops of near-duplicate but distinct items.
        try:
            env_thr = os.getenv("DEDUP_SIMILARITY_THRESH")
            if env_thr is not None:
                similarity_threshold = float(env_thr)
        except Exception:
            pass
        self.similarity_threshold = similarity_threshold

    def _simhash64(self, text: str) -> int:
        import hashlib
        if not text:
            return 0
        # Tokenize lightly and hash tokens; simple 64-bit simhash
        toks = re.findall(r"[A-Za-z0-9]+", text.lower())
        v = [0] * 64
        for t in toks[:200]:  # cap tokens for speed
            h = int(hashlib.blake2b(t.encode('utf-8'), digest_size=8).hexdigest(), 16)
            for i in range(64):
                v[i] += 1 if (h >> i) & 1 else -1
        out = 0
        for i in range(64):
            if v[i] >= 0:
                out |= (1 << i)
        return out

    @staticmethod
    def _hamdist64(a: int, b: int) -> int:
        x = a ^ b
        # Kernighan bit count
        c = 0
        while x:
            x &= x - 1
            c += 1
        return c

    async def deduplicate_results(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Remove duplicate results using URL and content similarity"""
        # Defensive guard – empty input
        if not results:
            return {
                "unique_results": [],
                "duplicates_removed": 0,
                "similarity_threshold": self.similarity_threshold,
            }

        unique_results: List[SearchResult] = []
        duplicates_removed = 0
        seen_urls: set[str] = set()

        # First pass: exact URL‐based deduplication.
        # Skip any items that do not expose a usable ``url`` attribute
        # to avoid unexpected ``AttributeError`` crashes that would halt
        # the entire research pipeline mid-progress (observed by UI
        # getting stuck at the *“Removing duplicate results…”* stage).
        url_deduplicated: List[SearchResult] = []
        for result in results:
            url = getattr(result, "url", None)

            # Verify a non-empty URL exists – otherwise drop the record
            if not url or not isinstance(url, str):
                logger.debug(
                    "[dedup] Dropping result without valid URL: %s", repr(result)[:100]
                )
                duplicates_removed += 1
                continue

            if url not in seen_urls:
                seen_urls.add(url)
                url_deduplicated.append(result)
            else:
                duplicates_removed += 1

        # Second pass: content-similarity via simhash bucketing to avoid O(n^2)
        # Build buckets on leading bits; compare only within bucket
        buckets: Dict[int, List[Tuple[int, SearchResult]]] = {}
        for r in url_deduplicated:
            try:
                basis = f"{getattr(r,'title','')} {getattr(r,'snippet','')}"
                sh = self._simhash64(basis)
                key = (sh >> 52)  # top 12 bits as bucket key
                buckets.setdefault(key, []).append((sh, r))
            except Exception:
                buckets.setdefault(0, []).append((0, r))

        for key, items in buckets.items():
            reps: List[Tuple[int, SearchResult]] = []
            for sh, r in items:
                is_dup = False
                for sh2, kept in reps:
                    try:
                        # Adaptive threshold based on result type and domain
                        result_type = getattr(r, 'result_type', None) or (r.metadata or {}).get('result_type', 'web')
                        domain = getattr(r, 'domain', None) or (r.metadata or {}).get('domain', '')

                        # More lenient for academic and government sources
                        if result_type == 'academic' or '.edu' in domain or '.gov' in domain or 'arxiv' in domain:
                            hamming_threshold = 8  # Very lenient for academic papers
                        elif result_type in ('news', 'blog'):
                            hamming_threshold = 5  # Moderate for news/blogs
                        else:
                            hamming_threshold = 3  # Strict for general web content

                        # Check SimHash with adaptive threshold
                        if self._hamdist64(sh, sh2) <= hamming_threshold:
                            is_dup = True
                        else:
                            sim = self._calculate_content_similarity(r, kept)
                            is_dup = sim > self.similarity_threshold
                    except Exception:
                        is_dup = False
                    if is_dup:
                        duplicates_removed += 1
                        break
                if not is_dup:
                    reps.append((sh, r))
                    unique_results.append(r)

        logger.info(
            "Deduplication: %s -> %s (%s duplicates removed)",
            len(results),
            len(unique_results),
            duplicates_removed,
        )

        return {
            "unique_results": unique_results,
            "duplicates_removed": duplicates_removed,
            "similarity_threshold": self.similarity_threshold,
        }

    def _calculate_content_similarity(
        self, result1: SearchResult, result2: SearchResult
    ) -> float:
        """Calculate similarity between two search results"""
        try:
            h1 = getattr(result1, "content_hash", None)
            h2 = getattr(result2, "content_hash", None)
            if h1 and h2 and isinstance(h1, str) and isinstance(h2, str) and h1 == h2:
                return 1.0
        except Exception:
            pass
        # Defensive extraction helpers – return empty set when value missing
        def words(text: Optional[str]) -> set[str]:
            if not text or not isinstance(text, str):
                return set()
            return set(text.lower().split())

        # Title similarity (Jaccard index)
        title1_words = words(getattr(result1, "title", ""))
        title2_words = words(getattr(result2, "title", ""))

        if not title1_words or not title2_words:
            title_similarity = 0.0
        else:
            intersection = len(title1_words.intersection(title2_words))
            union = len(title1_words.union(title2_words))
            title_similarity = intersection / union if union > 0 else 0.0

        # Domain similarity – ensure attributes exist
        domain1 = getattr(result1, "domain", "") or ""
        domain2 = getattr(result2, "domain", "") or ""
        domain_similarity = 1.0 if domain1 == domain2 and domain1 else 0.0

        # Snippet similarity
        snippet1_words = words(getattr(result1, "snippet", ""))
        snippet2_words = words(getattr(result2, "snippet", ""))

        if not snippet1_words or not snippet2_words:
            snippet_similarity = 0.0
        else:
            intersection = len(snippet1_words.intersection(snippet2_words))
            union = len(snippet1_words.union(snippet2_words))
            snippet_similarity = intersection / union if union > 0 else 0.0

        # Weighted combination (configurable + adaptive)
        try:
            w_title = float(os.getenv("DEDUP_TITLE_WEIGHT", "0.5"))
            w_domain = float(os.getenv("DEDUP_DOMAIN_WEIGHT", "0.3"))
            w_snippet = float(os.getenv("DEDUP_SNIPPET_WEIGHT", "0.2"))
        except Exception:
            w_title, w_domain, w_snippet = 0.5, 0.3, 0.2
        # Adapt weights for academic results: titles are often standardized; rely more on domain + snippet
        try:
            if (getattr(result1, "result_type", "") == "academic" or getattr(result2, "result_type", "") == "academic"):
                w_title, w_domain, w_snippet = 0.4, max(w_domain, 0.35), 0.25
        except Exception:
            pass
        total_w = max(1e-9, w_title + w_domain + w_snippet)
        w_title, w_domain, w_snippet = w_title/total_w, w_domain/total_w, w_snippet/total_w
        overall_similarity = (
            title_similarity * w_title
            + domain_similarity * w_domain
            + snippet_similarity * w_snippet
        )

        return overall_similarity


class EarlyRelevanceFilter:
    """Early-stage relevance filtering to remove obviously irrelevant results"""

    def __init__(self):
        self.spam_indicators = {
            'viagra', 'cialis', 'casino', 'poker', 'lottery',
            'weight loss', 'get rich quick', 'work from home',
            'singles in your area', 'hot deals', 'limited time offer'
        }

        # Allow custom spam indicators via environment
        try:
            custom_spam = os.getenv("EARLY_FILTER_SPAM_INDICATORS")
            if custom_spam:
                additional_spam = [s.strip().lower() for s in custom_spam.split(",") if s.strip()]
                self.spam_indicators.update(additional_spam)
        except Exception:
            pass

        # Domain blocklist can be disabled via EARLY_FILTER_BLOCK_DOMAINS=0
        try:
            if os.getenv("EARLY_FILTER_BLOCK_DOMAINS", "1").lower() in {"1", "true", "yes", "on"}:
                self.low_quality_domains = {
                    'ezinearticles.com', 'articlesbase.com', 'squidoo.com',
                    'hubpages.com', 'buzzle.com', 'ehow.com'
                }
            else:
                self.low_quality_domains = set()
        except Exception:
            self.low_quality_domains = set()

    def is_relevant(self, result: SearchResult, query: str, paradigm: str) -> bool:
        """Check if a result meets minimum relevance criteria"""

        # Check for spam content
        title_val = (getattr(result, "title", "") or "")
        snippet_val = (getattr(result, "snippet", "") or "")
        combined_text = f"{title_val} {snippet_val}".lower()
        if any(spam in combined_text for spam in self.spam_indicators):
            return False

        # Check for low-quality domains
        if (getattr(result, "domain", "") or "").lower() in self.low_quality_domains:
            return False

        # Minimum content check
        if not isinstance(title_val, str) or len(title_val.strip()) < 10:
            return False
        if not isinstance(snippet_val, str) or len(snippet_val.strip()) < 20:
            return False

        # Language detection (basic check for non-English) - configurable by context
        # Allow override via user context language or environment variable
        language = getattr(result, 'language', 'en') or 'en'  # some APIs provide language hint
        ascii_threshold = 0.3  # default 30%

        # Adjust threshold for non-English queries or when explicitly allowed
        if language != 'en':
            ascii_threshold = 0.8  # be more permissive for non-English content

        # Allow environment override
        try:
            ascii_threshold = float(os.getenv("EARLY_FILTER_ASCII_THRESHOLD", str(ascii_threshold)))
        except Exception:
            pass

        non_ascii_count = sum(1 for c in combined_text if ord(c) > 127)
        if len(combined_text) > 0 and non_ascii_count > len(combined_text) * ascii_threshold:
            return False

        # Query relevance check - use query_compressor for better keyword extraction
        try:
            import re as _re
            query_terms = set(query_compressor.extract_keywords(query))
            if query_terms:
                # Use regex tokenization for more flexible matching
                def _toks(text: str) -> set[str]:
                    return {t for t in _re.findall(r"[A-Za-z0-9]+", (text or "").lower()) if len(t) > 2}

                result_terms = _toks(combined_text)
                if result_terms:
                    # Check for direct term matches
                    has_query_term = bool(query_terms & result_terms)

                    # If no direct matches, check for partial matches with shorter keywords
                    if not has_query_term:
                        # Try with 2-character minimum for short keywords
                        short_query_terms = {t for t in query_terms if len(t) >= 2}
                        short_result_terms = {t for t in _re.findall(r"[A-Za-z0-9]+", (combined_text or "").lower()) if len(t) >= 2}
                        has_query_term = bool(short_query_terms & short_result_terms)

                    if not has_query_term:
                        return False
        except Exception:
            # Fallback to simple split if query_compressor fails
            query_terms = [term.lower() for term in query.split() if len(term) > 2]
            if query_terms:
                has_query_term = any(term in combined_text for term in query_terms)
                if not has_query_term:
                    return False

        # Check for duplicate/mirror sites
        if self._is_likely_duplicate_site((getattr(result, "domain", "") or "")):
            return False

        # Paradigm-specific early filters
        if paradigm == "bernard" and getattr(result, "result_type", "web") == "web":
            authoritative_indicators = ['.edu', '.gov', 'journal', 'research', 'study', 'analysis',
                                      'technology', 'innovation', 'science', 'ieee', 'acm', 'mit',
                                      'stanford', 'harvard', 'arxiv', 'nature', 'springer']
            tech_indicators = ['ai', 'artificial intelligence', 'machine learning', 'deep learning',
                             'neural', 'algorithm', 'technology', 'computing', 'software', 'innovation']

            has_authority = any(indicator in (getattr(result, "domain", "") or "").lower() or indicator in combined_text for indicator in authoritative_indicators)
            has_tech_content = any(indicator in combined_text for indicator in tech_indicators)

            # Be permissive when there is direct query overlap – do not penalize generic but relevant snippets
            has_direct_overlap = True
            try:
                q_terms = set(query_compressor.extract_keywords(query))
            except Exception:
                q_terms = {t for t in (query or "").lower().split() if len(t) > 2}
            if q_terms:
                has_direct_overlap = any(t in combined_text for t in q_terms)

            if not (has_authority or has_tech_content) and not has_direct_overlap:
                academic_terms = ['methodology', 'hypothesis', 'conclusion', 'abstract', 'citation',
                                'analysis', 'framework', 'approach', 'technique', 'evaluation']
                if not any(term in combined_text for term in academic_terms):
                    return False

        # Folded-in theme overlap (Jaccard) filter - with domain whitelisting
        try:
            thr = float(os.getenv("EARLY_THEME_OVERLAP_MIN", "0.08") or 0.08)
            q_terms = set(query_compressor.extract_keywords(query))
            if q_terms:
                import re as _re
                def _toks(text: str) -> set:
                    return set([t for t in _re.findall(r"[A-Za-z0-9]+", (text or "").lower()) if len(t) > 2])
                rtoks = _toks(combined_text)
                if rtoks:
                    inter = len(q_terms & rtoks)
                    union = len(q_terms | rtoks)
                    jac = (inter / float(union)) if union else 0.0

                    # Whitelist well-known domains even with low overlap
                    domain = (getattr(result, "domain", "") or "").lower()
                    whitelist_domains = {
                        'arxiv.org', 'scholar.google.com', 'pubmed.ncbi.nlm.nih.gov',
                        'nature.com', 'science.org', 'ieee.org', 'acm.org',
                        'springer.com', 'wiley.com', 'tandfonline.com', 'researchgate.net'
                    }

                    # Only apply threshold if not in whitelist
                    if domain not in whitelist_domains and jac < thr:
                        return False
        except Exception:
            pass

        return True

    def _is_likely_duplicate_site(self, domain: str) -> bool:
        """Check if domain is likely a duplicate/mirror site"""
        duplicate_patterns = [
            r'.*-mirror\.', r'.*-cache\.', r'.*-proxy\.',
            r'.*\.mirror\.', r'.*\.cache\.', r'.*\.proxy\.',
            r'webcache\.', r'cached\.'
        ]

        for pattern in duplicate_patterns:
            if re.match(pattern, domain.lower()):
                return True

        return False


class CostMonitor:
    """Monitors and tracks API costs"""

    def __init__(self):
        self.cost_per_call = {
            "google": 0.005,  # $5 per 1000 queries
            "brave": 0.0,     # Free tier
            "arxiv": 0.0,     # Free
            "pubmed": 0.0,    # Free
        }

    async def track_search_cost(self, api_name: str, queries_count: int) -> float:
        """Track cost for search API calls"""
        cost = self.cost_per_call.get(api_name, 0.0) * queries_count
        await cache_manager.track_api_cost(api_name, cost, queries_count)
        return cost

    async def estimate_and_track_aggregate(self, providers: List[str], queries_count: int) -> Dict[str, float]:
        """Estimate and record per‑provider costs for an aggregate search.

        Returns a mapping of provider -> cost for this batch. Unknown providers are
        treated as zero-cost (defensive default).
        """
        breakdown: Dict[str, float] = {}
        for name in providers:
            try:
                # Record per‑provider (not "aggregate") so daily roll‑ups stay accurate
                breakdown[name] = await self.track_search_cost(name, queries_count)
            except Exception:
                breakdown[name] = 0.0
        return breakdown

    async def get_daily_costs(self, date: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Get daily API costs"""
        return await cache_manager.get_daily_api_costs(date)


class RetryPolicy:
    """Standardized retry/backoff policy configuration."""
    def __init__(self, max_attempts: int = 3, base_delay_sec: float = 0.5, max_delay_sec: float = 8.0) -> None:
        self.max_attempts = max_attempts
        self.base_delay_sec = base_delay_sec
        self.max_delay_sec = max_delay_sec


@dataclass
class ToolCapability:
    name: str
    cost_per_call_usd: float = 0.0
    rpm_limit: Optional[int] = None
    rpd_limit: Optional[int] = None
    typical_latency_ms: Optional[int] = None
    failure_types: List[str] = field(default_factory=list)
    healthy: bool = True
    last_health_check: Optional[datetime] = None


class ToolRegistry:
    """Registry for tool capabilities, costs, limits, and health status."""
    def __init__(self) -> None:
        self._tools: Dict[str, ToolCapability] = {}

    def register(self, capability: ToolCapability) -> None:
        self._tools[capability.name] = capability

    def get(self, name: str) -> Optional[ToolCapability]:
        return self._tools.get(name)

    def list(self) -> List[ToolCapability]:
        return list(self._tools.values())

    def set_health(self, name: str, healthy: bool) -> None:
        cap = self._tools.get(name)
        if cap:
            cap.healthy = healthy
            cap.last_health_check = datetime.now()


@dataclass
class Budget:
    max_tokens: int
    max_cost_usd: float
    max_wallclock_minutes: int


@dataclass
class PlannerCheckpoint:
    name: str
    description: str
    done: bool = False


@dataclass
class Plan:
    objective: str
    checkpoints: List[PlannerCheckpoint]
    budget: Budget
    stop_conditions: Dict[str, Any] = field(default_factory=dict)
    consumed_cost_usd: float = 0.0
    consumed_tokens: int = 0
    started_at: datetime = field(default_factory=datetime.now)

    def can_spend(self, additional_cost_usd: float, additional_tokens: int) -> bool:
        within_cost = (self.consumed_cost_usd + additional_cost_usd) <= self.budget.max_cost_usd
        within_tokens = (self.consumed_tokens + additional_tokens) <= self.budget.max_tokens
        return within_cost and within_tokens

    def spend(self, cost_usd: float, tokens: int) -> None:
        self.consumed_cost_usd += max(0.0, cost_usd)
        self.consumed_tokens += max(0, tokens)


class BudgetAwarePlanner:
    """Simple budget-aware planner that selects tools and enforces spend."""
    def __init__(self, registry: ToolRegistry, retry_policy: Optional[RetryPolicy] = None) -> None:
        self.registry = registry
        self.retry_policy = retry_policy or RetryPolicy()

    def select_tools(self, preferred: List[str]) -> List[str]:
        tools: List[str] = []
        for name in preferred:
            cap = self.registry.get(name)
            if cap and cap.healthy:
                tools.append(name)
        return tools

    def estimate_cost(self, tool_name: str, calls: int = 1) -> float:
        cap = self.registry.get(tool_name)
        return (cap.cost_per_call_usd * calls) if cap else 0.0

    def record_tool_spend(self, plan: Plan, tool_name: str, calls: int, tokens: int = 0) -> bool:
        cost = self.estimate_cost(tool_name, calls)
        if not plan.can_spend(cost, tokens):
            return False
        plan.spend(cost, tokens)
        return True


class SearchMetrics(TypedDict, total=False):
    total_queries: int
    total_results: int
    apis_used: List[str]
    deduplication_rate: float
    retries_attempted: int
    task_timeouts: int
    exceptions_by_api: Dict[str, int]
    api_call_counts: Dict[str, int]
    dropped_no_url: int
    dropped_invalid_shape: int
    compression_plural_used: int
    compression_singular_used: int


class UnifiedResearchOrchestrator:
    """Unified Research Orchestrator combining all features"""

    def __init__(self):
        self.search_manager = None
        self.deduplicator = ResultDeduplicator()
        self.cost_monitor = CostMonitor()
        self.early_filter = EarlyRelevanceFilter()
        self.brave_enabled = False
        self.credibility_enabled = True
        self.research_store = research_store  # Add research store for cancellation checks

        # Orchestration additions
        self.tool_registry = ToolRegistry()
        self.retry_policy = RetryPolicy()
        self.planner = BudgetAwarePlanner(self.tool_registry, self.retry_policy)
        # Enforce that LLM-based synthesis must be available. If set to 1/true,
        # any failure to generate an answer will surface as a job failure rather
        # than silently degrading to evidence-only output. Default: on.
        try:
            self.require_synthesis = os.getenv("LLM_REQUIRED", "1").lower() in {"1", "true", "yes", "on"}
        except Exception:
            self.require_synthesis = True
        # Agentic loop configuration (can be overridden by env)
        self.agentic_config = {
            "enabled": True,
            # Enable LLM critic by default (can be disabled via env)
            "enable_llm_critic": True,
            "max_iterations": 2,
            "coverage_threshold": 0.75,
            "max_new_queries_per_iter": 4,
        }
        try:
            if os.getenv("AGENTIC_ENABLE_LLM_CRITIC", "0") == "1":
                self.agentic_config["enable_llm_critic"] = True
            if os.getenv("AGENTIC_ENABLE_LLM_CRITIC", "0") == "0" and "AGENTIC_ENABLE_LLM_CRITIC" in os.environ:
                self.agentic_config["enable_llm_critic"] = False
            if os.getenv("AGENTIC_DISABLE", "0") == "1":
                self.agentic_config["enabled"] = False
            # Optional tuning knobs
            if os.getenv("AGENTIC_MAX_ITERATIONS"):
                try:
                    self.agentic_config["max_iterations"] = int(os.getenv("AGENTIC_MAX_ITERATIONS", "2") or 2)
                except Exception:
                    pass
            if os.getenv("AGENTIC_COVERAGE_THRESHOLD"):
                try:
                    self.agentic_config["coverage_threshold"] = float(os.getenv("AGENTIC_COVERAGE_THRESHOLD", "0.75") or 0.75)
                except Exception:
                    pass
            if os.getenv("AGENTIC_MAX_NEW_QUERIES_PER_ITER"):
                try:
                    self.agentic_config["max_new_queries_per_iter"] = int(os.getenv("AGENTIC_MAX_NEW_QUERIES_PER_ITER", "4") or 4)
                except Exception:
                    pass
        except Exception:
            pass

        # Metrics
        self._search_metrics: SearchMetrics = {
            "total_queries": 0,
            "total_results": 0,
            "apis_used": [],
            "deduplication_rate": 0.0,
            # diagnostics
            "retries_attempted": 0,
            "task_timeouts": 0,
            "exceptions_by_api": {},
            "api_call_counts": {},
            "dropped_no_url": 0,
            "dropped_invalid_shape": 0,
            "compression_plural_used": 0,
            "compression_singular_used": 0,
        }
        self._diag_samples = {"no_url": []}
        self._supports_search_with_api = False
        # Capture deep research citations per research_id to merge into EvidenceBundle later
        self._deep_citations_map: Dict[str, List[Any]] = {}

        # Performance tracking (cap history to prevent unbounded memory growth)
        self.execution_history = deque(maxlen=100)

        # Diagnostics toggles
        self.diagnostics = {
            "log_task_creation": True,
            "log_result_normalization": True,
            "log_credibility_failures": True,
            "enforce_url_presence": True,
            "enforce_per_result_origin": True
        }

        # Per-search task timeout (seconds) for external API calls.
        # Ensure this exceeds the per-provider timeout so we don't cancel
        # the overall search while individual providers are still running.
        try:
            task_to = float(os.getenv("SEARCH_TASK_TIMEOUT_SEC", "30") or 30)
        except Exception:
            task_to = 30.0
        try:
            prov_to = float(os.getenv("SEARCH_PROVIDER_TIMEOUT_SEC", "25") or 25)
        except Exception:
            prov_to = 25.0
        # Add a small cushion above provider timeout
        self.search_task_timeout = max(task_to, prov_to + 5.0)

    # ──────────────────────────────────────────────────────────────────────
    # Small error/logging helper (applied in a few hot spots)
    # ──────────────────────────────────────────────────────────────────────
    def _log_and_continue(
        self,
        message: str,
        exc: Optional[Exception] = None,
        level: str = "warning",
    ) -> None:
        try:
            if exc:
                getattr(logger, level, logger.warning)(f"{message}: {exc}")
            else:
                getattr(logger, level, logger.warning)(message)
        except Exception:
            # Never raise from logging – absolute last resort
            logger.warning(message)

    # ──────────────────────────────────────────────────────────────────────
    # Metric coercion helpers
    # Ensure safe, typed extraction from heterogeneous _search_metrics dict
    # ──────────────────────────────────────────────────────────────────────
    def _safe_metric_int(self, key: str, default: int = 0) -> int:
        """Safely coerce metric to int for type checkers and runtime safety."""
        try:
            val = self._search_metrics.get(key, default)
            if isinstance(val, bool):
                return int(val)
            if isinstance(val, (int, float)):
                return int(val)
            if isinstance(val, str):
                sval = val.strip()
                if sval:
                    # Allow floats encoded as strings
                    return int(float(sval))
            return default
        except Exception:
            return default

    def _safe_metric_float(self, key: str, default: float = 0.0) -> float:
        """Safely coerce metric to float for type checkers and runtime safety."""
        try:
            val = self._search_metrics.get(key, default)
            if isinstance(val, bool):
                return float(val)
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                sval = val.strip()
                if sval:
                    return float(sval)
            return default
        except Exception:
            return default

    async def initialize(self):
        """Initialize the orchestrator"""
        self.search_manager = create_search_manager()
        # SearchAPIManager uses context manager protocol
        await self.search_manager.__aenter__()
        # Fail fast on empty provider set to avoid silent no-op searches
        try:
            if not getattr(self.search_manager, "apis", {}):
                error_msg = (
                    "CRITICAL: No search providers configured - Research pipeline initialization failed. "
                    "Required API keys are missing or invalid. "
                    "Please configure at least one of: "
                    "1) BRAVE_SEARCH_API_KEY for Brave Search "
                    "2) GOOGLE_CSE_API_KEY + GOOGLE_CSE_CX for Google CSE "
                    "3) Academic providers (ArXiv, PubMed, Semantic Scholar). "
                    "You can disable this check by setting SEARCH_DISABLE_FAILFAST=1 if using deep-research-only mode."
                )
                logger.critical(error_msg)
                raise RuntimeError(error_msg)
        except RuntimeError:
            # Allow opt-out via env (useful for deep-research-only scenarios)
            import os as _os
            if _os.getenv("SEARCH_DISABLE_FAILFAST", "0") not in {"1", "true", "yes"}:
                logger.critical("Search provider initialization failed - exiting. Set SEARCH_DISABLE_FAILFAST=1 to bypass.")
                raise
        await cache_manager.initialize()
        await initialize_deep_research()
        # SearchAPIManager only supports search_all, not per-API search
        self._supports_search_with_api = False

        # Try to initialize Brave MCP
        try:
            from services.brave_mcp_integration import initialize_brave_mcp
            self.brave_enabled = await initialize_brave_mcp()
        except Exception as e:
            logger.warning(f"Brave MCP initialization failed: {e}")

        logger.info("✓ Unified Research Orchestrator V2 initialized")
        # Register default tool capabilities for budget tracking
        try:
            self.tool_registry.register(ToolCapability(name="google", cost_per_call_usd=0.005, rpm_limit=100, typical_latency_ms=800))
            self.tool_registry.register(ToolCapability(name="brave", cost_per_call_usd=0.0, rpm_limit=100, typical_latency_ms=600))
        except Exception:
            pass

    async def cleanup(self):
        """Cleanup resources"""
        if self.search_manager:
            # SearchAPIManager uses context manager protocol
            await self.search_manager.__aexit__(None, None, None)
            logger.info("Search manager cleaned up")

    def get_capabilities(self) -> Dict[str, Any]:
        """Get current orchestrator capabilities"""
        return {
            "version": "2.0",
            "features": {
                # Multiple providers + concurrency imply non-determinism in ordering
                # and provider availability; report this accurately.
                "deterministic_results": False,
                "origin_tracking": True,
                "dynamic_compression": True,
                "full_context_preservation": True,
                "user_context_aware": True,
                "paradigm_optimization": True,
                "brave_mcp": self.brave_enabled,
                "deep_research": True,
                "early_relevance_filtering": True,
                "cost_monitoring": True
            }
        }


# Backwards-compatibility alias for legacy imports/tests
class ResearchOrchestrator(UnifiedResearchOrchestrator):
    pass

    async def execute_research(
        self,
        classification: ClassificationResultSchema,
        context_engineered: ContextEngineeredQuerySchema,
        user_context: Any,
        progress_callback: Optional[Any] = None,
        research_id: Optional[str] = None,
        enable_deep_research: bool = False,
        deep_research_mode: Optional[DeepResearchMode] = None,
        synthesize_answer: bool = False,
        answer_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute research (pure V2 flow; legacy path removed)"""

        async def check_cancelled():
            if not research_id:
                return False
            research_data = await self.research_store.get(research_id)
            return bool(research_data and research_data.get("status") == RuntimeResearchStatus.CANCELLED)

        start_time = datetime.now()
        # Use request‑scoped holders to avoid cross‑talk under concurrency
        cost_breakdown: Dict[str, float] = {}
        search_metrics_local: SearchMetrics = {
            "total_queries": 0,
            "total_results": 0,
            "apis_used": [],
            "deduplication_rate": 0.0,
            "retries_attempted": 0,
            "task_timeouts": 0,
            "exceptions_by_api": {},
            "api_call_counts": {},
            "dropped_no_url": 0,
            "dropped_invalid_shape": 0,
            "compression_plural_used": 0,
            "compression_singular_used": 0,
        }

        # Pure V2 query planning
        limited = self._get_query_limit(user_context)
        source_queries = list(getattr(context_engineered, "refined_queries", []) or [])
        if not source_queries:
            source_queries = [getattr(context_engineered, "original_query", "")]
        try:
            optimized_all = self._compress_and_dedup_queries(source_queries)
            optimized_queries = self._prioritize_queries(
                optimized_all,
                limited,
                getattr(classification, "primary_paradigm", None),
            )
        except Exception as e:
            logger.warning(f"Query compression failed; using fallback. {e}")
            optimized_queries = [getattr(context_engineered, "original_query", "")][:limited]

        logger.info(
            f"Optimized queries to {len(optimized_queries)} for {classification.primary_paradigm.value}"
        )

        # Update metrics for total queries
        # Update request-scoped metrics
        search_metrics_local["total_queries"] = int(search_metrics_local.get("total_queries", 0)) + len(optimized_queries)

        # Check for cancellation before executing searches
        if await check_cancelled():
            return {"status": "cancelled", "message": "Research was cancelled"}

        # Fail fast if search manager is missing or empty (misconfiguration)
        try:
            mgr = getattr(self, "search_manager", None)
            if not mgr or not getattr(mgr, "apis", {}):
                if progress_callback and research_id:
                    try:
                        await progress_callback.update_progress(
                            research_id,
                            phase="initialization",
                            message="No search providers available. Check API keys and provider flags.",
                            custom_data={"event": "no_search_providers"},
                        )
                    except Exception:
                        pass
                return {
                    "status": "error",
                    "message": "No search providers configured. Set BRAVE_SEARCH_API_KEY and/or GOOGLE_CSE_*; enable academic providers as needed.",
                }
        except Exception:
            pass

        # Execute searches with deterministic ordering
        search_results = await self._execute_searches_deterministic(
            optimized_queries,
            classification.primary_paradigm,
            user_context,
            progress_callback,
            research_id,
            check_cancelled,  # Pass cancellation check function
            cost_accumulator=cost_breakdown,
            metrics=search_metrics_local,
        )

        # Check for cancellation before processing results
        if await check_cancelled():
            return {"status": "cancelled", "message": "Research was cancelled"}

        # Process and enrich results
        processed_results = await self._process_results(
            search_results,
            classification,
            context_engineered,
            user_context,
            progress_callback,
            research_id,
            metrics=search_metrics_local,
        )

        # Check for cancellation before deep research
        if await check_cancelled():
            return {"status": "cancelled", "message": "Research was cancelled"}

        # Execute deep research if enabled
        if enable_deep_research:
            # Temporarily allow unlinked citations for deep research path
            prev_allow_unlinked = bool(self.diagnostics.get("allow_unlinked_citations", False))
            self.diagnostics["allow_unlinked_citations"] = True
            try:
                deep_results = await self._execute_deep_research_integration(
                    context_engineered,
                    classification,
                    user_context,
                    deep_research_mode or DeepResearchMode.PARADIGM_FOCUSED,
                    progress_callback,
                    research_id
                )
            finally:
                # Restore previous setting
                self.diagnostics["allow_unlinked_citations"] = prev_allow_unlinked

            if deep_results:
                # Avoid duplicates by URL against ranked results before appending deep results
                seen_urls = {
                    getattr(r, "url", None)
                    for r in processed_results.get("results", [])
                    if getattr(r, "url", None)
                }
                unique_deep: List[dict] = []
                for d in deep_results:
                    try:
                        u = (d.get("url") or "").strip()
                    except Exception:
                        u = ""
                    if u and u not in seen_urls:
                        unique_deep.append(d)
                processed_results["results"].extend(unique_deep)
                processed_results["metadata"]["deep_research_enabled"] = True
                # Update apis_used/sources_used to reflect deep results appended post‑processing
                try:
                    apis = set(processed_results.get("metadata", {}).get("apis_used", []) or [])
                except Exception:
                    apis = set()
                for d in deep_results:
                    try:
                        api = getattr(d, "source", None) or getattr(d, "source_api", None) or getattr(d, "search_api", None) or "deep_research"
                        if api:
                            apis.add(str(api))
                    except Exception:
                        continue
                if apis:
                    processed_results.setdefault("metadata", {})["apis_used"] = list(apis)
                try:
                    used = set(processed_results.get("sources_used", []) or [])
                except Exception:
                    used = set()
                for d in deep_results:
                    try:
                        used.add(getattr(d, "source", None) or getattr(d, "source_api", None) or getattr(d, "search_api", None) or "deep_research")
                    except Exception:
                        continue
                processed_results["sources_used"] = list(used)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        processed_results["metadata"]["processing_time"] = processing_time
        processed_results["metadata"]["processing_time_seconds"] = float(processing_time)

        # Compute UI-facing metrics that depend on the final result set
        credibility_summary = processed_results.setdefault("credibility_summary", {"average_score": 0.0})
        score_distribution = {"high": 0, "medium": 0, "low": 0}
        category_distribution: Dict[str, int] = {}
        total_sources_analyzed = 0
        high_quality_sources = 0

        def _score_band(value: Any) -> str:
            try:
                score = float(value if value is not None else 0.0)
            except Exception:
                score = 0.0
            if score >= 0.8:
                return "high"
            if score >= 0.6:
                return "medium"
            return "low"

        final_results = list(processed_results.get("results", []) or [])
        for result in final_results:
            adapter = ResultAdapter(result)
            score = adapter.credibility_score
            if score is None:
                try:
                    score = float(getattr(result, "credibility_score", 0.0) or 0.0)
                except Exception:
                    score = 0.0
            score = max(0.0, min(1.0, float(score or 0.0)))
            band = _score_band(score)
            score_distribution[band] += 1
            if score >= 0.8:
                high_quality_sources += 1
            total_sources_analyzed += 1

            raw_data = getattr(result, "raw_data", {}) or {}
            source_category = None
            if isinstance(raw_data, dict):
                source_category = raw_data.get("source_category")
            metadata = adapter.metadata or {}
            if not source_category and isinstance(metadata, dict):
                source_category = metadata.get("source_category")
            source_category = (str(source_category).strip().lower() or "general") if source_category else "general"
            category_distribution[source_category] = category_distribution.get(source_category, 0) + 1

        credibility_summary["score_distribution"] = score_distribution
        if total_sources_analyzed:
            credibility_summary["high_credibility_count"] = score_distribution["high"]
            credibility_summary["high_credibility_ratio"] = score_distribution["high"] / float(total_sources_analyzed)

        processed_results["metadata"]["total_sources_analyzed"] = total_sources_analyzed
        processed_results["metadata"]["high_quality_sources"] = high_quality_sources
        processed_results["metadata"]["category_distribution"] = category_distribution
        processed_results["metadata"].setdefault("bias_distribution", {})
        processed_results["metadata"].setdefault("agent_trace", [])

        normalized_sources: List[Dict[str, Any]] = []
        try:
            for result in final_results:
                adapter = ResultAdapter(result)
                url = (adapter.url or "").strip()
                if not url:
                    continue
                metadata = adapter.metadata or {}
                if not isinstance(metadata, dict):
                    metadata = {}
                raw_data = getattr(result, "raw_data", {}) or {}
                if isinstance(raw_data, dict):
                    extracted = raw_data.get("extracted_meta")
                    if extracted and "extracted_meta" not in metadata:
                        metadata["extracted_meta"] = extracted
                    if "source_category" in raw_data and "source_category" not in metadata:
                        metadata["source_category"] = raw_data["source_category"]
                    if "credibility_explanation" in raw_data and "credibility_explanation" not in metadata:
                        metadata["credibility_explanation"] = raw_data["credibility_explanation"]
                credibility = adapter.credibility_score
                if credibility is None:
                    try:
                        credibility = float(metadata.get("credibility_score", 0.0) or 0.0)
                    except Exception:
                        credibility = 0.0
                credibility = max(0.0, min(1.0, float(credibility or 0.0)))
                published_date = metadata.get("published_date")
                if published_date is None:
                    published_date = getattr(result, "published_date", None)
                # Ensure JSON-serializable date
                try:
                    from datetime import date, datetime as _dt
                    if isinstance(published_date, (_dt, date)):
                        published_date = published_date.isoformat()
                except Exception:
                    pass
                result_type = metadata.get("result_type") or getattr(result, "result_type", "web")
                normalized_sources.append(
                    {
                        "title": adapter.title,
                        "url": url,
                        "snippet": adapter.snippet,
                        "content": adapter.content,
                        "domain": adapter.domain,
                        "credibility_score": credibility,
                        "published_date": published_date,
                        "result_type": result_type or "web",
                        "source_api": adapter.source_api,
                        "source_category": metadata.get("source_category"),
                        "metadata": metadata,
                    }
                )
        except Exception:
            normalized_sources = []

        # Optional answer synthesis (P2)
        synthesized_answer = None
        if synthesize_answer:
            try:
                # Build evidence quotes from processed results (typed EvidenceQuote)
                evidence_quotes: List[Any] = []
                evidence_bundle = None
                try:
                    from services.evidence_builder import build_evidence_bundle, quotes_to_plain_dicts

                    # Check if we have any sources to build evidence from
                    if not normalized_sources:
                        logger.warning(
                            f"Evidence builder skipped - No search results to process. "
                            f"Research ID: {research_id or 'N/A'}"
                        )
                        evidence_quotes = []
                        evidence_bundle = None
                        evidence_quotes_payload = []
                    else:
                        max_docs = min(
                            EVIDENCE_MAX_DOCS_DEFAULT,
                            int(getattr(user_context, "source_limit", default_source_limit())),
                        )

                        logger.info(
                            f"Building evidence bundle from {len(normalized_sources)} sources, "
                            f"max_docs={max_docs}, quotes_per_doc={EVIDENCE_QUOTES_PER_DOC_DEFAULT}"
                        )

                        evidence_bundle = await build_evidence_bundle(
                            getattr(context_engineered, "original_query", ""),
                            normalized_sources,
                            max_docs=max_docs,
                            quotes_per_doc=EVIDENCE_QUOTES_PER_DOC_DEFAULT,
                            include_full_content=True,
                            full_text_budget=EVIDENCE_BUDGET_TOKENS_DEFAULT,
                        )

                        evidence_quotes = list(getattr(evidence_bundle, "quotes", []) or [])
                        evidence_quotes_payload = quotes_to_plain_dicts(evidence_quotes) if evidence_quotes else []

                        # Log warning if evidence building produced no quotes
                        if not evidence_quotes and normalized_sources:
                            logger.warning(
                                f"Evidence builder produced no quotes from {len(normalized_sources)} sources. "
                                f"Possible causes: "
                                f"1) URL fetching failures (network issues, timeouts) "
                                f"2) Content extraction failures (JavaScript-rendered pages, PDFs) "
                                f"3) No relevant quotes found in fetched content. "
                                f"Research ID: {research_id or 'N/A'}"
                            )
                        else:
                            logger.info(
                                f"Evidence builder succeeded: {len(evidence_quotes)} quotes from "
                                f"{len(getattr(evidence_bundle, 'documents', []))} documents"
                            )

                        # Signal evidence phase completion to UI
                        if progress_callback and research_id:
                            try:
                                await progress_callback.update_progress(
                                    research_id,
                                    phase="evidence",
                                    progress=75,
                                )
                            except Exception:
                                pass

                except ImportError as e:
                    error_msg = (
                        f"CRITICAL: Evidence builder import failed - Cannot build evidence bundle. "
                        f"Error: {str(e)}. "
                        f"Missing dependencies or module not found. "
                        f"Research ID: {research_id or 'N/A'}"
                    )
                    logger.critical(error_msg)
                    evidence_quotes = []
                    evidence_bundle = None
                    evidence_quotes_payload = []
                    # Don't raise - allow synthesis to proceed without evidence

                except Exception as e:
                    error_msg = (
                        f"ERROR: Evidence builder failed - Evidence extraction unsuccessful. "
                        f"Error: {str(e)}. "
                        f"Possible causes: "
                        f"1) Network issues fetching URLs "
                        f"2) Invalid URL formats in search results "
                        f"3) Memory issues with large documents "
                        f"4) Timeout during content fetching. "
                        f"Research ID: {research_id or 'N/A'}, "
                        f"Sources attempted: {len(normalized_sources)}"
                    )
                    logger.error(error_msg)
                    logger.debug("Evidence builder traceback:", exc_info=True)
                    evidence_quotes = []
                    evidence_bundle = None
                    evidence_quotes_payload = []
                    # Don't raise - allow synthesis to proceed without evidence
                # Respect cancellation just before heavy LLM synthesis
                try:
                    if await check_cancelled():
                        return {"status": "cancelled", "message": "Research was cancelled"}
                except Exception:
                    pass

                # Attempt synthesis even if evidence building failed
                if evidence_bundle is None and processed_results.get("results"):
                    logger.warning(
                        f"Proceeding with answer synthesis without evidence bundle. "
                        f"Using {len(processed_results.get('results', []))} raw search results. "
                        f"Research ID: {research_id or 'N/A'}"
                    )

                synthesized_answer = await self._synthesize_answer(
                    classification=classification,
                    context_engineered=context_engineered,
                    results=processed_results["results"],
                    research_id=research_id or f"research_{int(start_time.timestamp())}",
                    options=answer_options or {},
                    evidence_quotes=evidence_quotes_payload,
                    evidence_bundle=evidence_bundle,  # Can be None - synthesis should handle
                )
                # Honour cancellation immediately after synthesis returns
                try:
                    if await check_cancelled():
                        return {"status": "cancelled", "message": "Research was cancelled"}
                except Exception:
                    pass
                # Attach contradiction summary to answer metadata if available
                try:
                    contradictions = processed_results.get("contradictions", {})
                    if hasattr(synthesized_answer, "metadata") and isinstance(getattr(synthesized_answer, "metadata"), dict):
                        synthesized_answer.metadata.setdefault("signals", {})
                        synthesized_answer.metadata["signals"]["contradictions"] = contradictions
                except Exception:
                    pass
                try:
                    if evidence_bundle is not None:
                        metrics.record_o3_usage(
                            paradigm=normalize_to_internal_code(classification.primary_paradigm),
                            document_count=len(getattr(evidence_bundle, "documents", []) or []),
                            document_tokens=int(getattr(evidence_bundle, "documents_token_count", 0) or 0),
                            quote_count=len(evidence_quotes or []),
                            source_count=len(normalized_sources),
                            prompt_tokens=None,
                            completion_tokens=None,
                        )
                except Exception:
                    pass
            except asyncio.CancelledError:
                # Preserve cancellation semantics; do not misreport as LLM failure
                raise
            except Exception as e:
                error_msg = (
                    f"CRITICAL: LLM Answer synthesis failed - Cannot generate response. "
                    f"Error: {str(e)}. "
                    f"This typically indicates Azure OpenAI issues: "
                    f"1) Missing/invalid AZURE_OPENAI_API_KEY "
                    f"2) Incorrect AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_DEPLOYMENT "
                    f"3) Azure OpenAI service down/quota exceeded. "
                    f"Research ID: {research_id or 'N/A'}, Query: '{context_engineered.original_query[:100]}...' "
                    f"Search results collected: {len(normalized_sources)} sources"
                )
                logger.critical(error_msg)
                # If synthesis is required, escalate to hard failure so the
                # request never returns a partial result without an LLM answer.
                if getattr(self, "require_synthesis", True):
                    logger.critical(
                        f"Synthesis is required (require_synthesis=True) - Discarding {len(normalized_sources)} search results. "
                        f"Set LLM_REQUIRED=false to allow partial results without synthesis."
                    )
                    raise RuntimeError(error_msg) from e
                else:
                    logger.warning(
                        f"Continuing without synthesis (require_synthesis=False) - Returning {len(normalized_sources)} search results only."
                    )

        # Assemble contract-aligned response payload
        primary_paradigm = getattr(classification, "primary_paradigm", HostParadigm.BERNARD)
        primary_code = normalize_to_internal_code(primary_paradigm)
        secondary_paradigm = getattr(classification, "secondary_paradigm", None)
        secondary_code = normalize_to_internal_code(secondary_paradigm) if secondary_paradigm else None

        distribution_map: Dict[str, float] = {}
        secondary_confidence: Optional[float] = None
        raw_distribution = getattr(classification, "distribution", {}) or {}
        try:
            for host, value in raw_distribution.items():
                key = normalize_to_internal_code(host) if host is not None else str(host)
                distribution_map[key] = float(value or 0.0)
            if secondary_code and secondary_code in distribution_map:
                secondary_confidence = float(distribution_map.get(secondary_code, 0.0))
        except Exception:
            distribution_map = {}

        margin = 0.0
        if distribution_map:
            sorted_scores = sorted(distribution_map.values(), reverse=True)
            margin = float(sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else 0.0))

        paradigm_analysis: Dict[str, Any] = {
            "primary": {
                "paradigm": primary_code,
                "confidence": float(getattr(classification, "confidence", 0.0) or 0.0),
            }
        }
        if secondary_code:
            secondary_payload: Dict[str, Any] = {"paradigm": secondary_code}
            if secondary_confidence is not None:
                secondary_payload["confidence"] = secondary_confidence
            paradigm_analysis["secondary"] = secondary_payload

        integrated_synthesis: Optional[Dict[str, Any]] = None
        if synthesized_answer is not None:
            try:
                integrated_synthesis = {
                    "primary_answer": synthesized_answer,
                    "integrated_summary": getattr(synthesized_answer, "summary", ""),
                    "synergies": [],
                    "conflicts_identified": [],
                }
            except Exception:
                integrated_synthesis = None

        apis_used = list(search_metrics_local.get("apis_used", []))

        response = {
            "research_id": research_id or f"research_{int(start_time.timestamp())}",
            "query": getattr(context_engineered, "original_query", classification.query),
            "status": "ok",
            "results": processed_results["results"],
            "sources": normalized_sources,
            "paradigm_analysis": paradigm_analysis,
            "integrated_synthesis": integrated_synthesis,
            "metadata": {
                **processed_results["metadata"],
                "total_results": len(processed_results["results"]),
                "queries_executed": len(optimized_queries),
                "sources_used": processed_results["sources_used"],
                "credibility_summary": processed_results["credibility_summary"],
                "deduplication_stats": processed_results["dedup_stats"],
                "contradictions": processed_results.get("contradictions", {"count": 0, "examples": []}),
                "processing_time_seconds": float(processing_time),
                "paradigm_fit": {
                    "primary": primary_code,
                    "confidence": float(getattr(classification, "confidence", 0.0) or 0.0),
                    "margin": margin,
                },
                "research_depth": "deep" if enable_deep_research else "standard",
                # Prepare safe metric values for type checking
                # Report request‑scoped metrics, not global cumulative values
                "search_metrics": {
                    "total_queries": int(search_metrics_local.get("total_queries", 0)),
                    "total_results": int(search_metrics_local.get("total_results", 0)),
                    "apis_used": apis_used,
                    "deduplication_rate": float(search_metrics_local.get("deduplication_rate", 0.0)),
                },
                "paradigm": primary_code,
                # Record LLM backend info so callers can verify model/endpoint used
                "llm_backend": self._llm_backend_info(),
            },
            "export_formats": {},
        }

        # Ensure "classification_details" is properly typed using ClassificationDetailsSchema
        try:
            md_obj = response.get("metadata")
            if not isinstance(md_obj, dict):
                md_obj = {}
                response["metadata"] = md_obj

            # Build ClassificationDetailsSchema for type safety
            from models.context_models import ClassificationDetailsSchema

            # Build distribution data with explicit typing
            distribution_data: Dict[str, float] = {}
            if distribution_map:
                for key, value in distribution_map.items():
                    distribution_data[key] = float(value)

            # Build reasoning data properly
            reasoning_data: Dict[str, List[str]] = {}
            reasoning_raw = getattr(classification, "reasoning", {}) or {}
            if isinstance(reasoning_raw, dict):
                for host, steps in reasoning_raw.items():
                    normalized_key = normalize_to_internal_code(host)
                    if steps:
                        steps_list = [str(s) for s in (list(steps) or [])[:4]]
                    else:
                        steps_list = []
                    reasoning_data[normalized_key] = steps_list

            # Create the schema object and convert to dict for storage
            classification_details = ClassificationDetailsSchema(
                distribution=distribution_data,
                reasoning=reasoning_data
            )
            md_obj["classification_details"] = classification_details.model_dump()
        except Exception:
            # Best-effort enrichment; skip on any unexpected shape issues
            pass

        try:
            rid = response["research_id"]
            response["export_formats"] = {
                "json": f"/v1/research/{rid}/export/json",
                "csv": f"/v1/research/{rid}/export/csv",
                "pdf": f"/v1/research/{rid}/export/pdf",
                "markdown": f"/v1/research/{rid}/export/markdown",
                "excel": f"/v1/research/{rid}/export/excel",
            }
        except Exception:
            response["export_formats"] = {}
        # Cost info (search API costs currently tracked; LLM costs not yet integrated)
        try:
            search_cost = sum(float(v) for v in (cost_breakdown or {}).values())
            response["cost_info"] = {
                "search_api_costs": round(search_cost, 4),
                "llm_costs": 0.0,
                "total": round(search_cost, 4),
                # keep backward compat with callers referencing total_cost
                "total_cost": round(search_cost, 4),
                "breakdown": cost_breakdown,
            }
        except Exception:
            response["cost_info"] = {"search_api_costs": 0.0, "llm_costs": 0.0, "total": 0.0, "total_cost": 0.0}
        if synthesized_answer is not None:
            # Preserve object for downstream normalizers (routes/research.py)
            response["answer"] = synthesized_answer
            try:
                metadata_obj = getattr(synthesized_answer, "metadata", {}) or {}
                evidence_quotes_copy = metadata_obj.get("evidence_quotes")
                if evidence_quotes_copy:
                    response["metadata"].setdefault("evidence_quotes", evidence_quotes_copy)
            except Exception:
                pass
        # Append execution record for stats (best-effort)
        try:
            # Build a minimal per-execution record
            cred_map: Dict[str, float] = {}
            try:
                for r in processed_results.get("results", []) or []:
                    dom = getattr(r, "domain", None)
                    sc = getattr(r, "credibility_score", None)
                    if dom and isinstance(sc, (int, float)):
                        cred_map[str(dom)] = float(sc)
            except Exception:
                cred_map = {}
            exec_rec = ResearchExecutionResult(
                original_query=getattr(context_engineered, "original_query", ""),
                paradigm=primary_code,
                secondary_paradigm=secondary_code,
                search_queries_executed=[{"query": q} for q in (optimized_queries or [])],
                raw_results=search_results if isinstance(search_results, dict) else {},
                filtered_results=processed_results.get("results", []) or [],
                credibility_scores=cred_map,
                execution_metrics={
                    "processing_time_seconds": float(processing_time),
                    "final_results_count": len(processed_results.get("results", []) or []),
                    "queries_executed": len(optimized_queries),
                },
                cost_breakdown=cost_breakdown,
                status=ContractResearchStatus.OK,
            )
            self.execution_history.append(exec_rec)
        except Exception:
            pass
        return response

    def _llm_backend_info(self) -> Dict[str, Any]:
        """Provide a small diagnostic snapshot of the active LLM backend."""
        try:
            from services.llm_client import llm_client
            return llm_client.get_active_backend_info()
        except Exception:
            return {"azure_enabled": False, "openai_enabled": False}

    async def _synthesize_answer(
        self,
        *,
        classification: ClassificationResultSchema,
        context_engineered: ContextEngineeredQuerySchema,
        results: List[SearchResult],
        research_id: str,
        options: Dict[str, Any],
        evidence_quotes: Optional[List[Dict[str, Any]]] = None,
        evidence_bundle: Any | None = None,
    ) -> Any:
        """Build a SynthesisContext from research outputs and invoke answer generator."""
        if not _ANSWER_GEN_AVAILABLE:
            error_msg = (
                "CRITICAL: Answer generation module not available - Cannot synthesize response. "
                "This indicates a system configuration issue: "
                "1) Missing answer_generator module or dependencies "
                "2) Import failure during module initialization "
                "3) Azure OpenAI client initialization failed. "
                f"Research ID: {research_id}, Query: '{context_engineered.original_query[:100]}...' "
                f"Available search results: {len(results)} that will be discarded"
            )
            logger.critical(error_msg)
            raise RuntimeError(error_msg)

        # Prepare minimal list[dict] for generator consumption.  We normalize all
        # result objects through ResultAdapter so downstream helpers receive a
        # consistent, dict-friendly payload (the evidence builder expects
        # `.get(...)` access and previously crashed when handed dataclasses).
        normalized_sources: List[Dict[str, Any]] = []
        for result in results or []:
            try:
                adapter = ResultAdapter(result)
                url = (adapter.url or "").strip()
                if not url:
                    continue

                metadata: Dict[str, Any] = adapter.metadata or {}
                if not isinstance(metadata, dict):
                    metadata = {}

                is_dict_result = isinstance(result, dict)

                if not is_dict_result:
                    try:
                        raw_data = getattr(result, "raw_data", {}) or {}
                        if isinstance(raw_data, dict):
                            extracted = raw_data.get("extracted_meta")
                            if extracted and "extracted_meta" not in metadata:
                                metadata["extracted_meta"] = extracted
                            for key in ("source_category", "credibility_explanation"):
                                if key in raw_data and key not in metadata:
                                    metadata[key] = raw_data[key]
                    except Exception:
                        pass

                credibility = adapter.credibility_score
                if credibility is None:
                    try:
                        credibility = float(metadata.get("credibility_score", 0.0) or 0.0)
                    except Exception:
                        credibility = 0.0

                published_date = metadata.get("published_date")
                if published_date is None and not is_dict_result:
                    published_date = getattr(result, "published_date", None)
                # Ensure JSON-serializable date
                try:
                    from datetime import date, datetime as _dt
                    if isinstance(published_date, (_dt, date)):
                        published_date = published_date.isoformat()
                except Exception:
                    pass

                result_type = metadata.get("result_type")
                if not result_type and not is_dict_result:
                    result_type = getattr(result, "result_type", "web")

                normalized_sources.append(
                    {
                        "title": adapter.title,
                        "url": url,
                        "snippet": adapter.snippet,
                        "content": adapter.content,
                        "domain": adapter.domain,
                        "credibility_score": float(credibility or 0.0),
                        "published_date": published_date,
                        "result_type": result_type or "web",
                        "source_api": adapter.source_api,
                        "metadata": metadata,
                    }
                )
            except Exception:
                logger.debug("[synthesis] Failed to normalize result", exc_info=True)
                continue

        sources: List[Dict[str, Any]] = normalized_sources

        # Best effort context_engineering dict
        try:
            ce = context_engineered.model_dump() if hasattr(context_engineered, "model_dump") else context_engineered.dict()
        except Exception:
            ce = {}

        # Isolation-only extraction: build findings strictly from Isolate layer patterns
        try:
            patterns = []
            iso = getattr(context_engineered, "isolate_output", None)
            if iso:
                raw = getattr(iso, "extraction_patterns", []) or []
                import re as _re
                for p in raw:
                    try:
                        patterns.append(_re.compile(p, _re.IGNORECASE))
                    except Exception:
                        continue
            findings: Dict[str, Any] = {"matches": [], "by_domain": {}, "focus_areas": getattr(getattr(context_engineered, "isolate_output", None), "focus_areas", [])}
            for r in sources:
                try:
                    text = " ".join([
                        (r.get("title") or ""),
                        (r.get("snippet") or ""),
                    ])
                    dom = (r.get("domain") or "").strip()
                except Exception:
                    continue
                matched = []
                for pat in patterns:
                    for m in pat.finditer(text):
                        frag = text[max(0, m.start()-80):m.end()+80]
                        matched.append(frag.strip())
                if matched:
                    findings["matches"].append({"domain": dom, "fragments": matched[:3]})
                    findings["by_domain"].setdefault(dom, 0)
                    findings["by_domain"][dom] += 1
            ce["isolated_findings"] = findings
            ce["isolation_only"] = True
        except Exception:
            ce.setdefault("isolation_only", True)

        # Normalize paradigm code for generator
        from models.paradigms import normalize_to_internal_code
        paradigm_code = normalize_to_internal_code(classification.primary_paradigm)

        # Predeclare evidence bundle holder for type-checkers
        eb = None  # will be optionally assigned below

        # Build synthesis context (typed EvidenceBundle)
        if SynthesisContextModel:
            try:
                from models.evidence import EvidenceBundle, EvidenceQuote, EvidenceMatch  # type: ignore
                # Normalize isolation findings into typed matches
                iso = (ce.get("isolated_findings", {}) or {}) if isinstance(ce, dict) else {}
                raw_matches = list(iso.get("matches", []) or [])

                def _ensure_match(val: Any) -> EvidenceMatch | None:
                    try:
                        if isinstance(val, EvidenceMatch):
                            return val
                        return EvidenceMatch.model_validate(val)
                    except Exception:
                        if isinstance(val, dict) and val.get("domain"):
                            fragments = val.get("fragments") or []
                            if isinstance(fragments, list):
                                return EvidenceMatch(domain=val.get("domain"), fragments=fragments)
                        return None

                matches_typed = []
                for m in raw_matches:
                    hm = _ensure_match(m)
                    if hm is not None:
                        matches_typed.append(hm)

                # Merge quotes from evidence_builder and optional deep research
                deep_q: List[Any] = []
                try:
                    if research_id and hasattr(self, "_deep_citations_map"):
                        deep_q = list(getattr(self, "_deep_citations_map", {}).get(research_id, []) or [])
                        # clear after use to avoid leaks across requests
                        if research_id in self._deep_citations_map:
                            self._deep_citations_map.pop(research_id, None)
                except Exception:
                    deep_q = []

                if evidence_bundle is not None:
                    try:
                        base_bundle = evidence_bundle if isinstance(evidence_bundle, EvidenceBundle) else EvidenceBundle.model_validate(evidence_bundle)
                    except Exception:
                        base_bundle = EvidenceBundle()
                else:
                    base_bundle = EvidenceBundle()

                existing_quotes = list(getattr(base_bundle, "quotes", []) or [])
                quotes_typed: List[EvidenceQuote] = []

                def _ensure_quote(val: Any) -> EvidenceQuote | None:
                    try:
                        if isinstance(val, EvidenceQuote):
                            return val
                        return EvidenceQuote.model_validate(val)
                    except Exception:
                        if isinstance(val, dict) and val.get("quote"):
                            return EvidenceQuote(
                                id=str(val.get("id", f"q{len(quotes_typed)+len(existing_quotes)+1:03d}")),
                                url=val.get("url", ""),
                                title=val.get("title", ""),
                                domain=val.get("domain", ""),
                                quote=val.get("quote", ""),
                                start=val.get("start"),
                                end=val.get("end"),
                                published_date=val.get("published_date"),
                                credibility_score=val.get("credibility_score"),
                                suspicious=val.get("suspicious"),
                                doc_summary=val.get("doc_summary"),
                                source_type=val.get("source_type"),
                            )
                        return None

                for item in existing_quotes:
                    q = _ensure_quote(item)
                    if q is not None:
                        quotes_typed.append(q)
                for item in evidence_quotes or []:
                    q = _ensure_quote(item)
                    if q is not None:
                        quotes_typed.append(q)
                for item in deep_q or []:
                    q = _ensure_quote(item)
                    if q is not None:
                        quotes_typed.append(q)

                dedup_quotes: Dict[str, EvidenceQuote] = {}
                for q in quotes_typed:
                    key = getattr(q, "id", None) or getattr(q, "url", None) or f"idx-{len(dedup_quotes)}"
                    key = str(key)
                    if key not in dedup_quotes:
                        dedup_quotes[key] = q
                quotes_typed = list(dedup_quotes.values())

                base_matches = list(getattr(base_bundle, "matches", []) or [])
                combined_matches: List[EvidenceMatch] = []
                for item in base_matches + matches_typed:
                    hm = _ensure_match(item)
                    if hm is not None:
                        combined_matches.append(hm)
                if combined_matches:
                    seen_match: set[tuple[str, tuple[str, ...]]] = set()
                    unique_matches: List[EvidenceMatch] = []
                    for m in combined_matches:
                        dom = getattr(m, "domain", "") or ""
                        fragments = tuple(getattr(m, "fragments", []) or [])
                        key = (dom, fragments)
                        if key in seen_match:
                            continue
                        seen_match.add(key)
                        unique_matches.append(m)
                    combined_matches = unique_matches

                base_by_domain = dict(getattr(base_bundle, "by_domain", {}) or {})
                for dom, count in (iso.get("by_domain", {}) or {}).items():
                    try:
                        base_by_domain[dom] = base_by_domain.get(dom, 0) + int(count)
                    except Exception:
                        continue

                focus_combined = list(dict.fromkeys((getattr(base_bundle, "focus_areas", []) or []) + list(iso.get("focus_areas", []) or [])))
                documents_combined = list(getattr(base_bundle, "documents", []) or [])
                documents_token_count = int(getattr(base_bundle, "documents_token_count", 0) or 0)

                eb = EvidenceBundle(
                    quotes=quotes_typed,
                    matches=combined_matches,
                    by_domain=base_by_domain,
                    focus_areas=focus_combined,
                    documents=documents_combined,
                    documents_token_count=documents_token_count,
                )
            except Exception:
                eb = None  # type: ignore
            # Validate only for diagnostics; guard to avoid crashing synthesis on validation errors
            try:
                _ = SynthesisContextModel(
                    query=context_engineered.original_query,
                    paradigm=paradigm_code,
                    search_results=sources,
                    context_engineering=ce,
                    max_length=int(options.get("max_length", SYNTHESIS_MAX_LENGTH_DEFAULT)),
                    include_citations=bool(options.get("include_citations", True)),
                    tone=str(options.get("tone", "professional")),
                    metadata={"research_id": research_id},
                    evidence_quotes=[],  # legacy field suppressed by unified bundle
                    evidence_bundle=eb,
                )
            except Exception:
                # Continue with untyped payload; downstream generator accepts dict-based context
                pass

        # Emit synthesis started (mirrored as research_progress by tracker)
        try:
            from services.progress import progress as _pt
            if _pt:
                await _pt.report_synthesis_started(research_id)
        except Exception:
            pass

        # Optionally pass prompt variant override from research record
        try:
            rec = await self.research_store.get(research_id)
            pv = ((rec or {}).get("experiment") or {}).get("prompt_variant")
            if isinstance(pv, str) and pv in {"v1", "v2"}:
                options = {**options, "prompt_variant": pv}
        except Exception:
            pass

        # Cooperative cancellation just before the long LLM call
        try:
            if research_id:
                rec = await self.research_store.get(research_id)
                if rec and rec.get("status") == RuntimeResearchStatus.CANCELLED:
                    raise asyncio.CancelledError()
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

        # Call the generator using the legacy signature for broad compatibility
        assert answer_orchestrator is not None, "Answer generation not available"
        answer = await answer_orchestrator.generate_answer(
            paradigm=paradigm_code,
            query=getattr(context_engineered, "original_query", ""),
            search_results=sources,
            context_engineering=ce,
            options={
                "research_id": research_id,
                **options,
                "evidence_quotes": evidence_quotes or [],
                "evidence_bundle": eb if 'eb' in locals() else None,
            },
        )

        # Cooperative cancellation checkpoint after LLM call returns
        try:
            if research_id:
                rec = await self.research_store.get(research_id)
                if rec and rec.get("status") == RuntimeResearchStatus.CANCELLED:
                    raise asyncio.CancelledError()
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

        # Attach evidence quotes and isolation findings summary to answer metadata
        # so the frontend can render dedicated panels without re-fetching.
        try:
            if hasattr(answer, "metadata") and isinstance(getattr(answer, "metadata"), dict):
                # Avoid accidental overwrites
                # Publish canonical bundle; legacy field retained only when absent
                if "evidence_bundle" not in answer.metadata and 'eb' in locals() and eb is not None:
                    try:
                        answer.metadata["evidence_bundle"] = eb.model_dump()
                    except Exception:
                        pass
                if "evidence_quotes" not in answer.metadata and (evidence_quotes or []):
                    # Temporary back-compat for frontends expecting raw quotes
                    try:
                        if all(isinstance(q, dict) for q in (evidence_quotes or [])):
                            answer.metadata["evidence_quotes"] = list(evidence_quotes)  # already plain
                        else:
                            from services.evidence_builder import quotes_to_plain_dicts
                            answer.metadata["evidence_quotes"] = quotes_to_plain_dicts(evidence_quotes)  # type: ignore[arg-type]
                    except Exception:
                        # Best-effort fallback: preserve dicts as-is, coerce objects via model_dump/__dict__
                        safe_quotes = []
                        for q in (evidence_quotes or []):
                            if isinstance(q, dict):
                                safe_quotes.append(q)
                            else:
                                try:
                                    safe_quotes.append(q.model_dump())  # type: ignore[attr-defined]
                                except Exception:
                                    safe_quotes.append(getattr(q, "__dict__", {}))
                        answer.metadata["evidence_quotes"] = safe_quotes
                # Summarize isolation findings if available in CE dict
                try:
                    if isinstance(ce, dict) and isinstance(ce.get("isolated_findings"), dict):
                        iso = ce.get("isolated_findings", {})
                        summary = {
                            "matches_count": len(iso.get("matches", []) or []),
                            "domains": list((iso.get("by_domain", {}) or {}).keys()),
                            "focus_areas": list(iso.get("focus_areas", []) or []),
                        }
                        answer.metadata.setdefault("signals", {})
                        answer.metadata["signals"]["isolated_findings"] = summary
                except Exception:
                    pass
        except Exception:
            pass

        # Emit synthesis completed with simple stats
        try:
            sections = 0
            citations = 0
            try:
                sections = len(getattr(answer, "sections", []) or [])
            except Exception:
                pass
            try:
                citations = len(getattr(answer, "citations", []) or [])
            except Exception:
                pass
            from services.progress import progress as _pt
            if _pt:
                await _pt.report_synthesis_completed(research_id, sections, citations)
        except Exception:
            pass

        return answer


    def _get_query_limit(self, user_context: Any) -> int:
        """Get query limit based on user context"""
        # Allow environment overrides per role, else use more generous defaults
        defaults = {"FREE": 5, "BASIC": 8, "PRO": 15, "ENTERPRISE": 25, "ADMIN": 40}
        role = getattr(user_context, "role", "PRO")
        try:
            env_key = f"QUERY_LIMIT_{role}"
            if env_key in os.environ:
                return int(os.getenv(env_key, str(defaults.get(role, 5))) or defaults.get(role, 5))
        except Exception:
            pass
        return defaults.get(role, 5)

    def _compress_and_dedup_queries(self, queries: List[str]) -> List[str]:
        optimized_list: List[str] = []
        seen_set: Set[str] = set()
        for q in queries or []:
            try:
                compressed = query_compressor.compress(q, preserve_keywords=True)
            except Exception:
                compressed = q
            val = (compressed or q or "").strip()
            if not val:
                continue
            if val not in seen_set:
                seen_set.add(val)
                optimized_list.append(val)
        return optimized_list

    def _prioritize_queries(self, queries: List[str], limit: int, paradigm: HostParadigm | None) -> List[str]:
        """Sort queries using simple heuristics; return top `limit` without arbitrary slice bias.

        Heuristics:
        - Prefer quoted phrases (exact intent)
        - Prefer moderate length (20–140 chars)
        - Paradigm nudges (Bernard: academic terms; Dolores: accountability keywords)
        """
        if not queries:
            return []
        def score(q: str) -> float:
            s = 0.0
            ql = len(q or "")
            if '"' in q:
                s += 2.0
            if 20 <= ql <= 140:
                s += 1.0
            low = (q or "").lower()
            try:
                code = normalize_to_internal_code(paradigm) if paradigm else ""
            except Exception:
                code = ""
            if code == "bernard":
                for kw in ("site:edu", "doi", "arxiv", "journal", "study", "evidence"):
                    if kw in low:
                        s += 0.5
            elif code == "dolores":
                for kw in ("systemic", "accountability", "investigation", "whistleblower", "lawsuit"):
                    if kw in low:
                        s += 0.5
            elif code == "maeve":
                for kw in ("roi", "market", "kpi", "growth", "benchmark"):
                    if kw in low:
                        s += 0.4
            elif code == "teddy":
                for kw in ("support", "resources", "mental health", "community"):
                    if kw in low:
                        s += 0.4
            # Penalise very long (>220) or very short (<10)
            if ql > 220:
                s -= 0.5
            if ql < 10:
                s -= 0.5
            return s
        ordered = sorted(queries, key=score, reverse=True)
        return ordered[: max(1, limit)]

    async def _execute_searches_deterministic(
        self,
        optimized_queries: List[str],
        primary_paradigm: HostParadigm,
        user_context: Any,
        progress_callback: Optional[Any],
        research_id: Optional[str],
        check_cancelled: Optional[Any],
        *,
        cost_accumulator: Optional[Dict[str, float]] = None,
        metrics: Optional[SearchMetrics] = None,
    ) -> Dict[str, List[SearchResult]]:
        """
        Deterministically execute searches for V2 path. Returns {query_key: [SearchResult, ...]}.
        """
        all_results: Dict[str, List[SearchResult]] = {}
        max_results = int(getattr(user_context, "source_limit", default_source_limit()) or default_source_limit())
        paradigm_code = normalize_to_internal_code(primary_paradigm)
        min_rel = bernard_min_relevance() if paradigm_code == "bernard" else 0.25

        # Run per-query searches concurrently with a cap to avoid API bursts
        concurrency = int(os.getenv("SEARCH_QUERY_CONCURRENCY", "4"))
        sem = asyncio.Semaphore(max(1, concurrency))

        async def _run_query(idx: int, q: str) -> Tuple[str, List[SearchResult]]:
            if check_cancelled and await check_cancelled():
                logger.info("Research cancelled before executing query #%d", idx + 1)
                return q, []
            # Progress (started)
            if progress_callback and research_id:
                try:
                    try:
                        await progress_callback.report_search_started(
                            research_id, q, "multi", idx + 1, len(optimized_queries)
                        )
                    except Exception:
                        pass
                    await progress_callback.update_progress(
                        research_id,
                        phase="search",
                        message=f"Searching: {q[:40]}...",
                        progress=20 + int(((idx + 1) / max(1, len(optimized_queries))) * 35),
                        items_done=idx + 1,
                        items_total=len(optimized_queries),
                    )
                except Exception:
                    pass

            config = SearchConfig(
                max_results=max_results,
                language="en",
                region="us",
                min_relevance_score=min_rel,
            )

            async with sem:
                try:
                    results = await self._perform_search(
                        q,
                        "aggregate",
                        config,
                        check_cancelled=check_cancelled,
                        progress_callback=progress_callback,
                        research_id=research_id,
                        metrics=metrics,
                    )
                except Exception as e:
                    logger.error("Search failed for '%s': %s", q, e)
                    results = []

            # Attribute cost only to providers observed in results (fallback: configured list)
            try:
                seen = {
                    (str(getattr(r, "source_api", None) or getattr(r, "source", "")).strip().lower())
                    for r in results
                }
                seen.discard("")
                if not seen:
                    mgr = getattr(self, "search_manager", None)
                    seen = set(getattr(mgr, "apis", {}).keys()) if mgr else set()
                for name in seen:
                    c = await self.cost_monitor.track_search_cost(name, 1)
                    if cost_accumulator is not None:
                        cost_accumulator[name] = float(cost_accumulator.get(name, 0.0)) + float(c)
            except Exception:
                logger.debug("Cost attribution failed for query '%s'", q, exc_info=True)

            # Progress (completed + sample sources)
            if progress_callback and research_id:
                try:
                    await progress_callback.report_search_completed(research_id, q, len(results))
                    for r in results[:3]:
                        try:
                            await progress_callback.report_source_found(
                                research_id,
                                {
                                    "title": getattr(r, "title", ""),
                                    "url": getattr(r, "url", ""),
                                    "domain": getattr(r, "domain", ""),
                                    "snippet": (getattr(r, "snippet", "") or "")[:200],
                                    "credibility_score": float(getattr(r, "credibility_score", 0.5) or 0.5),
                                },
                            )
                        except Exception:
                            pass
                except Exception:
                    pass

            return q, results

        tasks = [asyncio.create_task(_run_query(idx, q)) for idx, q in enumerate(optimized_queries)]
        try:
            for fut in asyncio.as_completed(tasks):
                # Cooperative cancellation
                if check_cancelled and await check_cancelled():
                    pending = [t for t in tasks if not t.done()]
                    for t in pending:
                        t.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
                    break
                try:
                    q, res = await fut
                    all_results[q] = res
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # Ensure a failed task does not block others
                    continue
        finally:
            pending = [t for t in tasks if not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        return all_results

    async def _process_results(
        self,
        search_results: Dict[str, List[SearchResult]],
        classification: ClassificationResultSchema,
        context_engineered: ContextEngineeredQuerySchema,
        user_context: Any,
        progress_callback: Optional[Any] = None,
        research_id: Optional[str] = None,
        *,
        metrics: Optional[SearchMetrics] = None,
    ) -> Dict[str, Any]:
        """
        Combine, validate, dedup, early-filter, rank, and score credibility.
        Returns a dict with keys: results, metadata, sources_used, credibility_summary, dedup_stats, contradictions.
        """
        # 1) Combine & basic validation/repairs (mirror legacy path)
        combined: List[SearchResult] = []
        to_backfill: List[Tuple[SearchResult, str]] = []
        for qkey, batch in (search_results or {}).items():
            if not batch:
                continue
            for r in batch:
                if r is None:
                    continue
                # Enforce URL presence when enabled; count drops separately
                url_val = getattr(r, "url", None)
                if not url_val or not isinstance(url_val, str) or not url_val.strip():
                    try:
                        if self.diagnostics.get("enforce_url_presence", True):
                            if metrics is not None:
                                metrics["dropped_no_url"] = int(metrics.get("dropped_no_url", 0)) + 1
                            if self._diag_samples is not None:
                                self._diag_samples.setdefault("no_url", []).append(getattr(r, "title", "") or "")
                            continue
                    except Exception:
                        pass
                # Minimal repair mirroring legacy behavior
                content = getattr(r, "content", "") or ""
                if not str(content).strip():
                    try:
                        sn = getattr(r, "snippet", "") or ""
                        tt = getattr(r, "title", "") or ""
                        if sn.strip():
                            r.content = f"Summary from search results: {sn.strip()}"
                        elif tt.strip():
                            r.content = tt.strip()
                    except Exception:
                        pass
                # Stage a second‑chance fetch for empty content; perform concurrently later
                if not str(getattr(r, "content", "") or "").strip() and getattr(r, "url", None):
                    to_backfill.append((r, getattr(r, "url")))
                # Require content non-empty after repair/fetch; log and surface drops
                if not str(getattr(r, "content", "") or "").strip():
                    try:
                        logger.debug("[process_results] Dropping empty-content result: %s", getattr(r, "url", ""))
                        if progress_callback and research_id:
                            try:
                                await progress_callback.update_progress(
                                    research_id,
                                    phase="filtering",
                                    message="Dropped result with empty content",
                                    custom_data={
                                        "event": "empty_content_drop",
                                        "url": getattr(r, "url", ""),
                                    },
                                )
                            except Exception:
                                pass
                    except Exception:
                        pass
                    continue
                combined.append(r)

        # Concurrent backfill for staged empty-content items (bounded concurrency)
        if to_backfill:
            try:
                sm = getattr(self, "search_manager", None)
                if sm is not None:
                    session = sm._any_session()
                    limit = int(os.getenv("SEARCH_BACKFILL_CONCURRENCY", "8") or 8)
                    sem_fetch = asyncio.Semaphore(max(1, limit))

                    async def _fetch_and_set(item: Tuple[SearchResult, str]):
                        res, url = item
                        async with sem_fetch:
                            try:
                                fetched = await sm.fetcher.fetch(session, url)
                                if fetched and not str(getattr(res, "content", "") or "").strip():
                                    res.content = fetched
                            except Exception:
                                return

                    await asyncio.gather(*[_fetch_and_set(it) for it in to_backfill])
            except Exception:
                pass

        # 2) Dedup
        dedup = await self.deduplicator.deduplicate_results(combined)
        deduped: List[SearchResult] = list(dedup["unique_results"])

        # Update metrics (request-scoped when provided)
        if metrics is not None:
            metrics["total_results"] = len(combined)
            if combined:
                try:
                    metrics["deduplication_rate"] = 1.0 - (len(deduped) / len(combined))
                except Exception:
                    metrics["deduplication_rate"] = 0.0
        # Report deduplication stats to UI
        try:
            if progress_callback and research_id:
                await progress_callback.report_deduplication(
                    research_id,
                    before_count=len(combined),
                    after_count=len(deduped),
                )
        except Exception:
            pass

        # 3) Early filtering
        paradigm_code = normalize_to_internal_code(classification.primary_paradigm)
        early = self._apply_early_relevance_filter(
            deduped,
            getattr(context_engineered, "original_query", ""),
            paradigm_code,
        )

        # 4) Paradigm strategy ranking
        strategy = get_search_strategy(paradigm_code)

        # Safely normalize secondary paradigm only when present to satisfy type checker
        secondary_val = getattr(classification, "secondary_paradigm", None)
        secondary_code = (
            normalize_to_internal_code(secondary_val) if secondary_val is not None else None
        )

        search_ctx = SearchContext(
            original_query=getattr(context_engineered, "original_query", ""),
            paradigm=paradigm_code,
            secondary_paradigm=secondary_code,
        )
        ranked: List[SearchResult] = await strategy.filter_and_rank_results(early, search_ctx)

        # 5) Credibility on top-N (batched with concurrency)
        cred_summary = {"average_score": 0.0}
        creds: Dict[str, float] = {}
        user_cap = int(getattr(user_context, "source_limit", default_source_limit()) or default_source_limit())
        topN = ranked[: min(max(20, user_cap), len(ranked))]

        # Deduplicate by domain, then fetch concurrently
        unique_domains: List[str] = []
        for r in topN:
            d = getattr(r, "domain", "")
            if d and d not in unique_domains:
                unique_domains.append(d)

        if progress_callback and research_id:
            try:
                await progress_callback.update_progress(
                    research_id,
                    phase="analysis",
                    message="Evaluating source credibility",
                    progress=55,
                    items_done=0,
                    items_total=len(unique_domains),
                )
            except Exception:
                pass

        conc = 8
        try:
            conc = int(os.getenv("CREDIBILITY_CONCURRENCY", "8") or 8)
        except Exception:
            pass
        semc = asyncio.Semaphore(max(1, conc))

        async def _fetch_dom(dom: str) -> Tuple[str, float]:
            async with semc:
                try:
                    score, _, _ = await self.get_source_credibility_safe(dom, paradigm_code)
                    return dom, float(score)
                except Exception:
                    return dom, 0.5

        domain_scores: Dict[str, float] = {}
        for idx, (dom, score) in enumerate(await asyncio.gather(*[_fetch_dom(d) for d in unique_domains])):
            domain_scores[dom] = score
            try:
                if progress_callback and research_id:
                    pct = 55 + int(((idx + 1) / max(1, len(unique_domains))) * 20)
                    await progress_callback.update_progress(
                        research_id,
                        phase="analysis",
                        progress=pct,
                        items_done=idx + 1,
                        items_total=len(unique_domains),
                    )
            except Exception:
                pass

        # Assign scores to results (preserve mapping)
        for r in topN:
            dom = getattr(r, "domain", "")
            sc = domain_scores.get(dom, None)
            if sc is not None:
                r.credibility_score = sc
                creds[dom] = sc

        if creds:
            try:
                cred_summary["average_score"] = sum(creds.values()) / float(len(creds))
            except Exception:
                cred_summary["average_score"] = 0.0

        # 6) Final limit by user context
        final_results: List[SearchResult] = ranked[: int(getattr(user_context, "source_limit", default_source_limit()) or default_source_limit())]

        # Update apis_used from final results
        apis = set()
        for r in final_results:
            api = getattr(r, "source_api", getattr(r, "source", "unknown"))
            if api:
                apis.add(api)
        if apis:
            # Normalize to list for type stability (request‑scoped when provided)
            if metrics is not None:
                metrics["apis_used"] = list(apis)

        meta = {
            "processing_time": None,  # caller fills
            "paradigm": getattr(classification.primary_paradigm, "value", paradigm_code),
            "deep_research_enabled": False,
        }

        return {
            "results": final_results,
            "metadata": meta,
            "sources_used": list({getattr(r, "source", getattr(r, "source_api", "web")) for r in final_results}),
            "credibility_summary": cred_summary,
            "dedup_stats": {
                "original_count": len(combined),
                "final_count": len(final_results),
                "duplicates_removed": dedup.get("duplicates_removed", 0),
            },
            "contradictions": {"count": 0, "examples": []},
        }



    async def _perform_search(
        self,
        query: str,
        api: str,
        config: SearchConfig,
        check_cancelled: Optional[Any] = None,
        progress_callback: Optional[Any] = None,
        research_id: Optional[str] = None,
        *,
        metrics: Optional[SearchMetrics] = None,
    ) -> List[SearchResult]:
        """
        Unified single-search with retries/backoff and per-attempt timeout.
        Returns a list of SearchResult; empty list on failure.
        """
        sm = self.search_manager
        if sm is None:
            error_msg = (
                "CRITICAL: Search Manager not initialized - Research pipeline cannot proceed. "
                "This typically means required API keys are missing. "
                "Please ensure at least one of the following is configured: "
                "BRAVE_SEARCH_API_KEY, GOOGLE_CSE_API_KEY + GOOGLE_CSE_CX, "
                "or academic search providers. "
                f"Research ID: {research_id or 'N/A'}, Query: '{query[:100]}...'"
            )
            logger.critical(error_msg)
            raise RuntimeError(error_msg)

        # Request-scoped metrics only (avoid shared-state mutation)
        if metrics is not None:
            mcounts = dict(metrics.get("api_call_counts", {}))
            try:
                if api == "aggregate" and self.search_manager is not None:
                    for name in getattr(self.search_manager, "apis", {}).keys():
                        mcounts[name] = int(mcounts.get(name, 0)) + 1
                else:
                    mcounts[api] = int(mcounts.get(api, 0)) + 1
            except Exception:
                mcounts[api] = int(mcounts.get(api, 0)) + 1
            metrics["api_call_counts"] = mcounts

        attempt = 0
        delay = self.retry_policy.base_delay_sec
        results: List[SearchResult] = []

        while attempt < self.retry_policy.max_attempts:
            # Cancellation gate before each retry attempt
            if check_cancelled and await check_cancelled():
                logger.info("Search cancelled during retry phase for api=%s q='%s'", api, query[:80])
                break

            attempt += 1
            try:
                # Use search_all and emit per-provider progress updates
                coro = sm.search_all(query, config, progress_callback=progress_callback, research_id=research_id)
                task_result = await asyncio.wait_for(coro, timeout=self.search_task_timeout)
                if isinstance(task_result, list):
                    results = task_result
                else:
                    results = []
                break  # success
            except asyncio.TimeoutError:
                if metrics is not None:
                    metrics["task_timeouts"] = int(metrics.get("task_timeouts", 0)) + 1
                if attempt < self.retry_policy.max_attempts:
                    if metrics is not None:
                        metrics["retries_attempted"] = int(metrics.get("retries_attempted", 0)) + 1
                    await asyncio.sleep(min(delay, self.retry_policy.max_delay_sec))
                    delay = min(delay * 2.0, self.retry_policy.max_delay_sec)
                    continue
                else:
                    logger.error("Search task timeout api=%s q='%s'", api, query[:120])
            except asyncio.CancelledError:
                # Preserve cooperative cancellation semantics
                raise
            except Exception as e:
                # Track exceptions by api (request-scoped)
                if metrics is not None:
                    mex = dict(metrics.get("exceptions_by_api", {}))
                    try:
                        if api == "aggregate" and self.search_manager is not None:
                            for name in getattr(self.search_manager, "apis", {}).keys():
                                mex[name] = int(mex.get(name, 0)) + 1
                        else:
                            mex[api] = int(mex.get(api, 0)) + 1
                    except Exception:
                        mex[api] = int(mex.get(api, 0)) + 1
                    metrics["exceptions_by_api"] = mex
                if attempt < self.retry_policy.max_attempts:
                    if metrics is not None:
                        metrics["retries_attempted"] = int(metrics.get("retries_attempted", 0)) + 1
                    await asyncio.sleep(min(delay, self.retry_policy.max_delay_sec))
                    delay = min(delay * 2.0, self.retry_policy.max_delay_sec)
                    continue
                else:
                    logger.error("Search failed api=%s err=%s q='%s'", api, str(e), query[:120])

        return results

    def _apply_early_relevance_filter(
        self,
        results: List[SearchResult],
        query: str,
        paradigm: str
    ) -> List[SearchResult]:
        """Apply early-stage relevance filtering (single pass – theme overlap folded into EarlyRelevanceFilter)"""

        baseline: List[SearchResult] = []
        removed = 0
        for result in results:
            try:
                ok = self.early_filter.is_relevant(result, query, paradigm)
            except Exception:
                ok = True
            if ok:
                baseline.append(result)
            else:
                removed += 1
                logger.debug("[early] drop: %s - %s", getattr(result, "domain", ""), (getattr(result, "title", "") or "")[:60])

        if removed > 0:
            logger.info("Early filter removed %d results", removed)
        return baseline

    # [legacy normalizer removed]

    async def get_source_credibility_safe(self, domain: str, paradigm: str) -> Tuple[float, str, str]:
        """Get source credibility with non-blocking error handling"""
        try:
            if not self.credibility_enabled:
                return 0.5, "Credibility checking disabled", "skipped"

            credibility = await get_source_credibility(domain, paradigm)
            # Prefer structured to_dict if available; fall back to attribute
            explanation = ""
            if hasattr(credibility, "to_dict"):
                card = credibility.to_dict()
                explanation = (
                    f"bias={card.get('bias_rating')}, "
                    f"fact={card.get('fact_check_rating')}, "
                    f"cat={card.get('source_category')}"
                )
            elif hasattr(credibility, "explanation"):
                explanation = getattr(credibility, "explanation") or ""
            return getattr(credibility, "overall_score", 0.5), explanation, "success"
        except Exception as e:
            if self.diagnostics.get("log_credibility_failures", True):
                logger.warning("Credibility check failed for %s: %s", domain, str(e))
            return 0.5, f"Credibility check failed: {str(e)}", "failed"

    async def _execute_deep_research_integration(
        self,
        context_engineered: ContextEngineeredQuerySchema,
        classification: ClassificationResultSchema,
        user_context: Any,
        mode: DeepResearchMode,
        progress_callback: Optional[Any],
        research_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Execute deep research and convert to search results format"""

        if progress_callback and research_id:
            try:
                if hasattr(progress_callback, "update_progress"):
                    await progress_callback.update_progress(
                        research_id,
                        phase="deep_research",
                        message="Starting deep research analysis"
                    )
                elif callable(progress_callback):
                    await progress_callback("Starting deep research analysis")
            except Exception:
                pass

        deep_config = DeepResearchConfig(
            mode=mode,
            enable_web_search=True,
            enable_code_interpreter=False,
            background=True,
            include_paradigm_context=True,
        )

        # Apply experiment override from research record when available
        try:
            if research_id:
                rec = await self.research_store.get(research_id)
                pv = ((rec or {}).get("experiment") or {}).get("prompt_variant")
                if isinstance(pv, str) and pv in {"v1", "v2"}:
                    deep_config.prompt_variant = pv
        except Exception:
            pass

        try:
            deep_result = await deep_research_service.execute_deep_research(
                query=context_engineered.original_query,
                classification=classification,           # type: ignore[arg-type]
                context_engineering=context_engineered,   # type: ignore[arg-type]
                config=deep_config,
                progress_tracker=progress_callback,
                research_id=research_id,
            )
        except Exception as e:
            error_msg = (
                f"CRITICAL: Deep Research failed - Enhanced research not available. "
                f"Error: {str(e)}. "
                f"This may indicate: "
                f"1) Azure OpenAI issues (missing API key, wrong endpoint/deployment) "
                f"2) Deep research service initialization failure "
                f"3) Token limit exceeded or rate limiting. "
                f"Research ID: {research_id or 'N/A'}, "
                f"Query: '{context_engineered.original_query[:100]}...' "
                f"Mode: {mode.value if mode else 'default'}"
            )
            logger.critical(error_msg)
            logger.critical(f"Deep Research traceback: ", exc_info=True)
            # Return empty list to allow fallback to standard search
            logger.warning("Falling back to standard search results without deep research enhancement")
            return []

        if not getattr(deep_result, "content", None):
            logger.warning(
                f"Deep Research returned no content - Empty result. "
                f"Research ID: {research_id or 'N/A'}, Query: '{context_engineered.original_query[:100]}...'"
            )
            return []

        deep_search_results = []
        # Convert deep research citations into typed EvidenceQuote and stash for bundle merge
        try:
            from services.deep_research_service import convert_citations_to_evidence_quotes
            evq = convert_citations_to_evidence_quotes(
                getattr(deep_result, "citations", []) or [],
                getattr(deep_result, "content", "") or "",
            )
            if research_id and evq:
                self._deep_citations_map[research_id] = evq
        except Exception:
            pass

        for citation in getattr(deep_result, "citations", []) or []:
            title = getattr(citation, "title", "") or ""
            url = getattr(citation, "url", "") or ""
            s = int(getattr(citation, "start_index", 0) or 0)
            e = int(getattr(citation, "end_index", s) or s)
            body = getattr(deep_result, "content", "") or ""
            snippet = body[s:e][:200] if isinstance(body, str) else ""

            # Do not synthesize URLs for unlinked citations. These are preserved
            # via the evidence bundle path and should not be emitted as results.
            if not url or not isinstance(url, str) or not url.strip():
                continue

            try:
                domain = url.split('/')[2] if ('/' in url and len(url.split('/')) > 2) else url
            except Exception:
                domain = ""
            deep_search_results.append({
                "title": title or (url.split('/')[-1] if url else "(deep research)"),
                "url": url,
                "snippet": snippet,
                "domain": domain,
                "result_type": "deep_research",
                "credibility_score": 0.9,
                "origin_query": getattr(context_engineered, "original_query", ""),
                "search_api": "deep_research",
                "source_api": "deep_research",
                "source": "deep_research",
                "metadata": {}
            })

        return deep_search_results
    # [legacy → V2 adapter removed]


    async def execute_deep_research(
        self,
        context_engineered: ContextEngineeredQuerySchema,
        enable_standard_search: bool = True,  # ignored (legacy path removed)
        deep_research_mode: DeepResearchMode = DeepResearchMode.PARADIGM_FOCUSED,
        search_context_size: Optional[Any] = None,
        user_location: Optional[Dict[str, str]] = None,
        progress_tracker: Optional[Any] = None,
        research_id: Optional[str] = None,
    ) -> ResearchExecutionResult:
        """Run only the deep research path and return a minimal ResearchExecutionResult."""
        class _UserCtxShim:
            def __init__(self, role="PRO", source_limit: int = default_source_limit(), is_pro_user=True, preferences=None):
                self.role = role
                self.source_limit = source_limit
                self.is_pro_user = is_pro_user
                self.preferences = preferences or {}
        uc = getattr(context_engineered, "user_context", None)
        user_context = _UserCtxShim(
            role=getattr(uc, "role", "PRO") if uc else "PRO",
            source_limit=getattr(uc, "source_limit", default_source_limit()) if uc else default_source_limit(),
            is_pro_user=getattr(uc, "is_pro_user", True) if uc else True,
            preferences=getattr(uc, "preferences", {}) if uc else {},
        )

        classification = getattr(context_engineered, "classification", None)
        if classification is None:
            try:
                orig_q = getattr(context_engineered, "original_query", "")
                classification = ClassificationResultSchema(
                    query=orig_q,
                    primary_paradigm=HostParadigm.BERNARD,
                    secondary_paradigm=None,
                    distribution={HostParadigm.BERNARD: 1.0},
                    confidence=0.9,
                    features=QueryFeaturesSchema(text=orig_q),
                    reasoning={HostParadigm.BERNARD: ["default"]},
                )
            except Exception:
                class _C:
                    primary_paradigm = HostParadigm.BERNARD
                    secondary_paradigm = None
                    confidence = 0.9
                classification = _C()  # type: ignore

        deep_items = await self._execute_deep_research_integration(
            context_engineered=context_engineered,
            classification=classification,  # type: ignore[arg-type]
            user_context=user_context,
            mode=deep_research_mode,
            progress_callback=progress_tracker,
            research_id=research_id,
        )

        adapted: List[SearchResult] = []
        for d in deep_items or []:
            url = (d.get("url") or "").strip()
            domain = url.split("/")[2] if ("/" in url and len(url.split("/")) > 2) else url
            title = d.get("title") or (url.split("/")[-1] if url else "(deep research)")
            snippet = d.get("snippet") or d.get("summary") or ""
            src = d.get("source_api") or d.get("search_api") or "deep_research"
            try:
                adapted.append(
                    SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source=src,
                        domain=domain,
                        content=snippet or title,
                    )
                )
            except Exception:
                continue

        final_results = adapted[: int(getattr(user_context, "source_limit", default_source_limit()) or default_source_limit())]
        credibility_scores: Dict[str, float] = {}
        for r in final_results[:20]:
            try:
                score, _, _ = await self.get_source_credibility_safe(getattr(r, "domain", ""), "bernard")
                r.credibility_score = score
                if getattr(r, "domain", ""):
                    credibility_scores[getattr(r, "domain", "")] = score
            except Exception:
                pass

        return ResearchExecutionResult(
            original_query=getattr(context_engineered, "original_query", ""),
            paradigm=normalize_to_internal_code(getattr(classification, "primary_paradigm", HostParadigm.BERNARD)),
            secondary_paradigm=None,
            search_queries_executed=[],
            raw_results={"deep_research": adapted},
            filtered_results=final_results,
            secondary_results=[],
            credibility_scores=credibility_scores,
            execution_metrics={
                "queries_executed": 0,
                "raw_results_count": len(adapted),
                "deduplicated_count": len(adapted),
                "final_results_count": len(final_results),
                "deep_research_enabled": True,
            },
            cost_breakdown={"deep_research": 0.0},
            status=ContractResearchStatus.OK if final_results else ContractResearchStatus.FAILED_NO_SOURCES,
        )


    async def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {"message": "No executions yet"}

        total_executions = len(self.execution_history)

        avg_processing_time = (
            sum(
                r.execution_metrics.get("processing_time_seconds", 0)
                for r in self.execution_history
            )
            / total_executions
        )

        paradigm_counts = {}
        for result in self.execution_history:
            paradigm_counts[result.paradigm] = (
                paradigm_counts.get(result.paradigm, 0) + 1
            )

        total_cost = sum(
            sum(result.cost_breakdown.values()) for result in self.execution_history
        )

        return {
            "total_executions": total_executions,
            "average_processing_time": round(avg_processing_time, 2),
            "paradigm_distribution": paradigm_counts,
            "total_cost": round(total_cost, 4),
            "cache_stats": await cache_manager.get_cache_stats(),
            "capabilities": self.get_capabilities()
        }


# Global orchestrator instance (use subclass with execute_research)
research_orchestrator = ResearchOrchestrator()

# Convenience functions for backward compatibility
async def initialize_research_system():
    """Initialize the complete research system"""
    await research_orchestrator.initialize()
    logger.info("Complete research system initialized")

# [legacy convenience execute_research removed]

# Alias for imports expecting different names
research_orchestrator_v2 = research_orchestrator
unified_orchestrator = research_orchestrator
enhanced_orchestrator = research_orchestrator
