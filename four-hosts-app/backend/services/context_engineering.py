"""
Four Hosts Context Engineering Pipeline
W-S-C-I (Write-Select-Compress-Isolate) implementation
"""
# flake8: noqa: E501

import logging
import structlog
import os
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import re
from abc import ABC, abstractmethod

# Import from classification engine
from .classification_engine import (
    HostParadigm,
    ClassificationResult,
)
from . import paradigm_search
from services.query_planning.optimizer import QueryOptimizer  # type: ignore
from search.query_planner import QueryPlanner
from services.query_planning import PlannerConfig, build_planner_config
from services.llm_client import llm_client  # type: ignore
try:
    from . import experiments  # Optional A/B helper
except Exception:
    experiments = None  # type: ignore

# Configure logging
logger = structlog.get_logger(__name__)

# --- Data Models ---


@dataclass
class WriteLayerOutput:
    """Output from Write layer processing"""

    paradigm: HostParadigm
    documentation_focus: str
    key_themes: List[str]
    narrative_frame: str
    search_priorities: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SelectLayerOutput:
    """Output from Select layer processing"""

    paradigm: HostParadigm
    search_queries: List[Dict[str, Any]]  # Enhanced queries with metadata
    source_preferences: List[str]
    exclusion_filters: List[str]
    tool_selections: List[str]
    max_sources: int
    # New: normalized source types for SearchConfig (e.g., web/news/academic)
    normalized_source_types: List[str] = field(default_factory=list)
    # New: domain authority whitelist for paradigm
    authority_whitelist: List[str] = field(default_factory=list)


@dataclass
class CompressLayerOutput:
    """Output from Compress layer processing"""

    paradigm: HostParadigm
    compression_ratio: float
    compression_strategy: str
    priority_elements: List[str]
    removed_elements: List[str]
    token_budget: int
    # Optional per-category budget plan in tokens
    budget_plan: Dict[str, int] = field(default_factory=dict)


@dataclass
class IsolateLayerOutput:
    """Output from Isolate layer processing"""

    paradigm: HostParadigm
    isolation_strategy: str
    key_findings_criteria: List[str]
    extraction_patterns: List[str]
    focus_areas: List[str]
    output_structure: Dict[str, Any]


@dataclass
class ContextEngineeredQuery:
    """Complete context-engineered query ready for research execution"""

    original_query: str
    classification: ClassificationResult
    write_output: WriteLayerOutput
    select_output: SelectLayerOutput
    compress_output: CompressLayerOutput
    isolate_output: IsolateLayerOutput
    processing_time: float
    # New optional layer outputs
    rewrite_output: Optional[Dict[str, Any]] = None
    optimize_output: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    # Optional: per-layer durations in seconds
    layer_durations: Dict[str, float] = field(default_factory=dict)
    # Refined queries after rewrite/optimization (consumed by orchestrator)
    refined_queries: List[str] = field(default_factory=list)


# --- Base Layer Class ---


class ContextLayer(ABC):
    """Abstract base class for context engineering layers"""

    def __init__(self, name: str):
        self.name = name
        self.processing_history = []

    @abstractmethod
    async def process(
        self,
        classification: ClassificationResult,
        previous_outputs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Process input through this layer"""
        pass

    def log_processing(self, input_data: Any, output_data: Any, duration: float):
        """Log processing metrics"""
        self.processing_history.append(
            {
                "timestamp": datetime.now(),
                "duration": duration,
                "input_size": len(str(input_data)),
                "output_size": len(str(output_data)),
            }
        )


# --- Write Layer Implementation ---


class WriteLayer(ContextLayer):
    """Documents according to paradigm-specific focus"""

    def __init__(self):
        super().__init__("Write")

        # Paradigm-specific documentation strategies
        self.paradigm_strategies = {
            HostParadigm.DOLORES: {
                "focus": "Document systemic injustices and power imbalances",
                "themes": [
                    "oppression",
                    "resistance",
                    "truth",
                    "justice",
                    "revolution",
                ],
                "narrative": "victim-oppressor dynamics",
                "priorities": [
                    "whistleblower accounts",
                    "leaked documents",
                    "investigative reports",
                    "historical parallels",
                ],
            },
            HostParadigm.TEDDY: {
                "focus": "Create empathetic profiles and document care needs",
                "themes": [
                    "compassion",
                    "support",
                    "healing",
                    "community",
                    "protection",
                ],
                "narrative": "helper-recipient relationships",
                "priorities": [
                    "personal stories",
                    "community resources",
                    "best practices",
                    "success stories",
                ],
            },
            HostParadigm.BERNARD: {
                "focus": "Systematic documentation of variables and evidence",
                "themes": [
                    "analysis",
                    "causation",
                    "correlation",
                    "methodology",
                    "validation",
                ],
                "narrative": "hypothesis-evidence framework",
                "priorities": [
                    "peer-reviewed studies",
                    "data sets",
                    "meta-analyses",
                    "reproducible methods",
                ],
            },
            HostParadigm.MAEVE: {
                "focus": "Map strategic landscape and actionable opportunities",
                "themes": [
                    "leverage",
                    "advantage",
                    "optimization",
                    "influence",
                    "control",
                ],
                "narrative": "competitor-opportunity matrix",
                "priorities": [
                    "case studies",
                    "market analysis",
                    "implementation guides",
                    "success metrics",
                ],
            },
        }

    async def process(
        self,
        classification: ClassificationResult,
        previous_outputs: Optional[Dict[str, Any]] = None,
    ) -> WriteLayerOutput:
        """Process query through Write layer"""
        start_time = datetime.now()

        paradigm = classification.primary_paradigm
        strategy = self.paradigm_strategies[paradigm]

        # Extract key themes from query
        query_themes = self._extract_query_themes(classification)

        # Combine with paradigm themes
        all_themes = list(set(strategy["themes"] + query_themes))

        # Generate search priorities based on query + paradigm
        search_priorities = self._generate_search_priorities(
            classification, strategy["priorities"]
        )

        output = WriteLayerOutput(
            paradigm=paradigm,
            documentation_focus=strategy["focus"],
            key_themes=all_themes[:10],  # Top 10 themes
            narrative_frame=strategy["narrative"],
            search_priorities=search_priorities[:8],  # Top 8 priorities
        )

        # Log processing
        duration = (datetime.now() - start_time).total_seconds()
        self.log_processing(classification, output, duration)

        logger.info(f"Write layer processed with focus: {output.documentation_focus}")

        return output

    def _extract_query_themes(self, classification: ClassificationResult) -> List[str]:
        """Extract themes from the query itself."""
        themes: List[str] = []

        # Use extracted entities directly as themes
        try:
            themes.extend(classification.features.entities)
        except Exception:
            pass

        # Pull out important tokens (exclude filler words)
        try:
            important_tokens = [
                t
                for t in classification.features.tokens
                if isinstance(t, str)
                and len(t) > 5
                and t.lower() not in {"should", "would", "could", "about", "through"}
            ]
            themes.extend(important_tokens[:5])
        except Exception:
            pass

        # Deduplicate while preserving order
        seen: Set[str] = set()
        return [t for t in themes if not (t in seen or seen.add(t))]

    def _generate_search_priorities(
        self, classification: ClassificationResult, base_priorities: List[str]
    ) -> List[str]:
        """Generate search priorities based on query context."""
        priorities = list(base_priorities)

        # Add domain-specific priorities when available
        try:
            domain = classification.features.domain
        except Exception:
            domain = None

        domain_priorities = {
            "business": ["industry reports", "competitor analysis"],
            "healthcare": ["clinical studies", "patient outcomes"],
            "education": ["pedagogical research", "learning outcomes"],
            "technology": ["technical documentation", "benchmarks"],
            "social_justice": ["advocacy reports", "policy analysis"],
        }

        if domain and domain in domain_priorities:
            priorities.extend(domain_priorities[domain])

        # Consider temporal aspect if recency is emphasized
        try:
            if getattr(classification.features, "time_sensitive", False):
                priorities.append("recent developments")
        except Exception:
            pass

        # Deduplicate while preserving order
        seen: Set[str] = set()
        return [p for p in priorities if not (p in seen or seen.add(p))]


class RewriteLayer(ContextLayer):
    """Rewrites the original query for clarity and searchability"""

    def __init__(self):
        super().__init__("Rewrite")

    async def process(
        self,
        classification: ClassificationResult,
        previous_outputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        start_time = datetime.now()
        original = classification.query
        paradigm = classification.primary_paradigm.value
        rewrites: List[str] = []
        method = "heuristic"
        # Prefer LLM rewrite when available
        try:
            # Choose a variant if experiments are enabled
            variant = (previous_outputs or {}).get("_experiment", {}).get("context_rewrite_variant", "v1")
            if variant == "v2":
                prompt = (
                    "Rewrite the user query for search. Produce exactly 3 rewritten variants,"
                    " each on its own line. Keep named entities in quotes. Remove filler words,"
                    " prefer active verbs, and add one domain-specific keyword when obvious.\n\n"
                    f"Query: {original}"
                )
            else:
                prompt = (
                    "Rewrite the user query to be concise, specific, and search-friendly. "
                    "Preserve the intent. Quote named entities and key phrases.\n\n"
                    f"Query: {original}"
                )
            txt = await llm_client.generate_completion(prompt, paradigm=paradigm)
            if isinstance(txt, str) and txt.strip():
                lines_out = [s.strip("- ") for s in str(txt).splitlines() if s.strip()]
                for s in lines_out:
                    m = re.match(r"^(?:\d+\.|\(\d+\))\s*(.*)$", s)
                    rewrites.append(m.group(1) if m else s)
                rewrites = [r for r in rewrites if len(r) >= 6][:3]
                if rewrites:
                    method = "llm"
        except Exception:
            pass

        if not rewrites:
            # Heuristic fallback: quote extracted entities and trim filler
            entities = classification.features.entities
            core = original
            for e in entities:
                try:
                    core = re.sub(rf"\b{re.escape(e)}\b", f'"{e}"', core)
                except re.error:
                    continue
            core = re.sub(r"\b(please|kindly|can you|would you|tell me|about)\b", "", core, flags=re.I)
            core = re.sub(r"\s+", " ", core).strip()
            rewrites = [core] if core else [original]

        duration = (datetime.now() - start_time).total_seconds()
        output = {
            "method": method,
            "rewritten": rewrites[0],
            "alternatives": rewrites,
        }
        self.log_processing(original, output, duration)
        return output

        # Add urgency-based priorities
        if classification.features.urgency_score > 0.7:
            priorities.insert(0, "recent developments")
            priorities.insert(1, "breaking news")

        return priorities


# --- Select Layer Implementation ---


class SelectLayer(ContextLayer):
    """Selects methods and sources based on paradigm"""

    def __init__(self):
        super().__init__("Select")

        # Paradigm-specific selection strategies
        self.selection_strategies = {
            HostParadigm.DOLORES: {
                "query_modifiers": [
                    "expose",
                    "reveal",
                    "uncover",
                    "truth about",
                    "scandal",
                ],
                "source_types": [
                    "investigative",
                    "alternative",
                    "whistleblower",
                    "activist",
                ],
                "exclude": [
                    "corporate PR",
                    "government propaganda",
                    "sponsored content",
                ],
                "tools": ["deep_web_search", "document_analysis", "fact_checking"],
                "max_sources": 100,
            },
            HostParadigm.TEDDY: {
                "query_modifiers": [
                    "support",
                    "help",
                    "resources",
                    "guide",
                    "community",
                ],
                "source_types": [
                    "nonprofit",
                    "community",
                    "educational",
                    "testimonial",
                ],
                "exclude": ["commercial", "exploitative", "sensational"],
                "tools": ["resource_finder", "community_search", "support_networks"],
                "max_sources": 75,
            },
            HostParadigm.BERNARD: {
                "query_modifiers": [
                    "research",
                    "study",
                    "analysis",
                    "data",
                    "evidence",
                ],
                "source_types": [
                    "academic",
                    "peer-reviewed",
                    "governmental",
                    "statistical",
                ],
                "exclude": ["opinion", "anecdotal", "unverified"],
                "tools": ["academic_search", "data_analysis", "citation_network"],
                "max_sources": 150,
            },
            HostParadigm.MAEVE: {
                "query_modifiers": [
                    "strategy",
                    "tactics",
                    "competitive",
                    "optimize",
                    "leverage",
                ],
                "source_types": ["industry", "consultancy", "case study", "strategic"],
                "exclude": ["theoretical", "outdated", "generic"],
                "tools": ["market_analysis", "competitor_intel", "trend_analysis"],
                "max_sources": 80,
            },
        }

    async def process(
        self,
        classification: ClassificationResult,
        previous_outputs: Optional[Dict[str, Any]] = None,
    ) -> SelectLayerOutput:
        """Process through Select layer"""
        start_time = datetime.now()

        paradigm = classification.primary_paradigm
        strategy = self.selection_strategies[paradigm]
        write_output = previous_outputs.get("write") if previous_outputs else None

        # Generate enhanced search queries
        search_strategy = paradigm_search.get_search_strategy(paradigm.value)
        search_context = paradigm_search.SearchContext(
            original_query=classification.query,
            paradigm=paradigm.value,
            secondary_paradigm=classification.secondary_paradigm.value
            if classification.secondary_paradigm
            else None,
        )
        search_queries = await search_strategy.generate_search_queries(search_context)

        # Add secondary paradigm queries if significant
        # if (
        #     classification.secondary_paradigm
        #     and classification.distribution[classification.secondary_paradigm] > 0.25
        # ):
        #     secondary_queries = self._generate_secondary_queries(
        #         classification, classification.secondary_paradigm
        #     )
        #     search_queries.extend(secondary_queries[:3])  # Add top 3

        # Normalize source types for SearchConfig
        normalized_map = {
            HostParadigm.MAEVE: ["web", "news"],
            HostParadigm.BERNARD: ["academic", "web"],
            HostParadigm.DOLORES: ["news", "web"],
            HostParadigm.TEDDY: ["web", "news"],
        }

        # Authority whitelists per paradigm
        authority_whitelists = {
            HostParadigm.MAEVE: [
                "hbr.org", "mckinsey.com", "bcg.com", "bain.com", "strategy-business.com",
                "ft.com", "wsj.com", "bloomberg.com", "gartner.com", "forrester.com"
            ],
            HostParadigm.BERNARD: [
                "nature.com", "science.org", "arxiv.org", "pubmed.ncbi.nlm.nih.gov",
                "ieee.org", "acm.org", "springer.com", "sciencedirect.com", "jstor.org",
                "nih.gov", "nasa.gov"
            ],
            HostParadigm.DOLORES: [
                "propublica.org", "theintercept.com", "icij.org", "theguardian.com",
                "washingtonpost.com", "nytimes.com"
            ],
            HostParadigm.TEDDY: [
                "npr.org", "who.int", "unicef.org", "redcross.org", "cdc.gov", "usa.gov"
            ],
        }

        output = SelectLayerOutput(
            paradigm=paradigm,
            search_queries=search_queries,
            source_preferences=strategy["source_types"],
            exclusion_filters=strategy["exclude"],
            tool_selections=strategy["tools"],
            max_sources=strategy["max_sources"],
            normalized_source_types=normalized_map.get(paradigm, ["web"]),
            authority_whitelist=authority_whitelists.get(paradigm, []),
        )

        # Log processing
        duration = (datetime.now() - start_time).total_seconds()
        self.log_processing(classification, output, duration)

        logger.info(f"Select layer generated {len(search_queries)} search queries")

        return output




# --- Compress Layer Implementation ---


class CompressLayer(ContextLayer):
    """Compresses information based on paradigm priorities"""

    def __init__(self):
        super().__init__("Compress")

        # Paradigm-specific compression strategies
        self.compression_strategies = {
            HostParadigm.DOLORES: {
                "ratio": 0.7,  # Keep 70% - preserve emotional impact
                "strategy": "impact_preservation",
                "priorities": [
                    "evidence of wrongdoing",
                    "victim testimonies",
                    "power dynamics",
                    "calls to action",
                ],
                "remove": [
                    "neutral descriptions",
                    "corporate justifications",
                    "technical details",
                    "sidebar information",
                ],
            },
            HostParadigm.TEDDY: {
                "ratio": 0.6,  # Keep 60% - preserve human stories
                "strategy": "narrative_preservation",
                "priorities": [
                    "personal stories",
                    "emotional context",
                    "support resources",
                    "positive outcomes",
                ],
                "remove": [
                    "statistics without context",
                    "technical jargon",
                    "impersonal data",
                    "negative speculation",
                ],
            },
            HostParadigm.BERNARD: {
                "ratio": 0.5,  # Keep 50% - maximum pattern extraction
                "strategy": "data_distillation",
                "priorities": [
                    "statistical findings",
                    "causal relationships",
                    "methodology",
                    "reproducible results",
                ],
                "remove": [
                    "anecdotes",
                    "opinions",
                    "emotional language",
                    "unverified claims",
                ],
            },
            HostParadigm.MAEVE: {
                "ratio": 0.4,  # Keep 40% - only actionable intelligence
                "strategy": "action_extraction",
                "priorities": [
                    "specific tactics",
                    "success metrics",
                    "implementation steps",
                    "competitive advantages",
                ],
                "remove": [
                    "background theory",
                    "historical context",
                    "philosophical discussion",
                    "general advice",
                ],
            },
        }

    async def process(
        self,
        classification: ClassificationResult,
        previous_outputs: Optional[Dict[str, Any]] = None,
    ) -> CompressLayerOutput:
        """Process through Compress layer"""
        start_time = datetime.now()

        paradigm = classification.primary_paradigm
        strategy = self.compression_strategies[paradigm]

        # Calculate token budget based on query complexity and env knobs
        try:
            base_tokens = int(os.getenv("CE_BASE_TOKENS", "2000"))
        except Exception:
            base_tokens = 2000
        complexity_multiplier = 1.0 + float(getattr(classification.features, "complexity_score", 0.0) or 0.0)
        # Optional urgency nudge (e.g., investigative may need more)
        urgency = float(getattr(classification.features, "urgency_score", 0.0) or 0.0)
        urgency_multiplier = 1.0 + (0.25 * urgency)
        raw_budget = base_tokens * complexity_multiplier * urgency_multiplier * float(strategy["ratio"])
        # Clamp to sane bounds; allow env overrides per paradigm
        try:
            max_cap = int(os.getenv("CE_MAX_TOKENS", "6000"))
            min_cap = int(os.getenv("CE_MIN_TOKENS", "800"))
        except Exception:
            max_cap, min_cap = 6000, 800
        token_budget = int(max(min_cap, min(max_cap, raw_budget)))

        # Derive a simple budget plan for instructions/knowledge/tools/scratch
        try:
            from utils.token_budget import compute_budget_plan  # type: ignore
        except Exception:
            # Local import fallback when relative path differs
            try:
                from utils.token_budget import compute_budget_plan  # type: ignore
            except Exception:
                compute_budget_plan = None  # type: ignore

        budget_plan = {}
        if compute_budget_plan:
            budget_plan = compute_budget_plan(
                token_budget,
                {"instructions": 0.15, "knowledge": 0.70, "tools": 0.15, "scratch": 0.0},
            )

        output = CompressLayerOutput(
            paradigm=paradigm,
            compression_ratio=strategy["ratio"],
            compression_strategy=strategy["strategy"],
            priority_elements=strategy["priorities"],
            removed_elements=strategy["remove"],
            token_budget=token_budget,
            budget_plan=budget_plan,
        )

        # Log processing
        duration = (datetime.now() - start_time).total_seconds()
        self.log_processing(classification, output, duration)

        logger.info(
            f"Compress layer set ratio: {output.compression_ratio:.0%}, "
            f"token budget: {output.token_budget}"
        )

        return output


# --- Isolate Layer Implementation ---


class IsolateLayer(ContextLayer):
    """Isolates key findings based on paradigm strategy"""

    def __init__(self):
        super().__init__("Isolate")

        # Paradigm-specific isolation strategies
        self.isolation_strategies = {
            HostParadigm.DOLORES: {
                "strategy": "pattern_of_injustice",
                "criteria": [
                    "systemic patterns",
                    "power imbalances",
                    "victim impact",
                    "accountability gaps",
                ],
                "patterns": [
                    r"pattern of\s+\w+",
                    r"systematic\s+\w+",
                    r"repeated\s+\w+",
                    r"history of\s+\w+",
                ],
                "focus": [
                    "root causes",
                    "responsible parties",
                    "impact scale",
                    "resistance opportunities",
                ],
                "structure": {
                    "injustices_found": [],
                    "patterns_identified": [],
                    "accountability_gaps": [],
                    "action_opportunities": [],
                },
            },
            HostParadigm.TEDDY: {
                "strategy": "human_centered_needs",
                "criteria": [
                    "individual needs",
                    "available support",
                    "success stories",
                    "care strategies",
                ],
                "patterns": [
                    r"help(?:ing|s)?\s+\w+",
                    r"support(?:ing|s)?\s+\w+",
                    r"care\s+for\s+\w+",
                    r"protect(?:ing|s)?\s+\w+",
                ],
                "focus": [
                    "immediate needs",
                    "long-term support",
                    "community resources",
                    "individual dignity",
                ],
                "structure": {
                    "needs_identified": [],
                    "resources_available": [],
                    "support_strategies": [],
                    "success_examples": [],
                },
            },
            HostParadigm.BERNARD: {
                "strategy": "empirical_extraction",
                "criteria": [
                    "statistical significance",
                    "causal links",
                    "research gaps",
                    "methodological strengths",
                ],
                "patterns": [
                    r"\d+%\s+of\s+\w+",
                    r"correlation\s+between",
                    r"study\s+found",
                    r"evidence\s+suggests",
                ],
                "focus": [
                    "key findings",
                    "statistical patterns",
                    "knowledge gaps",
                    "future research",
                ],
                "structure": {
                    "empirical_findings": [],
                    "statistical_patterns": [],
                    "causal_relationships": [],
                    "research_directions": [],
                },
            },
            HostParadigm.MAEVE: {
                "strategy": "strategic_intelligence",
                "criteria": [
                    "competitive advantages",
                    "implementation tactics",
                    "success metrics",
                    "resource requirements",
                ],
                "patterns": [
                    r"strategy\s+to\s+\w+",
                    r"tactic\s+for\s+\w+",
                    r"advantage\s+of\s+\w+",
                    r"optimize\s+\w+",
                ],
                "focus": [
                    "quick wins",
                    "long-term positioning",
                    "resource allocation",
                    "success measurement",
                ],
                "structure": {
                    "strategic_opportunities": [],
                    "tactical_approaches": [],
                    "implementation_steps": [],
                    "success_metrics": [],
                },
            },
        }

    async def process(
        self,
        classification: ClassificationResult,
        previous_outputs: Optional[Dict[str, Any]] = None,
    ) -> IsolateLayerOutput:
        """Process through Isolate layer"""
        start_time = datetime.now()

        paradigm = classification.primary_paradigm
        strategy = self.isolation_strategies[paradigm]

        # Compile extraction patterns
        compiled_patterns = [re.compile(p, re.IGNORECASE) for p in strategy["patterns"]]

        output = IsolateLayerOutput(
            paradigm=paradigm,
            isolation_strategy=strategy["strategy"],
            key_findings_criteria=strategy["criteria"],
            extraction_patterns=[p.pattern for p in compiled_patterns],
            focus_areas=strategy["focus"],
            output_structure=strategy["structure"].copy(),
        )

        # Log processing
        duration = (datetime.now() - start_time).total_seconds()
        self.log_processing(classification, output, duration)

        logger.info(f"Isolate layer configured for: {output.isolation_strategy}")

        return output


class OptimizeLayer(ContextLayer):
    """Optimizes search terms and builds final query variations"""

    def __init__(self):
        super().__init__("Optimize")
        self.optimizer = QueryOptimizer()
        self._cached_planner: Optional[QueryPlanner] = None

    def _planner_config(self) -> PlannerConfig:
        max_candidates = int(os.getenv("CE_PLANNER_MAX_CANDIDATES", "10") or 10)
        base = PlannerConfig(max_candidates=max(1, max_candidates))
        cfg = build_planner_config(base=base)
        cfg.enable_agentic = False
        cfg.per_stage_caps["agentic"] = 0
        cfg.per_stage_caps["context"] = min(cfg.per_stage_caps.get("context", 6), 4)
        stage_order_env = os.getenv("CE_PLANNER_STAGE_ORDER")
        if stage_order_env:
            parts = [p.strip().lower() for p in stage_order_env.split(",") if p.strip()]
            valid = [p for p in parts if p in cfg.per_stage_caps]
            if valid:
                cfg.stage_order = valid
        dedup_env = os.getenv("CE_PLANNER_DEDUP_JACCARD")
        if dedup_env:
            try:
                cfg.dedup_jaccard = float(dedup_env)
            except Exception:
                pass
        rule_cap = os.getenv("CE_PLANNER_RULE_CAP")
        if rule_cap:
            try:
                cfg.per_stage_caps["rule_based"] = max(1, int(rule_cap))
            except Exception:
                pass
        return cfg

    async def process(
        self,
        classification: ClassificationResult,
        previous_outputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        start_time = datetime.now()
        paradigm = classification.primary_paradigm.value
        original = classification.query
        rewritten = None
        if previous_outputs and "rewrite" in previous_outputs:
            rewritten = previous_outputs["rewrite"].get("rewritten")
        base_query = rewritten or original

        planner_cfg = self._planner_config()
        planner = self._cached_planner or QueryPlanner(planner_cfg)
        self._cached_planner = planner

        additional_queries: List[str] = []
        try:
            select_output = previous_outputs.get("select") if previous_outputs else None
            if select_output and getattr(select_output, "search_queries", None):
                for item in select_output.search_queries or []:
                    if isinstance(item, dict):
                        candidate = item.get("query")
                    else:
                        candidate = getattr(item, "query", None)
                    if isinstance(candidate, str) and candidate.strip():
                        additional_queries.append(candidate.strip())
        except Exception:
            additional_queries = []

        terms = self.optimizer.get_key_terms(base_query)

        primary = base_query
        variations: Dict[str, str] = {}

        try:
            plan = await planner.initial_plan(
                seed_query=base_query,
                paradigm=paradigm,
                additional_queries=additional_queries,
            )
        except Exception as exc:
            logger.debug(f"Planner initial_plan failed in OptimizeLayer: {exc}")
            plan = []

        if plan:
            primary = plan[0].query
            for cand in plan:
                if cand.stage == "rule_based":
                    key = cand.label or "rule_based"
                else:
                    key = f"{cand.stage}:{cand.label}" if cand.label else cand.stage
                key = key.lower()
                if key not in variations:
                    variations[key] = cand.query
        else:
            variations = self.optimizer.generate_query_variations(base_query, paradigm)
            primary = self.optimizer.optimize_query(base_query, paradigm)
            if planner_cfg.enable_llm:
                try:
                    from services.llm_query_optimizer import propose_semantic_variations  # type: ignore

                    llm_vars = await propose_semantic_variations(
                        base_query,
                        paradigm,
                        max_variants=planner_cfg.per_stage_caps.get("llm", 4),
                        key_terms=terms[:8],
                    )
                    for i, v in enumerate(llm_vars):
                        key = f"llm:{i+1}"
                        if key not in variations:
                            variations[key] = v
                except Exception as e:
                    logger.debug(f"LLM query optimizer skipped: {e}")

        output = {
            "primary_query": primary,
            "optimized_terms": terms[:20],
            "variations": variations,
            "variations_count": len(variations),
        }
        duration = (datetime.now() - start_time).total_seconds()
        self.log_processing(base_query, output, duration)
        return output


# --- Context Engineering Pipeline ---


class ContextEngineeringPipeline:
    """Main pipeline orchestrating all W-S-C-I layers"""

    def __init__(self):
        self.write_layer = WriteLayer()
        self.rewrite_layer = RewriteLayer()
        self.select_layer = SelectLayer()
        self.optimize_layer = OptimizeLayer()
        self.compress_layer = CompressLayer()
        self.isolate_layer = IsolateLayer()
        self.processing_history = []

    async def process_query(
        self, classification: ClassificationResult,
        research_id: Optional[str] = None
    ) -> ContextEngineeredQuery:
        """Process classification through all context layers"""
        start_time = datetime.now()

        logger.info(
            f"Starting context engineering for paradigm: {classification.primary_paradigm.value}"
        )

        # Import progress tracker if available
        progress_tracker = None
        if research_id:
            try:
                from services.progress import progress as _pt
                progress_tracker = _pt
            except Exception:
                progress_tracker = None

        # Track outputs from each layer
        outputs = {}

        # Seed experiment context for downstream layers (e.g., Rewrite)
        try:
            unit_id = (research_id or classification.query or "").strip() or "anon"
            variant = (
                experiments.variant_or_default("context_rewrite_prompt", unit_id, default="v1")
                if experiments else "v1"
            )
            outputs["_experiment"] = {"unit_id": unit_id, "context_rewrite_variant": variant}
        except Exception:
            outputs["_experiment"] = {"unit_id": research_id or "anon", "context_rewrite_variant": "v1"}
        total_layers = 6  # W-S-C-I + Rewrite + Optimize

        # Process through Write layer
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="context_engineering",
                message="Processing Write layer - documenting paradigm focus",
                items_done=0,
                items_total=total_layers
            )
        _t0 = datetime.now()
        write_output = await self.write_layer.process(classification)
        write_time = (datetime.now() - _t0).total_seconds()
        outputs["write"] = write_output

        # Rewrite query
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="context_engineering",
                message="Rewriting query for clarity and searchability",
                items_done=1,
                items_total=total_layers
            )
        _tR = datetime.now()
        rewrite_output = await self.rewrite_layer.process(classification, outputs)
        rewrite_time = (datetime.now() - _tR).total_seconds()
        outputs["rewrite"] = rewrite_output

        # Process through Select layer
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="context_engineering",
                message="Selecting search methods and sources",
                items_done=2,
                items_total=total_layers
            )
        _t1 = datetime.now()
        select_output = await self.select_layer.process(classification, outputs)
        select_time = (datetime.now() - _t1).total_seconds()
        outputs["select"] = select_output

        # Optimize terms and queries
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="context_engineering",
                message="Optimizing search terms and query variations",
                items_done=3,
                items_total=total_layers
            )
        _tO = datetime.now()
        optimize_output = await self.optimize_layer.process(classification, outputs)
        optimize_time = (datetime.now() - _tO).total_seconds()
        outputs["optimize"] = optimize_output

        # Process through Compress layer
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="context_engineering",
                message="Compressing information by paradigm priorities",
                items_done=4,
                items_total=total_layers
            )
        _t2 = datetime.now()
        compress_output = await self.compress_layer.process(classification, outputs)
        compress_time = (datetime.now() - _t2).total_seconds()
        outputs["compress"] = compress_output

        # Process through Isolate layer
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="context_engineering",
                message="Isolating key findings extraction patterns",
                items_done=5,
                items_total=total_layers
            )
        _t3 = datetime.now()
        isolate_output = await self.isolate_layer.process(classification, outputs)
        isolate_time = (datetime.now() - _t3).total_seconds()
        outputs["isolate"] = isolate_output

        # Mark complete
        if progress_tracker and research_id:
            await progress_tracker.update_progress(
                research_id,
                phase="context_engineering",
                message="Context engineering complete",
                items_done=total_layers,
                items_total=total_layers
            )

        # Calculate total processing time
        processing_time = (datetime.now() - start_time).total_seconds()

        # Compose refined queries for orchestrator (variations + select queries)
        refined_queries: List[str] = []
        try:
            if optimize_output and isinstance(optimize_output.get("variations"), dict):
                refined_queries.extend(list(optimize_output["variations"].values()))
        except Exception:
            pass
        try:
            for q in getattr(select_output, "search_queries", []) or []:
                txt = q.get("query") if isinstance(q, dict) else None
                if isinstance(txt, str) and txt.strip():
                    refined_queries.append(txt)
        except Exception:
            pass
        # Deduplicate preserving order
        seen = set()
        refined_queries = [q for q in refined_queries if not (q in seen or seen.add(q))]

        # Create final engineered query
        engineered_query = ContextEngineeredQuery(
            original_query=classification.query,
            classification=classification,
            write_output=write_output,
            rewrite_output=rewrite_output,
            select_output=select_output,
            optimize_output=optimize_output,
            compress_output=compress_output,
            isolate_output=isolate_output,
            processing_time=processing_time,
            refined_queries=refined_queries,
            layer_durations={
                "write": write_time,
                "rewrite": rewrite_time,
                "select": select_time,
                "optimize": optimize_time,
                "compress": compress_time,
                "isolate": isolate_time,
            },
        )

        # Store in history
        self.processing_history.append(engineered_query)

        logger.info(f"Context engineering completed in {processing_time:.2f}s")
        self._log_summary(engineered_query)

        return engineered_query

    def _log_summary(self, engineered_query: ContextEngineeredQuery):
        """Log summary of context engineering"""
        eq = engineered_query
        logger.info(
            f"""
Context Engineering Summary:
- Query: {eq.original_query[:50]}...
- Paradigm: {eq.classification.primary_paradigm.value}
- Documentation Focus: {eq.write_output.documentation_focus[:50]}...
- Rewrites: {len((eq.rewrite_output or {}).get('alternatives', []))}
- Optimized Variations: {len(((eq.optimize_output or {}).get('variations') or {}))}
- Search Queries: {len(eq.select_output.search_queries)}
- Compression Ratio: {eq.compress_output.compression_ratio:.0%}
- Token Budget: {eq.compress_output.token_budget}
- Isolation Strategy: {eq.isolate_output.isolation_strategy}
        """
        )

    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the pipeline"""
        if not self.processing_history:
            return {}

        paradigm_counts = {p: 0 for p in HostParadigm}
        total_time = 0.0

        for eq in self.processing_history:
            paradigm_counts[eq.classification.primary_paradigm] += 1
            total_time += eq.processing_time

        return {
            "total_processed": len(self.processing_history),
            "paradigm_distribution": {
                p.value: count for p, count in paradigm_counts.items()
            },
            "average_processing_time": total_time / len(self.processing_history),
            "layer_metrics": {
                "write": len(self.write_layer.processing_history),
                "select": len(self.select_layer.processing_history),
                "compress": len(self.compress_layer.processing_history),
                "isolate": len(self.isolate_layer.processing_history),
            },
        }


# Global instance
context_pipeline = ContextEngineeringPipeline()
