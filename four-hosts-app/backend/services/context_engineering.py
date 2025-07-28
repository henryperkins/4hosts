"""
Four Hosts Context Engineering Pipeline
W-S-C-I (Write-Select-Compress-Isolate) implementation
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
from abc import ABC, abstractmethod

# Import from classification engine
from .classification_engine import HostParadigm, ClassificationResult, QueryFeatures

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


@dataclass
class CompressLayerOutput:
    """Output from Compress layer processing"""

    paradigm: HostParadigm
    compression_ratio: float
    compression_strategy: str
    priority_elements: List[str]
    removed_elements: List[str]
    token_budget: int


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
    timestamp: datetime = field(default_factory=datetime.now)


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
        previous_outputs: Dict[str, Any] = None,
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
        previous_outputs: Dict[str, Any] = None,
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
        """Extract themes from the query itself"""
        themes = []

        # Use entities as themes
        themes.extend(classification.features.entities)

        # Extract key concepts from tokens
        important_tokens = [
            t
            for t in classification.features.tokens
            if len(t) > 5 and t not in ["should", "would", "could", "about", "through"]
        ]
        themes.extend(important_tokens[:5])

        return themes

    def _generate_search_priorities(
        self, classification: ClassificationResult, base_priorities: List[str]
    ) -> List[str]:
        """Generate search priorities based on query context"""
        priorities = base_priorities.copy()

        # Add domain-specific priorities
        if classification.features.domain:
            domain_priorities = {
                "business": ["industry reports", "competitor analysis"],
                "healthcare": ["clinical studies", "patient outcomes"],
                "education": ["pedagogical research", "learning outcomes"],
                "technology": ["technical documentation", "benchmarks"],
                "social_justice": ["advocacy reports", "policy analysis"],
            }

            if classification.features.domain in domain_priorities:
                priorities.extend(domain_priorities[classification.features.domain])

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
        previous_outputs: Dict[str, Any] = None,
    ) -> SelectLayerOutput:
        """Process through Select layer"""
        start_time = datetime.now()

        paradigm = classification.primary_paradigm
        strategy = self.selection_strategies[paradigm]
        write_output = previous_outputs.get("write") if previous_outputs else None

        # Generate enhanced search queries
        search_queries = self._generate_search_queries(
            classification, strategy, write_output
        )

        # Add secondary paradigm queries if significant
        if (
            classification.secondary_paradigm
            and classification.distribution[classification.secondary_paradigm] > 0.25
        ):
            secondary_queries = self._generate_secondary_queries(
                classification, classification.secondary_paradigm
            )
            search_queries.extend(secondary_queries[:3])  # Add top 3

        output = SelectLayerOutput(
            paradigm=paradigm,
            search_queries=search_queries,
            source_preferences=strategy["source_types"],
            exclusion_filters=strategy["exclude"],
            tool_selections=strategy["tools"],
            max_sources=strategy["max_sources"],
        )

        # Log processing
        duration = (datetime.now() - start_time).total_seconds()
        self.log_processing(classification, output, duration)

        logger.info(f"Select layer generated {len(search_queries)} search queries")

        return output

    def _generate_search_queries(
        self,
        classification: ClassificationResult,
        strategy: Dict[str, Any],
        write_output: Optional[WriteLayerOutput],
    ) -> List[Dict[str, Any]]:
        """Generate paradigm-specific search queries"""
        base_query = classification.query
        queries = []

        # Base query
        queries.append(
            {
                "query": base_query,
                "type": "original",
                "weight": 1.0,
                "source_filter": None,
            }
        )

        # Modified queries with paradigm terms
        for modifier in strategy["query_modifiers"][:3]:
            modified = f"{base_query} {modifier}"
            queries.append(
                {
                    "query": modified,
                    "type": "paradigm_modified",
                    "weight": 0.8,
                    "source_filter": (
                        strategy["source_types"][0]
                        if strategy["source_types"]
                        else None
                    ),
                }
            )

        # Theme-based queries from Write layer
        if write_output and write_output.key_themes:
            for theme in write_output.key_themes[:2]:
                theme_query = f"{theme} {base_query}"
                queries.append(
                    {
                        "query": theme_query,
                        "type": "theme_enhanced",
                        "weight": 0.7,
                        "source_filter": None,
                    }
                )

        # Entity-focused queries
        for entity in classification.features.entities[:2]:
            if classification.features.tokens:
                entity_query = f'"{entity}" {classification.features.tokens[0]}'
                queries.append(
                    {
                        "query": entity_query,
                        "type": "entity_focused",
                        "weight": 0.6,
                        "source_filter": None,
                    }
                )

        return queries[:10]  # Limit to 10 queries

    def _generate_secondary_queries(
        self, classification: ClassificationResult, secondary_paradigm: HostParadigm
    ) -> List[Dict[str, Any]]:
        """Generate queries for secondary paradigm"""
        strategy = self.selection_strategies[secondary_paradigm]

        queries = []
        for modifier in strategy["query_modifiers"][:1]:
            queries.append(
                {
                    "query": f"{classification.query} {modifier}",
                    "type": "secondary_paradigm",
                    "weight": 0.5,
                    "source_filter": strategy["source_types"][0],
                }
            )

        return queries


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
        previous_outputs: Dict[str, Any] = None,
    ) -> CompressLayerOutput:
        """Process through Compress layer"""
        start_time = datetime.now()

        paradigm = classification.primary_paradigm
        strategy = self.compression_strategies[paradigm]

        # Calculate token budget based on query complexity
        base_tokens = 2000  # Base token budget
        complexity_multiplier = 1 + classification.features.complexity_score
        token_budget = int(base_tokens * complexity_multiplier * strategy["ratio"])

        output = CompressLayerOutput(
            paradigm=paradigm,
            compression_ratio=strategy["ratio"],
            compression_strategy=strategy["strategy"],
            priority_elements=strategy["priorities"],
            removed_elements=strategy["remove"],
            token_budget=token_budget,
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
        previous_outputs: Dict[str, Any] = None,
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


# --- Context Engineering Pipeline ---


class ContextEngineeringPipeline:
    """Main pipeline orchestrating all W-S-C-I layers"""

    def __init__(self):
        self.write_layer = WriteLayer()
        self.select_layer = SelectLayer()
        self.compress_layer = CompressLayer()
        self.isolate_layer = IsolateLayer()
        self.processing_history = []

    async def process_query(
        self, classification: ClassificationResult
    ) -> ContextEngineeredQuery:
        """Process classification through all context layers"""
        start_time = datetime.now()

        logger.info(
            f"Starting context engineering for paradigm: {classification.primary_paradigm.value}"
        )

        # Track outputs from each layer
        outputs = {}

        # Process through Write layer
        write_output = await self.write_layer.process(classification)
        outputs["write"] = write_output

        # Process through Select layer
        select_output = await self.select_layer.process(classification, outputs)
        outputs["select"] = select_output

        # Process through Compress layer
        compress_output = await self.compress_layer.process(classification, outputs)
        outputs["compress"] = compress_output

        # Process through Isolate layer
        isolate_output = await self.isolate_layer.process(classification, outputs)
        outputs["isolate"] = isolate_output

        # Calculate total processing time
        processing_time = (datetime.now() - start_time).total_seconds()

        # Create final engineered query
        engineered_query = ContextEngineeredQuery(
            original_query=classification.query,
            classification=classification,
            write_output=write_output,
            select_output=select_output,
            compress_output=compress_output,
            isolate_output=isolate_output,
            processing_time=processing_time,
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
