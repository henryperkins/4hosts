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
from models.paradigms import normalize_to_enum
from services.llm_client import llm_client
from services.text_compression import text_compressor
from services.cache import cache_manager
from utils.token_budget import (
    estimate_tokens,
    select_items_within_budget,
)
from utils.injection_hygiene import (
    sanitize_snippet,
    quarantine_note,
    guardrail_instruction,
)

logger = logging.getLogger(__name__)


# ============================================================================
# SHARED PATTERNS AND CONSTANTS
# ============================================================================

STATISTICAL_PATTERNS = {
    "correlation": r"correlat\w+\s+(?:of\s+)?([+-]?\d*\.?\d+)",
    "percentage": r"(\d+(?:\.\d+)?)\s*%",
    "p_value": r"p\s*[<=]\s*(\d*\.?\d+)",
    "sample_size": r"n\s*=\s*(\d+)",
    "confidence": r"(\d+(?:\.\d+)?)\s*%\s*(?:CI|confidence)",
    "effect_size": r"(?:Cohen's\s*d|effect\s*size)\s*=\s*([+-]?\d*\.?\d+)",
}

STRATEGIC_PATTERNS = {
    "market_size": r"\$(\d+(?:\.\d+)?)\s*(?:billion|million|B|M)",
    "growth_rate": r"(\d+(?:\.\d+)?)\s*%\s*(?:growth|CAGR|increase)",
    "market_share": r"(\d+(?:\.\d+)?)\s*%\s*(?:market\s*share|of\s*the\s*market)",
    "roi": r"(?:ROI|return)\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%",
    "cost_savings": r"(?:save|reduce\s*costs?)\s*(?:by\s*)?\$?(\d+(?:\.\d+)?)\s*(?:million|thousand|K|M)?",
    "timeline": r"(\d+)\s*(?:months?|years?|quarters?|weeks?)",
}

PARADIGM_KEYWORDS: Dict[HostParadigm, List[str]] = {
    HostParadigm.DOLORES: [
        "injustice", "oppression", "revolution", "resistance", "corruption",
        "exploitation", "systemic", "power", "expose", "inequality",
        "liberation", "uprising", "defiance", "overthrow", "rebel"
    ],
    HostParadigm.BERNARD: [
        "empirical", "statistical", "data", "analysis", "research",
        "evidence", "methodology", "correlation", "significance", "hypothesis",
        "variable", "quantitative", "systematic", "peer-reviewed", "meta-analysis"
    ],
    HostParadigm.MAEVE: [
        "strategy", "leverage", "opportunity", "competitive", "optimize",
        "roi", "market", "growth", "scale", "efficiency",
        "innovation", "disruption", "advantage", "execution", "performance"
    ],
    HostParadigm.TEDDY: [
        "support", "help", "care", "empathy", "community",
        "healing", "together", "hope", "compassion", "kindness",
        "assistance", "resources", "wellbeing", "comfort", "solidarity"
    ]
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


# ============================================================================
# CORE DATACLASSES
# ============================================================================

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


# ============================================================================
# BASE GENERATOR CLASS
# ============================================================================

class BaseAnswerGenerator:
    """Base class for all paradigm-specific generators"""
    
    def __init__(self, paradigm: str):
        self.paradigm = paradigm
        self.citation_counter = 0
        self.citations = {}
    
    def create_citation(self, source: Dict[str, Any], fact_type: str = "reference") -> Citation:
        """Create a citation from a source"""
        self.citation_counter += 1
        citation_id = f"cite_{self.citation_counter:03d}"
        
        citation = Citation(
            id=citation_id,
            source_title=source.get("title", ""),
            source_url=source.get("url", ""),
            domain=source.get("domain", ""),
            snippet=source.get("snippet", ""),
            credibility_score=float(source.get("credibility_score", 0.5)),
            fact_type=fact_type,
            metadata=source.get("metadata", {}),
            timestamp=source.get("published_date")
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


# ============================================================================
# PARADIGM-SPECIFIC GENERATORS
# ============================================================================

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
        
        summary = sections[0].content[:300] + "..." if sections else ""
        
        return GeneratedAnswer(
            research_id=context.metadata.get("research_id", "unknown"),
            query=context.query,
            paradigm=self.paradigm,
            summary=summary,
            sections=sections,
            action_items=self._generate_action_items(context),
            citations=self.citations,
            confidence_score=0.8,
            synthesis_quality=0.85,
            generation_time=(datetime.now() - start_time).total_seconds(),
            metadata={"tone": "investigative", "focus": "exposing injustice"}
        )
    
    async def _generate_section(self, context: SynthesisContext, section_def: Dict[str, Any]) -> AnswerSection:
        """Generate a single section"""
        # Filter relevant results
        relevant_results = [r for r in context.search_results[:5]]
        
        # Create citations
        citation_ids = []
        for result in relevant_results:
            citation = self.create_citation(result, "evidence")
            citation_ids.append(citation.id)
        
        # Generate content with LLM or fallback
        try:
            # Isolation-only support: summarize findings
            iso_lines = []
            try:
                if isinstance(context.context_engineering, dict):
                    for m in (context.context_engineering.get("isolated_findings", {}).get("matches", []) or [])[:5]:
                        dom = m.get("domain", "")
                        for frag in (m.get("fragments", []) or [])[:1]:
                            iso_lines.append(f"- [{dom}] {frag}")
            except Exception:
                pass
            iso_block = "\n".join(iso_lines) if iso_lines else "(no isolated findings)"

            prompt = f"""
            Write the "{section_def['title']}" section focusing on: {section_def['focus']}
            Query: {context.query}
            Use passionate, urgent language that exposes injustice and calls for change.
            Use only these Isolated Findings as evidence:
            {iso_block}
            Length: {int(2000 * section_def['weight'])} words
            """
            
            content = await llm_client.generate_paradigm_content(
                prompt=prompt,
                paradigm="dolores",
                max_tokens=int(3000 * section_def['weight']),
                temperature=0.7
            )
        except Exception as e:
            if os.getenv("LLM_STRICT", "0") == "1":
                raise
            logger.warning(f"LLM generation failed: {e}, using fallback")
            content = self._generate_fallback_content(section_def, relevant_results)
        
        return AnswerSection(
            title=section_def['title'],
            paradigm=self.paradigm,
            content=content,
            confidence=0.75,
            citations=citation_ids,
            word_count=len(content.split()),
            key_insights=self._extract_insights(content),
            metadata={"section_weight": section_def['weight']}
        )
    
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
        
        # Extract statistical insights from search results
        statistical_insights = self._extract_statistical_insights(context.search_results)
        
        # Perform meta-analysis if possible
        meta_analysis = self._perform_meta_analysis(context.search_results)
        
        # Generate sections
        sections = []
        for section_def in self.get_section_structure():
            section = await self._generate_analytical_section(
                context, section_def, statistical_insights, meta_analysis
            )
            sections.append(section)
        
        summary = self._generate_analytical_summary(sections, statistical_insights)
        
        return GeneratedAnswer(
            research_id=context.metadata.get("research_id", "unknown"),
            query=context.query,
            paradigm=self.paradigm,
            summary=summary,
            sections=sections,
            action_items=self._generate_research_action_items(statistical_insights),
            citations=self.citations,
            confidence_score=self._calculate_analytical_confidence(statistical_insights),
            synthesis_quality=0.9,
            generation_time=(datetime.now() - start_time).total_seconds(),
            metadata={
                "statistical_insights": len(statistical_insights),
                "meta_analysis_performed": meta_analysis is not None,
                "peer_reviewed_sources": self._count_peer_reviewed(context.search_results)
            }
        )
    
    def _extract_statistical_insights(self, search_results: List[Dict[str, Any]]) -> List[StatisticalInsight]:
        """Extract statistical insights from search results"""
        insights = []
        
        for result in search_results:
            text = f"{result.get('title', '')} {result.get('snippet', '')}"
            
            # Extract correlations
            for match in re.finditer(STATISTICAL_PATTERNS["correlation"], text):
                insights.append(StatisticalInsight(
                    metric="correlation",
                    value=float(match.group(1)),
                    unit="r",
                    context=text[max(0, match.start()-50):match.end()+50]
                ))
            
            # Extract p-values
            for match in re.finditer(STATISTICAL_PATTERNS["p_value"], text):
                insights.append(StatisticalInsight(
                    metric="p_value",
                    value=float(match.group(1)),
                    unit="",
                    context=text[max(0, match.start()-50):match.end()+50]
                ))
            
            # Extract sample sizes
            for match in re.finditer(STATISTICAL_PATTERNS["sample_size"], text):
                insights.append(StatisticalInsight(
                    metric="sample_size",
                    value=float(match.group(1)),
                    unit="n",
                    context=text[max(0, match.start()-50):match.end()+50]
                ))
            
            # Extract effect sizes
            for match in re.finditer(STATISTICAL_PATTERNS["effect_size"], text):
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
            effect_matches = re.findall(STATISTICAL_PATTERNS["effect_size"], text)
            if effect_matches:
                effect_sizes.extend([float(m) for m in effect_matches])
            
            # Extract sample sizes
            sample_matches = re.findall(STATISTICAL_PATTERNS["sample_size"], text)
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
        # Filter relevant results
        relevant_results = [r for r in context.search_results[:5]]
        
        # Create citations
        citation_ids = []
        for result in relevant_results:
            citation = self.create_citation(result, "empirical")
            citation_ids.append(citation.id)
        
        # Generate content
        try:
            insights_summary = self._format_statistical_insights(statistical_insights[:5])
            # Isolation-only support: include extracted findings if present
            isolated = {}
            try:
                if isinstance(context.context_engineering, dict):
                    isolated = context.context_engineering.get("isolated_findings", {}) or {}
            except Exception:
                isolated = {}
            iso_lines = []
            try:
                for m in (isolated.get("matches", []) or [])[:5]:
                    dom = m.get("domain", "")
                    for frag in (m.get("fragments", []) or [])[:1]:
                        iso_lines.append(f"- [{dom}] {frag}")
            except Exception:
                pass
            iso_block = "\n".join(iso_lines) if iso_lines else "(no isolated findings)"
            
            prompt = f"""
            Write the "{section_def['title']}" section focusing on: {section_def['focus']}
            Query: {context.query}

            Statistical insights available:
            {insights_summary}

            Isolated Findings (SSOTA Isolate Layer - use only these as evidence):
            {iso_block}

            {f"Meta-analysis results: {meta_analysis}" if meta_analysis else ""}

            Requirements:
            - Use precise scientific language
            - Include effect sizes, confidence intervals, and p-values where available
            - Distinguish correlation from causation
            - Acknowledge limitations
            - STRICT: Do not introduce claims not supported by the Isolated Findings above

            Length: {int(2000 * section_def['weight'])} words
            """
            
            content = await llm_client.generate_paradigm_content(
                prompt=prompt,
                paradigm="bernard",
                max_tokens=int(3000 * section_def['weight']),
                temperature=0.3
            )
        except Exception as e:
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
        
        # Extract strategic insights
        strategic_insights = self._extract_strategic_insights(context.search_results)
        
        # Generate SWOT analysis
        swot_analysis = self._generate_swot_analysis(context.query, context.search_results)
        
        # Generate sections
        sections = []
        for section_def in self.get_section_structure():
            section = await self._generate_strategic_section(
                context, section_def, strategic_insights, swot_analysis
            )
            sections.append(section)
        
        # Generate strategic recommendations
        recommendations = self._generate_strategic_recommendations(
            context.query, strategic_insights, swot_analysis
        )
        
        summary = self._generate_strategic_summary(sections, strategic_insights)
        
        return GeneratedAnswer(
            research_id=context.metadata.get("research_id", "unknown"),
            query=context.query,
            paradigm=self.paradigm,
            summary=summary,
            sections=sections,
            action_items=self._format_recommendations_as_actions(recommendations),
            citations=self.citations,
            confidence_score=0.85,
            synthesis_quality=0.88,
            generation_time=(datetime.now() - start_time).total_seconds(),
            metadata={
                "strategic_insights": len(strategic_insights),
                "swot_completed": swot_analysis is not None,
                "recommendations": len(recommendations)
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
        # Filter relevant results
        relevant_results = [r for r in context.search_results[:5]]
        
        # Create citations
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
                if isinstance(context.context_engineering, dict):
                    for m in (context.context_engineering.get("isolated_findings", {}).get("matches", []) or [])[:5]:
                        dom = m.get("domain", "")
                        for frag in (m.get("fragments", []) or [])[:1]:
                            iso_lines.append(f"- [{dom}] {frag}")
            except Exception:
                pass
            iso_block = "\n".join(iso_lines) if iso_lines else "(no isolated findings)"

            prompt = f"""
            Write the "{section_def['title']}" section focusing on: {section_def['focus']}
            Query: {context.query}

            SWOT Analysis:
            {swot_summary}

            Use only these Isolated Findings as domain evidence:
            {iso_block}

            Requirements:
            - Focus on actionable strategies and concrete recommendations
            - Include ROI considerations and resource implications
            - Emphasize competitive advantages and market opportunities
            - Use decisive, action-oriented language
            
            Length: {int(2000 * section_def['weight'])} words
            """
            
            content = await llm_client.generate_paradigm_content(
                prompt=prompt,
                paradigm="maeve",
                max_tokens=int(3000 * section_def['weight']),
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
        
        summary = self._generate_supportive_summary(sections)
        
        return GeneratedAnswer(
            research_id=context.metadata.get("research_id", "unknown"),
            query=context.query,
            paradigm=self.paradigm,
            summary=summary,
            sections=sections,
            action_items=self._generate_supportive_actions(context),
            citations=self.citations,
            confidence_score=0.82,
            synthesis_quality=0.86,
            generation_time=(datetime.now() - start_time).total_seconds(),
            metadata={"tone": "supportive", "focus": "community care"}
        )
    
    async def _generate_supportive_section(
        self,
        context: SynthesisContext,
        section_def: Dict[str, Any]
    ) -> AnswerSection:
        """Generate supportive section"""
        # Filter relevant results
        relevant_results = [r for r in context.search_results[:5]]
        
        # Create citations
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
                            iso_lines.append(f"- [{dom}] {frag}")
            except Exception:
                pass
            iso_block = "\n".join(iso_lines) if iso_lines else "(no isolated findings)"

            prompt = f"""
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
            
            Length: {int(2000 * section_def['weight'])} words
            """
            
            content = await llm_client.generate_paradigm_content(
                prompt=prompt,
                paradigm="teddy",
                max_tokens=int(3000 * section_def['weight']),
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
# ============================================================================

class AnswerGenerationOrchestrator:
    """Main orchestrator for answer generation"""
    
    def __init__(self):
        self.generators = {
            "dolores": DoloresAnswerGenerator(),
            "bernard": BernardAnswerGenerator(),
            "maeve": MaeveAnswerGenerator(),
            "teddy": TeddyAnswerGenerator()
        }
        logger.info("Answer Generation Orchestrator initialized")
    
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
        if paradigm not in self.generators:
            logger.error(f"Unknown paradigm: {paradigm}")
            paradigm = "bernard"  # Default to analytical
        
        # Create synthesis context
        context = SynthesisContext(
            query=query,
            paradigm=paradigm,
            search_results=search_results,
            context_engineering=context_engineering or {},
            max_length=(options or {}).get("max_length", 2000),
            include_citations=(options or {}).get("include_citations", True),
            tone=(options or {}).get("tone", "professional"),
            metadata=options or {},
        )
        
        # Generate answer with progress callbacks if research_id available
        generator = self.generators[paradigm]

        research_id = (options or {}).get("research_id") if options else None
        total_sections = len(generator.get_section_structure())

        if research_id and total_sections:
            # Late import to avoid circular
            try:
                from services.websocket_service import progress_tracker as _pt

                async def _report(idx: int):
                    if _pt:
                        await _pt.report_synthesis_progress(research_id, idx, total_sections)

                # Wrap generator._generate_section
                orig_generate = generator._generate_section  # type: ignore[attr-defined]

                async def _wrapped(sec_self, ctx, sec_def):  # noqa: ANN001
                    result = await orig_generate(ctx, sec_def)  # type: ignore[misc]
                    # section_index inferred by len progress
                    await _report(len(sec_self.citations) // 1)  # approx sections processed
                    return result

                # Monkeypatch for duration of call
                setattr(generator, "_generate_section", _wrapped)  # type: ignore[attr-defined]
                try:
                    answer = await generator.generate_answer(context)
                finally:
                    setattr(generator, "_generate_section", orig_generate)  # restore
            except Exception:
                answer = await generator.generate_answer(context)
        else:
            answer = await generator.generate_answer(context)
        
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
    
    # Core types
    'StatisticalInsight',
    'StrategicRecommendation',
]

# Create aliases for enhanced generators (for backward compatibility)
EnhancedBernardAnswerGenerator = BernardAnswerGenerator
EnhancedMaeveAnswerGenerator = MaeveAnswerGenerator
