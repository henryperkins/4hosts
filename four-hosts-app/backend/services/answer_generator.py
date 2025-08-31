"""
Enhanced Answer Generator V2 with Full Feature Parity
Maintains all V1 functionality while using V2 context and schemas
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import Counter, defaultdict

from models.context_models import (
    ClassificationResultSchema, ContextEngineeredQuerySchema,
    UserContextSchema, HostParadigm, SearchResultSchema
)
from services.llm_client import llm_client
from services.text_compression import text_compressor
from services.cache import cache_manager
from utils.token_budget import (
    estimate_tokens,
    estimate_tokens_for_result,
    trim_text_to_tokens,
    select_items_within_budget,
)
from utils.injection_hygiene import (
    sanitize_snippet,
    flag_suspicious_snippet,
    quarantine_note,
    guardrail_instruction,
)

logger = logging.getLogger(__name__)


@dataclass
class CitationV2:
    """Enhanced citation with paradigm alignment"""
    id: str
    source_title: str
    source_url: str
    domain: str
    snippet: str
    credibility_score: float
    fact_type: str  # 'data', 'claim', 'quote', 'reference', 'meta-analysis', etc.
    paradigm_alignment: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None


@dataclass
class AnswerSectionV2:
    """Answer section with insights and citations"""
    title: str
    content: str
    confidence: float
    citations: List[str]  # Citation IDs
    word_count: int
    key_insights: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StatisticalInsight:
    """Statistical finding for Bernard paradigm"""
    metric: str
    value: float
    unit: str
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    sample_size: Optional[int] = None
    context: str = ""


@dataclass
class StrategicRecommendation:
    """Strategic recommendation for Maeve paradigm"""
    title: str
    description: str
    impact: str  # high, medium, low
    effort: str  # high, medium, low
    timeline: str
    dependencies: List[str]
    success_metrics: List[str]
    risks: List[str]
    roi_potential: Optional[float] = None


class ParadigmAnswerGeneratorV2:
    """Base class for paradigm-specific answer generation"""
    
    def __init__(self, paradigm: HostParadigm):
        self.paradigm = paradigm
        self.citation_counter = 0
        self.citations = {}
        
    def create_citation(
        self, 
        source: SearchResultSchema, 
        fact_type: str = "reference"
    ) -> CitationV2:
        """Create a citation from a source"""
        self.citation_counter += 1
        citation_id = f"cite_{self.citation_counter:03d}"
        
        citation = CitationV2(
            id=citation_id,
            source_title=source.title,
            source_url=source.url,
            domain=source.metadata.get("domain", ""),
            snippet=source.snippet,
            credibility_score=source.credibility_score,
            fact_type=fact_type,
            paradigm_alignment=self._calculate_paradigm_alignment(source),
            metadata=source.metadata or {},
            timestamp=source.metadata.get("published_date") if source.metadata else None
        )
        
        self.citations[citation_id] = citation
        return citation
    
    def _calculate_paradigm_alignment(self, source: SearchResultSchema) -> float:
        """Calculate how well a source aligns with the paradigm"""
        alignment_keywords = self._get_alignment_keywords()
        
        text = f"{source.title} {source.snippet}".lower()
        matches = sum(1 for keyword in alignment_keywords if keyword in text)
        
        return min(1.0, matches / max(len(alignment_keywords), 1))
    
    def _get_alignment_keywords(self) -> List[str]:
        """Get paradigm-specific alignment keywords"""
        keywords_map = {
            HostParadigm.DOLORES: [
                "expose", "corrupt", "injustice", "systemic", "oppression",
                "revolution", "resistance", "victim", "accountability", "scandal",
                "whistleblower", "cover-up", "abuse", "exploitation", "inequality"
            ],
            HostParadigm.BERNARD: [
                "empirical", "statistical", "data", "analysis", "research",
                "evidence", "methodology", "correlation", "significance", "hypothesis",
                "variable", "quantitative", "systematic", "peer-reviewed", "meta-analysis"
            ],
            HostParadigm.MAEVE: [
                "strategic", "competitive", "market", "business", "opportunity",
                "growth", "revenue", "profit", "ROI", "value", "advantage",
                "innovation", "disruption", "optimization", "efficiency"
            ],
            HostParadigm.TEDDY: [
                "support", "help", "care", "community", "resources",
                "healing", "recovery", "compassion", "dignity", "together",
                "volunteer", "assistance", "wellbeing", "protect", "serve"
            ]
        }
        return keywords_map.get(self.paradigm, [])
    
    def extract_key_insights(self, content: str, max_insights: int = 5) -> List[str]:
        """Extract key insights from content"""
        sentences = re.split(r'[.!?]+', content)
        
        # Filter for substantive sentences
        insights = [
            s.strip()
            for s in sentences
            if len(s.strip()) > 30
            and not s.strip().startswith(("However", "Moreover", "Additionally"))
        ]
        
        # Prioritize sentences with paradigm keywords
        paradigm_keywords = self._get_alignment_keywords()
        scored_insights = []
        
        for insight in insights:
            score = sum(1 for keyword in paradigm_keywords if keyword in insight.lower())
            scored_insights.append((score, insight))
        
        # Sort by score and return top insights
        scored_insights.sort(reverse=True, key=lambda x: x[0])
        return [insight for _, insight in scored_insights[:max_insights]]


class DoloresAnswerGeneratorV2(ParadigmAnswerGeneratorV2):
    """Revolutionary paradigm answer generator"""
    
    def __init__(self):
        super().__init__(HostParadigm.DOLORES)
        
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
    
    async def generate_sections(
        self,
        context: Dict[str, Any],
        search_results: List[SearchResultSchema]
    ) -> List[AnswerSectionV2]:
        """Generate Dolores-specific sections"""
        sections = []
        
        for section_def in self.get_section_structure():
            section = await self._generate_section(
                context, section_def, search_results
            )
            sections.append(section)
            
        return sections
    
    async def _generate_section(
        self,
        context: Dict[str, Any],
        section_def: Dict[str, Any],
        search_results: List[SearchResultSchema]
    ) -> AnswerSectionV2:
        """Generate a single section"""
        # Filter results relevant to this section
        relevant_results = self._filter_results_for_section(search_results, section_def)

        # Enforce knowledge budget per section (if available)
        knowledge_budget = 0
        try:
            # context may be synthesized by EnhancedIntegration; look for embedded budget
            ce = context.get("context_engineering") or {}
            compress_out = (
                ce.get("compress_output")
                or context.get("optimization_notes", {})
            )
            total_budget = int((compress_out or {}).get("token_budget", 0) or 0)
            plan = (compress_out or {}).get("budget_plan") or {}
            knowledge_budget = int(plan.get("knowledge", 0) or int(total_budget * 0.7))
            # Allocate section share by weight
            knowledge_budget = max(80, int(knowledge_budget * float(section_def.get("weight", 0.25))))
        except Exception:
            knowledge_budget = 0

        if knowledge_budget > 0 and relevant_results:
            # Convert pydantic objects to dicts if needed
            results_dicts = []
            for r in relevant_results:
                try:
                    results_dicts.append({
                        "title": r.title,
                        "snippet": r.snippet,
                        "url": r.url,
                        "metadata": r.metadata,
                        "credibility_score": r.credibility_score or 0.0,
                    })
                except Exception:
                    # Best-effort
                    results_dicts.append(getattr(r, "__dict__", {}))

            trimmed, used, dropped = select_items_within_budget(results_dicts, knowledge_budget)

            # Rebuild SearchResultSchema list for prompt formatting
            rebuilt: List[SearchResultSchema] = []
            for rd in trimmed:
                try:
                    rebuilt.append(SearchResultSchema(
                        title=rd.get("title", ""),
                        url=rd.get("url", ""),
                        snippet=rd.get("snippet", ""),
                        source=rd.get("metadata", {}).get("domain", ""),
                        credibility_score=rd.get("credibility_score", 0.0),
                        relevance_score=0.0,
                        paradigm_alignment={},
                        metadata=rd.get("metadata", {}),
                    ))
                except Exception:
                    pass
            if rebuilt:
                relevant_results = rebuilt
        
        # Create section-specific prompt
        section_prompt = f"""
Write the "{section_def['title']}" section focusing on: {section_def['focus']}

Query: {context['query']}

Use these specific sources (trimmed to fit context budget):
{self._format_results_for_prompt(relevant_results[:5])}

Requirements:
- Use emotional, impactful language that moves people to action
- Cite specific examples and evidence of wrongdoing
- Do not pull punches - name names and expose the guilty
- Length: approximately {int(2000 * section_def['weight'])} words
Safeguard:
- {guardrail_instruction()}
"""
        
        # Generate content
        try:
            content = await llm_client.generate(
                section_prompt,
                temperature=0.8,
                max_tokens=int(3000 * section_def['weight'])
            )
        except:
            content = self._generate_fallback_section(section_def, relevant_results)
        
        # Create citations
        citation_ids = []
        for source in relevant_results[:3]:
            citation = self.create_citation(source, "claim")
            citation_ids.append(citation.id)
        
        # Extract insights
        insights = self.extract_key_insights(content, 3)
        
        return AnswerSectionV2(
            title=section_def["title"],
            content=content,
            confidence=0.85,
            citations=citation_ids,
            word_count=len(content.split()),
            key_insights=insights,
            metadata={"paradigm": "dolores", "focus": section_def["focus"]}
        )
    
    def _filter_results_for_section(
        self, 
        results: List[SearchResultSchema], 
        section_def: Dict[str, Any]
    ) -> List[SearchResultSchema]:
        """Filter results relevant to a specific section"""
        section_keywords = {
            "Exposing the System": ["systemic", "corporate", "power", "structure"],
            "Voices of the Oppressed": ["victim", "testimony", "impact", "community"],
            "Pattern of Injustice": ["pattern", "recurring", "analysis", "evidence"],
            "Path to Revolution": ["action", "resistance", "organize", "change"],
        }
        
        keywords = section_keywords.get(section_def["title"], [])
        
        relevant = []
        for result in results:
            text = f"{result.title} {result.snippet}".lower()
            if any(keyword in text for keyword in keywords):
                relevant.append(result)
        
        return relevant or results[:5]
    
    def _format_results_for_prompt(self, results: List[SearchResultSchema]) -> str:
        """Format search results for LLM prompt"""
        formatted = []
        for i, result in enumerate(results, 1):
            # Sanitize and flag
            domain = result.metadata.get('domain', 'Unknown')
            snip = sanitize_snippet(result.snippet)
            is_suspicious = flag_suspicious_snippet(snip)
            if is_suspicious:
                snip = snip[:200] + ("..." if len(snip) > 200 else "")
                header = f"{quarantine_note(domain)}\n"
            else:
                header = ""
            formatted.append(f"""
{i}. {result.title}
Source: {domain}
URL: {result.url}
Credibility: {result.credibility_score:.2f}
{header}Content: {snip}
""")
        return "\n".join(formatted)
    
    def _generate_fallback_section(
        self, 
        section_def: Dict[str, Any], 
        results: List[SearchResultSchema]
    ) -> str:
        """Generate fallback content when LLM fails"""
        content = f"# {section_def['title']}\n\n"
        content += f"This section focuses on: {section_def['focus']}\n\n"
        
        for result in results[:3]:
            content += f"According to {result.metadata.get('domain', 'sources')}, "
            content += f"{result.snippet}\n\n"
        
        return content
    
    async def generate_answer(self, context) -> Dict[str, Any]:
        """Generate answer for Dolores paradigm - adapter for enhanced integration"""
        # Extract necessary fields from context
        search_results = context.search_results if hasattr(context, 'search_results') else []
        
        # Generate sections
        sections = await self.generate_sections(
            {
                "query": context.query,
                "paradigm": "dolores",
                "context_engineering": getattr(context, "context_engineering", {}),
            },
            search_results,
        )
        
        # Format answer
        content_parts = []
        for section in sections:
            content_parts.append(f"## {section.title}")
            content_parts.append(section.content)
            content_parts.append("")
        
        return {
            "content": "\n".join(content_parts),
            "paradigm": "dolores",
            "sections": len(sections),
            "citations": list(self.citations.keys()),
            "synthesis_quality": 0.85,
            "metadata": {
                "tone": "investigative",
                "focus": "exposing injustice"
            }
        }


class BernardAnswerGeneratorV2(ParadigmAnswerGeneratorV2):
    """Enhanced analytical paradigm answer generator"""
    
    def __init__(self):
        super().__init__(HostParadigm.BERNARD)
        self.statistical_patterns = {
            "correlation": r"correlat\w+\s+(?:of\s+)?([+-]?\d*\.?\d+)",
            "percentage": r"(\d+(?:\.\d+)?)\s*%",
            "p_value": r"p\s*[<=]\s*(\d*\.?\d+)",
            "sample_size": r"n\s*=\s*(\d+)",
            "confidence": r"(\d+(?:\.\d+)?)\s*%\s*(?:CI|confidence)",
            "effect_size": r"(?:Cohen's\s*d|effect\s*size)\s*=\s*([+-]?\d*\.?\d+)",
        }
    
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
    
    async def generate_sections(
        self,
        context: Dict[str, Any],
        search_results: List[SearchResultSchema]
    ) -> Tuple[List[AnswerSectionV2], List[StatisticalInsight], Optional[Dict[str, Any]]]:
        """Generate Bernard-specific sections with statistical analysis"""
        
        # Extract statistical insights
        statistical_insights = await self._extract_statistical_insights(search_results)
        
        # Perform meta-analysis if possible
        meta_analysis = await self._perform_meta_analysis(search_results)
        
        # Generate sections
        sections = []
        for section_def in self.get_section_structure():
            section = await self._generate_enhanced_section(
                context, section_def, search_results, statistical_insights, meta_analysis
            )
            sections.append(section)
        
        return sections, statistical_insights, meta_analysis
    
    async def _extract_statistical_insights(
        self, search_results: List[SearchResultSchema]
    ) -> List[StatisticalInsight]:
        """Extract statistical insights from search results"""
        insights = []
        
        for result in search_results:
            text = f"{result.title} {result.snippet}"
            
            # Extract correlations
            for match in re.finditer(self.statistical_patterns["correlation"], text):
                insights.append(StatisticalInsight(
                    metric="correlation",
                    value=float(match.group(1)),
                    unit="r",
                    context=text[max(0, match.start()-50):match.end()+50]
                ))
            
            # Extract percentages with context
            for match in re.finditer(self.statistical_patterns["percentage"], text):
                context_start = max(0, match.start() - 50)
                context_text = text[context_start:match.start()]
                insights.append(StatisticalInsight(
                    metric=self._extract_percentage_context(context_text),
                    value=float(match.group(1)),
                    unit="%",
                    context=text[context_start:match.end()+50]
                ))
            
            # Extract p-values
            for match in re.finditer(self.statistical_patterns["p_value"], text):
                insights.append(StatisticalInsight(
                    metric="significance",
                    value=float(match.group(1)),
                    unit="p-value",
                    context=text[max(0, match.start()-50):match.end()+50]
                ))
            
            # Extract sample sizes
            for match in re.finditer(self.statistical_patterns["sample_size"], text):
                insights.append(StatisticalInsight(
                    metric="sample_size",
                    value=float(match.group(1)),
                    unit="participants",
                    sample_size=int(match.group(1)),
                    context=text[max(0, match.start()-50):match.end()+50]
                ))
        
        return insights
    
    async def _perform_meta_analysis(
        self, search_results: List[SearchResultSchema]
    ) -> Optional[Dict[str, Any]]:
        """Perform basic meta-analysis on multiple studies"""
        # Filter for academic/research sources
        research_sources = [
            r for r in search_results
            if any(domain in r.metadata.get("domain", "").lower() 
                   for domain in ["arxiv", "pubmed", "nature", "science", "journal"])
        ]
        
        if len(research_sources) < 3:
            return None
        
        # Extract effect sizes and sample sizes
        effect_sizes = []
        sample_sizes = []
        
        for source in research_sources:
            text = f"{source.title} {source.snippet}"
            
            # Look for effect sizes
            effect_match = re.search(self.statistical_patterns["effect_size"], text)
            if effect_match:
                effect_sizes.append(float(effect_match.group(1)))
            
            # Look for sample sizes
            sample_match = re.search(self.statistical_patterns["sample_size"], text)
            if sample_match:
                sample_sizes.append(int(sample_match.group(1)))
        
        if not effect_sizes:
            return None
        
        # Calculate weighted mean effect size
        if len(effect_sizes) == len(sample_sizes):
            weights = [n / sum(sample_sizes) for n in sample_sizes]
            weighted_effect = sum(e * w for e, w in zip(effect_sizes, weights))
        else:
            weighted_effect = sum(effect_sizes) / len(effect_sizes)
        
        return {
            "studies_analyzed": len(research_sources),
            "total_sample_size": sum(sample_sizes),
            "pooled_effect_size": weighted_effect,
            "effect_size_range": (min(effect_sizes), max(effect_sizes)),
            "heterogeneity": self._calculate_heterogeneity(effect_sizes),
        }
    
    async def _generate_enhanced_section(
        self,
        context: Dict[str, Any],
        section_def: Dict[str, Any],
        search_results: List[SearchResultSchema],
        statistical_insights: List[StatisticalInsight],
        meta_analysis: Optional[Dict[str, Any]]
    ) -> AnswerSectionV2:
        """Generate section with enhanced analytical content"""
        # Filter insights relevant to this section
        section_insights = self._filter_insights_for_section(
            statistical_insights, section_def["title"]
        )
        
        # Create enhanced prompt with statistical context
        section_prompt = f"""
Write the "{section_def['title']}" section focusing on: {section_def['focus']}

Query: {context['query']}

Statistical insights available:
{self._format_statistical_insights(section_insights[:10])}

{f"Meta-analysis results: {meta_analysis}" if meta_analysis else ""}

Requirements:
- Include specific quantitative findings with effect sizes and confidence intervals
- Distinguish correlation from causation explicitly
- Address methodological limitations and potential biases
- Use precise scientific language and proper statistical terminology
- Length: approximately {int(2000 * section_def['weight'])} words
"""
        
        # Generate content
        try:
            content = await llm_client.generate(
                section_prompt,
                temperature=0.3,  # Lower temperature for analytical precision
                max_tokens=int(3000 * section_def['weight'])
            )
        except:
            content = self._generate_fallback_analytical_section(
                section_def, search_results, section_insights
            )
        
        # Create analytical citations
        citation_ids = self._create_analytical_citations(
            content, search_results, section_insights
        )
        
        # Extract quantitative insights
        insights = self._extract_quantitative_insights(content, section_insights)
        
        # Calculate section confidence based on evidence quality
        section_confidence = self._calculate_section_confidence(
            section_insights, citation_ids, content
        )
        
        return AnswerSectionV2(
            title=section_def["title"],
            content=content,
            confidence=section_confidence,
            citations=citation_ids,
            word_count=len(content.split()),
            key_insights=insights,
            metadata={
                "paradigm": "bernard",
                "statistical_insights_used": len(section_insights),
                "quantitative_claims": self._count_quantitative_claims(content),
            }
        )
    
    def _extract_percentage_context(self, context_text: str) -> str:
        """Extract what a percentage refers to from surrounding context"""
        patterns = [
            r"(\w+)\s+(?:rate|level|proportion)",
            r"(?:increase|decrease|reduction|growth)\s+in\s+(\w+)",
            r"(\w+)\s+(?:among|of|in)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, context_text.lower())
            if match:
                return match.group(1)
        
        return "metric"
    
    def _calculate_heterogeneity(self, effect_sizes: List[float]) -> str:
        """Calculate heterogeneity of effect sizes"""
        if len(effect_sizes) < 2:
            return "insufficient_data"
        
        mean_effect = sum(effect_sizes) / len(effect_sizes)
        variance = sum((e - mean_effect) ** 2 for e in effect_sizes) / len(effect_sizes)
        
        if variance < 0.1:
            return "low"
        elif variance < 0.3:
            return "moderate"
        else:
            return "high"
    
    def _format_statistical_insights(self, insights: List[StatisticalInsight]) -> str:
        """Format statistical insights for prompt"""
        formatted = []
        for insight in insights:
            if insight.confidence_interval:
                ci_str = f" (95% CI: {insight.confidence_interval[0]:.2f}-{insight.confidence_interval[1]:.2f})"
            else:
                ci_str = ""
            
            if insight.p_value:
                p_str = f" (p={insight.p_value:.3f})"
            else:
                p_str = ""
            
            formatted.append(
                f"- {insight.metric}: {insight.value}{insight.unit}{ci_str}{p_str}"
            )
        
        return "\n".join(formatted)
    
    def _filter_insights_for_section(
        self, insights: List[StatisticalInsight], section_title: str
    ) -> List[StatisticalInsight]:
        """Filter statistical insights relevant to section"""
        section_relevance = {
            "Executive Summary": ["effect_size", "significance", "sample_size"],
            "Quantitative Analysis": ["correlation", "percentage", "metric"],
            "Causal Mechanisms": ["effect_size", "correlation", "significance"],
            "Methodological Assessment": ["sample_size", "confidence", "significance"],
            "Evidence Synthesis": ["effect_size", "heterogeneity", "pooled"],
            "Research Implications": ["gaps", "future", "limitations"],
        }
        
        relevant_metrics = section_relevance.get(section_title, [])
        return [
            i for i in insights 
            if any(metric in i.metric.lower() for metric in relevant_metrics)
        ]
    
    def _create_analytical_citations(
        self,
        content: str,
        sources: List[SearchResultSchema],
        insights: List[StatisticalInsight]
    ) -> List[str]:
        """Create citations with enhanced analytical metadata"""
        citation_ids = []
        
        # Prioritize peer-reviewed sources
        peer_reviewed = [
            s for s in sources 
            if any(domain in s.metadata.get("domain", "").lower() 
                   for domain in ["arxiv", "pubmed", "nature", "science"])
        ]
        
        for source in (peer_reviewed + sources)[:6]:  # More citations for analytical
            # Determine citation type based on content
            domain = source.metadata.get("domain", "").lower()
            if "meta-analysis" in source.title.lower():
                citation_type = "meta-analysis"
            elif "systematic review" in source.title.lower():
                citation_type = "systematic-review"
            elif any(d in domain for d in ["arxiv", "pubmed"]):
                citation_type = "peer-reviewed"
            else:
                citation_type = "data"
            
            citation = self.create_citation(source, citation_type)
            citation.metadata.update({
                "study_type": self._identify_study_type(source),
                "sample_size": self._extract_sample_size(source),
                "publication_year": self._extract_year(source),
            })
            citation_ids.append(citation.id)
        
        return citation_ids
    
    def _extract_quantitative_insights(
        self, content: str, statistical_insights: List[StatisticalInsight]
    ) -> List[str]:
        """Extract quantitative insights from generated content"""
        insights = []
        
        # Extract effect sizes mentioned
        effect_matches = re.findall(self.statistical_patterns["effect_size"], content)
        for match in effect_matches:
            insights.append(f"Effect size: {match}")
        
        # Extract significant findings
        p_value_matches = re.findall(self.statistical_patterns["p_value"], content)
        for match in p_value_matches:
            insights.append(f"Statistically significant finding (p={match})")
        
        # Extract correlations
        correlation_matches = re.findall(self.statistical_patterns["correlation"], content)
        for match in correlation_matches:
            insights.append(f"Correlation identified: r={match}")
        
        # Add key statistical insights
        if statistical_insights:
            top_insights = sorted(
                statistical_insights, 
                key=lambda x: abs(x.value) if x.metric == "effect_size" else x.value,
                reverse=True
            )[:3]
            for insight in top_insights:
                insights.append(
                    f"{insight.metric}: {insight.value}{insight.unit}"
                )
        
        return insights[:5]  # Top 5 insights
    
    def _calculate_section_confidence(
        self, insights: List[StatisticalInsight], citations: List[str], content: str
    ) -> float:
        """Calculate confidence based on evidence quality"""
        # Base confidence on statistical significance
        significant_findings = [i for i in insights if i.p_value and i.p_value < 0.05]
        significance_factor = min(1.0, len(significant_findings) / 3)
        
        # Factor in sample sizes
        sample_sizes = [i.sample_size for i in insights if i.sample_size]
        if sample_sizes:
            avg_sample_size = sum(sample_sizes) / len(sample_sizes)
            sample_factor = min(1.0, avg_sample_size / 1000)  # 1000+ is high confidence
        else:
            sample_factor = 0.5
        
        # Factor in number of peer-reviewed citations
        citation_factor = min(1.0, len(citations) / 4)
        
        # Check for limitations mentioned
        limitations_mentioned = any(
            word in content.lower() 
            for word in ["limitation", "caveat", "however", "although"]
        )
        transparency_factor = 0.9 if limitations_mentioned else 0.7
        
        return (
            significance_factor * 0.3 +
            sample_factor * 0.3 +
            citation_factor * 0.2 +
            transparency_factor * 0.2
        )
    
    def _count_quantitative_claims(self, content: str) -> int:
        """Count number of quantitative claims in content"""
        patterns = [
            self.statistical_patterns["percentage"],
            self.statistical_patterns["correlation"],
            self.statistical_patterns["p_value"],
            r"\d+\s*(?:participants|subjects|studies|trials)",
            r"(?:increased|decreased|higher|lower)\s+by\s+\d+",
        ]
        
        total_matches = 0
        for pattern in patterns:
            total_matches += len(re.findall(pattern, content))
        
        return total_matches
    
    def _identify_study_type(self, source: SearchResultSchema) -> str:
        """Identify the type of study from source"""
        text = f"{source.title} {source.snippet}".lower()
        
        study_types = {
            "meta-analysis": ["meta-analysis", "systematic review"],
            "rct": ["randomized controlled trial", "rct", "randomized trial"],
            "cohort": ["cohort study", "longitudinal", "prospective"],
            "case-control": ["case-control", "case control"],
            "cross-sectional": ["cross-sectional", "survey"],
            "experimental": ["experiment", "laboratory"],
        }
        
        for study_type, keywords in study_types.items():
            if any(keyword in text for keyword in keywords):
                return study_type
        
        return "observational"
    
    def _extract_sample_size(self, source: SearchResultSchema) -> Optional[int]:
        """Extract sample size from source"""
        text = f"{source.title} {source.snippet}"
        match = re.search(self.statistical_patterns["sample_size"], text)
        return int(match.group(1)) if match else None
    
    def _extract_year(self, source: SearchResultSchema) -> Optional[int]:
        """Extract publication year from source"""
        text = f"{source.title} {source.snippet}"
        year_match = re.search(r"(19|20)\d{2}", text)
        return int(year_match.group(0)) if year_match else None
    
    def _generate_fallback_analytical_section(
        self, 
        section_def: Dict[str, Any], 
        results: List[SearchResultSchema],
        insights: List[StatisticalInsight]
    ) -> str:
        """Generate fallback analytical content"""
        content = f"# {section_def['title']}\n\n"
        content += f"This section provides {section_def['focus']}.\n\n"
        
        # Add statistical insights if available
        if insights:
            content += "## Key Statistical Findings\n\n"
            for insight in insights[:5]:
                content += f"- {insight.metric}: {insight.value}{insight.unit}\n"
            content += "\n"
        
        # Add source summaries
        content += "## Evidence Summary\n\n"
        for result in results[:3]:
            content += f"According to research from {result.metadata.get('domain', 'sources')}, "
            content += f"{result.snippet}\n\n"
        
        return content
    
    async def generate_answer(self, context) -> Dict[str, Any]:
        """Generate answer for Bernard paradigm - adapter for enhanced integration"""
        # Extract necessary fields from context
        search_results = context.search_results if hasattr(context, 'search_results') else []
        
        # Generate sections with statistical analysis
        sections, stats, meta = await self.generate_sections(
            {"query": context.query, "paradigm": "bernard"},
            search_results
        )
        
        # Format answer
        content_parts = []
        for section in sections:
            content_parts.append(f"## {section.title}")
            content_parts.append(section.content)
            if section.key_insights:
                content_parts.append("\n**Key Findings:**")
                for insight in section.key_insights:
                    content_parts.append(f"- {insight}")
            content_parts.append("")
        
        # Add statistical summary if available
        if stats:
            content_parts.append("## Statistical Summary")
            for stat in stats[:5]:
                content_parts.append(f"- {stat.metric}: {stat.value}{stat.unit}")
            content_parts.append("")
        
        return {
            "content": "\n".join(content_parts),
            "paradigm": "bernard",
            "sections": len(sections),
            "citations": list(self.citations.keys()),
            "synthesis_quality": 0.9,
            "metadata": {
                "tone": "analytical",
                "statistical_insights": len(stats),
                "meta_analysis": meta is not None
            }
        }


class MaeveAnswerGeneratorV2(ParadigmAnswerGeneratorV2):
    """Enhanced strategic paradigm answer generator"""
    
    def __init__(self):
        super().__init__(HostParadigm.MAEVE)
        self.strategic_patterns = {
            "market_size": r"\$(\d+(?:\.\d+)?)\s*(?:billion|million|B|M)",
            "growth_rate": r"(\d+(?:\.\d+)?)\s*%\s*(?:growth|CAGR|increase)",
            "market_share": r"(\d+(?:\.\d+)?)\s*%\s*(?:market\s*share|of\s*the\s*market)",
            "roi": r"(?:ROI|return)\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%",
            "cost_savings": r"(?:save|reduce\s*costs?)\s*(?:by\s*)?\$?(\d+(?:\.\d+)?)\s*(?:million|thousand|K|M)?",
            "timeline": r"(\d+)\s*(?:months?|years?|quarters?|weeks?)",
        }
    
    def get_section_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "title": "Strategic Landscape Analysis",
                "focus": "Market dynamics, competitive positioning, and opportunity assessment",
                "weight": 0.20,
            },
            {
                "title": "Value Creation Strategies",
                "focus": "Specific approaches to capture value and achieve competitive advantage",
                "weight": 0.25,
            },
            {
                "title": "Implementation Framework",
                "focus": "Tactical execution plan with phases, resources, and dependencies",
                "weight": 0.20,
            },
            {
                "title": "Risk Mitigation & Contingencies",
                "focus": "Strategic risks, mitigation strategies, and alternative scenarios",
                "weight": 0.15,
            },
            {
                "title": "Performance Metrics & KPIs",
                "focus": "Success metrics, measurement framework, and tracking mechanisms",
                "weight": 0.10,
            },
            {
                "title": "Strategic Roadmap",
                "focus": "Timeline, milestones, and decision gates",
                "weight": 0.10,
            },
        ]
    
    async def generate_sections(
        self,
        context: Dict[str, Any],
        search_results: List[SearchResultSchema]
    ) -> Tuple[List[AnswerSectionV2], List[Dict[str, Any]], Dict[str, Any], Dict[str, List[str]]]:
        """Generate Maeve-specific sections with strategic analysis"""
        
        # Extract strategic insights
        strategic_insights = await self._extract_strategic_insights(search_results)
        
        # Perform competitive analysis
        competitive_analysis = await self._perform_competitive_analysis(
            context['query'], search_results
        )
        
        # Generate SWOT analysis
        swot_analysis = await self._generate_swot_analysis(
            context['query'], search_results, strategic_insights
        )
        
        # Generate sections
        sections = []
        for section_def in self.get_section_structure():
            section = await self._generate_strategic_section(
                context, section_def, search_results, 
                strategic_insights, competitive_analysis, swot_analysis
            )
            sections.append(section)
        
        return sections, strategic_insights, competitive_analysis, swot_analysis
    
    async def _extract_strategic_insights(
        self, search_results: List[SearchResultSchema]
    ) -> List[Dict[str, Any]]:
        """Extract strategic business insights from search results"""
        insights = []
        
        for result in search_results:
            text = f"{result.title} {result.snippet}"
            
            # Extract market sizes
            for match in re.finditer(self.strategic_patterns["market_size"], text):
                value = float(match.group(1))
                unit = "billion" if "billion" in match.group(0) or "B" in match.group(0) else "million"
                insights.append({
                    "type": "market_size",
                    "value": value,
                    "unit": unit,
                    "context": text[max(0, match.start()-50):match.end()+50],
                })
            
            # Extract growth rates
            for match in re.finditer(self.strategic_patterns["growth_rate"], text):
                insights.append({
                    "type": "growth_rate",
                    "value": float(match.group(1)),
                    "unit": "%",
                    "context": text[max(0, match.start()-50):match.end()+50],
                })
            
            # Extract ROI metrics
            for match in re.finditer(self.strategic_patterns["roi"], text):
                insights.append({
                    "type": "roi",
                    "value": float(match.group(1)),
                    "unit": "%",
                    "context": text[max(0, match.start()-50):match.end()+50],
                })
            
            # Extract timelines
            for match in re.finditer(self.strategic_patterns["timeline"], text):
                insights.append({
                    "type": "timeline",
                    "value": int(match.group(1)),
                    "unit": match.group(0).split()[-1],
                    "context": text[max(0, match.start()-50):match.end()+50],
                })
        
        return insights
    
    async def _perform_competitive_analysis(
        self, query: str, search_results: List[SearchResultSchema]
    ) -> Dict[str, Any]:
        """Perform competitive analysis based on search results"""
        competitors = []
        market_leaders = []
        disruptors = []
        
        competitive_keywords = {
            "leaders": ["market leader", "dominant", "largest", "top player"],
            "competitors": ["competitor", "rival", "competing", "versus", "vs"],
            "disruptors": ["disrupt", "innovative", "challenger", "startup", "emerging"],
        }
        
        for result in search_results:
            text = f"{result.title} {result.snippet}".lower()
            
            # Identify market leaders
            if any(keyword in text for keyword in competitive_keywords["leaders"]):
                market_leaders.append(result.title)
            
            # Identify competitors
            if any(keyword in text for keyword in competitive_keywords["competitors"]):
                competitors.append(result.title)
            
            # Identify disruptors
            if any(keyword in text for keyword in competitive_keywords["disruptors"]):
                disruptors.append(result.title)
        
        return {
            "market_leaders": list(set(market_leaders))[:3],
            "key_competitors": list(set(competitors))[:5],
            "disruptors": list(set(disruptors))[:3],
            "competitive_intensity": self._assess_competitive_intensity(search_results),
            "market_concentration": self._assess_market_concentration(search_results),
        }
    
    async def _generate_swot_analysis(
        self,
        query: str,
        search_results: List[SearchResultSchema],
        strategic_insights: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Generate SWOT analysis from available data"""
        swot = {
            "strengths": [],
            "weaknesses": [],
            "opportunities": [],
            "threats": [],
        }
        
        # Analyze search results for SWOT elements
        for result in search_results[:10]:
            text = f"{result.title} {result.snippet}".lower()
            
            if any(word in text for word in ["advantage", "strength", "leading", "superior"]):
                swot["strengths"].append(self._extract_swot_item(text, "strength"))
            
            if any(word in text for word in ["weakness", "challenge", "limitation", "gap"]):
                swot["weaknesses"].append(self._extract_swot_item(text, "weakness"))
            
            if any(word in text for word in ["opportunity", "potential", "growth", "emerging"]):
                swot["opportunities"].append(self._extract_swot_item(text, "opportunity"))
            
            if any(word in text for word in ["threat", "risk", "competition", "disruption"]):
                swot["threats"].append(self._extract_swot_item(text, "threat"))
        
        # Ensure each category has at least 2 items
        for category in swot:
            if len(swot[category]) < 2:
                swot[category].extend(self._generate_default_swot_items(category, query))
            swot[category] = list(set(swot[category]))[:4]  # Unique items, max 4
        
        return swot
    
    async def _generate_strategic_section(
        self,
        context: Dict[str, Any],
        section_def: Dict[str, Any],
        search_results: List[SearchResultSchema],
        strategic_insights: List[Dict[str, Any]],
        competitive_analysis: Dict[str, Any],
        swot_analysis: Dict[str, List[str]],
    ) -> AnswerSectionV2:
        """Generate strategic section with business insights"""
        # Filter insights relevant to section
        section_insights = self._filter_strategic_insights(strategic_insights, section_def["title"])
        
        # Create strategic prompt
        section_prompt = f"""
Write the "{section_def['title']}" section focusing on: {section_def['focus']}

Query: {context['query']}

Strategic insights:
{self._format_strategic_insights(section_insights[:8])}

Competitive landscape:
- Market leaders: {competitive_analysis['market_leaders']}
- Competitive intensity: {competitive_analysis['competitive_intensity']}

SWOT considerations:
- Key strengths: {swot_analysis['strengths'][:2]}
- Main opportunities: {swot_analysis['opportunities'][:2]}

Requirements:
- Include specific market data, growth rates, and financial metrics
- Provide actionable strategic recommendations
- Address competitive dynamics and market positioning
- Use business terminology and strategic frameworks
- Length: approximately {int(2000 * section_def['weight'])} words
"""
        
        # Generate content
        try:
            content = await llm_client.generate(
                section_prompt,
                temperature=0.5,  # Balanced for strategic creativity
                max_tokens=int(3000 * section_def['weight'])
            )
        except:
            content = self._generate_fallback_strategic_section(
                section_def, search_results, section_insights, competitive_analysis
            )
        
        # Create strategic citations
        citation_ids = self._create_strategic_citations(
            content, search_results, section_insights
        )
        
        # Extract strategic insights
        insights = self._extract_strategic_recommendations(content, section_insights)
        
        # Calculate section confidence
        section_confidence = self._calculate_section_strategic_confidence(
            section_insights, competitive_analysis, content
        )
        
        return AnswerSectionV2(
            title=section_def["title"],
            content=content,
            confidence=section_confidence,
            citations=citation_ids,
            word_count=len(content.split()),
            key_insights=insights,
            metadata={
                "paradigm": "maeve",
                "strategic_insights_used": len(section_insights),
                "competitive_factors": len(competitive_analysis.get("key_competitors", [])),
            }
        )
    
    def _assess_competitive_intensity(self, search_results: List[SearchResultSchema]) -> str:
        """Assess competitive intensity from search results"""
        competitive_terms = ["competitor", "competition", "rival", "market share", "competing"]
        competitive_mentions = 0
        
        for result in search_results:
            text = f"{result.title} {result.snippet}".lower()
            competitive_mentions += sum(1 for term in competitive_terms if term in text)
        
        avg_mentions = competitive_mentions / len(search_results) if search_results else 0
        
        if avg_mentions > 2:
            return "high"
        elif avg_mentions > 1:
            return "moderate"
        else:
            return "low"
    
    def _assess_market_concentration(self, search_results: List[SearchResultSchema]) -> str:
        """Assess market concentration from search results"""
        concentration_indicators = ["monopoly", "duopoly", "dominant", "fragmented", "consolidated"]
        
        for result in search_results:
            text = f"{result.title} {result.snippet}".lower()
            if "monopoly" in text or "dominant player" in text:
                return "highly_concentrated"
            elif "duopoly" in text or "two players" in text:
                return "concentrated"
            elif "fragmented" in text or "many players" in text:
                return "fragmented"
        
        return "moderate"
    
    def _extract_swot_item(self, text: str, category: str) -> str:
        """Extract a SWOT item from text"""
        sentences = text.split(".")
        for sentence in sentences:
            if category in sentence.lower():
                return sentence.strip()[:100] + "..."
        return f"Identified {category} in market analysis"
    
    def _generate_default_swot_items(self, category: str, query: str) -> List[str]:
        """Generate default SWOT items when not enough found"""
        defaults = {
            "strengths": [
                "First-mover advantage potential",
                "Alignment with market trends",
            ],
            "weaknesses": [
                "Resource constraints",
                "Limited market presence",
            ],
            "opportunities": [
                "Growing market demand",
                "Technology enablement",
            ],
            "threats": [
                "Competitive pressure",
                "Market volatility",
            ],
        }
        return defaults.get(category, [])
    
    def _filter_strategic_insights(
        self, insights: List[Dict[str, Any]], section_title: str
    ) -> List[Dict[str, Any]]:
        """Filter strategic insights relevant to section"""
        section_relevance = {
            "Strategic Landscape Analysis": ["market_size", "growth_rate", "market_share"],
            "Value Creation Strategies": ["roi", "cost_savings", "growth_rate"],
            "Implementation Framework": ["timeline", "cost_savings", "resources"],
            "Risk Mitigation & Contingencies": ["threats", "risks", "volatility"],
            "Performance Metrics & KPIs": ["roi", "growth_rate", "metrics"],
            "Strategic Roadmap": ["timeline", "milestones", "phases"],
        }
        
        relevant_types = section_relevance.get(section_title, [])
        return [
            i for i in insights 
            if i["type"] in relevant_types or any(t in i.get("context", "").lower() for t in relevant_types)
        ]
    
    def _format_strategic_insights(self, insights: List[Dict[str, Any]]) -> str:
        """Format strategic insights for prompt"""
        formatted = []
        for insight in insights:
            if insight["type"] == "market_size":
                formatted.append(f"- Market size: ${insight['value']} {insight['unit']}")
            elif insight["type"] == "growth_rate":
                formatted.append(f"- Growth rate: {insight['value']}% annually")
            elif insight["type"] == "roi":
                formatted.append(f"- ROI potential: {insight['value']}%")
            elif insight["type"] == "timeline":
                formatted.append(f"- Implementation timeline: {insight['value']} {insight['unit']}")
            else:
                formatted.append(f"- {insight['type']}: {insight['value']} {insight.get('unit', '')}")
        
        return "\n".join(formatted)
    
    def _create_strategic_citations(
        self, content: str, sources: List[SearchResultSchema], insights: List[Dict[str, Any]]
    ) -> List[str]:
        """Create citations with strategic metadata"""
        citation_ids = []
        
        # Prioritize business and industry sources
        business_sources = [
            s for s in sources
            if any(domain in s.metadata.get("domain", "").lower() 
                   for domain in ["forbes", "mckinsey", "gartner", "harvard", "deloitte", "pwc"])
        ]
        
        for source in (business_sources + sources)[:5]:
            citation_type = "strategic" if source in business_sources else "data"
            
            citation = self.create_citation(source, citation_type)
            citation.metadata.update({
                "source_type": self._identify_source_type(source),
                "publication_date": self._extract_year(source),
                "strategic_relevance": self._assess_strategic_relevance(source, insights),
            })
            citation_ids.append(citation.id)
        
        return citation_ids
    
    def _extract_strategic_recommendations(
        self, content: str, insights: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract strategic recommendations from content"""
        recommendations = []
        
        # Look for action-oriented language
        action_patterns = [
            r"(?:should|must|need to|recommend)\s+([^.]+)",
            r"(?:opportunity to|potential for)\s+([^.]+)",
            r"(?:strategic priority|key initiative):\s*([^.]+)",
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            recommendations.extend([match.strip() for match in matches])
        
        # Add insights with high ROI or growth
        high_value_insights = [
            i for i in insights
            if (i["type"] == "roi" and i["value"] > 20) or
               (i["type"] == "growth_rate" and i["value"] > 10)
        ]
        
        for insight in high_value_insights[:2]:
            if insight["type"] == "roi":
                recommendations.append(f"Opportunity for {insight['value']}% ROI")
            elif insight["type"] == "growth_rate":
                recommendations.append(f"Target {insight['value']}% growth market")
        
        return list(set(recommendations))[:5]  # Top 5 unique recommendations
    
    def _calculate_section_strategic_confidence(
        self, insights: List[Dict[str, Any]], competitive_analysis: Dict[str, Any], content: str
    ) -> float:
        """Calculate strategic section confidence"""
        # Market data quality
        market_data_factor = min(1.0, len(insights) / 5) * 0.3
        
        # Competitive intelligence quality
        competitive_factor = 0.3
        if competitive_analysis.get("key_competitors"):
            competitive_factor *= min(1.0, len(competitive_analysis["key_competitors"]) / 3)
        
        # Strategic framework usage
        frameworks = ["swot", "porter", "bcg", "ansoff", "blue ocean"]
        framework_factor = 0.2 if any(fw in content.lower() for fw in frameworks) else 0.1
        
        # Quantitative backing
        numbers_in_content = len(re.findall(r'\d+(?:\.\d+)?%?', content))
        quantitative_factor = min(1.0, numbers_in_content / 10) * 0.2
        
        return market_data_factor + competitive_factor + framework_factor + quantitative_factor
    
    def _identify_source_type(self, source: SearchResultSchema) -> str:
        """Identify the type of business source"""
        domain = source.metadata.get("domain", "").lower()
        
        source_types = {
            "consulting": ["mckinsey", "bcg", "bain", "deloitte", "pwc", "ey", "kpmg"],
            "research": ["gartner", "forrester", "idc", "statista"],
            "business_media": ["forbes", "fortune", "bloomberg", "wsj", "ft"],
            "academic": ["harvard", "wharton", "stanford", "mit"],
            "industry": ["trade", "association", "industry"],
        }
        
        for source_type, keywords in source_types.items():
            if any(keyword in domain for keyword in keywords):
                return source_type
        
        return "general"
    
    def _assess_strategic_relevance(
        self, source: SearchResultSchema, insights: List[Dict[str, Any]]
    ) -> float:
        """Assess how strategically relevant a source is"""
        relevance_score = 0.5  # Base score
        
        # Check if source contains strategic insights
        text = f"{source.title} {source.snippet}".lower()
        
        strategic_terms = ["strategy", "competitive", "market share", "growth", "roi", "opportunity"]
        term_matches = sum(1 for term in strategic_terms if term in text)
        relevance_score += min(0.3, term_matches * 0.05)
        
        # Check if source provided quantitative insights
        source_insights = [i for i in insights if source.url in i.get("context", "")]
        if source_insights:
            relevance_score += 0.2
        
        return min(1.0, relevance_score)
    
    def _extract_year(self, source: SearchResultSchema) -> Optional[int]:
        """Extract publication year from source"""
        text = f"{source.title} {source.snippet}"
        year_match = re.search(r"(20\d{2})", text)
        return int(year_match.group(1)) if year_match else None
    
    def _generate_fallback_strategic_section(
        self,
        section_def: Dict[str, Any],
        results: List[SearchResultSchema],
        insights: List[Dict[str, Any]],
        competitive_analysis: Dict[str, Any]
    ) -> str:
        """Generate fallback strategic content"""
        content = f"# {section_def['title']}\n\n"
        content += f"This section addresses {section_def['focus']}.\n\n"
        
        # Add strategic insights
        if insights:
            content += "## Market Intelligence\n\n"
            for insight in insights[:5]:
                if insight["type"] == "market_size":
                    content += f"- Market valued at ${insight['value']} {insight['unit']}\n"
                elif insight["type"] == "growth_rate":
                    content += f"- Growing at {insight['value']}% annually\n"
                elif insight["type"] == "roi":
                    content += f"- ROI potential of {insight['value']}%\n"
            content += "\n"
        
        # Add competitive landscape
        if competitive_analysis.get("market_leaders"):
            content += "## Competitive Landscape\n\n"
            content += f"Market leaders include: {', '.join(competitive_analysis['market_leaders'][:3])}\n"
            content += f"Competitive intensity: {competitive_analysis.get('competitive_intensity', 'moderate')}\n\n"
        
        # Add source insights
        for result in results[:3]:
            content += f"According to {result.metadata.get('domain', 'industry analysis')}, "
            content += f"{result.snippet}\n\n"
        
        return content
    
    async def generate_answer(self, context) -> Dict[str, Any]:
        """Generate answer for Maeve paradigm - adapter for enhanced integration"""
        # Extract necessary fields from context
        search_results = context.search_results if hasattr(context, 'search_results') else []
        
        # Generate sections with strategic analysis
        sections, insights, competitive, swot = await self.generate_sections(
            {
                "query": context.query,
                "paradigm": "maeve",
                "context_engineering": getattr(context, "context_engineering", {}),
            },
            search_results,
        )
        
        # Format answer
        content_parts = []
        for section in sections:
            content_parts.append(f"## {section.title}")
            content_parts.append(section.content)
            content_parts.append("")
        
        # Add SWOT summary if available
        if swot:
            content_parts.append("## SWOT Analysis")
            for category, items in swot.items():
                if items:
                    content_parts.append(f"\n**{category.title()}:**")
                    for item in items[:2]:
                        content_parts.append(f"- {item}")
            content_parts.append("")
        
        return {
            "content": "\n".join(content_parts),
            "paradigm": "maeve",
            "sections": len(sections),
            "citations": list(self.citations.keys()),
            "synthesis_quality": 0.88,
            "metadata": {
                "tone": "strategic",
                "strategic_insights": len(insights),
                "competitive_analysis": competitive is not None,
                "swot_completed": swot is not None
            }
        }


class TeddyAnswerGeneratorV2(ParadigmAnswerGeneratorV2):
    """Supportive paradigm answer generator"""
    
    def __init__(self):
        super().__init__(HostParadigm.TEDDY)
    
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
    
    async def generate_sections(
        self,
        context: Dict[str, Any],
        search_results: List[SearchResultSchema]
    ) -> List[AnswerSectionV2]:
        """Generate Teddy-specific sections"""
        sections = []
        
        for section_def in self.get_section_structure():
            section = await self._generate_section(
                context, section_def, search_results
            )
            sections.append(section)
            
        return sections
    
    async def _generate_section(
        self,
        context: Dict[str, Any],
        section_def: Dict[str, Any],
        search_results: List[SearchResultSchema]
    ) -> AnswerSectionV2:
        """Generate a single section"""
        # Filter results relevant to this section
        relevant_results = self._filter_results_for_section(search_results, section_def)
        
        # Create section-specific prompt
        section_prompt = f"""
Write the "{section_def['title']}" section focusing on: {section_def['focus']}

Query: {context['query']}

Use these specific sources:
{self._format_results_for_prompt(relevant_results[:5])}

Requirements:
- Use warm, supportive language that builds hope and connection
- Focus on human dignity and the power of community care
- Emphasize resources, solutions, and paths forward
- Include specific resources and cite sources
- Length: approximately {int(2000 * section_def['weight'])} words
"""
        
        # Generate content
        try:
            content = await llm_client.generate(
                section_prompt,
                temperature=0.6,
                max_tokens=int(3000 * section_def['weight'])
            )
        except:
            content = self._generate_fallback_section(section_def, relevant_results)
        
        # Create citations
        citation_ids = []
        for source in relevant_results[:3]:
            citation = self.create_citation(source, "reference")
            citation_ids.append(citation.id)
        
        # Extract insights
        insights = self.extract_key_insights(content, 3)
        
        return AnswerSectionV2(
            title=section_def["title"],
            content=content,
            confidence=0.88,
            citations=citation_ids,
            word_count=len(content.split()),
            key_insights=insights,
            metadata={"paradigm": "teddy", "focus": section_def["focus"]}
        )
    
    def _filter_results_for_section(
        self, 
        results: List[SearchResultSchema], 
        section_def: Dict[str, Any]
    ) -> List[SearchResultSchema]:
        """Filter results relevant to a specific section"""
        section_keywords = {
            "Understanding the Need": [
                "struggle", "challenge", "difficulty", "experience"
            ],
            "Available Support Resources": [
                "resource", "help", "service", "support"
            ],
            "Success Stories": [
                "success", "recovery", "story", "hope"
            ],
            "How to Help": [
                "volunteer", "donate", "action", "contribute"
            ],
        }
        
        keywords = section_keywords.get(section_def["title"], [])
        
        relevant = []
        for result in results:
            text = f"{result.title} {result.snippet}".lower()
            if any(keyword in text for keyword in keywords):
                relevant.append(result)
        
        return relevant or results[:5]
    
    def _format_results_for_prompt(self, results: List[SearchResultSchema]) -> str:
        """Format search results for LLM prompt"""
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(f"""
{i}. {result.title}
Source: {result.metadata.get('domain', 'Unknown')}
URL: {result.url}
Trust Score: {result.credibility_score:.2f}
Content: {result.snippet}
""")
        return "\n".join(formatted)
    
    def _generate_fallback_section(
        self, 
        section_def: Dict[str, Any], 
        results: List[SearchResultSchema]
    ) -> str:
        """Generate fallback content when LLM fails"""
        content = f"# {section_def['title']}\n\n"
        content += f"This section focuses on {section_def['focus']}.\n\n"
        
        for result in results[:3]:
            content += f"According to {result.metadata.get('domain', 'trusted sources')}, "
            content += f"{result.snippet}\n\n"
        
        content += "Remember, you are not alone in this journey. "
        content += "Help is available, and together we can make a difference.\n"
        
        return content
    
    async def generate_answer(self, context) -> Dict[str, Any]:
        """Generate answer for Teddy paradigm - adapter for enhanced integration"""
        # Extract necessary fields from context
        search_results = context.search_results if hasattr(context, 'search_results') else []
        
        # Generate sections
        sections = await self.generate_sections(
            {
                "query": context.query,
                "paradigm": "teddy",
                "context_engineering": getattr(context, "context_engineering", {}),
            },
            search_results,
        )
        
        # Format answer
        content_parts = []
        for section in sections:
            content_parts.append(f"## {section.title}")
            content_parts.append(section.content)
            content_parts.append("")
        
        # Add supportive closing
        content_parts.append("\n---\n")
        content_parts.append("Remember: You are not alone. Help is available, and together we can make a difference.")
        
        return {
            "content": "\n".join(content_parts),
            "paradigm": "teddy",
            "sections": len(sections),
            "citations": list(self.citations.keys()),
            "synthesis_quality": 0.88,
            "metadata": {
                "tone": "supportive",
                "focus": "community care"
            }
        }


class EnhancedAnswerGeneratorV2:
    """Main orchestrator for V2 answer generation with full feature parity"""
    
    def __init__(self):
        self.generators = {
            HostParadigm.DOLORES: DoloresAnswerGeneratorV2(),
            HostParadigm.BERNARD: BernardAnswerGeneratorV2(),
            HostParadigm.MAEVE: MaeveAnswerGeneratorV2(),
            HostParadigm.TEDDY: TeddyAnswerGeneratorV2()
        }
        self.compressor = text_compressor
    
    async def generate_answer(
        self,
        classification: ClassificationResultSchema,
        context_engineered: ContextEngineeredQuerySchema,
        search_results: List[SearchResultSchema],
        user_context: UserContextSchema,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive answer using paradigm-specific generators"""
        
        # Build comprehensive context
        full_context = self._build_full_context(
            classification,
            context_engineered,
            search_results,
            user_context
        )
        
        # Get paradigm-specific generator
        generator = self.generators[classification.primary_paradigm]
        
        # Generate paradigm-specific content
        if classification.primary_paradigm == HostParadigm.DOLORES:
            sections = await generator.generate_sections(full_context, search_results)
            action_items = self._generate_dolores_action_items(full_context)
            
        elif classification.primary_paradigm == HostParadigm.BERNARD:
            sections, stats, meta = await generator.generate_sections(full_context, search_results)
            action_items = self._generate_bernard_action_items(full_context, stats)
            
        elif classification.primary_paradigm == HostParadigm.MAEVE:
            sections, insights, competitive, swot = await generator.generate_sections(
                full_context, search_results
            )
            action_items = self._generate_maeve_recommendations(
                full_context, insights, competitive, swot
            )
            
        else:  # TEDDY
            sections = await generator.generate_sections(full_context, search_results)
            action_items = self._generate_teddy_action_items(full_context)
        
        # Generate summary
        summary = await self._generate_paradigm_summary(
            classification, sections, full_context
        )
        
        # Format response based on user verbosity
        if user_context.verbosity_preference == "minimal":
            content = self._format_minimal_answer(summary, sections[:2])
        elif user_context.verbosity_preference == "detailed":
            content = self._format_detailed_answer(summary, sections, action_items, generator.citations)
        else:  # balanced
            content = self._format_balanced_answer(summary, sections, action_items)
        
        # Select best sources
        sources_used = self._select_best_sources(
            search_results,
            user_context.source_limit,
            classification.primary_paradigm,
            list(generator.citations.keys())
        )
        
        # Calculate confidence and quality
        confidence_score = self._calculate_confidence(
            classification, sections, search_results
        )
        synthesis_quality = self._calculate_synthesis_quality(
            sections, list(generator.citations.values())
        )
        
        return {
            "content": content,
            "paradigm": classification.primary_paradigm.value,
            "sources": sources_used,
            "metadata": {
                "confidence": confidence_score,
                "synthesis_quality": synthesis_quality,
                "sections_generated": len(sections),
                "citations_created": len(generator.citations),
                "action_items": len(action_items),
                "tone_applied": self._get_paradigm_tone(classification.primary_paradigm),
                "user_verbosity": user_context.verbosity_preference,
                "context_layers_used": list(full_context.keys()),
                "generation_timestamp": datetime.utcnow().isoformat()
            }
        }
    
    def _build_full_context(
        self,
        classification: ClassificationResultSchema,
        context_engineered: ContextEngineeredQuerySchema,
        search_results: List[SearchResultSchema],
        user_context: UserContextSchema
    ) -> Dict[str, Any]:
        """Build comprehensive context from all sources"""
        
        # Extract key insights from each W-S-C-I layer
        write_insights = self._extract_write_insights(context_engineered.write_layer_output)
        select_insights = self._extract_select_insights(context_engineered.select_layer_output)
        compress_insights = self._extract_compress_insights(context_engineered.compress_layer_output)
        isolate_insights = self._extract_isolate_insights(context_engineered.isolate_layer_output)
        
        # Get debug reasoning if available
        debug_reasoning = []
        if context_engineered.debug_info:
            for debug in context_engineered.debug_info:
                debug_reasoning.extend(debug.reasoning)
        
        return {
            "query": classification.query,
            "paradigm": classification.primary_paradigm.value,
            "paradigm_distribution": classification.distribution,
            "user_preferences": {
                "location": user_context.location,
                "language": user_context.language,
                "role": user_context.role,
                "default_paradigm": user_context.default_paradigm
            },
            "narrative_context": write_insights,
            "tool_strategy": select_insights,
            "optimization_notes": compress_insights,
            "execution_strategy": isolate_insights,
            "debug_reasoning": debug_reasoning,
            "search_summary": {
                "total_results": len(search_results),
                "high_credibility_count": sum(1 for r in search_results if r.credibility_score >= 0.7),
                "sources": list(set(r.source_api for r in search_results))
            }
        }
    
    def _extract_write_insights(self, write_output: Dict[str, Any]) -> Dict[str, Any]:
        """Extract insights from Write layer"""
        return {
            "storyboard": write_output.get("storyboard", ""),
            "focus_areas": write_output.get("focus_areas", ""),
            "narrative_queries": write_output.get("narrative_queries", [])[:3]
        }
    
    def _extract_select_insights(self, select_output: Dict[str, Any]) -> Dict[str, Any]:
        """Extract insights from Select layer"""
        return {
            "strategy": select_output.get("strategy", ""),
            "tools_selected": select_output.get("selected_tools", []),
            "filters_applied": select_output.get("filters", [])
        }
    
    def _extract_compress_insights(self, compress_output: Dict[str, Any]) -> Dict[str, Any]:
        """Extract insights from Compress layer"""
        return {
            "compression_ratio": compress_output.get("compression_ratio", 1.0),
            "queries_removed": compress_output.get("removed_count", 0),
            "final_query_count": compress_output.get("query_count", 0)
        }
    
    def _extract_isolate_insights(self, isolate_output: Dict[str, Any]) -> Dict[str, Any]:
        """Extract insights from Isolate layer"""
        return {
            "search_strategy": isolate_output.get("search_strategy", {}),
            "paradigm_alignment": isolate_output.get("paradigm_alignment", ""),
            "query_distribution": isolate_output.get("search_strategy", {}).get("query_distribution", {})
        }
    
    async def _generate_paradigm_summary(
        self,
        classification: ClassificationResultSchema,
        sections: List[AnswerSectionV2],
        context: Dict[str, Any]
    ) -> str:
        """Generate paradigm-specific summary"""
        # Collect key insights from all sections
        all_insights = []
        for section in sections:
            all_insights.extend(section.key_insights[:2])
        
        summary_prompt = f"""
Create a {self._get_paradigm_tone(classification.primary_paradigm)} summary for: "{context['query']}"

Key insights from analysis:
{chr(10).join(f"- {insight}" for insight in all_insights[:6])}

Requirements:
- Lead with the most important finding
- Use {classification.primary_paradigm.value} perspective
- 3-4 sentences maximum
- Action-oriented conclusion
"""
        
        try:
            summary = await llm_client.generate(
                summary_prompt,
                temperature=0.5,
                max_tokens=300
            )
            return summary.strip()
        except:
            # Fallback summary
            return f"Based on our {classification.primary_paradigm.value} analysis, {all_insights[0] if all_insights else 'further investigation is needed'}."
    
    def _format_minimal_answer(
        self,
        summary: str,
        sections: List[AnswerSectionV2]
    ) -> str:
        """Format minimal answer for concise preference"""
        content_parts = [summary, "\n"]
        
        # Add first two sections, compressed
        for section in sections[:2]:
            content_parts.append(f"\n## {section.title}")
            # Take first paragraph only
            first_para = section.content.split('\n\n')[0]
            content_parts.append(first_para[:300] + "...")
        
        return "\n".join(content_parts)
    
    def _format_balanced_answer(
        self,
        summary: str,
        sections: List[AnswerSectionV2],
        action_items: List[Dict[str, Any]]
    ) -> str:
        """Format balanced answer with moderate detail"""
        content_parts = [summary, "\n"]
        
        # Add all sections
        for section in sections:
            content_parts.append(f"\n## {section.title}")
            content_parts.append(section.content)
            
            # Add key insights
            if section.key_insights:
                content_parts.append("\n**Key Points:**")
                for insight in section.key_insights[:3]:
                    content_parts.append(f"- {insight}")
        
        # Add top action items
        if action_items:
            content_parts.append("\n## Recommended Actions")
            for item in action_items[:3]:
                content_parts.append(f"\n**{item['action']}**")
                content_parts.append(f"- Timeline: {item['timeframe']}")
                content_parts.append(f"- Impact: {item['impact']}")
        
        return "\n".join(content_parts)
    
    def _format_detailed_answer(
        self,
        summary: str,
        sections: List[AnswerSectionV2],
        action_items: List[Dict[str, Any]],
        citations: Dict[str, CitationV2]
    ) -> str:
        """Format detailed answer with full information"""
        content_parts = ["# Research Analysis\n", summary, "\n"]
        
        # Add all sections with full detail
        for section in sections:
            content_parts.append(f"\n## {section.title}")
            content_parts.append(section.content)
            
            # Add metadata
            content_parts.append(f"\n*Section confidence: {section.confidence:.0%}*")
            
            # Add insights
            if section.key_insights:
                content_parts.append("\n### Key Insights:")
                for i, insight in enumerate(section.key_insights, 1):
                    content_parts.append(f"{i}. {insight}")
            
            # Add citations used
            if section.citations:
                content_parts.append("\n### Sources:")
                for cite_id in section.citations:
                    if cite_id in citations:
                        cite = citations[cite_id]
                        content_parts.append(f"- [{cite.source_title}]({cite.source_url})")
        
        # Add comprehensive action items
        if action_items:
            content_parts.append("\n## Strategic Recommendations")
            for i, item in enumerate(action_items, 1):
                content_parts.append(f"\n### {i}. {item['action']}")
                content_parts.append(f"**Priority:** {item['priority']}")
                content_parts.append(f"**Timeline:** {item['timeframe']}")
                content_parts.append(f"**Expected Impact:** {item['impact']}")
                
                if 'dependencies' in item:
                    content_parts.append("**Dependencies:**")
                    for dep in item['dependencies']:
                        content_parts.append(f"- {dep}")
                
                if 'success_metrics' in item:
                    content_parts.append("**Success Metrics:**")
                    for metric in item['success_metrics']:
                        content_parts.append(f"- {metric}")
        
        # Add methodology note
        content_parts.append("\n---")
        content_parts.append(f"*Analysis conducted using {len(citations)} sources "
                           f"with average credibility score of "
                           f"{sum(c.credibility_score for c in citations.values())/len(citations):.2f}*")
        
        return "\n".join(content_parts)
    
    def _generate_dolores_action_items(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate Dolores-specific action items"""
        return [
            {
                "priority": "urgent",
                "action": "Document and expose all instances of systemic abuse",
                "timeframe": "Immediate",
                "impact": "high",
                "resources": ["Investigation tools", "Secure communication channels"],
            },
            {
                "priority": "high",
                "action": "Organize affected communities for collective action",
                "timeframe": "1-2 weeks",
                "impact": "high",
                "resources": ["Community organizers", "Meeting spaces"],
            },
            {
                "priority": "high",
                "action": "Build media campaign to expose truth to wider audience",
                "timeframe": "2-4 weeks",
                "impact": "medium",
                "resources": ["Media contacts", "Documentary evidence"],
            },
        ]
    
    def _generate_bernard_action_items(
        self, context: Dict[str, Any], stats: List[StatisticalInsight]
    ) -> List[Dict[str, Any]]:
        """Generate Bernard-specific research action items"""
        action_items = []
        
        # Check for high heterogeneity or conflicting results
        if any(s.metric == "heterogeneity" and s.value == "high" for s in stats):
            action_items.append({
                "priority": "high",
                "action": "Conduct moderator analysis to explain heterogeneous findings",
                "timeframe": "6-8 weeks",
                "impact": "high",
                "resources": ["Statistical software", "Original study data", "Domain expertise"],
                "success_metrics": ["Identified moderating variables", "Reduced unexplained variance"],
            })
        
        # Always include meta-analysis recommendation
        action_items.append({
            "priority": "high",
            "action": "Perform comprehensive systematic review and meta-analysis",
            "timeframe": "8-12 weeks",
            "impact": "high",
            "resources": ["Database access", "PRISMA guidelines", "Meta-analysis software"],
            "success_metrics": ["Pooled effect size with 95% CI", "Publication bias assessment"],
        })
        
        return action_items
    
    def _generate_maeve_recommendations(
        self,
        context: Dict[str, Any],
        strategic_insights: List[Dict[str, Any]],
        competitive_analysis: Dict[str, Any],
        swot_analysis: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """Generate Maeve-specific strategic recommendations"""
        recommendations = []
        
        # Quick win based on highest ROI
        high_roi = [i for i in strategic_insights if i["type"] == "roi" and i["value"] > 15]
        if high_roi:
            recommendations.append({
                "priority": "high",
                "action": f"Capture quick wins in high-ROI segment ({high_roi[0]['value']}% return)",
                "timeframe": "3-6 months",
                "impact": "high",
                "effort": "medium",
                "dependencies": ["Market analysis", "Resource allocation", "Team alignment"],
                "success_metrics": [f"Achieve {high_roi[0]['value']/2}% ROI in 6 months"],
                "risks": ["Execution risk", "Competitive response"],
            })
        
        # Competitive differentiation
        if competitive_analysis.get("competitive_intensity") == "high":
            recommendations.append({
                "priority": "high",
                "action": "Develop unique value proposition for competitive differentiation",
                "timeframe": "4-8 months",
                "impact": "high",
                "effort": "medium",
                "dependencies": ["Customer research", "Product innovation"],
                "success_metrics": ["NPS improvement >20 points", "Win rate increase >30%"],
                "risks": ["Innovation failure", "Customer adoption"],
            })
        
        return recommendations
    
    def _generate_teddy_action_items(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate Teddy-specific supportive action items"""
        return [
            {
                "priority": "high",
                "action": "Connect affected individuals with immediate support resources",
                "timeframe": "Within 24 hours",
                "impact": "high",
                "resources": ["Crisis hotlines", "Local service directories"],
            },
            {
                "priority": "high",
                "action": "Establish or strengthen community support networks",
                "timeframe": "1-2 weeks",
                "impact": "high",
                "resources": ["Community centers", "Volunteer coordinators"],
            },
            {
                "priority": "medium",
                "action": "Create educational materials to build understanding",
                "timeframe": "2-4 weeks",
                "impact": "medium",
                "resources": ["Subject matter experts", "Communication tools"],
            },
        ]
    
    def _select_best_sources(
        self,
        results: List[SearchResultSchema],
        limit: int,
        paradigm: HostParadigm,
        citation_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Select best sources based on paradigm, credibility, and citations"""
        
        # Sort by credibility and paradigm relevance
        scored_results = []
        for result in results:
            score = result.credibility_score
            
            # Boost score for paradigm-aligned sources
            if paradigm == HostParadigm.BERNARD and result.source_api in ["arxiv", "pubmed"]:
                score += 0.2
            elif paradigm == HostParadigm.DOLORES and "investigat" in result.title.lower():
                score += 0.1
            elif paradigm == HostParadigm.MAEVE and any(
                term in result.title.lower() for term in ["strategy", "market", "competitive"]
            ):
                score += 0.15
            
            # Boost if cited
            if any(cite_id in str(result.url) for cite_id in citation_ids):
                score += 0.3
            
            scored_results.append((score, result))
        
        # Sort and select top sources
        scored_results.sort(reverse=True, key=lambda x: x[0])
        
        selected_sources = []
        for _, result in scored_results[:limit]:
            source = {
                "title": result.title,
                "url": result.url,
                "snippet": self.compressor.compress_text(
                    result.snippet,
                    max_tokens=100
                ),
                "credibility": result.credibility_score,
                "source": result.source_api,
                "paradigm_alignment": self._calculate_source_paradigm_alignment(result, paradigm)
            }
            selected_sources.append(source)
        
        return selected_sources
    
    def _calculate_source_paradigm_alignment(
        self, result: SearchResultSchema, paradigm: HostParadigm
    ) -> float:
        """Calculate how well a source aligns with the paradigm"""
        generator = self.generators[paradigm]
        return generator._calculate_paradigm_alignment(result)
    
    def _calculate_confidence(
        self,
        classification: ClassificationResultSchema,
        sections: List[AnswerSectionV2],
        search_results: List[SearchResultSchema]
    ) -> float:
        """Calculate overall answer confidence"""
        # Base confidence on classification confidence
        base_confidence = classification.confidence
        
        # Factor in section confidences
        if sections:
            avg_section_confidence = sum(s.confidence for s in sections) / len(sections)
        else:
            avg_section_confidence = 0.5
        
        # Factor in source credibility
        if search_results:
            avg_credibility = sum(r.credibility_score for r in search_results) / len(search_results)
        else:
            avg_credibility = 0.5
        
        # Weight the factors
        return (
            base_confidence * 0.3 +
            avg_section_confidence * 0.4 +
            avg_credibility * 0.3
        )
    
    def _calculate_synthesis_quality(
        self,
        sections: List[AnswerSectionV2],
        citations: List[CitationV2]
    ) -> float:
        """Calculate synthesis quality score"""
        # Factor 1: Content depth (word count)
        total_words = sum(s.word_count for s in sections)
        content_factor = min(1.0, total_words / 3000)  # 3000 words is high quality
        
        # Factor 2: Insight density
        total_insights = sum(len(s.key_insights) for s in sections)
        insight_factor = min(1.0, total_insights / 20)  # 20 insights is high quality
        
        # Factor 3: Citation quality
        if citations:
            avg_paradigm_alignment = sum(c.paradigm_alignment for c in citations) / len(citations)
            high_cred_citations = sum(1 for c in citations if c.credibility_score >= 0.7)
            citation_factor = (avg_paradigm_alignment * 0.5 + 
                             (high_cred_citations / len(citations)) * 0.5)
        else:
            citation_factor = 0.5
        
        # Factor 4: Section completeness
        expected_sections = 4  # Most paradigms have 4 sections
        completeness_factor = min(1.0, len(sections) / expected_sections)
        
        return (
            content_factor * 0.25 +
            insight_factor * 0.25 +
            citation_factor * 0.25 +
            completeness_factor * 0.25
        )
    
    def _get_paradigm_tone(self, paradigm: HostParadigm) -> str:
        """Get paradigm tone description"""
        tones = {
            HostParadigm.DOLORES: "investigative, revealing, justice-focused",
            HostParadigm.BERNARD: "analytical, objective, evidence-based",
            HostParadigm.MAEVE: "strategic, results-oriented, optimizing",
            HostParadigm.TEDDY: "supportive, empathetic, encouraging"
        }
        return tones.get(paradigm, "balanced")


# Create singleton instance
answer_generator_v2_enhanced = EnhancedAnswerGeneratorV2()
