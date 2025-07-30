"""
Enhanced Answer Generation System - Advanced Bernard and Maeve implementations
Includes sophisticated empirical analysis and strategic planning features
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio
import logging
import re
from collections import Counter
from dataclasses import dataclass

from .answer_generator import (
    BaseAnswerGenerator,
    Citation,
    AnswerSection,
    GeneratedAnswer,
    SynthesisContext,
)
from .llm_client import llm_client

logger = logging.getLogger(__name__)


@dataclass
class StatisticalInsight:
    """Represents a statistical finding from data analysis"""
    metric: str
    value: float
    unit: str
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    sample_size: Optional[int] = None


@dataclass
class StrategicRecommendation:
    """Represents a strategic recommendation with implementation details"""
    title: str
    description: str
    impact: str  # high, medium, low
    effort: str  # high, medium, low
    timeline: str
    dependencies: List[str]
    success_metrics: List[str]
    risks: List[str]


class EnhancedBernardAnswerGenerator(BaseAnswerGenerator):
    """Enhanced analytical paradigm answer generator with sophisticated empirical analysis"""

    def __init__(self):
        super().__init__("bernard")
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

    async def generate_answer(self, context: SynthesisContext) -> GeneratedAnswer:
        """Generate enhanced analytical answer with sophisticated data analysis"""
        start_time = datetime.now()
        
        # Reset citations
        self.citation_counter = 0
        self.citations = {}
        
        # Extract statistical insights from search results
        statistical_insights = await self._extract_statistical_insights(context.search_results)
        
        # Perform meta-analysis if multiple studies found
        meta_analysis = await self._perform_meta_analysis(context.search_results)
        
        # Generate sections with enhanced analytical depth
        sections = []
        for section_def in self.get_section_structure():
            section = await self._generate_enhanced_section(
                context, section_def, statistical_insights, meta_analysis
            )
            sections.append(section)
        
        # Generate evidence-based summary
        summary = await self._generate_analytical_summary(context, sections, statistical_insights)
        
        # Generate research-oriented action items
        action_items = self._generate_research_action_items(context, statistical_insights)
        
        # Calculate rigorous confidence score
        confidence_score = self._calculate_analytical_confidence(
            context, sections, statistical_insights, meta_analysis
        )
        
        synthesis_quality = self._calculate_synthesis_quality(sections, statistical_insights)
        
        return GeneratedAnswer(
            research_id=context.metadata.get("research_id", "unknown"),
            query=context.query,
            paradigm=self.paradigm,
            summary=summary,
            sections=sections,
            action_items=action_items,
            citations=self.citations,
            confidence_score=confidence_score,
            synthesis_quality=synthesis_quality,
            generation_time=(datetime.now() - start_time).total_seconds(),
            metadata={
                "statistical_insights": len(statistical_insights),
                "meta_analysis_performed": meta_analysis is not None,
                "peer_reviewed_sources": self._count_peer_reviewed(context.search_results),
                "total_sample_size": sum(s.sample_size or 0 for s in statistical_insights),
            },
        )

    async def _extract_statistical_insights(
        self, search_results: List[Dict[str, Any]]
    ) -> List[StatisticalInsight]:
        """Extract statistical insights from search results"""
        insights = []
        
        for result in search_results:
            text = f"{result.get('title', '')} {result.get('snippet', '')}"
            
            # Extract correlations
            for match in re.finditer(self.statistical_patterns["correlation"], text):
                insights.append(StatisticalInsight(
                    metric="correlation",
                    value=float(match.group(1)),
                    unit="r",
                ))
            
            # Extract percentages with context
            for match in re.finditer(self.statistical_patterns["percentage"], text):
                context_start = max(0, match.start() - 50)
                context_text = text[context_start:match.start()]
                insights.append(StatisticalInsight(
                    metric=self._extract_percentage_context(context_text),
                    value=float(match.group(1)),
                    unit="%",
                ))
            
            # Extract p-values
            for match in re.finditer(self.statistical_patterns["p_value"], text):
                insights.append(StatisticalInsight(
                    metric="significance",
                    value=float(match.group(1)),
                    unit="p-value",
                ))
            
            # Extract sample sizes
            for match in re.finditer(self.statistical_patterns["sample_size"], text):
                insights.append(StatisticalInsight(
                    metric="sample_size",
                    value=float(match.group(1)),
                    unit="participants",
                    sample_size=int(match.group(1)),
                ))
        
        return insights

    async def _perform_meta_analysis(
        self, search_results: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Perform basic meta-analysis on multiple studies"""
        # Filter for academic/research sources
        research_sources = [
            r for r in search_results
            if any(domain in r.get("domain", "").lower() 
                   for domain in ["arxiv", "pubmed", "nature", "science", "journal"])
        ]
        
        if len(research_sources) < 3:
            return None
        
        # Extract effect sizes and sample sizes
        effect_sizes = []
        sample_sizes = []
        
        for source in research_sources:
            text = f"{source.get('title', '')} {source.get('snippet', '')}"
            
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
        context: SynthesisContext, 
        section_def: Dict[str, Any],
        statistical_insights: List[StatisticalInsight],
        meta_analysis: Optional[Dict[str, Any]]
    ) -> AnswerSection:
        """Generate section with enhanced analytical content"""
        # Filter insights relevant to this section
        section_insights = self._filter_insights_for_section(
            statistical_insights, section_def["title"]
        )
        
        # Create enhanced prompt with statistical context
        section_prompt = f"""
Write the "{section_def['title']}" section focusing on: {section_def['focus']}

Query: {context.query}

Statistical insights available:
{self._format_statistical_insights(section_insights[:10])}

{f"Meta-analysis results: {meta_analysis}" if meta_analysis else ""}

Requirements:
- Include specific quantitative findings with effect sizes and confidence intervals
- Distinguish correlation from causation explicitly
- Address methodological limitations and potential biases
- Use precise scientific language and proper statistical terminology
- Length: approximately {int(context.max_length * section_def['weight'])} words
"""

        content = await llm_client.generate_paradigm_content(
            prompt=section_prompt,
            paradigm=self.paradigm,
            max_tokens=int(context.max_length * section_def["weight"] * 2),
            temperature=0.3,  # Lower temperature for analytical precision
        )
        
        # Create citations with enhanced metadata
        citation_ids = self._create_analytical_citations(
            content, context.search_results, section_insights
        )
        
        # Extract quantitative insights
        insights = self._extract_quantitative_insights(content, section_insights)
        
        # Calculate section confidence based on evidence quality
        section_confidence = self._calculate_section_confidence(
            section_insights, citation_ids, content
        )
        
        return AnswerSection(
            title=section_def["title"],
            paradigm=self.paradigm,
            content=content,
            confidence=section_confidence,
            citations=citation_ids,
            word_count=len(content.split()),
            key_insights=insights,
            metadata={
                "statistical_insights_used": len(section_insights),
                "quantitative_claims": self._count_quantitative_claims(content),
            }
        )

    def _extract_percentage_context(self, context_text: str) -> str:
        """Extract what a percentage refers to from surrounding context"""
        # Common patterns for percentage contexts
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
        sources: List[Dict[str, Any]], 
        insights: List[StatisticalInsight]
    ) -> List[str]:
        """Create citations with enhanced analytical metadata"""
        citation_ids = []
        
        # Prioritize peer-reviewed sources
        peer_reviewed = [
            s for s in sources 
            if any(domain in s.get("domain", "").lower() 
                   for domain in ["arxiv", "pubmed", "nature", "science"])
        ]
        
        for source in (peer_reviewed + sources)[:6]:  # More citations for analytical
            # Determine citation type based on content
            if "meta-analysis" in source.get("title", "").lower():
                citation_type = "meta-analysis"
            elif "systematic review" in source.get("title", "").lower():
                citation_type = "systematic-review"
            elif any(domain in source.get("domain", "").lower() for domain in ["arxiv", "pubmed"]):
                citation_type = "peer-reviewed"
            else:
                citation_type = "data"
            
            citation = self.create_citation(source, citation_type)
            citation.metadata = {
                "study_type": self._identify_study_type(source),
                "sample_size": self._extract_sample_size(source),
                "publication_year": self._extract_year(source),
            }
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

    def _identify_study_type(self, source: Dict[str, Any]) -> str:
        """Identify the type of study from source"""
        text = f"{source.get('title', '')} {source.get('snippet', '')}".lower()
        
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

    def _extract_sample_size(self, source: Dict[str, Any]) -> Optional[int]:
        """Extract sample size from source"""
        text = f"{source.get('title', '')} {source.get('snippet', '')}"
        match = re.search(self.statistical_patterns["sample_size"], text)
        return int(match.group(1)) if match else None

    def _extract_year(self, source: Dict[str, Any]) -> Optional[int]:
        """Extract publication year from source"""
        text = f"{source.get('title', '')} {source.get('snippet', '')}"
        year_match = re.search(r"(19|20)\d{2}", text)
        return int(year_match.group(0)) if year_match else None

    async def _generate_analytical_summary(
        self, 
        context: SynthesisContext, 
        sections: List[AnswerSection],
        statistical_insights: List[StatisticalInsight]
    ) -> str:
        """Generate evidence-based analytical summary"""
        # Aggregate key findings
        all_insights = []
        for section in sections:
            all_insights.extend(section.key_insights)
        
        # Get top statistical findings
        top_stats = sorted(
            statistical_insights,
            key=lambda x: abs(x.value) if x.metric in ["effect_size", "correlation"] else 0,
            reverse=True
        )[:3]
        
        summary_prompt = f"""
Create a rigorous executive summary for the research question: "{context.query}"

Key quantitative findings:
{self._format_statistical_insights(top_stats)}

Section insights:
{chr(10).join(f"- {insight}" for insight in all_insights[:5])}

Requirements:
- Lead with the most significant empirical finding
- Include effect sizes and confidence intervals where available
- Acknowledge any conflicting evidence or heterogeneity
- End with evidence quality assessment
- Length: 3-4 sentences, highly technical and precise
"""

        return await llm_client.generate_paradigm_content(
            prompt=summary_prompt,
            paradigm=self.paradigm,
            max_tokens=300,
            temperature=0.3,
        )

    def _generate_research_action_items(
        self, context: SynthesisContext, insights: List[StatisticalInsight]
    ) -> List[Dict[str, Any]]:
        """Generate research-oriented action items"""
        action_items = []
        
        # Check for high heterogeneity or conflicting results
        if any(i.metric == "heterogeneity" and i.value == "high" for i in insights):
            action_items.append({
                "priority": "high",
                "action": "Conduct moderator analysis to explain heterogeneous findings",
                "timeframe": "6-8 weeks",
                "impact": "high",
                "resources": ["Statistical software", "Original study data", "Domain expertise"],
                "success_metrics": ["Identified moderating variables", "Reduced unexplained variance"],
            })
        
        # Check for small sample sizes
        sample_sizes = [i.sample_size for i in insights if i.sample_size]
        if sample_sizes and max(sample_sizes) < 100:
            action_items.append({
                "priority": "high",
                "action": "Design and conduct adequately powered replication study",
                "timeframe": "12-16 weeks",
                "impact": "high",
                "resources": ["Funding for n=500+ participants", "IRB approval", "Research team"],
                "success_metrics": ["80% statistical power", "Pre-registered protocol"],
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
        
        # Add methodology improvement
        action_items.append({
            "priority": "medium",
            "action": "Develop standardized measurement protocol for key variables",
            "timeframe": "4-6 weeks",
            "impact": "medium",
            "resources": ["Psychometric expertise", "Validation sample"],
            "success_metrics": ["Reliability > 0.8", "Convergent validity established"],
        })
        
        return action_items

    def _calculate_analytical_confidence(
        self, 
        context: SynthesisContext, 
        sections: List[AnswerSection],
        insights: List[StatisticalInsight],
        meta_analysis: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate confidence with rigorous analytical criteria"""
        # Factor 1: Evidence quality (40%)
        peer_reviewed_ratio = self._count_peer_reviewed(context.search_results) / len(context.search_results)
        evidence_quality = peer_reviewed_ratio * 0.4
        
        # Factor 2: Statistical rigor (30%)
        significant_findings = len([i for i in insights if i.p_value and i.p_value < 0.05])
        large_samples = len([i for i in insights if i.sample_size and i.sample_size > 500])
        statistical_rigor = min(1.0, (significant_findings + large_samples) / 10) * 0.3
        
        # Factor 3: Convergence of findings (20%)
        if meta_analysis and meta_analysis.get("heterogeneity") == "low":
            convergence = 0.2
        elif meta_analysis and meta_analysis.get("heterogeneity") == "moderate":
            convergence = 0.15
        else:
            convergence = 0.1
        
        # Factor 4: Methodological transparency (10%)
        transparency = sum(s.confidence for s in sections) / len(sections) * 0.1
        
        return evidence_quality + statistical_rigor + convergence + transparency

    def _calculate_synthesis_quality(
        self, sections: List[AnswerSection], insights: List[StatisticalInsight]
    ) -> float:
        """Calculate synthesis quality for analytical paradigm"""
        # Quantitative content density
        total_quant_claims = sum(
            s.metadata.get("quantitative_claims", 0) for s in sections
            if s.metadata
        )
        quant_density = min(1.0, total_quant_claims / 30)
        
        # Statistical insight utilization
        insight_utilization = min(1.0, len(insights) / 20)
        
        # Citation quality
        total_citations = sum(len(s.citations) for s in sections)
        citation_quality = min(1.0, total_citations / 20)
        
        # Methodological discussion
        method_sections = [
            s for s in sections 
            if "method" in s.title.lower() or "assessment" in s.title.lower()
        ]
        method_quality = 1.0 if method_sections else 0.5
        
        return (
            quant_density * 0.3 +
            insight_utilization * 0.3 +
            citation_quality * 0.2 +
            method_quality * 0.2
        )

    def get_synthesis_prompt(self, context: SynthesisContext) -> str:
        """Get analytical paradigm-specific synthesis prompt"""
        return f"""
As an empirical researcher focused on data-driven insights, synthesize these search results
about "{context.query}" into a rigorous analytical narrative that:
1. Presents quantitative findings with statistical significance
2. Distinguishes correlation from causation
3. Evaluates methodological quality and research design
4. Identifies patterns, trends, and empirical relationships
5. Acknowledges limitations and confounding variables

Prioritize peer-reviewed research, meta-analyses, and studies with robust sample sizes.
Use precise scientific language and include effect sizes, confidence intervals, and p-values where available.
"""

    def _get_alignment_keywords(self) -> List[str]:
        """Get analytical paradigm-specific alignment keywords"""
        return [
            "empirical",
            "statistical",
            "data",
            "analysis",
            "research",
            "evidence",
            "methodology",
            "correlation",
            "significance",
            "hypothesis",
            "variable",
            "quantitative",
            "systematic",
            "peer-reviewed",
            "meta-analysis",
            "sample",
            "control",
            "experiment",
            "measurement",
            "validity"
        ]


class EnhancedMaeveAnswerGenerator(BaseAnswerGenerator):
    """Enhanced strategic paradigm answer generator with sophisticated business analysis"""

    def __init__(self):
        super().__init__("maeve")
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

    async def generate_answer(self, context: SynthesisContext) -> GeneratedAnswer:
        """Generate enhanced strategic answer with sophisticated business analysis"""
        start_time = datetime.now()
        
        # Reset citations
        self.citation_counter = 0
        self.citations = {}
        
        # Extract strategic insights
        strategic_insights = await self._extract_strategic_insights(context.search_results)
        
        # Perform competitive analysis
        competitive_analysis = await self._perform_competitive_analysis(
            context.query, context.search_results
        )
        
        # Generate SWOT analysis
        swot_analysis = await self._generate_swot_analysis(
            context.query, context.search_results, strategic_insights
        )
        
        # Generate sections with strategic depth
        sections = []
        for section_def in self.get_section_structure():
            section = await self._generate_strategic_section(
                context, section_def, strategic_insights, competitive_analysis, swot_analysis
            )
            sections.append(section)
        
        # Generate executive strategic summary
        summary = await self._generate_strategic_summary(
            context, sections, strategic_insights, swot_analysis
        )
        
        # Generate strategic recommendations
        action_items = self._generate_strategic_recommendations(
            context, strategic_insights, competitive_analysis, swot_analysis
        )
        
        # Calculate strategic confidence
        confidence_score = self._calculate_strategic_confidence(
            context, sections, strategic_insights, competitive_analysis
        )
        
        synthesis_quality = self._calculate_strategic_synthesis_quality(
            sections, strategic_insights, action_items
        )
        
        return GeneratedAnswer(
            research_id=context.metadata.get("research_id", "unknown"),
            query=context.query,
            paradigm=self.paradigm,
            summary=summary,
            sections=sections,
            action_items=action_items,
            citations=self.citations,
            confidence_score=confidence_score,
            synthesis_quality=synthesis_quality,
            generation_time=(datetime.now() - start_time).total_seconds(),
            metadata={
                "strategic_insights": len(strategic_insights),
                "competitive_analysis": competitive_analysis,
                "swot_completed": True,
                "recommendations": len(action_items),
            },
        )

    async def _extract_strategic_insights(
        self, search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract strategic business insights from search results"""
        insights = []
        
        for result in search_results:
            text = f"{result.get('title', '')} {result.get('snippet', '')}"
            
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
        self, query: str, search_results: List[Dict[str, Any]]
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
            text = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
            
            # Identify market leaders
            if any(keyword in text for keyword in competitive_keywords["leaders"]):
                market_leaders.append(result.get("title", "Unknown"))
            
            # Identify competitors
            if any(keyword in text for keyword in competitive_keywords["competitors"]):
                competitors.append(result.get("title", "Unknown"))
            
            # Identify disruptors
            if any(keyword in text for keyword in competitive_keywords["disruptors"]):
                disruptors.append(result.get("title", "Unknown"))
        
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
        search_results: List[Dict[str, Any]],
        strategic_insights: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Generate SWOT analysis from available data"""
        swot_prompt = f"""
Based on the research query "{query}" and available market data, generate a SWOT analysis.

Key insights:
- Market growth rates: {[i['value'] for i in strategic_insights if i['type'] == 'growth_rate']}
- ROI potential: {[i['value'] for i in strategic_insights if i['type'] == 'roi']}
- Market size: {[i for i in strategic_insights if i['type'] == 'market_size']}

Provide 3-4 items for each category.
"""

        # For now, generate based on patterns in search results
        swot = {
            "strengths": [],
            "weaknesses": [],
            "opportunities": [],
            "threats": [],
        }
        
        # Analyze search results for SWOT elements
        for result in search_results[:10]:
            text = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
            
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
        context: SynthesisContext,
        section_def: Dict[str, Any],
        strategic_insights: List[Dict[str, Any]],
        competitive_analysis: Dict[str, Any],
        swot_analysis: Dict[str, List[str]],
    ) -> AnswerSection:
        """Generate strategic section with business insights"""
        # Filter insights relevant to section
        section_insights = self._filter_strategic_insights(strategic_insights, section_def["title"])
        
        # Create strategic prompt
        section_prompt = f"""
Write the "{section_def['title']}" section focusing on: {section_def['focus']}

Query: {context.query}

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
- Length: approximately {int(context.max_length * section_def['weight'])} words
"""

        content = await llm_client.generate_paradigm_content(
            prompt=section_prompt,
            paradigm=self.paradigm,
            max_tokens=int(context.max_length * section_def["weight"] * 2),
            temperature=0.5,  # Balanced for strategic creativity
        )
        
        # Create strategic citations
        citation_ids = self._create_strategic_citations(
            content, context.search_results, section_insights
        )
        
        # Extract strategic insights
        insights = self._extract_strategic_recommendations(content, section_insights)
        
        # Calculate section confidence
        section_confidence = self._calculate_section_strategic_confidence(
            section_insights, competitive_analysis, content
        )
        
        return AnswerSection(
            title=section_def["title"],
            paradigm=self.paradigm,
            content=content,
            confidence=section_confidence,
            citations=citation_ids,
            word_count=len(content.split()),
            key_insights=insights,
            metadata={
                "strategic_insights_used": len(section_insights),
                "competitive_factors": len(competitive_analysis.get("key_competitors", [])),
            }
        )

    def _assess_competitive_intensity(self, search_results: List[Dict[str, Any]]) -> str:
        """Assess competitive intensity from search results"""
        competitive_terms = ["competitor", "competition", "rival", "market share", "competing"]
        competitive_mentions = 0
        
        for result in search_results:
            text = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
            competitive_mentions += sum(1 for term in competitive_terms if term in text)
        
        avg_mentions = competitive_mentions / len(search_results) if search_results else 0
        
        if avg_mentions > 2:
            return "high"
        elif avg_mentions > 1:
            return "moderate"
        else:
            return "low"

    def _assess_market_concentration(self, search_results: List[Dict[str, Any]]) -> str:
        """Assess market concentration from search results"""
        concentration_indicators = ["monopoly", "duopoly", "dominant", "fragmented", "consolidated"]
        
        for result in search_results:
            text = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
            if "monopoly" in text or "dominant player" in text:
                return "highly_concentrated"
            elif "duopoly" in text or "two players" in text:
                return "concentrated"
            elif "fragmented" in text or "many players" in text:
                return "fragmented"
        
        return "moderate"

    def _extract_swot_item(self, text: str, category: str) -> str:
        """Extract a SWOT item from text"""
        # Simple extraction - in production, use NLP
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
        self, content: str, sources: List[Dict[str, Any]], insights: List[Dict[str, Any]]
    ) -> List[str]:
        """Create citations with strategic metadata"""
        citation_ids = []
        
        # Prioritize business and industry sources
        business_sources = [
            s for s in sources
            if any(domain in s.get("domain", "").lower() 
                   for domain in ["forbes", "mckinsey", "gartner", "harvard", "deloitte", "pwc"])
        ]
        
        for source in (business_sources + sources)[:5]:
            citation_type = "strategic" if source in business_sources else "data"
            
            citation = self.create_citation(source, citation_type)
            citation.metadata = {
                "source_type": self._identify_source_type(source),
                "publication_date": self._extract_year(source),
                "strategic_relevance": self._assess_strategic_relevance(source, insights),
            }
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

    def _identify_source_type(self, source: Dict[str, Any]) -> str:
        """Identify the type of business source"""
        domain = source.get("domain", "").lower()
        
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
        self, source: Dict[str, Any], insights: List[Dict[str, Any]]
    ) -> float:
        """Assess how strategically relevant a source is"""
        relevance_score = 0.5  # Base score
        
        # Check if source contains strategic insights
        text = f"{source.get('title', '')} {source.get('snippet', '')}".lower()
        
        strategic_terms = ["strategy", "competitive", "market share", "growth", "roi", "opportunity"]
        term_matches = sum(1 for term in strategic_terms if term in text)
        relevance_score += min(0.3, term_matches * 0.05)
        
        # Check if source provided quantitative insights
        source_insights = [i for i in insights if source.get("url") in i.get("context", "")]
        if source_insights:
            relevance_score += 0.2
        
        return min(1.0, relevance_score)

    async def _generate_strategic_summary(
        self,
        context: SynthesisContext,
        sections: List[AnswerSection],
        strategic_insights: List[Dict[str, Any]],
        swot_analysis: Dict[str, List[str]],
    ) -> str:
        """Generate executive strategic summary"""
        # Get top strategic metrics
        top_metrics = []
        for insight in strategic_insights[:5]:
            if insight["type"] == "market_size":
                top_metrics.append(f"${insight['value']}{insight['unit'][0]} market")
            elif insight["type"] == "growth_rate":
                top_metrics.append(f"{insight['value']}% CAGR")
            elif insight["type"] == "roi":
                top_metrics.append(f"{insight['value']}% ROI potential")
        
        summary_prompt = f"""
Create an executive strategic summary for: "{context.query}"

Key metrics: {', '.join(top_metrics)}
Top opportunity: {swot_analysis['opportunities'][0] if swot_analysis['opportunities'] else 'Market expansion'}
Main threat: {swot_analysis['threats'][0] if swot_analysis['threats'] else 'Competitive pressure'}

Requirements:
- Lead with the most compelling business opportunity
- Include 1-2 key metrics that support the opportunity
- Mention the critical success factor
- End with a clear strategic imperative
- Length: 3-4 sentences, action-oriented and decisive
"""

        return await llm_client.generate_paradigm_content(
            prompt=summary_prompt,
            paradigm=self.paradigm,
            max_tokens=300,
            temperature=0.5,
        )

    def _generate_strategic_recommendations(
        self,
        context: SynthesisContext,
        strategic_insights: List[Dict[str, Any]],
        competitive_analysis: Dict[str, Any],
        swot_analysis: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        """Generate strategic recommendations with implementation details"""
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
                "success_metrics": [f"Achieve {high_roi[0]['value']/2}% ROI in 6 months", "Market entry completed"],
                "risks": ["Execution risk", "Competitive response"],
            })
        
        # Market expansion based on growth rate
        high_growth = [i for i in strategic_insights if i["type"] == "growth_rate" and i["value"] > 10]
        if high_growth:
            recommendations.append({
                "priority": "high",
                "action": f"Enter high-growth market segment ({high_growth[0]['value']}% CAGR)",
                "timeframe": "6-12 months",
                "impact": "high",
                "effort": "high",
                "dependencies": ["Market research", "Product adaptation", "Channel development"],
                "success_metrics": ["10% market share in 12 months", "Revenue growth >20%"],
                "risks": ["Market saturation", "Resource constraints"],
            })
        
        # Competitive differentiation
        if competitive_analysis.get("competitive_intensity") == "high":
            recommendations.append({
                "priority": "high",
                "action": "Develop unique value proposition for competitive differentiation",
                "timeframe": "4-8 months",
                "impact": "high",
                "effort": "medium",
                "dependencies": ["Customer research", "Product innovation", "Brand positioning"],
                "success_metrics": ["NPS improvement >20 points", "Win rate increase >30%"],
                "risks": ["Innovation failure", "Customer adoption"],
            })
        
        # Operational excellence
        cost_savings = [i for i in strategic_insights if i["type"] == "cost_savings"]
        if cost_savings or True:  # Always include operational recommendation
            recommendations.append({
                "priority": "medium",
                "action": "Implement operational excellence program for margin improvement",
                "timeframe": "6-9 months",
                "impact": "medium",
                "effort": "medium",
                "dependencies": ["Process mapping", "Technology enablement", "Change management"],
                "success_metrics": ["15% cost reduction", "Process efficiency +25%"],
                "risks": ["Implementation complexity", "Organizational resistance"],
            })
        
        # Strategic partnership
        if swot_analysis.get("weaknesses") and len(swot_analysis["weaknesses"]) > 2:
            recommendations.append({
                "priority": "medium",
                "action": "Form strategic partnerships to address capability gaps",
                "timeframe": "3-6 months",
                "impact": "medium",
                "effort": "low",
                "dependencies": ["Partner identification", "Due diligence", "Contract negotiation"],
                "success_metrics": ["2-3 strategic partnerships", "Capability gaps closed"],
                "risks": ["Partner alignment", "IP concerns"],
            })
        
        return recommendations[:4]  # Return top 4 recommendations

    def _calculate_strategic_confidence(
        self,
        context: SynthesisContext,
        sections: List[AnswerSection],
        strategic_insights: List[Dict[str, Any]],
        competitive_analysis: Dict[str, Any],
    ) -> float:
        """Calculate strategic confidence score"""
        # Market data completeness (30%)
        market_metrics = ["market_size", "growth_rate", "market_share"]
        market_coverage = sum(
            1 for metric in market_metrics 
            if any(i["type"] == metric for i in strategic_insights)
        ) / len(market_metrics)
        market_factor = market_coverage * 0.3
        
        # Competitive intelligence quality (25%)
        competitive_completeness = sum([
            0.1 if competitive_analysis.get("market_leaders") else 0,
            0.1 if competitive_analysis.get("key_competitors") else 0,
            0.05 if competitive_analysis.get("competitive_intensity") else 0,
        ])
        
        # Source authority (25%)
        business_sources = [
            r for r in context.search_results
            if any(domain in r.get("domain", "").lower()
                   for domain in ["mckinsey", "gartner", "forbes", "harvard"])
        ]
        authority_factor = min(1.0, len(business_sources) / 3) * 0.25
        
        # Quantitative backing (20%)
        quant_insights = [i for i in strategic_insights if isinstance(i.get("value"), (int, float))]
        quant_factor = min(1.0, len(quant_insights) / 10) * 0.2
        
        return market_factor + competitive_completeness + authority_factor + quant_factor

    def _calculate_strategic_synthesis_quality(
        self, sections: List[AnswerSection], insights: List[Dict[str, Any]], recommendations: List[Dict[str, Any]]
    ) -> float:
        """Calculate synthesis quality for strategic paradigm"""
        # Strategic framework usage
        framework_count = sum(
            1 for s in sections
            if any(fw in s.content.lower() for fw in ["swot", "porter", "value chain", "bcg"])
        )
        framework_factor = min(1.0, framework_count / 3) * 0.25
        
        # Quantitative support
        total_insights = len(insights)
        insight_factor = min(1.0, total_insights / 15) * 0.25
        
        # Actionability of recommendations
        detailed_recs = [r for r in recommendations if len(r.get("dependencies", [])) > 2]
        action_factor = min(1.0, len(detailed_recs) / 3) * 0.25
        
        # Competitive awareness
        competitive_mentions = sum(
            1 for s in sections
            if "competit" in s.content.lower()
        )
        competitive_factor = min(1.0, competitive_mentions / 4) * 0.25
        
        return framework_factor + insight_factor + action_factor + competitive_factor

    def _extract_year(self, source: Dict[str, Any]) -> Optional[int]:
        """Extract publication year from source"""
        text = f"{source.get('title', '')} {source.get('snippet', '')}"
        year_match = re.search(r"(20\d{2})", text)
        return int(year_match.group(1)) if year_match else None

    def get_synthesis_prompt(self, context: SynthesisContext) -> str:
        """Get strategic paradigm-specific synthesis prompt"""
        return f"""
As a strategic business advisor focused on competitive advantage and value creation, synthesize these search results
about "{context.query}" into an actionable strategic narrative that:
1. Identifies market opportunities and competitive dynamics
2. Proposes value creation strategies with clear ROI potential
3. Outlines implementation frameworks with timelines and milestones
4. Addresses risks with mitigation strategies
5. Provides measurable KPIs and success metrics

Prioritize business intelligence sources, market research, and strategic frameworks.
Use executive-level language and include specific market data, growth rates, and financial projections where available.
"""

    def _get_alignment_keywords(self) -> List[str]:
        """Get strategic paradigm-specific alignment keywords"""
        return [
            "strategic",
            "competitive",
            "market",
            "business",
            "opportunity",
            "growth",
            "revenue",
            "profit",
            "ROI",
            "value",
            "advantage",
            "innovation",
            "disruption",
            "transformation",
            "optimization",
            "efficiency",
            "scalable",
            "sustainable",
            "leverage",
            "synergy"
        ]