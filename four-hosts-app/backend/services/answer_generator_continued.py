"""
Continuation of Answer Generation System - Bernard and Maeve generators
Plus the main orchestrator
"""

from .answer_generator import (
    BaseAnswerGenerator, Citation, AnswerSection, GeneratedAnswer,
    SynthesisContext, DoloresAnswerGenerator, TeddyAnswerGenerator
)
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)

# --- Bernard (Analytical) Answer Generator ---

class BernardAnswerGenerator(BaseAnswerGenerator):
    """Analytical paradigm answer generator"""
    
    def __init__(self):
        super().__init__("bernard")
        
    def get_section_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "title": "Executive Summary",
                "focus": "Key findings and statistical overview",
                "weight": 0.15
            },
            {
                "title": "Data Analysis",
                "focus": "Empirical evidence and statistical patterns",
                "weight": 0.35
            },
            {
                "title": "Causal Relationships",
                "focus": "Identified correlations and causations",
                "weight": 0.25
            },
            {
                "title": "Research Methodology",
                "focus": "Sources, limitations, and confidence levels",
                "weight": 0.15
            },
            {
                "title": "Future Research Directions",
                "focus": "Knowledge gaps and recommended studies",
                "weight": 0.1
            }
        ]
    
    def get_synthesis_prompt(self, context: SynthesisContext) -> str:
        return f"""
As an analytical researcher focused on empirical evidence, synthesize these search results
about "{context.query}" into a rigorous analysis that:

1. Presents statistical findings and quantitative data
2. Identifies patterns, correlations, and causal relationships
3. Maintains scientific objectivity and acknowledges limitations
4. Provides evidence-based conclusions and recommendations

Use precise, academic language with proper citations.
Focus on data, methodology, and reproducible findings.
Avoid speculation and clearly distinguish correlation from causation.

Results to synthesize:
{self._format_results_for_prompt(context.search_results[:10])}

Write a {context.max_length} word analysis with scientific rigor and clarity.
"""
    
    def _get_alignment_keywords(self) -> List[str]:
        return [
            "study", "research", "data", "analysis", "evidence",
            "statistical", "correlation", "methodology", "findings", "empirical",
            "peer-reviewed", "hypothesis", "measurement", "significant", "variable"
        ]
    
    def _format_results_for_prompt(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for LLM prompt"""
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(f"""
[{i}] {result.get('title', 'Untitled')}
Journal/Source: {result.get('domain', 'Unknown')}
Reliability Score: {result.get('credibility_score', 0.5):.2f}
Abstract: {result.get('snippet', 'No abstract available')[:200]}...
""")
        return "\n".join(formatted)
    
    async def generate_answer(self, context: SynthesisContext) -> GeneratedAnswer:
        """Generate analytical paradigm answer"""
        start_time = datetime.now()
        
        # Reset citations for new answer
        self.citation_counter = 0
        self.citations = {}
        
        # Generate sections
        sections = []
        for section_def in self.get_section_structure():
            section = await self._generate_section(context, section_def)
            sections.append(section)
        
        # Generate summary
        summary = await self._generate_summary(context, sections)
        
        # Generate action items
        action_items = self._generate_action_items(context)
        
        # Calculate confidence and quality scores
        confidence_score = self._calculate_confidence(context, sections)
        synthesis_quality = self._calculate_synthesis_quality(sections)
        
        # Create final answer
        metadata = context.metadata if hasattr(context, 'metadata') else {}
        answer = GeneratedAnswer(
            research_id=metadata.get('research_id', 'unknown'),
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
                "sources_used": len(context.search_results),
                "citations_created": len(self.citations),
                "paradigm_alignment": self._calculate_overall_alignment(context.search_results),
                "peer_reviewed_sources": self._count_peer_reviewed(context.search_results)
            }
        )
        
        return answer
    
    async def _generate_section(self, context: SynthesisContext, 
                               section_def: Dict[str, Any]) -> AnswerSection:
        """Generate a single section"""
        # Filter results relevant to this section
        relevant_results = self._filter_results_for_section(context.search_results, section_def)
        
        # Create section-specific prompt
        section_prompt = f"""
Write the "{section_def['title']}" section focusing on: {section_def['focus']}

Analyze these sources:
{self._format_results_for_prompt(relevant_results[:5])}

Length: approximately {int(context.max_length * section_def['weight'])} words.
Include specific data points, statistics, and citations.
Maintain academic objectivity and precision.
"""
        
        # Generate content (mock for now - would use LLM)
        content = await self._mock_generate_content(section_prompt, section_def)
        
        # Extract citations from content
        citation_ids = self._extract_and_create_citations(content, relevant_results)
        
        # Extract key insights
        insights = self.extract_key_insights(content, 3)
        
        return AnswerSection(
            title=section_def['title'],
            paradigm=self.paradigm,
            content=content,
            confidence=0.91,
            citations=citation_ids,
            word_count=len(content.split()),
            key_insights=insights
        )
    
    async def _mock_generate_content(self, prompt: str, section_def: Dict[str, Any]) -> str:
        """Mock content generation (replace with actual LLM call)"""
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        templates = {
            "Executive Summary": """Analysis of available data reveals statistically significant patterns with p<0.05 confidence levels. Meta-analysis of 47 studies (n=12,847) indicates strong correlations between key variables. Effect sizes range from moderate (d=0.5) to large (d=0.8), suggesting meaningful practical significance. However, heterogeneity across studies (I²=72%) necessitates careful interpretation of aggregate findings.""",
            
            "Data Analysis": """Quantitative analysis reveals three primary clusters of findings: (1) Linear regression models show R²=0.67 explanatory power for primary outcomes, with β coefficients ranging from 0.23 to 0.58. (2) Time-series analysis indicates cyclical patterns with 18-month periodicity (ARIMA model AIC=1247.3). (3) Multivariate analysis identifies interaction effects between variables X₁ and X₂ (F(3,296)=14.7, p<0.001). Bootstrap resampling confirms robustness of findings across 10,000 iterations.""",
            
            "Causal Relationships": """Instrumental variable analysis supports causal interpretation for 3 of 5 hypothesized relationships. Granger causality tests indicate temporal precedence (χ²=23.4, p<0.01). However, unmeasured confounders may account for up to 15% of observed variance based on sensitivity analysis. Mediation analysis reveals indirect effects through intermediate variables M₁ (ab=0.31, 95% CI [0.18, 0.44]) and M₂ (ab=0.22, 95% CI [0.09, 0.35]).""",
            
            "Research Methodology": """Studies employed mixed methodologies: 62% quantitative (RCTs, quasi-experimental), 23% qualitative (ethnographic, phenomenological), 15% mixed-methods. Sample sizes ranged from n=12 to n=4,831 (median=287). Geographic distribution spans 23 countries with overrepresentation of WEIRD populations (78%). Publication bias assessment (Egger's test p=0.08) suggests minimal systematic bias. Inter-rater reliability for coded variables averaged κ=0.84.""",
            
            "Future Research Directions": """Critical gaps remain in understanding long-term effects (studies >5 years: n=3) and cross-cultural validity. Recommended priorities: (1) Longitudinal cohort studies with minimum 10-year follow-up, (2) Replication in non-WEIRD contexts, (3) Investigation of moderating variables using machine learning approaches, (4) Development of standardized measurement instruments with established psychometric properties. Estimated sample size for adequate power (0.80): n=850 per condition."""
        }
        
        return templates.get(section_def['title'], "Content generation in progress...")
    
    def _filter_results_for_section(self, results: List[Dict[str, Any]], 
                                   section_def: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter results relevant to a specific section"""
        section_keywords = {
            "Executive Summary": ["summary", "overview", "key findings", "abstract"],
            "Data Analysis": ["data", "statistics", "analysis", "quantitative"],
            "Causal Relationships": ["causal", "correlation", "relationship", "effect"],
            "Research Methodology": ["methodology", "method", "approach", "design"],
            "Future Research Directions": ["future", "gap", "limitation", "recommendation"]
        }
        
        keywords = section_keywords.get(section_def['title'], [])
        
        relevant = []
        for result in results:
            text = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
            if any(keyword in text for keyword in keywords):
                relevant.append(result)
        
        return relevant or results[:5]
    
    def _extract_and_create_citations(self, content: str, sources: List[Dict[str, Any]]) -> List[str]:
        """Extract facts from content and create citations"""
        citation_ids = []
        
        # Create citations for top sources
        for source in sources[:4]:  # More citations for analytical
            citation = self.create_citation(source, "data")
            citation_ids.append(citation.id)
        
        return citation_ids
    
    async def _generate_summary(self, context: SynthesisContext, 
                               sections: List[AnswerSection]) -> str:
        """Generate executive summary"""
        return f"""Systematic analysis of {context.query} based on {len(context.search_results)} sources reveals statistically significant findings. 
Evidence supports moderate to strong effects across primary outcome variables with acceptable confidence intervals. 
Methodological rigor varies across studies, necessitating cautious interpretation of aggregate results."""
    
    def _generate_action_items(self, context: SynthesisContext) -> List[Dict[str, Any]]:
        """Generate paradigm-specific action items"""
        return [
            {
                "priority": "high",
                "action": "Conduct systematic meta-analysis of existing literature",
                "timeframe": "4-6 weeks",
                "impact": "high",
                "resources": ["Statistical software", "Database access", "Research team"]
            },
            {
                "priority": "high",
                "action": "Design replication study with enhanced methodology",
                "timeframe": "8-12 weeks",
                "impact": "high",
                "resources": ["IRB approval", "Funding", "Subject recruitment"]
            },
            {
                "priority": "medium",
                "action": "Develop standardized measurement protocols",
                "timeframe": "6-8 weeks",
                "impact": "medium",
                "resources": ["Domain experts", "Psychometric validation"]
            }
        ]
    
    def _count_peer_reviewed(self, results: List[Dict[str, Any]]) -> int:
        """Count peer-reviewed sources"""
        peer_reviewed_domains = ['arxiv.org', 'pubmed', 'nature.com', 'sciencedirect.com']
        count = 0
        for result in results:
            domain = result.get('domain', '').lower()
            if any(pr_domain in domain for pr_domain in peer_reviewed_domains):
                count += 1
        return count
    
    def _calculate_confidence(self, context: SynthesisContext, 
                             sections: List[AnswerSection]) -> float:
        """Calculate overall confidence score"""
        avg_credibility = sum(r.get('credibility_score', 0.5) 
                             for r in context.search_results) / len(context.search_results)
        
        peer_reviewed_ratio = self._count_peer_reviewed(context.search_results) / len(context.search_results)
        citation_factor = min(1.0, len(self.citations) / 15)
        section_confidence = sum(s.confidence for s in sections) / len(sections)
        
        return (avg_credibility * 0.3 + peer_reviewed_ratio * 0.3 + 
                citation_factor * 0.2 + section_confidence * 0.2)
    
    def _calculate_synthesis_quality(self, sections: List[AnswerSection]) -> float:
        """Calculate synthesis quality score"""
        insight_count = sum(len(s.key_insights) for s in sections)
        citation_count = sum(len(s.citations) for s in sections)
        
        insight_factor = min(1.0, insight_count / 20)
        citation_factor = min(1.0, citation_count / 25)
        
        return (insight_factor * 0.4 + citation_factor * 0.6)
    
    def _calculate_overall_alignment(self, results: List[Dict[str, Any]]) -> float:
        """Calculate overall paradigm alignment of results"""
        if not results:
            return 0.0
        
        alignments = [self._calculate_paradigm_alignment(r) for r in results]
        return sum(alignments) / len(alignments)

# --- Maeve (Strategic) Answer Generator ---

class MaeveAnswerGenerator(BaseAnswerGenerator):
    """Strategic paradigm answer generator"""
    
    def __init__(self):
        super().__init__("maeve")
        
    def get_section_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "title": "Strategic Overview",
                "focus": "Competitive landscape and opportunity assessment",
                "weight": 0.2
            },
            {
                "title": "Tactical Approaches",
                "focus": "Specific strategies and implementation methods",
                "weight": 0.3
            },
            {
                "title": "Resource Optimization",
                "focus": "Efficient allocation and leverage points",
                "weight": 0.2
            },
            {
                "title": "Success Metrics",
                "focus": "KPIs and measurement frameworks",
                "weight": 0.15
            },
            {
                "title": "Implementation Roadmap",
                "focus": "Timeline and action steps",
                "weight": 0.15
            }
        ]
    
    def get_synthesis_prompt(self, context: SynthesisContext) -> str:
        return f"""
As a strategic advisor focused on competitive advantage, synthesize these search results
about "{context.query}" into an actionable strategy that:

1. Identifies opportunities for competitive differentiation
2. Provides specific tactical recommendations
3. Optimizes resource allocation for maximum impact
4. Defines clear success metrics and milestones

Use crisp, action-oriented language focused on results.
Emphasize practical implementation over theory.
Provide specific frameworks and methodologies.

Results to synthesize:
{self._format_results_for_prompt(context.search_results[:10])}

Write a {context.max_length} word strategic analysis with clear action steps.
"""
    
    def _get_alignment_keywords(self) -> List[str]:
        return [
            "strategy", "competitive", "advantage", "optimize", "leverage",
            "tactical", "implementation", "framework", "metrics", "ROI",
            "efficiency", "market", "positioning", "execution", "performance"
        ]
    
    def _format_results_for_prompt(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for LLM prompt"""
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(f"""
[{i}] {result.get('title', 'Untitled')}
Source: {result.get('domain', 'Unknown')} | Authority: {result.get('credibility_score', 0.5):.2f}
Key Insight: {result.get('snippet', 'No insight available')[:200]}...
""")
        return "\n".join(formatted)
    
    async def generate_answer(self, context: SynthesisContext) -> GeneratedAnswer:
        """Generate strategic paradigm answer"""
        start_time = datetime.now()
        
        # Reset citations for new answer
        self.citation_counter = 0
        self.citations = {}
        
        # Generate sections
        sections = []
        for section_def in self.get_section_structure():
            section = await self._generate_section(context, section_def)
            sections.append(section)
        
        # Generate summary
        summary = await self._generate_summary(context, sections)
        
        # Generate action items
        action_items = self._generate_action_items(context)
        
        # Calculate confidence and quality scores
        confidence_score = self._calculate_confidence(context, sections)
        synthesis_quality = self._calculate_synthesis_quality(sections)
        
        # Create final answer
        metadata = context.metadata if hasattr(context, 'metadata') else {}
        answer = GeneratedAnswer(
            research_id=metadata.get('research_id', 'unknown'),
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
                "sources_used": len(context.search_results),
                "citations_created": len(self.citations),
                "paradigm_alignment": self._calculate_overall_alignment(context.search_results),
                "actionable_insights": self._count_actionable_insights(sections)
            }
        )
        
        return answer
    
    async def _generate_section(self, context: SynthesisContext, 
                               section_def: Dict[str, Any]) -> AnswerSection:
        """Generate a single section"""
        # Filter results relevant to this section
        relevant_results = self._filter_results_for_section(context.search_results, section_def)
        
        # Create section-specific prompt
        section_prompt = f"""
Write the "{section_def['title']}" section focusing on: {section_def['focus']}

Leverage these strategic insights:
{self._format_results_for_prompt(relevant_results[:5])}

Target length: {int(context.max_length * section_def['weight'])} words.
Focus on actionable recommendations and specific tactics.
Include frameworks, tools, and methodologies where applicable.
"""
        
        # Generate content (mock for now - would use LLM)
        content = await self._mock_generate_content(section_prompt, section_def)
        
        # Extract citations from content
        citation_ids = self._extract_and_create_citations(content, relevant_results)
        
        # Extract key insights
        insights = self.extract_key_insights(content, 4)
        
        return AnswerSection(
            title=section_def['title'],
            paradigm=self.paradigm,
            content=content,
            confidence=0.87,
            citations=citation_ids,
            word_count=len(content.split()),
            key_insights=insights
        )
    
    async def _mock_generate_content(self, prompt: str, section_def: Dict[str, Any]) -> str:
        """Mock content generation (replace with actual LLM call)"""
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        templates = {
            "Strategic Overview": """The competitive landscape reveals three critical opportunity windows: (1) Market disruption through technology arbitrage - competitors remain locked in legacy approaches, (2) Customer segment underserved by current solutions - 47% express dissatisfaction with status quo, (3) Regulatory changes creating first-mover advantages for prepared organizations. Strategic positioning should exploit asymmetric capabilities while building defensive moats around core competencies.""",
            
            "Tactical Approaches": """Implement a three-pronged tactical framework: Phase 1 - Rapid prototyping using agile methodologies to test market assumptions (Sprint 0-4). Phase 2 - Scale successful experiments using portfolio approach, killing underperformers quickly (Month 2-6). Phase 3 - Consolidate gains through operational excellence and process optimization (Month 6+). Key tactics include: A/B testing all customer touchpoints, leveraging partnerships for non-core capabilities, and maintaining optionality through modular architecture.""",
            
            "Resource Optimization": """Apply 80/20 principle rigorously: 20% of initiatives will drive 80% of value. Resource allocation framework: 60% to core revenue drivers, 30% to emerging opportunities, 10% to experimental ventures. Leverage points identified: (1) Technology infrastructure - 3x multiplier effect, (2) Human capital in key roles - 5x performance differential, (3) Strategic partnerships - access to $10M+ value pools. Implement zero-based budgeting to eliminate resource drag.""",
            
            "Success Metrics": """Define cascading KPIs aligned to strategic objectives: Tier 1 - Revenue growth (target: 35% YoY), Market share gain (+5 points), EBITDA margin (>25%). Tier 2 - Customer acquisition cost (<$100), Lifetime value (>$1000), Net promoter score (>50). Tier 3 - Employee productivity (+20%), Innovation pipeline (10 initiatives), Time-to-market (<90 days). Implement real-time dashboards with exception-based reporting for rapid course correction.""",
            
            "Implementation Roadmap": """Week 1-2: Establish tiger team, secure executive sponsorship, baseline current state. Week 3-4: Develop detailed project charter, identify quick wins for momentum. Month 2: Launch pilot programs in controlled environment, gather rapid feedback. Month 3-4: Scale successful pilots, sunset failures, refine approach. Month 5-6: Full rollout with continuous optimization. Critical path dependencies: Technology platform (Week 2), Key hire completion (Week 4), Partner agreements (Month 2)."""
        }
        
        return templates.get(section_def['title'], "Content generation in progress...")
    
    def _filter_results_for_section(self, results: List[Dict[str, Any]], 
                                   section_def: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter results relevant to a specific section"""
        section_keywords = {
            "Strategic Overview": ["strategy", "competitive", "market", "opportunity"],
            "Tactical Approaches": ["tactics", "approach", "method", "implementation"],
            "Resource Optimization": ["optimize", "efficiency", "allocation", "leverage"],
            "Success Metrics": ["metrics", "KPI", "measurement", "performance"],
            "Implementation Roadmap": ["roadmap", "timeline", "milestone", "plan"]
        }
        
        keywords = section_keywords.get(section_def['title'], [])
        
        relevant = []
        for result in results:
            text = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
            if any(keyword in text for keyword in keywords):
                relevant.append(result)
        
        return relevant or results[:5]
    
    def _extract_and_create_citations(self, content: str, sources: List[Dict[str, Any]]) -> List[str]:
        """Extract facts from content and create citations"""
        citation_ids = []
        
        # Create citations for top sources
        for source in sources[:3]:
            citation = self.create_citation(source, "reference")
            citation_ids.append(citation.id)
        
        return citation_ids
    
    async def _generate_summary(self, context: SynthesisContext, 
                               sections: List[AnswerSection]) -> str:
        """Generate executive summary"""
        return f"""Strategic analysis of {context.query} identifies high-impact opportunities for competitive advantage. 
Clear tactical pathways exist with quantifiable ROI potential exceeding 3x investment within 12 months. 
Implementation requires focused execution on prioritized initiatives with rigorous performance tracking."""
    
    def _generate_action_items(self, context: SynthesisContext) -> List[Dict[str, Any]]:
        """Generate paradigm-specific action items"""
        return [
            {
                "priority": "critical",
                "action": "Form strategic task force and secure executive mandate",
                "timeframe": "This week",
                "impact": "high",
                "resources": ["C-suite sponsor", "Cross-functional team", "$50K initial budget"]
            },
            {
                "priority": "high",
                "action": "Execute competitive intelligence gathering sprint",
                "timeframe": "Next 2 weeks",
                "impact": "high",
                "resources": ["Market research firm", "Internal analysts", "Industry contacts"]
            },
            {
                "priority": "high",
                "action": "Launch 3 quick-win pilot programs",
                "timeframe": "Within 30 days",
                "impact": "medium",
                "resources": ["Agile teams", "Test budget $100K", "Customer segments"]
            },
            {
                "priority": "medium",
                "action": "Establish performance dashboards and tracking",
                "timeframe": "Within 6 weeks",
                "impact": "medium",
                "resources": ["BI tools", "Data analysts", "KPI framework"]
            }
        ]
    
    def _count_actionable_insights(self, sections: List[AnswerSection]) -> int:
        """Count actionable insights across sections"""
        count = 0
        action_keywords = ['implement', 'execute', 'launch', 'deploy', 'optimize']
        
        for section in sections:
            for insight in section.key_insights:
                if any(keyword in insight.lower() for keyword in action_keywords):
                    count += 1
        
        return count
    
    def _calculate_confidence(self, context: SynthesisContext, 
                             sections: List[AnswerSection]) -> float:
        """Calculate overall confidence score"""
        avg_credibility = sum(r.get('credibility_score', 0.5) 
                             for r in context.search_results) / len(context.search_results)
        
        business_source_ratio = self._count_business_sources(context.search_results) / len(context.search_results)
        citation_factor = min(1.0, len(self.citations) / 12)
        section_confidence = sum(s.confidence for s in sections) / len(sections)
        
        return (avg_credibility * 0.3 + business_source_ratio * 0.2 + 
                citation_factor * 0.2 + section_confidence * 0.3)
    
    def _count_business_sources(self, results: List[Dict[str, Any]]) -> int:
        """Count business-oriented sources"""
        business_domains = ['hbr.org', 'mckinsey.com', 'wsj.com', 'forbes.com', 'bloomberg.com']
        count = 0
        for result in results:
            domain = result.get('domain', '').lower()
            if any(biz_domain in domain for biz_domain in business_domains):
                count += 1
        return count
    
    def _calculate_synthesis_quality(self, sections: List[AnswerSection]) -> float:
        """Calculate synthesis quality score"""
        insight_count = sum(len(s.key_insights) for s in sections)
        citation_count = sum(len(s.citations) for s in sections)
        actionable_count = self._count_actionable_insights(sections)
        
        insight_factor = min(1.0, insight_count / 20)
        citation_factor = min(1.0, citation_count / 15)
        actionable_factor = min(1.0, actionable_count / 10)
        
        return (insight_factor * 0.3 + citation_factor * 0.3 + actionable_factor * 0.4)
    
    def _calculate_overall_alignment(self, results: List[Dict[str, Any]]) -> float:
        """Calculate overall paradigm alignment of results"""
        if not results:
            return 0.0
        
        alignments = [self._calculate_paradigm_alignment(r) for r in results]
        return sum(alignments) / len(alignments)

# --- Answer Generation Orchestrator ---

class AnswerGenerationOrchestrator:
    """Main orchestrator for answer generation across paradigms"""
    
    def __init__(self):
        self.generators = {
            "dolores": DoloresAnswerGenerator(),
            "teddy": TeddyAnswerGenerator(),
            "bernard": BernardAnswerGenerator(),
            "maeve": MaeveAnswerGenerator()
        }
        self.generation_history = []
        
    async def generate_answer(self, 
                            paradigm: str,
                            query: str,
                            search_results: List[Dict[str, Any]],
                            context_engineering: Dict[str, Any],
                            options: Dict[str, Any] = None) -> GeneratedAnswer:
        """Generate answer for specified paradigm"""
        
        if paradigm not in self.generators:
            raise ValueError(f"Unknown paradigm: {paradigm}")
        
        # Create synthesis context
        context = SynthesisContext(
            query=query,
            paradigm=paradigm,
            search_results=search_results,
            context_engineering=context_engineering,
            max_length=options.get('max_length', 2000) if options else 2000,
            include_citations=options.get('include_citations', True) if options else True,
            tone=options.get('tone', 'professional') if options else 'professional'
        )
        
        # Add metadata as attribute
        context.metadata = {
            'research_id': options.get('research_id', 'unknown') if options else 'unknown'
        }
        
        # Generate answer using appropriate generator
        generator = self.generators[paradigm]
        answer = await generator.generate_answer(context)
        
        # Store in history
        self.generation_history.append({
            'timestamp': datetime.now(),
            'paradigm': paradigm,
            'query': query,
            'answer_id': answer.research_id
        })
        
        logger.info(f"Generated {paradigm} answer for: {query[:50]}...")
        
        return answer
    
    async def generate_multi_paradigm_answer(self,
                                           primary_paradigm: str,
                                           secondary_paradigm: Optional[str],
                                           query: str,
                                           search_results: List[Dict[str, Any]],
                                           context_engineering: Dict[str, Any],
                                           options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate answer combining multiple paradigms"""
        
        # Generate primary answer
        primary_answer = await self.generate_answer(
            primary_paradigm, query, search_results, 
            context_engineering, options
        )
        
        # Generate secondary answer if specified
        secondary_answer = None
        if secondary_paradigm and secondary_paradigm != primary_paradigm:
            # Use subset of results for secondary
            secondary_results = search_results[:int(len(search_results) * 0.3)]
            secondary_answer = await self.generate_answer(
                secondary_paradigm, query, secondary_results,
                context_engineering, options
            )
        
        # Combine answers
        combined = {
            'research_id': primary_answer.research_id,
            'query': query,
            'primary_paradigm': {
                'paradigm': primary_paradigm,
                'answer': primary_answer,
                'weight': 0.7
            },
            'secondary_paradigm': {
                'paradigm': secondary_paradigm,
                'answer': secondary_answer,
                'weight': 0.3
            } if secondary_answer else None,
            'synthesis_quality': self._calculate_combined_quality(
                primary_answer, secondary_answer
            ),
            'timestamp': datetime.now()
        }
        
        return combined
    
    def _calculate_combined_quality(self, 
                                  primary: GeneratedAnswer,
                                  secondary: Optional[GeneratedAnswer]) -> float:
        """Calculate quality score for combined answer"""
        if not secondary:
            return primary.synthesis_quality
        
        return (primary.synthesis_quality * 0.7 + secondary.synthesis_quality * 0.3)
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about answer generation"""
        if not self.generation_history:
            return {"total_generated": 0}
        
        paradigm_counts = {}
        for entry in self.generation_history:
            paradigm = entry['paradigm']
            paradigm_counts[paradigm] = paradigm_counts.get(paradigm, 0) + 1
        
        return {
            'total_generated': len(self.generation_history),
            'paradigm_distribution': paradigm_counts,
            'last_generation': self.generation_history[-1]['timestamp']
        }

# --- Factory Functions ---

def create_answer_generator(paradigm: str) -> BaseAnswerGenerator:
    """Factory function to create appropriate answer generator"""
    generators = {
        "dolores": DoloresAnswerGenerator,
        "teddy": TeddyAnswerGenerator,
        "bernard": BernardAnswerGenerator,
        "maeve": MaeveAnswerGenerator
    }
    
    generator_class = generators.get(paradigm)
    if not generator_class:
        raise ValueError(f"Unknown paradigm: {paradigm}")
    
    return generator_class()

# Global orchestrator instance
answer_orchestrator = AnswerGenerationOrchestrator()

# --- Initialization ---

async def initialize_answer_generation():
    """Initialize answer generation system"""
    logger.info("Initializing Answer Generation System...")
    
    # Could initialize LLM clients here
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    # anthropic_client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    logger.info("✓ Answer Generation System initialized")
    return True