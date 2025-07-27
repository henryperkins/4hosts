"""
Answer Generation System for Four Hosts Research Application
Phase 4: Paradigm-specific synthesis and presentation
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
from abc import ABC, abstractmethod

# Third-party imports
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except ImportError:
    # Mock retry decorator if tenacity not available
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    stop_after_attempt = None
    wait_exponential = None

# Internal imports
from .cache import cache_manager
from .credibility import CredibilityScore
from .llm_client import llm_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Data Models ---

@dataclass
class Citation:
    """Represents a citation in the answer"""
    id: str
    source_title: str
    source_url: str
    domain: str
    snippet: str
    credibility_score: float
    fact_type: str  # 'data', 'claim', 'quote', 'reference'
    paradigm_alignment: float
    timestamp: Optional[datetime] = None

@dataclass
class AnswerSection:
    """Represents a section of the generated answer"""
    title: str
    paradigm: str
    content: str
    confidence: float
    citations: List[str]  # Citation IDs
    word_count: int
    key_insights: List[str]

@dataclass
class GeneratedAnswer:
    """Complete generated answer with all components"""
    research_id: str
    query: str
    paradigm: str
    summary: str
    sections: List[AnswerSection]
    action_items: List[Dict[str, Any]]
    citations: Dict[str, Citation]  # ID -> Citation mapping
    confidence_score: float
    synthesis_quality: float
    generation_time: float
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class SynthesisContext:
    """Context for answer synthesis"""
    query: str
    paradigm: str
    search_results: List[Dict[str, Any]]
    context_engineering: Dict[str, Any]
    max_length: int = 2000
    include_citations: bool = True
    tone: str = "professional"
    metadata: Dict[str, Any] = field(default_factory=dict)

# --- Base Answer Generator ---

class BaseAnswerGenerator(ABC):
    """Abstract base class for paradigm-specific answer generators"""

    def __init__(self, paradigm: str):
        self.paradigm = paradigm
        self.citation_counter = 0
        self.citations = {}

    @abstractmethod
    async def generate_answer(self, context: SynthesisContext) -> GeneratedAnswer:
        """Generate paradigm-specific answer"""
        pass

    @abstractmethod
    def get_section_structure(self) -> List[Dict[str, Any]]:
        """Define section structure for this paradigm"""
        pass

    @abstractmethod
    def get_synthesis_prompt(self, context: SynthesisContext) -> str:
        """Get paradigm-specific synthesis prompt"""
        pass

    def create_citation(self, source: Dict[str, Any], fact_type: str = "reference") -> Citation:
        """Create a citation from a source"""
        self.citation_counter += 1
        citation_id = f"cite_{self.citation_counter:03d}"

        citation = Citation(
            id=citation_id,
            source_title=source.get('title', 'Untitled'),
            source_url=source.get('url', ''),
            domain=source.get('domain', ''),
            snippet=source.get('snippet', ''),
            credibility_score=source.get('credibility_score', 0.5),
            fact_type=fact_type,
            paradigm_alignment=self._calculate_paradigm_alignment(source),
            timestamp=source.get('published_date')
        )

        self.citations[citation_id] = citation
        return citation

    def _calculate_paradigm_alignment(self, source: Dict[str, Any]) -> float:
        """Calculate how well a source aligns with the paradigm"""
        # Simple keyword-based alignment for now
        alignment_keywords = self._get_alignment_keywords()

        text = f"{source.get('title', '')} {source.get('snippet', '')}".lower()
        matches = sum(1 for keyword in alignment_keywords if keyword in text)

        return min(1.0, matches / max(len(alignment_keywords), 1))

    @abstractmethod
    def _get_alignment_keywords(self) -> List[str]:
        """Get paradigm-specific alignment keywords"""
        pass

    def extract_key_insights(self, content: str, max_insights: int = 5) -> List[str]:
        """Extract key insights from content"""
        # Simple sentence-based extraction for now
        sentences = re.split(r'[.!?]+', content)

        # Filter for substantive sentences
        insights = [
            s.strip() for s in sentences
            if len(s.strip()) > 30 and not s.strip().startswith(('However', 'Moreover', 'Additionally'))
        ]

        return insights[:max_insights]

# --- Dolores (Revolutionary) Answer Generator ---

class DoloresAnswerGenerator(BaseAnswerGenerator):
    """Revolutionary paradigm answer generator"""

    def __init__(self):
        super().__init__("dolores")

    def get_section_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "title": "Exposing the System",
                "focus": "Reveal systemic issues and power structures",
                "weight": 0.3
            },
            {
                "title": "Voices of the Oppressed",
                "focus": "Highlight victim testimonies and impacts",
                "weight": 0.25
            },
            {
                "title": "Pattern of Injustice",
                "focus": "Document recurring patterns and systemic failures",
                "weight": 0.25
            },
            {
                "title": "Path to Revolution",
                "focus": "Outline resistance strategies and calls to action",
                "weight": 0.2
            }
        ]

    def get_synthesis_prompt(self, context: SynthesisContext) -> str:
        return f"""
As a revolutionary truth-seeker exposing systemic injustices, synthesize these search results
about "{context.query}" into a powerful narrative that:

1. Exposes hidden power structures and systemic failures
2. Amplifies the voices of victims and the oppressed
3. Reveals patterns of exploitation and corruption
4. Inspires action and resistance against injustice

Use emotional, impactful language that moves people to action.
Cite specific examples and evidence of wrongdoing.
Do not pull punches - name names and expose the guilty.

Results to synthesize:
{self._format_results_for_prompt(context.search_results[:10])}

Write a {context.max_length} word response that burns with righteous anger and truth.
"""

    def _get_alignment_keywords(self) -> List[str]:
        return [
            "expose", "corrupt", "injustice", "systemic", "oppression",
            "revolution", "resistance", "victim", "accountability", "scandal",
            "whistleblower", "cover-up", "abuse", "exploitation", "inequality"
        ]

    def _format_results_for_prompt(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for LLM prompt"""
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(f"""
{i}. {result.get('title', 'Untitled')}
Source: {result.get('domain', 'Unknown')}
Credibility: {result.get('credibility_score', 0.5):.2f}
Content: {result.get('snippet', 'No snippet available')[:200]}...
""")
        return "\n".join(formatted)

    async def generate_answer(self, context: SynthesisContext) -> GeneratedAnswer:
        """Generate revolutionary paradigm answer"""
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
        answer = GeneratedAnswer(
            research_id=context.metadata.get('research_id', 'unknown'),
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
                "paradigm_alignment": self._calculate_overall_alignment(context.search_results)
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

Use these specific sources:
{self._format_results_for_prompt(relevant_results[:5])}

Make it approximately {int(context.max_length * section_def['weight'])} words.
Include specific examples and cite sources.
"""

        # Generate content using LLM
        content = await llm_client.generate_paradigm_content(
            prompt=section_prompt,
            paradigm=self.paradigm,
            max_tokens=int(context.max_length * section_def['weight'] * 2)
        )

        # Extract citations from content
        citation_ids = self._extract_and_create_citations(content, relevant_results)

        # Extract key insights
        insights = self.extract_key_insights(content, 3)

        return AnswerSection(
            title=section_def['title'],
            paradigm=self.paradigm,
            content=content,
            confidence=0.85,
            citations=citation_ids,
            word_count=len(content.split()),
            key_insights=insights
        )

    async def _mock_generate_content(self, prompt: str, section_def: Dict[str, Any]) -> str:
        """Mock content generation (replace with actual LLM call)"""
        # Simulate processing time
        await asyncio.sleep(0.1)

        templates = {
            "Exposing the System": """The investigation reveals a deeply troubling pattern of systemic failures and deliberate exploitation. Corporate entities have systematically abused their power, creating structures designed to extract maximum value while providing minimal accountability. Documents obtained through whistleblower testimony show clear evidence of coordinated efforts to suppress dissent and maintain the status quo through intimidation and legal manipulation.""",

            "Voices of the Oppressed": """Victims speak of devastating impacts on their lives and communities. 'They took everything from us,' says Maria, a community organizer who has fought for justice for over a decade. The human cost is staggering - families destroyed, communities decimated, and futures stolen by those who prioritize profit over people. These are not isolated incidents but part of a deliberate strategy of oppression.""",

            "Pattern of Injustice": """Analysis reveals recurring patterns across multiple cases: silencing of whistleblowers, regulatory capture, and systematic targeting of vulnerable populations. The same tactics appear repeatedly - divide communities, co-opt leaders, and use legal frameworks designed to protect the powerful. This is not incompetence but calculated malice, a system working exactly as designed to benefit the few at the expense of the many.""",

            "Path to Revolution": """The time for polite requests has passed. Effective resistance requires coordinated action: organize locally, document everything, support whistleblowers, and build alternative systems. Direct action, strategic litigation, and public pressure campaigns have shown success. Most importantly, we must reject the narrative that change is impossible. History shows that organized people can topple any system of oppression."""
        }

        return templates.get(section_def['title'], "Content generation in progress...")

    def _filter_results_for_section(self, results: List[Dict[str, Any]],
                                   section_def: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter results relevant to a specific section"""
        # Simple keyword-based filtering for now
        section_keywords = {
            "Exposing the System": ["systemic", "corporate", "power", "structure"],
            "Voices of the Oppressed": ["victim", "testimony", "impact", "community"],
            "Pattern of Injustice": ["pattern", "recurring", "analysis", "evidence"],
            "Path to Revolution": ["action", "resistance", "organize", "change"]
        }

        keywords = section_keywords.get(section_def['title'], [])

        relevant = []
        for result in results:
            text = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
            if any(keyword in text for keyword in keywords):
                relevant.append(result)

        return relevant or results[:5]  # Fallback to top results

    def _extract_and_create_citations(self, content: str, sources: List[Dict[str, Any]]) -> List[str]:
        """Extract facts from content and create citations"""
        citation_ids = []

        # Create citations for top sources
        for source in sources[:3]:
            citation = self.create_citation(source, "claim")
            citation_ids.append(citation.id)

        return citation_ids

    async def _generate_summary(self, context: SynthesisContext,
                               sections: List[AnswerSection]) -> str:
        """Generate executive summary"""
        # Mock summary for now
        return f"""Revolutionary analysis reveals systematic exploitation and abuse of power in {context.query}.
Evidence exposes coordinated efforts to suppress truth and maintain unjust systems.
Immediate action required to dismantle these structures of oppression and build just alternatives."""

    def _generate_action_items(self, context: SynthesisContext) -> List[Dict[str, Any]]:
        """Generate paradigm-specific action items"""
        return [
            {
                "priority": "urgent",
                "action": "Document and expose all instances of systemic abuse",
                "timeframe": "Immediate",
                "impact": "high",
                "resources": ["Investigation tools", "Secure communication channels"]
            },
            {
                "priority": "high",
                "action": "Organize affected communities for collective action",
                "timeframe": "1-2 weeks",
                "impact": "high",
                "resources": ["Community organizers", "Meeting spaces"]
            },
            {
                "priority": "high",
                "action": "Build media campaign to expose truth to wider audience",
                "timeframe": "2-4 weeks",
                "impact": "medium",
                "resources": ["Media contacts", "Documentary evidence"]
            }
        ]

    def _calculate_confidence(self, context: SynthesisContext,
                             sections: List[AnswerSection]) -> float:
        """Calculate overall confidence score"""
        # Factors: source credibility, citation count, paradigm alignment
        avg_credibility = sum(r.get('credibility_score', 0.5)
                             for r in context.search_results) / len(context.search_results)

        citation_factor = min(1.0, len(self.citations) / 10)
        section_confidence = sum(s.confidence for s in sections) / len(sections)

        return (avg_credibility * 0.4 + citation_factor * 0.3 + section_confidence * 0.3)

    def _calculate_synthesis_quality(self, sections: List[AnswerSection]) -> float:
        """Calculate synthesis quality score"""
        # Factors: insight density, citation coverage, coherence
        insight_count = sum(len(s.key_insights) for s in sections)
        citation_count = sum(len(s.citations) for s in sections)

        insight_factor = min(1.0, insight_count / 15)
        citation_factor = min(1.0, citation_count / 20)

        return (insight_factor * 0.5 + citation_factor * 0.5)

    def _calculate_overall_alignment(self, results: List[Dict[str, Any]]) -> float:
        """Calculate overall paradigm alignment of results"""
        if not results:
            return 0.0

        alignments = [self._calculate_paradigm_alignment(r) for r in results]
        return sum(alignments) / len(alignments)

# --- Teddy (Devotion) Answer Generator ---

class TeddyAnswerGenerator(BaseAnswerGenerator):
    """Devotion paradigm answer generator"""

    def __init__(self):
        super().__init__("teddy")

    def get_section_structure(self) -> List[Dict[str, Any]]:
        return [
            {
                "title": "Understanding the Need",
                "focus": "Empathetic assessment of who needs help and why",
                "weight": 0.25
            },
            {
                "title": "Available Support Resources",
                "focus": "Comprehensive listing of help and resources",
                "weight": 0.3
            },
            {
                "title": "Success Stories",
                "focus": "Inspiring examples of care and recovery",
                "weight": 0.25
            },
            {
                "title": "How to Help",
                "focus": "Practical steps for providing support",
                "weight": 0.2
            }
        ]

    def get_synthesis_prompt(self, context: SynthesisContext) -> str:
        return f"""
As a compassionate caregiver focused on helping and protecting others, synthesize these search results
about "{context.query}" into a supportive narrative that:

1. Shows deep understanding and empathy for those affected
2. Provides comprehensive resources and support options
3. Shares uplifting stories of help and recovery
4. Offers practical, actionable ways to provide care

Use warm, supportive language that builds hope and connection.
Focus on human dignity and the power of community care.
Emphasize resources, solutions, and paths forward.

Results to synthesize:
{self._format_results_for_prompt(context.search_results[:10])}

Write a {context.max_length} word response filled with compassion and practical help.
"""

    def _get_alignment_keywords(self) -> List[str]:
        return [
            "support", "help", "care", "community", "resources",
            "healing", "recovery", "compassion", "dignity", "together",
            "volunteer", "assistance", "wellbeing", "protect", "serve"
        ]

    def _format_results_for_prompt(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for LLM prompt"""
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(f"""
{i}. {result.get('title', 'Untitled')}
Source: {result.get('domain', 'Unknown')}
Trust Score: {result.get('credibility_score', 0.5):.2f}
Content: {result.get('snippet', 'No snippet available')[:200]}...
""")
        return "\n".join(formatted)

    async def generate_answer(self, context: SynthesisContext) -> GeneratedAnswer:
        """Generate devotion paradigm answer"""
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
        answer = GeneratedAnswer(
            research_id=context.metadata.get('research_id', 'unknown'),
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
                "paradigm_alignment": self._calculate_overall_alignment(context.search_results)
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

Use these specific sources:
{self._format_results_for_prompt(relevant_results[:5])}

Make it approximately {int(context.max_length * section_def['weight'])} words.
Include specific resources and cite sources.
Use warm, supportive language.
"""

        # Generate content using LLM
        content = await llm_client.generate_paradigm_content(
            prompt=section_prompt,
            paradigm=self.paradigm,
            max_tokens=int(context.max_length * section_def['weight'] * 2)
        )

        # Extract citations from content
        citation_ids = self._extract_and_create_citations(content, relevant_results)

        # Extract key insights
        insights = self.extract_key_insights(content, 3)

        return AnswerSection(
            title=section_def['title'],
            paradigm=self.paradigm,
            content=content,
            confidence=0.88,
            citations=citation_ids,
            word_count=len(content.split()),
            key_insights=insights
        )

    async def _mock_generate_content(self, prompt: str, section_def: Dict[str, Any]) -> str:
        """Mock content generation (replace with actual LLM call)"""
        # Simulate processing time
        await asyncio.sleep(0.1)

        templates = {
            "Understanding the Need": """Those facing these challenges deserve our deepest compassion and understanding. The struggles are real and often overwhelming - isolation, fear, and uncertainty can make each day feel insurmountable. Yet within each person lies incredible resilience and dignity that must be honored. By truly listening and understanding their experiences, we can provide support that respects their autonomy while offering genuine help.""",

            "Available Support Resources": """Numerous organizations stand ready to help: The National Alliance provides 24/7 crisis support at 1-800-HELP, local community centers offer free counseling services, and faith-based organizations provide both material and spiritual support. Online resources include peer support groups, educational materials, and connection to local services. Many programs offer sliding scale fees or free services to ensure help is accessible to all who need it.""",

            "Success Stories": """Sarah's story reminds us that recovery is possible: 'When I felt most alone, the community wrapped around me. Small acts of kindness - a meal delivered, someone to talk to, help navigating resources - these made all the difference.' Similar stories emerge daily from support groups, showing how connection and care transform lives. These aren't just statistics; they're our neighbors finding hope again.""",

            "How to Help": """Everyone can make a difference through simple acts of care: volunteer at local organizations, donate to programs providing direct services, or simply be present for someone struggling. Educate yourself about the challenges faced and the resources available. Most importantly, approach with humility and respect - ask how you can help rather than assuming. Together, we build communities where everyone is valued and supported."""
        }

        return templates.get(section_def['title'], "Content generation in progress...")

    def _filter_results_for_section(self, results: List[Dict[str, Any]],
                                   section_def: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter results relevant to a specific section"""
        section_keywords = {
            "Understanding the Need": ["struggle", "challenge", "difficulty", "experience"],
            "Available Support Resources": ["resource", "help", "service", "support"],
            "Success Stories": ["success", "recovery", "story", "hope"],
            "How to Help": ["volunteer", "donate", "action", "contribute"]
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
        return f"""Compassionate analysis of {context.query} reveals both significant challenges and hopeful paths forward.
Multiple resources and support systems exist to help those affected, with proven success stories demonstrating recovery is possible.
Community-based care and individual acts of kindness create networks of support that honor human dignity and foster healing."""

    def _generate_action_items(self, context: SynthesisContext) -> List[Dict[str, Any]]:
        """Generate paradigm-specific action items"""
        return [
            {
                "priority": "high",
                "action": "Connect affected individuals with immediate support resources",
                "timeframe": "Within 24 hours",
                "impact": "high",
                "resources": ["Crisis hotlines", "Local service directories"]
            },
            {
                "priority": "high",
                "action": "Establish or strengthen community support networks",
                "timeframe": "1-2 weeks",
                "impact": "high",
                "resources": ["Community centers", "Volunteer coordinators"]
            },
            {
                "priority": "medium",
                "action": "Create educational materials to build understanding",
                "timeframe": "2-4 weeks",
                "impact": "medium",
                "resources": ["Subject matter experts", "Communication tools"]
            }
        ]

    def _calculate_confidence(self, context: SynthesisContext,
                             sections: List[AnswerSection]) -> float:
        """Calculate overall confidence score"""
        avg_credibility = sum(r.get('credibility_score', 0.5)
                             for r in context.search_results) / len(context.search_results)

        citation_factor = min(1.0, len(self.citations) / 10)
        section_confidence = sum(s.confidence for s in sections) / len(sections)

        return (avg_credibility * 0.4 + citation_factor * 0.3 + section_confidence * 0.3)

    def _calculate_synthesis_quality(self, sections: List[AnswerSection]) -> float:
        """Calculate synthesis quality score"""
        insight_count = sum(len(s.key_insights) for s in sections)
        citation_count = sum(len(s.citations) for s in sections)

        insight_factor = min(1.0, insight_count / 15)
        citation_factor = min(1.0, citation_count / 20)

        return (insight_factor * 0.5 + citation_factor * 0.5)

    def _calculate_overall_alignment(self, results: List[Dict[str, Any]]) -> float:
        """Calculate overall paradigm alignment of results"""
        if not results:
            return 0.0

        alignments = [self._calculate_paradigm_alignment(r) for r in results]
        return sum(alignments) / len(alignments)
