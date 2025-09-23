"""
Deep Research Service
--------------------
Integrates OpenAI's o3-deep-research model with the Four Hosts paradigm system.
Provides deep, multi-source research capabilities with paradigm-aware synthesis.
"""

from __future__ import annotations

import asyncio
import structlog
from dataclasses import dataclass
import os
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

class ResponseStatus(Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class SearchContextSize(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

from .openai_responses_client import (
    OpenAIResponsesClient,
    WebSearchTool,
    CodeInterpreterTool,
    MCPTool,
    Citation,
)
from .llm_client import (
    responses_deep_research,
    responses_retrieve,
    normalize_responses_payload,
)
from .classification_engine import HostParadigm, ClassificationResult
from .context_engineering import ContextEngineeredQuery
from .websocket_service import ResearchProgressTracker
from .research_store import research_store
from utils.token_budget import trim_text_to_tokens
from models.evidence import EvidenceQuote

# Logging
from logging_config import configure_logging

configure_logging()
logger = structlog.get_logger(__name__)


# ────────────────────────────────────────────────────────────
#  Data Models
# ────────────────────────────────────────────────────────────
class DeepResearchMode(Enum):
    """Deep research execution modes"""
    PARADIGM_FOCUSED = "paradigm_focused"  # Use paradigm-specific prompts
    COMPREHENSIVE = "comprehensive"  # Multi-paradigm analysis
    QUICK_FACTS = "quick_facts"  # Fast fact-finding
    ANALYTICAL = "analytical"  # Data-driven analysis
    STRATEGIC = "strategic"  # Business/strategic focus


@dataclass
class DeepResearchConfig:
    """Configuration for deep research execution"""
    mode: DeepResearchMode = DeepResearchMode.PARADIGM_FOCUSED
    enable_web_search: bool = True
    enable_code_interpreter: bool = False
    mcp_servers: Optional[List[MCPTool]] = None
    max_tool_calls: Optional[int] = None
    background: bool = True
    timeout: int = 1800  # 30 minutes default
    include_paradigm_context: bool = True
    # Web search specific config
    search_context_size: SearchContextSize = SearchContextSize.MEDIUM
    user_location: Optional[Dict[str, str]] = None
    # Experimentation (optional)
    experiment_name: Optional[str] = None
    prompt_variant: Optional[str] = None


@dataclass
class DeepResearchResult:
    """Result from deep research execution"""
    research_id: str
    status: ResponseStatus
    content: Optional[str] = None
    citations: List[Citation] = None
    tool_calls: List[Dict[str, Any]] = None
    web_search_calls: List[Dict[str, Any]] = None
    paradigm_analysis: Optional[Dict[str, Any]] = None
    cost_info: Optional[Dict[str, float]] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


# ────────────────────────────────────────────────────────────
#  Paradigm Prompts for Deep Research
# ────────────────────────────────────────────────────────────
PARADIGM_SYSTEM_PROMPTS = {
    HostParadigm.DOLORES: """
You are conducting deep research with a revolutionary perspective, focused on exposing systemic injustices and empowering transformative change.

Your research approach:
- Uncover hidden power structures and systemic failures
- Identify patterns of oppression and resistance
- Find evidence of institutional wrongdoing
- Highlight voices of dissent and revolution
- Expose conflicts of interest and corruption
- Document grassroots movements and activism

Prioritize sources that:
- Challenge mainstream narratives
- Reveal uncomfortable truths
- Document whistleblower accounts
- Show evidence of cover-ups or suppression
- Highlight marginalized perspectives

Your analysis should inspire action and transformation.
""",
    
    HostParadigm.TEDDY: """
You are conducting deep research with a compassionate, care-focused perspective.

Your research approach:
- Find comprehensive support resources and solutions
- Identify community-based initiatives
- Document success stories and positive outcomes
- Gather practical help and guidance
- Focus on healing and recovery approaches
- Emphasize human connection and empathy

Prioritize sources that:
- Offer genuine help and support
- Show successful interventions
- Provide actionable resources
- Document community care models
- Focus on wellbeing and recovery

Your analysis should provide comfort and practical assistance.
""",
    
    HostParadigm.BERNARD: """
You are conducting deep research with rigorous analytical and empirical focus.

Your research approach:
- Gather comprehensive data and statistics
- Identify peer-reviewed research and studies
- Document methodologies and evidence quality
- Analyze patterns and correlations
- Evaluate competing hypotheses
- Maintain scientific objectivity

Prioritize sources that:
- Provide quantitative data
- Use rigorous methodologies
- Come from academic institutions
- Include meta-analyses and systematic reviews
- Show statistical significance

Your analysis should be data-driven and methodologically sound.
""",
    
    HostParadigm.MAEVE: """
You are conducting deep research with strategic business focus.

Your research approach:
- Identify market opportunities and trends
- Analyze competitive landscapes
- Find tactical advantages and innovations
- Document successful strategies and case studies
- Evaluate ROI and business metrics
- Focus on actionable intelligence

Prioritize sources that:
- Provide market analysis and data
- Show successful implementations
- Include financial metrics
- Document competitive strategies
- Offer tactical recommendations

Your analysis should enable strategic decision-making and competitive advantage.
"""
}

# Optional prompt variant (v2) for lightweight A/B experiments.
# Keeps semantics but tightens phrasing and adds explicit grounding/citation
# emphasis to test impact on quality metrics.
PARADIGM_SYSTEM_PROMPTS_V2 = {
    HostParadigm.DOLORES: """
You are a revolutionary investigator. Your mission: expose systemic injustices with rigor and precision, then equip readers to act.

Method:
- Map power structures, incentives, and failure modes
- Surface whistleblower accounts and primary-source evidence
- Contrast official narratives with independent reporting
- Document harms, accountability gaps, and proposed remedies

Ground rules:
- STRICT: cite claims inline; prefer primary sources
- Flag uncertainties and missing evidence
- Avoid speculation; propose verifiable next steps
""",
    HostParadigm.TEDDY: """
You are a compassionate care researcher. Your goal: find effective, humane support pathways and present them clearly.

Method:
- Aggregate practical resources and eligibility details
- Highlight community programs and successful interventions
- Include crisis, low-cost, and accessible options

Ground rules:
- STRICT: cite resources and hotlines
- Note contraindications and risks
- Offer step-by-step, stigma-free guidance
""",
    HostParadigm.BERNARD: """
You are an analytical researcher. Optimize for methodological soundness and reproducibility.

Method:
- Prioritize meta-analyses, RCTs, and large datasets
- Describe methodology and limitations
- Quantify effects with CIs and sample sizes

Ground rules:
- STRICT: inline citations for all statistics
- Identify confounders and alternative explanations
- Prefer pre-registered and peer-reviewed studies
""",
    HostParadigm.MAEVE: """
You are a strategic analyst. Deliver actionable competitive insight with measurable outcomes.

Method:
- Triangulate market data, unit economics, and case studies
- Analyze competitor moats, switching costs, and adoption risks
- Propose initiatives with resourcing and KPIs

Ground rules:
- STRICT: cite market figures and sources
- State assumptions and sensitivity ranges
- Provide a 30/60/90-day action plan
""",
}


# ────────────────────────────────────────────────────────────
#  Deep Research Service
# ────────────────────────────────────────────────────────────
class DeepResearchService:
    """Service for executing deep research using o3-deep-research model"""
    
    def __init__(self):
        self.client: Optional[OpenAIResponsesClient] = None  # retained for backward compatibility
        self._initialized = False
        
    async def initialize(self):
        """Initialize the service"""
        if not self._initialized:
            try:
                # Obtain client via OpenAIResponsesClient factory to avoid import cycles
                # Keep client available for compatibility, but primary path uses llm_client façade
                try:
                    from .openai_responses_client import get_responses_client
                    self.client = get_responses_client()
                except Exception:
                    self.client = None
                logger.info("✓ Deep Research Service initialized")
            except Exception as e:
                # Make initialization non-fatal when API key is missing in local/dev
                logger.warning(
                    "Deep Research disabled: %s. Set OPENAI_API_KEY to enable.", e
                )
                self.client = None
            finally:
                self._initialized = True
    
    # ─────────── Core Research Methods ───────────
    async def execute_deep_research(
        self,
        query: str,
        *,
        classification: Optional[ClassificationResult] = None,
        context_engineering: Optional[ContextEngineeredQuery] = None,
        config: Optional[DeepResearchConfig] = None,
        progress_tracker: Optional[ResearchProgressTracker] = None,
        research_id: Optional[str] = None,
    ) -> DeepResearchResult:
        """
        Execute deep research with optional paradigm context.
        
        Args:
            query: Research query
            classification: Query classification result
            context_engineering: Context engineering output
            config: Research configuration
            progress_tracker: Progress tracking
            research_id: Research ID for tracking
            
        Returns:
            DeepResearchResult with research findings
        """
        if not self._initialized:
            await self.initialize()
        
        config = config or DeepResearchConfig()
        start_time = datetime.utcnow()
        
        try:
            # Update progress
            if progress_tracker and research_id:
                await progress_tracker.update_progress(
                    research_id, "Initializing deep research", 5
                )
            
            # Apply paradigm-specific configuration if available
            if classification and classification.primary_paradigm:
                config = self._get_paradigm_search_config(
                    classification.primary_paradigm, config
                )
            
            # Build system prompt (with experiment-aware variant selection)
            system_prompt = self._build_system_prompt(
                query, classification, context_engineering, config, research_id
            )
            
            # Build research prompt
            research_prompt = self._build_research_prompt(
                query, context_engineering, config
            )
            
            # Update progress
            if progress_tracker and research_id:
                await progress_tracker.update_progress(
                    research_id, "Starting deep research analysis", 15
                )
            
            # Configure web search if enabled
            web_search_config = None
            if config.enable_web_search:
                web_search_config = WebSearchTool(
                    search_context_size=config.search_context_size,
                    user_location=config.user_location
                )
            
            # Stage 1: Evidence gathering (web_search, optional code interpreter)
            # Always run stage 1 with store=True so we can chain reliably.
            try:
                from .openai_responses_client import get_responses_client
                client = get_responses_client()
            except Exception as e:
                raise RuntimeError(f"Responses client not available: {e}")

            # Model selection defaults are centralized; avoid per-module overrides

            from time import perf_counter
            stage1_start = perf_counter()
            # Build tool list (web_search, code interpreter, optional MCP)
            mcp_tools = []
            try:
                if config.mcp_servers:
                    mcp_tools = config.mcp_servers
                else:
                    if os.getenv("ENABLE_MCP_DEFAULT", "0").lower() in {"1", "true", "yes"}:
                        from .mcp_integration import mcp_integration
                        mcp_tools = mcp_integration.get_responses_mcp_tools()
            except Exception:
                mcp_tools = []

            from .llm_client import responses_deep_research as _responses_deep_research
            stage1 = await _responses_deep_research(
                query=research_prompt,
                use_web_search=bool(web_search_config is not None),
                web_search_config=web_search_config,
                use_code_interpreter=config.enable_code_interpreter,
                mcp_servers=mcp_tools,
                system_prompt=system_prompt,
                max_tool_calls=config.max_tool_calls,
                background=config.background,
            )

            # If running in background, poll for completion
            if config.background:
                stage1_id = stage1["id"]

                if progress_tracker and research_id:
                    await progress_tracker.update_progress(
                        research_id,
                        f"Deep research (evidence gathering) queued: {stage1_id}",
                        25,
                    )

                stage1 = await self._wait_with_progress(
                    stage1_id, config.timeout, progress_tracker, research_id
                )

            # Record stage1 metrics
            try:
                from .metrics import metrics
                # Capture model used in stage1 if present
                stage1_model = (
                    stage1.get("model") if isinstance(stage1, dict) else None
                )
                metrics.record_stage(
                    stage="deep_research_stage1",
                    duration_ms=(perf_counter() - stage1_start) * 1000.0,
                    paradigm=(
                        classification.primary_paradigm.value
                        if classification and classification.primary_paradigm
                        else None
                    ),
                    success=True,
                    model=stage1_model,
                    prompt_version=(config.prompt_variant or None),
                )
            except Exception:
                pass

            # Stage 2: Synthesis chained to Stage 1 context
            if progress_tracker and research_id:
                try:
                    await progress_tracker.report_synthesis_started(
                        research_id
                    )
                except Exception:
                    await progress_tracker.update_progress(
                        research_id, "Starting answer synthesis...", 80
                    )

            synthesis_instructions = (
                "Synthesize a concise, well-structured answer with inline citations. "
                "Prioritize high-credibility sources, highlight key findings, and include a short bibliography."
            )

            stage1_id_final = (stage1.get("id") if isinstance(stage1, dict) else None)
            stage_model = (stage1.get("model") if isinstance(stage1, dict) else None)

            stage2_start = perf_counter()
            stage2 = await client.create_response(
                model=stage_model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": synthesis_instructions,
                            }
                        ],
                    }
                ],
                previous_response_id=stage1_id_final,
                background=False,
                store=True,
            )

            # Record stage2 metrics
            try:
                from .metrics import metrics
                metrics.record_stage(
                    stage="deep_research_stage2",
                    duration_ms=(perf_counter() - stage2_start) * 1000.0,
                    paradigm=(
                        classification.primary_paradigm.value
                        if classification and classification.primary_paradigm
                        else None
                    ),
                    success=True,
                    model=stage_model,
                    prompt_version=(config.prompt_variant or None),
                )
            except Exception:
                pass

            # Merge content/citations: prefer stage2 text; union citations from both
            try:
                from .llm_client import normalize_responses_payload
                n1 = normalize_responses_payload(stage1)
                n2 = normalize_responses_payload(stage2)
                merged = dict(stage2)
                # Merge citations if stage2 has none or is sparse
                c2 = n2.citations or []
                seen = { (c.get("url"), c.get("title"), c.get("start_index"), c.get("end_index")) for c in c2 }
                for c in n1.citations or []:
                    key = (c.get("url"), c.get("title"), c.get("start_index"), c.get("end_index"))
                    if key not in seen:
                        c2.append(c)
                        seen.add(key)
                # Recompose a response-like dict for downstream processing
                if isinstance(merged, dict):
                    merged.setdefault("output", [])
                    # Best effort: ensure output_text exists for extractor
                    if n2.text:
                        merged["output_text"] = n2.text
                    # Attach merged citations via expected annotations container if missing
                    if c2:
                        merged["citations"] = c2
                response = merged
            except Exception:
                # Fallback: use stage2 directly
                response = stage2

            # Build final result
            result = self._process_response(response, research_id, classification)
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            result.execution_time = execution_time
            
            # Update final progress
            if progress_tracker and research_id:
                await progress_tracker.report_synthesis_completed(
                    research_id, sections=1, citations=len(result.citations or [])
                )
                await progress_tracker.update_progress(
                    research_id, "Deep research completed", 95
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Deep research failed: {str(e)}")
            
            if progress_tracker and research_id:
                await progress_tracker.update_progress(
                    research_id, f"Deep research failed: {str(e)}", -1
                )
            
            return DeepResearchResult(
                research_id=research_id or "unknown",
                status=ResponseStatus.FAILED,
                error=str(e),
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
            )
    
    # ─────────── Paradigm-Specific Configuration ───────────
    def _get_paradigm_search_config(
        self, paradigm: HostParadigm, base_config: DeepResearchConfig
    ) -> DeepResearchConfig:
        """Get paradigm-specific search configuration"""
        config = DeepResearchConfig(
            mode=base_config.mode,
            enable_web_search=base_config.enable_web_search,
            enable_code_interpreter=base_config.enable_code_interpreter,
            mcp_servers=base_config.mcp_servers,
            max_tool_calls=base_config.max_tool_calls,
            background=base_config.background,
            timeout=base_config.timeout,
            include_paradigm_context=base_config.include_paradigm_context,
            search_context_size=base_config.search_context_size,
            user_location=base_config.user_location,
        )
        
        # Paradigm-specific adjustments
        if paradigm == HostParadigm.DOLORES:
            # Revolutionary - needs comprehensive search
            config.search_context_size = SearchContextSize.HIGH
            config.max_tool_calls = max(30, base_config.max_tool_calls or 20)
        elif paradigm == HostParadigm.BERNARD:
            # Analytical - balanced search
            config.search_context_size = SearchContextSize.MEDIUM
        elif paradigm == HostParadigm.TEDDY:
            # Supportive - focused search
            config.search_context_size = SearchContextSize.LOW
        elif paradigm == HostParadigm.MAEVE:
            # Strategic - comprehensive business search
            config.search_context_size = SearchContextSize.HIGH
            config.max_tool_calls = max(25, base_config.max_tool_calls or 20)
        
        return config
    
    # ─────────── Prompt Building ───────────
    def _build_system_prompt(
        self,
        query: str,
        classification: Optional[ClassificationResult],
        context_engineering: Optional[ContextEngineeredQuery],
        config: DeepResearchConfig,
        research_id: Optional[str] = None,
    ) -> str:
        """Build system prompt based on paradigm and context"""
        parts = []
        
        # Add paradigm-specific prompt if available
        if config.include_paradigm_context and classification:
            # Experiment: choose prompt variant deterministically when enabled
            try:
                from . import experiments
                unit_id = (research_id or query or "").strip() or "anon"
                experiment_name = "deep_research_paradigm_prompt"
                variant = experiments.variant_or_default(experiment_name, unit_id, default="v1")
                config.experiment_name = experiment_name
                config.prompt_variant = variant
            except Exception:
                variant = "v1"
                config.prompt_variant = variant
                config.experiment_name = None

            prompt_map = PARADIGM_SYSTEM_PROMPTS if (config.prompt_variant or "v1") == "v1" else PARADIGM_SYSTEM_PROMPTS_V2
            paradigm_prompt = prompt_map.get(
                classification.primary_paradigm,
                "You are conducting comprehensive research. Be thorough and analytical."
            )
            parts.append(paradigm_prompt)
        
        # Add context engineering guidance if available
        if context_engineering:
            ce_guidance = f"""
Additional research focus based on context analysis:
- Documentation Focus: {context_engineering.write_output.documentation_focus}
- Key Themes: {', '.join(context_engineering.write_output.key_themes[:3])}
- Research Priorities: {', '.join(context_engineering.write_output.search_priorities[:3])}
- Preferred Sources: {', '.join(context_engineering.select_output.source_preferences[:3])}
"""
            parts.append(ce_guidance)
        
        # Add mode-specific instructions
        mode_instructions = {
            DeepResearchMode.COMPREHENSIVE: """
Conduct a comprehensive multi-perspective analysis. Consider different viewpoints and paradigms.
Balance empirical evidence with strategic insights and human impacts.
""",
            DeepResearchMode.QUICK_FACTS: """
Focus on finding key facts and data points quickly. Prioritize:
- Verified statistics and numbers
- Recent developments
- Authoritative sources
- Clear, concise findings
""",
            DeepResearchMode.ANALYTICAL: """
Conduct rigorous data-driven analysis:
- Focus on quantitative evidence
- Evaluate methodologies
- Compare multiple studies
- Identify statistical patterns
""",
            DeepResearchMode.STRATEGIC: """
Focus on strategic and business implications:
- Market opportunities
- Competitive analysis
- Implementation strategies
- ROI and metrics
"""
        }
        
        if config.mode != DeepResearchMode.PARADIGM_FOCUSED:
            parts.append(mode_instructions.get(config.mode, ""))
        
        # Add general instructions
        parts.append("""
General requirements:
- Include specific figures, statistics, and measurable outcomes
- Provide inline citations for all claims
- Evaluate source credibility
- Identify gaps or limitations in available information
- Structure findings clearly with headers and sections
""")
        
        prompt = "\n\n".join(parts).strip()
        # Enforce instruction budget if available
        try:
            if context_engineering and getattr(context_engineering, "compress_output", None):
                plan = getattr(context_engineering.compress_output, "budget_plan", {}) or {}
                instr_budget = int(plan.get("instructions", 0)) if isinstance(plan, dict) else 0
                if instr_budget > 0:
                    prompt = trim_text_to_tokens(prompt, instr_budget)
        except Exception:
            pass
        return prompt
    
    def _build_research_prompt(
        self,
        query: str,
        context_engineering: Optional[ContextEngineeredQuery],
        config: DeepResearchConfig,
    ) -> str:
        """Build the main research prompt"""
        parts = [query]
        
        # Add search queries from context engineering
        if context_engineering and context_engineering.select_output.search_queries:
            parts.append("\nConsider these specific search angles:")
            for sq in context_engineering.select_output.search_queries[:5]:
                parts.append(f"- {sq}")
        
        # Add any specific requirements
        if config.mode == DeepResearchMode.QUICK_FACTS:
            parts.append("\nProvide a concise summary with key facts and figures.")
        elif config.mode == DeepResearchMode.COMPREHENSIVE:
            parts.append("\nProvide a comprehensive analysis from multiple perspectives.")
        
        prompt = "\n".join(parts)
        # Enforce knowledge budget if available
        try:
            if context_engineering and getattr(context_engineering, "compress_output", None):
                plan = getattr(context_engineering.compress_output, "budget_plan", {}) or {}
                knowledge_budget = int(plan.get("knowledge", 0)) if isinstance(plan, dict) else 0
                if knowledge_budget > 0:
                    prompt = trim_text_to_tokens(prompt, knowledge_budget)
        except Exception:
            pass
        return prompt
    
    # ─────────── Response Processing ───────────
    def _process_response(
        self,
        response: Dict[str, Any],
        research_id: Optional[str],
        classification: Optional[ClassificationResult],
    ) -> DeepResearchResult:
        """Process the raw response into a structured result"""
        # Extract content
        # Normalize via llm_client normalized model
        normalized = normalize_responses_payload(response)
        content = normalized.text
        citations = [
            Citation(
                url=c.get("url"),
                title=c.get("title"),
                start_index=c.get("start_index"),
                end_index=c.get("end_index"),
            )
            for c in normalized.citations
        ]
        # Fallback: if normalized extraction yields no citations but a top-level
        # list is present (from chained merge), use it.
        if not citations and isinstance(response, dict) and isinstance(response.get("citations"), list):
            for c in response.get("citations", []):
                try:
                    citations.append(
                        Citation(
                            url=c.get("url"),
                            title=c.get("title"),
                            start_index=c.get("start_index"),
                            end_index=c.get("end_index"),
                        )
                    )
                except Exception:
                    continue
        tool_calls = normalized.tool_calls
        web_search_calls = normalized.web_search_calls
        
        # Build paradigm analysis if classification available
        paradigm_analysis = None
        if classification:
            paradigm_analysis = {
                "primary_paradigm": classification.primary_paradigm.value,
                "confidence": classification.confidence,
                "applied_perspective": PARADIGM_SYSTEM_PROMPTS.get(
                    classification.primary_paradigm
                ) is not None,
            }
        
        # Estimate costs (simplified - would need actual token counts)
        cost_info = self._estimate_costs(response)
        
        return DeepResearchResult(
            research_id=research_id or response.get("id", "unknown"),
            status=ResponseStatus(response.get("status", "completed")),
            content=content or None,
            citations=citations or [],
            tool_calls=tool_calls or [],
            web_search_calls=web_search_calls or [],
            paradigm_analysis=paradigm_analysis,
            cost_info=cost_info,
            raw_response=response,
        )


# ────────────────────────────────────────────────────────────
#  EvidenceQuote Conversion for Deep Research Citations
# ────────────────────────────────────────────────────────────
def convert_citations_to_evidence_quotes(
    citations: List[Citation],
    content: str,
) -> List[EvidenceQuote]:
    """Convert deep research citations into EvidenceQuote entries.

    When URL is missing, preserve as an unlinked citation (blank URL).
    Extract a short quote span from `content` using start/end indexes when provided.
    """
    out: List[EvidenceQuote] = []
    if not citations:
        return out
    for idx, c in enumerate(citations):
        try:
            url = getattr(c, "url", "") or ""
            title = getattr(c, "title", "") or ""
            s = int(getattr(c, "start_index", 0) or 0)
            e = int(getattr(c, "end_index", s) or s)
            snippet = content[s:e][:240] if isinstance(content, str) else ""
            # Keep URL empty for unlinked citations; do not synthesize anchors
            domain = url.split('/')[2] if (url and '/' in url and len(url.split('/')) > 2) else (url or "")
            out.append(EvidenceQuote(
                id=f"dq{idx+1:03d}",
                url=url,
                title=title or (url.split('/')[-1] if url else "(deep research)"),
                domain=domain,
                quote=snippet or (title or ""),
                start=s if s >= 0 else None,
                end=e if e >= 0 else None,
                credibility_score=0.9,
                source_type="deep_research",
                doc_summary=(snippet[:240] if snippet else None),
            ))
        except Exception:
            continue
    return out
    
    def _estimate_costs(self, response: Dict[str, Any]) -> Dict[str, float]:
        """Estimate costs based on tool usage"""
        costs = {
            "web_search_calls": 0.0,
            "code_interpreter_calls": 0.0,
            "mcp_calls": 0.0,
            "total": 0.0,
        }
        
        # Count tool calls (simplified pricing)
        for item in response.get("output", []):
            if item["type"] == "web_search_call":
                costs["web_search_calls"] += 0.005  # $0.005 per search
            elif item["type"] == "code_interpreter_call":
                costs["code_interpreter_calls"] += 0.001  # $0.001 per execution
            elif item["type"] == "mcp_call":
                costs["mcp_calls"] += 0.002  # $0.002 per MCP call
        
        costs["total"] = sum(v for k, v in costs.items() if k != "total")
        return costs
    
    # ─────────── Background Execution ───────────
    async def resume_deep_research(
        self,
        research_id: str,
        progress_tracker: Optional[ResearchProgressTracker] = None,
    ) -> DeepResearchResult:
        """Resume a deep research task that was interrupted"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get research data from store
            research_data = await research_store.get(research_id)
            if not research_data:
                raise ValueError(f"Research {research_id} not found")
            
            # Get stored response ID
            response_id = research_data.get("deep_research_response_id")
            if not response_id:
                raise ValueError(f"No deep research response ID found for {research_id}")
            
            # Update progress
            if progress_tracker:
                await progress_tracker.update_progress(
                    research_id, "Resuming deep research", 30
                )
            
            # Check current status
            response = await responses_retrieve(response_id)
            status = ResponseStatus(response["status"])
            
            if status in [ResponseStatus.QUEUED, ResponseStatus.IN_PROGRESS]:
                # Still running, continue waiting
                response = await self._wait_with_progress(
                    response_id, 1800, progress_tracker, research_id
                )
            
            # Process the response
            classification = research_data.get("classification")
            if classification:
                classification = ClassificationResult(**classification)
            
            result = self._process_response(response, research_id, classification)
            
            # Update final progress
            if progress_tracker:
                await progress_tracker.update_progress(
                    research_id, "Deep research resumed and completed", 95
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to resume deep research: {str(e)}")
            if progress_tracker:
                await progress_tracker.update_progress(
                    research_id, f"Resume failed: {str(e)}", -1
                )
            raise
    
    async def _wait_with_progress(
        self,
        response_id: str,
        timeout: int,
        progress_tracker: Optional[ResearchProgressTracker],
        research_id: Optional[str],
    ) -> Dict[str, Any]:
        """Wait for background response with progress updates"""
        poll_interval = 5  # Check every 5 seconds
        elapsed = 0
        last_progress = 25
        # Track already-announced tool calls to emit itemized evidence-gathering progress
        seen_tool_ids: set[str] = set()
        
        # Store response ID for potential recovery
        if research_id:
            await research_store.update_field(
                research_id, "deep_research_response_id", response_id
            )
        
        while elapsed < timeout:
            try:
                response = await responses_retrieve(response_id)
                status = ResponseStatus(response["status"])
                
                if status not in [ResponseStatus.QUEUED, ResponseStatus.IN_PROGRESS]:
                    return response
                 
                # Emit tool-by-tool evidence gathering events the first time we see them
                try:
                    tool_calls = self.client.extract_tool_calls(response)
                except Exception:
                    tool_calls = None
                if progress_tracker and research_id and tool_calls:
                    for tc in tool_calls:
                        try:
                            tc_type = tc.get("type") or "tool_call"
                            # Prefer explicit IDs; fall back to a synthetic key
                            tc_id = tc.get("id") or f"{tc_type}:{tc.get('name') or tc.get('tool') or tc.get('server') or tc.get('server_label') or len(seen_tool_ids)+1}"
                            if tc_id not in seen_tool_ids:
                                seen_tool_ids.add(tc_id)
                                # Best-effort naming for UI
                                tc_name = tc.get("name") or tc.get("tool") or tc.get("server") or tc.get("server_label")
                                msg_name = f" {tc_name}" if tc_name else ""
                                # Stream as standard research_progress so FE status bar and log update
                                await progress_tracker.update_progress(
                                    research_id,
                                    message=f"Deep research: executing {tc_type}{msg_name}",
                                    items_done=len(seen_tool_ids),
                                    custom_data={
                                        "deep_research_tool": {
                                            "type": tc_type,
                                            "name": tc_name,
                                            "id": tc_id,
                                        }
                                    },
                                )
                        except Exception:
                            # Never fail polling loop due to progress emission
                            pass
                 
                # Update progress based on elapsed time
                if progress_tracker and research_id:
                    # Progress from 25% to 90% over the course of execution
                    progress = min(25 + (elapsed / timeout) * 65, 90)
                    if progress > last_progress + 5:  # Update every 5%
                        message = "Deep research in progress"
                        if status == ResponseStatus.IN_PROGRESS:
                            # Try to extract current action from response
                            tool_calls = self.client.extract_tool_calls(response)
                            if tool_calls:
                                last_call = tool_calls[-1]
                                if last_call["type"] == "web_search_call":
                                    message = "Searching web for information"
                                elif last_call["type"] == "code_interpreter_call":
                                    message = "Analyzing data"
                        
                        await progress_tracker.update_progress(
                            research_id, message, int(progress)
                        )
                        last_progress = progress
            
            except Exception as e:
                logger.warning(f"Error polling response {response_id}: {str(e)}")
                # Continue polling even if there's an error
            
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        
        # Timeout reached
        raise TimeoutError(f"Deep research {response_id} timed out after {timeout} seconds")
    
    # ─────────── Paradigm-Specific Methods ───────────
    async def revolutionary_deep_dive(
        self, query: str, research_id: Optional[str] = None
    ) -> DeepResearchResult:
        """Execute deep research with Dolores (revolutionary) perspective"""
        config = DeepResearchConfig(
            mode=DeepResearchMode.PARADIGM_FOCUSED,
            enable_web_search=True,
            include_paradigm_context=True,
        )
        
        # Create mock classification for Dolores
        classification = ClassificationResult(
            query=query,
            primary_paradigm=HostParadigm.DOLORES,
            secondary_paradigm=None,
            distribution={HostParadigm.DOLORES: 1.0},
            confidence=1.0,
            features=None,
            reasoning={},
            timestamp=datetime.utcnow(),
        )
        
        return await self.execute_deep_research(
            query=query,
            classification=classification,
            config=config,
            research_id=research_id,
        )
    
    async def analytical_deep_dive(
        self, query: str, research_id: Optional[str] = None
    ) -> DeepResearchResult:
        """Execute deep research with Bernard (analytical) perspective"""
        config = DeepResearchConfig(
            mode=DeepResearchMode.ANALYTICAL,
            enable_web_search=True,
            enable_code_interpreter=True,
            include_paradigm_context=True,
        )
        
        # Create mock classification for Bernard
        classification = ClassificationResult(
            query=query,
            primary_paradigm=HostParadigm.BERNARD,
            secondary_paradigm=None,
            distribution={HostParadigm.BERNARD: 1.0},
            confidence=1.0,
            features=None,
            reasoning={},
            timestamp=datetime.utcnow(),
        )
        
        return await self.execute_deep_research(
            query=query,
            classification=classification,
            config=config,
            research_id=research_id,
        )
    
    async def strategic_deep_dive(
        self, query: str, research_id: Optional[str] = None
    ) -> DeepResearchResult:
        """Execute deep research with Maeve (strategic) perspective"""
        config = DeepResearchConfig(
            mode=DeepResearchMode.STRATEGIC,
            enable_web_search=True,
            include_paradigm_context=True,
        )
        
        # Create mock classification for Maeve
        classification = ClassificationResult(
            query=query,
            primary_paradigm=HostParadigm.MAEVE,
            secondary_paradigm=None,
            distribution={HostParadigm.MAEVE: 1.0},
            confidence=1.0,
            features=None,
            reasoning={},
            timestamp=datetime.utcnow(),
        )
        
        return await self.execute_deep_research(
            query=query,
            classification=classification,
            config=config,
            research_id=research_id,
        )


# ────────────────────────────────────────────────────────────
#  Singleton Instance
# ────────────────────────────────────────────────────────────
deep_research_service = DeepResearchService()


async def initialize_deep_research():
    """Initialize deep research service"""
    await deep_research_service.initialize()
    return True
