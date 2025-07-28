Based on your Four Hosts architecture and paradigm concept, here's a comprehensive strategy for tools/MCPs and context management:

## 🛠️ Recommended MCP Tools

### 1. **Core MCPs for Research System**

```python
# MCP Configuration
MCP_TOOLS = {
    "filesystem": {
        "purpose": "Store/retrieve research artifacts",
        "usage": {
            "cache_dir": "/research_cache",
            "paradigm_templates": "/templates/{paradigm}",
            "user_sessions": "/sessions/{user_id}"
        }
    },
    "web_search": {
        "purpose": "Direct search integration",
        "paradigm_routing": True
    },
    "project_knowledge": {
        "purpose": "Paradigm knowledge base",
        "indexes": ["paradigm_rules", "source_credibility", "templates"]
    }
}
```

### 2. **Paradigm-Specific Tool Routing**

```python
class ParadigmAwareToolRouter:
    """Routes tool calls based on active paradigm"""

    PARADIGM_TOOL_MATRIX = {
        "DOLORES": {
            "primary_tools": ["web_search", "investigate_tool"],
            "search_modifiers": ["expose", "victims", "corruption"],
            "credibility_filters": ["independent_media", "whistleblowers"],
            "context_depth": "deep"  # Keep full investigation trail
        },
        "BERNARD": {
            "primary_tools": ["academic_search", "data_analysis"],
            "search_modifiers": ["peer-reviewed", "empirical", "statistical"],
            "credibility_filters": ["academic_journals", "research_institutes"],
            "context_depth": "structured"  # Methodical, organized
        },
        "TEDDY": {
            "primary_tools": ["community_search", "support_finder"],
            "search_modifiers": ["help", "support", "resources"],
            "credibility_filters": ["nonprofits", "community_orgs"],
            "context_depth": "empathetic"  # Focus on human stories
        },
        "MAEVE": {
            "primary_tools": ["market_intelligence", "strategy_analyzer"],
            "search_modifiers": ["competitive", "strategic", "opportunity"],
            "credibility_filters": ["business_intel", "consultancies"],
            "context_depth": "actionable"  # Strategic insights only
        }
    }

    async def route_tool_call(self, tool_name: str, params: dict, paradigm: str):
        """Route tool calls through paradigm lens"""
        config = self.PARADIGM_TOOL_MATRIX[paradigm]

        # Modify parameters based on paradigm
        if tool_name == "web_search":
            params["query"] += f" {' '.join(config['search_modifiers'])}"
            params["source_filter"] = config["credibility_filters"]

        return await self.execute_tool(tool_name, params)
```

## 📚 Context Management Strategy

### 1. **Hierarchical Context System**

```python
class ParadigmContextManager:
    """Manages context at multiple levels"""

    def __init__(self):
        self.contexts = {
            "session": {},      # User session context
            "paradigm": {},     # Active paradigm context
            "research": {},     # Current research context
            "knowledge": {}     # Persistent knowledge base
        }

    async def build_context(self, query: str, paradigm: str) -> dict:
        """Build layered context for LLM calls"""

        # Layer 1: Paradigm Identity
        paradigm_context = self.get_paradigm_identity(paradigm)

        # Layer 2: Research History (paradigm-filtered)
        research_context = await self.get_research_context(
            paradigm=paradigm,
            depth=self.PARADIGM_TOOL_MATRIX[paradigm]["context_depth"]
        )

        # Layer 3: Dynamic Context (from current search)
        dynamic_context = await self.get_dynamic_context(query, paradigm)

        # Layer 4: User Preferences
        user_context = self.get_user_preferences()

        return self.merge_contexts(
            paradigm_context,
            research_context,
            dynamic_context,
            user_context
        )
```

### 2. **Context Window Optimization**

```python
class ContextWindowOptimizer:
    """Optimize context for different paradigms"""

    PARADIGM_CONTEXT_STRATEGIES = {
        "DOLORES": {
            "max_tokens": 8000,
            "prioritize": ["evidence", "injustices", "patterns"],
            "compression": "narrative",  # Keep story coherent
            "memory": "long"  # Remember everything
        },
        "BERNARD": {
            "max_tokens": 6000,
            "prioritize": ["data", "methodology", "findings"],
            "compression": "structured",  # Bullet points, tables
            "memory": "selective"  # Only verified facts
        },
        "TEDDY": {
            "max_tokens": 5000,
            "prioritize": ["needs", "solutions", "stories"],
            "compression": "empathetic",  # Preserve human element
            "memory": "supportive"  # Focus on help given
        },
        "MAEVE": {
            "max_tokens": 4000,
            "prioritize": ["strategies", "tactics", "outcomes"],
            "compression": "executive",  # Key points only
            "memory": "strategic"  # Actionable insights
        }
    }

    def optimize_for_paradigm(self, context: dict, paradigm: str) -> dict:
        """Compress context based on paradigm needs"""
        strategy = self.PARADIGM_CONTEXT_STRATEGIES[paradigm]

        # Apply paradigm-specific compression
        if strategy["compression"] == "narrative":
            return self._narrative_compression(context)
        elif strategy["compression"] == "structured":
            return self._structured_compression(context)
        # etc...
```

### 3. **Context Persistence Strategy**

```python
class ParadigmMemorySystem:
    """Long-term memory per paradigm"""

    def __init__(self):
        self.memories = {
            "DOLORES": RevolutionaryMemory(),  # Tracks patterns of injustice
            "BERNARD": AnalyticalMemory(),     # Builds knowledge graphs
            "TEDDY": CompassionateMemory(),    # Remembers who needs help
            "MAEVE": StrategicMemory()         # Maps competitive landscape
        }

    async def store_finding(self, paradigm: str, finding: dict):
        """Store findings in paradigm-appropriate memory"""
        memory = self.memories[paradigm]

        # Each paradigm remembers differently
        if paradigm == "DOLORES":
            await memory.add_pattern(finding)  # Connect to larger patterns
        elif paradigm == "BERNARD":
            await memory.add_fact(finding)     # Verify and cross-reference
        elif paradigm == "TEDDY":
            await memory.add_need(finding)     # Track who needs what
        elif paradigm == "MAEVE":
            await memory.add_insight(finding)  # Strategic implications
```

## 🔄 Integrated Tool & Context Flow

```python
class IntegratedResearchSystem:
    """Combines tools and context management"""

    async def research_with_paradigm(self, query: str):
        # 1. Classification with minimal context
        paradigm = await self.classify_query(query)

        # 2. Load paradigm-specific context
        context = await self.context_manager.build_context(query, paradigm)

        # 3. Execute paradigm-aware tool calls
        tools_to_use = self.get_paradigm_tools(paradigm)

        results = []
        for tool in tools_to_use:
            # Each tool call includes paradigm context
            result = await self.call_tool_with_context(
                tool=tool,
                query=query,
                context=context,
                paradigm=paradigm
            )
            results.append(result)

            # Update context with findings
            context = await self.update_context(context, result, paradigm)

        # 4. Synthesize with full context
        answer = await self.synthesize_answer(
            query=query,
            results=results,
            context=context,
            paradigm=paradigm
        )

        # 5. Store in paradigm memory
        await self.memory_system.store_finding(paradigm, answer)

        return answer
```

## 💡 Best Practices

### 1. **Context Budgeting**
```python
# Allocate context tokens by importance
CONTEXT_BUDGET = {
    "paradigm_identity": 500,    # Core paradigm rules
    "search_results": 3000,      # Current findings
    "conversation_history": 1500, # Recent exchanges
    "knowledge_base": 1000       # Relevant facts
}
```

### 2. **Tool Call Optimization**
```python
# Batch compatible tool calls
async def batch_tool_calls(paradigm: str, queries: list):
    # Group by tool type
    grouped = group_by_tool(queries, paradigm)

    # Execute in parallel where possible
    results = await asyncio.gather(*[
        tool.batch_execute(group)
        for tool, group in grouped.items()
    ])
```

### 3. **Context Decay Strategy**
```python
# Different paradigms forget differently
CONTEXT_DECAY_RATES = {
    "DOLORES": 0.1,   # Never forget injustices
    "BERNARD": 0.3,   # Keep verified facts longer
    "TEDDY": 0.5,     # Focus on current needs
    "MAEVE": 0.7      # Rapid strategy adaptation
}
```

## 🚀 Implementation Priority

1. **Phase 1**: Implement `ParadigmAwareToolRouter` for existing tools
2. **Phase 2**: Build `ParadigmContextManager` with basic layering
3. **Phase 3**: Add `ContextWindowOptimizer` for efficiency
4. **Phase 4**: Implement `ParadigmMemorySystem` for learning

This approach ensures each paradigm maintains its unique perspective while efficiently managing tools and context. The key is that **context shapes tool usage**, and **tool results reshape context** - creating a dynamic, paradigm-aware research system.
