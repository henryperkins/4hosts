# Subagent Quick Reference - Four Hosts Project

## 🎯 Quick Trigger Phrases

### paradigm-analyzer
**Triggers:** 
- "analyze paradigm alignment"
- "review code for paradigm consistency"
- "which paradigm does this feature serve"
- "check if this matches Dolores/Teddy/Bernard/Maeve approach"

**Example:**
```
"Analyze the paradigm alignment of the new search feature in paradigm_search.py"
```

### research-optimizer
**Triggers:**
- "optimize search performance"
- "reduce API costs"
- "improve query efficiency"
- "enhance search quality"
- "optimize the research pipeline"

**Example:**
```
"Optimize our search queries to reduce Google API usage while maintaining quality"
```

### test-engineer
**Triggers:**
- "write tests for"
- "create test cases"
- "test paradigm behavior"
- "validate the implementation"
- "add test coverage"

**Example:**
```
"Create comprehensive tests for the Dolores answer generator including edge cases"
```

### api-integrator
**Triggers:**
- "integrate new API"
- "add search source"
- "implement rate limiting"
- "add support for [service]"
- "handle API authentication"

**Example:**
```
"Integrate the Wikipedia API as a new search source for Bernard paradigm"
```

### llm-prompt-engineer
**Triggers:**
- "optimize prompts"
- "reduce token usage"
- "improve answer quality"
- "refine paradigm tone"
- "enhance LLM responses"

**Example:**
```
"Optimize the Maeve prompts to be more strategic and action-oriented"
```

### react-component-builder
**Triggers:**
- "create React component"
- "build UI for"
- "implement frontend"
- "design interface"
- "add paradigm styling"

**Example:**
```
"Create a React component to display paradigm-specific search results"
```

## 🔄 Common Workflows

### Adding a New Feature
```
1. "Analyze which paradigm this feature aligns with"
2. "Design the API integration for [feature]"  
3. "Build the React components for [feature]"
4. "Write comprehensive tests for [feature]"
5. "Optimize the performance of [feature]"
```

### Improving Existing Code
```
1. "Review the paradigm consistency in [component]"
2. "Optimize the search queries in [component]"
3. "Refine the prompts used by [component]"
4. "Update tests to cover new edge cases"
```

### Debugging Issues
```
1. "Analyze why [paradigm] classification is failing"
2. "Review the search API integration for errors"
3. "Check if the React components handle [error case]"
4. "Write tests to reproduce the issue"
```

## 💡 Pro Tips

### Be Specific About Files
❌ "Fix the search"
✅ "Optimize the search performance in search_apis.py, focusing on rate limiting"

### Reference Paradigms Explicitly
❌ "Make it more analytical"
✅ "Enhance the Bernard paradigm's analytical tone in answer generation"

### Chain Subagents Naturally
❌ "Build the feature"
✅ "Let's build this feature: first analyze paradigm fit, then create the API, build UI, and add tests"

### Use Project Terminology
- W-S-C-I pipeline
- Context engineering
- Paradigm alignment
- Search orchestration
- Answer synthesis

## 📊 Subagent Selection Matrix

| Task Type | Primary Subagent | Supporting Subagents |
|-----------|------------------|---------------------|
| Code Review | paradigm-analyzer | test-engineer |
| Performance | research-optimizer | api-integrator |
| New API | api-integrator | paradigm-analyzer, test-engineer |
| UI Work | react-component-builder | paradigm-analyzer |
| Testing | test-engineer | All others for context |
| Prompts | llm-prompt-engineer | paradigm-analyzer |

## 🚀 Quick Start Examples

### "I need to improve Dolores search results"
```
1. Use paradigm-analyzer: "Review Dolores search strategy implementation"
2. Use research-optimizer: "Optimize Dolores search queries for investigative sources"
3. Use llm-prompt-engineer: "Enhance Dolores prompts for more revolutionary tone"
4. Use test-engineer: "Create tests for improved Dolores search"
```

### "Add a new visualization component"
```
1. Use paradigm-analyzer: "Determine paradigm-specific visualization needs"
2. Use react-component-builder: "Create paradigm-aware visualization component"
3. Use test-engineer: "Write tests for the visualization component"
```

### "System is too slow"
```
1. Use research-optimizer: "Identify performance bottlenecks in the research pipeline"
2. Use api-integrator: "Optimize API call patterns and caching"
3. Use llm-prompt-engineer: "Reduce token usage in prompts"
```

Remember: The more specific and paradigm-aware your requests, the better the subagents perform!