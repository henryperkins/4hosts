# Guide: Ensuring Claude Code Makes Effective Use of Subagents

This guide explains how to maximize the effectiveness of custom subagents in Claude Code for the Four Hosts project.

## How Claude Code Selects Subagents

### 1. **Automatic Delegation**
Claude Code automatically considers using subagents when:
- The task matches the subagent's description
- The current context suggests specialized knowledge is needed
- Multiple related tasks could benefit from focused expertise

### 2. **Explicit Invocation**
You can directly request a subagent:
```
"Use the paradigm-analyzer subagent to review the new search implementation"
"Use the react-component-builder to create a new results display component"
```

## Best Practices for Effective Subagent Usage

### 1. **Write Clear, Specific Descriptions**

**Good Description:**
```yaml
description: Analyzes code and system components to identify which Four Hosts paradigm (Dolores, Teddy, Bernard, Maeve) they align with. Use when examining system architecture or refactoring code.
```

**Why it works:**
- Mentions specific paradigms
- Clear use cases (examining, refactoring)
- Directly relates to project concepts

### 2. **Include Trigger Keywords in Descriptions**

Add keywords that match common tasks:
```yaml
# For paradigm-analyzer
description: ... Use when examining system architecture, refactoring code, reviewing paradigm alignment, or checking paradigm consistency.

# For test-engineer  
description: ... Use when writing tests, creating test cases, setting up test fixtures, or validating paradigm behavior.
```

### 3. **Provide Comprehensive Context in System Prompts**

Each subagent should know:
- Project-specific terminology
- File locations and patterns
- Common workflows
- Integration points

Example from our react-component-builder:
```markdown
## Actual Project Structure:
src/
  components/
    auth/              # LoginForm, RegisterForm, ProtectedRoute
    ui/                # Alert, Badge, Button, Card, Dialog, etc.
```

### 4. **Grant Appropriate Tools**

Match tools to the subagent's needs:
```yaml
# For code analysis
tools: Read, Grep, Glob

# For code modification
tools: Read, Write, MultiEdit, Bash

# For API testing
tools: Read, Write, Bash, WebFetch
```

## Strategies to Encourage Subagent Use

### 1. **Phrase Requests to Match Subagent Descriptions**

Instead of:
> "Fix the search feature"

Say:
> "Optimize the research pipeline for better search performance"
(Triggers research-optimizer)

### 2. **Break Complex Tasks into Specialized Parts**

Instead of:
> "Add a new feature for document analysis"

Say:
> "I need to add document analysis. First, let's analyze which paradigm this aligns with, then create the API integration, build the UI components, and write tests."

This triggers multiple subagents:
1. paradigm-analyzer (alignment check)
2. api-integrator (API work)
3. react-component-builder (UI)
4. test-engineer (tests)

### 3. **Use Domain-Specific Language**

Use terms from your subagent prompts:
- "paradigm alignment" → paradigm-analyzer
- "optimize search queries" → research-optimizer
- "create React component" → react-component-builder
- "integrate new API" → api-integrator
- "optimize prompts" → llm-prompt-engineer

### 4. **Reference Specific Files or Patterns**

> "Review the paradigm consistency in answer_generator.py"
> "Optimize the W-S-C-I pipeline in context_engineering.py"
> "Create tests for the classification engine"

## Examples of Effective Prompts

### Example 1: Multi-Subagent Task
```
"I want to add support for Wikipedia as a new search source. Let's start by analyzing which paradigms would benefit most from Wikipedia data, then integrate the API, and optimize the search queries for it."
```
This engages:
- paradigm-analyzer (paradigm benefits)
- api-integrator (API integration)
- research-optimizer (query optimization)

### Example 2: Focused Subagent Use
```
"Use the test-engineer subagent to create comprehensive tests for the new Dolores search strategy, including edge cases for revolutionary queries"
```

### Example 3: Sequential Subagent Tasks
```
"Let's improve the Bernard paradigm's analytical capabilities:
1. First, analyze the current Bernard implementation
2. Then optimize the prompts for more academic rigor
3. Create UI components to better display statistical data
4. Finally, write tests to validate the improvements"
```

## Monitoring Subagent Effectiveness

### Signs of Good Subagent Usage:
- Claude references specific files and line numbers from the subagent's knowledge
- Responses include project-specific patterns and conventions
- Code follows established paradigms without being told
- Appropriate tools are used for each task

### Signs Subagents Could Be Better Utilized:
- Claude seems unaware of project structure
- Generic solutions instead of paradigm-specific ones
- Missing project conventions in generated code
- Not referencing the specialized knowledge in subagents

## Advanced Tips

### 1. **Chain Subagents for Complex Workflows**
```
"Analyze this feature request → Design the API → Build components → Write tests"
```

### 2. **Provide Feedback on Subagent Performance**
If a subagent isn't performing well:
- Update its description to be more specific
- Add missing context to its system prompt
- Include more examples in its prompt

### 3. **Create Project-Specific Workflows**
Document common task patterns:
```markdown
## Adding a New Paradigm Feature:
1. Use paradigm-analyzer to ensure alignment
2. Use api-integrator if new data sources needed
3. Use react-component-builder for UI
4. Use test-engineer for comprehensive tests
5. Use research-optimizer to ensure performance
```

### 4. **Leverage CLAUDE.md**
Keep CLAUDE.md updated with:
- Current project state
- Recent architectural decisions
- Common patterns
- Known issues

This helps all subagents stay coordinated.

## Troubleshooting

### Subagent Not Being Used:
1. Check if description matches the task
2. Try explicit invocation
3. Verify tools are appropriate
4. Update description with more triggers

### Subagent Giving Generic Responses:
1. Add more project context to system prompt
2. Include specific examples
3. Reference actual files and patterns
4. Update with recent project changes

### Subagent Conflicts:
If multiple subagents could handle a task:
1. Be explicit about which to use
2. Break task into parts for each subagent
3. Update descriptions to be more distinct

## Maintenance

Regularly update subagents with:
- New project patterns
- Changed file structures  
- Updated dependencies
- Lessons learned from usage

Remember: Effective subagents are living documents that evolve with your project!