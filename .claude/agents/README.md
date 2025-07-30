# Four Hosts Custom Subagents

This directory contains specialized subagents for the Four Hosts application. Each subagent is designed to handle specific aspects of the system with deep domain knowledge.

## Available Subagents

### 1. paradigm-analyzer
**Purpose**: Analyzes code components for paradigm alignment
**Use Cases**:
- Reviewing new features for paradigm consistency
- Refactoring code to better match paradigm principles
- Identifying paradigm conflicts in implementations

**Example Usage**:
```
"Use the paradigm-analyzer subagent to review the new search implementation"
```

### 2. research-optimizer
**Purpose**: Optimizes the research pipeline for better performance and quality
**Use Cases**:
- Improving search query generation
- Enhancing answer quality
- Reducing API costs
- Performance optimization

**Example Usage**:
```
"Use the research-optimizer subagent to improve search performance for Dolores paradigm"
```

### 3. test-engineer
**Purpose**: Creates comprehensive tests for all system components
**Use Cases**:
- Writing unit tests for new features
- Creating paradigm-specific test cases
- Integration testing
- Performance benchmarking

**Example Usage**:
```
"Use the test-engineer subagent to create tests for the context engineering pipeline"
```

### 4. api-integrator
**Purpose**: Integrates new search APIs and optimizes existing ones
**Use Cases**:
- Adding new data sources
- Implementing rate limiting
- Error handling improvements
- API cost optimization

**Example Usage**:
```
"Use the api-integrator subagent to add support for Wikipedia API"
```

### 5. llm-prompt-engineer
**Purpose**: Optimizes LLM prompts for better paradigm alignment
**Use Cases**:
- Refining paradigm-specific prompts
- Reducing token usage
- Improving response quality
- Debugging LLM outputs

**Example Usage**:
```
"Use the llm-prompt-engineer subagent to optimize Bernard's analytical prompts"
```

### 6. react-component-builder
**Purpose**: Creates React components with paradigm-specific UI patterns
**Use Cases**:
- Building new frontend components
- Implementing paradigm-specific styling
- Creating reusable UI patterns
- Refactoring existing components
- Ensuring TypeScript type safety

**Example Usage**:
```
"Use the react-component-builder subagent to create a paradigm-aware results display component"
```

## Creating New Subagents

To create a new subagent:

1. Create a new `.md` file in this directory
2. Add the frontmatter with name, description, and tools
3. Write a detailed system prompt explaining the subagent's role
4. Include specific expertise areas and guidelines

### Template:
```markdown
---
name: your-subagent-name
description: Brief description of when to use this subagent
tools: Read, Write, MultiEdit, Bash, Grep
---

Detailed system prompt defining the subagent's role, expertise, and approach...
```

## Best Practices

1. **Focus**: Each subagent should have a clear, specific purpose
2. **Tools**: Only grant access to necessary tools
3. **Context**: Provide domain-specific knowledge in the prompt
4. **Examples**: Include concrete examples of tasks to handle
5. **Integration**: Explain how the subagent fits into the larger system

## Version Control

These subagents are tracked in git. When modifying:
1. Test changes thoroughly
2. Document the rationale for changes
3. Consider backward compatibility
4. Update this README if adding new subagents