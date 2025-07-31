# Anthropic Claude Integration

This document describes the integration of Anthropic's Claude Sonnet 4 and Opus 4 models into the Four Hosts research application.

## Overview

The Four Hosts application now supports both OpenAI and Anthropic LLM providers, allowing users to leverage the capabilities of Claude models alongside existing GPT models.

## Supported Models

### Claude Sonnet 4
- **Model ID**: `claude-3-5-sonnet-20250123`
- **Capabilities**: Balanced performance, good for most tasks
- **Paradigm Assignments**: Dolores (revolutionary), Teddy (devotion), Maeve (strategic)

### Claude Opus 4  
- **Model ID**: `claude-3-opus-20250123`
- **Capabilities**: Most capable model, best for complex analysis
- **Paradigm Assignment**: Bernard (analytical) - gets the most powerful model for detailed research

## Configuration

### Environment Variables

Add your Anthropic API key to enable Claude models:

```bash
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

The application will automatically detect and initialize the Anthropic client when this environment variable is set.

### Multiple Provider Support

The application can be configured with multiple LLM providers simultaneously:

```bash
# OpenAI
export OPENAI_API_KEY="your_openai_key"

# Azure OpenAI  
export AZURE_OPENAI_API_KEY="your_azure_key"
export AZURE_OPENAI_ENDPOINT="your_azure_endpoint"
export AZURE_OPENAI_DEPLOYMENT="your_deployment"

# Anthropic
export ANTHROPIC_API_KEY="your_anthropic_key"
```

## API Usage

### Basic Completion

```python
from services.llm_client import llm_client

# Use OpenAI model (default)
response = await llm_client.generate_completion(
    prompt="Analyze market trends",
    paradigm="bernard",
    provider="openai"
)

# Use Anthropic model
response = await llm_client.generate_completion(
    prompt="Analyze market trends", 
    paradigm="bernard",
    provider="anthropic"  # Will use claude-3-opus-20250123 for bernard
)
```

### Auto-Detection

The system can automatically detect the provider from explicit model names:

```python
# Automatically uses Anthropic
response = await llm_client.generate_completion(
    prompt="Research question",
    model="claude-3-5-sonnet-20250123",
    paradigm="dolores"
)

# Automatically uses OpenAI
response = await llm_client.generate_completion(
    prompt="Research question", 
    model="gpt-4o",
    paradigm="dolores"
)
```

### Paradigm-Specific Content

```python
# Generate content with paradigm-specific prompts and models
response = await llm_client.generate_paradigm_content(
    prompt="Investigate corporate malfeasance",
    paradigm="dolores",  # Revolutionary perspective
    provider="anthropic"  # Uses claude-3-5-sonnet-20250123
)
```

### Tool Calling

Claude models support tool calling with automatic format conversion:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "search_database",
        "description": "Search research database",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            }
        }
    }
}]

result = await llm_client.generate_with_tools(
    prompt="Find research on AI safety",
    tools=tools,
    paradigm="bernard",
    provider="anthropic"
)
```

## Paradigm Model Mappings

| Paradigm | OpenAI Model | Anthropic Model | Rationale |
|----------|-------------|-----------------|-----------|
| **Dolores** (Revolutionary) | gpt-4o | claude-3-5-sonnet-20250123 | Balanced model for investigative work |
| **Teddy** (Devotion) | gpt-4o-mini | claude-3-5-sonnet-20250123 | Sonnet for empathetic, helpful responses |
| **Bernard** (Analytical) | gpt-4o | claude-3-opus-20250123 | Most capable model for rigorous analysis |
| **Maeve** (Strategic) | gpt-4o-mini | claude-3-5-sonnet-20250123 | Balanced model for strategic insights |

## Streaming Support

Both providers support streaming responses:

```python
# Streaming with Anthropic
stream = await llm_client.generate_completion(
    prompt="Long research analysis",
    paradigm="bernard",
    provider="anthropic",
    stream=True
)

async for token in stream:
    print(token, end="", flush=True)
```

## Error Handling

The integration includes proper error handling and fallbacks:

```python
try:
    response = await llm_client.generate_completion(
        prompt="Research query",
        provider="anthropic"
    )
except RuntimeError as e:
    # Falls back or raises clear error about missing API key
    print(f"Anthropic integration error: {e}")
```

## Cost Considerations

### Model Costs (Approximate)
- **Claude Sonnet 4**: Moderate cost, good value for most use cases
- **Claude Opus 4**: Higher cost, reserved for complex analytical tasks (Bernard paradigm)
- **Comparison**: Opus 4 is used selectively for Bernard's analytical work to balance cost with capability

### Usage Recommendations
1. Use **Sonnet 4** for general research tasks (Dolores, Teddy, Maeve)
2. Use **Opus 4** for complex analytical work requiring maximum capability (Bernard)
3. Consider OpenAI models for simpler tasks where appropriate
4. Monitor API usage and costs across providers

## Implementation Details

### Architecture
- Anthropic integration added to existing `LLMClient` class
- Maintains backward compatibility with all existing OpenAI functionality
- Provider auto-detection based on model names
- Unified API interface across all providers

### Security
- API keys stored as environment variables
- No hardcoded credentials in source code
- Client initialization with secure key handling

### Testing
- Comprehensive test suite in `tests/test_anthropic_integration.py`
- Mock-based testing for safe CI/CD
- Validation of model mappings and paradigm assignments

## Troubleshooting

### Common Issues

1. **"Anthropic client not configured"**
   - Solution: Set `ANTHROPIC_API_KEY` environment variable

2. **"No LLM client configured"** 
   - Solution: Set at least one provider's API key (OpenAI, Azure, or Anthropic)

3. **Model not found errors**
   - Solution: Verify model names and API access for requested models

4. **Unexpected provider selection**
   - Solution: Use explicit `provider` parameter or check model name detection logic

### Debug Information

Enable debug logging to see provider selection:

```python
import logging
logging.getLogger("services.llm_client").setLevel(logging.DEBUG)
```

## Future Enhancements

- Support for additional Claude model variants
- Fine-tuning integration when available
- Advanced prompt engineering for Claude-specific optimizations
- Cost tracking and optimization features