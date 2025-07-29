# Azure OpenAI Integration Guide

This guide covers the enhanced Azure OpenAI implementation in the Four Hosts Research Application.

## Overview

The application now supports the latest Azure OpenAI API (2025-04-01-preview) with advanced features including:
- Structured outputs with JSON schema
- Streaming responses
- Tool/function calling
- Multi-turn conversations
- Paradigm-specific content generation

## Configuration

### Environment Variables

Add the following to your `.env` file:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_API_VERSION=2025-04-01-preview
AZURE_GPT4_DEPLOYMENT_NAME=gpt-4o
AZURE_GPT4_MINI_DEPLOYMENT_NAME=gpt-4o-mini
```

### Model Deployments

Ensure you have deployed the following models in your Azure OpenAI resource:
- `gpt-4o` - For advanced reasoning (Dolores, Bernard paradigms)
- `gpt-4o-mini` - For efficient processing (Teddy, Maeve paradigms)

## Usage Examples

### Basic Completion

```python
from services.llm_client import llm_client

response = await llm_client.generate_completion(
    prompt="Explain AI consciousness",
    paradigm="bernard",
    max_tokens=500,
    temperature=0.7
)
```

### Streaming Response

```python
stream = await llm_client.generate_completion(
    prompt="Write about AI ethics",
    paradigm="dolores",
    stream=True
)

async for chunk in stream:
    print(chunk, end="", flush=True)
```

### Structured Output

```python
schema = {
    "name": "analysis",
    "schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "key_points": {
                "type": "array",
                "items": {"type": "string"}
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1
            }
        },
        "required": ["summary", "key_points", "confidence"]
    }
}

result = await llm_client.generate_structured_output(
    prompt="Analyze this research paper",
    schema=schema,
    paradigm="bernard"
)
```

### Tool Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "search_database",
        "description": "Search the research database",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "filters": {"type": "object"}
            },
            "required": ["query"]
        }
    }
}]

response = await llm_client.generate_with_tools(
    prompt="Find recent papers on consciousness",
    tools=tools,
    tool_choice="auto"
)
```

### Multi-turn Conversation

```python
messages = [
    {"role": "user", "content": "What is consciousness?"},
    {"role": "assistant", "content": "Consciousness is..."},
    {"role": "user", "content": "How does it relate to AI?"}
]

response = await llm_client.create_conversation(
    messages=messages,
    paradigm="bernard"
)
```

## Paradigm-Specific Generation

The system supports four paradigms with different characteristics:

### Dolores (Revolutionary)
- Model: gpt-4o
- Temperature: 0.8
- Focus: Exposing systemic issues, emotional impact
- Use for: Critical analysis, investigative content

### Teddy (Supportive)
- Model: gpt-4o-mini
- Temperature: 0.6
- Focus: Empathy, practical help, resources
- Use for: Support content, user guidance

### Bernard (Analytical)
- Model: gpt-4o
- Temperature: 0.4
- Focus: Data-driven analysis, objectivity
- Use for: Research summaries, technical content

### Maeve (Strategic)
- Model: gpt-4o-mini
- Temperature: 0.6
- Focus: Competitive advantage, tactics
- Use for: Strategic recommendations, planning

## Error Handling

The implementation includes:
- Automatic retry with exponential backoff
- Fallback to OpenAI when Azure fails
- Detailed error logging
- Connection timeout handling

## Testing

Run the basic test:
```bash
python tests/test_azure_openai.py
```

Run the enhanced test suite:
```bash
python tests/test_azure_openai_enhanced.py
```

## Performance Considerations

1. **Model Selection**: Use gpt-4o-mini for routine tasks to optimize costs
2. **Streaming**: Enable streaming for long-form content generation
3. **Caching**: Consider implementing response caching for repeated queries
4. **Rate Limiting**: Monitor Azure OpenAI quotas and implement appropriate throttling

## Security Best Practices

1. Store API keys in environment variables, never in code
2. Use Azure Key Vault for production deployments
3. Implement request validation and sanitization
4. Monitor API usage for anomalies
5. Use managed identities when possible

## Troubleshooting

### Common Issues

1. **API Version Mismatch**
   - Ensure `AZURE_OPENAI_API_VERSION` is set to `2025-04-01-preview`
   - Check that your Azure resource supports the API version

2. **Model Not Found**
   - Verify deployment names match environment variables
   - Ensure models are deployed in your Azure resource

3. **Authentication Failures**
   - Confirm API key is correct and active
   - Check endpoint URL format (should end with `/`)

4. **Timeout Errors**
   - Increase timeout settings for long operations
   - Consider using streaming for large responses

## Future Enhancements

- [ ] Implement conversation state persistence
- [ ] Add support for image generation (DALL-E)
- [ ] Integrate with Azure Cognitive Search
- [ ] Add telemetry and monitoring
- [ ] Implement response caching layer