# Azure OpenAI o3 Integration Guide

This guide documents the integration of Azure OpenAI's o3 model into the Four Hosts Research Application for enhanced synthesis capabilities.

## Overview

We've replaced the existing summarization models (Pegasus/Flan-T5) with Azure OpenAI's o3 model, which provides:
- 200,000 token context window
- 100,000 token output limit
- Superior reasoning and synthesis capabilities
- Paradigm-aware content generation

## Configuration

### 1. Environment Variables

Add these to your `.env` file:

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_KEY=<your-api-key>
AZURE_OPENAI_DEPLOYMENT=o3-synthesis
AZURE_OPENAI_API_VERSION=2024-05-01-preview
```

### 2. Azure Portal Setup

1. Navigate to your Azure OpenAI resource
2. Go to **Deployments** → **Create**
3. Configure:
   - Model: **gpt-4o** (o3 variant)
   - Deployment Name: **o3-synthesis**
   - API Version: `2024-05-01-preview` or newer

## Implementation Details

### LLM Client Updates

The `services/llm_client.py` has been enhanced with:
- Azure OpenAI client initialization
- Model routing for "o3" requests to Azure
- Fallback to OpenAI for non-o3 models

### Answer Generator Updates

All answer generators now use o3 for synthesis:
- `EnhancedBernardAnswerGenerator`: Analytical synthesis with o3
- `EnhancedMaeveAnswerGenerator`: Strategic synthesis with o3
- Base generators in `answer_generator.py`: Updated for o3

### Key Changes

1. **Model Parameter**: All `generate_paradigm_content()` calls now include `model="o3"`
2. **Token Budgeting**: Respects o3's 200k context limit
3. **Temperature Settings**: Optimized for each paradigm (0.3 for Bernard, 0.5 for Maeve)

## Usage Examples

### Basic Synthesis
```python
content = await llm_client.generate_paradigm_content(
    prompt="Synthesize these research findings...",
    paradigm="bernard",
    max_tokens=2000,
    temperature=0.3,
    model="o3"  # Routes to Azure
)
```

### With Source Formatting
```python
prompt = """
=== SOURCE 1 ===
<title>Study Title</title>
<snippet>Key findings...</snippet>

=== SOURCE 2 ===
<title>Another Study</title>
<snippet>More findings...</snippet>

Write a coherent 3-paragraph synthesis with numbered citations [1]...
"""
```

## Testing

Run the test script to verify integration:

```bash
cd backend
source venv/bin/activate
python test_azure_o3.py
```

## Safety & Fallbacks

The system includes fallbacks for:
- Rate limit errors → Falls back to OpenAI
- Azure unavailable → Uses OpenAI with standard models
- Token limits exceeded → Truncates context appropriately

## Performance Considerations

1. **Context Management**: Never exceed ~180k prompt tokens
2. **Output Planning**: Reserve at least 2k tokens for responses
3. **Batch Processing**: Group related synthesis tasks when possible

## Monitoring

Check logs for:
- "✓ Azure OpenAI client initialised" - Successful connection
- "Azure OpenAI request failed" - Connection/auth issues
- Model routing decisions in synthesis operations

## Troubleshooting

### Common Issues

1. **Authentication Error**
   - Verify AZURE_OPENAI_KEY is correct
   - Check deployment name matches exactly

2. **404 Not Found**
   - Confirm deployment name in Azure Portal
   - Verify API version compatibility

3. **Rate Limits**
   - System automatically falls back to OpenAI
   - Consider implementing request throttling

## Future Enhancements

- Implement reasoning effort parameters for o3
- Add response format controls (JSON mode)
- Integrate tool calling capabilities
- Optimize prompt templates for o3's strengths