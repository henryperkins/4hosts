# Complete Groundedness Detection Guide

This guide provides comprehensive documentation for the Azure Content Safety groundedness detection integration in the Four Hosts application.

## Overview

Groundedness detection ensures AI-generated answers are properly grounded in source materials. The implementation supports three modes:
- **Basic**: Simple groundedness percentage detection
- **Reasoning**: Explains why content is ungrounded
- **Correction**: Suggests corrections for ungrounded content

## Environment Configuration

### Required Variables

```bash
# Enable groundedness detection
AZURE_CS_ENABLE_GROUNDEDNESS=1

# Content Safety endpoint and key
AZURE_CS_ENDPOINT=https://<name>.cognitiveservices.azure.com/
AZURE_CS_KEY=<your-content-safety-key>
```

### Optional Variables for Advanced Features

```bash
# Azure OpenAI (required for reasoning/correction)
AZURE_OPENAI_ENDPOINT=https://<name>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=<deployment-name>

# Feature modes (mutually exclusive)
AZURE_CS_GROUNDEDNESS_REASONING=1      # Enable reasoning mode
AZURE_CS_GROUNDEDNESS_CORRECTION=1     # Enable correction mode

# Performance tuning
AZURE_HTTP_TIMEOUT_SECONDS=12          # API timeout (default: 12)
```

## Feature Modes

### Basic Mode (Default)

Returns groundedness percentage and ungrounded text segments.

```json
{
  "ungrounded_detected": true,
  "ungrounded_percentage": 0.3,
  "ungrounded_details": [
    {
      "text": "ungrounded segment",
      "offset": {...},
      "length": {...}
    }
  ],
  "mode": "basic"
}
```

### Reasoning Mode

Provides explanations for why content is ungrounded.

Requirements:
- Azure OpenAI resource with GPT-4o deployment
- Content Safety MI with `Cognitive Services OpenAI User` role

```json
{
  "ungrounded_detected": true,
  "ungrounded_percentage": 0.3,
  "ungrounded_details": [...],
  "reasoning": [
    "The name 'John' doesn't match the source which states 'Jane'"
  ],
  "mode": "reasoning"
}
```

### Correction Mode

Suggests corrected text that aligns with sources.

Requirements:
- Same as reasoning mode
- Mutually exclusive with reasoning mode

```json
{
  "ungrounded_detected": true,
  "ungrounded_percentage": 0.3,
  "ungrounded_details": [...],
  "correction_text": "The patient's name is Jane.",
  "mode": "correction"
}
```

## RBAC Configuration

### 1. Enable Managed Identity on Content Safety

```bash
az cognitiveservices account identity assign \
  -g <resource-group> \
  -n <content-safety-resource>

CS_MI_ID=$(az cognitiveservices account show \
  -g <resource-group> \
  -n <content-safety-resource> \
  --query identity.principalId -o tsv)
```

### 2. Grant Access to Azure OpenAI

```bash
AOAI_ID=$(az cognitiveservices account show \
  -g <resource-group> \
  -n <openai-resource> \
  --query id -o tsv)

az role assignment create \
  --assignee-object-id "$CS_MI_ID" \
  --assignee-principal-type ServicePrincipal \
  --role "Cognitive Services OpenAI User" \
  --scope "$AOAI_ID"
```

### 3. Wait for Propagation

Role assignments can take up to 15 minutes to propagate.

## Testing

### Quick Test Script

```bash
# Basic mode test
curl -X POST "$AZURE_CS_ENDPOINT/contentsafety/text:detectGroundedness?api-version=2024-09-15-preview" \
  -H "Ocp-Apim-Subscription-Key: $AZURE_CS_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "Generic",
    "task": "Summarization",
    "text": "The cat weighs 50 pounds.",
    "groundingSources": ["The cat weighs 10 pounds."]
  }'

# Reasoning mode test (requires OpenAI config)
curl -X POST "$AZURE_CS_ENDPOINT/contentsafety/text:detectGroundedness?api-version=2024-09-15-preview" \
  -H "Ocp-Apim-Subscription-Key: $AZURE_CS_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "Generic",
    "task": "Summarization",
    "text": "The patient is John.",
    "groundingSources": ["The patient is Jane."],
    "reasoning": true,
    "llmResource": {
      "resourceType": "AzureOpenAI",
      "azureOpenAIEndpoint": "'$AZURE_OPENAI_ENDPOINT'",
      "azureOpenAIDeploymentName": "'$AZURE_OPENAI_DEPLOYMENT'"
    }
  }'
```

### Diagnostic Tool

```bash
# Run comprehensive diagnostics
python diagnose_azure_auth.py

# Run unit tests
pytest tests/test_groundedness_enhanced.py -v
```

## API Integration

The groundedness check is integrated at the evaluation level:

```python
from services.evaluation.context_evaluator import evaluate_context_package_async

# Automatically includes groundedness check when configured
report = await evaluate_context_package_async(
    package,
    answer="AI generated answer",
    retrieved_documents=["source1", "source2"],
    check_content_safety=True  # Enable Content Safety checks
)

# Access results
if report.content_safety_groundedness is not None:
    print(f"Groundedness: {report.content_safety_groundedness:.2%}")
    if report.content_safety_details:
        print(f"Mode: {report.content_safety_details['mode']}")
```

## Response Format in API

```json
{
  "metadata": {
    "context_evaluation": {
      "precision": 0.8,
      "utilization": 0.7,
      "groundedness": 0.9,
      "content_safety_groundedness": 0.85,
      "content_safety_details": {
        "ungrounded_detected": false,
        "ungrounded_percentage": 0.15,
        "ungrounded_details": [],
        "mode": "basic"
      }
    }
  }
}
```

## Error Handling

The implementation follows a fail-safe pattern:

1. **Never blocks answer pipeline** - Returns `None` on any error
2. **Automatic retry** - Uses exponential backoff for transient failures
3. **Mode downgrade** - Falls back to basic mode if advanced features unavailable
4. **Structured logging** - All errors logged with context

### Error Codes

| Status | Meaning | Action |
|--------|---------|--------|
| 401 | Invalid API key | Check `AZURE_CS_KEY` |
| 403 | RBAC missing | Check role assignments |
| 429 | Rate limit | Implement backoff |
| 404 | Wrong endpoint | Verify URL format |

## Performance Considerations

### Limits

- Text: Maximum 7,500 characters
- Sources: Maximum 20 sources, 10,000 chars each
- Rate: 50 requests/minute (groundedness)
- Timeout: Configurable, default 12 seconds

### Optimization Tips

1. **Cache results** - Groundedness is deterministic for same inputs
2. **Batch sources** - Combine related sources to reduce count
3. **Async processing** - Use async evaluation for better performance
4. **Mode selection** - Use basic mode when reasoning not needed

## Security Best Practices

1. **Never log API keys** - Only log key presence/length
2. **Use Managed Identity** - Preferred over API keys in production
3. **Validate inputs** - Truncate to API limits before sending
4. **Handle PII** - Consider redaction before sending to API

## Troubleshooting

### Common Issues

**Groundedness always returns None**
- Check `AZURE_CS_ENABLE_GROUNDEDNESS=1` is set
- Verify `AZURE_CS_ENDPOINT` and `AZURE_CS_KEY` are correct
- Run `python diagnose_azure_auth.py`

**Reasoning/Correction not working**
- Verify Azure OpenAI credentials
- Check RBAC: Content Safety MI needs `Cognitive Services OpenAI User` role
- Wait 15 minutes after role assignment
- Check only one mode is enabled (not both)

**Timeout errors**
- Increase `AZURE_HTTP_TIMEOUT_SECONDS`
- Check network connectivity
- Verify endpoint URL is correct region

**403 Forbidden**
- Check role assignments with:
  ```bash
  az role assignment list --assignee <principal-id> --all -o table
  ```
- Verify network restrictions (firewall rules)

## Migration from Old Variables

If you have existing deployments using old variable names:

```bash
# Old format
CONTENT_SAFETY_ENDPOINT=...
CONTENT_SAFETY_API_KEY=...

# New format
AZURE_CS_ENDPOINT=...
AZURE_CS_KEY=...
AZURE_CS_ENABLE_GROUNDEDNESS=1
```

The diagnostic tool supports both formats during transition.

## References

- [Azure Content Safety Groundedness](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/quickstart-groundedness)
- [Azure OpenAI RBAC](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/role-based-access-control)
- [Content Filter Concepts](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/concepts/content-filter-groundedness)