# Azure Content Safety Integration Setup

This document describes how to set up Azure Content Safety for groundedness detection in the Four Hosts application.

## Prerequisites

1. Azure Content Safety resource
2. Azure OpenAI resource (optional, for reasoning/correction features)
3. Proper RBAC roles configured

## RBAC Configuration for Content Safety with Azure OpenAI

When using the reasoning or correction features that require Azure OpenAI, you need to configure proper role assignments:

### Option 1: Using API Keys (Current Implementation)

The current implementation uses API keys for both services:

```bash
# In your .env file
CONTENT_SAFETY_ENDPOINT=https://4hosts.cognitiveservices.azure.com/
CONTENT_SAFETY_API_KEY=your-content-safety-key

# Optional: For reasoning/correction features
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-openai-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

### Option 2: Using Managed Identity (Recommended for Production)

For production environments, use Managed Identity for enhanced security:

1. **Enable Managed Identity for your application**
   - For Azure App Service/Container Apps: Enable system-assigned managed identity in the Identity blade
   - For VMs: Enable system-assigned managed identity in the VM configuration

2. **Assign roles to the Managed Identity**

   For Content Safety resource:
   ```bash
   # Assign Cognitive Services User role
   az role assignment create \
     --assignee <managed-identity-object-id> \
     --role "Cognitive Services User" \
     --scope /subscriptions/<subscription-id>/resourceGroups/<rg>/providers/Microsoft.CognitiveServices/accounts/<content-safety-resource>
   ```

   For Azure OpenAI resource (if using reasoning/correction):
   ```bash
   # Assign Cognitive Services OpenAI User role
   az role assignment create \
     --assignee <managed-identity-object-id> \
     --role "Cognitive Services OpenAI User" \
     --scope /subscriptions/<subscription-id>/resourceGroups/<rg>/providers/Microsoft.CognitiveServices/accounts/<openai-resource>
   ```

3. **Update the code to use DefaultAzureCredential**

   The enhanced context evaluator would need to be updated to support this:

   ```python
   from azure.identity import DefaultAzureCredential

   # Use DefaultAzureCredential instead of AzureKeyCredential
   credential = DefaultAzureCredential()
   ```

## Available Features

### Basic Groundedness Detection
- Detects if AI-generated text is grounded in source materials
- Returns ungrounded percentage and details
- No additional Azure OpenAI resource required

### Advanced Features (Requires Azure OpenAI)

1. **Reasoning**: Provides detailed explanations for why content is ungrounded
2. **Correction**: Suggests corrections to make the text more grounded

To enable these features:
1. Set up Azure OpenAI resource with GPT-4o deployment
2. Configure RBAC as described above
3. Set environment variables:
   ```bash
   CONTENT_SAFETY_ENABLE_REASONING=1
   CONTENT_SAFETY_ENABLE_CORRECTION=1
   ```

## API Limits

- **Text**: Maximum 7,500 characters
- **Grounding Sources**: Maximum 5 sources, 10,000 characters each
- **Rate Limits**: 50 requests per minute for groundedness detection

## Integration in Four Hosts

The groundedness detection is integrated at the evaluation level:

1. **Context Evaluator** (`services/evaluation/context_evaluator.py`):
   - Provides async groundedness checking via Content Safety API
   - Falls back gracefully if not configured

2. **Research Orchestrator** (`services/research_orchestrator.py`):
   - Uses async evaluation with Content Safety when available
   - Includes results in metadata

3. **Response Format**:
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
           "ungrounded_details": []
         }
       }
     }
   }
   ```

## Testing

Run the integration test:
```bash
export CONTENT_SAFETY_ENDPOINT="https://4hosts.cognitiveservices.azure.com/"
export CONTENT_SAFETY_API_KEY="your-key"
python test_content_safety_integration.py
```

## Troubleshooting

1. **401 Unauthorized**: Check API key is correct
2. **404 Not Found**: Verify endpoint URL includes the correct region
3. **429 Too Many Requests**: Implement rate limiting or retry logic
4. **No results returned**: Check if environment variables are set correctly

## Security Best Practices

1. Never commit API keys to version control
2. Use Azure Key Vault for production deployments
3. Implement proper retry and error handling
4. Use Managed Identity when possible
5. Apply principle of least privilege for role assignments

## References

- [Azure Content Safety Documentation](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/)
- [Groundedness Detection Guide](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/quickstart-groundedness)
- [Azure RBAC Documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/role-based-access-control)
- [Managed Identity Documentation](https://learn.microsoft.com/en-us/azure/active-directory/managed-identities-azure-resources/)