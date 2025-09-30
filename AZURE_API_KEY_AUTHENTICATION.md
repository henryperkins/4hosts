# Azure API Key Authentication Configuration

**Date:** 2025-09-30
**Authentication Method:** API Keys (Primary)

---

## Current Configuration

### Azure Services Using API Keys

```bash
# Azure OpenAI
AZURE_OPENAI_API_KEY=<your-azure-openai-api-key>
AZURE_OPENAI_ENDPOINT=https://oaisubresource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=o3

# Azure AI Search
AZURE_AI_SEARCH_API_KEY=<your-azure-search-api-key>
AZURE_AI_SEARCH_ENDPOINT=https://oaisearch2.search.windows.net
AZURE_AI_SEARCH_API_VERSION=2025-03-01-preview

# Azure AI Foundry
AZURE_AI_PROJECT_ENDPOINT=https://oaisubresource.services.ai.azure.com/api/projects/oaisubresource
AZURE_AI_PROJECT_NAME=oaisubresource
```

### Identity Information

```bash
AZURE_TENANT_ID=41f88ecb-ca63-404d-97dd-ab0a169fd138
AZURE_SUBSCRIPTION_ID=fe000daf-8df4-49e4-99d8-6e789060f760
AZURE_RESOURCE_GROUP_NAME=rg-hperkin4-8776
```

---

## Authentication Strategy

**Primary Method:** API Keys
**Reason:** Simplicity and direct access control

### Services Configuration

1. **Azure OpenAI** → API Key authentication
2. **Azure AI Search** → API Key authentication
3. **Azure AI Foundry** → Endpoint-based (uses Azure OpenAI key)

---

## API Key Management

### Current Setup

- **OpenAI API Key:** Admin key for full access
- **Search API Key:** Admin key for index management
- **Storage:** Environment variables in `.env` file

### Security Measures

1. **Not in version control:** `.env` is gitignored
2. **File permissions:** Restricted to application user
3. **Rotation:** Keys can be regenerated in Azure Portal
4. **Monitoring:** Key usage tracked via Azure metrics

---

## Usage in Code

### Azure OpenAI

```python
# services/llm_client.py
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
```

### Azure AI Search

```python
# When using Azure AI Search
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

search_client = SearchClient(
    endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
    index_name="your-index",
    credential=AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_API_KEY"))
)
```

### Azure AI Foundry

```python
# Uses Azure OpenAI endpoint and key
# services/mcp/azure_ai_foundry_mcp_integration.py
config.ai_project_endpoint  # Endpoint-based configuration
```

---

## DefaultAzureCredential Support

While API keys are the primary method, the system also supports DefaultAzureCredential as a **fallback option**:

```python
# services/mcp/azure_ai_foundry_auth.py
# This provides Azure CLI / Managed Identity support
# But API keys take precedence when configured
```

**Priority:**
1. ✅ API Keys (if configured) ← **Primary**
2. Azure CLI authentication (fallback)
3. Managed Identity (fallback)

---

## Key Rotation Process

### When to Rotate

- Regular schedule (e.g., every 90 days)
- After team member changes
- If key exposure suspected
- Compliance requirements

### How to Rotate

1. **Azure Portal:**
   - Navigate to your Azure OpenAI resource
   - Go to "Keys and Endpoint"
   - Click "Regenerate Key 2"
   - Update `.env` with new key
   - Restart application
   - Regenerate Key 1 after verification

2. **Zero Downtime:**
   - Azure provides two keys (Key 1 and Key 2)
   - Rotate one at a time
   - No service interruption

---

## Environment Variables Reference

### Required for API Key Authentication

```bash
# Azure OpenAI (Required)
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key-here>
AZURE_OPENAI_DEPLOYMENT=o3
AZURE_OPENAI_API_VERSION=preview

# Azure AI Search (Required if using search)
AZURE_AI_SEARCH_ENDPOINT=https://<resource>.search.windows.net
AZURE_AI_SEARCH_API_KEY=<your-key-here>
AZURE_AI_SEARCH_API_VERSION=2025-03-01-preview

# Azure AI Foundry (Required for MCP)
AZURE_AI_PROJECT_ENDPOINT=https://<project>.services.ai.azure.com/api/projects/<project>
AZURE_AI_PROJECT_NAME=<project-name>

# Identity (Optional but recommended)
AZURE_TENANT_ID=<tenant-id>
AZURE_SUBSCRIPTION_ID=<subscription-id>
AZURE_RESOURCE_GROUP_NAME=<resource-group>
```

---

## Benefits of API Key Authentication

### Advantages

1. **Simple Setup:** Just copy/paste keys from Azure Portal
2. **No Azure CLI Required:** Works without `az login`
3. **Explicit Control:** Know exactly which key is being used
4. **Easy Debugging:** Clear error messages about key issues
5. **No Token Refresh:** Keys don't expire like tokens do

### Use Cases

- ✅ Development environments
- ✅ CI/CD pipelines
- ✅ Quick prototyping
- ✅ Simple deployments
- ✅ Services without identity support

---

## Security Best Practices

### Do's ✅

- ✅ Store keys in `.env` file (gitignored)
- ✅ Use environment variables
- ✅ Rotate keys regularly
- ✅ Monitor key usage in Azure Portal
- ✅ Use separate keys for dev/staging/prod
- ✅ Restrict file permissions on `.env`

### Don'ts ❌

- ❌ Don't commit keys to git
- ❌ Don't hardcode keys in source code
- ❌ Don't share keys in chat/email
- ❌ Don't use production keys in development
- ❌ Don't log API keys
- ❌ Don't expose keys in error messages

---

## Verification

### Check Current Configuration

```bash
# Check if API keys are configured
python -c "
import os
from dotenv import load_dotenv
load_dotenv()

print('Azure OpenAI Key:', '✓ SET' if os.getenv('AZURE_OPENAI_API_KEY') else '✗ NOT SET')
print('Azure Search Key:', '✓ SET' if os.getenv('AZURE_AI_SEARCH_API_KEY') else '✗ NOT SET')
print('Azure Project Endpoint:', '✓ SET' if os.getenv('AZURE_AI_PROJECT_ENDPOINT') else '✗ NOT SET')
"
```

### Test Authentication

```python
from openai import AzureOpenAI
import os

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

# Test with a simple call
response = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    messages=[{"role": "user", "content": "test"}],
    max_tokens=5
)
print("✓ Authentication successful")
```

---

## Troubleshooting

### Common Issues

**Error: "Invalid API key"**
- Check key is copied correctly (no extra spaces)
- Verify key hasn't expired or been regenerated
- Confirm endpoint matches the key's resource

**Error: "Access denied"**
- Verify the key has proper permissions
- Check if key is admin vs query key
- Confirm resource isn't paused or deleted

**Error: "Rate limit exceeded"**
- Keys have usage quotas
- Check Azure Portal for quota status
- Consider upgrading tier or requesting increase

---

## Migration to Managed Identity (Future)

If you later want to migrate to Managed Identity:

1. Enable Managed Identity in Azure
2. Assign RBAC roles to the identity
3. Remove API keys from `.env`
4. System will automatically fall back to DefaultAzureCredential

**For now:** API Keys are the configured authentication method. ✅

---

## Summary

- **Authentication Method:** API Keys
- **Configuration:** Complete and working
- **Security:** Keys stored in `.env` (gitignored)
- **Rotation:** Manual via Azure Portal
- **Fallback:** DefaultAzureCredential available if keys removed

**Status:** ✅ API Key authentication is properly configured and operational.