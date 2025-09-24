# Azure RBAC Setup for Content Safety with Azure OpenAI

This guide follows the official Azure AI Foundry documentation for setting up role-based access control between Content Safety and Azure OpenAI services.

## Overview

When Content Safety needs to use Azure OpenAI for reasoning or correction features, it requires proper RBAC configuration. This can be done using either API keys or Managed Identity (recommended).

## Prerequisites

1. Azure Content Safety resource (already created: `content-safety-hperkin4`)
2. Azure OpenAI resource with a deployed model (e.g., `gpt-4o`)
3. Appropriate permissions to assign roles

## Role Requirements

### For Direct API Usage

| Scenario | Required Role | Scope |
|----------|--------------|-------|
| Inference only (chat completions) | `Cognitive Services OpenAI User` | Azure OpenAI resource |
| Manage + use deployments | `Cognitive Services OpenAI Contributor` | Azure OpenAI resource |
| Full control | `Cognitive Services Contributor` | Azure OpenAI resource |

### For Content Safety to Access Azure OpenAI

Content Safety needs to act as a client to Azure OpenAI. The Content Safety's Managed Identity needs:
- **Role**: `Cognitive Services OpenAI User`
- **Scope**: Azure OpenAI resource
- **Purpose**: Allows Content Safety to call OpenAI for reasoning/correction

## Step-by-Step Setup

### 1. Enable Managed Identity on Content Safety

```bash
# Check if Managed Identity is enabled
az cognitiveservices account show \
  --name content-safety-hperkin4 \
  --resource-group <your-rg> \
  --query identity

# Enable system-assigned managed identity if not already enabled
az cognitiveservices account identity assign \
  --name content-safety-hperkin4 \
  --resource-group <your-rg>
```

### 2. Get the Managed Identity Principal ID

```bash
# Get the principal ID of Content Safety's managed identity
CS_PRINCIPAL_ID=$(az cognitiveservices account show \
  --name content-safety-hperkin4 \
  --resource-group <your-rg> \
  --query identity.principalId -o tsv)

echo "Content Safety MI Principal ID: $CS_PRINCIPAL_ID"
```

### 3. Assign Role to Content Safety MI

```bash
# Get Azure OpenAI resource ID
OPENAI_RESOURCE_ID=$(az cognitiveservices account show \
  --name <your-openai-resource> \
  --resource-group <your-rg> \
  --query id -o tsv)

# Assign Cognitive Services OpenAI User role
az role assignment create \
  --assignee-object-id "$CS_PRINCIPAL_ID" \
  --assignee-principal-type ServicePrincipal \
  --role "Cognitive Services OpenAI User" \
  --scope "$OPENAI_RESOURCE_ID"
```

### 4. Verify Role Assignment

```bash
# List role assignments for the OpenAI resource
az role assignment list \
  --scope "$OPENAI_RESOURCE_ID" \
  --query "[?principalId=='$CS_PRINCIPAL_ID'].{role:roleDefinitionName,principal:principalName}" \
  -o table
```

### 5. Wait for Propagation

Role assignments can take up to 15 minutes to propagate. You can verify it's working by:

```bash
# Test Content Safety with reasoning enabled
curl -X POST "https://4hosts.cognitiveservices.azure.com/contentsafety/text:detectGroundedness?api-version=2024-09-15-preview" \
  -H "Ocp-Apim-Subscription-Key: <your-content-safety-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Test text",
    "groundingSources": ["Source text"],
    "task": "Summarization",
    "reasoning": true,
    "llmResource": {
      "resourceType": "AzureOpenAI",
      "azureOpenAIEndpoint": "https://<your-resource>.openai.azure.com/",
      "azureOpenAIDeploymentName": "gpt-4o"
    }
  }'
```

## Application Configuration

### Using API Keys (Development)

```bash
# .env file
CONTENT_SAFETY_ENDPOINT=https://4hosts.cognitiveservices.azure.com/
CONTENT_SAFETY_API_KEY=<your-key>

# For reasoning/correction features
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

### Using Managed Identity (Production)

When deployed to Azure (App Service, Container Apps, etc.):

1. Enable system-assigned managed identity on your compute resource
2. Assign roles to your app's managed identity:
   - `Cognitive Services User` on Content Safety resource
   - `Cognitive Services OpenAI User` on Azure OpenAI resource (if calling directly)

## Troubleshooting

### Common Issues and Solutions

1. **401 Unauthorized**
   - Verify API key is correct
   - Check key hasn't been regenerated
   - Ensure using correct header (`api-key` for OpenAI, `Ocp-Apim-Subscription-Key` for Content Safety)

2. **403 Forbidden**
   - Check role assignments are complete
   - Wait 15 minutes for propagation
   - Verify no Deny assignments blocking access
   - Check network restrictions (firewall rules)

3. **404 Not Found**
   - Verify deployment name matches exactly
   - Check endpoint URL format
   - Ensure resource exists in specified region

### Diagnostic Commands

```bash
# Check Content Safety MI status
az cognitiveservices account show \
  --name content-safety-hperkin4 \
  --resource-group <your-rg> \
  --query "{name:name, endpoint:properties.endpoint, identity:identity}"

# List all role assignments for a principal
az role assignment list \
  --assignee "$CS_PRINCIPAL_ID" \
  --all \
  -o table

# Check activity logs for authorization failures
az monitor activity-log list \
  --correlation-id <correlation-id-from-error> \
  --query "[].{operation:operationName.value,status:status.value,code:properties.statusCode}" \
  -o table
```

## Security Best Practices

1. **Use Managed Identity in Production**
   - Eliminates key management overhead
   - Automatic credential rotation
   - Better audit trail

2. **Principle of Least Privilege**
   - Only assign `Cognitive Services OpenAI User` (not Contributor/Owner)
   - Scope assignments to specific resources, not subscription

3. **Network Security**
   - Use private endpoints when possible
   - Configure firewall rules to allow only necessary IPs
   - Enable diagnostic logging for audit trail

4. **Key Management (if using API keys)**
   - Store in Azure Key Vault
   - Rotate regularly
   - Never commit to source control

## Validation Script

Run the diagnostic script to verify setup:

```bash
python diagnose_azure_auth.py
```

Expected output when properly configured:
- ✅ Azure OpenAI endpoint format
- ✅ Content Safety endpoint format
- ✅ Azure OpenAI API connection
- ✅ Content Safety API connection
- ✅ Content Safety with reasoning

## References

- [Azure AI Foundry RBAC Guide](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/role-based-access-control)
- [Content Safety Documentation](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/)
- [Azure OpenAI Service Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Managed Identities for Azure Resources](https://learn.microsoft.com/en-us/azure/active-directory/managed-identities-azure-resources/)