# Azure OpenAI Integration for Four Hosts Research App

This document explains how to configure and use Azure OpenAI services with the Four Hosts Research Application.

## Prerequisites

1. Azure subscription with access to Azure OpenAI services
2. Azure OpenAI resource deployed
3. GPT-4 and GPT-4o-mini models deployed in your Azure OpenAI resource

## Configuration

To enable Azure OpenAI integration, you need to set the following environment variables in your `.env` file:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_API_VERSION=2024-07-01-preview
AZURE_GPT4_DEPLOYMENT_NAME=gpt-4
AZURE_GPT4_MINI_DEPLOYMENT_NAME=gpt-4o-mini
```

### Environment Variable Details

- `AZURE_OPENAI_ENDPOINT`: The endpoint URL for your Azure OpenAI resource
- `AZURE_OPENAI_API_KEY`: The API key for your Azure OpenAI resource
- `AZURE_OPENAI_API_VERSION`: The API version to use (default: 2024-07-01-preview)
- `AZURE_GPT4_DEPLOYMENT_NAME`: The deployment name for your GPT-4 model
- `AZURE_GPT4_MINI_DEPLOYMENT_NAME`: The deployment name for your GPT-4o-mini model

## Model Selection by Paradigm

The system automatically selects the appropriate model based on the research paradigm:

- **Dolores (Revolutionary)**: Uses GPT-4 for creative and impactful content
- **Teddy (Devotion)**: Uses GPT-4o-mini for balanced and supportive content
- **Bernard (Analytical)**: Uses GPT-4 for detailed analytical content
- **Maeve (Strategic)**: Uses GPT-4o-mini for efficient strategic content

## Fallback Mechanism

If Azure OpenAI is not configured or unavailable, the system will automatically fall back to:

1. OpenAI API (if `OPENAI_API_KEY` is configured)
2. Mock content generation (for development)

## Testing the Integration

To verify that Azure OpenAI is properly configured:

1. Start the backend server:
   ```bash
   cd four-hosts-app/backend
   python main_new.py
   ```

2. Submit a research query through the API:
   ```bash
   curl -X POST http://localhost:8000/research/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the impacts of climate change on coastal communities?", "options": {"enable_real_search": true}}'
   ```

3. Check the logs for Azure OpenAI usage:
    ```
    INFO:     ✓ Generated completion using Azure OpenAI (gpt-4o-mini)
    ```

4. Verify LLM health-check endpoint:
   ```bash
   curl http://localhost:8000/v1/system/llm-ping
   ```
   Expected output:
   ```json
   {
     "status": "ok",
     "llm_response": "ping"
   }
   ```

## Troubleshooting

### Common Issues

1. **Authentication Error**: Verify that your `AZURE_OPENAI_API_KEY` is correct and has the necessary permissions.

2. **Model Not Found**: Ensure that your model deployments (`AZURE_GPT4_DEPLOYMENT_NAME` and `AZURE_GPT4_MINI_DEPLOYMENT_NAME`) exist in your Azure OpenAI resource.

3. **API Version Error**: Check that the `AZURE_OPENAI_API_VERSION` is supported by your Azure OpenAI resource.

### Logging

The system provides detailed logging for LLM interactions. Set `LOG_LEVEL=DEBUG` in your `.env` file for more detailed information.

## Cost Management

Azure OpenAI usage is billed based on token consumption. The system uses different models for different paradigms to optimize costs:

- GPT-4o-mini for most use cases (lower cost)
- GPT-4 for complex analytical tasks (higher cost, better quality)

Monitor your Azure portal for usage and costs.

## Redis & Multi-Worker Support

When `REDIS_URL` is configured the backend will automatically launch **4 Uvicorn workers**
for improved concurrency. Override this with `WORKERS` env var if you need a different count.

```env
# Example .env
REDIS_URL=redis://localhost:6379
WORKERS=8          # optional – overrides automatic worker scaling
```

Ensure Redis is running locally (or via Docker) before starting the backend to avoid fallback
to single-worker mode.
