## SDK Snippets & Templates

### Python (API Key)

```python
import os
from openai import OpenAI

AZURE_OPENAI_ENDPOINT = "https://YOUR-RESOURCE-NAME.openai.azure.com"
DEPLOYMENT_NAME = "YOUR_DEPLOYMENT_NAME"

client = OpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/v1/",
)

response = client.responses.create(
    model=DEPLOYMENT_NAME,
    input="YOUR_PROMPT_HERE",
    temperature=0.2,   # optional
    max_output_tokens=512,
)

print(response.output_text)
```

### Python (Microsoft Entra ID)

```python
import os
from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

AZURE_OPENAI_ENDPOINT = "https://YOUR-RESOURCE-NAME.openai.azure.com"
DEPLOYMENT_NAME = "YOUR_DEPLOYMENT_NAME"

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default",
)

client = OpenAI(
    api_key=token_provider,
    base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/v1/",
)

response = client.responses.create(
    model=DEPLOYMENT_NAME,
    input="YOUR_PROMPT_HERE",
    reasoning={"effort": "medium"},  # optional
)

print(response.output_text)
```

### REST (cURL)

```bash
curl -X POST "https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/responses?api-version=2025-05-01-preview" \
  -H "Content-Type: application/json" \
  -H "api-key: $AZURE_OPENAI_API_KEY" \
  -d '{
        "model": "YOUR_DEPLOYMENT_NAME",
        "input": "YOUR_PROMPT_HERE",
        "max_output_tokens": 400
      }'
```

### Background Task Template

```python
from time import sleep
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_AUTH_MECHANISM",
    base_url="https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/",
)

job = client.responses.create(
    model="BACKGROUND_DEPLOYMENT",
    input="Describe the quarterly performance of our retail business.",
    background=True,
    store=True,
)

while job.status in {"queued", "in_progress"}:
    sleep(3)
    job = client.responses.retrieve(job.id)

if job.status == "completed":
    print(job.output_text)
else:
    raise RuntimeError(f"Job ended with status {job.status}")
```

---

## Tool Schema Templates

### Function Tool (JSON Schema)

```json
{
  "type": "function",
  "name": "FUNCTION_NAME",
  "description": "One-sentence purpose statement.",
  "parameters": {
    "type": "object",
    "properties": {
      "param_one": {
        "type": "string",
        "description": "Explain the expected value and format."
      },
      "param_two": {
        "type": "integer",
        "description": "Optional integer parameter.",
        "minimum": 0
      }
    },
    "required": ["param_one"]
  }
}
```

### Code Interpreter Tool

```json
{
  "type": "code_interpreter",
  "container": {
    "type": "auto",          // or "persistent" when available
    "files": ["file-id-1"],  // optional preloaded files
    "environment": "python3.11" // optional when multiple runtimes exist
  }
}
```

### MCP Tool Entry

```json
{
  "type": "mcp",
  "server_label": "github-docs",
  "server_url": "https://YOUR-MCP-ENDPOINT",
  "require_approval": "auto",     // auto | never | always
  "headers": {
    "Authorization": "Bearer YOUR_TOKEN"
  }
}
```

### Image Generation Tool

```json
{
  "type": "image_generation",
  "parameters": {
    "size": "1024x1024",
    "background": "transparent"
  }
}
```

---

## Quick-Reference Tables

### Deployment Support Matrix |

| Deployment | Reasoning Effort | Tool Calling | Vision Inputs | Background Mode |
| --- | --- | --- | --- | --- |
| `gpt-4.1` | No | Yes (parallel) | Yes | No |
| `gpt-4.1-mini` | No | Yes (parallel) | Yes | No |
| `gpt-4.1-nano` | No | Yes (basic) | Yes | No |
| `gpt-4o` (2024+) | No | Yes (parallel) | Yes | No |
| `gpt-4o-mini` | No | Yes (parallel) | Yes | No |
| `o3` | `low`/`medium`/`high` | Yes (basic) | Yes | Yes |
| `o4-mini` | `low`/`medium`/`high` | Yes (basic) | Yes | Planned |
| `o3-pro` | `low`/`medium`/`high` | Yes (basic) | No | Yes |
| `gpt-5` family | `minimal`/`low`/`medium`/`high` | Yes (parallel) | Yes | Yes |
| `o1` / `o1-mini` | No reasoning controls | Yes (basic) | Limited | Yes |

> *Parallel* = multiple tool calls per response.  
> *Basic* = single or sequential tool usage within one turn.

### Feature-to-Tool Mapping |

| Capability | Required Tool Entry | Notes |
| --- | --- | --- |
| Call REST API / internal service | `function` | Validate JSON arguments before execution. |
| Execute Python analysis | `code_interpreter` | Incurs container billing; supports file uploads. |
| Fetch remote knowledge via MCP | `mcp` | Approval flow defaults to manual unless overridden. |
| Generate images (preview) | `image_generation` | No streaming; place deployment name in `x-ms-oai-image-generation-deployment`. |
| Run browser automation | `computer-use` (preview model) | Configure per Playwright guide; background recommended. |
| Enforce constrained grammar | `custom` → `lark_tool` | Works best with GPT-5 reasoning series. |

### Prompt Engineering Levers |

| Goal | Suggested Approach | Applicable Models |
| --- | --- | --- |
| Force Markdown output | Developer message: “Formatting re-enabled—use Markdown code fences.” | `o3-mini`, `o1`, other reasoning models |
| Encourage clarification before tool call | System prompt: “Ask for missing fields before invoking tools.” | All |
| Restrict tool usage | System prompt: “Use only provided tools; never fabricate external actions.” | All |
| Optimize cost/time | Set `reasoning.effort` to `minimal`/`low` (where supported). | GPT-5 series |
| Retry invalid JSON | Wrap function schema with explicit examples and instruct “return valid JSON or ask for clarification.” | All |

---

### How to Use These Assets

- **Paste the snippets** into template notes or QuickAdd scripts for rapid project scaffolding.
- **Embed the tables** near the top of [[Azure OpenAI Unified Responses API]] as visual cheat sheets.
- **Adapt the tool schemas** by swapping placeholders with production metadata before deployment.

If you’d like Dataview queries, callout-based quick actions, or separate snippet notes per SDK, let me know!
