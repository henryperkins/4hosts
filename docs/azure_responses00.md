# Azure OpenAI Unified Responses API

- **Published**: 2025-09-19  
- **Estimated read time**: 21 minutes

The Azure OpenAI **Responses API** unifies the former Chat Completions and Assistants APIs into a single, stateful workflow. It unlocks advanced features such as reasoning models, code interpreter, MCP tooling, and preview image generation. This guide consolidates official documentation into an actionable reference (region/matrix data intentionally omitted).

---

## At a Glance

### Key Highlights
- Stateful sessions with `previous_response_id` for automatic context.
- Unified tooling surface for function calling, code interpreter, MCP, and image generation.
- Expanded model support (GPT-5 series, O-series, GPT-4.1 family, etc.).
- Background processing for long-running tasks.

### Current Gaps
> [!NOTE]
> - Web search tool is unavailable.  
> - Image generation lacks multi-turn editing & streaming.  
> - File-uploaded images cannot be referenced as inputs yet.

> [!WARNING]
> - PDF uploads require `purpose="assistants"` (no `user_data` support yet).  
> - Streaming + background mode may exhibit latency (fix pending).

---

## Prerequisites

1. **API Version**: Azure OpenAI **v1** is required (see [API version lifecycle](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle#api-evolution)).
2. **SDK Upgrade**:

```bash
pip install --upgrade openai
```

3. **Security**: Store credentials in secure vaults (e.g., Azure Key Vault). Review [Azure AI authentication](https://learn.microsoft.com/en-us/azure/ai-services/authentication).

---

## Reference Documentation

- [Responses API Reference](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/reference-preview-latest?#create-response)
- Companion notes: [[Azure OpenAI Reasoning Models]], [[How to Use Function Calling with Azure OpenAI (Azure AI Foundry Models)]]

---

## Core Workflow

### Generate a Text Response (Python SDK)

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url="https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/",
)

response = client.responses.create(
    model="gpt-4.1-nano",  # replace with your deployment name
    input="This is a test.",
)

print(response.model_dump_json(indent=2))
```

Equivalent REST, Entra ID, and alternate language samples are available in the official quickstart.

---

## Managing Responses

### Retrieve

```python
response = client.responses.retrieve("resp_67cb61fa3a448190bcf2c42d96f0d1a8")
```

### Delete

```python
response = client.responses.delete("resp_67cb61fa3a448190bcf2c42d96f0d1a8")
print(response)
```

> [!NOTE]
> Responses are retained for **30 days** by default.

### Chaining (Automatic Context)

```python
response = client.responses.create(
    model="gpt-4o",
    input="Define and explain the concept of catastrophic forgetting?"
)

second_response = client.responses.create(
    model="gpt-4o",
    previous_response_id=response.id,
    input=[
        {
            "role": "user",
            "content": "Explain this at a level that could be understood by a college freshman"
        }
    ]
)

print(second_response.model_dump_json(indent=2))
```

### Manual Chaining

```python
inputs = [
    {
        "type": "message",
        "role": "user",
        "content": "Define and explain the concept of catastrophic forgetting?"
    }
]

response = client.responses.create(model="gpt-4o", input=inputs)
inputs += response.output
inputs.append({
    "role": "user",
    "type": "message",
    "content": "Explain this at a level that could be understood by a college freshman"
})

second_response = client.responses.create(model="gpt-4o", input=inputs)
print(second_response.model_dump_json(indent=2))
```

---

## Streaming & Background Processing

### Live Streaming

```python
response = client.responses.create(
    input="This is a test",
    model="o4-mini",
    stream=True
)

for event in response:
    if event.type == "response.output_text.delta":
        print(event.delta, end="")
```

### Background Mode

```python
response = client.responses.create(
    model="o3",
    input="Write me a very long story",
    background=True
)

print(response.status)
```

#### Polling Status

```python
from time import sleep

while response.status in {"queued", "in_progress"}:
    print(f"Current status: {response.status}")
    sleep(2)
    response = client.responses.retrieve(response.id)

print(f"Final status: {response.status}\nOutput:\n{response.output_text}")
```

#### Cancel Background Task

```python
cancelled = client.responses.cancel("resp_1234567890")
print(cancelled.status)
```

#### Background Streaming

```python
stream = client.responses.create(
    model="o3",
    input="Write me a very long story",
    background=True,
    stream=True,
)

cursor = None
for event in stream:
    print(event)
    cursor = event["sequence_number"]
```

Resume a stream from a specific sequence:

```bash
curl https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/responses/resp_1234567890?stream=true&starting_after=42 \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AZURE_OPENAI_AUTH_TOKEN"
```

> [!NOTE]
> Background mode requires `store=true`, introduces higher time-to-first-token latency, and can be cancelled by terminating the connection for synchronous calls.

---

## Tooling & Advanced Capabilities

### Function Calling (Responses API)

```python
response = client.responses.create(
    model="gpt-4o",
    tools=[
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
    ],
    input=[{
        "role": "user",
        "content": "What's the weather in San Francisco?"
    }],
)

tool_followup = []
for output in response.output:
    if output.type == "function_call":
        if output.name == "get_weather":
            tool_followup.append(
                {
                    "type": "function_call_output",
                    "call_id": output.call_id,
                    "output": '{"temperature": "70 degrees"}',
                }
            )
        else:
            raise ValueError(f"Unknown function call: {output.name}")

second_response = client.responses.create(
    model="gpt-4o",
    previous_response_id=response.id,
    input=tool_followup,
)

print(second_response.model_dump_json(indent=2))
```

See [[How to Use Function Calling with Azure OpenAI (Azure AI Foundry Models)]] for comprehensive tool schemas, parallel invocation strategies, and responsible-use guidance.

---

### Code Interpreter

The Python-based code interpreter runs within a managed container for analytics, visualization, and iterative coding.

#### REST Sample

```bash
curl https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/responses?api-version=preview \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AZURE_OPENAI_AUTH_TOKEN" \
  -d '{
        "model": "gpt-4.1",
        "tools": [
            { "type": "code_interpreter", "container": {"type": "auto"} }
        ],
        "instructions": "You are a personal math tutor. When asked a math question, write and run code using the python tool to answer the question.",
        "input": "I need to solve the equation 3x + 11 = 14. Can you help me?"
    }'
```

#### Python Sample

```python
instructions = (
    "You are a personal math tutor. When asked a math question, "
    "write and run code using the python tool to answer the question."
)

response = client.responses.create(
    model="gpt-4.1",
    tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
    instructions=instructions,
    input="I need to solve the equation 3x + 11 = 14. Can you help me?",
)

print(response.output)
```

#### Container Controls

```json
"tools": [
  {
    "type": "code_interpreter",
    "container": { "type": "auto", "files": ["file-1", "file-2"] }
  }
]
```

> [!IMPORTANT]
> - Additional billing applies per active container.  
> - Containers last 1 hour (30-minute idle timeout).  
> - Unused containers expire after 20 minutes.

#### File Handling
- Request files auto-upload into the container.
- Generated artifacts (CSV, plots, etc.) can be cited in outputs.

#### Supported File Types

| File format | MIME type |
| --- | --- |
| `.c` | text/x-c |
| `.cs` | text/x-csharp |
| `.cpp` | text/x-c++ |
| `.csv` | text/csv |
| `.doc` | application/msword |
| `.docx` | application/vnd.openxmlformats-officedocument.wordprocessingml.document |
| `.html` | text/html |
| `.java` | text/x-java |
| `.json` | application/json |
| `.md` | text/markdown |
| `.pdf` | application/pdf |
| `.php` | text/x-php |
| `.pptx` | application/vnd.openxmlformats-officedocument.presentationml.presentation |
| `.py` | text/x-python |
| `.rb` | text/x-ruby |
| `.tex` | text/x-tex |
| `.txt` | text/plain |
| `.css` | text/css |
| `.js` | text/javascript |
| `.sh` | application/x-sh |
| `.ts` | application/typescript |
| `.jpeg` | image/jpeg |
| `.jpg` | image/jpeg |
| `.gif` | image/gif |
| `.pkl` | application/octet-stream |
| `.png` | image/png |
| `.tar` | application/x-tar |
| `.xlsx` | application/vnd.openxmlformats-officedocument.spreadsheetml.sheet |
| `.xml` | application/xml or text/xml |
| `.zip` | application/zip |

---

### Image Generation (Preview)

```python
from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import base64

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default",
)

client = OpenAI(
    base_url="https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/",
    api_key=token_provider,
    default_headers={
        "x-ms-oai-image-generation-deployment": "gpt-image-1",
        "api_version": "preview"
    }
)

response = client.responses.create(
    model="o3",
    input="Generate an image of gray tabby cat hugging an otter with an orange scarf",
    tools=[{"type": "image_generation"}],
)

image_data = [
    output.result
    for output in response.output
    if output.type == "image_generation_call"
]

if image_data:
    with open("otter.png", "wb") as f:
        f.write(base64.b64decode(image_data[0]))
```

> [!NOTE]
> - No partial-image streaming via Responses API (use Images API if needed).  
> - Specify `image_generation` in `tools`.

---

### Remote MCP Servers

```bash
curl https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AZURE_OPENAI_AUTH_TOKEN" \
  -d '{
    "model": "gpt-4.1",
    "tools": [
      {
        "type": "mcp",
        "server_label": "github",
        "server_url": "https://contoso.com/Azure/azure-rest-api-specs",
        "require_approval": "never"
      }
    ],
    "input": "What is this repo in 100 words?"
  }'
```

```python
response = client.responses.create(
    model="gpt-4.1",
    tools=[
        {
            "type": "mcp",
            "server_label": "github",
            "server_url": "https://contoso.com/Azure/azure-rest-api-specs",
            "require_approval": "never"
        },
    ],
    input="What transport protocols are supported in the 2025-03-26 version of the MCP spec?",
)

print(response.output_text)
```

> [!NOTE]
> - MCP tooling is exclusive to the Responses API.  
> - Token billing only; importing/calling tools adds no extra fees.

#### Approval Workflow

```json
{
  "id": "mcpr_682bd9cd428c8198b170dc6b549d66fc016e86a03f4cc828",
  "type": "mcp_approval_request",
  "arguments": {},
  "name": "fetch_azure_rest_api_docs",
  "server_label": "github"
}
```

Approve via follow-up:

```python
approval_response = client.responses.create(
    model="gpt-4.1",
    tools=[{
        "type": "mcp",
        "server_label": "github",
        "server_url": "https://contoso.com/Azure/azure-rest-api-specs",
        "require_approval": "never"
    }],
    previous_response_id="resp_682f750c5f9c8198aee5b480980b5cf60351aee697a7cd77",
    input=[{
        "type": "mcp_approval_response",
        "approve": True,
        "approval_request_id": "mcpr_682bd9cd428c8198b170dc6b549d66fc016e86a03f4cc828"
    }],
)
```

#### Authentication Headers

```python
response = client.responses.create(
    model="gpt-4.1",
    input="What is this repo in 100 words?",
    tools=[
        {
            "type": "mcp",
            "server_label": "github",
            "server_url": "https://gitmcp.io/Azure/azure-rest-api-specs",
            "headers": {
                "Authorization": "Bearer $YOUR_API_KEY"
            }
        }
    ]
)

print(response.output_text)
```

---

### Computer Use

The `computer-use-preview` model leverages Playwright-based automation. Refer to the dedicated [Computer Use model documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/computer-use#playwright-integration) for setup and limitations.

---

## Handling Inputs & Files

### List Input Items

```python
response = client.responses.input_items.list("resp_67d856fcfba0819081fd3cffee2aa1c0")
print(response.model_dump_json(indent=2))
```

### Image Inputs

#### Remote URL

```python
response = client.responses.create(
    model="gpt-4o",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "what is in this image?"},
                {"type": "input_image", "image_url": "<image_URL>"}
            ]
        }
    ]
)

print(response)
```

#### Base64

```python
import base64

def encode_image(path: str) -> str:
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image("path_to_your_image.jpg")

response = client.responses.create(
    model="gpt-4o",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "what is in this image?"},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            ]
        }
    ]
)

print(response)
```

### PDF & File Inputs

> [!CAUTION]
> - Large PDFs consume many tokens; ensure content fits within the model context.  
> - Use `purpose="assistants"` until `user_data` support rolls out.

#### Base64 Ingestion

```python
import base64

with open("PDF-FILE-NAME.pdf", "rb") as f:
    data = f.read()

base64_string = base64.b64encode(data).decode("utf-8")

response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "filename": "PDF-FILE-NAME.pdf",
                    "file_data": f"data:application/pdf;base64,{base64_string}",
                },
                {"type": "input_text", "text": "Summarize this PDF"},
            ],
        },
    ]
)

print(response.output_text)
```

#### File Upload by ID

```python
file = client.files.create(
    file=open("nucleus_sampling.pdf", "rb"),
    purpose="assistants"
)

response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_file", "file_id": file.id},
                {"type": "input_text", "text": "Summarize this PDF"},
            ],
        },
    ]
)

print(response.output_text)
```

---

## Reasoning Enhancements

### Encrypted Reasoning Items

Use encrypted reasoning tokens when running stateless (`store=false`) to preserve context securely:

```bash
curl https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AZURE_OPENAI_AUTH_TOKEN" \
  -d '{
        "model": "o4-mini",
        "reasoning": {"effort": "medium"},
        "input": "What is the weather like today?",
        "tools": ["<YOUR_FUNCTION_GOES_HERE>"],
        "include": ["reasoning.encrypted_content"]
      }'
```

### Reasoning Models Overview

See [[Azure OpenAI Reasoning Models]] for deeper coverage. Highlights:

| Capability | Notes |
| --- | --- |
| Developer/system roles | Functionally equivalent; choose one. |
| Structured outputs | Supported alongside function/tool calling. |
| Context window | Up to **400k tokens** (272k input / 128k output). |
| Reasoning effort | `minimal` (GPT-5 only), `low`, `medium`, `high`. |
| Streaming | Available across reasoning models. |
| Token controls | Use `max_output_tokens` (Responses API). |

#### New GPT-5 Features

| Feature | Summary |
| --- | --- |
| `reasoning_effort="minimal"` | Exclusive to GPT-5 series except `gpt-5-codex`. |
| `verbosity` | Adjust answer length (`low`, `medium`, `high`). |
| `preamble` | Encourage pre-tool deliberation via `instructions`. |
| `allowed tools` | Restrict tool selection in `tool_choice`. |
| `custom tool type` | Enable raw text outputs instead of JSON. |
| `lark_tool` | Enforce output structure via Lark grammar. |

> GPT-5 reasoning jobs benefit from background mode for long tasks. Note that `o3-pro` cannot generate images.

#### Unsupported Parameters

`temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `logprobs`, `top_logprobs`, `logit_bias`, `max_tokens` are ignored by reasoning deployments.

---

## Function Calling Playbook

Summarized guidance (full detail in [[How to Use Function Calling with Azure OpenAI (Azure AI Foundry Models)]]):
- Use the `tools` array with well-described schemas (≤1024 characters per description).
- `tool_choice="auto"` lets the model decide; specify a function to force its use.
- Always validate generated arguments before execution and append tool results with `tool_call_id`.
- For parallel responses, provide one tool message per call.

> [!TIP]
> Developer messages (`role="developer"`) can steer tool usage, request clarification, or restrict unlisted functions.

---

## Prompting Best Practices

- **Enrich definitions**: Include precise parameter descriptions and examples.
- **Clarify expectations**: Use system/developer prompts to enforce tool usage or output formats.
- **Markdown control**: Reasoning models like `o3-mini`/`o1` may avoid Markdown. Prepend a developer instruction such as “Formatting re-enabled—please use Markdown code fences.”

---

## Responsible Use Checklist

- Validate model outputs and tool arguments before executing.
- Use trusted data sources and sanitize content against prompt injection.
- Apply least privilege for downstream systems (read-only creds, scoped APIs).
- Seek explicit confirmation for impactful or irreversible actions.

For broader policy context, review the [Responsible AI overview for Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-foundry/responsible-ai/openai/overview).

---

## Next Steps

- Extract dedicated snippets/templates for SDK usage or tool schemas.
- Build quick-reference tables per deployment or capability.
- Evaluate whether related content should split into focused notes (e.g., reasoning-only playbook, MCP integration guide).

Let me know how you’d like this note adapted—happy to generate task checklists, Dataview tables, or streamline into templates for your vault.
