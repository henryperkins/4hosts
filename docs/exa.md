---
title: Exa AI Master Note
source: "Compiled from Obsidian vault notes: [[Python SDK Specification]], [[Exa Research]], [[OpenAI Responses API]], [[OpenAI SDK Compatibility]], [[Exa OpenAPI Specification v12]], [[Exa MCP 1]]"
author:
  - "[[Exa]]"
published: 2025-09-16
created: 2025-09-17
description: A comprehensive master note compiling all information from Exa AI-related clippings in the vault. This includes Python SDK details, Research API, OpenAI integrations, OpenAPI specs, and MCP tools. Duplicates are minimized, and content is organized thematically.
tags:
  - clippings
  - exa
  - api
  - sdk
  - research
  - openai
  - mcp
aliases:
  - Exa Master Documentation
  - Exa AI Compilation
---
# Exa AI Master Note

This master note aggregates and organizes all content from the provided Exa AI-related notes in your Obsidian vault. It draws from:

- [[Exa Python SDK Methods and Types]]: Detailed SDK methods and examples.
- [[Exa Research]]: Async research pipeline, models, pricing, and examples.
- [[Exa OpenAI Responses API Guide]]: Using Exa with OpenAI's Responses API and tool calling.
- [[Exa as a Drop-In OpenAI SDK Replacement]]: Drop-in compatibility with OpenAI SDK for chat and responses.
- [[Exa OpenAPI Specification v12]]: Raw OpenAPI YAML spec for endpoints.
- [[Exa MCP Server Overview]]: MCP server for AI assistants like Claude, including tools and setup.

The content is structured into logical sections for easy navigation. Overlaps (e.g., research methods in SDK and Research API) are merged where possible. Use Obsidian's search or links to cross-reference originals.

## Introduction to Exa AI

Exa is a search engine built for AI, using neural (semantic) search to understand query meanings beyond keywords. It delivers links, content excerpts, highlights, summaries, and structured data, making it ideal for LLMs and RAG (retrieval-augmented generation). Key features include:

- **Search Types**: Neural (embeddings-based), keyword (SERP-like), auto (combines both), fast (streamlined).
- **Categories**: Focused searches on types like company, research paper, news, LinkedIn profile, GitHub, tweet, movie, song, personal site, PDF, financial report.
- **Integrations**: Python SDK, OpenAI compatibility, OpenAPI endpoints, MCP for tools like Claude.
- **Research**: Async, multi-step tasks for complex queries with structured JSON outputs.
- **Pricing**: Usage-based (e.g., $5/1k searches, varies by model/operation). Only charged for successful tasks.

For API keys: Visit [Exa Dashboard](https://dashboard.exa.ai/api-keys).

## Python SDK Specification

From [[Exa Python SDK Methods and Types]].

### Getting Started

Install via:

```bash
pip install exa_py
```

Instantiate client:

```python
from exa_py import Exa
import os

exa = Exa(os.getenv('EXA_API_KEY'))
```

Get your API key from [Exa Dashboard](https://dashboard.exa.ai/login?redirect=/docs?path=/reference/python-sdk-specification).

### search Method

Perform a search and retrieve links.

#### Input Example:

```python
result = exa.search(
  "hottest AI startups",
  num_results=2
)
```

#### Input Parameters:

| Parameter            | Type                                         | Description                                                                                                                                                                                                                                                  | Default  |
| -------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- |
| query                | str                                          | The input query string.                                                                                                                                                                                                                                      | Required |
| num_results          | Optional[int]                                | Number of search results to return.                                                                                                                                                                                                                          | 10       |
| include_domains      | Optional[List[str]]                          | List of domains to include in the search.                                                                                                                                                                                                                    | None     |
| exclude_domains      | Optional[List[str]]                          | List of domains to exclude in the search.                                                                                                                                                                                                                    | None     |
| start_crawl_date     | Optional[str]                                | Results will only include links **crawled** after this date.                                                                                                                                                                                                 | None     |
| end_crawl_date       | Optional[str]                                | Results will only include links **crawled** before this date.                                                                                                                                                                                                | None     |
| start_published_date | Optional[str]                                | Results will only include links with a **published** date after this date.                                                                                                                                                                                   | None     |
| end_published_date   | Optional[str]                                | Results will only include links with a **published** date before this date.                                                                                                                                                                                  | None     |
| type                 | Optional[str]                                | [The type of search](https://docs.exa.ai/sdks/#), keyword or neural.                                                                                                                                                                                         | "auto"   |
| category             | Optional[str]                                | A data category to focus on when searching, with higher comprehensivity and data cleanliness. Currently, the available categories are: company, research paper, news, linkedin profile, github, tweet, movie, song, personal site, pdf and financial report. | None     |
| include_text         | Optional[List[str]]                          | List of strings that must be present in webpage text of results. Currently, only 1 string is supported, of up to 5 words.                                                                                                                                    | None     |
| exclude_text         | Optional[List[str]]                          | List of strings that must not be present in webpage text of results. Currently, only 1 string is supported, of up to 5 words. Checks from the first 1000 words of the webpage text.                                                                          | None     |
| context              | Union[ContextContentsOptions, Literal[True]] | If true, concatenates results into a context string.                                                                                                                                                                                                         | None     |

#### Returns Example:

```json
{
  "autopromptString": "Here is a link to one of the hottest AI startups:",
  "results": [
    {
      "title": "Adept: Useful General Intelligence",
      "id": "https://www.adept.ai/",
      "url": "https://www.adept.ai/",
      "publishedDate": "2000-01-01",
      "author": null
    },
    {
      "title": "Home | Tenyx, Inc.",
      "id": "https://www.tenyx.com/",
      "url": "https://www.tenyx.com/",
      "publishedDate": "2019-09-10",
      "author": null
    }
  ],
  "requestId": "a78ebce717f4d712b6f8fe0d5d7753f8"
}
```

#### Return Parameters:

`SearchResponse[Result]`

| Field | Type | Description |
| --- | --- | --- |
| results | List[Result] | List of Result objects |
| context | Optional[str] | Results concatenated into a string |

#### Result Object:

| Field | Type | Description |
| --- | --- | --- |
| url | str | URL of the search result |
| id | str | Temporary ID for the document |
| title | Optional[str] | Title of the search result |
| published_date | Optional[str] | Estimated creation date |
| author | Optional[str] | Author of the content, if available |

### search_and_contents Method

Search and retrieve links with optional content (text, highlights, summaries).

#### Input Example:

```python
# Search with full text content
result_with_text = exa.search_and_contents(
    "AI in healthcare",
    text=True,
    num_results=2
)

# Search with highlights
result_with_highlights = exa.search_and_contents(
    "AI in healthcare",
    highlights=True,
    num_results=2
)

# Search with both text and highlights
result_with_text_and_highlights = exa.search_and_contents(
    "AI in healthcare",
    text=True,
    highlights=True,
    num_results=2
)

# Search with structured summary schema
company_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Company Information",
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "The name of the company"
        },
        "industry": {
            "type": "string",
            "description": "The industry the company operates in"
        },
        "foundedYear": {
            "type": "number",
            "description": "The year the company was founded"
        },
        "keyProducts": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "List of key products or services offered by the company"
        },
        "competitors": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "List of main competitors"
        }
    },
    "required": ["name", "industry"]
}

result_with_structured_summary = exa.search_and_contents(
    "OpenAI company information",
    summary={
        "schema": company_schema
    },
    category="company",
    num_results=3
)

# Parse the structured summary (returned as a JSON string)
first_result = result_with_structured_summary.results[0]
if first_result.summary:
    import json
    structured_data = json.loads(first_result.summary)

    print(structured_data["name"])        # e.g. "OpenAI"
    print(structured_data["industry"])    # e.g. "Artificial Intelligence"
    print(structured_data["keyProducts"]) # e.g. ["GPT-4", "DALL-E", "ChatGPT"]
```

#### Input Parameters:

Same as `search` method, plus:

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| text | Union[TextContentsOptions, Literal[True]] | If provided, includes the full text of the content in the results. | None |
| highlights | Union[HighlightsContentsOptions, Literal[True]] | If provided, includes highlights of the content in the results. | None |

#### Returns Example:

```json
{
  "results": [
    {
      "title": "2023 AI Trends in Health Care",
      "id": "https://aibusiness.com/verticals/2023-ai-trends-in-health-care-",
      "url": "https://aibusiness.com/verticals/2023-ai-trends-in-health-care-",
      "publishedDate": "2022-12-29",
      "author": "Wylie Wong",
      "text": "While the health care industry was initially slow to [... TRUNCATED IN THESE DOCS FOR BREVITY ...]",
      "highlights": [
        "But to do so, many health care institutions would like to share data, so they can build a more comprehensive dataset to use to train an AI model. Traditionally, they would have to move the data to one central repository. However, with federated or swarm learning, the data does not have to move. Instead, the AI model goes to each individual health care facility and trains on the data, he said. This way, health care providers can maintain security and governance over their data."
      ],
      "highlightScores": [
        0.5566554069519043
      ]
    },
    {
      "title": "AI in healthcare: Innovative use cases and applications",
      "id": "https://www.leewayhertz.com/ai-use-cases-in-healthcare",
      "url": "https://www.leewayhertz.com/ai-use-cases-in-healthcare",
      "publishedDate": "2023-02-13",
      "author": "Akash Takyar",
      "text": "The integration of AI in healthcare is not [... TRUNCATED IN THESE DOCS FOR BREVITY ...]",
      "highlights": [
        "The ability of AI to analyze large amounts of medical data and identify patterns has led to more accurate and timely diagnoses. This has been especially helpful in identifying complex medical conditions, which may be difficult to detect using traditional methods. Here are some examples of successful implementation of AI in healthcare. IBM Watson Health: IBM Watson Health is an AI-powered system used in healthcare to improve patient care and outcomes. The system uses natural language processing and machine learning to analyze large amounts of data and provide personalized treatment plans for patients."
      ],
      "highlightScores": [
        0.6563674807548523
      ]
    }
  ],
  "requestId": "d8fd59c78d34afc9da173f1fe5aa8965"
}
```

#### Return Parameters:

- Depends on `text` and `highlights`:
  - `SearchResponse[ResultWithText]`: Only text.
  - `SearchResponse[ResultWithHighlights]`: Only highlights.
  - `SearchResponse[ResultWithTextAndHighlights]`: Both.

#### SearchResponse[ResultWithTextAndHighlights]

| Field | Type | Description |
| --- | --- | --- |
| results | List[ResultWithTextAndHighlights] | List of ResultWithTextAndHighlights objects |
| context | Optional[str] | Results concatenated into a string |

#### ResultWithTextAndHighlights Object

| Field | Type | Description |
| --- | --- | --- |
| url | str | URL of the search result |
| id | str | Temporary ID for the document |
| title | Optional[str] | Title of the search result |
| published_date | Optional[str] | Estimated creation date |
| author | Optional[str] | Author of the content, if available |
| text | str | Text of the search result page (always present) |
| highlights | List[str] | Highlights of the search result (always present) |
| highlight_scores | List[float] | Scores of the highlights (always present) |

Note: Defaults to full text if neither is specified.

### find_similar Method

Find similar results to a URL.

#### Input Example:

```python
similar_results = exa.find_similar(
    "miniclip.com",
    num_results=2,
    exclude_source_domain=True
)
```

#### Input Parameters:

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| url | str | The URL of the webpage to find similar results for. | Required |
| num_results | Optional[int] | Number of similar results to return. | None |
| include_domains | Optional[List[str]] | List of domains to include in the search. | None |
| exclude_domains | Optional[List[str]] | List of domains to exclude from the search. | None |
| start_crawl_date | Optional[str] | Results will only include links **crawled** after this date. | None |
| end_crawl_date | Optional[str] | Results will only include links **crawled** before this date. | None |
| start_published_date | Optional[str] | Results will only include links with a **published** date after this date. | None |
| end_published_date | Optional[str] | Results will only include links with a **published** date before this date. | None |
| exclude_source_domain | Optional[bool] | If true, excludes results from the same domain as the input URL. | None |
| category | Optional[str] | A data category to focus on when searching, with higher comprehensivity and data cleanliness. | None |
| context | Union[ContextContentsOptions, Literal[True]] | If true, concatenates results into a context string. | None |

#### Returns Example:

```json
{
  "results": [
    {
      "title": "Play New Free Online Games Every Day",
      "id": "https://www.minigames.com/new-games",
      "url": "https://www.minigames.com/new-games",
      "publishedDate": "2000-01-01",
      "author": null
    },
    {
      "title": "Play The best Online Games",
      "id": "https://www.minigames.com/",
      "url": "https://www.minigames.com/",
      "publishedDate": "2000-01-01",
      "author": null
    }
  ],
  "requestId": "08fdc6f20e9f3ea87f860af3f6ccc30f"
}
```

#### Return Parameters:

`SearchResponse[Result]`

| Field | Type | Description |
| --- | --- | --- |
| results | List[Result] | List of Result objects |
| context | Optional[String] | Results concatenated into a string |

#### Result Object

Same as in `search`.

### find_similar_and_contents Method

Find similar with optional content.

#### Input Example:

```python
# Find similar with full text content
similar_with_text = exa.find_similar_and_contents(
    "https://example.com/article",
    text=True,
    num_results=2
)

# Find similar with highlights
similar_with_highlights = exa.find_similar_and_contents(
    "https://example.com/article",
    highlights=True,
    num_results=2
)

# Find similar with both text and highlights
similar_with_text_and_highlights = exa.find_similar_and_contents(
    "https://example.com/article",
    text=True,
    highlights=True,
    num_results=2
)
```

#### Input Parameters:

Same as `find_similar`, plus `text` and `highlights` like in `search_and_contents`.

#### Returns:

Depends on parameters (e.g., `SearchResponse[ResultWithTextAndHighlights]` for both). Defaults to full text.

### answer Method

Generate non-streaming answer with citations.

#### Input Example:

```python
response = exa.answer("What is the capital of France?")
print(response.answer)       # e.g. "Paris"
print(response.citations)    # list of citations used

# If you want the full text of the citations in the response:
response_with_text = exa.answer(
    "What is the capital of France?",
    text=True
)

print(response_with_text.citations[0].text)  # Full page text
```

#### Input Parameters:

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| query | str | The question to answer. | Required |
| text | Optional[bool] | If true, the full text of each citation is included in the result. | False |
| stream | Optional[bool] | Note: If true, an error is thrown. Use stream_answer() instead for streaming responses. | None |

#### Returns Example:

```json
{
  "answer": "The capital of France is Paris.",
  "citations": [
    {
      "id": "https://www.example.com/france",
      "url": "https://www.example.com/france",
      "title": "France - Wikipedia",
      "publishedDate": "2023-01-01",
      "author": null,
      "text": "France, officially the French Republic, is a country in... [truncated for brevity]"
    }
  ]
}
```

#### Return Parameters:

`AnswerResponse`

| Field | Type | Description |
| --- | --- | --- |
| answer | str | The generated answer text |
| citations | List[AnswerResult] | List of citations used to generate the answer |

#### AnswerResult Object

| Field | Type | Description |
| --- | --- | --- |
| id | str | Temporary ID for the document |
| url | str | URL of the citation |
| title | Optional[str] | Title of the content, if available |
| published_date | Optional[str] | Estimated creation date |
| author | Optional[str] | The author of the content, if available |
| text | Optional[str] | The full text of the content (if text=True) |

### stream_answer Method

Streaming version of answer.

#### Input Example:

```python
stream = exa.stream_answer("What is the capital of France?", text=True)
for chunk in stream:
    if chunk.content:
        print("Partial answer:", chunk.content)
    if chunk.citations:
        for citation in chunk.citations:
            print("Citation found:", citation.url)
```

#### Input Parameters:

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| query | str | The question to answer. | Required |
| text | Optional[bool] | If true, includes full text of each citation in the streamed response. | False |

#### Return Type:

Iterable `StreamAnswerResponse` yielding `StreamChunk`:

| Field | Type | Description |
| --- | --- | --- |
| content | Optional[str] | Partial text content of the answer so far. |
| citations | Optional[List[AnswerResult]] | Citations discovered in this chunk, if any. |

Use `stream.close()` if needed.

### Research Methods

#### research.create_task

Create async research task.

#### Input Example:

```python
from exa_py import Exa
import os

exa = Exa(os.environ["EXA_API_KEY"])

# Create a simple research task
instructions = "What is the latest valuation of SpaceX?"
schema = {
    "type": "object",
    "properties": {
        "valuation": {"type": "string"},
        "date": {"type": "string"},
        "source": {"type": "string"}
    }
}

task = exa.research.create_task(
    instructions=instructions,
    output_schema=schema
)

# Or even simpler - let the model infer the schema
simple_task = exa.research.create_task(
    instructions="What are the main benefits of meditation?",
    infer_schema=True
)

print(f"Task created with ID: {task.id}")
```

#### Input Parameters:

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| instructions | str | Natural language instructions describing what the research task should accomplish. | Required |
| model | Optional[str] | The research model to use. Options: “exa-research” (default), “exa-research-pro”. | “exa-research” |
| output_schema | Optional[Dict] | JSON Schema specification for the desired output structure. See json-schema.org/draft-07. | None |
| infer_schema | Optional[bool] | When true and no output schema is provided, an LLM will generate an output schema. | None |

#### Returns:

`ResearchTask`

| Field | Type | Description |
| --- | --- | --- |
| id | str | The unique identifier for the task |

#### Return Example:

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

#### research.get_task

Get task status.

#### Input Example:

```python
# Get a research task by ID
task_id = "your-task-id-here"
task = exa.research.get_task(task_id)

print(f"Task status: {task.status}")

if task.status == "completed":
    print(f"Results: {task.data}")
    print(f"Citations: {task.citations}")
```

#### Input Parameters:

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| task_id | str | The unique identifier of the task | Required |

#### Returns:

`ResearchTaskDetails`

| Field | Type | Description |
| --- | --- | --- |
| id | str | The unique identifier for the task |
| status | str | Task status: “running”, “completed”, or “failed” |
| instructions | str | The original instructions provided |
| schema | Optional[Dict] | The JSON schema specification used |
| data | Optional[Dict] | The research results (when completed) |
| citations | Optional[Dict[str, List]] | Citations grouped by root field (when completed) |

#### research.poll_task

Poll until complete.

#### Input Example:

```python
# Create and poll a task until completion
task = exa.research.create_task(
    instructions="Get information about Paris, France",
    output_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "population": {"type": "string"},
            "founded_date": {"type": "string"}
        }
    }
)

# Poll until completion
result = exa.research.poll_task(task.id)
print(f"Research complete: {result.data}")
```

#### Input Parameters:

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| task_id | str | The unique identifier of the task | Required |
| poll_interval | Optional[int] | Seconds between polling attempts | 2 |
| max_wait_time | Optional[int] | Maximum seconds to wait before timing out | 300 |

#### Returns:

`ResearchTaskDetails` (same as `get_task`).

#### research.list_tasks

List tasks with pagination.

#### Input Example:

```python
# List all research tasks
response = exa.research.list_tasks()
print(f"Found {len(response['data'])} tasks")

# List with pagination
response = exa.research.list_tasks(limit=10)
if response['hasMore']:
    next_page = exa.research.list_tasks(cursor=response['nextCursor'])
```

#### Input Parameters:

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| cursor | Optional[str] | Pagination cursor from previous request | None |
| limit | Optional[int] | Number of results to return (1-200) | 25 |

#### Returns:

Dictionary with:

| Field | Type | Description |
| --- | --- | --- |
| data | List[ResearchTaskDetails] | List of research task objects |
| hasMore | bool | Whether there are more results to paginate |
| nextCursor | Optional[str] | Cursor for the next page (if hasMore is true) |

#### Return Example:

```json
{
  "data": [
    {
      "id": "task-1",
      "status": "completed",
      "instructions": "Research SpaceX valuation",
      ...
    },
    {
      "id": "task-2",
      "status": "running",
      "instructions": "Compare GPU specifications",
      ...
    }
  ],
  "hasMore": true,
  "nextCursor": "eyJjcmVhdGVkQXQiOiIyMDI0LTAxLTE1VDE4OjMwOjAwWiIsImlkIjoidGFzay0yIn0="
}
```

[TypeScript SDK Specification](https://docs.exa.ai/sdks/typescript-sdk-specification)

## Exa Research API

From [[Exa Research]]. Merged with SDK research methods for completeness.

### How It Works

The Research API is an **asynchronous, multi-step pipeline** that transforms open-ended questions into grounded reports. You provide natural-language instructions (e.g., *"Compare the hardware roadmaps of the top GPU manufacturers"*) and an optional JSON Schema describing the output you want.

Under the hood, Exa agents perform multiple steps:
1. **Planning** – Your natural-language `instructions` are parsed by an LLM that decomposes the task into one or more research steps.
2. **Searching** – Specialized search agents issue semantic and keyword queries to Exa’s search engine, continuously expanding and refining the result set until they can fulfill the request.
3. **Reasoning & synthesis** – Reasoning models combine facts across sources and return structured JSON (if you provide `outputSchema`) or a detailed markdown report.

Because tasks are **asynchronous**, you submit a request and immediately receive a `researchId`. You can [poll the request](https://docs.exa.ai/reference/research/get-a-task) until it is complete or failed, or [list all tasks](https://docs.exa.ai/reference/research/list-tasks) to monitor progress in bulk.

### Best Practices

- **Be explicit** – Clear, scoped instructions lead to faster tasks and higher-quality answers. You should describe (1) what information you want (2) how the agent should find that information and (3) how the agent should compose its final report.
- **Keep schemas small** – 1-5 root fields is the sweet spot. If you need more, create multiple tasks.
- **Use enums** – Tight schema constraints improve accuracy and reduce hallucinations.

### Models

The Research API offers two advanced agentic researcher models that break down your instructions, search the web, extract and reason over facts, and return structured answers with citations.

- **exa-research** (default) adapts to the difficulty of the task, using more or less compute for individual steps. Recommended for most use cases.
- **exa-research-pro** maximizes quality by using the highest reasoning capability for every step. Recommended for the most complex, multi-step research tasks.

Here are typical completion times for each model:

| Model | p50 (seconds) | p90 (seconds) |
| --- | --- | --- |
| exa-research | 45 | 90 |
| exa-research-pro | 90 | 180 |

### Pricing

The Research API now uses **variable usage-based pricing**. You are billed based on how much work and reasoning the research agent does.

You are ONLY charged for tasks that complete successfully.

| Operation | exa-research | exa-research-pro | Notes |
| --- | --- | --- | --- |
| **Search** | $5/1k searches | $5/1k searches | Each unique search query issued by the agent |
| **Page read** | $5/1k pages read | $10/1k pages read | One “page” = 1,000 tokens from the web |
| **Reasoning tokens** | $5/1M tokens | $5/1M tokens | Specific LLM tokens used for reasoning and synthesis |

**Example:**  
A research task with `exa-research` that performs 6 searches, reads 20 pages of content, and uses 1,000 reasoning tokens would cost:  
$0.03 (6 searches × $5/1000) + $0.10 (20 pages × $5/1000) + $0.005 (1,000 reasoning tokens × $5/1,000,000) = $0.135

 For `exa-research-pro`, the same task would cost:  
$0.03 (6 searches × $5/1000) + $0.20 (20 pages × $10/1000) + $0.005 (1,000 reasoning tokens × $5/1,000,000) = $0.235

### Examples

#### Competitive Landscape Table

Compare the current flagship GPUs from NVIDIA, AMD, and Intel and extract pricing, TDP, and release date.

```python
import os
from exa_py import Exa

exa = Exa(os.environ["EXA_API_KEY"])

instructions = "Compare the current flagship GPUs from NVIDIA, AMD and Intel. Return a table of model name, MSRP USD, TDP watts, and launch date. Include citations for each cell."

schema = {
    "type": "object",
    "required": ["gpus"],
    "properties": {
        "gpus": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["manufacturer", "model", "msrpUsd", "tdpWatts", "launchDate"],
                "properties": {
                    "manufacturer": {"type": "string"},
                    "model": {"type": "string"},
                    "msrpUsd": {"type": "number"},
                    "tdpWatts": {"type": "integer"},
                    "launchDate": {"type": "string"}
                }
            }
        }
    },
    "additionalProperties": False
}

research = exa.research.create(
    model="exa-research",
    instructions=instructions,
    output_schema=schema
)
# Poll until completion
result = exa.research.poll_until_finished(research.researchId)
print(result)
```

#### Market Size Estimate

Estimate the total global market size (USD) for battery recycling in 2030 with a clear methodology.

```python
import os
from exa_py import Exa

exa = Exa(os.environ["EXA_API_KEY"])

instructions = "Estimate the global market size for battery recycling in 2030. Provide reasoning steps and cite sources."

schema = {
    "type": "object",
    "required": ["estimateUsd", "methodology"],
    "properties": {
        "estimateUsd": {"type": "number"},
        "methodology": {"type": "string"}
    },
    "additionalProperties": False
}
research = exa.research.create(
    model="exa-research",
    instructions=instructions,
    output_schema=schema
)
# Poll until completion
result = exa.research.poll_until_finished(research.researchId)
print(result)
```

#### Timeline of Key Events

Build a timeline of major OpenAI product releases from 2015 – 2023.

```python
import os
from exa_py import Exa

exa = Exa(os.environ["EXA_API_KEY"])

instructions = "Create a chronological timeline (year, month, brief description) of major OpenAI product releases from 2015 to 2023."

schema = {
    "type": "object",
    "required": ["events"],
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["date", "description"],
                "properties": {
                    "date": {"type": "string"},
                    "description": {"type": "string"}
                }
            }
        }
    },
    "additionalProperties": False
}

research = exa.research.create(
    model="exa-research",
    instructions=instructions,
    output_schema=schema
)
# Poll until completion
result = exa.research.poll_until_finished(research.researchId)
print(result)
```

### FAQs

Product teams, analysts, researchers, and anyone who needs **structured answers** that require reading multiple web sources — without having to build their own search + scraping + LLM pipeline.

`/answer` is designed for **single-shot Q&A**. The Research API handles **long-running, multi-step investigations**. It’s suitable for tasks that require complex reasoning over web data.

Tasks generally complete in 20–40 seconds. Simple tasks that can be solved with few searches complete faster, while complex schemas targeting niche subjects may take longer.

Be explicit about the objective and any constraints - Specify the **time range** or **types of sources** to consult if important - Use imperative verbs (“Compare”, “List”, “Summarize”) - Keep it under 4096 characters

You must have ≤ 8 root fields. It must not be more than 5 fields deep.

If your schema is not valid, an error will surface *before the task is created* with a message about what is invalid. You will not be charged for such requests.

[Livecrawling Contents](https://docs.exa.ai/reference/livecrawling-contents) [Migrating from Bing](https://docs.exa.ai/reference/migrating-from-bing)

## OpenAI Integrations

### OpenAI Responses API

From [[Exa OpenAI Responses API Guide]].

#### What is Exa?

Exa is the search engine built for AI. It finds information from across the web and delivers both links and the actual content from pages, making it easy to use with AI models. Exa uses neural search technology to understand the meaning of queries, not just keywords. The API works with both semantic search and traditional keyword methods.

#### Get Started

First, you’ll need API keys from both OpenAI and Exa:
- Get your Exa API key from the [Exa Dashboard](https://dashboard.exa.ai/api-keys)
- Get your OpenAI API key from the [OpenAI Dashboard](https://platform.openai.com/api-keys)

#### Complete Example

```python
import json
from openai import OpenAI
from exa_py import Exa

OPENAI_API_KEY = ""  # Add your OpenAI API key here
EXA_API_KEY = ""     # Add your Exa API key here

# Define the tool for Exa web search
tools = [{
    "type": "function",
    "name": "exa_websearch",
    "description": "Search the web using Exa. Provide relevant links in your answer.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query for Exa."
            }
        },
        "required": ["query"],
        "additionalProperties": False
    },
    "strict": True
}]

# Define the system message
system_message = {"role": "system", "content": "You are a helpful assistant. Use exa_websearch to find info when relevant. Always list sources."}

def run_exa_search(user_query):
    """Run an Exa web search with a dynamic user query."""
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    exa = Exa(api_key=EXA_API_KEY)
    
    # Create messages with the dynamic user query
    messages = [
        system_message,
        {"role": "user", "content": user_query}
    ]
    
    # Send initial request
    print("Sending initial request to OpenAI...")
    response = openai_client.responses.create(
        model="gpt-4o",
        input=messages,
        tools=tools
    )
    print("Initial OpenAI response:", response.output)

    # Check if the model returned a function call
    function_call = None
    for item in response.output:
        if item.type == "function_call" and item.name == "exa_websearch":
            function_call = item
            break

    # If exa_websearch was called
    if function_call:
        call_id = function_call.call_id
        args = json.loads(function_call.arguments)
        query = args.get("query", "")
        print(f"\nOpenAI requested a web search for: {query}")
        search_results = exa.search_and_contents(
            query=query,
            text = {
              "max_characters": 4000
            },
            type="auto"
        )

        # Store citations for later use in formatting
        citations = [{"url": result.url, "title": result.title} for result in search_results.results]
        search_results_str = str(search_results)
        
        # Provide the function call + function_call_output to the conversation
        messages.append({
            "type": "function_call",
            "name": function_call.name,
            "arguments": function_call.arguments,
            "call_id": call_id
        })

        messages.append({
            "type": "function_call_output",
            "call_id": call_id,
            "output": search_results_str
        })

        print("\nSending search results back to OpenAI for a final answer...")

        response = openai_client.responses.create(
            model="gpt-4o",
            input=messages,
            tools=tools
        )

        # Format the final response to include citations
        if hasattr(response, 'output_text') and response.output_text:

            # Add citations to the final output
            formatted_response = format_response_with_citations(response.output_text, citations)

            # Create a custom response object with citations
            if hasattr(response, 'model_dump'):

                # For newer versions of the OpenAI library that use Pydantic
                response_dict = response.model_dump()
            else:
                # For older versions or if model_dump is not available
                response_dict = response.dict() if hasattr(response, 'dict') else response.__dict__
            # Update the output with annotations
            if response.output and len(response.output) > 0:
                response_dict['output'] = [{
                    "type": "message",
                    "id": response.output[0].id if hasattr(response.output[0], 'id') else "msg_custom",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": formatted_response["text"],
                        "annotations": formatted_response["annotations"]
                    }]
                }]

                # Update the output_text property
                response_dict['output_text'] = formatted_response["text"]
                # Create a new response object (implementation may vary based on the OpenAI SDK version)
                try:
                    response = type(response)(**response_dict)
                except:
                    # If we can't create a new instance, we'll just print the difference
                    print("\nFormatted response with citations would be:", formatted_response)
    # Print final answer text
    print("\nFinal Answer:\n", response.output_text)
    print("\nAnnotations:", json.dumps(response.output[0].content[0].annotations if hasattr(response, 'output') and response.output and hasattr(response.output[0], 'content') else [], indent=2))
    print("\nFull Response with Citations:", response)
    return response

def format_response_with_citations(text, citations):
    """Format the response to include citations as annotations."""
    annotations = []
    formatted_text = text
    
    # For each citation, append a numbered reference to the text
    for i, citation in enumerate(citations):
        # Create annotation object
        start_index = len(formatted_text)
        citation_text = f"\n\n[{i+1}] {citation['url']}"
        end_index = start_index + len(citation_text)
        annotation = {
            "type": "url_citation",
            "start_index": start_index,
            "end_index": end_index,
            "url": citation["url"],
            "title": citation["title"]
        }

        # Add annotation to the array
        annotations.append(annotation)
        # Append citation to text
        formatted_text += citation_text
    return {
        "text": formatted_text,
        "annotations": annotations
    }
if __name__ == "__main__":

    # Example of how to use with a dynamic query
    user_query = input("Enter your question: ")
    run_exa_search(user_query)
```

Both examples show how to:
1. Set up the OpenAI Response API with Exa as a tool
2. Make a request to OpenAI
3. Handle the search function call
4. Send the search results back to OpenAI
5. Get the final response

Remember to replace the empty API key strings with your actual API keys when trying these examples.

#### How Tool Calling Works

Let’s break down how the Exa web search tool works with OpenAI’s Response API:
1. **Tool Definition**: First, we define our Exa search as a tool that OpenAI can use:
	```json
	{
	  "type": "function",
	  "name": "exa_websearch",
	  "description": "Search the web using Exa...",
	  "parameters": {
	    "query": "string"  // The search query parameter
	  }
	}
	```
2. **Initial Request**: When you send a message to OpenAI, the API looks at your message and decides if it needs to search the web. If it does, instead of giving a direct answer, it will return a “function call” in its output.
3. **Function Call**: If OpenAI decides to search, it returns something like:
	```json
	{
	  "type": "function_call",
	  "name": "exa_websearch",
	  "arguments": { "query": "your search query" }
	}
	```
4. **Search Execution**: Your code then:
	- Takes this search query
	- Calls Exa’s API to perform the actual web search
	- Gets real web results back
5. **Final Response**: You send these web results back to OpenAI, and it gives you a final answer using the fresh information from the web.

This back-and-forth process happens automatically in the code above, letting OpenAI use Exa’s web search when it needs to find current information.

#### Direct Research with Responses API

In addition to using Exa as a search tool, you can also access Exa’s powerful research capabilities directly through the OpenAI Responses API format. This provides a familiar interface for running complex research tasks.

Simply point the OpenAI client to Exa’s API and use our research models.

#### Available Models

- **`exa-research`** - Adapts compute to task difficulty. Best for most use cases.
- **`exa-research-pro`** - Maximum quality with highest reasoning capability. Best for complex, multi-step research.

Choose the right approach for your use case:

| Feature | Web Search Tool (Function Calling) | Direct Research |
| --- | --- | --- |
| **Use Case** | Augment LLM conversations with web data | Get comprehensive research reports |
| **Control** | Full control over search queries and integration | Automated multi-step research |
| **Response Time** | Fast (seconds) | Longer (45-180 seconds) |
| **Best For** | Interactive chatbots, real-time Q&A | In-depth analysis, research reports |

For detailed information about research capabilities, structured outputs, and pricing, see the [Exa Research documentation](https://docs.exa.ai/reference/exa-research).

[OpenAI SDK Compatibility](https://docs.exa.ai/reference/openai-sdk) [Vercel AI SDK](https://docs.exa.ai/reference/vercel)

### OpenAI SDK Compatibility

From [[Exa as a Drop-In OpenAI SDK Replacement]].

#### Overview

Exa provides OpenAI-compatible endpoints that work seamlessly with the OpenAI SDK:

| Endpoint | OpenAI Interface | Models Available | Use Case |
| --- | --- | --- | --- |
| `/chat/completions` | Chat Completions API | `exa`, `exa-research`, `exa-research-pro` | Traditional chat interface |
| `/responses` | Responses API | `exa-research`, `exa-research-pro` | Modern, simplified interface |

Exa will parse through your messages and send only the last message to `/answer` or `/research`.

#### Answer

To use Exa’s `/answer` endpoint via the chat completions interface:
1. Replace base URL with `https://api.exa.ai`
2. Replace API key with your Exa API key
3. Replace model name with `exa`.

See the full `/answer` endpoint reference [here](https://docs.exa.ai/reference/answer).

```python
from openai import OpenAI

client = OpenAI(
  base_url="https://api.exa.ai", # use exa as the base url
  api_key="YOUR_EXA_API_KEY", # update your api key
)

completion = client.chat.completions.create(
  model="exa",
  messages = [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What are the latest developments in quantum computing?"}
],
# use extra_body to pass extra parameters to the /answer endpoint
  extra_body={
    "text": True # include full text from sources
  }
)

print(completion.choices[0].message.content)  # print the response content
print(completion.choices[0].message.citations)  # print the citations
```

#### Research

To use Exa’s research models via the chat completions interface:
1. Replace base URL with `https://api.exa.ai`
2. Replace API key with your Exa API key
3. Replace model name with `exa-research` or `exa-research-pro`

See the full `/research` endpoint reference [here](https://docs.exa.ai/reference/research/create-a-task).

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.exa.ai",
    api_key=os.environ["EXA_API_KEY"],
)

completion = client.chat.completions.create(
    model="exa-research", # or exa-research-pro
    messages=[
        {"role": "user", "content": "What makes some LLMs so much better than others?"}
    ],
    stream=True,
)

for chunk in completion:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

#### Research via Responses API

You can also access Exa’s research models using OpenAI’s newer Responses API format.

#### Chat Wrapper

Exa provides a Python wrapper that automatically enhances any OpenAI chat completion with RAG capabilities. With one line of code, you can turn any OpenAI chat completion into an Exa-powered RAG system that handles search, chunking, and prompting automatically.

```python
from openai import OpenAI
from exa_py import Exa

# Initialize clients
openai = OpenAI(api_key='OPENAI_API_KEY')
exa = Exa('EXA_API_KEY')

# Wrap the OpenAI client
exa_openai = exa.wrap(openai)

# Use exactly like the normal OpenAI client
completion = exa_openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is the latest climate tech news?"}]
)

print(completion.choices[0].message.content)
```

The wrapped client works exactly like the native OpenAI client, except it automatically improves your completions with relevant search results when needed. The wrapper supports any parameters from the `exa.search()` function.

```python
completion = exa_openai.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    use_exa="auto",              # "auto", "required", or "none"
    num_results=5,               # defaults to 3
    result_max_len=1024,         # defaults to 2048 characters
    include_domains=["arxiv.org"],
    category="research paper",
    start_published_date="2019-01-01"
)
```

[OpenAI Tool Calling](https://docs.exa.ai/reference/openai-tool-calling) [OpenAI Responses API](https://docs.exa.ai/reference/openai-responses-api-with-exa)

## Exa OpenAPI Specification v1.2

From [[Exa OpenAPI Specification v12]]. This is the raw YAML spec for Exa's API endpoints. Below is the full content for reference.

```yaml
openapi: 3.1.0
info:
  version: 1.2.0
  title: Exa Search API
  description: A comprehensive API for internet-scale search, allowing users to perform queries and retrieve results from a wide variety of sources using embeddings-based and traditional search.
servers:
  - url: https://api.exa.ai
security:
  - apikey: []
paths:
  /search:
    post:
      operationId: search
      summary: Search
      description: Perform a search with a Exa prompt-engineered query and retrieve a list of relevant results. Optionally get contents.
      x-codeSamples:
        - lang: bash
          label: Simple search and contents
          source: |
            curl -X POST 'https://api.exa.ai/search' \
              -H 'x-api-key: YOUR-EXA-API-KEY' \
              -H 'Content-Type: application/json' \
              -d '{
                "query": "Latest research in LLMs",
                "text": true
              }'
        - lang: python
          label: Simple search and contents
          source: |
            # pip install exa-py
            from exa_py import Exa
            exa = Exa('YOUR_EXA_API_KEY')

            results = exa.search_and_contents(
                "Latest research in LLMs", 
                text=True
            )

            print(results)
        - lang: javascript
          label: Simple search and contents
          source: |
            // npm install exa-js
            import Exa from 'exa-js';
            const exa = new Exa('YOUR_EXA_API_KEY');

            const results = await exa.searchAndContents(
                'Latest research in LLMs', 
                { text: true }
            );

            console.log(results);
        - lang: php
          label: Simple search and contents
          source: ""
        - lang: go
          label: Simple search and contents
          source: ""
        - lang: java
          label: Simple search and contents
          source: ""
        - lang: bash
          label: Advanced search with filters
          source: |
            curl --request POST \
              --url https://api.exa.ai/search \
              --header 'x-api-key: <token>' \
              --header 'Content-Type: application/json' \
              --data '{
              "query": "Latest research in LLMs",
              "type": "auto",
              "category": "research paper",
              "numResults": 10,
              "contents": {
                "text": true,
                "summary": {
                  "query": "Main developments"
                },
                "subpages": 1,
                "subpageTarget": "sources",
                "extras": {
                  "links": 1,
                  "imageLinks": 1
                }
              }
            }'
        - lang: python
          label: Advanced search with filters
          source: |
            # pip install exa-py
            from exa_py import Exa
            exa = Exa('YOUR_EXA_API_KEY')

            results = exa.search_and_contents(
                "Latest research in LLMs",
                type="auto",
                category="research paper",
                num_results=10,
                text=True,
                summary={
                    "query": "Main developments"
                },
                subpages=1,
                subpage_target="sources",
                extras={
                    "links": 1,
                    "image_links": 1
                }
            )

            print(results)
        - lang: javascript
          label: Advanced search with filters
          source: |
            // npm install exa-js
            import Exa from 'exa-js';
            const exa = new Exa('YOUR_EXA_API_KEY');

            const results = await exa.searchAndContents('Latest research in LLMs', {
                type: 'auto',
                category: 'research paper',
                numResults: 10,
                contents: {
                    text: true,
                    summary: {
                        query: 'Main developments'
                    },
                    subpages: 1,
                    subpageTarget: 'sources',
                    extras: {
                        links: 1,
                        imageLinks: 1
                    }
                }
            });

            console.log(results);
        - lang: php
          label: Advanced search with filters
          source: ""
        - lang: go
          label: Advanced search with filters
          source: ""
        - lang: java
          label: Advanced search with filters
          source: ""
      requestBody:
        required: true
        content:
          application/json:
            schema:
              allOf:
                - type: object
                  properties:
                    query:
                      type: string
                      example: "Latest developments in LLM capabilities"
                      description: The query string for the search.
                    type:
                      type: string
                      enum:
                        - keyword
                        - neural
                        - fast
                        - auto
                      description: The type of search. Neural uses an embeddings-based model, keyword is google-like SERP, and auto (default) intelligently combines the two. Fast uses streamlined versions of the neural and keyword models.
                      example: "auto"
                      default: "auto"
                    category:
                      type: string
                      enum:
                        - company
                        - research paper
                        - news
                        - pdf
                        - github
                        - tweet
                        - personal site
                        - linkedin profile
                        - financial report
                      description: A data category to focus on.
                      example: "research paper"
                    userLocation:
                      type: string
                      description: The two-letter ISO country code of the user, e.g. US.
                      example: "US"
                  required:
                    - query
                - $ref: "#/components/schemas/CommonRequest"
      responses:
        "200":
          $ref: "#/components/responses/SearchResponse"
  /findSimilar:
    post:
      operationId: findSimilar
      summary: Find similar links
      description: Find similar links to the link provided. Optionally get contents.
      x-codeSamples:
        - lang: bash
          label: Find similar links
          source: |
            curl -X POST 'https://api.exa.ai/findSimilar' \
              -H 'x-api-key: YOUR-EXA-API-KEY' \
              -H 'Content-Type: application/json' \
              -d '{
                "url": "https://arxiv.org/abs/2307.06435",
                "text": true
              }'
        - lang: python
          label: Find similar links
          source: |
            # pip install exa-py
            from exa_py import Exa
            exa = Exa('YOUR_EXA_API_KEY')

            results = exa.find_similar_and_contents(
                url="https://arxiv.org/abs/2307.06435",
                text=True
            )

            print(results)
        - lang: javascript
          label: Find similar links
          source: |
            // npm install exa-js
            import Exa from 'exa-js';
            const exa = new Exa('YOUR_EXA_API_KEY');

            const results = await exa.findSimilarAndContents(
                'https://arxiv.org/abs/2307.06435',
                { text: true }
            );

            console.log(results);
        - lang: php
          label: Find similar links
          source: ""
        - lang: go
          label: Find similar links
          source: ""
        - lang: java
          label: Find similar links
          source: ""
      requestBody:
        required: true
        content:
          application/json:
            schema:
              allOf:
                - type: object
                  properties:
                    url:
                      type: string
                      example: "https://arxiv.org/abs/2307.06435"
                      description: The url for which you would like to find similar links.
                  required:
                    - url
                - $ref: "#/components/schemas/CommonRequest"
      responses:
        "200":
          $ref: "#/components/responses/FindSimilarResponse"
  /contents:
    post:
      summary: Get Contents
      operationId: "getContents"
      x-codeSamples:
        - lang: bash
          label: Simple contents retrieval
          source: |
            curl -X POST 'https://api.exa.ai/contents' \
              -H 'x-api-key: YOUR-EXA-API-KEY' \
              -H 'Content-Type: application/json' \
              -d '{
                "urls": ["https://arxiv.org/abs/2307.06435"],
                "text": true
              }'
        - lang: python
          label: Simple contents retrieval
          source: |
            # pip install exa-py
            from exa_py import Exa
            exa = Exa('YOUR_EXA_API_KEY')

            results = exa.get_contents(
                urls=["https://arxiv.org/abs/2307.06435"],
                text=True
            )

            print(results)
        - lang: javascript
          label: Simple contents retrieval
          source: |
            // npm install exa-js
            import Exa from 'exa-js';
            const exa = new Exa('YOUR_EXA_API_KEY');

            const results = await exa.getContents(
                ["https://arxiv.org/abs/2307.06435"],
                { text: true }
            );

            console.log(results);
        - lang: php
          label: Simple contents retrieval
          source: ""
        - lang: go
          label: Simple contents retrieval
          source: ""
        - lang: java
          label: Simple contents retrieval
          source: ""
        - lang: bash
          label: Advanced contents retrieval
          source: |
            curl --request POST \
              --url https://api.exa.ai/contents \
              --header 'x-api-key: YOUR-EXA-API-KEY' \
              --header 'Content-Type: application/json' \
              --data '{
                "urls": ["https://arxiv.org/abs/2307.06435"],
                "text": {
                  "maxCharacters": 1000,
                  "includeHtmlTags": false
                },
                "highlights": {
                  "numSentences": 3,
                  "highlightsPerUrl": 2,
                  "query": "Key findings"
                },
                "summary": {
                  "query": "Main research contributions"
                },
                "subpages": 1,
                "subpageTarget": "references",
                "extras": {
                  "links": 2,
                  "imageLinks": 1
                }
              }'
        - lang: python
          label: Advanced contents retrieval
          source: |
            # pip install exa-py
            from exa_py import Exa
            exa = Exa('YOUR_EXA_API_KEY')

            results = exa.get_contents(
                urls=["https://arxiv.org/abs/2307.06435"],
                text={
                    "maxCharacters": 1000,
                    "includeHtmlTags": False
                },
                highlights={
                    "numSentences": 3,
                    "highlightsPerUrl": 2,
                    "query": "Key findings"
                },
                summary={
                    "query": "Main research contributions"
                },
                subpages=1,
                subpage_target="references",
                extras={
                    "links": 2,
                    "image_links": 1
                }
            )

            print(results)
        - lang: javascript
          label: Advanced contents retrieval
          source: |
            // npm install exa-js
            import Exa from 'exa-js';
            const exa = new Exa('YOUR_EXA_API_KEY');

            const results = await exa.getContents(
                ["https://arxiv.org/abs/2307.06435"],
                {
                    text: {
                        maxCharacters: 1000,
                        includeHtmlTags: false
                    },
                    highlights: {
                        numSentences: 3,
                        highlightsPerUrl: 2,
                        query: "Key findings"
                    },
                    summary: {
                        query: "Main research contributions"
                    },
                    subpages: 1,
                    subpageTarget: "references",
                    extras: {
                        links: 2,
                        imageLinks: 1
                    }
                }
            );

            console.log(results);
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              allOf:
                - type: object
                  properties:
                    urls:
                      type: array
                      description: Array of URLs to crawl (backwards compatible with 'ids' parameter).
                      items:
                        type: string
                      example: ["https://arxiv.org/pdf/2307.06435"]
                    ids:
                      type: array
                      deprecated: true
                      description: Deprecated - use 'urls' instead. Array of document IDs obtained from searches.
                      items:
                        type: string
                      example: ["https://arxiv.org/pdf/2307.06435"]
                  required:
                    - urls
                - $ref: "#/components/schemas/ContentsRequest"
      responses:
        "200":
          $ref: "#/components/responses/ContentsResponse"
  /answer:
    post:
      operationId: answer
      summary: Generate an answer from search results
      description: |
        Performs a search based on the query and generates either a direct answer or a detailed summary with citations, depending on the query type.
      x-codeSamples:
        - lang: bash
          label: Simple answer
          source: |
            curl -X POST 'https://api.exa.ai/answer' \
              -H 'x-api-key: YOUR-EXA-API-KEY' \
              -H 'Content-Type: application/json' \
              -d '{
                "query": "What is the latest valuation of SpaceX?",
                "text": true
              }'
        - lang: python
          label: Simple answer
          source: |
            # pip install exa-py
            from exa_py import Exa
            exa = Exa('YOUR_EXA_API_KEY')

            result = exa.answer(
                "What is the latest valuation of SpaceX?",
                text=True
            )

            print(result)
        - lang: javascript
          label: Simple answer
          source: |
            // npm install exa-js
            import Exa from 'exa-js';
            const exa = new Exa('YOUR_EXA_API_KEY');

            const result = await exa.answer(
                'What is the latest valuation of SpaceX?',
                { text: true }
            );

            console.log(result);
        - lang: php
          label: Simple answer
          source: ""
        - lang: go
          label: Simple answer
          source: ""
        - lang: java
          label: Simple answer
          source: ""
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - query
              properties:
                query:
                  type: string
                  description: The question or query to answer.
                  example: "What is the latest valuation of SpaceX?"
                  minLength: 1
                stream:
                  type: boolean
                  default: false
                  description: If true, the response is returned as a server-sent events (SSS) stream.
                text:
                  type: boolean
                  default: false
                  description: If true, the response includes full text content in the search results
      responses:
        "200":
          $ref: "#/components/responses/AnswerResponse"
  /research/v1:
    get:
      description: Get a paginated list of research requests
      operationId: ResearchController_listResearch
      parameters:
        - name: cursor
          required: false
          in: query
          description: The cursor to paginate through the results
          schema:
            minLength: 1
            type: string
        - name: limit
          required: false
          in: query
          description: The number of results to return
          schema:
            minimum: 1
            maximum: 50
            default: 10
            type: number
      responses:
        '200':
          description: List of research requests
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ListResearchResponseDto'
      summary: List research requests
      tags: &ref_0
        - Research
    post:
      operationId: ResearchController_createResearch
      parameters: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ResearchCreateRequestDtoClass'
      responses:
        '201':
          description: Research request created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ResearchDtoClass'
      summary: Create a new research request
      tags: *ref_0
  /research/v1/{researchId}:
    get:
      description: Retrieve research by ID. Add ?stream=true for real-time SSE updates.
      operationId: ResearchController_getResearch
      parameters:
        - name: researchId
          required: true
          in: path
          schema:
            type: string
        - name: stream
          required: true
          in: query
          schema:
            type: string
        - name: events
          required: true
          in: query
          schema:
            type: string
      responses:
        '200':
          description: ''
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ResearchDtoClass'
      summary: Get a research request by id
      tags: *ref_0
components:
  securitySchemes:
    apikey:
      type: apiKey
      name: x-api-key
      in: header
      description: API key can be provided either via x-api-key header or Authorization header with Bearer scheme
    bearer:
      type: http
      scheme: bearer
      description: API key can be provided either via x-api-key header or Authorization header with Bearer scheme
  schemas:
    AnswerCitation:
      type: object
      properties:
        id:
          type: string
          description: The temporary ID for the document.
          example: "https://www.theguardian.com/science/2024/dec/11/spacex-valued-at-350bn-as-company-agrees-to-buy-shares-from-employees"
        url:
          type: string
          format: uri
          description: The URL of the search result.
          example: "https://www.theguardian.com/science/2024/dec/11/spacex-valued-at-350bn-as-company-agrees-to-buy-shares-from-employees"
        title:
          type: string
          description: The title of the search result.
          example: "SpaceX valued at $350bn as company agrees to buy shares from ..."
        author:
          type: string
          nullable: true
          description: If available, the author of the content.
          example: "Dan Milmon"
        publishedDate:
          type: string
          nullable: true
          description: An estimate of the creation date, from parsing HTML content. Format is YYYY-MM-DD.
          example: "2023-11-16T01:36:32.547Z"
        text:
          type: string
          description: The full text content of each source. Only present when includeText is enabled.
          example: "SpaceX valued at $350bn as company agrees to buy shares from ..."
        image:
          type: string
          format: uri
          description: The URL of the image associated with the search result, if available.
          example: "https://i.guim.co.uk/img/media/7cfee7e84b24b73c97a079c402642a333ad31e77/0_380_6176_3706/master/6176.jpg?width=1200&height=630&quality=85&auto=format&fit=crop&overlay-align=bottom%2Cleft&overlay-width=100p&overlay-base64=L2ltZy9zdGF0aWMvb3ZlcmxheXMvdGctZGVmYXVsdC5wbmc&enable=upscale&s=71ebb2fbf458c185229d02d380c01530"
        favicon:
          type: string
          format: uri
          description: The URL of the favicon for the search result's domain, if available.
          example: "https://assets.guim.co.uk/static/frontend/icons/homescreen/apple-touch-icon.svg"
    AnswerResult:
      type: object
      properties:
        answer:
          type: string
          description: The generated answer based on search results.
          example: "$350 billion."
        citations:
          type: array
          description: Search results used to generate the answer.
          items:
            $ref: "#/components/schemas/AnswerCitation"
    ContentsRequest:
      type: object
      properties:
        text:
          oneOf:
            - type: boolean
              title: "Simple text retrieval"
              description: If true, returns full page text with default settings. If false, disables text return.
            - type: object
              title: "Advanced text options"
              description: Advanced options for controlling text extraction. Use this when you need to limit text length or include HTML structure.
              properties:
                maxCharacters:
                  type: integer
                  description: Maximum character limit for the full page text. Useful for controlling response size and API costs.
                  example: 1000
                includeHtmlTags:
                  type: boolean
                  default: false
                  description: Include HTML tags in the response, which can help LLMs understand text structure and formatting.
                  example: false
        highlights:
          type: object
          description: Text snippets the LLM identifies as most relevant from each page.
          properties:
            numSentences:
              type: integer
              minimum: 1
              description: The number of sentences to return for each snippet.
              example: 1
            highlightsPerUrl:
              type: integer
              minimum: 1
              description: The number of snippets to return for each result.
              example: 1
            query:
              type: string
              description: Custom query to direct the LLM's selection of highlights.
              example: "Key advancements"
        summary:
          type: object
          description: Summary of the webpage
          properties:
            query:
              type: string
              description: Custom query for the LLM-generated summary.
              example: "Main developments"
            schema:
              type: object
              description: |
                JSON schema for structured output from summary. 
                See https://json-schema.org/overview/what-is-jsonschema for JSON Schema documentation.
              example:
                {
                  "$schema": "http://json-schema.org/draft-07/schema#",
                  "title": "Title",
                  "type": "object",
                  "properties":
                    {
                      "Property 1":
                        { "type": "string", "description": "Description" },
                      "Property 2":
                        {
                          "type": "string",
                          "enum": ["option 1", "option 2", "option 3"],
                          "description": "Description",
                        },
                    },
                  "required": ["Property 1"],
                }
        livecrawl:
          type: string
          enum: [never, fallback, always, preferred]
          description: |
            Options for livecrawling pages.
            'never': Disable livecrawling (default for neural search).
            'fallback': Livecrawl when cache is empty (default for keyword search).
            'always': Always livecrawl.
            'preferred': Always try to livecrawl, but fall back to cache if crawling fails.
          example: "always"
        livecrawlTimeout:
          type: integer
          default: 10000
          description: The timeout for livecrawling in milliseconds.
          example: 1000
        subpages:
          type: integer
          default: 0
          description: The number of subpages to crawl. The actual number crawled may be limited by system constraints.
          example: 1
        subpageTarget:
          oneOf:
            - type: string
            - type: array
              items:
                type: string
          description: Keyword to find specific subpages of search results. Can be a single string or an array of strings, comma delimited.
          example: "sources"
        extras:
          type: object
          description: Extra parameters to pass.
          properties:
            links:
              type: integer
              default: 0
              description: Number of URLs to return from each webpage.
              example: 1
            imageLinks:
              type: integer
              default: 0
              description: Number of images to return for each result.
              example: 1
        context:
          oneOf:
            - type: boolean
              description: Formats the search resutls into a context string ready for LLMs.
              example: true
            - type: object
              description: Formats the search resutls into a context string ready for LLMs.
              properties:
                maxCharacters:
                  type: integer
                  description: Maximum character limit.
                  example: 10000

    CommonRequest:
      type: object
      properties:
        numResults:
          type: integer
          maximum: 100
          default: 10
          minimum: 1
          description: Number of results to return (up to thousands of results available for custom plans)
          example: 10
        includeDomains:
          type: array
          items:
            type: string
          description: List of domains to include in the search. If specified, results will only come from these domains.
          example:
            - arxiv.org
            - paperswithcode.com
        excludeDomains:
          type: array
          items:
            type: string
          description: List of domains to exclude from search results. If specified, no results will be returned from these domains.
        startCrawlDate:
          type: string
          format: date-time
          description: Crawl date refers to the date that Exa discovered a link. Results will include links that were crawled after this date. Must be specified in ISO 8601 format.
          example: 2023-01-01
        endCrawlDate:
          type: string
          format: date-time
          description: Crawl date refers to the date that Exa discovered a link. Results will include links that were crawled before this date. Must be specified in ISO 8601 format.
          example: 2023-12-31
        startPublishedDate:
          type: string
          format: date-time
          description: Only links with a published date after this will be returned. Must be specified in ISO 8601 format.
          example: 2023-01-01
        endPublishedDate:
          type: string
          format: date-time
          description: Only links with a published date before this will be returned. Must be specified in ISO 8601 format.
          example: 2023-12-31
        includeText:
          type: array
          items:
            type: string
          description: List of strings that must be present in webpage text of results. Currently, only 1 string is supported, of up to 5 words.
          example:
            - large language model
        excludeText:
          type: array
          items:
            type: string
          description: List of strings that must not be present in webpage text of results. Currently, only 1 string is supported, of up to 5 words. Checks from the first 1000 words of the webpage text.
          example:
            - course
        context:
          oneOf:
            - type: boolean
              description: Formats the search results into a context string ready for LLMs.
              example: true
            - type: object
              description: Formats the search results into a context string ready for LLMs.
              properties:
                maxCharacters:
                  type: integer
                  description: Maximum character limit.
                  example: 10000

        contents:
          $ref: "#/components/schemas/ContentsRequest"

    Result:
      type: object
      properties:
        title:
          type: string
          description: The title of the search result.
          example: "A Comprehensive Overview of Large Language Models"
        url:
          type: string
          format: uri
          description: The URL of the search result.
          example: "https://arxiv.org/pdf/2307.06435.pdf"
        publishedDate:
          type: string
          nullable: true
          description: An estimate of the creation date, from parsing HTML content. Format is YYYY-MM-DD.
          example: "2023-11-16T01:36:32.547Z"
        author:
          type: string
          nullable: true
          description: If available, the author of the content.
          example: "Humza  Naveed, University of Engineering and Technology (UET), Lahore, Pakistan"
        score:
          type: number
          nullable: true
          description: A number from 0 to 1 representing similarity between the query/url and the result.
          example: 0.4600165784358978
        id:
          type: string
          description: The temporary ID for the document. Useful for /contents endpoint.
          example: "https://arxiv.org/abs/2307.06435"
        image:
          type: string
          format: uri
          description: The URL of an image associated with the search result, if available.
          example: "https://arxiv.org/pdf/2307.06435.pdf/page_1.png"
        favicon:
          type: string
          format: uri
          description: The URL of the favicon for the search result's domain.
          example: "https://arxiv.org/favicon.ico"

    ResultWithContent:
      allOf:
        - $ref: "#/components/schemas/Result"
        - type: object
          properties:
            text:
              type: string
              description: The full content text of the search result.
              example: "Abstract Large Language Models (LLMs) have recently demonstrated remarkable capabilities..."
            highlights:
              type: array
              items:
                type: string
              description: Array of highlights extracted from the search result content.
              example:
                - "Such requirements have limited their adoption..."
            highlightScores:
              type: array
              items:
                type: number
                format: float
              description: Array of cosine similarity scores for each highlighted
              example: [0.4600165784358978]
            summary:
              type: string
              description: Summary of the webpage
              example: "This overview paper on Large Language Models (LLMs) highlights key developments..."
            subpages:
              type: array
              items:
                $ref: "#/components/schemas/ResultWithContent"
              description: Array of subpages for the search result.
              example:
                [
                  {
                    "id": "https://arxiv.org/abs/2303.17580",
                    "url": "https://arxiv.org/pdf/2303.17580.pdf",
                    "title": "HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face",
                    "author": "Yongliang  Shen, Microsoft Research Asia, Kaitao  Song, Microsoft Research Asia, Xu  Tan, Microsoft Research Asia, Dongsheng  Li, Microsoft Research Asia, Weiming  Lu, Microsoft Research Asia, Yueting  Zhuang, Microsoft Research Asia, yzhuang@zju.edu.cn, Zhejiang  University, Microsoft Research Asia, Microsoft  Research, Microsoft Research Asia",
                    "publishedDate": "2023-11-16T01:36:20.486Z",
                    "text": "HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face Date Published: 2023-05-25 Authors: Yongliang Shen, Microsoft Research Asia Kaitao Song, Microsoft Research Asia Xu Tan, Microsoft Research Asia Dongsheng Li, Microsoft Research Asia Weiming Lu, Microsoft Research Asia Yueting Zhuang, Microsoft Research Asia, yzhuang@zju.edu.cn Zhejiang University, Microsoft Research Asia Microsoft Research, Microsoft Research Asia Abstract Solving complicated AI tasks with different domains and modalities is a key step toward artificial general intelligence. While there are abundant AI models available for different domains and modalities, they cannot handle complicated AI tasks. Considering large language models (LLMs) have exhibited exceptional ability in language understanding, generation, interaction, and reasoning, we advocate that LLMs could act as a controller to manage existing AI models to solve complicated AI tasks and language could be a generic interface to empower t",
                    "summary": "HuggingGPT is a framework using ChatGPT as a central controller to orchestrate various AI models from Hugging Face to solve complex tasks. ChatGPT plans the task, selects appropriate models based on their descriptions, executes subtasks, and summarizes the results. This approach addresses limitations of LLMs by allowing them to handle multimodal data (vision, speech) and coordinate multiple models for complex tasks, paving the way for more advanced AI systems.",
                    "highlights":
                      [
                        "2) Recently, some researchers started to investigate the integration of using tools or models in LLMs  .",
                      ],
                    "highlightScores": [0.32679107785224915],
                  },
                ]
            extras:
              type: object
              description: Results from extras.
              properties:
                links:
                  type: array
                  items:
                    type: string
                  description: Array of links from the search result.
                  example: []

    CostDollars:
      type: object
      properties:
        total:
          type: number
          format: float
          description: Total dollar cost for your request
          example: 0.005
        breakDown:
          type: array
          description: Breakdown of costs by operation type
          items:
            type: object
            properties:
              search:
                type: number
                format: float
                description: Cost of your search operations
                example: 0.005
              contents:
                type: number
                format: float
                description: Cost of your content operations
                example: 0
              breakdown:
                type: object
                properties:
                  keywordSearch:
                    type: number
                    format: float
                    description: Cost of your keyword search operations
                    example: 0
                  neuralSearch:
                    type: number
                    format: float
                    description: Cost of your neural search operations
                    example: 0.005
                  contentText:
                    type: number
                    format: float
                    description: Cost of your text content retrieval
                    example: 0
                  contentHighlight:
                    type: number
                    format: float
                    description: Cost of your highlight generation
                    example: 0
                  contentSummary:
                    type: number
                    format: float
                    description: Cost of your summary generation
                    example: 0
        perRequestPrices:
          type: object
          description: Standard price per request for different operations
          properties:
            neuralSearch_1_25_results:
              type: number
              format: float
              description: Standard price for neural search with 1-25 results
              example: 0.005
            neuralSearch_26_100_results:
              type: number
              format: float
              description: Standard price for neural search with 26-100 results
              example: 0.025
            neuralSearch_100_plus_results:
              type: number
              format: float
              description: Standard price for neural search with 100+ results
              example: 1
            keywordSearch_1_100_results:
              type: number
              format: float
              description: Standard price for keyword search with 1-100 results
              example: 0.0025
            keywordSearch_100_plus_results:
              type: number
              format: float
              description: Standard price for keyword search with 100+ results
              example: 3
        perPagePrices:
          type: object
          description: Standard price per page for different content operations
          properties:
            contentText:
              type: number
              format: float
              description: Standard price per page for text content
              example: 0.001
            contentHighlight:
              type: number
              format: float
              description: Standard price per page for highlights
              example: 0.001
            contentSummary:
              type: number
              format: float
              description: Standard price per page for summaries
              example: 0.001

    ListResearchResponseDto:
      type:
        - object
      properties:
        data:
          type:
            - array
          items:
            discriminator:
              propertyName: status
            examples: &ref_1
              - researchId: 01jszdfs0052sg4jc552sg4jc5
                model: exa-research
                instructions: What species of ant are similar to honeypot ants?
                status: running
              - researchId: 01jszdfs0052sg4jc552sg4jc5
                model: exa-research
                instructions: What species of ant are similar to honeypot ants?
                status: completed
                output: Melophorus bagoti
            $ref: '#/components/schemas/ResearchDtoClass'
          description: The list of research requests
        hasMore:
          type:
            - boolean
          description: Whether there are more results to paginate through
        nextCursor:
          type:
            - string
            - 'null'
          description: The cursor to paginate through the next set of results
      required:
        - data
        - hasMore
        - nextCursor
    ResearchCreateRequestDtoClass:
      type:
        - object
      properties:
        model:
          default: exa-research
          type:
            - string
          enum:
            - exa-research
            - exa-research-pro
        instructions:
          type:
            - string
          maxLength: 4096
          description: Instructions for what research should be conducted
        outputSchema:
          type:
            - object
          additionalProperties: {}
      required:
        - instructions
      examples:
        - model: exa-research
          instructions: What species of ant are similar to honeypot ants?
    ResearchDtoClass:
      discriminator:
        propertyName: status
      oneOf:
        - type:
            - object
          properties:
            researchId:
              type:
                - string
              description: The unique identifier for the research request
            createdAt:
              type:
                - number
              description: Milliseconds since epoch time
            model:
              default: exa-research
              type:
                - string
              enum:
                - exa-research
                - exa-research-pro
              description: The model used for the research request
            instructions:
              type:
                - string
              description: The instructions given to this research request
            status:
              type:
                - string
              enum:
                - pending
          required:
            - researchId
            - createdAt
            - instructions
            - status
        - type:
            - object
          properties:
            researchId:
              type:
                - string
              description: The unique identifier for the research request
            createdAt:
              type:
                - number
              description: Milliseconds since epoch time
            model:
              default: exa-research
              type:
                - string
              enum:
                - exa-research
                - exa-research-pro
              description: The model used for the research request
            instructions:
              type:
                - string
              description: The instructions given to this research request
            status:
              type:
                - string
              enum:
                - running
            events:
              type:
                - array
              items:
                $ref: '#/components/schemas/ResearchEventDtoClass'
          required:
            - researchId
            - createdAt
            - instructions
            - status
        - type:
            - object
          properties:
            researchId:
              type:
                - string
              description: The unique identifier for the research request
            createdAt:
              type:
                - number
              description: Milliseconds since epoch time
            model:
              default: exa-research
              type:
                - string
              enum:
                - exa-research
                - exa-research-pro
              description: The model used for the research request
            instructions:
              type:
                - string
              description: The instructions given to this research request
            status:
              type:
                - string
              enum:
                - completed
            events:
              type:
                - array
              items:
                $ref: '#/components/schemas/ResearchEventDtoClass'
            output:
              type:
                - object
              properties:
                content:
                  type:
                    - string
                parsed:
                  type:
                    - object
                  additionalProperties: {}
              required:
                - content
            costDollars:
              type:
                - object
              properties:
                total:
                  type:
                    - number
                numSearches:
                  type:
                    - number
                numPages:
                  type:
                    - number
                reasoningTokens:
                  type:
                    - number
              required:
                - total
                - numSearches
                - numPages
                - reasoningTokens
          required:
            - researchId
            - createdAt
            - instructions
            - status
            - output
            - costDollars
        - type:
            - object
          properties:
            researchId:
              type:
                - string
              description: The unique identifier for the research request
            createdAt:
              type:
                - number
              description: Milliseconds since epoch time
            model:
              default: exa-research
              type:
                - string
              enum:
                - exa-research
                - exa-research-pro
              description: The model used for the research request
            instructions:
              type:
                - string
              description: The instructions given to this research request
            status:
              type:
                - string
              enum:
                - canceled
            events:
              type:
                - array
              items:
                $ref: '#/components/schemas/ResearchEventDtoClass'
          required:
            - researchId
            - createdAt
            - instructions
            - status
        - type:
            - object
          properties:
            researchId:
              type:
                - string
              description: The unique identifier for the research request
            createdAt:
              type:
                - number
              description: Milliseconds since epoch time
            model:
              default: exa-research
              type:
                - string
              enum:
                - exa-research
                - exa-research-pro
              description: The model used for the research request
            instructions:
              type:
                - string
              description: The instructions given to this research request
            status:
              type:
                - string
              enum:
                - failed
            events:
              type:
                - array
              items:
                $ref: '#/components/schemas/ResearchEventDtoClass'
            error:
              type:
                - string
              description: A message indicating why the request failed
          required:
            - researchId
            - createdAt
            - instructions
            - status
            - error
      examples: *ref_1
    ResearchOperationDtoClass:
      discriminator:
        propertyName: type
      oneOf:
        - type:
            - object
          properties:
            type:
              type:
                - string
              enum:
                - think
            content:
              type:
                - string
          required:
            - type
            - content
        - type:
            - object
          properties:
            type:
              type:
                - string
              enum:
                - search
            searchType:
              type:
                - string
              enum:
                - neural
                - keyword
                - auto
                - fast
            goal:
              type:
                - string
            query:
              type:
                - string
            results:
              type:
                - array
              items:
                type:
                  - object
                properties:
                  url:
                    type:
                      - string
                required:
                  - url
            pageTokens:
              type:
                - number
          required:
            - type
            - searchType
            - query
            - results
            - pageTokens
        - type:
            - object
          properties:
            type:
              type:
                - string
              enum:
                - crawl
            goal:
              type:
                - string
            result:
              type:
                - object
              properties:
                url:
                  type:
                    - string
              required:
                - url
            pageTokens:
              type:
                - number
          required:
            - type
            - result
            - pageTokens
    ResearchEventDtoClass:
      oneOf:
        - discriminator:
            propertyName: eventType
          oneOf:
            - type:
                - object
              properties:
                eventType:
                  type:
                    - string
                  enum:
                    - research-definition
                instructions:
                  type:
                    - string
                outputSchema:
                  type:
                    - object
                  additionalProperties: {}
                createdAt:
                  type:
                    - number
                  description: Milliseconds since epoch time
                researchId:
                  type:
                    - string
              required:
                - eventType
                - instructions
                - createdAt
                - researchId
            - type:
                - object
              properties:
                eventType:
                  type:
                    - string
                  enum:
                    - research-output
                output:
                  discriminator:
                    propertyName: outputType
                  oneOf:
                    - type:
                        - object
                      properties:
                        outputType:
                          type:
                            - string
                          enum:
                            - completed
                        costDollars:
                          type:
                            - object
                          properties:
                            total:
                              type:
                                - number
                            numSearches:
                              type:
                                - number
                            numPages:
                              type:
                                - number
                            reasoningTokens:
                              type:
                                - number
                          required:
                            - total
                            - numSearches
                            - numPages
                            - reasoningTokens
                        content:
                          type:
                            - string
                        parsed:
                          type:
                            - object
                          additionalProperties: {}
                      required:
                        - outputType
                        - costDollars
                        - content
                    - type:
                        - object
                      properties:
                        outputType:
                          type:
                            - string
                          enum:
                            - failed
                        error:
                          type:
                            - string
                      required:
                        - outputType
                        - error
                createdAt:
                  type:
                    - number
                  description: Milliseconds since epoch time
                researchId:
                  type:
                    - string
              required:
                - eventType
                - output
                - createdAt
                - researchId
        - discriminator:
            propertyName: eventType
          oneOf:
            - type:
                - object
              properties:
                eventType:
                  type:
                    - string
                  enum:
                    - plan-definition
                planId:
                  type:
                    - string
                createdAt:
                  type:
                    - number
                  description: Milliseconds since epoch time
                researchId:
                  type:
                    - string
              required:
                - eventType
                - planId
                - createdAt
                - researchId
            - type:
                - object
              properties:
                eventType:
                  type:
                    - string
                  enum:
                    - plan-operation
                planId:
                  type:
                    - string
                operationId:
                  type:
                    - string
                data:
                  discriminator:
                    propertyName: type
                  $ref: '#/components/schemas/ResearchOperationDtoClass'
                createdAt:
                  type:
                    - number
                  description: Milliseconds since epoch time
                researchId:
                  type:
                    - string
              required:
                - eventType
                - planId
                - operationId
                - data
                - createdAt
                - researchId
            - type:
                - object
              properties:
                eventType:
                  type:
                    - string
                  enum:
                    - plan-output
                planId:
                  type:
                    - string
                output:
                  discriminator:
                    propertyName: outputType
                  oneOf:
                    - type:
                        - object
                      properties:
                        outputType:
                          type:
                            - string
                          enum:
                            - tasks
                        reasoning:
                          type:
                            - string
                        tasksInstructions:
                          type:
                            - array
                          items:
                            type:
                              - string
                      required:
                        - outputType
                        - reasoning
                        - tasksInstructions
                    - type:
                        - object
                      properties:
                        outputType:
                          type:
                            - string
                          enum:
                            - stop
                        reasoning:
                          type:
                            - string
                      required:
                        - outputType
                        - reasoning
                createdAt:
                  type:
                    - number
                  description: Milliseconds since epoch time
                researchId:
                  type:
                    - string
              required:
                - eventType
                - planId
                - output
                - createdAt
                - researchId
        - discriminator:
            propertyName: eventType
          oneOf:
            - type:
                - object
              properties:
                eventType:
                  type:
                    - string
                  enum:
                    - task-definition
                planId:
                  type:
                    - string
                taskId:
                  type:
                    - string
                instructions:
                  type:
                    - string
                createdAt:
                  type:
                    - number
                  description: Milliseconds since epoch time
                researchId:
                  type:
                    - string
              required:
                - eventType
                - planId
                - taskId
                - instructions
                - createdAt
                - researchId
            - type:
                - object
              properties:
                eventType:
                  type:
                    - string
                  enum:
                    - task-operation
                planId:
                  type:
                    - string
                taskId:
                  type:
                    - string
                operationId:
                  type:
                    - string
                data:
                  discriminator:
                    propertyName: type
                  $ref: '#/components/schemas/ResearchOperationDtoClass'
                createdAt:
                  type:
                    - number
                  description: Milliseconds since epoch time
                researchId:
                  type:
                    - string
              required:
                - eventType
                - planId
                - taskId
                - operationId
                - data
                - createdAt
                - researchId
            - type:
                - object
              properties:
                eventType:
                  type:
                    - string
                  enum:
                    - task-output
                planId:
                  type:
                    - string
                taskId:
                  type:
                    - string
                output:
                  type:
                    - object
                  properties:
                    outputType:
                      type:
                        - string
                      enum:
                        - completed
                    content:
                      type:
                        - string
                  required:
                    - outputType
                    - content
                createdAt:
                  type:
                    - number
                  description: Milliseconds since epoch time
                researchId:
                  type:
                    - string
              required:
                - eventType
                - planId
                - taskId
                - output
                - createdAt
                - researchId
  responses:
    SearchResponse:
      description: OK
      content:
        application/json:
          schema:
            type: object
            properties:
              requestId:
                type: string
                description: Unique identifier for the request
                example: "b5947044c4b78efa9552a7c89b306d95"
              resolvedSearchType:
                type: string
                enum: [neural, keyword]
                description: The search type that was actually used for this request
                example: "neural"
              results:
                type: array
                description: A list of search results containing title, URL, published date, author, and score.
                items:
                  $ref: "#/components/schemas/ResultWithContent"
              searchType:
                type: string
                enum: [neural, keyword]
                description: For auto searches, indicates which search type was selected.
                example: "auto"
              context:
                type: string
                description: A formatted string of the search results ready for LLMs.
              costDollars:
                $ref: "#/components/schemas/CostDollars"

    FindSimilarResponse:
      description: OK
      content:
        application/json:
          schema:
            type: object
            properties:
              requestId:
                type: string
                description: Unique identifier for the request
                example: "c6958155d5c89ffa0663b7c90c407396"
              context:
                type: string
                description: A formatted string of the search results ready for LLMs.
              results:
                type: array
                description: A list of search results containing title, URL, published date, author, and score.
                items:
                  $ref: "#/components/schemas/ResultWithContent"
              costDollars:
                $ref: "#/components/schemas/CostDollars"

    ContentsResponse:
      description: OK
      content:
        application/json:
          schema:
            type: object
            properties:
              requestId:
                type: string
                description: Unique identifier for the request
                example: "e492118ccdedcba5088bfc4357a8a125"
              results:
                type: array
                items:
                  $ref: "#/components/schemas/ResultWithContent"
              context:
                type: string
                description: A formatted string of the search results ready for LLMs.
              statuses:
                type: array
                description: Status information for each requested URL
                items:
                  type: object
                  properties:
                    id:
                      type: string
                      description: The URL that was requested
                      example: "https://example.com"
                    status:
                      type: string
                      enum: ["success", "error"]
                      description: Status of the content fetch operation
                      example: "success"
                    error:
                      type: object
                      nullable: true
                      description: Error details, only present when status is "error"
                      properties:
                        tag:
                          type: string
                          enum:
                            [
                              "CRAWL_NOT_FOUND",
                              "CRAWL_TIMEOUT",
                              "CRAWL_LIVECRAWL_TIMEOUT",
                              "SOURCE_NOT_AVAILABLE",
                              "CRAWL_UNKNOWN_ERROR",
                            ]
                          description: Specific error type
                          example: "CRAWL_NOT_FOUND"
                        httpStatusCode:
                          type: integer
                          nullable: true
                          description: The corresponding HTTP status code
                          example: 404
              costDollars:
                $ref: "#/components/schemas/CostDollars"

    AnswerResponse:
      description: OK
      content:
        application/json:
          schema:
            allOf:
              - $ref: "#/components/schemas/AnswerResult"
              - type: object
                properties:
                  costDollars:
                    $ref: "#/components/schemas/CostDollars"
        text/event-stream:
          schema:
            type: object
            properties:
              answer:
                type: string
                description: Partial answer chunk when streaming is enabled.
              citations:
                type: array
                items:
                  $ref: "#/components/schemas/AnswerCitation"
```

## Exa MCP

From [[Exa MCP Server Overview]].

Exa MCP Server enables AI assistants like Claude to perform real-time web searches through the Exa Search API, allowing them to access up-to-date information from the internet. It is open-source, check out [GitHub](https://github.com/exa-labs/exa-mcp-server/).

### Get your API key

Before you begin, you’ll need an Exa API key: [Get your Exa API key](https://dashboard.exa.ai/api-keys)

### Remote Exa MCP

Connect directly to Exa’s hosted MCP server using this URL instead of running it locally:

```
https://mcp.exa.ai/mcp?exaApiKey=your-exa-api-key
```

To configure Claude Desktop, add this to your configuration file:

```
{
  "mcpServers": {
    "exa": {
      "command": "npx",
      "args": [
        "-y",
        "mcp-remote",
        "https://mcp.exa.ai/mcp?exaApiKey=your-exa-api-key"
      ]
    }
  }
}
```

### Available Tools

Exa MCP includes several specialized search tools:

| Tool | Description |
| --- | --- |
| `deep_researcher_start` | Start a smart AI researcher for complex questions. The AI will search the web, read many sources, and think deeply about your question to create a detailed research report |
| `deep_researcher_check` | Check if your research is ready and get the results. Use this after starting a research task to see if it’s done and get your comprehensive report |
| `web_search_exa` | Performs real-time web searches with optimized results and content extraction |
| `company_research` | Comprehensive company research tool that crawls company websites to gather detailed information about businesses |
| `crawling` | Extracts content from specific URLs, useful for reading articles, PDFs, or any web page when you have the exact URL |
| `linkedin_search` | Search LinkedIn for companies and people using Exa AI. Simply include company names, person names, or specific LinkedIn URLs in your query |

### Usage Examples

Once configured, you can ask Claude to perform searches:
- “Research the company exa.ai and find information about their pricing”
- “Start a deep research project on the impact of artificial intelligence on healthcare, then check when it’s complete to get a comprehensive report”

### Local Installation

#### Prerequisites

- [Node.js](https://nodejs.org/) v18 or higher.
- [Claude Desktop](https://claude.ai/download) installed (optional). Exa MCP also works with other MCP-compatible clients like Cursor, Windsurf, and more).
- An Exa API key (see above).

#### Using Claude Code

The quickest way to set up Exa MCP is using Claude Code:

```
claude mcp add exa -e EXA_API_KEY=YOUR_API_KEY -- npx -y exa-mcp-server
```

Replace `YOUR_API_KEY` with your Exa API key from above.

#### Using NPX

The simplest way to install and run Exa MCP is via NPX:

```
# Install globally
npm install -g exa-mcp-server

# Or run directly with npx
npx exa-mcp-server
```

To specify which tools to enable:

```
# Enable only web search
npx exa-mcp-server --tools=web_search

# Enable deep researcher tools
npx exa-mcp-server --tools=deep_researcher_start,deep_researcher_check

# List all available tools
npx exa-mcp-server --list-tools
```

### Configuring Claude Desktop

To configure Claude Desktop to use Exa MCP:
1. **Enable Developer Mode in Claude Desktop**
	- Open Claude Desktop
	- Click on the top-left menu
	- Enable Developer Mode
2. **Open the Configuration File**
	- After enabling Developer Mode, go to Settings
	- Navigate to the Developer Option
	- Click “Edit Config” to open the configuration file
	Alternatively, you can open it directly:**macOS:**
	```
	code ~/Library/Application\ Support/Claude/claude_desktop_config.json
	```
	**Windows:**
	```
	code %APPDATA%\Claude\claude_desktop_config.json
	```
3. **Add Exa MCP Configuration** Add the following to your configuration:
	```
	{
	  "mcpServers": {
	    "exa": {
	      "command": "npx",
	      "args": [
	        "-y",
	       "exa-mcp-server"
	       ],
	      "env": {
	        "EXA_API_KEY": "your-api-key-here"
	      }
	    }
	  }
	}
	```
	Replace `your-api-key-here` with your actual Exa API key.
4. **Enabling Specific Tools** To enable only specific tools:
	```
	{
	  "mcpServers": {
	    "exa": {
	      "command": "npx",
	      "args": [
	        "-y",
	        "exa-mcp-server",
	        "--tools=web_search"
	      ],
	      "env": {
	        "EXA_API_KEY": "your-api-key-here"
	      }
	    }
	  }
	}
	```
	To enable deep researcher tools:
	```
	{
	  "mcpServers": {
	    "exa": {
	      "command": "npx",
	      "args": [
	        "-y",
	        "exa-mcp-server",
	        "--tools=deep_researcher_start,deep_researcher_check"
	      ],
	      "env": {
	        "EXA_API_KEY": "your-api-key-here"
	      }
	    }
	  }
	}
	```
5. **Restart Claude Desktop**
	- Completely quit Claude Desktop (not just close the window)
	- Start Claude Desktop again
	- Look for the 🔌 icon to verify the Exa server is connected

### Troubleshooting

#### Common Issues

1. **Server Not Found**
	- Ensure the npm package is correctly installed
2. **API Key Issues**
	- Confirm your EXA_API_KEY is valid
	- Make sure there are no spaces or quotes around the API key
3. **Connection Problems**
	- Restart Claude Desktop completely

### Additional Resources

For more information, visit the [Exa MCP Server GitHub repository](https://github.com/exa-labs/exa-mcp-server/).

[Quickstart](https://docs.exa.ai/reference/quickstart) [Search](https://docs.exa.ai/reference/search)

## Additional Resources and Links

- [TypeScript SDK Specification](https://docs.exa.ai/sdks/typescript-sdk-specification)
- [Livecrawling Contents](https://docs.exa.ai/reference/livecrawling-contents)
- [Migrating from Bing](https://docs.exa.ai/reference/migrating-from-bing)
- [OpenAI Tool Calling](https://docs.exa.ai/reference/openai-tool-calling)
- [Vercel AI SDK](https://docs.exa.ai/reference/vercel)
- [Exa MCP GitHub](https://github.com/exa-labs/exa-mcp-server/)

This master note serves as a self-contained reference. If you need updates or expansions, link back to originals or query @vault for more context!
```
