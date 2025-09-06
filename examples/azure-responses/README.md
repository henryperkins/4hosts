Azure OpenAI Responses API — o3 + v1 preview (code-first)

This folder contains minimal, runnable examples that demonstrate:
- Background jobs with polling and cancel (Node)
- Background jobs with streaming + resume via cursor (Node)
- Response chaining, with and without tools/functions (Node)
- A quick Python polling sample

Requirements
- Node 20+ for the TypeScript samples
- Python 3.11+ for the Python sample
- Azure OpenAI resource with a Responses-capable deployment (e.g., `o3` reasoning)

Environment
- Copy `.env.sample` to `.env` or export the variables in your shell. These examples default to the `o3` reasoning model and support the Azure v1 preview style.

  export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com"
  export AZURE_OPENAI_API_KEY="<key>"
  export AZURE_OPENAI_DEPLOYMENT="o3"
  # Optional if your resource requires explicit preview flag:
  # export AZURE_OPENAI_API_VERSION="preview"

Install

  npm i

Run
- Background polling:

  npm run dev:background

- Background streaming + resume:

  npm run dev:background:stream

- Simple chaining with previous_response_id:

  npm run dev:chain

- Chaining with tools (submit function outputs next turn):

  npm run dev:chain:tools

Python quickie

  python3 background.py

Notes & Gotchas
- v1 preview: All clients point at `${AZURE_OPENAI_ENDPOINT}/openai/v1`. If your resource expects an explicit preview flag, set `AZURE_OPENAI_API_VERSION=preview` and the examples will append `?api-version=preview` to requests (including stream resume URLs).
- Background requires `store: true`; otherwise Azure will reject background jobs.
- Streaming + background is supported; expect higher TTFT. Use `sequence_number` and resume with `?starting_after=<cursor>`.
- Chaining: prefer `previous_response_id` for server-side context. Or manually include prior outputs.
- Tools with Responses: submit `function_call_output` items on the next `responses.create`, matching each `call_id`.
