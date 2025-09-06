import OpenAI from "openai";

const endpoint = process.env.AZURE_OPENAI_ENDPOINT!;
const apiVersion = process.env.AZURE_OPENAI_API_VERSION; // e.g., "preview"
const apiKey = process.env.AZURE_OPENAI_API_KEY!;
const model = process.env.AZURE_OPENAI_DEPLOYMENT!; // default to o3

const baseURL = `${endpoint}/openai/v1${apiVersion ? `?api-version=${apiVersion}` : ""}`;
const client = new OpenAI({ apiKey, baseURL });

async function run() {
  // Kick off a background + stream job
  const stream = (await client.responses.create({
    model,
    input: "Write a long essay about the history of timekeeping.",
    background: true,
    stream: true,
    store: true,
  })) as any;

  let cursor: number | null = null;
  let responseId: string | null = null;

  for await (const event of stream) {
    if (event.type === "response.created") responseId = event.response?.id ?? responseId;
    if ("sequence_number" in event) cursor = event.sequence_number as number;
    if (event.type === "response.output_text.delta") process.stdout.write(event.delta);
  }

  // Demonstrate resume using the `starting_after` cursor.
  if (responseId && cursor != null) {
    console.log("\n--- resume from cursor:", cursor, "response:", responseId, "---\n");

    // Use native fetch to resume SSE stream and print only text deltas
    const url = `${endpoint}/openai/v1/responses/${responseId}?${apiVersion ? `api-version=${apiVersion}&` : ""}stream=true&starting_after=${cursor}`;
    const res = await fetch(url, {
      headers: {
        Authorization: `Bearer ${apiKey}`,
        Accept: "text/event-stream",
      },
    });

    // Simple SSE reader that extracts output_text deltas
    const reader = (res.body as any).getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const rawEvent = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const lines = rawEvent.split(/\r?\n/);
        let ev: string | null = null;
        let data = "";
        for (const line of lines) {
          if (line.startsWith("event:")) ev = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (ev === "response.output_text.delta") {
          try {
            const parsed = JSON.parse(data);
            if (parsed?.delta) process.stdout.write(parsed.delta);
          } catch {
            // ignore parse errors for non-JSON keepalives
          }
        }
      }
    }
  }
}

run().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
