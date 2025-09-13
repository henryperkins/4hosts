import OpenAI from "openai";

const endpoint = process.env.AZURE_OPENAI_ENDPOINT!;
const apiVersion = process.env.AZURE_OPENAI_API_VERSION; // e.g., "preview"
const baseURL = `${endpoint}/openai/v1${apiVersion ? `?api-version=${apiVersion}` : ""}`;

const client = new OpenAI({
  apiKey: process.env.AZURE_OPENAI_API_KEY!,
  baseURL,
});

const model = process.env.AZURE_OPENAI_DEPLOYMENT!;

(async () => {
  const r1 = await client.responses.create({ model, input: "Summarize dropout vs. batchnorm tradeoffs." });

  const responseFormat = {
    type: "json_schema",
    json_schema: {
      name: "action_items",
      strict: true,
      schema: {
        type: "object",
        additionalProperties: false,
        properties: {
          action_items: {
            type: "array",
            description: "List of extracted action items.",
            items: {
              type: "object",
              additionalProperties: false,
              properties: {
                description: { type: "string", description: "Concise, verb-first task." },
                owner: { type: "string", description: "Responsible person or team." },
                due_date: { type: ["string", "null"], description: "YYYY-MM-DD or null." }
              },
              required: ["description", "owner", "due_date"]
            }
          }
        },
        required: ["action_items"]
      }
    }
  } as const;

  const r2 = await client.responses.create({
    model,
    previous_response_id: r1.id, // preserves full server-side context
    input: [{ role: "user", content: "Now give a 3-point checklist for interviews." }],
    response_format: responseFormat,
    max_output_tokens: 600,
  });

  console.log(r2.output_text);
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
