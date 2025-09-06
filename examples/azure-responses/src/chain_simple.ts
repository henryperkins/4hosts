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

  const r2 = await client.responses.create({
    model,
    previous_response_id: r1.id, // preserves full server-side context
    input: [{ role: "user", content: "Now give a 3-point checklist for interviews." }],
  });

  console.log(r2.output_text);
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
