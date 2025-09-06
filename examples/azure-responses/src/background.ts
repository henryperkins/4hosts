import OpenAI from "openai";

const endpoint = process.env.AZURE_OPENAI_ENDPOINT!;
const apiVersion = process.env.AZURE_OPENAI_API_VERSION; // e.g., "preview"
const baseURL = `${endpoint}/openai/v1${apiVersion ? `?api-version=${apiVersion}` : ""}`;

const client = new OpenAI({
  apiKey: process.env.AZURE_OPENAI_API_KEY!,
  baseURL,
});

async function run() {
  if (!process.env.AZURE_OPENAI_DEPLOYMENT) throw new Error("AZURE_OPENAI_DEPLOYMENT is required");

  const created = await client.responses.create({
    model: process.env.AZURE_OPENAI_DEPLOYMENT, // default o3 via .env.sample
    input: "Generate 10 whimsical, distinct startup ideas with a 1‑line pitch each.",
    background: true,
    store: true, // required for background
  });

  console.log("job id:", created.id, "status:", created.status);

  let resp = created;
  while (resp.status === "queued" || resp.status === "in_progress") {
    await new Promise((r) => setTimeout(r, 1500));
    resp = await client.responses.retrieve(resp.id);
    process.stdout.write(`\rstatus: ${resp.status}          `);
  }
  console.log("\n");

  if (resp.status === "completed") {
    console.log("DONE:\n", resp.output_text);
  } else {
    console.error("FAILED:", resp.error);
  }

  // Example: cancel if still running (commented out)
  // await client.responses.cancel(created.id);
}

run().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
