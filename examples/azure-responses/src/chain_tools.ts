import OpenAI from "openai";

const endpoint = process.env.AZURE_OPENAI_ENDPOINT!;
const apiVersion = process.env.AZURE_OPENAI_API_VERSION; // e.g., "preview"
const baseURL = `${endpoint}/openai/v1${apiVersion ? `?api-version=${apiVersion}` : ""}`;

const client = new OpenAI({
  apiKey: process.env.AZURE_OPENAI_API_KEY!,
  baseURL,
});

const model = process.env.AZURE_OPENAI_DEPLOYMENT!; // default o3

type Weather = { location: string };
async function getWeather({ location }: Weather) {
  // In real usage, call a weather API here.
  return { temperatureF: 72, condition: "clear", location };
}

(async () => {
  // 1) model decides to call a function
  const step1 = await client.responses.create({
    model,
    input: [{ role: "user", content: "What's the weather in Chicago?" }],
    tools: [
      {
        type: "function",
        name: "get_weather",
        description: "Get the current weather for a location",
        strict: true,
        parameters: {
          type: "object",
          properties: { location: { type: "string" } },
          required: ["location"],
          additionalProperties: false,
        },
      },
    ],
  });

  // 2) execute function(s) and send outputs back, chained via previous_response_id
  const outputs: any[] = [];
  for (const item of step1.output ?? []) {
    if (item.type === "function_call" && item.name === "get_weather") {
      // arguments is a JSON string
      const args = JSON.parse((item as any).arguments ?? "{}");
      const result = await getWeather(args);
      outputs.push({
        type: "function_call_output",
        call_id: item.call_id,
        output: JSON.stringify(result),
      });
    }
  }

  const step2 = await client.responses.create({
    model,
    previous_response_id: step1.id,
    input: outputs,
  });

  console.log(step2.output_text);
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
