import os, time
from openai import OpenAI

endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
api_version = os.getenv("AZURE_OPENAI_API_VERSION")  # e.g., "preview"
base_url = f"{endpoint}/openai/v1" + (f"?api-version={api_version}" if api_version else "")

client = OpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"], base_url=base_url)

model = os.environ["AZURE_OPENAI_DEPLOYMENT"]  # default o3

r = client.responses.create(
    model=model,
    input="Write an unusually long limerick about compilers.",
    background=True,
    store=True,
)

while r.status in {"queued", "in_progress"}:
    time.sleep(1.5)
    r = client.responses.retrieve(r.id)

print(r.status)
print((r.output_text or "")[:400])
