import json
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1",
                api_key="x")
MODEL = "Qwen/Qwen3-4B"

SYSTEM = ("You are a helpful assistant for a law firm. "
          "Be concise. If the answer is not in the "
          "provided text or is uncertain, say so.")

results = []
for line in open("golden_set.jsonl"):
    item = json.loads(line)
    start = time.perf_counter()
    r = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": item["prompt"]},
        ],
        temperature=0,
        max_tokens=512,
        extra_body={"chat_template_kwargs":
                    {"enable_thinking": False}},
    )
    results.append({
        "id": item["id"],
        "prompt": item["prompt"],
        "criteria": item["criteria"],
        "answer": r.choices[0].message.content,
        "tokens": r.usage.completion_tokens,
        "seconds": round(time.perf_counter() - start, 2),
    })

out = f"answers_{MODEL.split('/')[-1]}.jsonl"
with open(out, "w") as f:
    for row in results:
        f.write(json.dumps(row) + "\n")
print(f"{len(results)} answers written to {out}")
