"""Mini-lab 5.4: prefix caching, fresh vs repeated prompt.

Sends the same long prompt twice. The first send pays full
prefill; the repeat hits the prefix cache. Recorded on one T4,
vLLM 0.24.0, Qwen/Qwen3-4B:
  fresh ~1.91 s to first token, repeat 0.16-0.18 s (11x).
Check the server counters afterwards:
  curl -s localhost:8000/metrics | grep prefix_cache
"""
import time

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")


def ttft(prompt):
    start = time.perf_counter()
    stream = client.chat.completions.create(
        model="Qwen/Qwen3-4B",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=32,
        temperature=0,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            return round(time.perf_counter() - start, 3)


long_prompt = (
    "Summarize the key obligations in this clause. "
    + "The Supplier shall maintain in confidence all "
      "non-public information disclosed by the Customer. " * 120
)

print("fresh: ", ttft(long_prompt), "s")
print("repeat:", ttft(long_prompt), "s")
print("repeat:", ttft(long_prompt), "s")
