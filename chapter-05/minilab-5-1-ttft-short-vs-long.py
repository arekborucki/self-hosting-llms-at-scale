"""Mini-lab 5.1: time to first token, short vs long prompt.

Each try uses a unique prompt (random prefix) so prefix caching
cannot hide the prefill cost. See minilab-5-4 for the cached case.
Recorded on one T4, vLLM 0.24.0, Qwen/Qwen3-4B:
  short: 0.066-0.139 s   long (~1,570 words): ~1.91 s
"""
import time
import uuid

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


base = (
    "The Supplier shall maintain in confidence all "
    "non-public information disclosed by the Customer. " * 120
)

for name, body in [("short", "What is an indemnification clause?"),
                   ("long", base)]:
    times = []
    for _ in range(3):
        unique = f"[case {uuid.uuid4().hex[:8]}] Summarize: {body}"
        times.append(ttft(unique))
    print(f"{name}: {times} s")
