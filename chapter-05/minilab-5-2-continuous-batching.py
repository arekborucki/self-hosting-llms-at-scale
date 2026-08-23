"""Mini-lab 5.2: eight requests, sequential vs concurrent.

Continuous batching in two numbers. Recorded on one T4,
vLLM 0.24.0, Qwen/Qwen3-4B:
  sequential total 14.17 s, concurrent total 2.08 s,
  per-request latency almost unchanged.
"""
import concurrent.futures
import time
import uuid

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")


def one_request(i):
    p = f"[case {uuid.uuid4().hex[:8]}] In one sentence: risk {i} in M&A?"
    start = time.perf_counter()
    client.chat.completions.create(
        model="Qwen/Qwen3-4B",
        messages=[{"role": "user", "content": p}],
        max_tokens=64,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return round(time.perf_counter() - start, 2)


seq_start = time.perf_counter()
seq_times = [one_request(i) for i in range(8)]
seq_total = round(time.perf_counter() - seq_start, 2)

par_start = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    par_times = list(ex.map(one_request, range(8)))
par_total = round(time.perf_counter() - par_start, 2)

print(f"sequential: total {seq_total} s, per-request {seq_times}")
print(f"concurrent: total {par_total} s, per-request {par_times}")
