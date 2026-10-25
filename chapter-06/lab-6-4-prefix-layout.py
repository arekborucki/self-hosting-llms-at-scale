"""Chapter 6, listing 6.5: what prompt layout costs.

Sends the same 24-request series twice with the same content in a
different order: shared block first, then shared block buried behind
a unique marker. Same tokens, different cache behavior.

Recorded on one T4, vLLM 0.24.0, Qwen/Qwen3-4B:
    shared-first : TTFT p50 0.182 s, 25,760 / 27,704 tokens cached
    shared-buried: TTFT p50 1.112 s,      0 / 27,700 tokens cached
6.1x from layout alone, at an identical token cost.

The recorded script used simple order-statistic indexing for the
percentiles; this script uses NumPy, as listing 6.5 does.
"""
import re
import time
import uuid

import numpy as np
import requests
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

PARA = ("Meridian Legal standard review policy: confidentiality, "
        "liability caps, governing law, notice periods, and "
        "termination rights must be checked in every contract. ")
SHARED = PARA * 40          # roughly 1,200 tokens of identical text


def ttft(content):
    start = time.perf_counter()
    stream = client.chat.completions.create(
        model="Qwen/Qwen3-4B",
        messages=[{"role": "user", "content": content}],
        max_tokens=32,
        temperature=0,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            return time.perf_counter() - start


def snapshot():
    text = requests.get("http://localhost:8000/metrics").text
    out = {}
    for name in ["prompt_tokens_total", "prefix_cache_hits_total"]:
        m = re.search(rf"^vllm:{name}(?:{{[^}}]*}})?\s+([0-9.e+]+)",
                      text, re.MULTILINE)
        out[name] = float(m.group(1)) if m else 0.0
    return out


def series(layout, n=24):
    before = snapshot()
    times = []
    for i in range(n):
        question = f"Is clause {i} about liability? One word."
        marker = f"[case {uuid.uuid4().hex[:8]}]"
        if layout == "shared-first":
            content = f"{SHARED}\n{marker} {question}"
        else:
            content = f"{marker} {SHARED}\n{question}"
        times.append(ttft(content))
    after = snapshot()
    p50, p95 = np.percentile(times, [50, 95])
    hits = after["prefix_cache_hits_total"] - before["prefix_cache_hits_total"]
    toks = after["prompt_tokens_total"] - before["prompt_tokens_total"]
    print(f"{layout}: TTFT p50={p50:.3f}s p95={p95:.3f}s "
          f"cache_hits={hits:.0f}/{toks:.0f}")


if __name__ == "__main__":
    series("shared-first")
    series("shared-buried")
