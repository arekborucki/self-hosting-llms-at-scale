---
name: vllm-first-endpoint
description: Guides someone through serving an open-weight model with vLLM behind an OpenAI-compatible API, sending the first requests, and reading the server's metrics. Use this skill whenever someone wants to run vLLM, start a local or cloud LLM endpoint, call a self-hosted model with the OpenAI client, or check whether their vLLM server is healthy, even if they do not mention vLLM by name.
---

# Serving a first model with vLLM

Based on chapter 2 of *Self-Hosting LLMs at Scale* (Manning).

## Before starting

1. Estimate memory first (use the gpu-memory-sizing skill). Weights at 16-bit precision take about 2 bytes per parameter, and the KV cache needs room on top.
2. Pin the vLLM version you test with, and record it.

## Launch

```bash
vllm serve <model-id> --max-model-len <tokens>
```

- On GPUs without BF16 support (for example the NVIDIA T4), add `--dtype half`.
- Set `--max-model-len` to what the workload needs. It limits the longest conversation and affects how much KV cache each request can use.
- Wait for `Application startup complete.` in the log. Read the lines that report the KV cache size in tokens and the maximum concurrency at the chosen length.

## First requests

Use the OpenAI client and point it at the server:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
r = client.chat.completions.create(
    model="<model-id>",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    max_tokens=64,
    temperature=0,
)
print(r.choices[0].message.content, r.usage)
```

- Use `stream=True` for chat interfaces, so users see the first tokens early.
- Always set `max_tokens`; without a limit a single long answer can hold the server busy.
- Check `usage` in the response: it shows how many prompt and completion tokens the request really used, including tokens added by the chat template.
- For reasoning models, check the model card for template settings (for example a thinking mode) and pass them through `extra_body` if needed.

## Health check with metrics

Read `http://localhost:8000/metrics` and look at:

- `vllm:num_requests_running` and `vllm:num_requests_waiting`: work in progress and the queue
- `vllm:kv_cache_usage_perc`: how full the KV cache is
- `vllm:num_preemptions_total`: requests pushed out of the cache because memory ran out
- prompt and generation token counters, and prefix cache queries and hits

A healthy idle server shows no waiting requests and no preemptions. For performance problems, use the vllm-performance-diagnosis skill.
