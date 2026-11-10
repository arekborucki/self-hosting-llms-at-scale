---
name: vllm-performance-diagnosis
description: Diagnoses why a vLLM or similar inference server is slow, stalls, or serves fewer users than expected, using the mechanics of prefill, decode, continuous batching, the KV cache, and prefix caching. Use this skill whenever someone reports a slow LLM endpoint, long time to first token, stuttering streaming, a growing queue, preemptions, or asks how to make vLLM faster, even if they do not mention any of these terms.
---

# Diagnosing a slow inference server

Based on chapter 5 of *Self-Hosting LLMs at Scale* (Manning).

## How a request spends its time

- **Prefill** processes the whole prompt at once. It is compute-heavy and sets most of the time to first token (TTFT) on a lightly loaded server. Long prompts mean long prefill.
- **Decode** generates one token per step and reads all weights and the request's KV cache each time. It is limited by memory bandwidth and sets the time per output token (TPOT).
- Under load, **queueing** adds to TTFT before prefill even starts.

## Diagnosis by symptom

1. **Long silence before the first word, then normal speed**: long prompts in prefill. Check prompt length in tokens (the server's tokenize endpoint or `usage`). Options: shorter prompts, prefix caching with shared content at the front, a faster GPU for prefill.
2. **First word slow only at busy times**: queueing. Check `vllm:num_requests_waiting`. If it rises when `vllm:kv_cache_usage_perc` is near 1.0, the KV cache is full. Options: more KV memory, shorter `--max-model-len`, quantized weights, more GPUs.
3. **Waiting rises while the KV cache is not full**: a scheduler or admission limit, such as `--max-num-seqs`.
4. **Streaming stutters for everyone when one long prompt arrives**: long prefill sharing steps with other requests' decode. Chunked prefill splits it into chunks of at most `--max-num-batched-tokens`; the chunk size trades the long request's TTFT against everyone else's smoothness.
5. **Preemptions climbing** (`vllm:num_preemptions_total`): the server admitted more work than the KV cache can hold, and evicted requests are recomputed from the start. Options: lower admitted concurrency, more KV memory, shorter contexts.
6. **Repeated system prompts but no speedup**: prefix caching only reuses identical tokens from the start of the prompt up to the first difference. Put shared content first and per-request content last; check prefix cache hits against queries.

## Rules

- Change one setting at a time and measure before and after (use the llm-endpoint-benchmarking skill).
- Read counters as before-and-after differences, not as totals since startup.
- Record the vLLM version; defaults change between releases.

## Output

State the most likely cause, the metric that supports it, what to change first, and how to confirm the change helped.
