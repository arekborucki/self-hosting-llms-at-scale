---
name: llm-endpoint-benchmarking
description: Plans and runs a reliable benchmark of an LLM serving endpoint, finds its capacity limit, sets a service level objective, and tunes the server one setting at a time. Use this skill whenever someone wants to load test an LLM endpoint, measure latency or throughput, use vllm bench serve, compare configurations, find how many users a server supports, or define an SLO for an LLM service.
---

# Benchmarking and tuning an LLM endpoint

Based on chapter 6 of *Self-Hosting LLMs at Scale* (Manning).

## Metrics

- **TTFT**: time to first token, what users feel as waiting.
- **TPOT**: time per output token, the reading speed of the answer. ITL shows the stutter between individual tokens.
- **E2EL**: end-to-end latency, the main metric for batch jobs and a diagnostic one for streaming chat.
- **Throughput**: requests and tokens per second for the whole system.
- **Goodput**: completed requests per second that met the latency targets.

Report percentiles (p50 and p95), never only averages.

## Method

1. Define the **workload shape**: input length, output length, and arrival pattern of the real traffic. Benchmark each very different shape separately.
2. Run a short **warm-up** and discard it.
3. Know whether the test prompts share prefixes; random prompts keep prefix caching out of the result.
4. Pin `--temperature 0` and the engine version, and change one variable per comparison.
5. **Sweep** the offered rate (for example 0.5, 1, 2, 4 requests per second) and find the **knee**, where throughput flattens and TTFT p95 rises sharply. Concurrency is a result of load, not a client setting.
6. Watch `num_requests_running`, `num_requests_waiting`, `kv_cache_usage_perc`, and preemptions during the run.

```bash
vllm bench serve --model <model-id> --dataset-name random \
  --random-input-len 600 --random-output-len 150 \
  --num-prompts 80 --request-rate 1 --temperature 0 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95 \
  --goodput ttft:2000 tpot:200
```

## Service level objective

Write the objective first, from user needs, then test against it. State the metric and percentile, the offered load, the workload shape, and the hardware and versions. Do not set the target just above whatever the benchmark already shows.

## Tuning one setting at a time

- `--max-num-seqs`: fewer sequences per step lowers TPOT but raises queueing and TTFT.
- `--max-num-batched-tokens`: the token budget per step and the prefill chunk size.
- `--gpu-memory-utilization`: the share of GPU memory vLLM uses, which sets the KV cache size. Too high can fail at startup.
- Prompt layout: shared content first, so prefix caching can reuse it.

Rerun the same benchmark after each change and compare with the baseline.

## Output

A short report: workload shape, configuration and versions, a results table (offered rate, achieved rate, TTFT p50/p95, TPOT p50/p95, goodput), the knee, whether the objective is met, and the next setting to try.
