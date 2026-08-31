# Chapter 6: Measuring your endpoint

Scripts from the recorded benchmarking session (one NVIDIA T4, vLLM 0.24.0,
Qwen/Qwen3-4B, 600 input / 150 output tokens per request).

Start the endpoint first:

```
vllm serve Qwen/Qwen3-4B --dtype half --max-model-len 8192
```

Then run a warm-up and discard it, as section 6.2 explains:

```
vllm bench serve --model Qwen/Qwen3-4B --dataset-name random \
  --random-input-len 600 --random-output-len 150 \
  --num-prompts 10 --request-rate 1
```

| Script | What it does | Recorded result |
|---|---|---|
| `lab-6-1-load-sweep.py` | Sweeps offered load at 0.5, 1, 2, 4 req/s | Knee between 1 and 2 req/s; throughput flattens at ~160 tok/s |
| `lab-6-2-metrics-delta.py` | Before-and-after counter delta (listing 6.1) | Per-experiment footprint instead of cumulative totals |
| `lab-6-3-gauge-monitor.py` | Samples running / waiting / KV / GPU each second | Queue forms only after KV pins at 0.99; preemptions 0 |
| `lab-6-4-prefix-layout.py` | Shared-first vs shared-buried prompt layout | 0.182 s vs 1.112 s TTFT p50; 93% vs 0% tokens cached |
| `lab-6-5-single-stream.sh` | Single-stream decode with `--max-concurrency 1` | 40.7 ms TPOT p50, about 24.6 tok/s |

Requirements: `pip install vllm==0.24.0 openai numpy requests` (pandas optional,
for the monitor summary).

Your numbers will differ. The point of these scripts is the method, not the values:
measure your own workload shape, then compare runs that changed one variable at a time.
