# Chapter 6: Benchmarking and tuning single-node performance

Scripts from the recorded benchmarking sessions (one NVIDIA T4, vLLM 0.24.0,
Qwen/Qwen3-4B). Most use 600 input and 150 output tokens per request; the
long-context scripts use longer prompts.

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
| `lab-6-6-goodput.py` | Goodput at 1 and 2 req/s (listing 6.4) | 0.85 and 0.01 req/s |
| `lab-6-7-tuning.py` | `--max-num-seqs` and `--gpu-memory-utilization`, one at a time | 32 sequences gives the best goodput at 2 req/s |
| `lab-6-8-prompt-length.py` | Single-stream TTFT at 600 to 7,000 prompt tokens (listing 6.6) | 0.38 s to 20.6 s |
| `lab-6-9-long-sweep.py` | Load sweep with 4,000-token prompts (listing 6.7) | About 0.10 req/s capacity |
| `lab-6-10-mixed.py` | Short questions and long contracts together (listing 6.8) | Short-question TTFT p95 above 8 s |

Requirements: `pip install vllm==0.24.0 openai numpy requests` (pandas optional,
for the monitor summary).

Your numbers will differ. Measure your own workload shape, and compare runs
that change one variable at a time.
