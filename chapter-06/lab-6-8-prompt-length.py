"""Chapter 6, listing 6.6: time to first token at four prompt lengths.

One request at a time, 150-token answers.

Recorded on one T4, vLLM 0.24.0, Qwen/Qwen3-4B:
    600 tokens   TTFT p50  0.38 s, TPOT p50 39 ms
    2,000        TTFT p50  1.88 s, TPOT p50 46 ms
    4,000        TTFT p50  5.15 s, TPOT p50 54 ms
    7,000        TTFT p50 20.6 s,  TPOT p50 67 ms
"""
import subprocess

for n in ["600", "2000", "4000", "7000"]:
    print(f"=== input {n} tokens ===", flush=True)
    subprocess.run([
        "vllm", "bench", "serve",
        "--model", "Qwen/Qwen3-4B",
        "--dataset-name", "random",
        "--random-input-len", n,
        "--random-output-len", "150",
        "--num-prompts", "10",
        "--max-concurrency", "1",
        "--temperature", "0",
        "--percentile-metrics", "ttft,tpot,e2el",
        "--metric-percentiles", "50,95",
    ])
