"""Chapter 6, listing 6.7: a load sweep with 4,000-token prompts.

Recorded on one T4, vLLM 0.24.0, Qwen/Qwen3-4B (150-token answers):
    0.05 req/s -> achieved 0.05, TTFT p50   7.5 s
    0.25 req/s -> achieved 0.10, TTFT p50  77.7 s
    0.5  req/s -> achieved 0.10, TTFT p50 163 s
    1    req/s -> achieved 0.10, TTFT p50 268 s
The card completes about 0.10 requests per second at this shape.
"""
import subprocess

for rate, n in [("0.05", 20), ("0.25", 30), ("0.5", 40), ("1", 60)]:
    print(f"=== rate {rate} req/s ===", flush=True)
    subprocess.run([
        "vllm", "bench", "serve",
        "--model", "Qwen/Qwen3-4B",
        "--dataset-name", "random",
        "--random-input-len", "4000",
        "--random-output-len", "150",
        "--num-prompts", str(n),
        "--request-rate", rate,
        "--temperature", "0",
        "--metric-percentiles", "50,95",
    ])
