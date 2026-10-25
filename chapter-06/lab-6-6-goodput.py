"""Chapter 6, listing 6.4: goodput against Meridian's objective.

Goodput counts only requests that meet the targets:
TTFT under 2,000 ms and TPOT under 200 ms.

Recorded on one T4, vLLM 0.24.0, Qwen/Qwen3-4B (600 in / 150 out,
80 requests per level):
    1 req/s -> throughput 0.85 req/s, goodput 0.85 req/s
    2 req/s -> throughput 1.05 req/s, goodput 0.01 req/s
"""
import subprocess

for rate in ["1", "2"]:
    print(f"=== rate {rate} req/s ===", flush=True)
    subprocess.run([
        "vllm", "bench", "serve",
        "--model", "Qwen/Qwen3-4B",
        "--dataset-name", "random",
        "--random-input-len", "600",
        "--random-output-len", "150",
        "--num-prompts", "80",
        "--request-rate", rate,
        "--temperature", "0",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,95",
        "--goodput", "ttft:2000", "tpot:200",
    ])
