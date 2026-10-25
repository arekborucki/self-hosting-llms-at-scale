"""Chapter 6, listing 6.8: short questions and long contracts together.

Runs two benchmarks at the same time against one server. Run it once
with the default server and once after restarting it with
--max-num-batched-tokens 8192, then compare short.txt in both runs.

Recorded on one T4, vLLM 0.24.0, Qwen/Qwen3-4B (2,048 and 8,192 gave
the same results within noise):
    short questions  TTFT p50 0.53 s, TTFT p95 8.5-9.1 s,
                     TPOT p95 186 ms, ITL p95 91 ms
    long contracts   TTFT p50 about 15.5 s (4 requests per run)
"""
import subprocess


def bench(input_len, rate, n, out):
    return subprocess.Popen([
        "vllm", "bench", "serve",
        "--model", "Qwen/Qwen3-4B",
        "--dataset-name", "random",
        "--random-input-len", str(input_len),
        "--random-output-len", "150",
        "--num-prompts", str(n),
        "--request-rate", str(rate),
        "--temperature", "0",
        "--percentile-metrics", "ttft,tpot,itl",
        "--metric-percentiles", "50,95",
    ], stdout=open(out, "w"))


short = bench(600, 0.2, 40, "short.txt")
long_ = bench(6000, 0.02, 4, "long.txt")
short.wait()
long_.wait()
print(open("short.txt").read())
print(open("long.txt").read())
