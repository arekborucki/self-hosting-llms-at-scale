"""Chapter 6, section 6.4: one setting at a time against the baseline.

Restarts the server with each configuration and measures the same
workload at 2 offered requests per second (80 requests, 600 in /
150 out), with goodput against TTFT < 2,000 ms and TPOT < 200 ms.
Run it with no other vLLM server on the GPU.

Recorded on one T4, vLLM 0.24.0, Qwen/Qwen3-4B:
    default               throughput 1.05, goodput 0.01, TPOT p95 310 ms
    --max-num-seqs 32     throughput 1.01, goodput 0.47, TPOT p95 193 ms
    --max-num-seqs 16     throughput 0.89, goodput 0.18, TPOT p95 117 ms
    --gpu-memory-utilization 0.80
                          KV cache 27,744 tokens, goodput 0.10, TPOT p95 236 ms
    --gpu-memory-utilization 0.95
                          server did not start (CUDA out of memory)
In the same session the default 0.92 reported 40,464 tokens of KV cache.
"""
import subprocess
import time

import requests

CONFIGS = [
    [],
    ["--max-num-seqs", "32"],
    ["--max-num-seqs", "16"],
    ["--gpu-memory-utilization", "0.80"],
    ["--gpu-memory-utilization", "0.95"],
]


def start(extra):
    subprocess.run("pkill -f 'vllm serve'", shell=True)
    time.sleep(15)
    server = subprocess.Popen(
        ["vllm", "serve", "Qwen/Qwen3-4B",
         "--dtype", "half", "--max-model-len", "8192"] + extra,
        stdout=open("vllm.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(120):
        if server.poll() is not None:
            return False
        try:
            if requests.get("http://localhost:8000/health",
                            timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(5)
    return False


def bench(args):
    subprocess.run([
        "vllm", "bench", "serve",
        "--model", "Qwen/Qwen3-4B",
        "--dataset-name", "random",
        "--random-input-len", "600",
        "--random-output-len", "150",
        "--temperature", "0",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,95",
    ] + args)


for extra in CONFIGS:
    name = " ".join(extra) or "default"
    print(f"\n=== {name} ===", flush=True)
    if not start(extra):
        print("server did not start; see vllm.log")
        continue
    for line in open("vllm.log"):
        if "KV cache size" in line:
            print(line.strip())
    bench(["--num-prompts", "10", "--request-rate", "1"])   # warm-up
    bench(["--num-prompts", "80", "--request-rate", "2",
           "--goodput", "ttft:2000", "tpot:200"])
