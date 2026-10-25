"""Chapter 6, LAB 1: a four-level load sweep with vllm bench serve.

Runs the same workload shape at rising offered rates and saves one
JSON result file per level. Start the server first:

    vllm serve Qwen/Qwen3-4B --dtype half --max-model-len 8192

Recorded on one T4, vLLM 0.24.0, Qwen/Qwen3-4B (600 in / 150 out):
    0.5 req/s -> achieved 0.46,  69 tok/s, TTFT p95  1.31 s
    1   req/s -> achieved 0.85, 127 tok/s, TTFT p95  1.40 s
    2   req/s -> achieved 1.04, 157 tok/s, TTFT p95 29.7 s
    4   req/s -> achieved 1.07, 160 tok/s, TTFT p95 79.8 s
Knee between 1 and 2 offered requests per second.

The recorded sweep ran without --temperature 0; this script adds it
so that reruns are reproducible.

Run a warm-up first and discard it:
    vllm bench serve --model Qwen/Qwen3-4B --dataset-name random \
      --random-input-len 600 --random-output-len 150 \
      --num-prompts 10 --request-rate 1
"""
import subprocess

MODEL = "Qwen/Qwen3-4B"
LEVELS = [("0.5", 40), ("1", 80), ("2", 120), ("4", 160)]

for rate, num_prompts in LEVELS:
    print(f"=== rate {rate} req/s ===", flush=True)
    subprocess.run([
        "vllm", "bench", "serve",
        "--model", MODEL,
        "--dataset-name", "random",
        "--random-input-len", "600",
        "--random-output-len", "150",
        "--num-prompts", str(num_prompts),
        "--request-rate", rate,
        "--temperature", "0",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,95",
        "--save-result",
        "--result-filename", f"sweep_r{rate}.json",
    ])
