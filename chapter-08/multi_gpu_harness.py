"""Chapter 8: helpers for the multi-GPU labs on two T4 cards.

Uses lab_harness.py from chapter-07 for start_server, show_log, and bench.
Tested on a Kaggle notebook with two Tesla T4 GPUs and vLLM 0.24.0.
"""
import os
import subprocess
import time

import requests

M = "Qwen/Qwen3-8B-AWQ"


def gpu_clean():
    """Listing 8.2: stop every vLLM process that still holds GPU memory."""
    for pat in ["vllm serve", "VLLM::", "EngineCore",
                "rr_proxy.py", "toy_proxy_server.py"]:
        subprocess.run(["pkill", "-9", "-f", pat])
    pids = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.split()
    for p in pids:
        subprocess.run(["kill", "-9", p])
    time.sleep(15)
    print(subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv"],
        capture_output=True, text=True).stdout)


def launch(gpu, port, log, extra=[], env_extra={}, model=M):
    """Start one vLLM server on one GPU and port."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), **env_extra)
    cmd = ["vllm", "serve", model, "--port", str(port), "--dtype", "half",
           "--max-model-len", "8192", "--gpu-memory-utilization", "0.80"] + extra
    return subprocess.Popen(cmd, env=env, stdout=open(log, "w"),
                            stderr=subprocess.STDOUT)


def wait(port, proc, log, limit=180):
    """Wait until the server on this port answers /health."""
    for _ in range(limit):
        if proc.poll() is not None:
            print(f"port {port} exited")
            print("".join(open(log).readlines()[-15:]))
            return False
        try:
            if requests.get(f"http://localhost:{port}/health",
                            timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    print(f"port {port} timeout")
    return False


def mixed(tag, model=M, seed_short="2001", seed_long="2002"):
    """Short questions at 1 req/s and long contracts at 0.05 req/s together.

    Each benchmark has its own seed so repeated prompts do not hit
    the prefix cache.
    """
    def run(inlen, n, rate, seed, out):
        return subprocess.Popen(
            ["vllm", "bench", "serve", "--model", model,
             "--dataset-name", "random", "--random-input-len", inlen,
             "--random-output-len", "150", "--temperature", "0",
             "--num-prompts", n, "--request-rate", rate, "--seed", seed,
             "--percentile-metrics", "ttft,tpot,itl,e2el",
             "--metric-percentiles", "50,95"],
            stdout=open(out, "w"), stderr=subprocess.STDOUT)
    s = run("600", "180", "1", seed_short, f"{tag}_mixed_short.txt")
    l = run("4000", "9", "0.05", seed_long, f"{tag}_mixed_long.txt")
    s.wait()
    l.wait()
    keep = ("Request throughput", "P50 TTFT", "P95 TTFT", "P50 TPOT",
            "P95 TPOT", "P50 ITL", "P95 ITL", "Successful")
    for part in ["short", "long"]:
        print(f"\n===== {tag}: {part} =====")
        for line in open(f"{tag}_mixed_{part}.txt").read().splitlines():
            if line.startswith(keep):
                print(line)
