"""Chapter 6, listing 6.3: sample the three gauges during a load run.

Writes monitor.csv with one row per second: running, waiting,
KV-cache fraction, GPU utilization. Run a benchmark alongside it.

Recorded trace (rate 2, one T4): running ramps 8 -> 22 -> 44 -> 55
while KV climbs 0.14 -> 0.99; waiting rises only AFTER the cache
pins at 0.99, peaking at 48; preemptions stay at zero.

The recorded run used no --temperature 0; this script adds it so
that reruns are reproducible.
"""
import os
import subprocess
import time

if os.path.exists("monitor.csv"):
    os.remove("monitor.csv")

SAMPLER = """
while true; do
  ts=$(date +%s)
  m=$(curl -s localhost:8000/metrics)
  run=$(echo "$m" | grep '^vllm:num_requests_running' | awk '{print $2}')
  wait=$(echo "$m" | grep '^vllm:num_requests_waiting{' | awk '{print $2}')
  kv=$(echo "$m" | grep '^vllm:kv_cache_usage_perc' | awk '{print $2}')
  gpu=$(nvidia-smi --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits | head -1)
  echo "$ts,$run,$wait,$kv,$gpu" >> monitor.csv
  sleep 1
done
"""

monitor = subprocess.Popen(["bash", "-c", SAMPLER])
time.sleep(2)

subprocess.run([
    "vllm", "bench", "serve",
    "--model", "Qwen/Qwen3-4B",
    "--dataset-name", "random",
    "--random-input-len", "600",
    "--random-output-len", "150",
    "--num-prompts", "120",
    "--request-rate", "2",
    "--temperature", "0",
    "--percentile-metrics", "ttft,tpot,itl,e2el",
    "--metric-percentiles", "50,95",
])

time.sleep(5)          # capture the drain after the last request
monitor.terminate()

try:
    import pandas as pd
    df = pd.read_csv("monitor.csv",
                     names=["ts", "running", "waiting", "kv", "gpu"])
    print(df.describe().loc[["max", "mean"]].round(2))
    print(df.iloc[::10].to_string(index=False))
except ImportError:
    print("monitor.csv written; install pandas for the summary")
