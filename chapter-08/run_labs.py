"""Chapter 8: labs 8.1 to 8.5 on the two T4 cards of a Kaggle notebook.

Run in a notebook cell with:  %run run_labs.py   (about two hours)
Needs lab_harness.py from chapter-07 in the same folder, and
`pip install vllm==0.24.0 openai nixl`.
"""
import os
import subprocess
import time
import urllib.request

import requests

from lab_harness import start_server, show_log, bench
from multi_gpu_harness import M, gpu_clean, launch, wait, mixed

RATES = [("0.5", "40"), ("1", "80"), ("2", "80"), ("3", "120")]


def sweep(model, tag, extra_rates=RATES):
    bench(model, ["--num-prompts", "10", "--request-rate", "1",
                  "--seed", "11"], "warmup.txt")
    bench(model, ["--num-prompts", "15", "--max-concurrency", "1",
                  "--seed", "12"], f"{tag}_single.txt")
    for i, (rate, n) in enumerate(extra_rates):
        bench(model, ["--num-prompts", n, "--request-rate", rate,
                      "--goodput", "ttft:2000", "tpot:200",
                      "--seed", str(20 + i)], f"{tag}_r{rate}.txt")


def long_single(model, tag):
    subprocess.run(["vllm", "bench", "serve", "--model", model,
                    "--dataset-name", "random", "--random-input-len", "4000",
                    "--random-output-len", "150", "--temperature", "0",
                    "--num-prompts", "5", "--max-concurrency", "1",
                    "--seed", "31", "--percentile-metrics", "ttft,tpot",
                    "--metric-percentiles", "50,95"],
                   stdout=open(f"{tag}_long_single.txt", "w"))


# Lab 8.1: one replica against two (--data-parallel-size 2)
gpu_clean(); os.environ["CUDA_VISIBLE_DEVICES"] = "0"
start_server(M); show_log(); sweep(M, "awq-1rep")
gpu_clean(); os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
start_server(M, ["--data-parallel-size", "2", "--gpu-memory-utilization", "0.80"])
show_log(); sweep(M, "awq-dp2"); mixed("awq-dp2")

# Lab 8.1b: two servers behind the router in listing 8.1
gpu_clean()
a = launch(0, 8001, "srv0.log"); b = launch(1, 8002, "srv1.log")
wait(8001, a, "srv0.log") and wait(8002, b, "srv1.log")
subprocess.Popen(["python3", "rr_proxy.py"], stdout=open("proxy.log", "w"),
                 stderr=subprocess.STDOUT)
time.sleep(8); sweep(M, "router"); mixed("router")

# Labs 8.2, 8.2b, and 8.3: tensor and pipeline parallelism
for model, tag, extra in [
        ("Qwen/Qwen3-8B", "tp2", ["--tensor-parallel-size", "2"]),
        ("Qwen/Qwen3-8B", "pp2", ["--pipeline-parallel-size", "2"]),
        (M, "tp2awq", ["--tensor-parallel-size", "2"])]:
    gpu_clean()
    start_server(model, extra + ["--gpu-memory-utilization", "0.80"])
    show_log(); sweep(model, tag); long_single(model, tag)
    if model == M:
        mixed(tag)

# Lab 8.5: disaggregated prefill and decode with the NixlConnector
gpu_clean()
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/vllm-project/vllm/v0.24.0/"
    "tests/v1/kv_connector/nixl_integration/toy_proxy_server.py",
    "toy_proxy_server.py")
kv = ["--kv-transfer-config",
      '{"kv_connector":"NixlConnector","kv_role":"kv_both"}']
p = launch(0, 8100, "prefill.log", kv, {"UCX_NET_DEVICES": "all",
           "UCX_PROTO_INFO": "y", "VLLM_NIXL_SIDE_CHANNEL_PORT": "5600"})
d = launch(1, 8200, "decode.log", kv, {"UCX_NET_DEVICES": "all",
           "UCX_PROTO_INFO": "y", "VLLM_NIXL_SIDE_CHANNEL_PORT": "5601"})
wait(8100, p, "prefill.log") and wait(8200, d, "decode.log")
subprocess.Popen(["python3", "toy_proxy_server.py", "--port", "8000",
                  "--prefiller-hosts", "localhost", "--prefiller-ports", "8100",
                  "--decoder-hosts", "localhost", "--decoder-ports", "8200"],
                 stdout=open("proxy.log", "w"), stderr=subprocess.STDOUT)
time.sleep(10); mixed("disagg")
gpu_clean()
