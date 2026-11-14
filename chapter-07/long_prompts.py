"""Chapter 7, lab 7.5: 4,000-token prompts on one T4.

Compares the 16-bit 4B model and the 8B AWQ model with long prompts:
five requests one at a time, then twenty requests at 0.1 req/s.
"""
import os
import subprocess
from lab_harness import start_server, KEEP

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def bench_long(model, args, out):
    cmd = ["vllm", "bench", "serve", "--model", model,
           "--dataset-name", "random",
           "--random-input-len", "4000", "--random-output-len", "150",
           "--temperature", "0",
           "--percentile-metrics", "ttft,tpot,e2el",
           "--metric-percentiles", "50,95"] + args
    r = subprocess.run(cmd, capture_output=True, text=True)
    open(out, "w").write(r.stdout)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise RuntimeError("benchmark failed")
    lines = [l for l in r.stdout.splitlines() if l.startswith(KEEP)]
    print("\n".join(lines), flush=True)


for model, tag in [("Qwen/Qwen3-4B", "4b-fp16"),
                   ("Qwen/Qwen3-8B-AWQ", "8b-awq")]:
    start_server(model)
    print(f"\n===== {tag}: long, single stream =====", flush=True)
    bench_long(model, ["--num-prompts", "5", "--max-concurrency", "1"],
               f"{tag}_long_single.txt")
    print(f"\n===== {tag}: long, 0.1 req/s =====", flush=True)
    bench_long(model, ["--num-prompts", "20", "--request-rate", "0.1",
                       "--goodput", "ttft:2000", "tpot:200"],
               f"{tag}_long_r0.1.txt")
