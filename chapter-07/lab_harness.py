"""Chapter 7: helper functions for the quantization labs.

Starts a vLLM server, prints the startup log lines that matter for
quantization, runs the golden set from chapter 4, and runs the load
sweep from chapter 6. Used by labs 7.1 to 7.4 (listing 7.5 shows run_all).

Tested on a Kaggle notebook with Tesla T4 GPUs and vLLM 0.24.0.
"""
import json
import os
import subprocess
import time
import urllib.request

import requests
from openai import OpenAI

GOLDEN_URL = ("https://raw.githubusercontent.com/arekborucki/"
              "self-hosting-llms-at-scale/main/chapter-04/golden_set.jsonl")
KEEP = ("Request throughput", "Request goodput",
        "Output token throughput", "Peak concurrent",
        "P50 TTFT", "P95 TTFT", "P50 TPOT", "P95 TPOT",
        "Successful")
LOGKEYS = ["quantization=", "Model loading took", "KV cache size",
           "Maximum concurrency", "WARNING", "Marlin", "unquantized",
           "attention backend", "ValueError"]
SYSTEM = ("You are a helpful assistant for a law firm. "
          "Be concise. If the answer is not in the "
          "provided text or is uncertain, say so.")


def start_server(model, extra=[]):
    """Stop any running server and start vLLM with the chapter's settings."""
    subprocess.run("pkill -f 'vllm serve'", shell=True)
    time.sleep(10)
    cmd = ["vllm", "serve", model, "--dtype", "half",
           "--max-model-len", "8192"] + extra
    proc = subprocess.Popen(cmd, stdout=open("vllm.log", "w"),
                            stderr=subprocess.STDOUT)
    for _ in range(240):
        if proc.poll() is not None:
            raise RuntimeError("server exited; see vllm.log")
        try:
            r = requests.get("http://localhost:8000/health", timeout=2)
            if r.status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError("server did not start in time")


def show_log():
    """Print quantization method, kernel, memory, cache size, warnings."""
    for line in open("vllm.log"):
        if any(k in line for k in LOGKEYS):
            print(line.strip()[:250])


def golden(model, tag):
    """Run the chapter 4 golden set and save answers with finish_reason."""
    if not os.path.exists("golden_set.jsonl"):
        urllib.request.urlretrieve(GOLDEN_URL, "golden_set.jsonl")
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
    rows = []
    for line in open("golden_set.jsonl"):
        item = json.loads(line)
        t0 = time.perf_counter()
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": item["prompt"]}],
            temperature=0, max_tokens=512,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        rows.append({"id": item["id"], "criteria": item["criteria"],
                     "answer": r.choices[0].message.content,
                     "finish_reason": r.choices[0].finish_reason,
                     "served_model": r.model,
                     "tokens": r.usage.completion_tokens,
                     "seconds": round(time.perf_counter() - t0, 2)})
    with open(f"answers_{tag}.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    cut = [x["id"] for x in rows if x["finish_reason"] == "length"]
    print("cut off:", cut)


def bench(model, args, out):
    """Run vllm bench serve with the chapter 6 workload (600 in, 150 out)."""
    cmd = ["vllm", "bench", "serve", "--model", model,
           "--dataset-name", "random",
           "--random-input-len", "600", "--random-output-len", "150",
           "--temperature", "0",
           "--percentile-metrics", "ttft,tpot,itl,e2el",
           "--metric-percentiles", "50,95"] + args
    r = subprocess.run(cmd, capture_output=True, text=True)
    open(out, "w").write(r.stdout)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise RuntimeError("benchmark failed")
    lines = [l for l in r.stdout.splitlines() if l.startswith(KEEP)]
    print("\n".join(lines), flush=True)


def run_all(model, tag, extra=[]):
    """Listing 7.5: one configuration through the quality and capacity labs."""
    start_server(model, extra)
    print(f"===== {tag}: log =====")
    show_log()
    print(f"\n===== {tag}: golden set =====")
    golden(model, tag)
    bench(model, ["--num-prompts", "10", "--request-rate", "1"],
          "warmup.txt")
    print(f"\n===== {tag}: single stream =====", flush=True)
    bench(model, ["--num-prompts", "15", "--max-concurrency", "1"],
          f"{tag}_single.txt")
    for rate, n in [("0.5", "40"), ("1", "80"), ("2", "80")]:
        print(f"\n===== {tag}: rate {rate} =====", flush=True)
        bench(model, ["--num-prompts", n, "--request-rate", rate,
                      "--goodput", "ttft:2000", "tpot:200"],
              f"{tag}_r{rate}.txt")


def print_answers(tag):
    """Print answers next to their criteria for human judging."""
    for line in open(f"answers_{tag}.jsonl"):
        r = json.loads(line)
        print(f"=== {r['id']}\nCRITERIA: {r['criteria']}\n"
              f"ANSWER: {r['answer']}\n")
