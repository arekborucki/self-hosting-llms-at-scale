"""Chapter 6, listing 6.1: before-and-after counter delta.

Snapshot the server counters, run exactly one experiment, snapshot
again, and subtract. The delta is that experiment's own footprint,
which is what chapter 5's cumulative reads could not give.
"""
import re

import requests

METRICS_URL = "http://localhost:8000/metrics"
NAMES = [
    "prompt_tokens_total",
    "generation_tokens_total",
    "prefix_cache_queries_total",
    "prefix_cache_hits_total",
    "num_preemptions_total",
]


def snapshot():
    text = requests.get(METRICS_URL).text
    out = {}
    for name in NAMES:
        m = re.search(rf"vllm:{name}{{[^}}]*}}\s+([0-9.e+]+)", text)
        out[name] = float(m.group(1)) if m else 0.0
    return out


if __name__ == "__main__":
    before = snapshot()
    input("Run your experiment, then press Enter...")
    after = snapshot()
    for name in NAMES:
        print(f"{name}: +{after[name] - before[name]:.0f}")
