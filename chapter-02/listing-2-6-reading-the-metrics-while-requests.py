import concurrent.futures
import re
import time

import requests
from openai import OpenAI


client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="x",
)


def ask(i):
    response = client.chat.completions.create(
        model="Qwen/Qwen3-4B",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question {i}: explain force majeure in detail, "
                    "including several examples from commercial contracts."
                ),
            }
        ],
        max_tokens=512,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        },
    )
    return response.choices[0].message.content


def metric_value(text, name):
    match = re.search(
        rf'{re.escape(name)}\{{[^}}]*\}}\s+([0-9.]+)',
        text,
    )
    return float(match.group(1)) if match else 0.0


max_running = 0
max_waiting = 0
max_kv = 0.0

with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    futures = [
        executor.submit(ask, i)
        for i in range(32)
    ]

    while not all(future.done() for future in futures):
        metrics = requests.get(
            "http://localhost:8000/metrics",
            timeout=5,
        ).text

        running = metric_value(
            metrics,
            "vllm:num_requests_running",
        )
        waiting = metric_value(
            metrics,
            "vllm:num_requests_waiting",
        )
        kv = metric_value(
            metrics,
            "vllm:kv_cache_usage_perc",
        )

        max_running = max(max_running, running)
        max_waiting = max(max_waiting, waiting)
        max_kv = max(max_kv, kv)

        print(
            f"running={running:.0f}, "
            f"waiting={waiting:.0f}, "
            f"kv_cache={kv:.3f}"
        )

        time.sleep(0.25)

    results = [future.result() for future in futures]


print(
    "max running:", max_running,
    "max waiting:", max_waiting,
    "max KV:", max_kv,
    "completed:", len(results),
)
