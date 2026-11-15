# Chapter 8: Scaling: multi-GPU, multi-node, and disaggregated serving

Scripts for the labs in chapter 8. Tested on a Kaggle notebook with two Tesla T4 GPUs and vLLM 0.24.0.

| File | What it does |
|---|---|
| `multi_gpu_harness.py` | Helpers: `gpu_clean` (listing 8.2), starting a server on one GPU and port, waiting for health, and the mixed-traffic test with its own seeds. |
| `rr_proxy.py` | Listing 8.1: a round-robin router for two vLLM servers on ports 8001 and 8002. |
| `run_labs.py` | Labs 8.1 to 8.5: one replica against two, the router, tensor and pipeline parallelism, the 4-bit model with tensor parallelism, and disaggregated prefill and decode. |

Setup in a notebook with two GPUs:

```
!pip install -q vllm==0.24.0 openai nixl
!wget -q https://raw.githubusercontent.com/arekborucki/self-hosting-llms-at-scale/main/chapter-07/lab_harness.py
%run run_labs.py
```

Run `gpu_clean()` before every new server; worker processes from a previous server can keep both cards full. Every benchmark gets its own `--seed`, so repeated prompts do not hit the prefix cache. Results depend on the session; compare layouts measured in the same session.

Lab 8.4 (one model on two machines) needs two rented machines and is not part of this script.
