---
name: multi-gpu-layout
description: Choose and test a multi-GPU layout for serving an LLM with vLLM - replicas (data parallel or separate servers behind a router), tensor parallelism, pipeline parallelism, multi-node, or disaggregated prefill and decode. Use when one GPU runs out of memory, capacity, or time for long prompts, or when comparing layouts on the same hardware. Covers memory division, startup problems, communication checks in the logs, benchmarking with per-run seeds, and recording the decision. Based on chapter 8 of "Self-Hosting LLMs at Scale" (Manning).
---

# Multi-GPU layout

Use this procedure to decide how to use more than one GPU for one model. Every layout choice rests on measurements on the actual cards.

## 1. Name the limit

Decide which limit one GPU hits, using the load sweep from the benchmarking chapter:
- memory: the weights and the KV cache the workload needs do not fit
- compute: goodput falls below throughput as offered load rises
- time: a single request misses its latency target on an idle card (often long prompts)

Replicas address compute. Tensor or pipeline parallelism addresses memory and the time of long prompts.

## 2. Estimate memory per layout

- Replicas: each GPU holds the full model and its own KV cache.
- Tensor parallelism (TP=N): each GPU holds 1/N of every weight matrix and 1/N of the KV heads, so the KV cost per token per GPU divides by N. The model's sharded dimensions, especially the attention-head count, must be compatible with N.
- Pipeline parallelism (PP=N): each GPU holds a contiguous group of layers, their weights, and their part of the KV cache.

Combining TP with 4-bit weights can give a much larger KV cache than either alone.

## 3. Start each layout cleanly

- Before every start, stop all vLLM processes, including workers named `VLLM::Worker_*`, and check that GPU memory is free. Run each pkill pattern as its own command, without a shell.
- If a multi-GPU start runs out of memory, lower `--gpu-memory-utilization` (for example to 0.85 or 0.80) and use the same value for every layout you compare.
- Read the startup log for the communication path: peer-to-peer support, custom all-reduce, and, across machines, the NCCL transport (`NCCL_DEBUG=TRACE`; `NET/Socket` means TCP, `NET/IB` means InfiniBand).

## 4. Benchmark every layout the same way

- Same model, workload shape, objective, and memory setting; measure layouts in the same session.
- Give every benchmark run its own `--seed`; repeated prompts hit the prefix cache.
- Measure single-request TTFT and TPOT, goodput at rising offered rates, the highest tested rate within the objective, single-request TTFT for the longest realistic prompts, and mixed traffic (short and long requests together) with ITL p95 for the short ones.

## 5. Know what each layout trades

- Replicas: more requests within the objective; no change to the time of one request; a replica can fail alone.
- Tensor parallelism: needs fast links between GPUs; can shorten long-prompt TTFT; the whole group fails together.
- Pipeline parallelism: less communication; a single request uses one stage at a time; the documentation recommends it for GPUs without NVLink, but measure both.
- Disaggregated prefill and decode (experimental): lowers tail ITL under mixed traffic; does not raise throughput; the prefill-to-decode capacity ratio must match the traffic, or TTFT rises sharply. Check the transport with `UCX_PROTO_INFO=y`.
- Multi-node: run on an isolated private network; inter-node traffic is not encrypted. Prefer TP inside a machine and PP between machines.

## 6. Record the decision

Write into the decision file:
- the layout, flags, and memory setting
- the highest tested rate within the objective, single-request TTFT for long prompts, and mixed-traffic results, with the session date
- separate objectives the organization sets for long documents (a product decision, not a benchmark result)
- the failure behavior and the fallback configuration
- the trigger for the next review
