---
name: quantization-decision
description: Decide whether and how to quantize an LLM for serving with vLLM. Use when a model does not fit on a GPU, when more KV cache or serving capacity is needed on the same card, or when choosing between AWQ, GPTQ, INT8, FP8, or 16-bit checkpoints. Covers memory estimates per precision, hardware support checks, reading the vLLM startup log, quality checks on a golden set, capacity checks with a load sweep, and recording the decision. Based on chapter 7 of "Self-Hosting LLMs at Scale" (Manning).
---

# Quantization decision

Use this procedure to decide whether to serve a quantized checkpoint, and which one. Every decision rests on measurements of the specific checkpoint on the specific card.

## 1. Estimate memory per precision

Run `scripts/quant_memory.py` with the model's Hugging Face ID and the usable GPU memory:

```
python scripts/quant_memory.py Qwen/Qwen3-8B --params 8.2e9 --gpu-gib 15
```

Pass the total parameter count from the model card with --params. The script reads config.json and prints weight memory at 16, 8, and 4 bits, keeping the embedding and output layers in 16-bit precision, plus the KV cache per token. For Qwen3-8B it estimates 5.56 GiB at 4 bits and 8.80 GiB at 8 bits; vLLM reported 5.71 GiB and 8.9 GiB, the difference being scales and other quantization data. If the 16-bit model fits with enough room for the KV cache to meet the service level objective, serve it in 16 bits and stop here.

## 2. Check what the card supports

Check the vLLM supported-hardware table: https://docs.vllm.ai/en/stable/features/quantization/

- Turing (T4): AWQ and GPTQ 4-bit weights run through Marlin. FP8 checkpoints run as weight-only W8A16 with a logged warning. FP8 KV cache is refused by the Triton attention backend.
- Ada, Hopper, and newer: FP8 W8A8 and FP8 KV cache are available; test them like any other format.

## 3. Prefer a published checkpoint

Look for the original publisher's quantized checkpoint first, then well-known third parties. Before serving any checkpoint, read `quantization_config` in its config.json (bits, group size, symmetric or not, ignored layers). Trust the configuration over the model card.

## 4. Serve it and read the startup log

Start vLLM with the same settings as the baseline and record from the log:
- the quantization method and the kernel selected (for example MarlinLinearKernel)
- any fallback warning (for example weight-only FP8 on cards without FP8 compute)
- model weight memory, KV cache size in tokens, and maximum concurrency

Compare the weight memory with the estimate from step 1.

## 5. Measure quality on your own golden set

Run the golden set at temperature 0 on the quantized checkpoint and on the model you serve today, in the same session. Judge by the criteria written in advance and compare question by question, especially trap prompts (invented laws, cases, or facts). Do not over-interpret one-point score differences on a small set. Two checkpoints of the same model with nearly identical performance can still differ on a trap prompt.

## 6. Measure capacity with the same load sweep

Run the baseline's benchmark (same input and output lengths, same objective, warm-up first) for both models in the same session:
- single-stream TPOT
- goodput at rising offered rates
- the highest tested rate at which goodput matches throughput

Repeat with your longest realistic prompts. Long prompts can miss a TTFT objective even with one request at a time, regardless of KV cache size.

## 7. Record the decision

Write into the decision file next to the deployment configuration:
- the exact checkpoint repository and revision, quantization method, and kernel from the log
- golden set score and trap prompt results, with the session date
- the highest tested rate that met the objective, and the prompt lengths it covers
- the tested fallback configuration
- the trigger for re-evaluation (for example peak load approaching the tested rate)
