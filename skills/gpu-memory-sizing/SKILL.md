---
name: gpu-memory-sizing
description: Estimates GPU memory, KV cache capacity, concurrency, and rough cost for serving an LLM on given hardware, before anything is bought or deployed. Use this skill whenever someone asks whether a model fits on a GPU, how many GPUs they need, how many users a card can serve, how big the KV cache is, or what self-hosting a model will cost, even if they only name a model and a GPU.
---

# Sizing GPU memory and cost for LLM serving

Based on chapter 3 of *Self-Hosting LLMs at Scale* (Manning).

## The memory equation

Total memory = weights + KV cache + runtime overhead.

1. **Weights** = parameters × bytes per parameter. Use the exact parameter count from the model card, not the rounded name. 16-bit = 2 bytes, 8-bit = 1 byte, 4-bit ≈ 0.5 bytes plus some extra for scales and layers kept in 16-bit precision.
2. **KV cache per token** = 2 (K and V) × layers × KV heads × head dimension × bytes per value. Read layers, KV heads, and head dimension from the model's `config.json` (`num_hidden_layers`, `num_key_value_heads`, `head_dim` or hidden size / attention heads).
3. **KV cache for the workload** = KV bytes per token × tokens per conversation × peak concurrent conversations.
4. **Runtime overhead**: leave about 1 to 2 GiB for CUDA graphs, activations, and allocator overhead, and check it later in the server's startup log.

Run `scripts/kv_cache_calc.py` to do the arithmetic.

## Mixture of Experts models

All experts must be in memory, even though only a few run per token. Use the **total** parameter count for weight memory and the active count only for compute.

## From memory to users

- Peak concurrency, not the number of users, sets the KV cache need. Estimate how many conversations are active at the busiest moment.
- Capacity in tokens ÷ tokens per conversation ≈ conversations that fit at once.
- Running out of KV cache does not crash a vLLM server; requests wait in the queue, and latency grows.

## Cost

- Hosted API: tokens per month × price per token.
- Rented GPUs: cards × hours × hourly rate, plus engineering time to run the deployment.
- The break-even point depends on utilization; compute it with the organization's own numbers.

## Output

Report: weight memory, KV cache per token, KV capacity left on the chosen card, concurrent conversations at the chosen context length, whether it fits, and the options if it does not (smaller model, quantization, shorter context, more or larger GPUs). State every assumption and recommend checking the estimate against the server's startup log.
