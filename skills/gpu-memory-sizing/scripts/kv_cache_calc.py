"""Estimate weight memory, KV cache per token, and concurrency for one GPU.

Usage:
    python kv_cache_calc.py --params 8.2e9 --bytes-per-param 2 \
        --layers 36 --kv-heads 8 --head-dim 128 \
        --gpu-gib 15 --overhead-gib 1.5 --tokens-per-conv 4000

All numbers are estimates. Check them against the server's startup log.
"""
import argparse

GIB = 2**30

p = argparse.ArgumentParser()
p.add_argument("--params", type=float, required=True, help="total parameters")
p.add_argument("--bytes-per-param", type=float, default=2.0)
p.add_argument("--layers", type=int, required=True)
p.add_argument("--kv-heads", type=int, required=True)
p.add_argument("--head-dim", type=int, required=True)
p.add_argument("--kv-bytes", type=float, default=2.0, help="2 for 16-bit KV cache")
p.add_argument("--gpu-gib", type=float, required=True, help="usable GPU memory")
p.add_argument("--overhead-gib", type=float, default=1.5)
p.add_argument("--tokens-per-conv", type=int, required=True)
a = p.parse_args()

weights = a.params * a.bytes_per_param / GIB
kv_per_token = 2 * a.layers * a.kv_heads * a.head_dim * a.kv_bytes
free = a.gpu_gib - weights - a.overhead_gib

print(f"weights:              {weights:7.2f} GiB")
print(f"KV cache per token:   {kv_per_token / 1024:7.1f} KiB")
if free <= 0:
    print(f"does not fit: short by {-free:.2f} GiB")
else:
    tokens = free * GIB / kv_per_token
    print(f"room for KV cache:    {free:7.2f} GiB")
    print(f"KV cache capacity:    {tokens:9,.0f} tokens")
    print(f"conversations at {a.tokens_per_conv} tokens: {tokens / a.tokens_per_conv:.1f}")
