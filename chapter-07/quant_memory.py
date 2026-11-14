"""Estimate weight memory at 16, 8, and 4 bits and KV cache per token.

Usage:
    python quant_memory.py Qwen/Qwen3-8B --params 8.2e9 --gpu-gib 15

Embedding and output layers stay in 16-bit precision, as in most
weight-only checkpoints. The estimate ignores scales and zero points
(a few percent at 4 bits with group size 128) and runtime overhead.
"""
import argparse
import json
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("--params", type=float, required=True,
               help="total parameter count from the model card, e.g. 8.2e9")
p.add_argument("--gpu-gib", type=float, default=15.0)
a = p.parse_args()

url = f"https://huggingface.co/{a.model}/resolve/main/config.json"
cfg = json.load(urllib.request.urlopen(url))
hidden = cfg["hidden_size"]
vocab = cfg["vocab_size"]
layers = cfg["num_hidden_layers"]
kv_heads = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
head_dim = cfg.get("head_dim", hidden // cfg["num_attention_heads"])
tied = cfg.get("tie_word_embeddings", False)

emb = vocab * hidden * (1 if tied else 2)
GiB = 2 ** 30
print(f"{a.model}: {a.params/1e9:.2f}B parameters, "
      f"{emb/1e9:.2f}B in embedding/output layers")
for bits in (16, 8, 4):
    w = emb * 2 + (a.params - emb) * bits / 8
    left = a.gpu_gib - w / GiB
    print(f"  {bits:>2}-bit weights: {w/GiB:6.2f} GiB, "
          f"left on a {a.gpu_gib:g} GiB card: {left:6.2f} GiB")
kv = 2 * layers * kv_heads * head_dim * 2
print(f"  KV cache per token at 16 bits: {kv/1024:.1f} KiB")
