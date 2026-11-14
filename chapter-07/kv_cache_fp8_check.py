"""Chapter 7: does the card accept an 8-bit KV cache?

On a Tesla T4 with vLLM 0.24.0, both fp8 and fp8_e5m2 fail at startup
because the Triton attention backend requires SM89 or newer for FP8.
On Ada or newer GPUs, the server should start; then rerun the quality
and capacity labs with the same option.
"""
import os
from lab_harness import start_server, show_log

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
for dtype in ["fp8", "fp8_e5m2"]:
    print(f"===== --kv-cache-dtype {dtype} =====")
    try:
        start_server("Qwen/Qwen3-8B-AWQ", ["--kv-cache-dtype", dtype])
        show_log()
    except RuntimeError as e:
        print("FAILED:", e)
        show_log()
