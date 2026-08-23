import subprocess

server = subprocess.Popen(
    ["vllm", "serve", "Qwen/Qwen3-4B",
     "--dtype", "half",
     "--max-model-len", "8192"],
    stdout=open("vllm.log", "w"),
    stderr=subprocess.STDOUT,
)
print("vLLM starting, PID:", server.pid)
