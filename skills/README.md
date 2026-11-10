# AI skills for Self-Hosting LLMs at Scale

Each folder is a skill: a set of instructions an AI assistant can load to apply one procedure from the book. They follow the Agent Skills format (a folder with a `SKILL.md` file), so they work with assistants that support skills, such as Claude.

| Skill | Chapter | What it does |
|---|---|---|
| `self-hosting-decision` | 1 | Decide whether to self-host or stay on a hosted API |
| `vllm-first-endpoint` | 2 | Serve a model with vLLM, send the first requests, read the metrics |
| `gpu-memory-sizing` | 3 | Estimate weights, KV cache, concurrency, and cost before buying hardware |
| `model-selection` | 4 | Choose a model with a license check, a golden set, and a decision file |
| `vllm-performance-diagnosis` | 5 | Find why an inference server is slow, using prefill, decode, batching, and the KV cache |
| `llm-endpoint-benchmarking` | 6 | Benchmark an endpoint, find its capacity, set an SLO, and tune it |

To use a skill, add its folder to your assistant's skills (for example, upload it in Claude's settings or place it in your project's skills directory).
