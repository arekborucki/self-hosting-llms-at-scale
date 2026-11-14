# Chapter 7: Quantizing large models

Scripts for the labs in chapter 7. Tested on a Kaggle notebook with Tesla T4 GPUs and vLLM 0.24.0.

| File | What it does |
|---|---|
| `lab_harness.py` | Helper functions: start the server, read the startup log, run the golden set from chapter 4, run the load sweep from chapter 6. `run_all` is listing 7.5. |
| `run_labs.py` | Labs 7.1 to 7.4: every configuration from tables 7.2 and 7.4, plus the 16-bit 8B baseline on two T4s. |
| `kv_cache_fp8_check.py` | Checks whether the card accepts an 8-bit KV cache (it does not on a T4). |

Setup in a notebook:

```
!pip install -q vllm==0.24.0 openai
%run run_labs.py
```

Judge the golden set answers by hand with `print_answers("8b-awq")` and the strict rule from chapter 4. Results depend on the session; compare configurations measured in the same session.
