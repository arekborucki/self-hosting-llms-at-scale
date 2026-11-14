"""Chapter 7: labs 7.1 to 7.4 in one session on one T4 (plus a 16-bit
baseline on two T4s for the question-by-question comparison).

Run in a notebook cell with:  %run run_labs.py
Each configuration takes about 15 minutes. Judge the golden set answers
by hand with print_answers(tag), using the strict rule from chapter 4.
"""
import os
from lab_harness import run_all, start_server, show_log, golden

os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # one T4 for all single-card labs

# Labs 7.1 to 7.3: the 4-bit 8B model and the 16-bit 4B pilot
run_all("Qwen/Qwen3-8B-AWQ", "8b-awq")
run_all("Qwen/Qwen3-4B", "4b-fp16")

# Lab 7.4: other checkpoints on the same card
run_all("Qwen/Qwen3-4B-AWQ", "4b-awq")
run_all("RedHatAI/Qwen3-8B-quantized.w4a16", "8b-w4a16")
run_all("Qwen/Qwen3-8B-FP8", "8b-fp8")     # runs as W8A16 on a T4

# 16-bit 8B on both T4s, golden set only
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
start_server("Qwen/Qwen3-8B", ["--tensor-parallel-size", "2"])
show_log()
golden("Qwen/Qwen3-8B", "8b-fp16")
