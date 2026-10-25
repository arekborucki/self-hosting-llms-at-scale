#!/usr/bin/env bash
# Chapter 6, LAB 4: clean single-stream decode measurement.
#
# --max-concurrency 1 sends one request at a time, which is the right
# setup for comparing with the single-stream bandwidth ceiling
# estimated from memory bandwidth (about 37 tok/s on a T4).
#
# Recorded on one T4, vLLM 0.24.0, Qwen/Qwen3-4B:
#   TTFT p50 385 ms, TPOT p50 40.7 ms (~24.6 tok/s), E2EL p50 6.45 s
#   24.6 tok/s is 66% of the 37 tok/s bandwidth-only ceiling.
#
# The recorded run used no --temperature 0; this script adds it so
# that reruns are reproducible.
set -euo pipefail

vllm bench serve \
  --model Qwen/Qwen3-4B \
  --dataset-name random \
  --random-input-len 600 \
  --random-output-len 150 \
  --num-prompts 15 \
  --max-concurrency 1 \
  --temperature 0 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,95
