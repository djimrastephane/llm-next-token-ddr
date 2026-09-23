#!/usr/bin/env bash
# Capture the two traces used by the project from the same DDR context:
#   data/generated/inference_trace.json         top-p sampling (drives the video)
#   data/generated/inference_trace_greedy.json  greedy decoding (for comparison)
# Extra arguments are passed to both runs, e.g. --context-id completions_packer or --model Qwen/Qwen2.5-0.5B-Instruct
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
"$PY" -m src.inference.capture_trace --mode sampling --temperature 0.7 --top-p 0.90 --seed 42 --steps 5 "$@"
"$PY" -m src.inference.capture_trace --mode greedy --seed 42 --steps 5 \
  --output data/generated/inference_trace_greedy.json "$@"
