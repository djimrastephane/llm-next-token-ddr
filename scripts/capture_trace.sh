#!/usr/bin/env bash
# Capture the two traces used by the project from the same DDR context:
#   data/generated/inference_trace.json         top-p sampling (drives the video)
#   data/generated/inference_trace_greedy.json  greedy decoding (for comparison)
# Both stop when the model ends the sentence (cap: 40 tokens).
# Extra arguments are passed to both runs, e.g. --context-id completions_packer or --model Qwen/Qwen2.5-0.5B-Instruct
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
"$PY" -m src.inference.capture_trace --mode sampling --temperature 0.7 --top-p 0.90 --seed 42 --steps 40 --until-sentence-end "$@"
"$PY" -m src.inference.capture_trace --mode greedy --seed 42 --steps 40 --until-sentence-end \
  --output data/generated/inference_trace_greedy.json "$@"
