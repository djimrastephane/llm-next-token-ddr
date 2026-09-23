#!/usr/bin/env bash
# Capture the two traces used by the project from the same DDR context:
#   $OUT_DIR/inference_trace.json         top-p sampling (drives the video)
#   $OUT_DIR/inference_trace_greedy.json  greedy decoding (for comparison)
# Both stop when the model ends the sentence (cap: 40 tokens).
# OUT_DIR defaults to data/generated. Extra arguments are passed to both runs, e.g.
#   scripts/capture_trace.sh --context-id completions_packer
#   OUT_DIR=/tmp/run scripts/capture_trace.sh --model Qwen/Qwen2.5-0.5B-Instruct
set -euo pipefail
cd "$(dirname "$0")/.."
for a in "$@"; do
  case "$a" in
    --output | --output=*)
      echo "capture_trace.sh writes two traces, so --output would make one overwrite the other." >&2
      echo "Set OUT_DIR=<directory> instead." >&2
      exit 2
      ;;
  esac
done
OUT_DIR="${OUT_DIR:-data/generated}"
PY="${PYTHON:-.venv/bin/python}"
"$PY" -m src.inference.capture_trace --mode sampling --temperature 0.7 --top-p 0.90 --seed 42 --steps 40 --until-sentence-end \
  --output "$OUT_DIR/inference_trace.json" "$@"
"$PY" -m src.inference.capture_trace --mode greedy --seed 42 --steps 40 --until-sentence-end \
  --output "$OUT_DIR/inference_trace_greedy.json" "$@"
