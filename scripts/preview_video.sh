#!/usr/bin/env bash
# Open Remotion Studio to scrub through the video (reads data/generated/inference_trace.json).
set -euo pipefail
cd "$(dirname "$0")/.."
npx remotion studio src/video/index.ts
