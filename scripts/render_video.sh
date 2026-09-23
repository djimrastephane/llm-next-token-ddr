#!/usr/bin/env bash
# Render the final 1080x1920 60 fps MP4 from the current inference trace.
set -euo pipefail
cd "$(dirname "$0")/.."
npx remotion render src/video/index.ts NextTokenDDR out/next_token_ddr.mp4 "$@"
