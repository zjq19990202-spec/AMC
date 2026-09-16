#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

exec "$PROJECT_ROOT/scripts/start_la4vla_inference.sh" \
  --prompt-file "$PROJECT_ROOT/runtime_assets/prompts/la4vla_fruit_three_boxes.json" \
  --initial-key 1 \
  --quit-key 0 \
  "$@"
