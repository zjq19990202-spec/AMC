#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PI05_PYTHON=${PI05_PYTHON:-/home/admin123/zjq/ws/pi0.5/.venv/bin/python-stable}

if [[ ! -x "$PI05_PYTHON" ]]; then
  echo "PI0.5 Python is not executable: $PI05_PYTHON" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTHONPATH="/home/admin123/zjq/ws/pi0.5/src:$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PI05_PYTHON" "$PROJECT_ROOT/scripts/serve_no_force_pi05_keyboard_50.py" \
  --plain-pi05 \
  --host 0.0.0.0 \
  --port 12000 \
  --checkpoint "$PROJECT_ROOT/runtime_assets/checkpoints/plain_pi05_union2375_masked_subtask_25k" \
  --norm-assets-dir "$PROJECT_ROOT/runtime_assets/norm" \
  --norm-asset-id openpi_norm_union2375_allframes_v1 \
  --prompt-file "$PROJECT_ROOT/runtime_assets/prompts/la4vla_fruit_three_boxes.json" \
  --prompt-mode continuous \
  --disable-prompt-transition-compose \
  --initial-key 1 \
  --quit-key 0 \
  "$@"
