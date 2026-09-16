#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PI05_PYTHON=${PI05_PYTHON:-/home/admin123/yc/conda_envs/pi05/bin/python}

if [[ ! -x "$PI05_PYTHON" ]]; then
  echo "PI0.5 Python is not executable: $PI05_PYTHON" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTHONPATH="$PROJECT_ROOT/vendor/pi0.5/src:$PROJECT_ROOT/vendor/pi0.5/packages/openpi-client/src:$PROJECT_ROOT/src:$PROJECT_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

exec "$PI05_PYTHON" "$PROJECT_ROOT/scripts/serve_la4vla_pi05_keyboard.py" "$@"
