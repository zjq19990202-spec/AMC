#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_root}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTHONPATH="/home/admin123/zjq/ws/pi0.5/src:${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
checkpoint=${CHECKPOINT:-/home/admin123/target2058_zm_latest_inference/runtime_assets/checkpoints/target2058_zm_40000}
session_id=${TRACE_SESSION_ID:-$(date +%Y%m%d_%H%M%S)}
trace_root=${TRACE_ROOT:-${repo_root}/runtime_assets/inference_traces/right_drawer_charger_markers_${session_id}}
trace_csv=${TRACE_CSV:-${trace_root}/inference.csv}
trace_images_dir=${TRACE_IMAGES_DIR:-${trace_root}/images}

exec /home/admin123/zjq/ws/pi0.5/.venv/bin/python \
  scripts/serve_layerwise_pi05_keyboard.py \
  --host 0.0.0.0 \
  --port 12000 \
  --checkpoint "${checkpoint}" \
  --norm-assets-dir "${repo_root}/runtime_assets/norm" \
  --norm-asset-id openpi_norm_compact_accepted_v3_no_short_no_idle \
  --coefficient-target joint_delta \
  --prompt-mode one-shot \
  --initial-key 1 \
  --trace-csv "${trace_csv}" \
  --trace-images-dir "${trace_images_dir}" \
  --prompt-file "${repo_root}/runtime_assets/prompts/right_drawer_charger_markers.json"
