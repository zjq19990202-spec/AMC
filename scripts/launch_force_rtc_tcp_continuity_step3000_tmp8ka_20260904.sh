#!/usr/bin/env bash
set -euo pipefail

PYTHON=/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python
SOURCE=/mnt/cunchu/zjq/atomic_latent_vla_snapshots/force_spherical_b2_subtaskpad_matched50k_20260904
RAM_SOURCE=/dev/shm/atomic_latent_vla_eval_tcp_continuity_step4000_20260904
ROOT=/mnt/cunchu/zjq/atomic_pi05_force_runs
RUN=force_spherical45h20_crossatt2_finalzm40_subtaskpad_from_matchedspherical50k_b2s6000_20260904
CKPT=${ROOT}/${RUN}/${RUN}/4000
OUT=${ROOT}/evals/${RUN}/step4000_rtc_tcp_continuity_seed20260824

cleanup() {
  status=$?
  if [[ "$RAM_SOURCE" == /dev/shm/atomic_latent_vla_eval_* && -d "$RAM_SOURCE" ]]; then
    cd /
    rm -rf -- "$RAM_SOURCE"
  fi
  exit "$status"
}
trap cleanup EXIT

test -f "$CKPT/params/_METADATA"
test ! -e "$RAM_SOURCE"
test ! -e "$OUT"
cp -a "$SOURCE" "$RAM_SOURCE"
cd "$RAM_SOURCE"
export PYTHONPATH="$RAM_SOURCE/vendor/pi0.5/src:$RAM_SOURCE/vendor/pi0.5/packages/openpi-client/src:$RAM_SOURCE/src"
mkdir -p "$OUT"

common=(
  --model-kind b2 --checkpoint "$CKPT"
  --pad-subtask-horizon --skip-wrong-tokens
  --norm-assets-dir /mnt/cunchu/zjq/target
  --norm-asset-id openpi_norm_union2375_allframes_v1
  --force-norm /mnt/cunchu/zjq/force_assets/force_norm_plug_vase_shared_lr_v1_20260822/norm_stats.json
  --episode 0 --batch-size 4 --max-windows 16 --seed 20260824
  --encoder-width 512 --encoder-heads 8 --encoder-mlp-dim 1024 --force-latent-dim 512
  --full-token-force-adapter --full-token-force-adapter-heads 2
)

CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/evaluate_force_rtc_offsets.py "${common[@]}" \
  --dataset-root /mnt/cunchu/zjq/target/lerobot_v3_4to1_plug_force \
  --subtask-sidecar /mnt/cunchu/zjq/force_assets/subtask_sidecars_plug_vase_20260904/plug_episode_subtasks.jsonl \
  --output-dir "$OUT/plug_episode0_first16" >"$OUT/plug.log" 2>&1 & p0=$!

CUDA_VISIBLE_DEVICES=1 "$PYTHON" scripts/evaluate_force_rtc_offsets.py "${common[@]}" \
  --dataset-root /mnt/cunchu/zjq/target/lerobot_v3_4to1_vase_force \
  --subtask-sidecar /mnt/cunchu/zjq/force_assets/subtask_sidecars_plug_vase_20260904/vase_episode_subtasks.jsonl \
  --output-dir "$OUT/vase_episode0_first16" >"$OUT/vase.log" 2>&1 & p1=$!

wait "$p0" "$p1"
date --iso-8601=seconds >"$OUT/COMPLETE"
