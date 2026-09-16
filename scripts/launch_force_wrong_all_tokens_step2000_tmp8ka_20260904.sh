#!/usr/bin/env bash
set -euo pipefail

PYTHON=/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python
SOURCE=/mnt/cunchu/zjq/atomic_latent_vla_snapshots/force_spherical_b2_subtaskpad_matched50k_20260904
RAM_SOURCE=/dev/shm/atomic_latent_vla_eval_wrong_all_tokens_step2000_20260904
ROOT=/mnt/cunchu/zjq/atomic_pi05_force_runs
RUN=force_spherical45h20_crossatt2_finalzm40_subtaskpad_from_matchedspherical50k_b2s6000_20260904
B2=${ROOT}/${RUN}/${RUN}/2000
OUT=${ROOT}/evals/${RUN}/step2000_wrong_all_slow_fast_tokens_seed20260824
NORM_ROOT=/mnt/cunchu/zjq/target
NORM_ID=openpi_norm_union2375_allframes_v1
FORCE_NORM=/mnt/cunchu/zjq/force_assets/force_norm_plug_vase_shared_lr_v1_20260822/norm_stats.json

cleanup() {
  status=$?
  if [[ "$RAM_SOURCE" == /dev/shm/atomic_latent_vla_eval_* && -d "$RAM_SOURCE" ]]; then
    cd /
    rm -rf -- "$RAM_SOURCE"
  fi
  exit "$status"
}
trap cleanup EXIT

test -f "$B2/params/_METADATA"
test ! -e "$RAM_SOURCE"
test ! -e "$OUT"
cp -a "$SOURCE" "$RAM_SOURCE"
cd "$RAM_SOURCE"
export PYTHONPATH="$RAM_SOURCE/vendor/pi0.5/src:$RAM_SOURCE/vendor/pi0.5/packages/openpi-client/src:$RAM_SOURCE/src"
mkdir -p "$OUT"

run_one() {
  local gpu=$1 domain=$2 data=$3 sidecar=$4
  local target=${OUT}/${domain}_episode0_first16
  mkdir -p "$target"
  CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  "$PYTHON" scripts/evaluate_force_rtc_offsets.py \
    --model-kind b2 --checkpoint "$B2" \
    --dataset-root "$data" --subtask-sidecar "$sidecar" --pad-subtask-horizon \
    --norm-assets-dir "$NORM_ROOT" --norm-asset-id "$NORM_ID" --force-norm "$FORCE_NORM" \
    --episode 0 --batch-size 4 --max-windows 16 --seed 20260824 \
    --encoder-width 512 --encoder-heads 8 --encoder-mlp-dim 1024 --force-latent-dim 512 \
    --full-token-force-adapter --full-token-force-adapter-heads 2 \
    --output-dir "$target" >"${target}.log" 2>&1
}

run_one 0 plug /mnt/cunchu/zjq/target/lerobot_v3_4to1_plug_force \
  /mnt/cunchu/zjq/force_assets/subtask_sidecars_plug_vase_20260904/plug_episode_subtasks.jsonl & p0=$!
run_one 1 vase /mnt/cunchu/zjq/target/lerobot_v3_4to1_vase_force \
  /mnt/cunchu/zjq/force_assets/subtask_sidecars_plug_vase_20260904/vase_episode_subtasks.jsonl & p1=$!
wait "$p0" "$p1"
date --iso-8601=seconds >"$OUT/COMPLETE"
