#!/usr/bin/env bash
set -euo pipefail

PYTHON=/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python
SOURCE=/mnt/cunchu/zjq/atomic_latent_vla_snapshots/force_spherical_b2_subtaskpad_matched50k_20260904
RAM_SOURCE=/dev/shm/atomic_latent_vla_eval_force_b2_subtaskpad_rmse_20260904
RUN=force_spherical45h20_crossatt2_finalzm40_subtaskpad_from_matchedspherical50k_b2s6000_20260904
ROOT=/mnt/cunchu/zjq/atomic_pi05_force_runs
B2=${ROOT}/${RUN}/${RUN}/6000
BASE_RUN=union2375_afro50k_matched_spherical20_w005_from25k_to50k_fsdp8_bs256_20260902
BASE=/mnt/cunchu/zjq/atomic_pi05_runs/${BASE_RUN}/${BASE_RUN}/50000
PIPELINE_LOG=${ROOT}/logs/${RUN}.pipeline.log
WATCH_LOG=${ROOT}/logs/${RUN}_rmse_eval.pipeline.log
LOCK=${ROOT}/.locks/${RUN}_rmse_eval.lock
OUT_ROOT=${ROOT}/evals/${RUN}/training_faithful_episode50_seed20260824
NORM_ROOT=/mnt/cunchu/zjq/target
NORM_ID=openpi_norm_union2375_allframes_v1
FORCE_NORM=/mnt/cunchu/zjq/force_assets/force_norm_plug_vase_shared_lr_v1_20260822/norm_stats.json
PLUG_ROOT=/mnt/cunchu/zjq/target/lerobot_v3_4to1_plug_force
VASE_ROOT=/mnt/cunchu/zjq/target/lerobot_v3_4to1_vase_force
PLUG_SUBTASK=/mnt/cunchu/zjq/force_assets/subtask_sidecars_plug_vase_20260904/plug_episode_subtasks.jsonl
VASE_SUBTASK=/mnt/cunchu/zjq/force_assets/subtask_sidecars_plug_vase_20260904/vase_episode_subtasks.jsonl

cleanup() {
  status=$?
  if [[ "$RAM_SOURCE" == /dev/shm/atomic_latent_vla_eval_* && -d "$RAM_SOURCE" ]]; then
    cd /
    rm -rf -- "$RAM_SOURCE"
  fi
  exit "$status"
}
trap cleanup EXIT

mkdir -p "$(dirname "$LOCK")" "$OUT_ROOT"
exec 9>"$LOCK"
flock -n 9 || { echo "evaluation lock already held: $LOCK" >&2; exit 70; }
exec >>"$WATCH_LOG" 2>&1

echo "$(date --iso-8601=seconds) waiting for corrected subtask-padded B2 step6000"
until [[ -f "$B2/params/_METADATA" ]] &&
      [[ -f "$B2/params/manifest.ocdbt" ]] &&
      grep -q 'corrected-subtask B2 6000 finalized' "$PIPELINE_LOG"; do
  sleep 30
done
echo "$(date --iso-8601=seconds) finalized checkpoint detected"

test ! -e "$RAM_SOURCE"
cp -a "$SOURCE" "$RAM_SOURCE"
cd "$RAM_SOURCE"
export PYTHONPATH="$RAM_SOURCE/vendor/pi0.5/src:$RAM_SOURCE/vendor/pi0.5/packages/openpi-client/src:$RAM_SOURCE/src"

run_domain() {
  local gpu=$1
  local name=$2
  local data=$3
  local sidecar=$4
  local out=${OUT_ROOT}/${name}_episode0
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" \
  XLA_PYTHON_CLIENT_PREALLOCATE=true \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  "$PYTHON" scripts/evaluate_force_b2_episode50_ablation.py \
    --checkpoint "$B2" \
    --base-checkpoint "$BASE" \
    --dataset-root "$data" \
    --subtask-sidecar "$sidecar" \
    --pad-subtask-horizon \
    --norm-assets-dir "$NORM_ROOT" \
    --norm-asset-id "$NORM_ID" \
    --force-norm "$FORCE_NORM" \
    --episode 0 \
    --batch-size 4 \
    --seed 20260824 \
    --encoder-width 512 \
    --encoder-heads 8 \
    --encoder-mlp-dim 1024 \
    --force-latent-dim 512 \
    --full-token-force-adapter \
    --full-token-force-adapter-heads 2 \
    --output-dir "$out" \
    >"${ROOT}/logs/${RUN}_rmse_${name}_episode0.log" 2>&1
  test -f "$out/summary.json"
}

run_domain 0 plug "$PLUG_ROOT" "$PLUG_SUBTASK" &
plug_pid=$!
run_domain 1 vase "$VASE_ROOT" "$VASE_SUBTASK" &
vase_pid=$!
wait "$plug_pid"
wait "$vase_pid"

echo "$(date --iso-8601=seconds) plug/vase training-faithful RMSE evaluations complete"
