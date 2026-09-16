#!/usr/bin/env bash
set -euo pipefail

PYTHON=/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python
SOURCE=/mnt/cunchu/zjq/atomic_latent_vla_snapshots/force_spherical_b2_subtaskpad_matched50k_20260904
RAM_SOURCE=/dev/shm/atomic_latent_vla_eval_b2_subtaskpad_step2000_20260904
ROOT=/mnt/cunchu/zjq/atomic_pi05_force_runs
RUN=force_spherical45h20_crossatt2_finalzm40_subtaskpad_from_matchedspherical50k_b2s6000_20260904
B2=${ROOT}/${RUN}/${RUN}/2000
BASE_RUN=union2375_afro50k_matched_spherical20_w005_from25k_to50k_fsdp8_bs256_20260902
BASE=/mnt/cunchu/zjq/atomic_pi05_runs/${BASE_RUN}/${BASE_RUN}/50000
OUT=${ROOT}/evals/${RUN}/step2000_exact_subtaskpad_diagnosis_seed20260824
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

test -f "$B2/params/_METADATA"
test -f "$BASE/params/_METADATA"
test ! -e "$RAM_SOURCE"
test ! -e "$OUT"
cp -a "$SOURCE" "$RAM_SOURCE"
cd "$RAM_SOURCE"
export PYTHONPATH="$RAM_SOURCE/vendor/pi0.5/src:$RAM_SOURCE/vendor/pi0.5/packages/openpi-client/src:$RAM_SOURCE/src"
mkdir -p "$OUT"

episode_eval() {
  local gpu=$1 domain=$2 data=$3 sidecar=$4
  local target=${OUT}/episode50_${domain}_episode0
  mkdir -p "$target"
  CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  "$PYTHON" scripts/evaluate_force_b2_episode50_ablation.py \
    --checkpoint "$B2" --base-checkpoint "$BASE" \
    --dataset-root "$data" --subtask-sidecar "$sidecar" --pad-subtask-horizon \
    --norm-assets-dir "$NORM_ROOT" --norm-asset-id "$NORM_ID" --force-norm "$FORCE_NORM" \
    --episode 0 --batch-size 4 --seed 20260824 \
    --encoder-width 512 --encoder-heads 8 --encoder-mlp-dim 1024 --force-latent-dim 512 \
    --full-token-force-adapter --full-token-force-adapter-heads 2 \
    --output-dir "$target" >"${target}.log" 2>&1
}

offset_eval() {
  local gpu=$1 domain=$2 data=$3 sidecar=$4
  local target=${OUT}/offsets_${domain}_episode0_first16
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

episode_eval 0 plug "$PLUG_ROOT" "$PLUG_SUBTASK" & p0=$!
episode_eval 1 vase "$VASE_ROOT" "$VASE_SUBTASK" & p1=$!
offset_eval 2 plug "$PLUG_ROOT" "$PLUG_SUBTASK" & p2=$!
offset_eval 3 vase "$VASE_ROOT" "$VASE_SUBTASK" & p3=$!
wait "$p0" "$p1" "$p2" "$p3"
date --iso-8601=seconds >"$OUT/COMPLETE"
