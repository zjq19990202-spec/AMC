#!/usr/bin/env bash
set -euo pipefail

SOURCE_SNAPSHOT=${SOURCE_SNAPSHOT:-/mnt/cunchu/zjq/atomic_latent_vla_snapshots/force_b2_unfreeze_ae_zf_20260907}
RAM_SOURCE=${RAM_SOURCE:-/dev/shm/atomic_latent_vla_run_force_b2_unfreeze_ae_zf_20260907}
PYTHON=/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python
RUN=${RUN:-force_spherical_b2s6000_unfreeze_ae_zf_rtc5k_20260907}
SOURCE_RUN=force_spherical45h20_crossatt2_finalzm40_subtaskpad_from_matchedspherical50k_b2s6000_20260904
SOURCE_PARAMS=/mnt/cunchu/zjq/atomic_pi05_force_runs/${SOURCE_RUN}/${SOURCE_RUN}/6000/params
CHECKPOINT_ROOT=/mnt/cunchu/zjq/atomic_pi05_force_runs
LOG_ROOT=${CHECKPOINT_ROOT}/logs
PIPELINE_LOG=${LOG_ROOT}/${RUN}.pipeline.log
SMOKE_LOG=${LOG_ROOT}/${RUN}_smoke1.log
FORMAL_LOG=${LOG_ROOT}/${RUN}.log
LOCK=${CHECKPOINT_ROOT}/.locks/${RUN}.lock
PLUG_ROOT=/mnt/cunchu/zjq/target/lerobot_v3_4to1_plug_force
VASE_ROOT=/mnt/cunchu/zjq/target/lerobot_v3_4to1_vase_force
PLUG_SUBTASK=/mnt/cunchu/zjq/force_assets/subtask_sidecars_plug_vase_20260904/plug_episode_subtasks.jsonl
VASE_SUBTASK=/mnt/cunchu/zjq/force_assets/subtask_sidecars_plug_vase_20260904/vase_episode_subtasks.jsonl
FORCE_NORM=/mnt/cunchu/zjq/force_assets/force_norm_plug_vase_shared_lr_v1_20260822/norm_stats.json

cleanup() {
  status=$?
  if [[ "$RAM_SOURCE" == /dev/shm/atomic_latent_vla_run_* && -d "$RAM_SOURCE" ]]; then
    cd /
    rm -rf -- "$RAM_SOURCE"
  fi
  exit "$status"
}
trap cleanup EXIT

mkdir -p "$LOG_ROOT" "$(dirname "$LOCK")"
exec 9>"$LOCK"
flock -n 9 || { echo "run lock already held: $LOCK" >&2; exit 70; }
exec >>"$PIPELINE_LOG" 2>&1

test -x "$PYTHON"
test -d "$SOURCE_SNAPSHOT/src"
test -f "$SOURCE_SNAPSHOT/scripts/train_force_stage_b2.py"
test -f "$SOURCE_PARAMS/_METADATA"
test -f "$SOURCE_PARAMS/manifest.ocdbt"
test ! -e "$CHECKPOINT_ROOT/$RUN"
test ! -e "$RAM_SOURCE"

mapfile -t used_mb < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
for value in "${used_mb[@]}"; do
  if (( value > 1000 )); then
    echo "GPU is not free; memory_mb=${used_mb[*]}" >&2
    exit 71
  fi
done

cp -a "$SOURCE_SNAPSHOT" "$RAM_SOURCE"
cd "$RAM_SOURCE"
export PYTHONPATH="$RAM_SOURCE/vendor/pi0.5/src:$RAM_SOURCE/vendor/pi0.5/packages/openpi-client/src:$RAM_SOURCE/src"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

common_args=(
  --dataset-root "$PLUG_ROOT"
  --dataset-root "$VASE_ROOT"
  --subtask-sidecar "$PLUG_SUBTASK"
  --subtask-sidecar "$VASE_SUBTASK"
  --pad-subtask-horizon
  --norm-assets-dir /mnt/cunchu/zjq/target
  --norm-asset-id openpi_norm_union2375_allframes_v1
  --force-norm "$FORCE_NORM"
  --b1-params "$SOURCE_PARAMS"
  --checkpoint-base-dir "$CHECKPOINT_ROOT"
  --coefficient-target joint_delta
  --batch-size 256
  --devices 8
  --fsdp-devices 1
  --num-workers 32
  --max-token-len 192
  --encoder-width 512
  --encoder-depth 2
  --encoder-heads 8
  --encoder-mlp-dim 1024
  --force-latent-dim 512
  --force-only-zf
  --future-condition-on-zm
  --full-token-force-adapter
  --full-token-force-adapter-heads 2
  --unfreeze-action-path
  --unfreeze-zf-path
  --no-layerwise-atomic-flow
  --spherical-visual-latent
  --visual-max-update-angle-deg 45
  --spherical-force-update
  --force-max-update-angle-deg 45
  --force-rotation-loss-weight 0.005
  --force-rotation-free-angle-deg 20
  --steps 5000
  --warmup-steps 200
  --peak-lr 5e-6
  --decay-steps 5000
  --decay-lr 1e-6
  --future-force-loss-weight 0
  --flow-loss-weight 1
  --delta-regularization-weight 1e-4
  --force-improvement-weight 1
  --force-improvement-margin 0.001
  --force-update-action-steps 10
  --force-update-offsets 0,10,20,30,40
  --save-interval 1000
  --keep-period 1000
  --log-interval 10
  --validation-modulus 10
  --validation-remainder 0
  --eval-batches 2
)

echo "$(date --iso-8601=seconds) smoke: B2 6K -> unfreeze Action Expert + zF"
"$PYTHON" scripts/train_force_stage_b2.py "${common_args[@]}" \
  --run-name "${RUN}_smoke1" --steps 1 --log-interval 1 --skip-checkpoint \
  >"$SMOKE_LOG" 2>&1
grep -q 'unfreeze_action_path=True unfreeze_zf_path=True' "$SMOKE_LOG"
grep -q 'step=1 ' "$SMOKE_LOG"
if grep -Eq 'Involuntary full rematerialization|RESOURCE_EXHAUSTED|CUDA_ERROR|NCCL.*(error|fail)|out of memory|Traceback' "$SMOKE_LOG"; then
  echo "smoke violated error gate" >&2
  exit 72
fi

echo "$(date --iso-8601=seconds) formal 5K launch"
"$PYTHON" scripts/train_force_stage_b2.py "${common_args[@]}" --run-name "$RUN" >"$FORMAL_LOG" 2>&1
test -f "$CHECKPOINT_ROOT/$RUN/$RUN/5000/params/_METADATA"
test -f "$CHECKPOINT_ROOT/$RUN/$RUN/5000/params/manifest.ocdbt"
echo "$(date --iso-8601=seconds) completed"
