#!/usr/bin/env bash
set -euo pipefail

: "${RAM_REPO:?set a unique /dev/shm/atomic_latent_vla_run_* source copy}"
: "${B1_PARAMS:?set the completed domain B1 params directory}"
: "${DATASET:?set one validated force dataset root}"
: "${RUN:?set a unique run name}"
: "${CUDA_DEVICES:?set the physical GPU list, for example 0,1,2,3,4,5,6,7}"

PI05_PYTHON=${PI05_PYTHON:-/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python}
NORM_ROOT=${NORM_ROOT:-/mnt/cunchu/zjq/target}
NORM_ID=${NORM_ID:-openpi_norm_union2375_allframes_v1}
FORCE_NORM=${FORCE_NORM:-/mnt/cunchu/zjq/force_assets/force_norm_plug_vase_shared_lr_v1_20260822/norm_stats.json}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/mnt/cunchu/zjq/atomic_pi05_force_b2_runs}
LOG_ROOT=${LOG_ROOT:-/mnt/cunchu/zjq/atomic_pi05_logs}
LOCK_ROOT=${LOCK_ROOT:-${CHECKPOINT_ROOT}/.locks}
STEPS=${STEPS:-6000}
WARMUP_STEPS=${WARMUP_STEPS:-300}
DECAY_STEPS=${DECAY_STEPS:-${STEPS}}
PEAK_LR=${PEAK_LR:-3.0e-5}
DECAY_LR=${DECAY_LR:-3.0e-6}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
KEEP_PERIOD=${KEEP_PERIOD:-5000}
EVAL_BATCHES=${EVAL_BATCHES:-2}
LOG_INTERVAL=${LOG_INTERVAL:-10}
SEED=${SEED:-0}
FSDP_DEVICES=${FSDP_DEVICES:-1}
XLA_MEM_FRACTION=${XLA_MEM_FRACTION:-0.75}
SKIP_CHECKPOINT=${SKIP_CHECKPOINT:-0}
FUTURE_FORCE_LOSS_WEIGHT=${FUTURE_FORCE_LOSS_WEIGHT:-0}
FORCE_UPDATE_ACTION_STEPS=${FORCE_UPDATE_ACTION_STEPS:-10}
FORCE_UPDATE_OFFSETS=${FORCE_UPDATE_OFFSETS:-0,10,20,30,40}
FREEZE_ACTION_PATH=${FREEZE_ACTION_PATH:-0}

IFS=',' read -r -a gpu_list <<<"${CUDA_DEVICES}"
DEVICES=${#gpu_list[@]}
if (( DEVICES <= 0 )); then
  echo "CUDA_DEVICES must contain at least one GPU" >&2
  exit 1
fi
BATCH_SIZE=${BATCH_SIZE:-$((DEVICES * 32))}
NUM_WORKERS=${NUM_WORKERS:-$((DEVICES * 4))}
if (( BATCH_SIZE % DEVICES != 0 )); then
  echo "BATCH_SIZE must be divisible by DEVICES" >&2
  exit 1
fi
if (( DEVICES % FSDP_DEVICES != 0 )); then
  echo "FSDP_DEVICES must divide DEVICES" >&2
  exit 1
fi

LOG=${LOG_ROOT}/${RUN}.log
RUN_DIR=${CHECKPOINT_ROOT}/${RUN}

cleanup() {
  status=$?
  if [[ "${RAM_REPO}" == /dev/shm/atomic_latent_vla_run_* ]]; then
    cd /
    rm -rf -- "${RAM_REPO}"
  fi
  exit "${status}"
}
trap cleanup EXIT

require_file() { [[ -f "$1" ]] || { echo "missing file: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "missing directory: $1" >&2; exit 1; }; }

[[ -x "${PI05_PYTHON}" ]] || { echo "python is not executable: ${PI05_PYTHON}" >&2; exit 1; }
require_dir "${RAM_REPO}/src"
require_dir "${RAM_REPO}/vendor/pi0.5/src/openpi"
require_file "${RAM_REPO}/scripts/train_force_stage_b2.py"
require_file "${B1_PARAMS}/_METADATA"
require_file "${B1_PARAMS}/manifest.ocdbt"
require_file "${DATASET}/meta/info.json"
require_file "${NORM_ROOT}/${NORM_ID}/norm_stats.json"
require_file "${FORCE_NORM}"

mkdir -p "${CHECKPOINT_ROOT}" "${LOG_ROOT}" "${LOCK_ROOT}"
[[ ! -e "${RUN_DIR}" ]] || {
  echo "checkpoint destination already exists: ${RUN_DIR}" >&2
  exit 1
}
exec 9>"${LOCK_ROOT}/${RUN}.lock"
flock -n 9 || { echo "lock held: ${RUN}" >&2; exit 1; }

{
  echo "started=$(date --iso-8601=seconds)"
  echo "host=$(hostname) cuda_visible_devices=${CUDA_DEVICES}"
  echo "stage=B2 force-conditioned action flow"
  echo "source=${RAM_REPO}"
  echo "b1_params=${B1_PARAMS}"
  echo "merge=restore_complete_B1_parameter_tree fresh_B2_Adam=true"
  echo "dataset=${DATASET}"
  echo "prompt_route=PromptFromLeRobotTask source=meta/tasks.parquet mixing=none"
  echo "base_norm=${NORM_ROOT}/${NORM_ID}/norm_stats.json"
  echo "force_norm=${FORCE_NORM}"
  echo "force_normalization=per-channel-q01-q99-no-clip pooled-left-right"
  echo "state_normalization=per-channel-q01-q99-no-clip adapt_to_pi=true"
  echo "state_clip=false action_clip=false inference_output_clip=false optimizer_gradient_clip=1.0"
  echo "load_compensation=false static_sensor_bias_already_removed=true"
  echo "objective=flow_1.0+future_${FUTURE_FORCE_LOSS_WEIGHT}+delta_z_reg_1e-4 train_fast=true"
  echo "rtc=training_time_clean_prefix tokenwise_flow_time=true learned_commit_embedding=false committed_prefix_weight=0 suffix_weight=1"
  echo "force_update_action_steps=${FORCE_UPDATE_ACTION_STEPS} offsets=${FORCE_UPDATE_OFFSETS}"
  echo "freeze_action_path=${FREEZE_ACTION_PATH}"
  echo "batch=${BATCH_SIZE} per_gpu=$((BATCH_SIZE / DEVICES)) devices=${DEVICES} fsdp_devices=${FSDP_DEVICES} mesh=$((DEVICES / FSDP_DEVICES))x${FSDP_DEVICES}"
  echo "steps=${STEPS} warmup=${WARMUP_STEPS} peak_lr=${PEAK_LR} decay_steps=${DECAY_STEPS} decay_lr=${DECAY_LR}"
  echo "save_interval=${SAVE_INTERVAL} keep_period=${KEEP_PERIOD} max_to_keep=1 skip_checkpoint=${SKIP_CHECKPOINT}"
  sha256sum \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/force.py" \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/model.py" \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/gemma_adapter.py" \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/config.py" \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/force_training_data.py" \
    "${RAM_REPO}/scripts/train_force_stage_b2.py" \
    "${NORM_ROOT}/${NORM_ID}/norm_stats.json" \
    "${FORCE_NORM}" \
    "${DATASET}/meta/info.json" \
    "${B1_PARAMS}/_METADATA"
} >"${LOG}.contract.txt"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="${RAM_REPO}/vendor/pi0.5/src:${RAM_REPO}/vendor/pi0.5/packages/openpi-client/src:${RAM_REPO}/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION}"
cd "${RAM_REPO}"

args=(
  --dataset-root "${DATASET}"
  --norm-assets-dir "${NORM_ROOT}"
  --norm-asset-id "${NORM_ID}"
  --force-norm "${FORCE_NORM}"
  --b1-params "${B1_PARAMS}"
  --checkpoint-base-dir "${CHECKPOINT_ROOT}"
  --run-name "${RUN}"
  --coefficient-target joint_delta
  --batch-size "${BATCH_SIZE}"
  --devices "${DEVICES}"
  --fsdp-devices "${FSDP_DEVICES}"
  --num-workers "${NUM_WORKERS}"
  --max-token-len 200
  --validation-modulus 10
  --validation-remainder 0
  --eval-batches "${EVAL_BATCHES}"
  --steps "${STEPS}"
  --warmup-steps "${WARMUP_STEPS}"
  --peak-lr "${PEAK_LR}"
  --decay-steps "${DECAY_STEPS}"
  --decay-lr "${DECAY_LR}"
  --future-force-loss-weight "${FUTURE_FORCE_LOSS_WEIGHT}"
  --force-update-action-steps "${FORCE_UPDATE_ACTION_STEPS}"
  --force-update-offsets "${FORCE_UPDATE_OFFSETS}"
  --flow-loss-weight 1.0
  --delta-regularization-weight 1.0e-4
  --save-interval "${SAVE_INTERVAL}"
  --keep-period "${KEEP_PERIOD}"
  --log-interval "${LOG_INTERVAL}"
  --seed "${SEED}"
)
if [[ "${FREEZE_ACTION_PATH}" == 1 ]]; then
  args+=(--freeze-action-path)
fi
if [[ "${SKIP_CHECKPOINT}" == 1 ]]; then
  args+=(--skip-checkpoint)
fi

"${PI05_PYTHON}" scripts/train_force_stage_b2.py "${args[@]}" 2>&1 | tee "${LOG}"
