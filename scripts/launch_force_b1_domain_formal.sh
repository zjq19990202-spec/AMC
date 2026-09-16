#!/usr/bin/env bash
set -euo pipefail

: "${RAM_REPO:?set a unique /dev/shm/atomic_latent_vla_run_* source copy}"
: "${BASE:?set the immutable domain policy params directory}"
: "${DATASET:?set one validated force dataset root}"
: "${RUN:?set a unique run name}"
: "${CUDA_DEVICES:?set the physical GPU list, for example 4,5,6,7}"

PI05_PYTHON=${PI05_PYTHON:-/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python}
NORM_ROOT=${NORM_ROOT:-/mnt/cunchu/zjq/target}
NORM_ID=${NORM_ID:-openpi_norm_union2375_allframes_v1}
FORCE_NORM=${FORCE_NORM:-/mnt/cunchu/zjq/force_assets/force_norm_plug_vase_shared_lr_v1_20260822/norm_stats.json}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/mnt/cunchu/zjq/atomic_pi05_force_runs}
LOG_ROOT=${LOG_ROOT:-/mnt/cunchu/zjq/atomic_pi05_logs}
LOCK_ROOT=${LOCK_ROOT:-${CHECKPOINT_ROOT}/.locks}
STEPS=${STEPS:-6000}
WARMUP_STEPS=${WARMUP_STEPS:-300}
DECAY_STEPS=${DECAY_STEPS:-${STEPS}}
PEAK_LR=${PEAK_LR:-1.0e-4}
DECAY_LR=${DECAY_LR:-1.0e-5}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
KEEP_PERIOD=${KEEP_PERIOD:-5000}
EVAL_BATCHES=${EVAL_BATCHES:-2}
LOG_INTERVAL=${LOG_INTERVAL:-10}
SEED=${SEED:-0}

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
require_file "${RAM_REPO}/scripts/train_force_encoder_b1.py"
require_dir "${BASE}"
require_file "${BASE}/_METADATA"
require_file "${BASE}/manifest.ocdbt"
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
  echo "stage=B1 shared force encoder pretraining"
  echo "source=${RAM_REPO}"
  echo "base_params=${BASE}"
  echo "dataset=${DATASET}"
  echo "prompt_route=PromptFromLeRobotTask source=meta/tasks.parquet mixing=none"
  echo "base_norm=${NORM_ROOT}/${NORM_ID}/norm_stats.json"
  echo "force_norm=${FORCE_NORM}"
  echo "force_normalization=per-channel-q01-q99-no-clip pooled-left-right"
  echo "state_normalization=per-channel-q01-q99-no-clip adapt_to_pi=true"
  echo "state_clip=false action_clip=false inference_output_clip=false"
  echo "load_compensation=false static_sensor_bias_already_removed=true"
  echo "decoder=phase_mlp gru_steps=50 phase_count=4 raw_force_samples=200"
  echo "objective=raw_smoothl1_plus_0.25_mean4_coarse train_fast=false"
  echo "batch=${BATCH_SIZE} per_gpu=$((BATCH_SIZE / DEVICES)) devices=${DEVICES} fsdp_devices=1"
  echo "steps=${STEPS} warmup=${WARMUP_STEPS} peak_lr=${PEAK_LR} decay_steps=${DECAY_STEPS} decay_lr=${DECAY_LR}"
  echo "save_interval=${SAVE_INTERVAL} keep_period=${KEEP_PERIOD} max_to_keep=1"
  sha256sum \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/force.py" \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/model.py" \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/gemma_adapter.py" \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/config.py" \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/force_training_data.py" \
    "${RAM_REPO}/scripts/train_force_encoder_b1.py" \
    "${NORM_ROOT}/${NORM_ID}/norm_stats.json" \
    "${FORCE_NORM}" \
    "${DATASET}/meta/info.json" \
    "${BASE}/_METADATA"
} >"${LOG}.contract.txt"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="${RAM_REPO}/vendor/pi0.5/src:${RAM_REPO}/vendor/pi0.5/packages/openpi-client/src:${RAM_REPO}/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
cd "${RAM_REPO}"

"${PI05_PYTHON}" scripts/train_force_encoder_b1.py \
  --dataset-root "${DATASET}" \
  --norm-assets-dir "${NORM_ROOT}" \
  --norm-asset-id "${NORM_ID}" \
  --force-norm "${FORCE_NORM}" \
  --base-params "${BASE}" \
  --checkpoint-base-dir "${CHECKPOINT_ROOT}" \
  --run-name "${RUN}" \
  --coefficient-target joint_delta \
  --batch-size "${BATCH_SIZE}" \
  --devices "${DEVICES}" \
  --fsdp-devices 1 \
  --num-workers "${NUM_WORKERS}" \
  --max-token-len 200 \
  --future-decoder-stride 4 \
  --future-decoder-kind phase_mlp \
  --encoder-depth 2 \
  --position-base 10000 \
  --history-train-lengths 120 \
  --validation-modulus 10 \
  --validation-remainder 0 \
  --eval-batches "${EVAL_BATCHES}" \
  --steps "${STEPS}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --peak-lr "${PEAK_LR}" \
  --decay-steps "${DECAY_STEPS}" \
  --decay-lr "${DECAY_LR}" \
  --save-interval "${SAVE_INTERVAL}" \
  --keep-period "${KEEP_PERIOD}" \
  --log-interval "${LOG_INTERVAL}" \
  --seed "${SEED}" \
  2>&1 | tee "${LOG}"
