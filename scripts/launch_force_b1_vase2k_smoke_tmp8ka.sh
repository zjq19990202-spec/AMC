#!/usr/bin/env bash
set -euo pipefail

: "${RAM_REPO:?set the unique /dev/shm source copy}"

PI05_PYTHON=${PI05_PYTHON:-/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python}
BASE=${BASE:-/mnt/cunchu/zjq/atomic_pi05_runs/union2375_vase_zm25k_bs128_ddp4_resume1000_r2_20260823/union2375_vase_zm25k_bs128_ddp4_resume1000_r2_20260823/2000/params}
DATASET=/mnt/cunchu/zjq/target/lerobot_v3_4to1_vase_force
NORM_ROOT=/mnt/cunchu/zjq/target
NORM_ID=openpi_norm_union2375_allframes_v1
FORCE_NORM=/mnt/cunchu/zjq/force_assets/force_norm_plug_vase_shared_lr_v1_20260822/norm_stats.json
CHECKPOINT_ROOT=/mnt/cunchu/zjq/atomic_pi05_smoke
LOG_ROOT=/mnt/cunchu/zjq/atomic_pi05_logs
LOCK_ROOT=/mnt/cunchu/zjq/atomic_pi05_smoke/.locks
RUN=${RUN:-force_b1_vase2k_raw200_smoke_20260823}
DECODER_STRIDE=${DECODER_STRIDE:-4}
DECODER_KIND=${DECODER_KIND:-phase_mlp}
ENCODER_DEPTH=${ENCODER_DEPTH:-2}
POSITION_BASE=${POSITION_BASE:-10000}
HISTORY_LENGTHS=${HISTORY_LENGTHS:-120}
STEPS=${STEPS:-20}
BATCH_SIZE=${BATCH_SIZE:-4}
WARMUP_STEPS=${WARMUP_STEPS:-5}
DECAY_STEPS=${DECAY_STEPS:-20}
PEAK_LR=${PEAK_LR:-1.0e-4}
VALIDATION_MODULUS=${VALIDATION_MODULUS:-10}
VALIDATION_REMAINDER=${VALIDATION_REMAINDER:-0}
EVAL_BATCHES=${EVAL_BATCHES:-20}
LOG=${LOG_ROOT}/${RUN}.log

mkdir -p "${CHECKPOINT_ROOT}" "${LOG_ROOT}" "${LOCK_ROOT}"
exec 9>"${LOCK_ROOT}/${RUN}.lock"
flock -n 9 || { echo "lock held: ${RUN}" >&2; exit 1; }

cleanup() {
  status=$?
  if [[ "${KEEP_RAM:-0}" != 1 && "${RAM_REPO}" == /dev/shm/atomic_latent_vla_run_* ]]; then
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
require_dir "${BASE}"
require_file "${DATASET}/meta/info.json"
require_file "${NORM_ROOT}/${NORM_ID}/norm_stats.json"
require_file "${FORCE_NORM}"
[[ ! -e "${CHECKPOINT_ROOT}/${RUN}" ]] || {
  echo "checkpoint destination already exists: ${CHECKPOINT_ROOT}/${RUN}" >&2
  exit 1
}

{
  echo "started=$(date --iso-8601=seconds)"
  echo "host=$(hostname) gpu=0"
  echo "purpose=force_b1_raw200_vase_smoke"
  echo "base_params=${BASE}"
  echo "dataset=${DATASET}"
  echo "norm_root=${NORM_ROOT} norm_id=${NORM_ID}"
  echo "force_norm=${FORCE_NORM}"
  echo "load_compensation=false static_sensor_bias_already_removed=true"
  echo "state_views=right:[right8,left8],left:[left8,right8]"
  echo "future_objective=raw200_smoothl1_plus_0.25_mean4_coarse50"
  echo "history_train_lengths=${HISTORY_LENGTHS} position_base=${POSITION_BASE} end_aligned=true"
  echo "future_decoder_stride=${DECODER_STRIDE}"
  echo "future_decoder_kind=${DECODER_KIND}"
  echo "force_encoder_depth=${ENCODER_DEPTH}"
  echo "episode_holdout=index_mod_${VALIDATION_MODULUS}_eq_${VALIDATION_REMAINDER} eval_batches=${EVAL_BATCHES}"
  echo "batch=${BATCH_SIZE} devices=1 fsdp_devices=1 workers=2 steps=${STEPS} checkpoint=false"
  echo "warmup=${WARMUP_STEPS} peak_lr=${PEAK_LR} decay_steps=${DECAY_STEPS} decay_lr=1e-5"
  sha256sum \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/force.py" \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/model.py" \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/config.py" \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/force_training_data.py" \
    "${RAM_REPO}/scripts/train_force_encoder_b1.py" \
    "${RAM_REPO}/scripts/launch_force_b1_vase2k_smoke_tmp8ka.sh"
} >"${LOG}.contract.txt"

export CUDA_VISIBLE_DEVICES=0
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
  --devices 1 \
  --fsdp-devices 1 \
  --num-workers 2 \
  --max-token-len 200 \
  --future-decoder-stride "${DECODER_STRIDE}" \
  --future-decoder-kind "${DECODER_KIND}" \
  --encoder-depth "${ENCODER_DEPTH}" \
  --position-base "${POSITION_BASE}" \
  --history-train-lengths "${HISTORY_LENGTHS}" \
  --validation-modulus "${VALIDATION_MODULUS}" \
  --validation-remainder "${VALIDATION_REMAINDER}" \
  --eval-batches "${EVAL_BATCHES}" \
  --steps "${STEPS}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --peak-lr "${PEAK_LR}" \
  --decay-steps "${DECAY_STEPS}" \
  --decay-lr 1.0e-5 \
  --log-interval 1 \
  --skip-checkpoint \
  2>&1 | tee "${LOG}"
