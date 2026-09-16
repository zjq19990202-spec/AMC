#!/usr/bin/env bash
set -euo pipefail

# Required per branch.
: "${RAM_REPO:?set RAM_REPO to the per-host /dev/shm source directory}"
: "${SOURCE_PARAMS:?set SOURCE_PARAMS to the source 61K params directory}"
: "${ZT_RUN:?set a unique ZT run name}"
: "${ZM_RUN:?set a unique ZM run name}"
: "${FULL_ATOMIC_LOSS_WEIGHT:?set the ZM Q1 distillation weight}"
: "${ATOMIC_PROJECTION_LOSS_WEIGHT:?set the ZM drop weighted-cos weight}"

OPENPI_ROOT=${OPENPI_ROOT:-${RAM_REPO}/vendor/pi0.5}
PI05_PYTHON=${PI05_PYTHON:-/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python}

DATASET=${DATASET:-/mnt/cunchu/zjq/target/lerobot_compact_accepted_final}
NORM_ROOT=${NORM_ROOT:-/mnt/cunchu/zjq/target}
NORM_ID=${NORM_ID:-openpi_norm_compact_accepted_v3}
TCP_NORM=${TCP_NORM:-${DATASET}/meta/tcp_twist_norm_bimanual_tcp200.json}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/mnt/cunchu/zjq/atomic_pi05_runs}
TEACHER_ROOT=${TEACHER_ROOT:-${CHECKPOINT_ROOT}/teachers}
ZT_TEACHER=${ZT_TEACHER:-${TEACHER_ROOT}/${ZT_RUN}}
ZT_CHECKPOINT=${CHECKPOINT_ROOT}/${ZT_RUN}/${ZT_RUN}/1000
LOCK_ROOT=${CHECKPOINT_ROOT}/.locks
LOG_ROOT=${CHECKPOINT_ROOT}/logs
ATOMIC_COMPOSITION_SIDECAR=${ATOMIC_COMPOSITION_SIDECAR:-fk_horizon_3hz_gate_top5_stay_v2}
CLEAN_RAM_ON_EXIT=${CLEAN_RAM_ON_EXIT:-1}

mkdir -p "${LOCK_ROOT}" "${LOG_ROOT}" "${TEACHER_ROOT}"
exec 9>"${LOCK_ROOT}/${ZM_RUN}.pipeline.lock"
if ! flock -n 9; then
  echo "pipeline lock is already held: ${ZM_RUN}" >&2
  exit 1
fi

cleanup() {
  local status=$?
  if [[ "${CLEAN_RAM_ON_EXIT}" == 1 && "${RAM_REPO}" == /dev/shm/atomic_latent_vla_run_* ]]; then
    cd /
    rm -rf -- "${RAM_REPO}"
  fi
  exit "${status}"
}
trap cleanup EXIT

test -d "${RAM_REPO}/src"
test -d "${SOURCE_PARAMS}"
test -f "${DATASET}/meta/info.json"
test -f "${NORM_ROOT}/${NORM_ID}/norm_stats.json"
test -f "${TCP_NORM}"
test ! -e "${CHECKPOINT_ROOT}/${ZT_RUN}"
test ! -e "${CHECKPOINT_ROOT}/${ZM_RUN}"
test ! -e "${ZT_TEACHER}"

export PYTHONPATH=${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${RAM_REPO}/src
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
cd "${RAM_REPO}"

echo "pipeline_start=$(date --iso-8601=seconds)"
echo "source_params=${SOURCE_PARAMS}"
echo "zt_run=${ZT_RUN}"
echo "zt_teacher=${ZT_TEACHER}"
echo "zm_run=${ZM_RUN}"
echo "zm_prompt=subtask_only"
echo "full_atomic_loss_weight=${FULL_ATOMIC_LOSS_WEIGHT}"
echo "drop_weighted_cos_weight=${ATOMIC_PROJECTION_LOSS_WEIGHT}"

# Stage 1: initialize from the complete 61K model parameters, but use a fresh
# optimizer/global step so all 1,000 updates modify both Q1 and the codebook.
"${PI05_PYTHON}" scripts/train_atomic_zt_fast.py \
  --dataset-root "${DATASET}" \
  --norm-assets-dir "${NORM_ROOT}" \
  --norm-asset-id "${NORM_ID}" \
  --tcp-twist-norm "${TCP_NORM}" \
  --coefficient-target joint_delta \
  --base-params "${SOURCE_PARAMS}" \
  --checkpoint-base-dir "${CHECKPOINT_ROOT}" \
  --run-name "${ZT_RUN}" \
  --batch-size 256 \
  --devices 8 \
  --fsdp-devices 1 \
  --max-token-len 200 \
  --num-workers 64 \
  --steps 1000 \
  --save-interval 1000 \
  --log-interval 10 \
  --warmup-steps 1000 \
  --peak-lr 2.5e-5 \
  --decay-lr 2.5e-6 \
  --lr-decay-steps 30000 \
  --codebook-freeze-step 1000 \
  --flow-loss-weight 1.0 \
  --coefficient-loss-weight 0.0 \
  --atomic-loss-weight 0.1 \
  --atomic-projection-loss-weight 0.1 \
  --codebook-loss-weight 0.0

test -d "${ZT_CHECKPOINT}/params"

# Stage 2: the teacher sidecar stores directions in the new codebook frame.
# It must be regenerated rather than reusing the old ZT27K sidecar.
env \
  CUDA_VISIBLE_DEVICES=0 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH="${PYTHONPATH}" \
  "${PI05_PYTHON}" scripts/precompute_target_zt_q_teacher.py \
    --checkpoint "${ZT_CHECKPOINT}" \
    --dataset-root "${DATASET}" \
    --norm-assets-dir "${NORM_ROOT}" \
    --norm-asset-id "${NORM_ID}" \
    --output "${ZT_TEACHER}" \
    --batch-size 256 \
    --num-workers 32 \
    --max-token-len 200 \
    --coefficient-target joint_delta

test -f "${ZT_TEACHER}/manifest.json"
test -f "${ZT_TEACHER}/codebook.npy"
test -f "${ZT_TEACHER}/directions.npy"
test -f "${ZT_TEACHER}/valid.npy"

# Stage 3: subtask-only ZM. Strict atomic arms distill the freshly generated
# Q1 target; drop arms use the existing stay-aware Top-5 weighted-cos target.
REPO="${RAM_REPO}" \
ZT_RUN="${ZT_RUN}" \
ZT_PARAMS="${ZT_CHECKPOINT}/params" \
ZT_TEACHER="${ZT_TEACHER}" \
CHECKPOINT_ROOT="${CHECKPOINT_ROOT}" \
RUN="${ZM_RUN}" \
STEPS=5000 \
PHYSICAL_BATCH_SIZE=256 \
FULL_ATOMIC_LOSS_WEIGHT="${FULL_ATOMIC_LOSS_WEIGHT}" \
ATOMIC_PROJECTION_LOSS_WEIGHT="${ATOMIC_PROJECTION_LOSS_WEIGHT}" \
ATOMIC_COMPOSITION_SIDECAR="${ATOMIC_COMPOSITION_SIDECAR}" \
INITIAL_STEP=62000 \
PHASE_START_STEP=29000 \
ATOMIC_PROMPT_PROBABILITY=0.0 \
RESUME=0 \
bash scripts/run_target_v3_offline_zm_from_zt27k_q005_kl0025_ddp8_bs128_40k.sh

echo "pipeline_complete=$(date --iso-8601=seconds)"
