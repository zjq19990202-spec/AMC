#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/mnt/cunchu/zjq/atomic_latent_vla_layerwise}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/mnt/cunchu/zjq/atomic_pi05_runs}
RUN=${RUN:?RUN must name a new checkpoint destination}
FULL_ATOMIC_LOSS_WEIGHT=${FULL_ATOMIC_LOSS_WEIGHT:?FULL_ATOMIC_LOSS_WEIGHT is required}
ATOMIC_PROJECTION_LOSS_WEIGHT=${ATOMIC_PROJECTION_LOSS_WEIGHT:?ATOMIC_PROJECTION_LOSS_WEIGHT is required}
STEPS=${STEPS:-40000}
PHYSICAL_BATCH_SIZE=${PHYSICAL_BATCH_SIZE:-256}

LOG_ROOT=${CHECKPOINT_ROOT}/logs
LOCK_ROOT=${CHECKPOINT_ROOT}/.locks
LOG=${LOG_ROOT}/${RUN}.log
PID_FILE=${LOG_ROOT}/${RUN}.pid
LOCK=${LOCK_ROOT}/${RUN}.lock
RUN_DIR=${CHECKPOINT_ROOT}/${RUN}

mkdir -p "${LOG_ROOT}" "${LOCK_ROOT}"
test -d "${REPO}"
test ! -e "${RUN_DIR}"
test ! -e "${LOG}"

exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "run lock is already held: ${LOCK}" >&2
  exit 1
fi

echo "$$" >"${PID_FILE}"
trap 'rm -f -- "${PID_FILE}"' EXIT

{
  echo "launch_time=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "repo=${REPO}"
  echo "run=${RUN}"
  echo "base_zt_step=27000"
  echo "updates=${STEPS}"
  echo "resume=0"
  echo "global_batch=${PHYSICAL_BATCH_SIZE}"
  echo "devices=8"
  echo "fsdp_devices=1"
  echo "vision_frozen=0"
  echo "full_atomic_loss_weight=${FULL_ATOMIC_LOSS_WEIGHT}"
  echo "atomic_projection_loss_weight=${ATOMIC_PROJECTION_LOSS_WEIGHT}"
  sha256sum \
    "${REPO}/src/atomic_latent_vla/pi05/model.py" \
    "${REPO}/src/atomic_latent_vla/pi05/training_data.py" \
    "${REPO}/scripts/train_atomic_pi05.py"
  exec env \
    REPO="${REPO}" \
    CHECKPOINT_ROOT="${CHECKPOINT_ROOT}" \
    RUN="${RUN}" \
    STEPS="${STEPS}" \
    SMOKE_STEPS=0 \
    RESUME=0 \
    PHYSICAL_BATCH_SIZE="${PHYSICAL_BATCH_SIZE}" \
    FULL_ATOMIC_LOSS_WEIGHT="${FULL_ATOMIC_LOSS_WEIGHT}" \
    ATOMIC_PROJECTION_LOSS_WEIGHT="${ATOMIC_PROJECTION_LOSS_WEIGHT}" \
    "${REPO}/scripts/run_target_v3_offline_zm_from_zt27k_q005_kl0025_ddp8_bs128_40k.sh"
} >"${LOG}" 2>&1
