#!/usr/bin/env bash
set -euo pipefail

: "${RAM_REPO:?set the unique per-host /dev/shm source directory}"
: "${SOURCE_CHECKPOINT:?set the exact old 61K full checkpoint step directory}"
: "${ZT_PARAMS:?set the matching ZT1K params directory used for structural initialization}"
: "${ZT_TEACHER:?set the existing matching Q1/codebook teacher sidecar}"
: "${RUN:?set a unique destination run name}"
: "${FULL_ATOMIC_LOSS_WEIGHT:?set the Q1 teacher cosine weight}"
: "${ATOMIC_PROJECTION_LOSS_WEIGHT:?set the Drop Top-5 weighted-cos weight}"

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/mnt/cunchu/zjq/atomic_pi05_runs}
LOCK_ROOT=${CHECKPOINT_ROOT}/.locks
ATOMIC_COMPOSITION_SIDECAR=${ATOMIC_COMPOSITION_SIDECAR:-fk_horizon_3hz_gate_top5_stay_v2}
CLEAN_RAM_ON_EXIT=${CLEAN_RAM_ON_EXIT:-1}

mkdir -p "${LOCK_ROOT}"
exec 9>"${LOCK_ROOT}/${RUN}.lock"
if ! flock -n 9; then
  echo "training lock is already held: ${RUN}" >&2
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
test -d "${SOURCE_CHECKPOINT}/train_state"
test -d "${SOURCE_CHECKPOINT}/params"
test -d "${ZT_PARAMS}"
test -f "${ZT_TEACHER}/manifest.json"
test -f "${ZT_TEACHER}/codebook.npy"
test -f "${ZT_TEACHER}/directions.npy"
test ! -e "${CHECKPOINT_ROOT}/${RUN}"

echo "launch_time=$(date --iso-8601=seconds)"
echo "ram_repo=${RAM_REPO}"
echo "source_checkpoint=${SOURCE_CHECKPOINT}"
echo "zt_params=${ZT_PARAMS}"
echo "zt_teacher=${ZT_TEACHER}"
echo "run=${RUN}"
echo "global_batch=256 per_gpu_batch=32 devices=8 mesh=ddp8"
echo "norm_asset_id=openpi_norm_compact_accepted_v3"
echo "coefficient_target=joint_delta max_token_len=200"
echo "prompt=subtask_only normalization=q01_q99_affine_unbounded vision_frozen=false"
echo "q1_weight=${FULL_ATOMIC_LOSS_WEIGHT} drop_weight=${ATOMIC_PROJECTION_LOSS_WEIGHT}"
echo "optimizer=restored_full_adam schedule=floor_lr_2.5e-6"

REPO="${RAM_REPO}" \
ZT_PARAMS="${ZT_PARAMS}" \
ZT_TEACHER="${ZT_TEACHER}" \
CHECKPOINT_ROOT="${CHECKPOINT_ROOT}" \
RUN="${RUN}" \
STEPS=5000 \
PHYSICAL_BATCH_SIZE=256 \
FULL_ATOMIC_LOSS_WEIGHT="${FULL_ATOMIC_LOSS_WEIGHT}" \
ATOMIC_PROJECTION_LOSS_WEIGHT="${ATOMIC_PROJECTION_LOSS_WEIGHT}" \
ATOMIC_COMPOSITION_SIDECAR="${ATOMIC_COMPOSITION_SIDECAR}" \
RESTORE_FULL_STATE="${SOURCE_CHECKPOINT}" \
INITIAL_STEP=61000 \
PHASE_START_STEP=29000 \
ATOMIC_PROMPT_PROBABILITY=0.0 \
RESUME=0 \
bash "${RAM_REPO}/scripts/run_target_v3_offline_zm_from_zt27k_q005_kl0025_ddp8_bs128_40k.sh"

echo "training_complete=$(date --iso-8601=seconds)"
