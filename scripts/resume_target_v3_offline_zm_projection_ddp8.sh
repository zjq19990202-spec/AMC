#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/mnt/cunchu/zjq/atomic_pi05_runs}
RUN=${RUN:?RUN is required}
REPO=${REPO:?REPO is required}
RESUME_STEP=${RESUME_STEP:?RESUME_STEP is required}
FINAL_STEP=${FINAL_STEP:-67000}
FULL_ATOMIC_LOSS_WEIGHT=${FULL_ATOMIC_LOSS_WEIGHT:?FULL_ATOMIC_LOSS_WEIGHT is required}
ATOMIC_PROJECTION_LOSS_WEIGHT=${ATOMIC_PROJECTION_LOSS_WEIGHT:?ATOMIC_PROJECTION_LOSS_WEIGHT is required}
SOURCE_COMMIT=${SOURCE_COMMIT:-unknown}
ATOMIC_COMPOSITION_SIDECAR=${ATOMIC_COMPOSITION_SIDECAR:-fk_horizon_3hz_gate_top5_v1}
CLEANUP_REPO=${CLEANUP_REPO:-0}
LOG=${BASE}/logs/${RUN}.projection_resume_from_${RESUME_STEP}.log
PID_FILE=${BASE}/logs/${RUN}.projection_resume.pid
LOCK=${BASE}/.locks/${RUN}.lock
CHECKPOINT_DIR=${BASE}/${RUN}/${RUN}/${RESUME_STEP}

if (( FINAL_STEP <= RESUME_STEP )); then
  echo "FINAL_STEP must be greater than RESUME_STEP" >&2
  exit 64
fi
STEPS=$((FINAL_STEP - RESUME_STEP))

mkdir -p "${BASE}/logs" "${BASE}/.locks"
test -d "${REPO}"
test -d "${CHECKPOINT_DIR}/params"

exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "run lock is already held: ${LOCK}" >&2
  exit 70
fi

cleanup() {
  if (( CLEANUP_REPO > 0 )); then
    case "${REPO}" in
      /dev/shm/atomic_latent_vla_run_*) rm -rf -- "${REPO}" ;;
      *) echo "refusing to clean non-RAM repository: ${REPO}" >&2 ;;
    esac
  fi
  rm -f -- "${PID_FILE}"
}
trap cleanup EXIT

exec >>"${LOG}" 2>&1
echo "$(date --iso-8601=seconds) host=$(hostname) run=${RUN} resume_step=${RESUME_STEP} final_step=${FINAL_STEP}"
echo "source_commit=${SOURCE_COMMIT} batch=256 devices=8 fsdp_devices=1 vision=unfrozen"
echo "teacher_plus_twoway_weight=${FULL_ATOMIC_LOSS_WEIGHT} projection_weight=${ATOMIC_PROJECTION_LOSS_WEIGHT} codebook=frozen"
echo "projection=dual_top2_drop_top5 source_sidecar=${ATOMIC_COMPOSITION_SIDECAR} kl=disabled"
sha256sum \
  "${REPO}/src/atomic_latent_vla/pi05/model.py" \
  "${REPO}/src/atomic_latent_vla/pi05/training_data.py" \
  "${REPO}/src/atomic_latent_vla/pi05/gemma_adapter.py" \
  "${REPO}/scripts/train_atomic_pi05.py"

env \
  REPO="${REPO}" \
  RUN="${RUN}" \
  STEPS="${STEPS}" \
  RESUME=1 \
  SMOKE_STEPS=0 \
  PHYSICAL_BATCH_SIZE=256 \
  FULL_ATOMIC_LOSS_WEIGHT="${FULL_ATOMIC_LOSS_WEIGHT}" \
  ATOMIC_PROJECTION_LOSS_WEIGHT="${ATOMIC_PROJECTION_LOSS_WEIGHT}" \
  ATOMIC_COMPOSITION_SIDECAR="${ATOMIC_COMPOSITION_SIDECAR}" \
  bash "${REPO}/scripts/run_target_v3_offline_zm_from_zt27k_q005_kl0025_ddp8_bs128_40k.sh"
