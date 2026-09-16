#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/mnt/cunchu/zjq/atomic_latent_vla_layerwise}
OPENPI_ROOT=${OPENPI_ROOT:-${REPO}/vendor/pi0.5}
PI05_PYTHON=${PI05_PYTHON:-/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python}
DATASET=/mnt/cunchu/zjq/target/lerobot_compact_accepted_final
NORM_ROOT=/mnt/cunchu/zjq/target
NORM_ID=openpi_norm_compact_accepted_v3
TCP_NORM=${DATASET}/meta/tcp_twist_norm_bimanual_tcp200.json
ZT_RUN=${ZT_RUN:-target_v3_bimanual_zt_layerwise_shared_flow_ddp8_seed0_bs256_40k}
ZT_PARAMS=${ZT_PARAMS:-/mnt/cunchu/zjq/atomic_pi05_runs/${ZT_RUN}/${ZT_RUN}/27000/params}
ZT_TEACHER=${ZT_TEACHER:-${DATASET}/meta/zt_q_teacher_target_v3_layerwise_shared_flow_zt27k_atomic_v1}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/mnt/cunchu/zjq/atomic_pi05_runs}
RUN=${RUN:-target_v3_offline_zm_layerwise_from_zt27k_q005_kl0025_unfreezevit_ddp8_bs256_40k}
STEPS=${STEPS:-40000}
SMOKE_STEPS=${SMOKE_STEPS:-0}
PHYSICAL_BATCH_SIZE=${PHYSICAL_BATCH_SIZE:-256}
FULL_ATOMIC_LOSS_WEIGHT=${FULL_ATOMIC_LOSS_WEIGHT:-0.05}
ATOMIC_COMPOSITION_LOSS_WEIGHT=${ATOMIC_COMPOSITION_LOSS_WEIGHT:-0.025}
ATOMIC_PROJECTION_LOSS_WEIGHT=${ATOMIC_PROJECTION_LOSS_WEIGHT:-${ATOMIC_COMPOSITION_LOSS_WEIGHT}}
RESUME=${RESUME:-0}
RESTORE_FULL_STATE=${RESTORE_FULL_STATE:-}
ATOMIC_COMPOSITION_SIDECAR=${ATOMIC_COMPOSITION_SIDECAR:-fk_horizon_3hz_gate_top5_v1}
INITIAL_STEP=${INITIAL_STEP:-27000}
PHASE_START_STEP=${PHASE_START_STEP:-27000}
ATOMIC_PROMPT_PROBABILITY=${ATOMIC_PROMPT_PROBABILITY:-0.5}

export PYTHONPATH=${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${REPO}/src
# The replicated 8-GPU graph needs a bounded JAX pool so NCCL/FSDP runtime
# allocations retain headroom. Dynamic allocation fragments the remaining
# memory and fails on the first backward collective.
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-true}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd "${REPO}"

test -d "${ZT_PARAMS}"
test -f "${TCP_NORM}"
test -d "${ZT_TEACHER}"
if [[ "${ATOMIC_COMPOSITION_SIDECAR}" = /* ]]; then
  test -d "${ATOMIC_COMPOSITION_SIDECAR}"
else
  test -d "${DATASET}/meta/${ATOMIC_COMPOSITION_SIDECAR}"
fi

extra_args=()
if (( SMOKE_STEPS > 0 )); then
  extra_args+=(--smoke-steps "${SMOKE_STEPS}" --skip-checkpoint)
fi
if (( RESUME > 0 )); then
  extra_args+=(--resume)
fi
if [[ -n "${RESTORE_FULL_STATE}" ]]; then
  test -d "${RESTORE_FULL_STATE}/train_state"
  test -d "${RESTORE_FULL_STATE}/params"
  extra_args+=(--restore-full-state "${RESTORE_FULL_STATE}")
fi

exec "${PI05_PYTHON}" scripts/train_atomic_pi05.py \
  --dataset-root "${DATASET}" \
  --norm-assets-dir "${NORM_ROOT}" \
  --norm-asset-id "${NORM_ID}" \
  --zt-teacher-sidecar "${ZT_TEACHER}" \
  --atomic-composition-sidecar "${ATOMIC_COMPOSITION_SIDECAR}" \
  --tcp-twist-norm "${TCP_NORM}" \
  --coefficient-target joint_delta \
  --base-params "${ZT_PARAMS}" \
  --checkpoint-base-dir "${CHECKPOINT_ROOT}" \
  --run-name "${RUN}" \
  --batch-size "${PHYSICAL_BATCH_SIZE}" \
  --devices 8 \
  --fsdp-devices 1 \
  --gradient-accumulation-steps 1 \
  --max-token-len 200 \
  --num-workers 64 \
  --steps "${STEPS}" \
  --initial-step "${INITIAL_STEP}" \
  --phase-start-step "${PHASE_START_STEP}" \
  --zt-fraction 0.0 \
  --zt-loss-weight 0.0 \
  --zm-loss-weight 1.0 \
  --text-flow-loss-weight 0.0 \
  --coefficient-loss-weight 0.0 \
  --text-atomic-loss-weight 0.0 \
  --full-atomic-loss-weight "${FULL_ATOMIC_LOSS_WEIGHT}" \
  --codebook-loss-weight 0.0 \
  --atomic-projection-loss-weight "${ATOMIC_PROJECTION_LOSS_WEIGHT}" \
  --subtask-ce-loss-weight 0.0 \
  --subtask-ce-batch-size 0 \
  --atomic-prompt-probability "${ATOMIC_PROMPT_PROBABILITY}" \
  --atomic-text-ce-probability 0.0 \
  --freeze-codebook \
  --unfreeze-vision \
  --warmup-steps 1000 \
  --peak-lr 2.5e-5 \
  --decay-lr 2.5e-6 \
  --lr-decay-steps 30000 \
  --save-interval 1000 \
  --log-interval 10 \
  "${extra_args[@]}"
