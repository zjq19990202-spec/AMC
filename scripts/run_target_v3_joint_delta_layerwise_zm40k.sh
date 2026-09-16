#!/usr/bin/env bash
set -euo pipefail

REPO=/mnt/cunchu/zjq/atomic_latent_vla_layerwise
OPENPI_ROOT=${OPENPI_ROOT:-${REPO}/vendor/pi0.5}
PI05_PYTHON=${PI05_PYTHON:-/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python}
DATASET=/mnt/cunchu/zjq/target/lerobot_compact_accepted_final
NORM_ROOT=/mnt/cunchu/zjq/target
NORM_ID=openpi_norm_compact_accepted_v3
TCP_NORM=${DATASET}/meta/tcp_twist_norm_bimanual_tcp200.json
ZT_TEACHER=${DATASET}/meta/zt_q_teacher_target_v3_shared_flow_40k_v1
ZT_RUN=target_v3_bimanual_zt_shared_flow_seed0_bs256_40k
ZT_PARAMS=/mnt/cunchu/zjq/atomic_pi05_runs/${ZT_RUN}/${ZT_RUN}/40000/params
CHECKPOINT_ROOT=/mnt/cunchu/zjq/atomic_pi05_layerwise_runs
RUN=target_v3_bimanual_zm_layerwise_from_shared_flow_zt40k_teacherq_bs256_40k

export PYTHONPATH=${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${REPO}/src
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd "${REPO}"
test -d "${ZT_PARAMS}"
test -f "${TCP_NORM}"
test -d "${ZT_TEACHER}"

exec "${PI05_PYTHON}" scripts/train_atomic_pi05.py \
  --dataset-root "${DATASET}" \
  --norm-assets-dir "${NORM_ROOT}" \
  --norm-asset-id "${NORM_ID}" \
  --zt-teacher-sidecar "${ZT_TEACHER}" \
  --tcp-twist-norm "${TCP_NORM}" \
  --coefficient-target joint_delta \
  --base-params "${ZT_PARAMS}" \
  --checkpoint-base-dir "${CHECKPOINT_ROOT}" \
  --run-name "${RUN}" \
  --batch-size 256 \
  --devices 8 \
  --fsdp-devices 1 \
  --gradient-accumulation-steps 1 \
  --max-token-len 200 \
  --num-workers 64 \
  --steps 40000 \
  --initial-step 40000 \
  --phase-start-step 40000 \
  --zt-fraction 0.0 \
  --zt-loss-weight 0.0 \
  --zm-loss-weight 1.0 \
  --coefficient-loss-weight 0.0 \
  --text-atomic-loss-weight 0.0 \
  --full-atomic-loss-weight 0.1 \
  --codebook-loss-weight 0.0 \
  --atomic-composition-loss-weight 0.05 \
  --subtask-ce-loss-weight 0.0 \
  --subtask-ce-batch-size 0 \
  --atomic-prompt-probability 1.0 \
  --atomic-text-ce-probability 0.0 \
  --freeze-codebook \
  --unfreeze-vision \
  --warmup-steps 1000 \
  --peak-lr 2.5e-5 \
  --decay-lr 2.5e-6 \
  --lr-decay-steps 30000 \
  --save-interval 5000 \
  --log-interval 10
