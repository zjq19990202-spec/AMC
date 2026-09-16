#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/mnt/cunchu/zjq/atomic_latent_vla_layerwise}
OPENPI_ROOT=${OPENPI_ROOT:-${REPO}/vendor/pi0.5}
PI05_PYTHON=${PI05_PYTHON:-/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python}
DATASET=/mnt/cunchu/zjq/target/lerobot_compact_accepted_final
NORM_ROOT=/mnt/cunchu/zjq/target
NORM_ID=openpi_norm_compact_accepted_v3
TCP_NORM=${DATASET}/meta/tcp_twist_norm_bimanual_tcp200.json
ZT_RUN=target_v3_bimanual_zt_layerwise_shared_flow_ddp8_seed0_bs256_40k
ZT_PARAMS=/mnt/cunchu/zjq/atomic_pi05_runs/${ZT_RUN}/${ZT_RUN}/27000/params
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/mnt/cunchu/zjq/atomic_pi05_runs}
RUN=${RUN:-target_v3_joint_paired_layerwise_from_zt27k_ddp8_micro64_acc2_eff128_40k}
STEPS=${STEPS:-40000}
SMOKE_STEPS=${SMOKE_STEPS:-0}
PHYSICAL_BATCH_SIZE=${PHYSICAL_BATCH_SIZE:-64}
ACCUMULATION_STEPS=${ACCUMULATION_STEPS:-2}

export PYTHONPATH=${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${REPO}/src
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd "${REPO}"

test -d "${ZT_PARAMS}"
test -f "${TCP_NORM}"

extra_args=()
if (( SMOKE_STEPS > 0 )); then
  extra_args+=(--smoke-steps "${SMOKE_STEPS}" --skip-checkpoint)
fi

# Each physical row is reused once by text/state ZT and once by full ZM.
# The default 64-row microbatch accumulated twice therefore supplies exactly
# 128 ZT routes + 128 ZM routes per optimizer update, without a third prefix.
exec "${PI05_PYTHON}" scripts/train_atomic_pi05.py \
  --dataset-root "${DATASET}" \
  --norm-assets-dir "${NORM_ROOT}" \
  --norm-asset-id "${NORM_ID}" \
  --tcp-twist-norm "${TCP_NORM}" \
  --coefficient-target joint_delta \
  --base-params "${ZT_PARAMS}" \
  --checkpoint-base-dir "${CHECKPOINT_ROOT}" \
  --run-name "${RUN}" \
  --batch-size "${PHYSICAL_BATCH_SIZE}" \
  --devices 8 \
  --fsdp-devices 1 \
  --gradient-accumulation-steps "${ACCUMULATION_STEPS}" \
  --max-token-len 200 \
  --num-workers 64 \
  --steps "${STEPS}" \
  --initial-step 27000 \
  --phase-start-step 27000 \
  --paired-zt-zm \
  --zt-fraction 0.5 \
  --zt-loss-weight 0.5 \
  --zm-loss-weight 0.5 \
  --text-flow-loss-weight 1.0 \
  --coefficient-loss-weight 0.0 \
  --text-atomic-loss-weight 0.1 \
  --full-atomic-loss-weight 0.1 \
  --codebook-loss-weight 0.0 \
  --atomic-composition-loss-weight 0.05 \
  --subtask-ce-loss-weight 0.0 \
  --subtask-ce-batch-size 0 \
  --atomic-prompt-probability 0.5 \
  --atomic-text-ce-probability 0.0 \
  --warmup-steps 1000 \
  --peak-lr 2.5e-5 \
  --decay-lr 2.5e-6 \
  --lr-decay-steps 30000 \
  --save-interval 1000 \
  --log-interval 10 \
  "${extra_args[@]}"
