#!/usr/bin/env bash
set -euo pipefail

: "${RAM_REPO:?set a unique /dev/shm RAM_REPO}"
: "${DOMAIN:?set DOMAIN}"
: "${DATASET:?set DATASET}"
: "${GPU_LIST:?set four comma-separated GPU ids}"

PI05_PYTHON=${PI05_PYTHON:-/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/mnt/cunchu/zjq/atomic_pi05_runs}
NORM_ROOT=${NORM_ROOT:-/mnt/cunchu/zjq/target}
NORM_ID=${NORM_ID:-openpi_norm_union2375_allframes_v1}
EXPECTED_NORM_SHA256=512debfe8b5a905287ab9993b29cbcb8bb246c10ac692f5ad6d341637c4aebcc
TCP_NORM=${NORM_ROOT}/${NORM_ID}/tcp_twist_norm_bimanual_tcp200.json
EXPECTED_TCP_NORM_SHA256=aa30efd013b1c656d126fd47d518321875c15a6bb5ea15ed0dd4a4248aa46b73
COMPOSITION=fk_horizon_3hz_gate_top5_stay_v2
ZT_RUN=union2375_allframes_zt_layerwise_25k_ddp8_bs256_20260822
ZT_CHECKPOINT=${CHECKPOINT_ROOT}/${ZT_RUN}/${ZT_RUN}/25000
RUN=${RUN:-union2375_${DOMAIN}_zm25k_bs128_4gpu_from_zt25k_20260822}
TEACHER=${TEACHER:-${CHECKPOINT_ROOT}/teachers/${ZT_RUN}_${DOMAIN}_stay_v2}
LOG_ROOT=${CHECKPOINT_ROOT}/logs
LOCK_ROOT=${CHECKPOINT_ROOT}/.locks
LOG=${LOG_ROOT}/${RUN}.log

mkdir -p "${LOG_ROOT}" "${LOCK_ROOT}" "${CHECKPOINT_ROOT}/teachers"
exec 9>"${LOCK_ROOT}/${RUN}.lock"
flock -n 9 || { echo "lock held: ${RUN}" >&2; exit 1; }

cleanup() {
  status=$?
  if [[ "${RAM_REPO}" == /dev/shm/atomic_latent_vla_run_* ]]; then
    cd /
    rm -rf -- "${RAM_REPO}"
  fi
  exit "${status}"
}
trap cleanup EXIT

test -d "${RAM_REPO}/src"
test -d "${RAM_REPO}/vendor/pi0.5/src/openpi"
test -x "${PI05_PYTHON}"
test -d "${ZT_CHECKPOINT}/params"
test -f "${DATASET}/meta/info.json"
test -d "${DATASET}/meta/${COMPOSITION}"
test -f "${DATASET}/meta/tcp_pose_bimanual_base_tcp200.npy"
test -f "${NORM_ROOT}/${NORM_ID}/norm_stats.json"
test -f "${TCP_NORM}"
test ! -e "${CHECKPOINT_ROOT}/${RUN}"
test ! -e "${TEACHER}"

norm_sha=$(sha256sum "${NORM_ROOT}/${NORM_ID}/norm_stats.json" | awk '{print $1}')
tcp_norm_sha=$(sha256sum "${TCP_NORM}" | awk '{print $1}')
[[ "${norm_sha}" == "${EXPECTED_NORM_SHA256}" ]]
[[ "${tcp_norm_sha}" == "${EXPECTED_TCP_NORM_SHA256}" ]]

contract=${LOG_ROOT}/${RUN}.contract.txt
{
  echo "started=$(date --iso-8601=seconds)"
  echo "host=$(hostname) gpus=${GPU_LIST} domain=${DOMAIN}"
  echo "dataset=${DATASET} sampling=natural_rows_no_dataset_weight"
  echo "zt_checkpoint=${ZT_CHECKPOINT} teacher=${TEACHER}"
  echo "norm_root=${NORM_ROOT} norm_id=${NORM_ID} norm_sha256=${norm_sha}"
  echo "tcp_norm=${TCP_NORM} tcp_norm_sha256=${tcp_norm_sha}"
  echo "batch=128 devices=4 per_gpu=32 fsdp_devices=4 accumulation=1"
  echo "steps=25000 warmup=1000 peak_lr=2.5e-5 decay_lr=2.5e-6 decay_steps=30000"
  echo "prompt=official_subtask_sidecar_only atomic_prompt_probability=0 max_token_len=200"
  echo "zt_fraction=0 q1_loss_weight=0.05 drop_weighted_cos_weight=0.015 codebook_frozen=true vision_frozen=false"
  echo "normalization=q01_q99_affine_unbounded adapt_to_pi=true coefficient_target=joint_delta"
  echo "state_clip=false action_clip=false output_clip=false optimizer_gradient_clip=1.0"
  sha256sum \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/training_data.py" \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/model.py" \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/gemma_adapter.py" \
    "${RAM_REPO}/scripts/precompute_target_zt_q_teacher.py" \
    "${RAM_REPO}/scripts/train_atomic_pi05.py"
} >"${contract}"

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export PYTHONPATH="${RAM_REPO}/vendor/pi0.5/src:${RAM_REPO}/vendor/pi0.5/packages/openpi-client/src:${RAM_REPO}/src"
cd "${RAM_REPO}"

echo "teacher_start=$(date --iso-8601=seconds)"
XLA_PYTHON_CLIENT_PREALLOCATE=false "${PI05_PYTHON}" scripts/precompute_target_zt_q_teacher.py \
  --checkpoint "${ZT_CHECKPOINT}" \
  --dataset-root "${DATASET}" \
  --norm-assets-dir "${NORM_ROOT}" \
  --norm-asset-id "${NORM_ID}" \
  --atomic-composition-sidecar "${COMPOSITION}" \
  --output "${TEACHER}" \
  --batch-size 1024 \
  --devices 4 \
  --num-workers 8 \
  --max-token-len 200 \
  --coefficient-target joint_delta
echo "teacher_complete=$(date --iso-8601=seconds)"

export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
"${PI05_PYTHON}" scripts/train_atomic_pi05.py \
  --dataset-root "${DATASET}" \
  --norm-assets-dir "${NORM_ROOT}" \
  --norm-asset-id "${NORM_ID}" \
  --zt-teacher-sidecar "${TEACHER}" \
  --atomic-composition-sidecar "${COMPOSITION}" \
  --tcp-twist-norm "${TCP_NORM}" \
  --coefficient-target joint_delta \
  --base-params "${ZT_CHECKPOINT}/params" \
  --checkpoint-base-dir "${CHECKPOINT_ROOT}" \
  --run-name "${RUN}" \
  --batch-size 128 \
  --devices 4 \
  --fsdp-devices 4 \
  --gradient-accumulation-steps 1 \
  --max-token-len 200 \
  --num-workers 16 \
  --steps 25000 \
  --initial-step 0 \
  --phase-start-step 0 \
  --zt-fraction 0.0 \
  --zt-loss-weight 0.0 \
  --zm-loss-weight 1.0 \
  --text-flow-loss-weight 0.0 \
  --coefficient-loss-weight 0.0 \
  --text-atomic-loss-weight 0.0 \
  --full-atomic-loss-weight 0.05 \
  --codebook-loss-weight 0.0 \
  --atomic-projection-loss-weight 0.015 \
  --subtask-ce-loss-weight 0.0 \
  --subtask-ce-batch-size 0 \
  --atomic-prompt-probability 0.0 \
  --atomic-text-ce-probability 0.0 \
  --freeze-codebook \
  --unfreeze-vision \
  --warmup-steps 1000 \
  --peak-lr 2.5e-5 \
  --decay-lr 2.5e-6 \
  --lr-decay-steps 30000 \
  --save-interval 1000 \
  --log-interval 10

echo "training_complete=$(date --iso-8601=seconds)"
