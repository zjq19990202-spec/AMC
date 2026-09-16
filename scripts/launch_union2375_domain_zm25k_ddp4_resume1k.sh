#!/usr/bin/env bash
set -euo pipefail

: "${RAM_REPO:?set a unique /dev/shm RAM_REPO}"
: "${DOMAIN:?set DOMAIN}"
: "${DATASET:?set DATASET}"
: "${GPU_LIST:?set four comma-separated GPU ids}"
: "${SOURCE_STEP:?set latest complete checkpoint step}"

PI05_PYTHON=${PI05_PYTHON:-/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python}
ROOT=/mnt/cunchu/zjq/atomic_pi05_runs
NORM_ROOT=/mnt/cunchu/zjq/target
NORM_ID=openpi_norm_union2375_allframes_v1
ZT_RUN=union2375_allframes_zt_layerwise_25k_ddp8_bs256_20260822
OLD_RUN=union2375_${DOMAIN}_zm25k_bs128_4gpu_from_zt25k_20260822
RUN=union2375_${DOMAIN}_zm25k_bs128_ddp4_resume${SOURCE_STEP}_r2_20260823
SOURCE=${ROOT}/${OLD_RUN}/${OLD_RUN}/${SOURCE_STEP}
ADDITIONAL_STEPS=$((25000 - SOURCE_STEP))
TEACHER=${ROOT}/teachers/${ZT_RUN}_${DOMAIN}_stay_v2
LOG=${ROOT}/logs/${RUN}.log

mkdir -p "${ROOT}/logs" "${ROOT}/.locks"
exec 9>"${ROOT}/.locks/${RUN}.lock"
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
test -d "${SOURCE}/params"
test -f "${SOURCE}/train_state/_METADATA"
test -d "${TEACHER}"
test ! -e "${ROOT}/${RUN}"

{
  echo "started=$(date --iso-8601=seconds)"
  echo "host=$(hostname) gpus=${GPU_LIST} domain=${DOMAIN}"
  echo "dataset=${DATASET} sampling=natural_rows_no_dataset_weight"
  echo "restore_full_state=${SOURCE} preserve_adam=true restore_step=${SOURCE_STEP}"
  echo "mesh=[4,1] devices=4 fsdp_devices=1 mode=pure_ddp"
  echo "batch=128 per_gpu=32 accumulation=1 additional_steps=${ADDITIONAL_STEPS} final_step=25000"
  echo "norm_root=${NORM_ROOT} norm_id=${NORM_ID} coefficient_target=joint_delta"
  echo "prompt=official_subtask_sidecar_only atomic_prompt_probability=0 max_token_len=200"
  echo "q1_loss_weight=0.05 drop_weighted_cos_weight=0.015 codebook_frozen=true vision_frozen=false"
  echo "state_clip=false action_clip=false output_clip=false optimizer_gradient_clip=1.0"
  sha256sum "${RAM_REPO}/src/atomic_latent_vla/pi05/model.py" \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/gemma_adapter.py" \
    "${RAM_REPO}/scripts/train_atomic_pi05.py"
} >"${ROOT}/logs/${RUN}.contract.txt"

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export PYTHONPATH="${RAM_REPO}/vendor/pi0.5/src:${RAM_REPO}/vendor/pi0.5/packages/openpi-client/src:${RAM_REPO}/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
cd "${RAM_REPO}"

"${PI05_PYTHON}" scripts/train_atomic_pi05.py \
  --dataset-root "${DATASET}" \
  --norm-assets-dir "${NORM_ROOT}" \
  --norm-asset-id "${NORM_ID}" \
  --zt-teacher-sidecar "${TEACHER}" \
  --atomic-composition-sidecar fk_horizon_3hz_gate_top5_stay_v2 \
  --tcp-twist-norm "${NORM_ROOT}/${NORM_ID}/tcp_twist_norm_bimanual_tcp200.json" \
  --coefficient-target joint_delta \
  --restore-full-state "${SOURCE}" \
  --checkpoint-base-dir "${ROOT}" \
  --run-name "${RUN}" \
  --batch-size 128 --devices 4 --fsdp-devices 1 \
  --gradient-accumulation-steps 1 --max-token-len 200 --num-workers 16 \
  --steps "${ADDITIONAL_STEPS}" --phase-start-step 0 \
  --zt-fraction 0.0 --zt-loss-weight 0.0 --zm-loss-weight 1.0 \
  --text-flow-loss-weight 0.0 --coefficient-loss-weight 0.0 \
  --text-atomic-loss-weight 0.0 --full-atomic-loss-weight 0.05 \
  --codebook-loss-weight 0.0 --atomic-projection-loss-weight 0.015 \
  --subtask-ce-loss-weight 0.0 --subtask-ce-batch-size 0 \
  --atomic-prompt-probability 0.0 --atomic-text-ce-probability 0.0 \
  --freeze-codebook --unfreeze-vision \
  --warmup-steps 1000 --peak-lr 2.5e-5 --decay-lr 2.5e-6 --lr-decay-steps 30000 \
  --save-interval 1000 --log-interval 10
