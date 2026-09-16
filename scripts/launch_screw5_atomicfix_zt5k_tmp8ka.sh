#!/usr/bin/env bash
set -euo pipefail

: "${RAM_REPO:?set the unique tmp8ka RAM source directory}"

RUN=union2375_zt25k_screw5_atomicfix_zt5k_ddp8_bs256_20260825
DATASET=/mnt/cunchu/zjq/target/lerobot_2058_screw5_subtask_atomicfix_20260825
NORM_ROOT=/mnt/cunchu/zjq/target
NORM_ID=openpi_norm_union2375_allframes_v1
CHECKPOINT_ROOT=/mnt/cunchu/zjq/atomic_pi05_runs
LOG=${CHECKPOINT_ROOT}/logs/${RUN}.log
CONTRACT=${CHECKPOINT_ROOT}/logs/${RUN}.contract.txt
LOCK=${CHECKPOINT_ROOT}/.locks/${RUN}.lock
PI05_PYTHON=/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python
OPENPI_ROOT=${RAM_REPO}/vendor/pi0.5

cleanup() {
  status=$?
  if [[ "${RAM_REPO}" == /dev/shm/atomic_latent_vla_run_* ]]; then
    cd /
    rm -rf -- "${RAM_REPO}"
  fi
  exit "${status}"
}
trap cleanup EXIT

mkdir -p "${CHECKPOINT_ROOT}/logs" "${CHECKPOINT_ROOT}/.locks"
exec 9>"${LOCK}"
flock -n 9 || { echo "run lock already held: ${LOCK}" >&2; exit 1; }

test -d "${RAM_REPO}/src"
test -d "${OPENPI_ROOT}/src"
test -f "${DATASET}/meta/episode_subtasks.jsonl"
test -f "${DATASET}/meta/atomic_horizon_prompts_3hz.jsonl"
test -f "${NORM_ROOT}/${NORM_ID}/norm_stats.json"
test -d "${CHECKPOINT_ROOT}/${RUN}/${RUN}/25000/train_state"
test ! -e "${CHECKPOINT_ROOT}/${RUN}/${RUN}/30000"

export PYTHONPATH=${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${RAM_REPO}/src
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

{
  echo "started=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "run=${RUN}"
  echo "transition=union2375 ZT25K -> corrected Screw5 atomic-prompt ZT30K"
  echo "resume=full params+optimizer+global_step from seeded step 25000"
  echo "dataset=${DATASET}"
  echo "prompt_route=atomic_horizon_prompts_3hz.jsonl; reliable atomic rows only"
  echo "atomic_composition=fk_horizon_3hz_gate_top5_stay_v2"
  echo "norm_root=${NORM_ROOT} norm_id=${NORM_ID} mode=q01_q99 adapt_to_pi=true"
  echo "state_clip=off action_clip=off inference_clip=not_applicable optimizer_grad_clip=1.0"
  echo "batch=256 devices=8 per_gpu=32 fsdp_devices=1 accumulation=1"
  echo "steps_global=25000_to_30000 codebook_train_until=30000"
  echo "loss_flow=1.0 loss_atomic_twoway=0.1 loss_dual_weighted_cos=0.1 loss_dct=0 loss_kl=0"
  sha256sum \
    "${DATASET}/meta/episode_subtasks.jsonl" \
    "${DATASET}/meta/atomic_horizon_prompts_3hz.jsonl" \
    "${NORM_ROOT}/${NORM_ID}/norm_stats.json" \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/training_data.py" \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/model.py" \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/gemma_adapter.py" \
    "${RAM_REPO}/scripts/train_atomic_zt_fast.py"
} >"${CONTRACT}"

cd "${RAM_REPO}"
"${PI05_PYTHON}" scripts/train_atomic_zt_fast.py \
  --dataset-root "${DATASET}" \
  --norm-assets-dir "${NORM_ROOT}" \
  --norm-asset-id "${NORM_ID}" \
  --atomic-composition-sidecar fk_horizon_3hz_gate_top5_stay_v2 \
  --tcp-twist-norm "${DATASET}/meta/tcp_twist_norm_bimanual_tcp200.json" \
  --coefficient-target joint_delta \
  --base-params /mnt/cunchu/zjq/.cache/openpi/openpi-assets/checkpoints/pi05_base/params \
  --checkpoint-base-dir "${CHECKPOINT_ROOT}" \
  --run-name "${RUN}" \
  --batch-size 256 \
  --devices 8 \
  --fsdp-devices 1 \
  --max-token-len 200 \
  --num-workers 64 \
  --steps 30000 \
  --resume \
  --save-interval 1000 \
  --log-interval 10 \
  --warmup-steps 1000 \
  --peak-lr 2.5e-5 \
  --decay-lr 2.5e-6 \
  --lr-decay-steps 30000 \
  --codebook-freeze-step 30000 \
  --flow-loss-weight 1.0 \
  --coefficient-loss-weight 0.0 \
  --atomic-loss-weight 0.1 \
  --atomic-projection-loss-weight 0.1 \
  --codebook-loss-weight 0.0 \
  >"${LOG}" 2>&1
