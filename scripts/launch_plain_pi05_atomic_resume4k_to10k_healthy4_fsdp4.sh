#!/usr/bin/env bash
set -euo pipefail

RUN_NAME=${RUN_NAME:-plain_pi05_atomic_if_valid_else_subtask_fullvision_resume4k_to10k_healthy4_fsdp4_bs256_20260911}
RAM_REPO=${RAM_REPO:?set RAM_REPO to the unique /dev/shm source copy}
CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1,3,4}
DEVICES=${DEVICES:-4}
FSDP_DEVICES=${FSDP_DEVICES:-4}
BATCH_SIZE=${BATCH_SIZE:-256}
PYTHON=/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python
SOURCE_RUN=/mnt/cunchu/zjq/pi05_method_baselines_20260830/checkpoints/plain_pi05_atomic_if_valid_else_subtask_fullvision_from_ft25k_ddp8_bs256_10k_nvls0_20260910/plain_pi05_atomic_if_valid_else_subtask_fullvision_from_ft25k_ddp8_bs256_10k_nvls0_20260910
RESTORE_STEP=${SOURCE_RUN}/4000
BASE_CHECKPOINT=/mnt/cunchu/zjq/pi05_method_baselines_20260830/checkpoints/plain_pi05_union2375_masked_subtask_ddp8_bs256_nw64_25k_20260830/plain_pi05_union2375_masked_subtask_ddp8_bs256_nw64_25k_20260830/25000
OUTPUT_ROOT=/mnt/cunchu/zjq/pi05_method_baselines_20260830/checkpoints
LOG_ROOT=/mnt/cunchu/zjq/pi05_method_baselines_20260830/logs
LOCK_ROOT=/mnt/cunchu/zjq/pi05_method_baselines_20260830/locks
RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_PATH="${LOG_ROOT}/${RUN_NAME}.log"
CONTRACT_PATH="${LOG_ROOT}/${RUN_NAME}.contract.txt"
LOCK_PATH="${LOCK_ROOT}/${RUN_NAME}.lock"
NORM_DIR=/mnt/cunchu/zjq/target/openpi_norm_union2375_allframes_v1

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${LOCK_ROOT}"
test -f "${RESTORE_STEP}/_CHECKPOINT_METADATA"
test -d "${RESTORE_STEP}/params"
test -d "${RESTORE_STEP}/train_state"
test -d "${BASE_CHECKPOINT}/params"
test -d "${NORM_DIR}"
test -x "${PYTHON}"
if [[ -e "${RUN_ROOT}" ]]; then
  echo "refusing to overwrite existing run: ${RUN_ROOT}" >&2
  exit 3
fi
exec 9>"${LOCK_PATH}"
flock -n 9 || { echo "run lock is held: ${LOCK_PATH}" >&2; exit 4; }

cleanup() {
  if [[ "${RAM_REPO}" == /dev/shm/atomic_latent_vla_run_* && -d "${RAM_REPO}" ]]; then
    rm -rf -- "${RAM_REPO}"
  fi
}
trap cleanup EXIT INT TERM

NORM_SHA256=$(
  find "${NORM_DIR}" -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    | sha256sum \
    | awk '{print $1}'
)

cat >"${CONTRACT_PATH}" <<EOF
run_name=${RUN_NAME}
method=stock PI0.5 full-vision atomic-route exact full-state resume
restore_type=complete train_state: params, AdamW optimizer state, and global step
restore_checkpoint=${RESTORE_STEP}
restore_step=4000
discarded_unsaved_steps=4001-4490 from stalled source process
final_step=10000
base_checkpoint=${BASE_CHECKPOINT}
vision_mask_probability=0.0
prompt_route=if any arm atomic_supervision_mask is true, use reviewed arm-specific atomic_prompt; otherwise use native SUBtask
prompt_mixing_probability=data-determined; atomic prompt has strict priority and SUBtask is fallback only
global_prompt=never passed to model
tokenizer=stock PI0.5 tokenizer through atomic_latent_vla.pi05.training_data
max_token_len=200
dataset=union2375 seven domain roots
start_inactive_arm_mask=meta/start_inactive_arm_block_mask.json in every root; 80 masked episodes total
norm_assets_dir=/mnt/cunchu/zjq/target
norm_asset_id=openpi_norm_union2375_allframes_v1
norm_tree_sha256=${NORM_SHA256}
normalization=q01_q99 affine without state/action clipping
adapt_to_pi=training_data.batch_to_observation stock PI0.5 contract
state=16 physical dims padded to 32
action=joint_delta 16 physical dims padded to 32
delta_action_order=14 arm joints followed by 2 grippers; padded to 32
action_horizon=50
subtask_horizon_padding=true
state_clipping=false
action_target_clipping=false
inference_output_clipping=false
optimizer_gradient_clipping=global_norm_1.0
model=stock PI0.5 gemma_2b + action_expert_300m
vision_frozen=false
global_batch_size=${BATCH_SIZE}
per_device_batch_size=$((BATCH_SIZE / DEVICES))
physical_gpus=${CUDA_DEVICE_LIST}
devices=${DEVICES}
fsdp_devices=${FSDP_DEVICES}
mesh=[1,4] FSDP across four healthy H800 GPUs; global batch unchanged
optimizer=restored AdamW b1=0.9 b2=0.95 eps=1e-8 weight_decay=1e-10 grad_clip=1.0
lr=continued warmup1000 peak2.5e-5 cosine_decay30000 floor2.5e-6
seed=0
save_interval=1000
num_workers=64
xla_preallocate=true
xla_memory_fraction=0.75
nccl_nvls_enable=0
source_ram=${RAM_REPO}
source_commit=$(git -C "${RAM_REPO}" rev-parse HEAD 2>/dev/null || echo external-drive-copy-without-git)
entrypoint_sha256=$(sha256sum "${RAM_REPO}/scripts/train_plain_pi05_la4_baseline.py" | awk '{print $1}')
training_data_sha256=$(sha256sum "${RAM_REPO}/src/atomic_latent_vla/pi05/training_data.py" | awk '{print $1}')
model_sha256=$(sha256sum "${RAM_REPO}/vendor/pi0.5/src/openpi/models/pi0.py" | awk '{print $1}')
gemma_adapter_sha256=$(sha256sum "${RAM_REPO}/vendor/pi0.5/src/openpi/models/gemma.py" | awk '{print $1}')
EOF

cd "${RAM_REPO}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}"
export PYTHONPATH="${RAM_REPO}/vendor/pi0.5/src:${RAM_REPO}/vendor/pi0.5/packages/openpi-client/src:${RAM_REPO}/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export NCCL_NVLS_ENABLE=0

"${PYTHON}" scripts/train_plain_pi05_la4_baseline.py \
  --norm-assets-dir /mnt/cunchu/zjq/target \
  --norm-asset-id openpi_norm_union2375_allframes_v1 \
  --base-params "${BASE_CHECKPOINT}/params" \
  --restore-full-state "${RESTORE_STEP}" \
  --checkpoint-base-dir "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --batch-size "${BATCH_SIZE}" \
  --devices "${DEVICES}" \
  --fsdp-devices "${FSDP_DEVICES}" \
  --steps 10000 \
  --warmup-steps 1000 \
  --peak-lr 2.5e-5 \
  --decay-lr 2.5e-6 \
  --lr-decay-steps 30000 \
  --max-token-len 200 \
  --num-workers 64 \
  --save-interval 1000 \
  --log-interval 10 \
  --seed 0 \
  --vision-mask-probability 0.0 \
  --prompt-route atomic_if_valid_else_subtask \
  >>"${LOG_PATH}" 2>&1
