#!/usr/bin/env bash
set -euo pipefail

RUN_NAME=${RUN_NAME:-plain_pi05_atomic_if_valid_else_subtask_fullvision_from_ft25k_ddp8_bs256_10k_nvls0_20260910}
RAM_REPO=${RAM_REPO:?set RAM_REPO to the unique /dev/shm source copy}
PYTHON=/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python
SOURCE_CHECKPOINT=/mnt/cunchu/zjq/pi05_method_baselines_20260830/checkpoints/plain_pi05_union2375_masked_subtask_ddp8_bs256_nw64_25k_20260830/plain_pi05_union2375_masked_subtask_ddp8_bs256_nw64_25k_20260830/25000
SOURCE_PARAMS=${SOURCE_CHECKPOINT}/params
OUTPUT_ROOT=/mnt/cunchu/zjq/pi05_method_baselines_20260830/checkpoints
LOG_ROOT=/mnt/cunchu/zjq/pi05_method_baselines_20260830/logs
LOCK_ROOT=/mnt/cunchu/zjq/pi05_method_baselines_20260830/locks
RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_PATH="${LOG_ROOT}/${RUN_NAME}.log"
CONTRACT_PATH="${LOG_ROOT}/${RUN_NAME}.contract.txt"
LOCK_PATH="${LOCK_ROOT}/${RUN_NAME}.lock"
NORM_DIR=/mnt/cunchu/zjq/target/openpi_norm_union2375_allframes_v1

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${LOCK_ROOT}"
test -f "${SOURCE_CHECKPOINT}/_CHECKPOINT_METADATA"
test -d "${SOURCE_PARAMS}"
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
method=stock PI0.5 full-vision atomic-route post-finetuning
initialization_type=params_only; fresh optimizer and fresh step counter
source_checkpoint=${SOURCE_CHECKPOINT}
source_params=${SOURCE_PARAMS}
source_training=25k full-vision native-SUBtask PI0.5 finetuning
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
global_batch_size=256
per_device_batch_size=32
devices=8
fsdp_devices=1
mesh=[8,1] DDP replicated parameters
optimizer=AdamW b1=0.9 b2=0.95 eps=1e-8 weight_decay=1e-10 grad_clip=1.0; freshly initialized
steps=10000 new atomic-route updates
lr=warmup1000 peak2.5e-5 cosine_decay30000 floor2.5e-6; fresh schedule, stopped at 10k
seed=0
save_interval=1000
num_workers=64
xla_preallocate=true
xla_memory_fraction=0.75
nccl_nvls_enable=0; required because this rebuilt node advertises NVLS but CUDA rejects NVLS multicast setup; ordinary 8-GPU NCCL P2P all-reduce passed
source_ram=${RAM_REPO}
source_commit=$(git -C "${RAM_REPO}" rev-parse HEAD 2>/dev/null || echo external-drive-copy-without-git)
entrypoint_sha256=$(sha256sum "${RAM_REPO}/scripts/train_plain_pi05_la4_baseline.py" | awk '{print $1}')
training_data_sha256=$(sha256sum "${RAM_REPO}/src/atomic_latent_vla/pi05/training_data.py" | awk '{print $1}')
model_sha256=$(sha256sum "${RAM_REPO}/vendor/pi0.5/src/openpi/models/pi0.py" | awk '{print $1}')
gemma_adapter_sha256=$(sha256sum "${RAM_REPO}/vendor/pi0.5/src/openpi/models/gemma.py" | awk '{print $1}')
EOF

cd "${RAM_REPO}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="${RAM_REPO}/vendor/pi0.5/src:${RAM_REPO}/vendor/pi0.5/packages/openpi-client/src:${RAM_REPO}/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export NCCL_NVLS_ENABLE=0

"${PYTHON}" scripts/train_plain_pi05_la4_baseline.py \
  --norm-assets-dir /mnt/cunchu/zjq/target \
  --norm-asset-id openpi_norm_union2375_allframes_v1 \
  --base-params "${SOURCE_PARAMS}" \
  --checkpoint-base-dir "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --batch-size 256 \
  --devices 8 \
  --fsdp-devices 1 \
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
