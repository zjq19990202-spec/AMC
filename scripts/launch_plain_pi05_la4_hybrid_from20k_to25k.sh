#!/usr/bin/env bash
set -euo pipefail

RUN_NAME=${RUN_NAME:-pi05_la4mixed50_atom_if_valid_else_subtask_from20k_to25k_20260901}
RAM_REPO=${RAM_REPO:?set RAM_REPO to the unique /dev/shm source copy}
PYTHON=/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python
OUTPUT_ROOT=/mnt/cunchu/zjq/pi05_method_baselines_20260830/checkpoints
LOG_ROOT=/mnt/cunchu/zjq/pi05_method_baselines_20260830/logs
LOCK_ROOT=/mnt/cunchu/zjq/pi05_method_baselines_20260830/locks
SOURCE_STEP=/mnt/cunchu/zjq/pi05_method_baselines_20260830/checkpoints/pi05_la4mixed50_union2375_masked_subtask_ddp8_bs256_nw64_25k_20260830/pi05_la4mixed50_union2375_masked_subtask_ddp8_bs256_nw64_25k_20260830/20000
RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_PATH="${LOG_ROOT}/${RUN_NAME}.log"
CONTRACT_PATH="${LOG_ROOT}/${RUN_NAME}.contract.txt"
LOCK_PATH="${LOCK_ROOT}/${RUN_NAME}.lock"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${LOCK_ROOT}"
test -d "${SOURCE_STEP}"
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

NORM_DIR=/mnt/cunchu/zjq/target/openpi_norm_union2375_allframes_v1
NORM_SHA256=$(
  find "${NORM_DIR}" -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    | sha256sum \
    | awk '{print $1}'
)

cat >"${CONTRACT_PATH}" <<EOF
run_name=${RUN_NAME}
method=PI0.5 + LA4VLA 50% per-sample vision masking with atomic-if-valid prompt routing
data_contract_transition=step 0..20000 used subtask_only; step 20000..25000 uses atomic_if_valid_else_subtask
source_full_state=${SOURCE_STEP}
source_step=20000
optimizer_state=restored from source checkpoint
lr_schedule=original warmup1000 peak2.5e-5 cosine_decay30000 floor2.5e-6, continued at global step 20000
vision_mask_probability=0.5
prompt_route=if any arm atomic_supervision_mask is true, use reviewed arm-specific atomic_prompt; otherwise use native SUBtask
stay_policy=reviewed/restored stay is a valid atomic label
global_prompt=never passed to model
dataset=union2375 seven domain roots
start_inactive_arm_mask=meta/start_inactive_arm_block_mask.json in every root; 80 masked episodes total
norm_assets_dir=/mnt/cunchu/zjq/target
norm_asset_id=openpi_norm_union2375_allframes_v1
norm_tree_sha256=${NORM_SHA256}
normalization=q01_q99 affine without state/action clipping
state=16 physical dims padded to 32
action=joint_delta 16 physical dims padded to 32
action_horizon=50
subtask_horizon_padding=true
model=stock PI0.5 gemma_2b + action_expert_300m
vision_frozen=false
global_batch_size=256
devices=8
fsdp_devices=1
mesh=[8,1] DDP replicated parameters
optimizer=AdamW b1=0.9 b2=0.95 eps=1e-8 weight_decay=1e-10 grad_clip=1.0
final_global_step=25000
additional_updates=5000
save_interval=1000
num_workers=64
source_ram=${RAM_REPO}
source_commit=$(git -C "${RAM_REPO}" rev-parse HEAD 2>/dev/null || echo external-drive-copy-without-git)
entrypoint_sha256=$(sha256sum "${RAM_REPO}/scripts/train_plain_pi05_la4_baseline.py" | awk '{print $1}')
training_data_sha256=$(sha256sum "${RAM_REPO}/src/atomic_latent_vla/pi05/training_data.py" | awk '{print $1}')
model_sha256=$(sha256sum "${RAM_REPO}/vendor/pi0.5/src/openpi/models/pi0.py" | awk '{print $1}')
EOF

cd "${RAM_REPO}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="${RAM_REPO}/vendor/pi0.5/src:${RAM_REPO}/vendor/pi0.5/packages/openpi-client/src:${RAM_REPO}/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

"${PYTHON}" scripts/train_plain_pi05_la4_baseline.py \
  --norm-assets-dir /mnt/cunchu/zjq/target \
  --norm-asset-id openpi_norm_union2375_allframes_v1 \
  --checkpoint-base-dir "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --batch-size 256 \
  --devices 8 \
  --fsdp-devices 1 \
  --steps 25000 \
  --warmup-steps 1000 \
  --peak-lr 2.5e-5 \
  --decay-lr 2.5e-6 \
  --lr-decay-steps 30000 \
  --num-workers 64 \
  --save-interval 1000 \
  --log-interval 10 \
  --vision-mask-probability 0.5 \
  --prompt-route atomic_if_valid_else_subtask \
  --restore-full-state "${SOURCE_STEP}" \
  >>"${LOG_PATH}" 2>&1
