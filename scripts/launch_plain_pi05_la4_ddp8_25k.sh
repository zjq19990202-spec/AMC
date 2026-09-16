#!/usr/bin/env bash
set -euo pipefail

RECIPE=${RECIPE:?set RECIPE=plain or la4}
RUN_NAME=${RUN_NAME:?set RUN_NAME}
RAM_REPO=${RAM_REPO:?set RAM_REPO}
VISION_MASK_PROBABILITY=0.0
if [[ "${RECIPE}" == "la4" ]]; then
  VISION_MASK_PROBABILITY=0.5
elif [[ "${RECIPE}" != "plain" ]]; then
  echo "RECIPE must be plain or la4" >&2
  exit 2
fi

PYTHON=/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python
OUTPUT_ROOT=/mnt/cunchu/zjq/pi05_method_baselines_20260830/checkpoints
LOG_ROOT=/mnt/cunchu/zjq/pi05_method_baselines_20260830/logs
LOCK_ROOT=/mnt/cunchu/zjq/pi05_method_baselines_20260830/locks
RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_PATH="${LOG_ROOT}/${RUN_NAME}.log"
CONTRACT_PATH="${LOG_ROOT}/${RUN_NAME}.contract.txt"
LOCK_PATH="${LOCK_ROOT}/${RUN_NAME}.lock"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${LOCK_ROOT}"
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

cat >"${CONTRACT_PATH}" <<EOF
run_name=${RUN_NAME}
recipe=${RECIPE}
method=$( [[ "${RECIPE}" == la4 ]] && echo 'PI0.5 + LA4VLA per-sample random vision masking' || echo 'stock PI0.5' )
vision_mask_probability=${VISION_MASK_PROBABILITY}
prompt=subtask_only (global_prompt never passed to model)
dataset=union2375 seven domain roots
start_inactive_arm_mask=meta/start_inactive_arm_block_mask.json in every root; 80 masked episodes total
norm_assets_dir=/mnt/cunchu/zjq/target
norm_asset_id=openpi_norm_union2375_allframes_v1
normalization=q01_q99
state=16 physical dims padded to 32
action=joint_delta 16 physical dims padded to 32
action_horizon=50
subtask_horizon_padding=true
model=stock PI0.5 gemma_2b + action_expert_300m
initialization=/mnt/cunchu/zjq/.cache/openpi/openpi-assets/checkpoints/pi05_base/params
vision_frozen=false
global_batch_size=256
devices=8
fsdp_devices=1
mesh=[8,1] (DDP replicated parameters)
optimizer=AdamW b1=0.9 b2=0.95 eps=1e-8 weight_decay=1e-10 grad_clip=1.0
steps=25000
lr=warmup1000 peak2.5e-5 cosine_decay30000 floor2.5e-6
save_interval=1000
num_workers=64
source_ram=${RAM_REPO}
entrypoint_sha256=$(sha256sum "${RAM_REPO}/scripts/train_plain_pi05_la4_baseline.py" | awk '{print $1}')
model_sha256=$(sha256sum "${RAM_REPO}/vendor/pi0.5/src/openpi/models/pi0.py" | awk '{print $1}')
gemma_adapter_sha256=$(sha256sum "${RAM_REPO}/vendor/pi0.5/src/openpi/models/gemma.py" | awk '{print $1}')
EOF

cd "${RAM_REPO}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="${RAM_REPO}/vendor/pi0.5/src:${RAM_REPO}/vendor/pi0.5/packages/openpi-client/src:${RAM_REPO}/src"
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
  --vision-mask-probability "${VISION_MASK_PROBABILITY}" \
  >>"${LOG_PATH}" 2>&1
