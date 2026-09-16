#!/usr/bin/env bash
set -euo pipefail

: "${RAM_REPO:?set the unique tmp8ka RAM source directory}"

RUN=union2375_corrected_screw_vase_zt25k_ddp8_bs256_20260825
DOMAIN=/mnt/cunchu/zjq/target/lerobot_2058_domain_split_20260822
PLUG_VASE=/mnt/cunchu/zjq/target/lerobot_plug_vase_split_20260822
SCREW=/mnt/cunchu/zjq/target/lerobot_2058_screw5_subtask_atomicfix_20260825
VASE=/mnt/cunchu/zjq/target/lerobot_vase167_manual_atomicfix_20260825
DATASETS=(
  "${DOMAIN}/cabinet"
  "${DOMAIN}/drawer"
  "${DOMAIN}/fruit"
  "${DOMAIN}/mixed_rest"
  "${SCREW}"
  "${PLUG_VASE}/plug"
  "${VASE}"
)
NORM_ROOT=/mnt/cunchu/zjq/target
NORM_ID=openpi_norm_union2375_allframes_v1
EXPECTED_NORM_SHA256=512debfe8b5a905287ab9993b29cbcb8bb246c10ac692f5ad6d341637c4aebcc
TCP_NORM=${NORM_ROOT}/${NORM_ID}/tcp_twist_norm_bimanual_tcp200.json
EXPECTED_TCP_NORM_SHA256=aa30efd013b1c656d126fd47d518321875c15a6bb5ea15ed0dd4a4248aa46b73
COMPOSITION=fk_horizon_3hz_gate_top5_stay_v2
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
test ! -e "${CHECKPOINT_ROOT}/${RUN}"
test -d "${RAM_REPO}/src"
test -d "${OPENPI_ROOT}/src/openpi"

total=0
for dataset in "${DATASETS[@]}"; do
  test -f "${dataset}/meta/info.json"
  test -f "${dataset}/meta/episode_subtasks.jsonl"
  test -f "${dataset}/meta/atomic_horizon_prompts_3hz.jsonl"
  test -d "${dataset}/meta/${COMPOSITION}"
  count=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["total_episodes"])' "${dataset}/meta/info.json")
  total=$((total + count))
done
[[ "${total}" -eq 2375 ]]

norm_sha=$(sha256sum "${NORM_ROOT}/${NORM_ID}/norm_stats.json" | awk '{print $1}')
tcp_sha=$(sha256sum "${TCP_NORM}" | awk '{print $1}')
[[ "${norm_sha}" == "${EXPECTED_NORM_SHA256}" ]]
[[ "${tcp_sha}" == "${EXPECTED_TCP_NORM_SHA256}" ]]

export PYTHONPATH=${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${RAM_REPO}/src
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

{
  echo "started=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "run=${RUN}"
  echo "recipe=fresh ZT from released PI0.5 base"
  echo "episodes=${total} domains=cabinet544,drawer183,fruit269,mixed_rest849,corrected_screw213,plug150,corrected_vase167"
  printf 'dataset=%s\n' "${DATASETS[@]}"
  echo "prompt_route=atomic_horizon_prompts_3hz.jsonl"
  echo "atomic_composition=${COMPOSITION}"
  echo "norm_root=${NORM_ROOT} norm_id=${NORM_ID} norm_sha256=${norm_sha} mode=q01_q99 adapt_to_pi=true"
  echo "tcp_norm=${TCP_NORM} tcp_norm_sha256=${tcp_sha}"
  echo "state_clip=off action_clip=off inference_clip=not_applicable optimizer_grad_clip=1.0"
  echo "batch=256 devices=8 per_gpu=32 fsdp_devices=1 accumulation=1"
  echo "steps=25000 warmup=1000 peak_lr=2.5e-5 decay_lr=2.5e-6 decay_steps=30000"
  echo "loss_flow=1.0 loss_atomic_twoway=0.1 loss_weighted_cos=0.1 loss_dct=0 loss_kl=0"
  sha256sum \
    "${SCREW}/meta/episode_subtasks.jsonl" \
    "${SCREW}/meta/atomic_horizon_prompts_3hz.jsonl" \
    "${VASE}/meta/episode_subtasks.jsonl" \
    "${VASE}/meta/atomic_horizon_prompts_3hz.jsonl" \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/training_data.py" \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/model.py" \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/gemma_adapter.py" \
    "${RAM_REPO}/scripts/train_atomic_zt_fast.py"
} >"${CONTRACT}"

args=()
for dataset in "${DATASETS[@]}"; do args+=(--dataset-root "${dataset}"); done

cd "${RAM_REPO}"
"${PI05_PYTHON}" scripts/train_atomic_zt_fast.py \
  "${args[@]}" \
  --norm-assets-dir "${NORM_ROOT}" \
  --norm-asset-id "${NORM_ID}" \
  --atomic-composition-sidecar "${COMPOSITION}" \
  --tcp-twist-norm "${TCP_NORM}" \
  --coefficient-target joint_delta \
  --base-params /mnt/cunchu/zjq/.cache/openpi/openpi-assets/checkpoints/pi05_base/params \
  --checkpoint-base-dir "${CHECKPOINT_ROOT}" \
  --run-name "${RUN}" \
  --batch-size 256 \
  --devices 8 \
  --fsdp-devices 1 \
  --max-token-len 200 \
  --num-workers 64 \
  --steps 25000 \
  --save-interval 1000 \
  --log-interval 10 \
  --warmup-steps 1000 \
  --peak-lr 2.5e-5 \
  --decay-lr 2.5e-6 \
  --lr-decay-steps 30000 \
  --codebook-freeze-step 25000 \
  --flow-loss-weight 1.0 \
  --coefficient-loss-weight 0.0 \
  --atomic-loss-weight 0.1 \
  --atomic-projection-loss-weight 0.1 \
  --codebook-loss-weight 0.0 \
  >"${LOG}" 2>&1
