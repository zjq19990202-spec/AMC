#!/usr/bin/env bash
set -euo pipefail

: "${RAM_REPO:?set a unique /dev/shm source directory}"
: "${DOMAIN:?set fruit, vase, or plug}"
: "${DATASET:?set the matching dataset root}"
: "${GPU_LIST:?set four comma-separated GPU ids}"

PI05_PYTHON=/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python
CHECKPOINT_ROOT=/mnt/cunchu/zjq/atomic_pi05_runs
NORM_ROOT=/mnt/cunchu/zjq/target
NORM_ID=openpi_norm_union2375_allframes_v1
EXPECTED_NORM_SHA256=512debfe8b5a905287ab9993b29cbcb8bb246c10ac692f5ad6d341637c4aebcc
TCP_NORM=${NORM_ROOT}/${NORM_ID}/tcp_twist_norm_bimanual_tcp200.json
EXPECTED_TCP_NORM_SHA256=aa30efd013b1c656d126fd47d518321875c15a6bb5ea15ed0dd4a4248aa46b73
COMPOSITION=fk_horizon_3hz_gate_top5_stay_v2
ZT_RUN=union2375_corrected_screw_vase_zt25k_ddp8_bs256_20260825
ZT_CHECKPOINT=${CHECKPOINT_ROOT}/${ZT_RUN}/${ZT_RUN}/25000
TEACHER=${CHECKPOINT_ROOT}/teachers/union2375_corrected_screw_vase_zt25k_20260825_${DOMAIN}_stay_v2
RUN=${RUN_NAME:-union2375_${DOMAIN}_final_zm_all18_fsdp4_bs128_20k_20260827}
TRAIN_STEPS=${TRAIN_STEPS:-20000}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
RESUME=${RESUME:-false}
RESUME_EXPECT_STEP=${RESUME_EXPECT_STEP:-}
RESUME_TAG=${RESUME_TAG:-}
LOG_ROOT=${CHECKPOINT_ROOT}/logs
LOCK=${CHECKPOINT_ROOT}/.locks/${RUN}.lock
OPENPI_ROOT=${RAM_REPO}/vendor/pi0.5

case "${RESUME}" in
  true)
    [[ "${RESUME_EXPECT_STEP}" =~ ^[0-9]+$ ]]
    RESUME_TAG=${RESUME_TAG:-resume${RESUME_EXPECT_STEP}_save${SAVE_INTERVAL}}
    LOG=${LOG_ROOT}/${RUN}.${RESUME_TAG}.log
    CONTRACT=${LOG_ROOT}/${RUN}.${RESUME_TAG}.contract.txt
    RESUME_ARGS=(--resume)
    ;;
  false)
    LOG=${LOG_ROOT}/${RUN}.log
    CONTRACT=${LOG_ROOT}/${RUN}.contract.txt
    RESUME_ARGS=()
    ;;
  *)
    echo "RESUME must be true or false, got ${RESUME}" >&2
    exit 1
    ;;
esac

cleanup() {
  status=$?
  if [[ "${RAM_REPO}" == /dev/shm/atomic_latent_vla_run_* ]]; then
    cd /
    rm -rf -- "${RAM_REPO}"
  fi
  exit "${status}"
}
trap cleanup EXIT

mkdir -p "${LOG_ROOT}" "${CHECKPOINT_ROOT}/.locks"
exec 9>"${LOCK}"
flock -n 9 || { echo "run lock already held: ${LOCK}" >&2; exit 1; }

if [[ "${RESUME}" == true ]]; then
  RUN_CHECKPOINT_DIR=${CHECKPOINT_ROOT}/${RUN}/${RUN}
  test -f "${RUN_CHECKPOINT_DIR}/${RESUME_EXPECT_STEP}/_CHECKPOINT_METADATA"
  test -f "${RUN_CHECKPOINT_DIR}/${RESUME_EXPECT_STEP}/params/_METADATA"
  test -f "${RUN_CHECKPOINT_DIR}/${RESUME_EXPECT_STEP}/train_state/_METADATA"
  latest_step=$(
    find "${RUN_CHECKPOINT_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
      | grep -E '^[0-9]+$' | sort -n | tail -1
  )
  [[ "${latest_step}" == "${RESUME_EXPECT_STEP}" ]]
else
  test ! -e "${CHECKPOINT_ROOT}/${RUN}"
  latest_step=0
fi
test -d "${RAM_REPO}/src"
test -d "${OPENPI_ROOT}/src/openpi"
test -d "${ZT_CHECKPOINT}/params"
test -f "${DATASET}/meta/info.json"
test -f "${DATASET}/meta/episode_subtasks.jsonl"
test -d "${DATASET}/meta/${COMPOSITION}"
test -f "${TEACHER}/manifest.json"
test -f "${TEACHER}/directions.npy"
test -f "${TEACHER}/valid.npy"
test -f "${TEACHER}/codebook.npy"

norm_sha=$(sha256sum "${NORM_ROOT}/${NORM_ID}/norm_stats.json" | awk '{print $1}')
tcp_sha=$(sha256sum "${TCP_NORM}" | awk '{print $1}')
[[ "${norm_sha}" == "${EXPECTED_NORM_SHA256}" ]]
[[ "${tcp_sha}" == "${EXPECTED_TCP_NORM_SHA256}" ]]

python3 - "${TEACHER}/manifest.json" "${DATASET}" "${ZT_CHECKPOINT}/params" <<'PY'
import json
import pathlib
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
if pathlib.Path(manifest["dataset_root"]).resolve() != pathlib.Path(sys.argv[2]).resolve():
    raise SystemExit("teacher dataset contract mismatch")
if pathlib.Path(manifest["checkpoint_params"]).resolve() != pathlib.Path(sys.argv[3]).resolve():
    raise SystemExit("teacher checkpoint contract mismatch")
if manifest["norm_asset_id"] != "openpi_norm_union2375_allframes_v1":
    raise SystemExit("teacher norm contract mismatch")
if manifest["coefficient_target"] != "joint_delta":
    raise SystemExit("teacher coefficient-target contract mismatch")
PY

export CUDA_VISIBLE_DEVICES=${GPU_LIST}
export PYTHONPATH=${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${RAM_REPO}/src
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

{
  echo "started=$(date --iso-8601=seconds)"
  echo "host=$(hostname) domain=${DOMAIN} gpus=${GPU_LIST}"
  echo "run=${RUN}"
  echo "source_git_head=${SOURCE_GIT_HEAD:-unavailable_dirty_snapshot} source_tree=external_drive_authoritative_snapshot"
  echo "dataset=${DATASET} sampling=natural_rows_no_dataset_weight"
  echo "zt_checkpoint=${ZT_CHECKPOINT} teacher=${TEACHER} codebook_frozen=true"
  echo "norm_root=${NORM_ROOT} norm_id=${NORM_ID} norm_sha256=${norm_sha} mode=q01_q99 adapt_to_pi=true"
  echo "tcp_norm=${TCP_NORM} tcp_norm_sha256=${tcp_sha} coefficient_target=joint_delta"
  echo "batch=128 devices=4 per_gpu=32 fsdp_devices=4 mesh=1x4 accumulation=1"
  echo "resume=${RESUME} restored_step=${latest_step} additional_updates=${TRAIN_STEPS} save_interval=${SAVE_INTERVAL}"
  echo "warmup=1000 peak_lr=2.5e-5 decay_lr=2.5e-6 decay_steps=30000"
  echo "latent_route=one_final_zm_reused_in_all_18_action_expert_film_adapters layerwise_atomic_flow=false"
  echo "prompt=single_active_subtask_only atomic_prompt_probability=0 max_token_len=200 tokenizer=paligemma"
  echo "subtask_source=${DATASET}/meta/episode_subtasks.jsonl"
  echo "subtask_boundary=repeat_active_subtask_final_absolute_action_to_50 composition_supervision=masked_on_padded_rows"
  echo "zt_fraction=0 q1_teacher_cos_weight=0.05 drop_weighted_cos_weight=0.015"
  echo "state_clip=false action_clip=false output_clip=false optimizer_gradient_clip=1.0"
  sha256sum \
    "${DATASET}/meta/episode_subtasks.jsonl" \
    "${TEACHER}/manifest.json" \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/training_data.py" \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/model.py" \
    "${RAM_REPO}/src/atomic_latent_vla/pi05/gemma_adapter.py" \
    "${RAM_REPO}/scripts/train_atomic_pi05.py"
} >"${CONTRACT}"

cd "${RAM_REPO}"
"${PI05_PYTHON}" scripts/train_atomic_pi05.py \
  --dataset-root "${DATASET}" \
  --norm-assets-dir "${NORM_ROOT}" \
  --norm-asset-id "${NORM_ID}" \
  --zt-teacher-sidecar "${TEACHER}" \
  --atomic-composition-sidecar "${COMPOSITION}" \
  --pad-subtask-horizon \
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
  --steps "${TRAIN_STEPS}" \
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
  --no-layerwise-atomic-flow \
  --warmup-steps 1000 \
  --peak-lr 2.5e-5 \
  --decay-lr 2.5e-6 \
  --lr-decay-steps 30000 \
  --save-interval "${SAVE_INTERVAL}" \
  --log-interval 10 \
  "${RESUME_ARGS[@]}" \
  >"${LOG}" 2>&1
