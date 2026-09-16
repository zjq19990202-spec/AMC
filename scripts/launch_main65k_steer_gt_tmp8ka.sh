#!/usr/bin/env bash
set -euo pipefail

: "${RAM_REPO:?set RAM_REPO to the unique tmp8ka RAM source copy}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to a unique persistent output directory}"

OPENPI_ROOT=${OPENPI_ROOT:-${RAM_REPO}/vendor/pi0.5}
PI05_PYTHON=${PI05_PYTHON:-/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python}

CHECKPOINT=${CHECKPOINT:-/mnt/cunchu/zjq/atomic_pi05_runs/target_v3_action_unclip_main61k_fulladam_zm5k_subonly_q005_p0015_ddp8_bs256/target_v3_action_unclip_main61k_fulladam_zm5k_subonly_q005_p0015_ddp8_bs256/65000}
DATASET=${DATASET:-/mnt/cunchu/zjq/target/lerobot_compact_accepted_final}
NORM_ROOT=${NORM_ROOT:-/mnt/cunchu/zjq/target}
NORM_ID=${NORM_ID:-openpi_norm_compact_accepted_v3}
ATOMIC_SIDECAR=${ATOMIC_SIDECAR:-fk_horizon_3hz_gate_top5_stay_v2}
MANIFEST=${MANIFEST:-${RAM_REPO}/eval_inputs/isolated_arm_selection_manifest.json}
CLEAN_RAM_ON_EXIT=${CLEAN_RAM_ON_EXIT:-1}

pids=()
cleanup() {
  local status=$?
  for pid in "${pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  if [[ "${CLEAN_RAM_ON_EXIT}" == 1 && "${RAM_REPO}" == /dev/shm/atomic_latent_vla_eval_* ]]; then
    cd /
    rm -rf -- "${RAM_REPO}"
  fi
  exit "${status}"
}
trap cleanup EXIT

test -d "${RAM_REPO}/src"
test -d "${CHECKPOINT}/params"
test -f "${DATASET}/meta/info.json"
test -f "${NORM_ROOT}/${NORM_ID}/norm_stats.json"
test -d "${DATASET}/meta/${ATOMIC_SIDECAR}"
test -f "${MANIFEST}"
test ! -e "${OUTPUT_ROOT}"

mkdir -p "${OUTPUT_ROOT}/steer"
cp "${MANIFEST}" "${OUTPUT_ROOT}/isolated_arm_selection_manifest.json"
cp "${RAM_REPO}/eval_inputs/run_contract.md" "${OUTPUT_ROOT}/run_contract.md"

export PYTHONPATH=${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${RAM_REPO}/src
cd "${RAM_REPO}"

echo "evaluation_start=$(date --iso-8601=seconds)"
echo "checkpoint=${CHECKPOINT}"
echo "dataset=${DATASET}"
echo "norm=${NORM_ID} norm_sha256=$(sha256sum "${NORM_ROOT}/${NORM_ID}/norm_stats.json" | awk '{print $1}')"
echo "manifest_sha256=$(sha256sum "${MANIFEST}" | awk '{print $1}')"
sha256sum \
  scripts/evaluate_500_anchor_atomic_sweep.py \
  scripts/analyze_cluster_seen_atomic_sweep.py \
  scripts/evaluate_subtask_atomic_episode_chunks.py \
  src/atomic_latent_vla/pi05/training_data.py \
  src/atomic_latent_vla/pi05/model.py

# Port 45567 exposes one H800. Run GT first, then evaluate the complete steering
# manifest in one process so JAX only pays the model-load/compile cost once.
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "${PI05_PYTHON}" scripts/evaluate_subtask_atomic_episode_chunks.py \
    --checkpoint "${CHECKPOINT}" \
    --checkpoint-label "main65k" \
    --source-revision "52796fb-dirty" \
    --dataset-root "${DATASET}" \
    --episode 184 \
    --episode 891 \
    --episode 1308 \
    --episode 1703 \
    --norm-assets-dir "${NORM_ROOT}" \
    --norm-asset-id "${NORM_ID}" \
    --atomic-composition-sidecar "${ATOMIC_SIDECAR}" \
    --seed 20260821 \
    --num-steps 10 \
    --batch-size 32 \
    --output-dir "${OUTPUT_ROOT}/gt" \
    >"${OUTPUT_ROOT}/gt.evaluate.log" 2>&1
mv "${OUTPUT_ROOT}/gt.evaluate.log" "${OUTPUT_ROOT}/gt/evaluate.log"

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "${PI05_PYTHON}" scripts/evaluate_500_anchor_atomic_sweep.py \
    --checkpoint "${CHECKPOINT}" \
    --checkpoint-name main65k \
    --dataset-root "${DATASET}" \
    --selection-manifest "${MANIFEST}" \
    --output "${OUTPUT_ROOT}/steer/shard_0.json" \
    --num-shards 1 \
    --shard-index 0 \
    --noise-repeats 1 \
    --cluster-seen-only \
    --batch-size 64 \
    --max-token-len 200 \
    --coefficient-target-kind joint_delta \
    --coefficient-target-dim 14 \
    --enable-layerwise-atomic-flow \
    --norm-assets-dir "${NORM_ROOT}" \
    --norm-asset-id "${NORM_ID}" \
    --atomic-composition-sidecar "${ATOMIC_SIDECAR}" \
    --seed 20260820 \
    >"${OUTPUT_ROOT}/steer/shard_0.log" 2>&1

"${PI05_PYTHON}" scripts/analyze_cluster_seen_atomic_sweep.py \
  --input-dir "${OUTPUT_ROOT}/steer" \
  --output-dir "${OUTPUT_ROOT}/steer/report" \
  --translation-threshold-mm 5 \
  --rotation-threshold-deg 1 \
  >"${OUTPUT_ROOT}/steer/analyze.log" 2>&1

echo "evaluation_complete=$(date --iso-8601=seconds)"
