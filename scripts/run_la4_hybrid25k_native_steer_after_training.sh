#!/usr/bin/env bash
set -euo pipefail

EVAL_RAM=${EVAL_RAM:?set EVAL_RAM}
TRAIN_PID=${TRAIN_PID:?set TRAIN_PID}
PYTHON=/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python
BASE_MANAGER=/mnt/cunchu/zjq/pi05_method_baselines_20260830/checkpoints/pi05_la4mixed50_union2375_masked_subtask_ddp8_bs256_nw64_25k_20260830/pi05_la4mixed50_union2375_masked_subtask_ddp8_bs256_nw64_25k_20260830
HYBRID_MANAGER=/mnt/cunchu/zjq/pi05_method_baselines_20260830/checkpoints/pi05_la4mixed50_atom_if_valid_else_subtask_from20k_to25k_20260901/pi05_la4mixed50_atom_if_valid_else_subtask_from20k_to25k_20260901
FINAL_METADATA=${HYBRID_MANAGER}/25000/_CHECKPOINT_METADATA
DATASET=/mnt/cunchu/zjq/target/lerobot_2058_domain_split_20260822/fruit
MANIFEST=${EVAL_RAM}/eval_manifests/fruit64_isolated_arm_selection_manifest.json
OUT_ROOT=/mnt/cunchu/zjq/pi05_method_baselines_20260830/evals/la4_native_atomic_steer_fruit64_20260901

cleanup() {
  if [[ "${EVAL_RAM}" == /dev/shm/atomic_latent_vla_eval_la4hybrid_* && -d "${EVAL_RAM}" ]]; then
    rm -rf -- "${EVAL_RAM}"
  fi
}
trap cleanup EXIT INT TERM

if [[ -e "${OUT_ROOT}" ]]; then
  echo "refusing to overwrite evaluation output: ${OUT_ROOT}" >&2
  exit 3
fi
test -x "${PYTHON}"
test -f "${MANIFEST}"

while [[ ! -f "${FINAL_METADATA}" ]]; do
  if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
    echo "training exited before the finalized 25K checkpoint appeared" >&2
    exit 4
  fi
  sleep 30
done
while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  sleep 10
done

mkdir -p "${OUT_ROOT}"/{base20k,base25k,hybrid25k}
cd "${EVAL_RAM}"
export PYTHONPATH="${EVAL_RAM}/vendor/pi0.5/src:${EVAL_RAM}/vendor/pi0.5/packages/openpi-client/src:${EVAL_RAM}/src:${EVAL_RAM}/scripts"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

run_eval() {
  local gpu=$1
  local name=$2
  local checkpoint=$3
  local output_dir=${OUT_ROOT}/${name}
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" \
    scripts/evaluate_plain_pi05_native_atomic_steering.py \
    --checkpoint "${checkpoint}" \
    --checkpoint-name "${name}" \
    --dataset-root "${DATASET}" \
    --selection-manifest "${MANIFEST}" \
    --output "${output_dir}/shard_0.json" \
    --num-shards 1 \
    --shard-index 0 \
    --noise-repeats 2 \
    --batch-size 32 \
    --seed 20260827 \
    --norm-assets-dir /mnt/cunchu/zjq/target \
    --norm-asset-id openpi_norm_union2375_allframes_v1 \
    >"${output_dir}/eval.log" 2>&1
  "${PYTHON}" scripts/analyze_plain_pi05_native_atomic_steering.py \
    --input-dir "${output_dir}" \
    --output-dir "${output_dir}/report_5mm_1deg" \
    --translation-threshold-mm 5 \
    --rotation-threshold-deg 1 \
    >"${output_dir}/analyze_5mm_1deg.log" 2>&1
  "${PYTHON}" scripts/analyze_plain_pi05_native_atomic_steering.py \
    --input-dir "${output_dir}" \
    --output-dir "${output_dir}/report_10mm_2deg" \
    --translation-threshold-mm 10 \
    --rotation-threshold-deg 2 \
    >"${output_dir}/analyze_10mm_2deg.log" 2>&1
}

run_eval 0 base20k "${BASE_MANAGER}/20000" & p0=$!
run_eval 1 base25k "${BASE_MANAGER}/25000" & p1=$!
run_eval 2 hybrid25k "${HYBRID_MANAGER}/25000" & p2=$!
wait "${p0}" "${p1}" "${p2}"
date -Is >"${OUT_ROOT}/DONE"
