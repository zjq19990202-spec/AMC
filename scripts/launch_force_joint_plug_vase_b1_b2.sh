#!/usr/bin/env bash
set -euo pipefail

: "${RAM_REPO:?set a unique /dev/shm/atomic_latent_vla_run_* source copy}"
: "${BASE_PARAMS:?set the common force-free base params directory}"
: "${RUN_PREFIX:?set a unique run prefix}"
: "${CUDA_DEVICES:?set the physical GPU list}"

PI05_PYTHON=${PI05_PYTHON:-/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python}
NORM_ROOT=${NORM_ROOT:-/mnt/cunchu/zjq/target}
NORM_ID=${NORM_ID:-openpi_norm_union2375_allframes_v1}
FORCE_NORM=${FORCE_NORM:-/mnt/cunchu/zjq/force_assets/force_norm_plug_vase_shared_lr_v1_20260822/norm_stats.json}
PLUG_ROOT=${PLUG_ROOT:-/mnt/cunchu/zjq/target/lerobot_v3_4to1_plug_force}
VASE_ROOT=${VASE_ROOT:-/mnt/cunchu/zjq/target/lerobot_v3_4to1_vase_force}
B1_ROOT=${B1_ROOT:-/mnt/cunchu/zjq/atomic_pi05_force_runs}
B2_ROOT=${B2_ROOT:-/mnt/cunchu/zjq/atomic_pi05_force_b2_runs}
LOG_ROOT=${LOG_ROOT:-/mnt/cunchu/zjq/atomic_pi05_logs}
B1_STEPS=${B1_STEPS:-6000}
B2_STEPS=${B2_STEPS:-6000}
B1_WARMUP=${B1_WARMUP:-300}
B2_WARMUP=${B2_WARMUP:-300}
WIDTH=${WIDTH:-512}
HEADS=${HEADS:-8}
MLP_DIM=${MLP_DIM:-1024}
FORCE_LATENT_DIM=${FORCE_LATENT_DIM:-512}
ENCODER_DEPTH=${ENCODER_DEPTH:-2}

IFS=',' read -r -a gpu_list <<<"${CUDA_DEVICES}"
DEVICES=${#gpu_list[@]}
BATCH_SIZE=${BATCH_SIZE:-$((DEVICES * 32))}
NUM_WORKERS=${NUM_WORKERS:-$((DEVICES * 4))}
FSDP_DEVICES=${FSDP_DEVICES:-1}
B1_RUN=${RUN_PREFIX}_b1_s${B1_STEPS}
B2_RUN=${RUN_PREFIX}_b2_s${B2_STEPS}

cleanup() {
  status=$?
  if [[ "${RAM_REPO}" == /dev/shm/atomic_latent_vla_run_* ]]; then
    cd /
    rm -rf -- "${RAM_REPO}"
  fi
  exit "${status}"
}
trap cleanup EXIT

for file in \
  "${BASE_PARAMS}/_METADATA" "${BASE_PARAMS}/manifest.ocdbt" \
  "${PLUG_ROOT}/meta/info.json" "${VASE_ROOT}/meta/info.json" \
  "${NORM_ROOT}/${NORM_ID}/norm_stats.json" "${FORCE_NORM}"; do
  [[ -f "${file}" ]] || { echo "missing file: ${file}" >&2; exit 1; }
done
[[ ${WIDTH} -gt 0 && $((WIDTH % HEADS)) -eq 0 ]] || {
  echo "WIDTH must be positive and divisible by HEADS" >&2; exit 1;
}
[[ ! -e "${B1_ROOT}/${B1_RUN}" ]] || { echo "B1 destination exists" >&2; exit 1; }
[[ ! -e "${B2_ROOT}/${B2_RUN}" ]] || { echo "B2 destination exists" >&2; exit 1; }
mkdir -p "${B1_ROOT}" "${B2_ROOT}" "${LOG_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="${RAM_REPO}/vendor/pi0.5/src:${RAM_REPO}/vendor/pi0.5/packages/openpi-client/src:${RAM_REPO}/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_MEM_FRACTION:-0.75}
cd "${RAM_REPO}"

common_data=(
  --dataset-root "${PLUG_ROOT}"
  --dataset-root "${VASE_ROOT}"
  --norm-assets-dir "${NORM_ROOT}"
  --norm-asset-id "${NORM_ID}"
  --force-norm "${FORCE_NORM}"
  --coefficient-target joint_delta
  --batch-size "${BATCH_SIZE}"
  --devices "${DEVICES}"
  --fsdp-devices "${FSDP_DEVICES}"
  --num-workers "${NUM_WORKERS}"
  --max-token-len 200
  --encoder-width "${WIDTH}"
  --encoder-depth "${ENCODER_DEPTH}"
  --encoder-heads "${HEADS}"
  --encoder-mlp-dim "${MLP_DIM}"
  --force-latent-dim "${FORCE_LATENT_DIM}"
  --validation-modulus 10
  --validation-remainder 0
  --save-interval 1000
  --keep-period 5000
  --log-interval 10
  --seed 0
)

"${PI05_PYTHON}" scripts/train_force_encoder_b1.py \
  "${common_data[@]}" \
  --base-params "${BASE_PARAMS}" \
  --checkpoint-base-dir "${B1_ROOT}" \
  --run-name "${B1_RUN}" \
  --future-decoder-stride 4 \
  --future-decoder-kind phase_mlp \
  --position-base 10000 \
  --history-train-lengths 120 \
  --eval-batches 20 \
  --steps "${B1_STEPS}" \
  --warmup-steps "${B1_WARMUP}" \
  --peak-lr 1.0e-4 \
  --decay-steps "${B1_STEPS}" \
  --decay-lr 1.0e-5 \
  2>&1 | tee "${LOG_ROOT}/${B1_RUN}.log"

B1_PARAMS="${B1_ROOT}/${B1_RUN}/${B1_RUN}/${B1_STEPS}/params"
[[ -f "${B1_PARAMS}/_METADATA" ]] || { echo "B1 params missing: ${B1_PARAMS}" >&2; exit 1; }

"${PI05_PYTHON}" scripts/train_force_stage_b2.py \
  "${common_data[@]}" \
  --b1-params "${B1_PARAMS}" \
  --checkpoint-base-dir "${B2_ROOT}" \
  --run-name "${B2_RUN}" \
  --eval-batches 2 \
  --steps "${B2_STEPS}" \
  --warmup-steps "${B2_WARMUP}" \
  --peak-lr 3.0e-5 \
  --decay-steps "${B2_STEPS}" \
  --decay-lr 3.0e-6 \
  --future-force-loss-weight 0 \
  --force-update-action-steps 10 \
  --force-update-offsets 0,10,20,30,40 \
  --flow-loss-weight 1.0 \
  --delta-regularization-weight 1.0e-4 \
  2>&1 | tee "${LOG_ROOT}/${B2_RUN}.log"
