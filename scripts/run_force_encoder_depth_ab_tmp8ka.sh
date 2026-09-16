#!/usr/bin/env bash
set -euo pipefail

: "${RAM_REPO:?set the unique /dev/shm source copy}"
: "${BASE:?set the immutable base params snapshot}"
: "${POSITION_BASE:?set the position base selected by the preceding ablation}"

AB_STEPS=${AB_STEPS:-60}
AB_WARMUP_STEPS=${AB_WARMUP_STEPS:-10}
AB_DECAY_STEPS=${AB_DECAY_STEPS:-${AB_STEPS}}
RUN_TAG=${RUN_TAG:-s${AB_STEPS}}

cleanup() {
  status=$?
  if [[ "${RAM_REPO}" == /dev/shm/atomic_latent_vla_run_* ]]; then
    cd /
    rm -rf -- "${RAM_REPO}"
  fi
  exit "${status}"
}
trap cleanup EXIT

for encoder_depth in 1 2; do
  run="force_b1_vase3k_encoder_d${encoder_depth}_gru50x4_b${POSITION_BASE}_h120_${RUN_TAG}_20260823"
  env \
    KEEP_RAM=1 \
    RAM_REPO="${RAM_REPO}" \
    BASE="${BASE}" \
    RUN="${run}" \
    DECODER_STRIDE=4 \
    ENCODER_DEPTH="${encoder_depth}" \
    POSITION_BASE="${POSITION_BASE}" \
    HISTORY_LENGTHS=120 \
    STEPS="${AB_STEPS}" \
    BATCH_SIZE=4 \
    WARMUP_STEPS="${AB_WARMUP_STEPS}" \
    DECAY_STEPS="${AB_DECAY_STEPS}" \
    PEAK_LR=1e-4 \
    bash "${RAM_REPO}/scripts/launch_force_b1_vase2k_smoke_tmp8ka.sh"
done
