#!/usr/bin/env bash
set -euo pipefail

: "${RAM_REPO:?set the unique /dev/shm source copy}"
: "${BASE:?set the immutable base params snapshot}"

AB_STEPS=${AB_STEPS:-500}
AB_WARMUP_STEPS=${AB_WARMUP_STEPS:-25}
AB_DECAY_STEPS=${AB_DECAY_STEPS:-${AB_STEPS}}
EVAL_BATCHES=${EVAL_BATCHES:-40}
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

for decoder_kind in linear_chunk phase_mlp; do
  run="force_b1_vase16k_${decoder_kind}_gru50x4_h120_${RUN_TAG}_20260823"
  env \
    KEEP_RAM=1 \
    RAM_REPO="${RAM_REPO}" \
    BASE="${BASE}" \
    RUN="${run}" \
    DECODER_STRIDE=4 \
    DECODER_KIND="${decoder_kind}" \
    ENCODER_DEPTH=2 \
    POSITION_BASE=10000 \
    HISTORY_LENGTHS=120 \
    STEPS="${AB_STEPS}" \
    BATCH_SIZE=4 \
    WARMUP_STEPS="${AB_WARMUP_STEPS}" \
    DECAY_STEPS="${AB_DECAY_STEPS}" \
    PEAK_LR=1e-4 \
    VALIDATION_MODULUS=10 \
    VALIDATION_REMAINDER=0 \
    EVAL_BATCHES="${EVAL_BATCHES}" \
    bash "${RAM_REPO}/scripts/launch_force_b1_vase2k_smoke_tmp8ka.sh"
done
