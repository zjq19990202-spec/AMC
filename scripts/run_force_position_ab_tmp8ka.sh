#!/usr/bin/env bash
set -euo pipefail

: "${RAM_REPO:?set the unique /dev/shm source copy}"
: "${BASE:?set the immutable base params snapshot}"

cleanup() {
  status=$?
  if [[ "${RAM_REPO}" == /dev/shm/atomic_latent_vla_run_* ]]; then
    cd /
    rm -rf -- "${RAM_REPO}"
  fi
  exit "${status}"
}
trap cleanup EXIT

for position_base in 100 1000 10000; do
  run="force_b1_vase3k_pe_gru50x4_b${position_base}_h120_s60_20260823"
  env \
    KEEP_RAM=1 \
    RAM_REPO="${RAM_REPO}" \
    BASE="${BASE}" \
    RUN="${run}" \
    DECODER_STRIDE=4 \
    POSITION_BASE="${position_base}" \
    HISTORY_LENGTHS=120 \
    STEPS=60 \
    BATCH_SIZE=4 \
    WARMUP_STEPS=10 \
    DECAY_STEPS=60 \
    PEAK_LR=1e-4 \
    bash "${RAM_REPO}/scripts/launch_force_b1_vase2k_smoke_tmp8ka.sh"
done
