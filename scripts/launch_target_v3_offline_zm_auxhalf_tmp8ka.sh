#!/usr/bin/env bash
set -euo pipefail

BASE=/mnt/cunchu/zjq/atomic_pi05_runs
RUN=target_v3_offline_zm_layerwise_from_zt27k_q0025_kl00125_unfreezevit_ddp8_bs256_40k
RAM=/dev/shm/atomic_latent_vla_run_target_v3_offline_zm_layerwise_from_zt27k_q0025_kl00125_ddp8_bs256_40k
LOG=${BASE}/logs/${RUN}.log
LOCK=${BASE}/.locks/${RUN}.lock
SOURCE_COMMIT=52796fb3bd863419c69f815503a406aaaeb574f7

mkdir -p "${BASE}/logs" "${BASE}/.locks"
test -d "${RAM}"
test ! -e "${BASE}/${RUN}"

exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "run lock is already held: ${LOCK}" >&2
  exit 70
fi

cleanup() {
  rm -rf -- "${RAM}"
}
trap cleanup EXIT

exec >"${LOG}" 2>&1
echo "$(date --iso-8601=seconds) host=$(hostname) run=${RUN}"
echo "source_commit=${SOURCE_COMMIT} batch=256 devices=8 fsdp_devices=1 vision=unfrozen steps=40000 initial_step=27000"
echo "q_cosine_weight=0.025 top5_kl_weight=0.0125 codebook=frozen norm=openpi_norm_compact_accepted_v3 coefficient_target=joint_delta max_token_len=200"
sha256sum \
  "${RAM}/src/atomic_latent_vla/pi05/model.py" \
  "${RAM}/src/atomic_latent_vla/pi05/gemma_adapter.py" \
  "${RAM}/scripts/train_atomic_pi05.py"

env \
  REPO="${RAM}" \
  RUN="${RUN}" \
  STEPS=40000 \
  SMOKE_STEPS=0 \
  PHYSICAL_BATCH_SIZE=256 \
  FULL_ATOMIC_LOSS_WEIGHT=0.025 \
  ATOMIC_COMPOSITION_LOSS_WEIGHT=0.0125 \
  bash "${RAM}/scripts/run_target_v3_offline_zm_from_zt27k_q005_kl0025_ddp8_bs128_40k.sh"
