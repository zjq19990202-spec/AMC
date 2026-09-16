#!/usr/bin/env bash
set -euo pipefail

: "${RAM_REPO:?set RAM_REPO to a unique per-host /dev/shm source copy}"

OPENPI_ROOT=${OPENPI_ROOT:-${RAM_REPO}/vendor/pi0.5}
PI05_PYTHON=${PI05_PYTHON:-/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python}

DATASET=${DATASET:-/mnt/cunchu/zjq/target/lerobot_compact_accepted_v3_no_short_no_idle}
NORM_ROOT=${NORM_ROOT:-/mnt/cunchu/zjq/target}
NORM_ID=${NORM_ID:-openpi_norm_compact_accepted_v3_no_short_no_idle}
EXPECTED_NORM_SHA256=${EXPECTED_NORM_SHA256:-44acd8bfde7b2b3aada0925ff374085e0e046b7c9ea487e22cfced08d0dbcff5}
TCP_NORM=${TCP_NORM:-${DATASET}/meta/tcp_twist_norm_bimanual_tcp200.json}
ATOMIC_COMPOSITION_SIDECAR=${ATOMIC_COMPOSITION_SIDECAR:-fk_horizon_3hz_gate_top5_stay_v2}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/mnt/cunchu/zjq/atomic_pi05_runs}
ZT_RUN=${ZT_RUN:-target2058_marvin_pi05_zt_layerwise_25k_ddp8_bs256}
ZT_CHECKPOINT=${CHECKPOINT_ROOT}/${ZT_RUN}/${ZT_RUN}/25000
ZT_TEACHER=${ZT_TEACHER:-${CHECKPOINT_ROOT}/teachers/${ZT_RUN}_stay_v2}
EXPECTED_TEACHER_ROWS=${EXPECTED_TEACHER_ROWS:-1814110}
ZM_RUN=${ZM_RUN:-target2058_marvin_pi05_zm_subonly_q005_p0015_from_zt25k_40k_ddp8_bs256}
LOCK_ROOT=${CHECKPOINT_ROOT}/.locks
CLEAN_RAM_ON_EXIT=${CLEAN_RAM_ON_EXIT:-1}

mkdir -p "${LOCK_ROOT}" "${CHECKPOINT_ROOT}/teachers"
exec 9>"${LOCK_ROOT}/${ZM_RUN}.lock"
if ! flock -n 9; then
  echo "ZM lock is already held: ${ZM_RUN}" >&2
  exit 1
fi

cleanup() {
  local status=$?
  if [[ "${CLEAN_RAM_ON_EXIT}" == 1 && "${RAM_REPO}" == /dev/shm/atomic_latent_vla_run_* ]]; then
    cd /
    rm -rf -- "${RAM_REPO}"
  fi
  exit "${status}"
}
trap cleanup EXIT

test -d "${RAM_REPO}/src"
test -d "${ZT_CHECKPOINT}/params"
test -f "${DATASET}/meta/info.json"
test -f "${NORM_ROOT}/${NORM_ID}/norm_stats.json"
test -f "${TCP_NORM}"
test -d "${DATASET}/meta/${ATOMIC_COMPOSITION_SIDECAR}"
test ! -e "${ZT_TEACHER}"
test ! -e "${CHECKPOINT_ROOT}/${ZM_RUN}"

actual_norm_sha256=$(sha256sum "${NORM_ROOT}/${NORM_ID}/norm_stats.json" | awk '{print $1}')
if [[ "${actual_norm_sha256}" != "${EXPECTED_NORM_SHA256}" ]]; then
  echo "unexpected 2058 norm SHA-256: ${actual_norm_sha256}" >&2
  exit 1
fi

test -d "${OPENPI_ROOT}/src/openpi"
test -d "${OPENPI_ROOT}/packages/openpi-client/src/openpi_client"
test -x "${PI05_PYTHON}"
export PYTHONPATH=${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${RAM_REPO}/src
cd "${RAM_REPO}"

echo "continuation_start=$(date --iso-8601=seconds)"
echo "dataset=${DATASET} episodes=2058"
echo "norm=${NORM_ID} norm_sha256=${actual_norm_sha256}"
echo "zt_checkpoint=${ZT_CHECKPOINT}"
echo "teacher=${ZT_TEACHER} sidecar=${ATOMIC_COMPOSITION_SIDECAR}"
echo "zm_run=${ZM_RUN} steps=40000 batch=256 prompt=subtask_only q_weight=0.05 drop_weight=0.015"
sha256sum \
  src/atomic_latent_vla/pi05/training_data.py \
  src/atomic_latent_vla/pi05/model.py \
  src/atomic_latent_vla/pi05/gemma_adapter.py \
  scripts/precompute_target_zt_q_teacher.py \
  scripts/train_atomic_pi05.py

# Regenerate the frozen Q teacher using exactly the same Stay-aware strict-row
# contract as ZT and ZM. The previous incomplete teacher remains untouched.
env \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "${PI05_PYTHON}" scripts/precompute_target_zt_q_teacher.py \
    --checkpoint "${ZT_CHECKPOINT}" \
    --dataset-root "${DATASET}" \
    --norm-assets-dir "${NORM_ROOT}" \
    --norm-asset-id "${NORM_ID}" \
    --atomic-composition-sidecar "${ATOMIC_COMPOSITION_SIDECAR}" \
    --output "${ZT_TEACHER}" \
    --batch-size 2048 \
    --devices 8 \
    --num-workers 32 \
    --max-token-len 200 \
    --coefficient-target joint_delta

"${PI05_PYTHON}" - "${ZT_TEACHER}" "${EXPECTED_TEACHER_ROWS}" <<'PY'
import json
import sys
from pathlib import Path

teacher = Path(sys.argv[1])
expected = int(sys.argv[2])
manifest = json.loads((teacher / "manifest.json").read_text(encoding="utf-8"))
if manifest.get("atomic_composition_sidecar") != "fk_horizon_3hz_gate_top5_stay_v2":
    raise SystemExit(f"teacher sidecar mismatch: {manifest}")
if int(manifest["encoded_rows"]) != expected:
    raise SystemExit(f"teacher coverage mismatch: {manifest['encoded_rows']} != {expected}")
print(f"verified_teacher_rows={expected}")
PY

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

"${PI05_PYTHON}" scripts/train_atomic_pi05.py \
  --dataset-root "${DATASET}" \
  --norm-assets-dir "${NORM_ROOT}" \
  --norm-asset-id "${NORM_ID}" \
  --zt-teacher-sidecar "${ZT_TEACHER}" \
  --atomic-composition-sidecar "${ATOMIC_COMPOSITION_SIDECAR}" \
  --tcp-twist-norm "${TCP_NORM}" \
  --coefficient-target joint_delta \
  --base-params "${ZT_CHECKPOINT}/params" \
  --checkpoint-base-dir "${CHECKPOINT_ROOT}" \
  --run-name "${ZM_RUN}" \
  --batch-size 256 \
  --devices 8 \
  --fsdp-devices 1 \
  --gradient-accumulation-steps 1 \
  --max-token-len 200 \
  --num-workers 32 \
  --steps 40000 \
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
  --warmup-steps 1000 \
  --peak-lr 2.5e-5 \
  --decay-lr 2.5e-6 \
  --lr-decay-steps 30000 \
  --save-interval 1000 \
  --log-interval 10

echo "continuation_complete=$(date --iso-8601=seconds)"
