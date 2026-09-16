#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 ZT_STEP_DIR [OUTPUT_DIR]" >&2
  exit 2
fi

REPO=/mnt/cunchu/zjq/atomic_latent_vla_layerwise
OPENPI_ROOT=${OPENPI_ROOT:-${REPO}/vendor/pi0.5}
PI05_PYTHON=${PI05_PYTHON:-/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python}
DATASET=/mnt/cunchu/zjq/target/lerobot_compact_accepted_final
NORM_ROOT=/mnt/cunchu/zjq/target
NORM_ID=openpi_norm_compact_accepted_v3
ZT_CHECKPOINT=$1
OUTPUT=${2:-${DATASET}/meta/zt_q_teacher_target_v3_shared_flow_40k_v1}

export PYTHONPATH=${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${REPO}/src
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "${REPO}"

test -d "${ZT_CHECKPOINT}/params" || test -d "${ZT_CHECKPOINT}"
test ! -e "${OUTPUT}"

"${PI05_PYTHON}" scripts/precompute_target_zt_q_teacher.py \
  --checkpoint "${ZT_CHECKPOINT}" \
  --dataset-root "${DATASET}" \
  --norm-assets-dir "${NORM_ROOT}" \
  --norm-asset-id "${NORM_ID}" \
  --output "${OUTPUT}" \
  --batch-size 256 \
  --num-workers 32 \
  --max-token-len 200 \
  --coefficient-target joint_delta
