#!/usr/bin/env bash
set -euo pipefail

PI05_PYTHON=${PI05_PYTHON:-/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python}

BASE=/mnt/cunchu/zjq/atomic_pi05_runs
FORMAL=/mnt/cunchu/zjq/atomic_latent_vla_layerwise
DATASET=/mnt/cunchu/zjq/target/lerobot_compact_accepted_final
TEACHER=${DATASET}/meta/zt_q_teacher_target_v3_layerwise_shared_flow_zt27k_atomic_v1
PRECOMPUTE_PID_FILE=${BASE}/logs/precompute_zt27k_atomic_teacher.pid
RUN=target_v3_offline_zm_layerwise_from_zt27k_q005_kl0025_ddp8_bs256_40k
SMOKE_RUN=${RUN}_smoke1
RAM=/dev/shm/atomic_latent_vla_run_${RUN}
LOG=${BASE}/logs/${RUN}.log
SMOKE_LOG=${BASE}/logs/${SMOKE_RUN}.log
WATCH_LOG=${BASE}/logs/${RUN}.watch.log
PID_FILE=${BASE}/logs/${RUN}.pid
LOCK=${BASE}/.locks/${RUN}.launch.lock

mkdir -p "${BASE}/logs" "${BASE}/.locks"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "offline ZM launch watcher already active" >&2
  exit 70
fi
exec >>"${WATCH_LOG}" 2>&1
echo "$(date --iso-8601=seconds) waiting for offline teacher ${TEACHER}"

while [[ ! -f "${TEACHER}/manifest.json" ]]; do
  if [[ -f "${PRECOMPUTE_PID_FILE}" ]]; then
    precompute_pid=$(<"${PRECOMPUTE_PID_FILE}")
    if ! kill -0 "${precompute_pid}" 2>/dev/null; then
      echo "teacher precompute exited before publishing manifest"
      exit 71
    fi
  fi
  sleep 30
done

"${PI05_PYTHON}" - "${TEACHER}" <<'PY'
import json
import pathlib
import sys
import numpy as np

root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
assert manifest["version"] == 1
assert manifest["prompt"] == "atomic"
assert manifest["norm_asset_id"] == "openpi_norm_compact_accepted_v3"
assert manifest["max_token_len"] == 144
assert manifest["coefficient_target"] == "joint_delta"
directions = np.load(root / "directions.npy", mmap_mode="r")
valid = np.load(root / "valid.npy", mmap_mode="r")
codebook = np.load(root / "codebook.npy")
assert directions.shape == (manifest["dataset_rows"], 2, manifest["latent_dim"])
assert directions.dtype == np.float16
assert valid.shape == (manifest["dataset_rows"],) and valid.dtype == np.bool_
assert int(valid.sum()) == manifest["encoded_rows"]
assert codebook.shape == (2, 13, manifest["latent_dim"])
np.testing.assert_allclose(np.linalg.norm(codebook, axis=-1), 1.0, atol=2e-3)
print(json.dumps({"encoded_rows": int(valid.sum()), "dataset_rows": len(valid)}))
PY
echo "$(date --iso-8601=seconds) offline teacher validated"

if [[ -e "${RAM}" ]]; then
  echo "refusing to overwrite RAM source ${RAM}"
  exit 72
fi
cp -a "${FORMAL}" "${RAM}"

echo "$(date --iso-8601=seconds) running offline ZM DDP batch256 smoke"
env \
  REPO="${RAM}" \
  RUN="${SMOKE_RUN}" \
  STEPS=1 \
  SMOKE_STEPS=1 \
  PHYSICAL_BATCH_SIZE=256 \
  bash "${RAM}/scripts/run_target_v3_offline_zm_from_zt27k_q005_kl0025_ddp8_bs128_40k.sh" \
  >"${SMOKE_LOG}" 2>&1
echo "$(date --iso-8601=seconds) offline ZM smoke passed"

if [[ -e "${BASE}/${RUN}" || -f "${PID_FILE}" ]]; then
  echo "refusing to overwrite existing offline ZM run or pid file"
  exit 73
fi
setsid env \
  REPO="${RAM}" \
  RUN="${RUN}" \
  STEPS=40000 \
  SMOKE_STEPS=0 \
  PHYSICAL_BATCH_SIZE=256 \
  bash "${RAM}/scripts/run_target_v3_offline_zm_from_zt27k_q005_kl0025_ddp8_bs128_40k.sh" \
  >"${LOG}" 2>&1 &
pid=$!
printf '%s\n' "${pid}" >"${PID_FILE}"
echo "$(date --iso-8601=seconds) launched offline ZM pid=${pid}"

for _ in $(seq 1 20); do
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "offline ZM exited during startup"
    tail -n 100 "${LOG}"
    exit 74
  fi
  if grep -q "step=2700[1-9]" "${LOG}"; then
    echo "$(date --iso-8601=seconds) offline ZM produced its first update"
    exit 0
  fi
  sleep 30
done
echo "offline ZM is alive but no first update appeared within ten minutes"
exit 75
