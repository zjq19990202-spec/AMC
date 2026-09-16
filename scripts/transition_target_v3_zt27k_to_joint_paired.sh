#!/usr/bin/env bash
set -euo pipefail

ZT_RUN=target_v3_bimanual_zt_layerwise_shared_flow_ddp8_seed0_bs256_40k
JOINT_RUN=target_v3_joint_paired_layerwise_from_zt27k_ddp8_bs128_40k
BASE=/mnt/cunchu/zjq/atomic_pi05_runs
FORMAL=/mnt/cunchu/zjq/atomic_latent_vla_layerwise
ZT_ROOT=${BASE}/${ZT_RUN}/${ZT_RUN}
ZT_CHECKPOINT=${ZT_ROOT}/27000
ZT_PID_FILE=${BASE}/logs/${ZT_RUN}.pid
JOINT_LOG=${BASE}/logs/${JOINT_RUN}.log
JOINT_PID_FILE=${BASE}/logs/${JOINT_RUN}.pid
TRANSITION_LOG=${BASE}/logs/${JOINT_RUN}.transition.log
TRANSITION_LOCK=${BASE}/.locks/${JOINT_RUN}.transition.lock
SMOKE_RUN=${JOINT_RUN}_smoke1
SMOKE_LOG=${BASE}/logs/${SMOKE_RUN}.log
RAM=/dev/shm/atomic_latent_vla_run_${JOINT_RUN}

mkdir -p "${BASE}/logs" "${BASE}/.locks"
exec 9>"${TRANSITION_LOCK}"
if ! flock -n 9; then
  echo "another ZT-to-joint transition watcher already holds ${TRANSITION_LOCK}" >&2
  exit 70
fi
exec >>"${TRANSITION_LOG}" 2>&1
echo "$(date --iso-8601=seconds) waiting for complete ${ZT_CHECKPOINT}"

while [[ ! -f "${ZT_CHECKPOINT}/_CHECKPOINT_METADATA" || ! -f "${ZT_CHECKPOINT}/params/_METADATA" ]]; do
  if ! pgrep -f "scripts/train_atomic_zt_fast.py.*--run-name ${ZT_RUN}" >/dev/null; then
    echo "$(date --iso-8601=seconds) ZT exited before checkpoint 27000 completed"
    exit 71
  fi
  sleep 30
done
echo "$(date --iso-8601=seconds) checkpoint 27000 is atomically visible"

if [[ ! -f "${ZT_PID_FILE}" ]]; then
  echo "missing ZT wrapper pid file: ${ZT_PID_FILE}"
  exit 72
fi
zt_pid=$(<"${ZT_PID_FILE}")
zt_cmd=$(ps -o args= -p "${zt_pid}" || true)
if [[ "${zt_cmd}" != *"${ZT_RUN}"* ]]; then
  echo "refusing to stop unverified pid ${zt_pid}: ${zt_cmd}"
  exit 73
fi
zt_pgid=$(ps -o pgid= -p "${zt_pid}" | tr -d ' ')
if [[ ! "${zt_pgid}" =~ ^[0-9]+$ ]]; then
  echo "invalid ZT process group: ${zt_pgid}"
  exit 74
fi
echo "$(date --iso-8601=seconds) stopping ZT process group ${zt_pgid}"
kill -TERM -- "-${zt_pgid}"
for _ in $(seq 1 60); do
  if ! pgrep -f "scripts/train_atomic_zt_fast.py.*--run-name ${ZT_RUN}" >/dev/null; then
    break
  fi
  sleep 2
done
if pgrep -f "scripts/train_atomic_zt_fast.py.*--run-name ${ZT_RUN}" >/dev/null; then
  echo "$(date --iso-8601=seconds) escalating verified ZT process group ${zt_pgid}"
  kill -KILL -- "-${zt_pgid}"
  sleep 5
fi
if pgrep -f "scripts/train_atomic_zt_fast.py.*--run-name ${ZT_RUN}" >/dev/null; then
  echo "ZT processes remain after verified process-group stop"
  exit 75
fi

if [[ -e "${RAM}" ]]; then
  echo "refusing to overwrite RAM source: ${RAM}"
  exit 76
fi
cp -a "${FORMAL}" "${RAM}"
echo "$(date --iso-8601=seconds) running one-update paired smoke"
env \
  REPO="${RAM}" \
  RUN="${SMOKE_RUN}" \
  STEPS=1 \
  SMOKE_STEPS=1 \
  bash "${RAM}/scripts/run_target_v3_joint_paired_from_zt27k_ddp8_bs128_40k.sh" \
  >"${SMOKE_LOG}" 2>&1
echo "$(date --iso-8601=seconds) paired smoke passed"

if [[ -e "${BASE}/${JOINT_RUN}" || -f "${JOINT_PID_FILE}" ]]; then
  echo "refusing to overwrite an existing joint run or pid file"
  exit 77
fi
echo "$(date --iso-8601=seconds) launching ${JOINT_RUN}"
setsid env \
  REPO="${RAM}" \
  RUN="${JOINT_RUN}" \
  STEPS=40000 \
  SMOKE_STEPS=0 \
  bash "${RAM}/scripts/run_target_v3_joint_paired_from_zt27k_ddp8_bs128_40k.sh" \
  >"${JOINT_LOG}" 2>&1 &
joint_pid=$!
printf '%s\n' "${joint_pid}" >"${JOINT_PID_FILE}"

for _ in $(seq 1 20); do
  if ! kill -0 "${joint_pid}" 2>/dev/null; then
    echo "joint process exited during startup"
    tail -n 100 "${JOINT_LOG}"
    exit 78
  fi
  if grep -q "step=2700[1-9]" "${JOINT_LOG}"; then
    echo "$(date --iso-8601=seconds) joint training produced its first update"
    exit 0
  fi
  sleep 30
done
echo "joint process is alive but no first-step metric appeared within 10 minutes"
exit 79
