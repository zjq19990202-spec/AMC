#!/usr/bin/env bash
set -euo pipefail

SOURCE_SNAPSHOT=/mnt/cunchu/zjq/atomic_latent_vla_snapshots/force_spherical45h20_crossatt_b2_matched50k_20260904
RAM_SOURCE=/dev/shm/atomic_latent_vla_eval_force_spherical45h20_crossatt_zf_20260904
PYTHON=/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python
B2_RUN=force_spherical45h20_crossatt2_finalzm40_from_matchedspherical50k_b2s6000_20260904
ROOT=/mnt/cunchu/zjq/atomic_pi05_force_runs
CKPT=${ROOT}/${B2_RUN}/${B2_RUN}/6000
PIPELINE_LOG=${ROOT}/logs/${B2_RUN}.pipeline.log
OUT=${ROOT}/evals/${B2_RUN}/zf_attribution_20260904
LOG=${ROOT}/logs/${B2_RUN}_zf_attribution.pipeline.log
LOCK=${ROOT}/.locks/${B2_RUN}_zf_attribution.lock
NORM_ROOT=/mnt/cunchu/zjq/target
NORM_ID=openpi_norm_union2375_allframes_v1
FORCE_NORM=/mnt/cunchu/zjq/force_assets/force_norm_plug_vase_shared_lr_v1_20260822/norm_stats.json

cleanup() {
  status=$?
  if [[ "$RAM_SOURCE" == /dev/shm/atomic_latent_vla_eval_* && -d "$RAM_SOURCE" ]]; then
    cd /
    rm -rf -- "$RAM_SOURCE"
  fi
  exit "$status"
}
trap cleanup EXIT

mkdir -p "$(dirname "$LOCK")" "$(dirname "$LOG")"
exec 9>"$LOCK"
flock -n 9 || { echo "evaluation lock already held: $LOCK" >&2; exit 70; }
exec >>"$LOG" 2>&1
echo "$(date --iso-8601=seconds) waiting for finalized B2 6000"
while :; do
  if [[ -f "$CKPT/params/_METADATA" && -f "$CKPT/params/manifest.ocdbt" ]] && \
     grep -q 'B2 6000 finalized' "$PIPELINE_LOG"; then
    break
  fi
  sleep 30
done
echo "$(date --iso-8601=seconds) finalized B2 detected"

while :; do
  mapfile -t used_mb < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '1,2p' | tr -d ' '
  )
  if (( ${used_mb[0]} < 1000 && ${used_mb[1]} < 1000 )); then
    break
  fi
  sleep 30
done

test ! -e "$RAM_SOURCE"
cp -a "$SOURCE_SNAPSHOT" "$RAM_SOURCE"
mkdir -p "$OUT/plug" "$OUT/vase"
cd "$RAM_SOURCE"
export PYTHONPATH="$RAM_SOURCE/vendor/pi0.5/src:$RAM_SOURCE/vendor/pi0.5/packages/openpi-client/src:$RAM_SOURCE/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

echo "$(date --iso-8601=seconds) launching paired Plug/Vase zF attribution"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/evaluate_force_b2_spherical_zf_attribution.py \
  --checkpoint "$CKPT" \
  --dataset-root /mnt/cunchu/zjq/target/lerobot_v3_4to1_plug_force \
  --norm-assets-dir "$NORM_ROOT" \
  --norm-asset-id "$NORM_ID" \
  --force-norm "$FORCE_NORM" \
  --sample-count 64 \
  --batch-size 8 \
  --seed 20260904 \
  --output-dir "$OUT/plug" \
  >"$OUT/plug/eval.log" 2>&1 &
plug_pid=$!

CUDA_VISIBLE_DEVICES=1 "$PYTHON" scripts/evaluate_force_b2_spherical_zf_attribution.py \
  --checkpoint "$CKPT" \
  --dataset-root /mnt/cunchu/zjq/target/lerobot_v3_4to1_vase_force \
  --norm-assets-dir "$NORM_ROOT" \
  --norm-asset-id "$NORM_ID" \
  --force-norm "$FORCE_NORM" \
  --sample-count 64 \
  --batch-size 8 \
  --seed 20260905 \
  --output-dir "$OUT/vase" \
  >"$OUT/vase/eval.log" 2>&1 &
vase_pid=$!

status=0
wait "$plug_pid" || status=1
wait "$vase_pid" || status=1
if (( status != 0 )); then
  echo "zF attribution failed; inspect per-domain logs" >&2
  exit 71
fi

"$PYTHON" scripts/summarize_force_b2_zf_attribution.py \
  --input "$OUT/plug/summary.json" \
  --input "$OUT/vase/summary.json" \
  --output-dir "$OUT" \
  >"$OUT/macro.log" 2>&1
echo "$(date --iso-8601=seconds) zF attribution complete: $OUT"
