#!/usr/bin/env bash
set -euo pipefail

SOURCE=/mnt/cunchu/zjq/atomic_latent_vla_snapshots/force_spherical45h20_crossatt_b2_matched50k_20260904
RAM_SOURCE=/dev/shm/atomic_latent_vla_eval_b2_6k_vs_afro50k_plug_ep0_20260904
PYTHON=/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python
RUN=force_spherical45h20_crossatt2_finalzm40_from_matchedspherical50k_b2s6000_20260904
B2=/mnt/cunchu/zjq/atomic_pi05_force_runs/${RUN}/${RUN}/6000
BASE=/mnt/cunchu/zjq/atomic_pi05_runs/union2375_afro50k_matched_spherical20_w005_from25k_to50k_fsdp8_bs256_20260902/union2375_afro50k_matched_spherical20_w005_from25k_to50k_fsdp8_bs256_20260902/50000
DATA=/mnt/cunchu/zjq/target/lerobot_v3_4to1_plug_force
NORM_ROOT=/mnt/cunchu/zjq/target
NORM_ID=openpi_norm_union2375_allframes_v1
FORCE_NORM=/mnt/cunchu/zjq/force_assets/force_norm_plug_vase_shared_lr_v1_20260822/norm_stats.json
OUT=/mnt/cunchu/zjq/atomic_pi05_force_runs/evals/${RUN}/plug_episode0_b2_6k_vs_afro50k_seed20260824
LOG=/mnt/cunchu/zjq/atomic_pi05_force_runs/logs/${RUN}_plug_episode0_vs_afro50k.log
LOCK=/mnt/cunchu/zjq/atomic_pi05_force_runs/.locks/${RUN}_plug_episode0_vs_afro50k.lock

cleanup() {
  status=$?
  if [[ "$RAM_SOURCE" == /dev/shm/atomic_latent_vla_eval_* && -d "$RAM_SOURCE" ]]; then
    cd /
    rm -rf -- "$RAM_SOURCE"
  fi
  exit "$status"
}
trap cleanup EXIT

test -f "$B2/params/_METADATA"
test -f "$BASE/params/_METADATA"
test ! -e "$RAM_SOURCE"
mkdir -p "$(dirname "$LOCK")" "$OUT"
exec 9>"$LOCK"
flock -n 9 || { echo "evaluation lock already held: $LOCK" >&2; exit 70; }
cp -a "$SOURCE" "$RAM_SOURCE"
cd "$RAM_SOURCE"
export PYTHONPATH="$RAM_SOURCE/vendor/pi0.5/src:$RAM_SOURCE/vendor/pi0.5/packages/openpi-client/src:$RAM_SOURCE/src"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

"$PYTHON" scripts/evaluate_force_b2_episode50_ablation.py \
  --checkpoint "$B2" \
  --base-checkpoint "$BASE" \
  --dataset-root "$DATA" \
  --norm-assets-dir "$NORM_ROOT" \
  --norm-asset-id "$NORM_ID" \
  --force-norm "$FORCE_NORM" \
  --episode 0 \
  --batch-size 4 \
  --seed 20260824 \
  --encoder-width 512 \
  --encoder-heads 8 \
  --encoder-mlp-dim 1024 \
  --force-latent-dim 512 \
  --full-token-force-adapter \
  --full-token-force-adapter-heads 2 \
  --output-dir "$OUT" \
  >"$LOG" 2>&1

test -f "$OUT/summary.json"
echo "evaluation complete: $OUT"
