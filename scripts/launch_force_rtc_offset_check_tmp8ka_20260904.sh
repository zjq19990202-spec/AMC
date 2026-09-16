#!/usr/bin/env bash
set -euo pipefail

PYTHON=/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python
SOURCE=/mnt/cunchu/zjq/atomic_latent_vla_snapshots/force_spherical_b2_subtaskpad_matched50k_20260904
RAM_SOURCE=/dev/shm/atomic_latent_vla_eval_force_rtc_offsets_20260904
ROOT=/mnt/cunchu/zjq/atomic_pi05_force_runs
B2_RUN=force_spherical45h20_crossatt2_finalzm40_subtask_from_matchedspherical50k_b2s6000_20260904
B2=${ROOT}/${B2_RUN}/${B2_RUN}/6000
BASE_RUN=union2375_afro50k_matched_spherical20_w005_from25k_to50k_fsdp8_bs256_20260902
BASE=/mnt/cunchu/zjq/atomic_pi05_runs/${BASE_RUN}/${BASE_RUN}/50000
OUT=${ROOT}/evals/${B2_RUN}/correct_subtask_unpadded_rtc_offsets_seed20260824
NORM_ROOT=/mnt/cunchu/zjq/target
NORM_ID=openpi_norm_union2375_allframes_v1
FORCE_NORM=/mnt/cunchu/zjq/force_assets/force_norm_plug_vase_shared_lr_v1_20260822/norm_stats.json
PLUG_ROOT=/mnt/cunchu/zjq/target/lerobot_v3_4to1_plug_force
VASE_ROOT=/mnt/cunchu/zjq/target/lerobot_v3_4to1_vase_force
PLUG_SUBTASK=/mnt/cunchu/zjq/force_assets/subtask_sidecars_plug_vase_20260904/plug_episode_subtasks.jsonl
VASE_SUBTASK=/mnt/cunchu/zjq/force_assets/subtask_sidecars_plug_vase_20260904/vase_episode_subtasks.jsonl

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
rm -rf -- "$RAM_SOURCE"
cp -a "$SOURCE" "$RAM_SOURCE"
cd "$RAM_SOURCE"
export PYTHONPATH="$RAM_SOURCE/vendor/pi0.5/src:$RAM_SOURCE/vendor/pi0.5/packages/openpi-client/src:$RAM_SOURCE/src"
mkdir -p "$OUT"

run_one() {
  local gpu=$1 kind=$2 checkpoint=$3 domain=$4 data=$5 sidecar=$6
  local target=${OUT}/${kind}_${domain}_episode0_first16
  mkdir -p "$target"
  CUDA_VISIBLE_DEVICES="$gpu" \
  XLA_PYTHON_CLIENT_PREALLOCATE=true \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  "$PYTHON" scripts/evaluate_force_rtc_offsets.py \
    --model-kind "$kind" \
    --checkpoint "$checkpoint" \
    --dataset-root "$data" \
    --subtask-sidecar "$sidecar" \
    --norm-assets-dir "$NORM_ROOT" \
    --norm-asset-id "$NORM_ID" \
    --force-norm "$FORCE_NORM" \
    --episode 0 \
    --batch-size 4 \
    --max-windows 16 \
    --seed 20260824 \
    --encoder-width 512 \
    --encoder-heads 8 \
    --encoder-mlp-dim 1024 \
    --force-latent-dim 512 \
    --full-token-force-adapter \
    --full-token-force-adapter-heads 2 \
    --output-dir "$target" \
    >"${target}.log" 2>&1
}

run_one 0 b2 "$B2" plug "$PLUG_ROOT" "$PLUG_SUBTASK" & p0=$!
run_one 1 base25k "$BASE" plug "$PLUG_ROOT" "$PLUG_SUBTASK" & p1=$!
run_one 2 b2 "$B2" vase "$VASE_ROOT" "$VASE_SUBTASK" & p2=$!
run_one 3 base25k "$BASE" vase "$VASE_ROOT" "$VASE_SUBTASK" & p3=$!
wait "$p0" "$p1" "$p2" "$p3"

"$PYTHON" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = {}
for path in sorted(root.glob("*_episode0_first16/summary.json")):
    doc = json.loads(path.read_text())
    rows[path.parent.name] = doc["offset_metrics"]
(root / "combined.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
PY
