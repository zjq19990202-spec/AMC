#!/usr/bin/env bash
set -euo pipefail

SOURCE_SNAPSHOT=${SOURCE_SNAPSHOT:-/mnt/cunchu/zjq/atomic_latent_vla_snapshots/force_spherical45h20_b2_matched50k_20260903}
RAM_SOURCE=/dev/shm/atomic_latent_vla_run_force_spherical45h20_b2_matched50k_20260903
PYTHON=/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python
ZM_RUN=union2375_afro50k_matched_spherical20_w005_from25k_to50k_fsdp8_bs256_20260902
B2_RUN=force_spherical45h20_finalzm40_from_matchedspherical50k_b2s6000_20260903
ZM_ROOT=/mnt/cunchu/zjq/atomic_pi05_runs/${ZM_RUN}/${ZM_RUN}
ZM_PARAMS=${ZM_ROOT}/50000/params
ZM_LOG=/mnt/cunchu/zjq/atomic_pi05_runs/logs/${ZM_RUN}.log
B1_PARAMS=/mnt/cunchu/zjq/atomic_pi05_force_runs/forceonly_zf_zmforecast_plug_vase_afro50k_w512_b1_s6000_20260831/forceonly_zf_zmforecast_plug_vase_afro50k_w512_b1_s6000_20260831/6000/params
CHECKPOINT_ROOT=/mnt/cunchu/zjq/atomic_pi05_force_runs
LOG_ROOT=${CHECKPOINT_ROOT}/logs
PIPELINE_LOG=${LOG_ROOT}/${B2_RUN}.pipeline.log
SMOKE_LOG=${LOG_ROOT}/${B2_RUN}_smoke1.log
FORMAL_LOG=${LOG_ROOT}/${B2_RUN}.log
CONTRACT=${LOG_ROOT}/${B2_RUN}.contract.txt
LOCK=${CHECKPOINT_ROOT}/.locks/${B2_RUN}.lock
NORM=/mnt/cunchu/zjq/target/openpi_norm_union2375_allframes_v1/norm_stats.json
FORCE_NORM=/mnt/cunchu/zjq/force_assets/force_norm_plug_vase_shared_lr_v1_20260822/norm_stats.json

cleanup() {
  status=$?
  if [[ "$RAM_SOURCE" == /dev/shm/atomic_latent_vla_run_* && -d "$RAM_SOURCE" ]]; then
    cd /
    rm -rf -- "$RAM_SOURCE"
  fi
  exit "$status"
}
trap cleanup EXIT

mkdir -p "$LOG_ROOT" "$(dirname "$LOCK")"
exec 9>"$LOCK"
flock -n 9 || { echo "run lock already held: $LOCK" >&2; exit 70; }
exec >>"$PIPELINE_LOG" 2>&1

echo "$(date --iso-8601=seconds) B2 watcher started on $(hostname)"
test -x "$PYTHON"
test -d "$SOURCE_SNAPSHOT/src"
test -f "$SOURCE_SNAPSHOT/scripts/train_force_stage_b2.py"
test -f "$NORM"
test -f "$FORCE_NORM"
test -f "$B1_PARAMS/_METADATA"
test -f "$B1_PARAMS/manifest.ocdbt"
test ! -e "$CHECKPOINT_ROOT/$B2_RUN"
test ! -e "$RAM_SOURCE"

last_notice=0
while :; do
  if [[ -f "$ZM_PARAMS/_METADATA" && -f "$ZM_PARAMS/manifest.ocdbt" ]] && \
     grep -q '\[step=50000\] CheckpointManager Save Finalize is done on all hosts' "$ZM_LOG"; then
    break
  fi
  now=$(date +%s)
  if (( now - last_notice >= 600 )); then
    latest=$(grep ' INFO step=' "$ZM_LOG" | tail -1 | sed -n 's/.*INFO step=\([0-9]\+\).*/\1/p')
    echo "$(date --iso-8601=seconds) waiting for finalized ZM 50000; latest=${latest:-unknown}"
    last_notice=$now
  fi
  sleep 30
done
echo "$(date --iso-8601=seconds) finalized ZM 50000 detected"

last_notice=0
while :; do
  mapfile -t used_mb < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '
  )
  busy=0
  for value in "${used_mb[@]}"; do
    if (( value > 1000 )); then
      busy=1
      break
    fi
  done
  (( busy == 0 )) && break
  now=$(date +%s)
  if (( now - last_notice >= 300 )); then
    echo "$(date --iso-8601=seconds) waiting for all eight GPUs; memory_mb=${used_mb[*]}"
    last_notice=$now
  fi
  sleep 30
done
echo "$(date --iso-8601=seconds) all eight GPUs are free"

cp -a "$SOURCE_SNAPSHOT" "$RAM_SOURCE"
cd "$RAM_SOURCE"
export PYTHONPATH="$RAM_SOURCE/vendor/pi0.5/src:$RAM_SOURCE/vendor/pi0.5/packages/openpi-client/src:$RAM_SOURCE/src"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

norm_sha=$(sha256sum "$NORM" | awk '{print $1}')
force_norm_sha=$(sha256sum "$FORCE_NORM" | awk '{print $1}')
cat >"$CONTRACT" <<EOF
stage=B2 force-conditioned action flow
host=$(hostname)
run=$B2_RUN
base_params=$ZM_PARAMS
force_init_params=$B1_PARAMS
merge=restore matched spherical ZM50K policy; overlay matching completed B1 force_conditioner tensors; initialize full-token adapter output at zero
datasets=/mnt/cunchu/zjq/target/lerobot_v3_4to1_plug_force,/mnt/cunchu/zjq/target/lerobot_v3_4to1_vase_force
prompt_route=current_subtask; no_global_prompt
norm=$NORM sha256=$norm_sha mode=q01_q99_no_clip adapt_to_pi=true
force_norm=$FORCE_NORM sha256=$force_norm_sha
state_clip=off action_clip=off inference_clip=not_applicable optimizer_grad_clip=1.0
zm_route=one final right/left pair reused in all 18 Action-Expert blocks; layerwise_atomic_flow=false
visual_geometry=unit-sphere zM inherited from base; visual hard cap 45deg and trained 20deg free-cone hinge
force_memory=30 SlowProj + 10 FastProj tokens per arm; query=Proj(zM)+RTC phase; 2-head cross-attention
force_geometry=direct 512D force residual projected onto final zM tangent plane; smooth 45deg cap; unit-sphere zM_force; 20deg free-cone geodesic hinge weight 0.005
trainable=FastProj + full-token force adapter + zero-init direct 512D force output + layer gates
frozen=VLM ActionExpert final-zM path Q/codebook completed B1 slow encoder/SlowProj/future decoder
rtc_offsets=0,10,20,30,40 committed_prefix_loss_weight=0 suffix_loss_weight=1
batch=256 devices=8 per_gpu=32 fsdp_devices=1 mesh=[8,1] accumulation=1
tokenizer=official_pi05_paligemma max_token_len=192
schedule=steps6000 warmup300 peak_lr=1e-4 decay_steps=6000 decay_lr=1e-5 fresh_adam=true
loss=flow1 future0 delta_regularization1e-4 improvement1 margin0.001 force_rotation_hinge0.005(free20deg,max45deg)
EOF
sha256sum \
  src/atomic_latent_vla/pi05/config.py \
  src/atomic_latent_vla/pi05/model.py \
  src/atomic_latent_vla/pi05/gemma_adapter.py \
  src/atomic_latent_vla/pi05/force.py \
  src/atomic_latent_vla/pi05/force_training_data.py \
  src/atomic_latent_vla/pi05/weights.py \
  scripts/train_force_stage_b2.py \
  vendor/pi0.5/src/openpi/training/sharding.py >>"$CONTRACT"

common_args=(
  --dataset-root /mnt/cunchu/zjq/target/lerobot_v3_4to1_plug_force
  --dataset-root /mnt/cunchu/zjq/target/lerobot_v3_4to1_vase_force
  --norm-assets-dir /mnt/cunchu/zjq/target
  --norm-asset-id openpi_norm_union2375_allframes_v1
  --force-norm "$FORCE_NORM"
  --b1-params "$ZM_PARAMS"
  --force-init-params "$B1_PARAMS"
  --checkpoint-base-dir "$CHECKPOINT_ROOT"
  --coefficient-target joint_delta
  --batch-size 256
  --devices 8
  --fsdp-devices 1
  --num-workers 32
  --max-token-len 192
  --encoder-width 512
  --encoder-depth 2
  --encoder-heads 8
  --encoder-mlp-dim 1024
  --force-latent-dim 512
  --force-only-zf
  --future-condition-on-zm
  --full-token-force-adapter
  --full-token-force-adapter-heads 2
  --no-layerwise-atomic-flow
  --spherical-visual-latent
  --visual-max-update-angle-deg 45
  --spherical-force-update
  --force-max-update-angle-deg 45
  --force-rotation-loss-weight 0.005
  --force-rotation-free-angle-deg 20
  --freeze-action-path
  --steps 6000
  --warmup-steps 300
  --peak-lr 1e-4
  --decay-steps 6000
  --decay-lr 1e-5
  --future-force-loss-weight 0
  --flow-loss-weight 1
  --delta-regularization-weight 1e-4
  --force-improvement-weight 1
  --force-improvement-margin 0.001
  --force-update-action-steps 10
  --force-update-offsets 0,10,20,30,40
  --save-interval 1000
  --keep-period 5000
  --log-interval 10
  --validation-modulus 10
  --validation-remainder 0
  --eval-batches 2
)

echo "$(date --iso-8601=seconds) launching one-step B2 smoke"
"$PYTHON" scripts/train_force_stage_b2.py \
  "${common_args[@]}" \
  --run-name "${B2_RUN}_smoke1" \
  --steps 1 \
  --log-interval 1 \
  --skip-checkpoint \
  >"$SMOKE_LOG" 2>&1
grep -q 'layerwise_atomic_flow=False' "$SMOKE_LOG"
grep -q 'step=1 ' "$SMOKE_LOG"
if grep -Eq 'Involuntary full rematerialization|RESOURCE_EXHAUSTED|CUDA_ERROR|NCCL.*(error|fail)|out of memory|Traceback' "$SMOKE_LOG"; then
  echo "B2 smoke violated error gate" >&2
  exit 72
fi
echo "$(date --iso-8601=seconds) B2 smoke passed"

echo "$(date --iso-8601=seconds) launching formal B2 6K"
"$PYTHON" scripts/train_force_stage_b2.py \
  "${common_args[@]}" \
  --run-name "$B2_RUN" \
  >"$FORMAL_LOG" 2>&1

test -f "$CHECKPOINT_ROOT/$B2_RUN/$B2_RUN/6000/params/_METADATA"
test -f "$CHECKPOINT_ROOT/$B2_RUN/$B2_RUN/6000/params/manifest.ocdbt"
echo "$(date --iso-8601=seconds) B2 6000 finalized"
