#!/usr/bin/env bash
set -euo pipefail

run_name=force_hidden_crossatt2_phase_from_b1_6k_plug_vase_w512_b2_s6000_20260901
run_src=/dev/shm/atomic_force_hidden_crossatt2_phase_b2_20260901
log_path=/mnt/cunchu/zjq/atomic_pi05_logs/${run_name}.log
lock_path=/mnt/cunchu/zjq/atomic_pi05_force_runs/.locks/${run_name}.lock
b1_params=/mnt/cunchu/zjq/atomic_pi05_force_runs/forceonly_zf_zmforecast_plug_vase_afro50k_w512_b1_s6000_20260831/forceonly_zf_zmforecast_plug_vase_afro50k_w512_b1_s6000_20260831/6000/params

mkdir -p "$(dirname "$lock_path")"
exec 9>"$lock_path"
flock -n 9 || { echo "run lock is already held: $run_name" >&2; exit 1; }
trap 'rm -rf -- "$run_src"' EXIT

cd "$run_src"
export PYTHONPATH="$run_src/vendor/pi0.5/src:$run_src/vendor/pi0.5/packages/openpi-client/src:$run_src/src"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python scripts/train_force_stage_b2.py \
  --dataset-root /mnt/cunchu/zjq/target/lerobot_v3_4to1_plug_force \
  --dataset-root /mnt/cunchu/zjq/target/lerobot_v3_4to1_vase_force \
  --norm-assets-dir /mnt/cunchu/zjq/target \
  --norm-asset-id openpi_norm_union2375_allframes_v1 \
  --force-norm /mnt/cunchu/zjq/force_assets/force_norm_plug_vase_shared_lr_v1_20260822/norm_stats.json \
  --coefficient-target joint_delta \
  --b1-params "$b1_params" \
  --checkpoint-base-dir /mnt/cunchu/zjq/atomic_pi05_force_runs \
  --run-name "$run_name" \
  --batch-size 256 --devices 8 --fsdp-devices 1 --num-workers 32 \
  --max-token-len 200 \
  --encoder-width 512 --encoder-depth 2 --encoder-heads 8 --encoder-mlp-dim 1024 \
  --force-latent-dim 512 --force-only-zf --future-condition-on-zm \
  --force-hidden-cross-attention --force-hidden-cross-attention-heads 2 \
  --steps 6000 --warmup-steps 300 --peak-lr 3e-5 --decay-steps 6000 --decay-lr 3e-6 \
  --flow-loss-weight 1.0 --future-force-loss-weight 0.05 \
  --delta-regularization-weight 0 \
  --force-improvement-weight 0.1 --force-improvement-margin 0.001 \
  --force-update-action-steps 10 --force-update-offsets 0,10,20,30,40 \
  --save-interval 1000 --keep-period 5000 --log-interval 10 \
  --validation-modulus 10 --validation-remainder 0 --eval-batches 2 \
  >>"$log_path" 2>&1
