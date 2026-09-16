#!/usr/bin/env bash
set -euo pipefail

RAM_SOURCE=/dev/shm/atomic_latent_vla_run_spherical_zm25k_b2s6k_20260901
PYTHON=/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python
LOG_ROOT=/mnt/cunchu/zjq/atomic_pi05_logs
ZM_RUN=union2375_spherical_zm25k_from_afrozt25k_fsdp8_bs256_20260901
B2_RUN=force_spherical15_fulltoken40_from_union2375_spherical50k_b2s6000_20260901
ZT_PARAMS=/mnt/cunchu/zjq/atomic_pi05_runs/union2375_afro_pi05tok32_zt25k_ddp8_bs256_noacc_tok192_20260827/union2375_afro_pi05tok32_zt25k_ddp8_bs256_noacc_tok192_20260827/25000/params
ZM_PARAMS=/mnt/cunchu/zjq/atomic_pi05_runs/${ZM_RUN}/${ZM_RUN}/50000/params
B1_PARAMS=/mnt/cunchu/zjq/atomic_pi05_force_runs/forceonly_zf_zmforecast_plug_vase_afro50k_w512_b1_s6000_20260831/forceonly_zf_zmforecast_plug_vase_afro50k_w512_b1_s6000_20260831/6000/params
PIPELINE_LOG=${LOG_ROOT}/${ZM_RUN}_to_${B2_RUN}.pipeline.log
LOCK=/mnt/cunchu/zjq/atomic_pi05_force_runs/.locks/${ZM_RUN}_to_${B2_RUN}.lock

mkdir -p "$LOG_ROOT" "$(dirname "$LOCK")"
exec 9>"$LOCK"
flock -n 9 || { echo "pipeline lock is already held: $LOCK"; exit 1; }
printf '%s\n' "$$" > "${PIPELINE_LOG}.pid"
trap 'rm -rf -- /dev/shm/atomic_latent_vla_run_spherical_zm25k_b2s6k_20260901' EXIT

if [[ ! -x "$PYTHON" || ! -d "$RAM_SOURCE/src" ]]; then
    echo "missing Python or RAM source" >&2
    exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')" ]]; then
    echo "tmp8ka is no longer idle; refusing to collide with another job" >&2
    exit 1
fi

install -m 0644 "$RAM_SOURCE/configs/union2375_spherical_zm25k_from_zt25k_20260901.contract.txt" "${LOG_ROOT}/${ZM_RUN}.contract.txt"
install -m 0644 "$RAM_SOURCE/configs/force_spherical_b2s6000_from_union2375_spherical50k_20260901.contract.txt" "${LOG_ROOT}/${B2_RUN}.contract.txt"

DATA_ARGS=(
    --dataset-root /mnt/cunchu/zjq/target/lerobot_2058_domain_split_20260822/cabinet
    --dataset-root /mnt/cunchu/zjq/target/lerobot_2058_domain_split_20260822/drawer
    --dataset-root /mnt/cunchu/zjq/target/lerobot_2058_domain_split_20260822/fruit
    --dataset-root /mnt/cunchu/zjq/target/lerobot_2058_domain_split_20260822/mixed_rest
    --dataset-root /mnt/cunchu/zjq/target/lerobot_2058_screw5_subtask_atomicfix_20260825
    --dataset-root /mnt/cunchu/zjq/target/lerobot_plug_vase_split_20260822/plug
    --dataset-root /mnt/cunchu/zjq/target/lerobot_vase167_manual_atomicfix_20260825
)
TEACHER_ARGS=(
    --zt-teacher-sidecar /mnt/cunchu/zjq/atomic_pi05_teachers/union2375_afro_pi05tok32_zt25000_maskv1_20260828/cabinet
    --zt-teacher-sidecar /mnt/cunchu/zjq/atomic_pi05_teachers/union2375_afro_pi05tok32_zt25000_maskv1_20260828/drawer
    --zt-teacher-sidecar /mnt/cunchu/zjq/atomic_pi05_teachers/union2375_afro_pi05tok32_zt25000_maskv1_20260828/fruit
    --zt-teacher-sidecar /mnt/cunchu/zjq/atomic_pi05_teachers/union2375_afro_pi05tok32_zt25000_maskv1_20260828/mixed_rest
    --zt-teacher-sidecar /mnt/cunchu/zjq/atomic_pi05_teachers/union2375_afro_pi05tok32_zt25000_maskv1_20260828/screw
    --zt-teacher-sidecar /mnt/cunchu/zjq/atomic_pi05_teachers/union2375_afro_pi05tok32_zt25000_maskv1_20260828/plug
    --zt-teacher-sidecar /mnt/cunchu/zjq/atomic_pi05_teachers/union2375_afro_pi05tok32_zt25000_maskv1_20260828/vase
)
ZM_COMMON=(
    "${DATA_ARGS[@]}" "${TEACHER_ARGS[@]}"
    --norm-assets-dir /mnt/cunchu/zjq/target
    --norm-asset-id openpi_norm_union2375_allframes_v1
    --tcp-twist-norm /mnt/cunchu/zjq/target/openpi_norm_union2375_allframes_v1/tcp_twist_norm_bimanual_tcp200.json
    --coefficient-target joint_delta
    --base-params "$ZT_PARAMS"
    --checkpoint-base-dir /mnt/cunchu/zjq/atomic_pi05_runs
    --batch-size 256 --devices 8 --fsdp-devices 8 --gradient-accumulation-steps 1
    --max-token-len 192 --num-workers 64 --seed 42
    --initial-step 25000 --phase-start-step 25000
    --warmup-steps 1000 --peak-lr 2.5e-5 --lr-decay-steps 30000 --decay-lr 2.5e-6
    --zt-fraction 0 --zt-loss-weight 0 --zm-loss-weight 1
    --text-flow-loss-weight 0 --coefficient-loss-weight 0 --text-atomic-loss-weight 0
    --full-atomic-loss-weight 0.05 --codebook-loss-weight 0 --freeze-codebook
    --subtask-ce-loss-weight 0 --atomic-composition-loss-weight 0.015
    --atomic-prompt-probability 0 --atomic-text-ce-probability 0
    --subprompt-warmup-steps 0 --subprompt-probability-after-warmup 1
    --atomic-composition-sidecar fk_horizon_3hz_gate_top5_stay_v2
    --pad-subtask-horizon --unfreeze-vision --layerwise-atomic-flow
    --spherical-visual-latent --visual-max-update-angle-deg 45
)

export PYTHONPATH="$RAM_SOURCE/src:$RAM_SOURCE/vendor/pi0.5/src:$RAM_SOURCE/vendor/pi0.5/packages/openpi-client/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "$(date --iso-8601=seconds) starting 8-GPU spherical ZM smoke" | tee -a "$PIPELINE_LOG"
"$PYTHON" "$RAM_SOURCE/scripts/train_atomic_pi05.py" \
    "${ZM_COMMON[@]}" --run-name "${ZM_RUN}_smoke1" --steps 25000 \
    --smoke-steps 1 --skip-checkpoint --log-interval 1 \
    > "${LOG_ROOT}/${ZM_RUN}.smoke1.log" 2>&1
echo "$(date --iso-8601=seconds) spherical ZM smoke passed" | tee -a "$PIPELINE_LOG"

echo "$(date --iso-8601=seconds) starting formal spherical ZM 25K" | tee -a "$PIPELINE_LOG"
"$PYTHON" "$RAM_SOURCE/scripts/train_atomic_pi05.py" \
    "${ZM_COMMON[@]}" --run-name "$ZM_RUN" --steps 25000 \
    --save-interval 1000 --log-interval 100 \
    > "${LOG_ROOT}/${ZM_RUN}.log" 2>&1

for marker in "$ZM_PARAMS/_METADATA" "$ZM_PARAMS/manifest.ocdbt"; do
    [[ -f "$marker" ]] || { echo "missing finalized ZM marker: $marker" >&2; exit 1; }
done
echo "$(date --iso-8601=seconds) ZM step50000 finalized; starting spherical B2 6K" | tee -a "$PIPELINE_LOG"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
"$PYTHON" "$RAM_SOURCE/scripts/train_force_stage_b2.py" \
    --dataset-root /mnt/cunchu/zjq/target/lerobot_v3_4to1_plug_force \
    --dataset-root /mnt/cunchu/zjq/target/lerobot_v3_4to1_vase_force \
    --norm-assets-dir /mnt/cunchu/zjq/target \
    --norm-asset-id openpi_norm_union2375_allframes_v1 \
    --force-norm /mnt/cunchu/zjq/force_assets/force_norm_plug_vase_shared_lr_v1_20260822/norm_stats.json \
    --b1-params "$ZM_PARAMS" --force-init-params "$B1_PARAMS" \
    --checkpoint-base-dir /mnt/cunchu/zjq/atomic_pi05_force_runs --run-name "$B2_RUN" \
    --coefficient-target joint_delta --batch-size 256 --devices 8 --fsdp-devices 1 \
    --num-workers 32 --max-token-len 200 \
    --encoder-width 512 --encoder-depth 2 --encoder-heads 8 \
    --encoder-mlp-dim 1024 --force-latent-dim 512 \
    --force-only-zf --future-condition-on-zm \
    --steps 6000 --warmup-steps 300 --peak-lr 1e-4 --decay-steps 6000 --decay-lr 1e-5 \
    --future-force-loss-weight 0 --flow-loss-weight 1 --delta-regularization-weight 1e-4 \
    --force-improvement-weight 1 --force-improvement-margin 0.001 \
    --full-token-force-adapter --full-token-force-adapter-heads 2 \
    --spherical-visual-latent --visual-max-update-angle-deg 45 \
    --spherical-force-update --force-max-update-angle-deg 15 \
    --force-update-action-steps 10 --force-update-offsets 0,10,20,30,40 \
    --freeze-action-path --save-interval 1000 --keep-period 5000 --log-interval 10 \
    --validation-modulus 10 --validation-remainder 0 --eval-batches 2 \
    > "${LOG_ROOT}/${B2_RUN}.log" 2>&1

echo "$(date --iso-8601=seconds) spherical B2 step6000 finalized" | tee -a "$PIPELINE_LOG"
