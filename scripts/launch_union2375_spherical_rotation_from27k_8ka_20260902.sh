#!/usr/bin/env bash
set -euo pipefail

RAM_SOURCE=${RAM_SOURCE:-/dev/shm/atomic_latent_vla_run_rotloss_from27k_20260902}
PYTHON=/mnt/cunchu/zjq/pi0.5_env/.venv/bin/python
LOG_ROOT=/mnt/cunchu/zjq/atomic_pi05_logs
SOURCE_RUN=union2375_spherical_zm25k_from_afrozt25k_fsdp8_bs256_20260901
SOURCE_STEP=28000
SOURCE_CHECKPOINT=/mnt/cunchu/zjq/atomic_pi05_runs/${SOURCE_RUN}/${SOURCE_RUN}/${SOURCE_STEP}
FSDP_DEVICES=${FSDP_DEVICES:-8}
RUN_NAME=union2375_spherical_rotfree20_w005_from28k_to50k_fsdp${FSDP_DEVICES}_bs256_20260902
LOG=${LOG_ROOT}/${RUN_NAME}.log
LOCK=/mnt/cunchu/zjq/atomic_pi05_runs/.locks/${RUN_NAME}.lock

mkdir -p "$LOG_ROOT" "$(dirname "$LOCK")"
exec 9>"$LOCK"
flock -n 9 || { echo "training lock is already held: $LOCK" >&2; exit 1; }

[[ -x "$PYTHON" ]] || { echo "missing Python: $PYTHON" >&2; exit 1; }
[[ -d "$RAM_SOURCE/src" ]] || { echo "missing RAM source: $RAM_SOURCE" >&2; exit 1; }
for marker in \
    "$SOURCE_CHECKPOINT/params/_METADATA" \
    "$SOURCE_CHECKPOINT/params/manifest.ocdbt" \
    "$SOURCE_CHECKPOINT/train_state/_METADATA"; do
    [[ -f "$marker" ]] || { echo "checkpoint is not finalized: $marker" >&2; exit 1; }
done
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')" ]]; then
    echo "GPU compute process already exists; refusing to collide" >&2
    exit 1
fi

install -m 0644 \
    "$RAM_SOURCE/configs/union2375_spherical_rotation_from27k_20260902.contract.txt" \
    "${LOG_ROOT}/${RUN_NAME}.contract.txt"

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

export PYTHONPATH="$RAM_SOURCE/src:$RAM_SOURCE/vendor/pi0.5/src:$RAM_SOURCE/vendor/pi0.5/packages/openpi-client/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export JAX_COMPILATION_CACHE_DIR=/mnt/cunchu/zjq/jax_compilation_cache/atomic_pi05
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=10
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

"$PYTHON" "$RAM_SOURCE/scripts/train_atomic_pi05.py" \
    "${DATA_ARGS[@]}" "${TEACHER_ARGS[@]}" \
    --norm-assets-dir /mnt/cunchu/zjq/target \
    --norm-asset-id openpi_norm_union2375_allframes_v1 \
    --tcp-twist-norm /mnt/cunchu/zjq/target/openpi_norm_union2375_allframes_v1/tcp_twist_norm_bimanual_tcp200.json \
    --coefficient-target joint_delta \
    --base-params "$SOURCE_CHECKPOINT/params" \
    --restore-full-state "$SOURCE_CHECKPOINT" \
    --checkpoint-base-dir /mnt/cunchu/zjq/atomic_pi05_runs \
    --batch-size 256 --devices 8 --fsdp-devices "$FSDP_DEVICES" --gradient-accumulation-steps 1 \
    --max-token-len 192 --num-workers 64 --seed 42 \
    --initial-step "$SOURCE_STEP" --phase-start-step "$SOURCE_STEP" \
    --warmup-steps 1000 --peak-lr 2.5e-5 --lr-decay-steps 30000 --decay-lr 2.5e-6 \
    --zt-fraction 0 --zt-loss-weight 0 --zm-loss-weight 1 \
    --text-flow-loss-weight 0 --coefficient-loss-weight 0 --text-atomic-loss-weight 0 \
    --full-atomic-loss-weight 0.05 --codebook-loss-weight 0 --freeze-codebook \
    --subtask-ce-loss-weight 0 --atomic-composition-loss-weight 0.015 \
    --atomic-prompt-probability 0 --atomic-text-ce-probability 0 \
    --subprompt-warmup-steps 0 --subprompt-probability-after-warmup 1 \
    --atomic-composition-sidecar fk_horizon_3hz_gate_top5_stay_v2 \
    --pad-subtask-horizon --unfreeze-vision --layerwise-atomic-flow \
    --spherical-visual-latent --visual-max-update-angle-deg 45 \
    --visual-rotation-loss-weight 0.005 --visual-rotation-free-angle-deg 20 \
    --visual-rotation-loss-warmup-steps 1000 \
    --run-name "$RUN_NAME" --steps 22000 --save-interval 1000 --log-interval 100 \
    > "$LOG" 2>&1
