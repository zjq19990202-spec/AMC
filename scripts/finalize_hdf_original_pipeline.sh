#!/usr/bin/env bash
# Finish HDF conversion only after the existing converter has every expected
# three-camera episode. A count mismatch stops instead of creating partial data.
set -euo pipefail

PID="${1:?converter PID required}"
PY="/home/admin123/anaconda3/envs/lingbotvla2_local/bin/python"
ROOT="/media/admin123/T5 EVO/original_atomic_lerobot"
NATIVE="$ROOT/record_data_gap_split_native_v2"
MIRROR="$ROOT/record_data_gap_split_mirror_v2"
RIGHT="/media/admin123/T5 EVO/record_data_gap_split_atomic_qwen/annotations/right"
LEFT="/media/admin123/T5 EVO/record_data_gap_split_atomic_qwen/annotations/left_mirror"
SCRIPTS="/home/admin123/zjq/atomic_latent_vla/scripts"
EXPECTED_EPISODES=204
EXPECTED_VIDEOS=$((EXPECTED_EPISODES * 3))

while kill -0 "$PID" 2>/dev/null; do
    videos=$(find "$NATIVE/videos" -type f -name '*.mp4' 2>/dev/null | wc -l)
    echo "$(date -Is) native converter still running: $videos/$EXPECTED_VIDEOS videos"
    sleep 30
done

videos=$(find "$NATIVE/videos" -type f -name '*.mp4' 2>/dev/null | wc -l)
if [[ "$videos" -ne "$EXPECTED_VIDEOS" || ! -s "$NATIVE/source_episode_map.json" ]]; then
    echo "$(date -Is) ERROR: converter ended but native is incomplete ($videos/$EXPECTED_VIDEOS videos or map missing)" >&2
    exit 1
fi

echo "$(date -Is) injecting right-arm annotations"
"$PY" "$SCRIPTS/inject_atomic_annotations_lerobot.py" --dataset "$NATIVE" --annotations "$RIGHT"
"$PY" "$SCRIPTS/validate_original_atomic_lerobot.py" "$NATIVE"

echo "$(date -Is) creating full three-camera left/right mirror"
"$PY" "$SCRIPTS/mirror_original_lerobot.py" --source "$NATIVE" --output "$MIRROR"

echo "$(date -Is) injecting mirrored-left annotations"
"$PY" "$SCRIPTS/inject_atomic_annotations_lerobot.py" --dataset "$MIRROR" --annotations "$LEFT"
"$PY" "$SCRIPTS/validate_original_atomic_lerobot.py" "$MIRROR"
"$PY" "$SCRIPTS/verify_bimanual_mirror.py" --source "$NATIVE" --mirror "$MIRROR"

echo "$(date -Is) COMPLETE: HDF native and mirror passed full validation"
