from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# Verified against CR1ARML/CR1ARMR FK.  Joint order is
# [shoulder_y, shoulder_x, shoulder_z, elbow, wrist_z, wrist_y, wrist_x].
CR1_LEFT_TO_RIGHT_JOINT_SIGN = np.asarray(
    [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float64
)
CR1_LEFT_TO_RIGHT_STATE_SIGN = np.concatenate(
    [CR1_LEFT_TO_RIGHT_JOINT_SIGN, np.ones(1, dtype=np.float64)]
)


def mirror_left_arm_values(values: np.ndarray) -> np.ndarray:
    """Map left-arm joints or joints+gripper into the right-arm convention."""

    array = np.asarray(values)
    if array.ndim < 1 or array.shape[-1] not in (7, 8):
        raise ValueError(
            f"left-arm values must end in 7 joints or 8 joints+gripper, got {array.shape}"
        )
    sign = (
        CR1_LEFT_TO_RIGHT_JOINT_SIGN
        if array.shape[-1] == 7
        else CR1_LEFT_TO_RIGHT_STATE_SIGN
    )
    return np.ascontiguousarray(array * sign.astype(array.dtype, copy=False))


def mirror_left_from_bimanual(values: np.ndarray) -> np.ndarray:
    """Extract ``[..., 0:8]`` from a CR1 bimanual vector and mirror it to right."""

    array = np.asarray(values)
    if array.ndim < 1 or array.shape[-1] != 16:
        raise ValueError(
            "CR1 bimanual values must end in 16 dimensions ordered as "
            "[left joints, left gripper, right joints, right gripper]"
        )
    return mirror_left_arm_values(array[..., :8])


def flip_frame_horizontal(frame: np.ndarray) -> np.ndarray:
    if frame.ndim not in (2, 3):
        raise ValueError(f"video frame must have 2 or 3 dimensions, got {frame.shape}")
    return cv2.flip(frame, 1)


def _write_flipped_video(source: Path, destination: Path, *, overwrite: bool) -> int:
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"mirrored video already exists: {destination}; pass --overwrite-mirror"
        )
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"cannot open source video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"invalid video metadata: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot create mirrored video: {temporary}")

    written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(flip_frame_horizontal(frame))
            written += 1
    finally:
        capture.release()
        writer.release()
    if written == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"source video contains no decodable frames: {source}")
    temporary.replace(destination)
    return written


@dataclass(frozen=True)
class MirroredEpisodeBundle:
    root: Path
    base_video: Path
    right_wrist_video: Path
    trajectory: Path
    metadata: Path

    @property
    def videos(self) -> tuple[Path, Path]:
        return self.base_video, self.right_wrist_video


def write_mirrored_episode_bundle(
    *,
    output_dir: str | Path,
    source_root: str | Path,
    source_episode_index: int,
    task: str,
    source_base_video: str | Path,
    source_left_wrist_video: str | Path,
    timestamps: np.ndarray,
    right_state: np.ndarray,
    right_action: np.ndarray | None,
    overwrite: bool = False,
) -> MirroredEpisodeBundle:
    """Materialize an auditable left-to-right mirror bundle before annotation."""

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    base_video = root / "base.mp4"
    right_wrist_video = root / "right_wrist.mp4"
    trajectory = root / "trajectory.npz"
    metadata = root / "metadata.json"
    existing = [
        path
        for path in (base_video, right_wrist_video, trajectory, metadata)
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"mirrored bundle files already exist: {existing}; pass --overwrite-mirror"
        )

    base_frames = _write_flipped_video(
        Path(source_base_video).expanduser().resolve(), base_video, overwrite=overwrite
    )
    wrist_frames = _write_flipped_video(
        Path(source_left_wrist_video).expanduser().resolve(),
        right_wrist_video,
        overwrite=overwrite,
    )
    arrays = {
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "right_state": np.asarray(right_state, dtype=np.float32),
    }
    if right_action is not None:
        arrays["right_action"] = np.asarray(right_action, dtype=np.float32)
    np.savez_compressed(trajectory, **arrays)
    metadata.write_text(
        json.dumps(
            {
                "augmentation": "mirror_left_to_right",
                "source_root": str(Path(source_root).expanduser().resolve()),
                "source_episode_index": source_episode_index,
                "task": task,
                "joint_sign": CR1_LEFT_TO_RIGHT_JOINT_SIGN.astype(int).tolist(),
                "base_frames": base_frames,
                "right_wrist_frames": wrist_frames,
                "trajectory_rows": int(len(timestamps)),
                "has_action": right_action is not None,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return MirroredEpisodeBundle(
        root=root,
        base_video=base_video,
        right_wrist_video=right_wrist_video,
        trajectory=trajectory,
        metadata=metadata,
    )
