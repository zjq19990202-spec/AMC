from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from atomic_latent_vla.tcp import TCP_LOCAL_Z_OFFSET_M
from atomic_latent_vla.timebase import timestamps_in_seconds

from .mirror import mirror_left_from_bimanual
from .motion import (
    ForwardKinematics,
    JointMotionTrace,
    LocalOffsetFK,
    PinocchioCR1FK,
    default_mount_xyz,
)


DEFAULT_BASE_VIDEO_KEY = "observation.images.base_0_rgb"
DEFAULT_LEFT_VIDEO_KEY = "observation.images.left_wrist_0_rgb"
DEFAULT_RIGHT_VIDEO_KEY = "observation.images.right_wrist_0_rgb"
LEFT_STATE_SLICE = slice(0, 8)
RIGHT_JOINT_SLICE = slice(8, 15)
RIGHT_STATE_SLICE = slice(8, 16)


def _read_parquet(path: Path, columns: list[str] | None = None) -> dict[str, list[Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "LeRobot episode input requires pyarrow; install the 'annotation' extra"
        ) from error
    if not path.is_file():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    available = set(parquet.schema_arrow.names)
    selected = None if columns is None else [name for name in columns if name in available]
    return parquet.read(columns=selected).to_pydict()


def _format_path(template: str, **values: Any) -> Path:
    try:
        return Path(template.format(**values))
    except KeyError as error:
        raise ValueError(f"unsupported LeRobot path placeholder {error} in {template!r}") from error


def _video_keys(info: dict[str, Any]) -> list[str]:
    return [
        key
        for key, feature in info.get("features", {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]


def _resolve_video_key(available: list[str], requested: str | None, *, view: str) -> str:
    if requested:
        if requested not in available:
            raise ValueError(
                f"LeRobot {view} video key {requested!r} is absent; available keys: {available}"
            )
        return requested
    preferred = {
        "base": DEFAULT_BASE_VIDEO_KEY,
        "left": DEFAULT_LEFT_VIDEO_KEY,
        "right": DEFAULT_RIGHT_VIDEO_KEY,
    }[view]
    if preferred in available:
        return preferred
    if view == "base":
        candidates = [key for key in available if "base" in key.lower()]
    else:
        candidates = [
            key
            for key in available
            if view in key.lower()
            and any(word in key.lower() for word in ("wrist", "hand"))
        ]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"cannot uniquely resolve the {view} video; pass --lerobot-{view}-video-key. "
        f"Available video keys: {available}"
    )


def _load_v2_episode_metadata(root: Path, episode_index: int) -> dict[str, Any]:
    metadata_path = root / "meta" / "episodes.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    with metadata_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            if int(item["episode_index"]) == episode_index:
                return item
    raise ValueError(f"episode {episode_index} is absent from {metadata_path}")


def _load_v3_episode_metadata(
    root: Path, episode_index: int, video_keys: tuple[str, str]
) -> dict[str, Any]:
    metadata_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not metadata_files:
        raise FileNotFoundError(root / "meta" / "episodes")
    columns = ["episode_index", "tasks", "data/chunk_index", "data/file_index"]
    for key in video_keys:
        prefix = f"videos/{key}"
        columns.extend(
            [
                f"{prefix}/chunk_index",
                f"{prefix}/file_index",
                f"{prefix}/from_timestamp",
                f"{prefix}/to_timestamp",
            ]
        )
    for path in metadata_files:
        table = _read_parquet(path, columns)
        indices = table.get("episode_index", [])
        for row, value in enumerate(indices):
            if int(value) == episode_index:
                return {key: values[row] for key, values in table.items()}
    raise ValueError(f"episode {episode_index} is absent from {root / 'meta' / 'episodes'}")


def _task_from_metadata(metadata: dict[str, Any]) -> str:
    tasks = metadata.get("tasks", [])
    if isinstance(tasks, str):
        return tasks
    if isinstance(tasks, (list, tuple)) and tasks:
        return str(tasks[0])
    return ""


def _as_state_matrix(values: list[Any]) -> np.ndarray:
    state = np.asarray(values, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != 16:
        raise ValueError(
            "LeRobot observation.state must have shape (T, 16) ordered as "
            "[left joints 0:7, left gripper, right joints 8:15, right gripper]"
        )
    if len(state) < 2 or not np.isfinite(state).all():
        raise ValueError("LeRobot observation.state needs at least two finite rows")
    return state


def _as_action_matrix(values: list[Any]) -> np.ndarray:
    action = np.asarray(values, dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != 16:
        raise ValueError(
            "LeRobot action must have shape (T, 16) ordered as "
            "[left joints 0:7, left gripper, right joints 8:15, right gripper]"
        )
    if not np.isfinite(action).all():
        raise ValueError("LeRobot action must contain only finite values")
    return action


@dataclass(frozen=True)
class LeRobotEpisode:
    root: Path
    episode_index: int
    task: str
    data_path: Path
    videos: tuple[Path, Path]
    video_keys: tuple[str, str]
    timestamps: np.ndarray
    right_state: np.ndarray
    right_action: np.ndarray | None
    horizontal_flip: tuple[bool, bool]
    augmentation: str

    @property
    def right_joints(self) -> np.ndarray:
        return self.right_state[:, :7]

    def make_trace(
        self,
        *,
        urdf_path: str | Path,
        tcp_frame: str | None = None,
        mount_xyz: tuple[float, float, float] | None = None,
        fk: ForwardKinematics | None = None,
    ) -> JointMotionTrace:
        if fk is not None:
            kinematics = fk
        else:
            resolved_frame = tcp_frame or "right_wrist_x_link"
            base_fk = PinocchioCR1FK(
                urdf_path=urdf_path,
                tcp_frame=resolved_frame,
                mount_xyz=mount_xyz or default_mount_xyz("right"),
            )
            kinematics = (
                base_fk
                if tcp_frame is not None
                else LocalOffsetFK(
                    base_fk, (0.0, 0.0, TCP_LOCAL_Z_OFFSET_M)
                )
            )
        return JointMotionTrace(
            timestamps=self.timestamps,
            qpos=self.right_joints,
            source=self.data_path,
            fk=kinematics,
            robot_state=self.right_state,
        )


def load_lerobot_episode(
    root: str | Path,
    episode_index: int,
    *,
    base_video_key: str | None = None,
    left_video_key: str | None = None,
    right_video_key: str | None = None,
    mirror_left_to_right: bool = False,
) -> LeRobotEpisode:
    """Resolve one LeRobot v2.x/v3.x episode for right-arm annotation.

    Normally FK consumes columns ``8:15`` and retains ``state/action[8:16]``.
    In mirror mode, ``state/action[0:8]`` are mapped into the right-arm joint
    convention and base + left-wrist video are horizontally mirrored.
    """

    root = Path(root).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"{info_path} is missing; point --lerobot-root at a complete LeRobot dataset"
        )
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    available_videos = _video_keys(info)
    base_key = _resolve_video_key(available_videos, base_video_key, view="base")
    if mirror_left_to_right:
        if right_video_key is not None:
            raise ValueError(
                "--lerobot-right-video-key cannot be used with --mirror-left-to-right"
            )
        arm_video_key = _resolve_video_key(
            available_videos, left_video_key, view="left"
        )
    else:
        if left_video_key is not None:
            raise ValueError(
                "--lerobot-left-video-key requires --mirror-left-to-right"
            )
        arm_video_key = _resolve_video_key(
            available_videos, right_video_key, view="right"
        )

    version = str(info.get("codebase_version", ""))
    is_v3 = version.startswith("v3") or (root / "meta" / "episodes").is_dir()
    episode_chunk = episode_index // int(info.get("chunks_size", 1000))
    if is_v3:
        metadata = _load_v3_episode_metadata(
            root, episode_index, (base_key, arm_video_key)
        )
        data_context = {
            "chunk_index": int(metadata["data/chunk_index"]),
            "file_index": int(metadata["data/file_index"]),
            "episode_chunk": episode_chunk,
            "episode_index": episode_index,
        }
    else:
        metadata = _load_v2_episode_metadata(root, episode_index)
        data_context = {
            "chunk_index": episode_chunk,
            "file_index": episode_index,
            "episode_chunk": episode_chunk,
            "episode_index": episode_index,
        }

    data_template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    data_path = root / _format_path(data_template, **data_context)
    data = _read_parquet(
        data_path, ["observation.state", "action", "timestamp", "episode_index"]
    )
    if "observation.state" not in data or "timestamp" not in data:
        raise ValueError(f"{data_path} lacks observation.state or timestamp")
    row_mask = np.ones(len(data["timestamp"]), dtype=bool)
    if "episode_index" in data:
        row_mask = np.asarray(data["episode_index"], dtype=np.int64) == episode_index
    state = _as_state_matrix(
        [value for value, keep in zip(data["observation.state"], row_mask, strict=True) if keep]
    )
    timestamp_values = np.asarray(data["timestamp"], dtype=np.float64)[row_mask]
    timestamps = timestamps_in_seconds(timestamp_values, require_strict=True)
    action = None
    if "action" in data:
        action = _as_action_matrix(
            [value for value, keep in zip(data["action"], row_mask, strict=True) if keep]
        )
        if len(action) != len(state):
            raise ValueError("LeRobot action and observation.state row counts differ")
    if mirror_left_to_right:
        right_state = mirror_left_from_bimanual(state)
        right_action = mirror_left_from_bimanual(action) if action is not None else None
    else:
        right_state = np.ascontiguousarray(state[:, RIGHT_STATE_SLICE])
        right_action = (
            np.ascontiguousarray(action[:, RIGHT_STATE_SLICE])
            if action is not None
            else None
        )

    video_template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    video_paths: list[Path] = []
    for key in (base_key, arm_video_key):
        if is_v3:
            prefix = f"videos/{key}"
            from_s = float(metadata.get(f"{prefix}/from_timestamp", 0.0))
            if abs(from_s) > 1e-6:
                raise ValueError(
                    f"episode {episode_index} starts at {from_s:.3f}s inside a packed video; "
                    "packed-video clipping is not supported by this annotation entry point"
                )
            context = {
                **data_context,
                "chunk_index": int(metadata[f"{prefix}/chunk_index"]),
                "file_index": int(metadata[f"{prefix}/file_index"]),
                "video_key": key,
            }
        else:
            context = {**data_context, "video_key": key}
        path = root / _format_path(video_template, **context)
        if not path.is_file():
            raise FileNotFoundError(
                f"episode {episode_index} requires {key!r}, but the video is missing: {path}"
            )
        video_paths.append(path)

    return LeRobotEpisode(
        root=root,
        episode_index=episode_index,
        task=_task_from_metadata(metadata),
        data_path=data_path,
        videos=(video_paths[0], video_paths[1]),
        video_keys=(base_key, arm_video_key),
        timestamps=timestamps,
        right_state=right_state,
        right_action=right_action,
        horizontal_flip=(mirror_left_to_right, mirror_left_to_right),
        augmentation=("mirror_left_to_right" if mirror_left_to_right else "none"),
    )
