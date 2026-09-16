from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from atomic_latent_vla.atomic import AtomicSkill
from atomic_latent_vla.tcp import TCP_LOCAL_Z_OFFSET_M
from atomic_latent_vla.timebase import timestamps_in_seconds


POSE_COLUMNS = ("x", "y", "z", "rotate_x", "rotate_y", "rotate_z")
LEGACY_POSE_COLUMNS = ("x", "y", "z", "roll", "pitch", "yaw")
POSITIVE_SKILLS = (
    AtomicSkill.MOVE_X_POS,
    AtomicSkill.MOVE_Y_POS,
    AtomicSkill.MOVE_Z_POS,
    AtomicSkill.ROTATE_X_POS,
    AtomicSkill.ROTATE_Y_POS,
    AtomicSkill.ROTATE_Z_POS,
)
NEGATIVE_SKILLS = (
    AtomicSkill.MOVE_X_NEG,
    AtomicSkill.MOVE_Y_NEG,
    AtomicSkill.MOVE_Z_NEG,
    AtomicSkill.ROTATE_X_NEG,
    AtomicSkill.ROTATE_Y_NEG,
    AtomicSkill.ROTATE_Z_NEG,
)


@dataclass(frozen=True)
class GeometricLabel:
    skill: AtomicSkill | None
    confidence: float
    normalized_scores: tuple[float, ...]


@dataclass(frozen=True)
class CartesianPose:
    translation: np.ndarray
    rotation: np.ndarray


class ForwardKinematics(Protocol):
    def pose(self, q: np.ndarray) -> CartesianPose: ...


class LocalOffsetFK:
    """Apply a fixed translation in an existing FK frame's local axes."""

    def __init__(self, base: ForwardKinematics, offset_xyz_m: tuple[float, float, float]):
        self._base = base
        self._offset = np.asarray(offset_xyz_m, dtype=np.float64)
        if self._offset.shape != (3,):
            raise ValueError("offset_xyz_m must contain exactly three values")

    def pose(self, q: np.ndarray) -> CartesianPose:
        pose = self._base.pose(q)
        return CartesianPose(
            pose.translation + pose.rotation @ self._offset,
            pose.rotation,
        )


def _xyz_euler_to_rotation(
    rotate_x: float, rotate_y: float, rotate_z: float
) -> np.ndarray:
    cr, sr = math.cos(rotate_x), math.sin(rotate_x)
    cp, sp = math.cos(rotate_y), math.sin(rotate_y)
    cy, sy = math.cos(rotate_z), math.sin(rotate_z)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _rotation_log(rotation: np.ndarray) -> np.ndarray:
    """SO(3) logarithm as a rotation vector in the matrix's reference frame."""
    rotation = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    skew = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1e-7:
        return 0.5 * skew
    if math.pi - angle < 1e-5:
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        axis_index = int(np.argmin(np.abs(eigenvalues - 1.0)))
        axis = np.real(eigenvectors[:, axis_index])
        axis /= max(float(np.linalg.norm(axis)), np.finfo(np.float64).eps)
        nonzero = int(np.argmax(np.abs(axis)))
        if axis[nonzero] < 0:
            axis = -axis
        return angle * axis
    return angle / (2.0 * math.sin(angle)) * skew


def base_frame_pose_delta(start: CartesianPose, end: CartesianPose) -> np.ndarray:
    """Return ``[dx, dy, dz, rx, ry, rz]`` expressed in CR1 base axes.

    Translation is the future TCP position minus the current TCP position.
    ``R_end @ R_start.T`` is the spatial/base-frame rotation increment, which
    matches the fixed base-axis convention used by the twelve atomic labels.
    """

    translation = np.asarray(end.translation) - np.asarray(start.translation)
    rotation = _rotation_log(np.asarray(end.rotation) @ np.asarray(start.rotation).T)
    return np.concatenate([translation, rotation]).astype(np.float32, copy=False)


def tcp_twist_delta_sequence(
    current_q: np.ndarray,
    future_q: np.ndarray,
    fk: ForwardKinematics,
) -> np.ndarray:
    """FK a future joint chunk into cumulative base-frame TCP pose deltas.

    The returned row at step ``h`` is relative to the *same* current TCP pose,
    not to step ``h-1``.  It therefore describes the chunk's macroscopic
    displacement while preserving all 50 future time positions.
    """

    current_q = np.asarray(current_q, dtype=np.float64)
    future_q = np.asarray(future_q, dtype=np.float64)
    if current_q.shape != (7,):
        raise ValueError(f"current_q must have shape (7,), got {current_q.shape}")
    if future_q.ndim != 2 or future_q.shape[1] != 7:
        raise ValueError(f"future_q must have shape (H, 7), got {future_q.shape}")
    start = fk.pose(current_q)
    return np.stack([base_frame_pose_delta(start, fk.pose(q)) for q in future_q]).astype(
        np.float32, copy=False
    )


class BaseMotionTrace:
    timestamps: np.ndarray
    source: Path
    source_type: str

    def _pose_at(self, timestamp: float) -> CartesianPose:
        raise NotImplementedError

    def _delta_at(self, start_s: float, end_s: float) -> np.ndarray:
        start = self._pose_at(start_s)
        end = self._pose_at(end_s)
        return base_frame_pose_delta(start, end)

    def label_interval(
        self,
        start_s: float,
        end_s: float,
        *,
        translation_scale_m: float,
        rotation_scale_rad: float,
    ) -> GeometricLabel:
        delta = self._delta_at(start_s, end_s)
        scales = np.asarray(
            [translation_scale_m] * 3 + [rotation_scale_rad] * 3,
            dtype=np.float64,
        )
        scores = np.abs(delta) / scales
        order = np.argsort(scores)[::-1]
        top_index = int(order[0])
        top = float(scores[top_index])
        second = float(scores[int(order[1])])
        if top < 0.15:
            return GeometricLabel(None, 0.0, tuple(float(x) for x in scores))
        magnitude = float(np.clip((top - 0.15) / 0.85, 0.0, 1.0))
        dominance = float(np.clip((top - second) / max(top, 1e-8), 0.0, 1.0))
        confidence = magnitude * (0.35 + 0.65 * dominance)
        skill = (
            POSITIVE_SKILLS[top_index]
            if delta[top_index] >= 0
            else NEGATIVE_SKILLS[top_index]
        )
        return GeometricLabel(skill, confidence, tuple(float(x) for x in scores))

    def atomic_probabilities_interval(
        self,
        start_s: float,
        end_s: float,
        *,
        translation_scale_m: float,
        rotation_scale_rad: float,
    ) -> tuple[np.ndarray, float]:
        """Return signed 12-atom FK evidence and its strongest unnormalized score."""
        atomic_scores = self.atomic_scores_interval(
            start_s,
            end_s,
            translation_scale_m=translation_scale_m,
            rotation_scale_rad=rotation_scale_rad,
        )
        strongest = float(atomic_scores.max())
        total = float(atomic_scores.sum())
        probabilities = atomic_scores / total if total > 0 else atomic_scores
        return probabilities, strongest

    def atomic_scores_interval(
        self,
        start_s: float,
        end_s: float,
        *,
        translation_scale_m: float,
        rotation_scale_rad: float,
    ) -> np.ndarray:
        """Return unnormalized signed FK evidence in the fixed 12-atom order."""
        delta = self._delta_at(start_s, end_s)
        scales = np.asarray(
            [translation_scale_m] * 3 + [rotation_scale_rad] * 3,
            dtype=np.float64,
        )
        axis_scores = np.abs(delta) / scales
        atomic_scores = np.zeros(12, dtype=np.float64)
        for axis, score in enumerate(axis_scores):
            atomic_scores[2 * axis + (1 if delta[axis] < 0 else 0)] = score
        return atomic_scores

    def prompt_summary(
        self, timestamps_s: list[float], max_rows: int | None = None
    ) -> str:
        """Return FK poses aligned to the exact timestamps used by the video montages."""
        if len(timestamps_s) < 2:
            return ""
        row_count = (
            len(timestamps_s) if max_rows is None else min(max_rows, len(timestamps_s))
        )
        if row_count < 2:
            raise ValueError("max_rows must be at least 2")
        indices = np.linspace(0, len(timestamps_s) - 1, row_count).astype(int)
        chosen = [timestamps_s[index] for index in sorted(set(indices.tolist()))]
        lines = [
            "time_s,x,y,z,rotvec_x,rotvec_y,rotvec_z,"
            "dx_prev,dy_prev,dz_prev,drotate_x_prev,drotate_y_prev,drotate_z_prev"
        ]
        previous_time = chosen[0]
        for row_index, timestamp in enumerate(chosen):
            pose = self._pose_at(timestamp)
            absolute = np.concatenate([pose.translation, _rotation_log(pose.rotation)])
            delta = (
                np.zeros(6, dtype=np.float64)
                if row_index == 0
                else self._delta_at(previous_time, timestamp)
            )
            values = np.concatenate([absolute, delta])
            lines.append(
                f"{timestamp:.3f}," + ",".join(f"{value:+.5f}" for value in values)
            )
            previous_time = timestamp
        return "\n".join(lines)


@dataclass(frozen=True)
class MotionTrace(BaseMotionTrace):
    timestamps: np.ndarray
    positions: np.ndarray
    rotations: np.ndarray
    source: Path
    source_type: str = "tcp_pose"

    @classmethod
    def load(cls, path: str | Path) -> "MotionTrace":
        source = Path(path).expanduser().resolve()
        if source.suffix.lower() == ".csv":
            with source.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        elif source.suffix.lower() in {".json", ".jsonl"}:
            with source.open(encoding="utf-8") as stream:
                if source.suffix.lower() == ".jsonl":
                    rows = [json.loads(line) for line in stream if line.strip()]
                else:
                    payload = json.load(stream)
                    rows = payload["samples"] if isinstance(payload, dict) else payload
        else:
            raise ValueError("TCP trace must be .csv, .json, or .jsonl")
        if len(rows) < 2:
            raise ValueError("TCP trace requires at least two samples")

        timestamps: list[float] = []
        poses: list[list[float]] = []
        for row in rows:
            timestamps.append(float(row["timestamp"]))
            if "tcp_pose" in row:
                pose = [float(value) for value in row["tcp_pose"]]
            elif all(name in row for name in POSE_COLUMNS):
                pose = [float(row[name]) for name in POSE_COLUMNS]
            else:
                pose = [float(row[name]) for name in LEGACY_POSE_COLUMNS]
            if len(pose) != 6:
                raise ValueError(
                    "each tcp_pose must contain [x,y,z,rotate_x,rotate_y,rotate_z]"
                )
            poses.append(pose)
        time_array = np.asarray(timestamps, dtype=np.float64)
        pose_array = np.asarray(poses, dtype=np.float64)
        order = np.argsort(time_array)
        time_array = timestamps_in_seconds(time_array[order], require_strict=True)
        pose_array = pose_array[order]
        rotations = np.stack([_xyz_euler_to_rotation(*pose[3:]) for pose in pose_array])
        return cls(time_array, pose_array[:, :3], rotations, source)

    def _pose_at(self, timestamp: float) -> CartesianPose:
        # TCP JSON traces are an optional interchange format. Linear RPY
        # interpolation is used only here; the preferred joint path interpolates q.
        timestamp = float(np.clip(timestamp, self.timestamps[0], self.timestamps[-1]))
        position = np.asarray(
            [
                np.interp(timestamp, self.timestamps, self.positions[:, dim])
                for dim in range(3)
            ]
        )
        index = int(np.searchsorted(self.timestamps, timestamp, side="left"))
        index = min(max(index, 0), len(self.timestamps) - 1)
        if index == 0:
            rotation = self.rotations[index]
        else:
            left = index - 1
            right = index
            span = self.timestamps[right] - self.timestamps[left]
            weight = float((timestamp - self.timestamps[left]) / span)
            relative = self.rotations[right] @ self.rotations[left].T
            rotation = (
                _axis_angle_to_rotation(weight * _rotation_log(relative))
                @ self.rotations[left]
            )
        return CartesianPose(position, rotation)


def _axis_angle_to_rotation(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-10:
        return np.eye(3)
    axis = np.asarray(vector, dtype=np.float64) / angle
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


@dataclass(frozen=True)
class JointMotionTrace(BaseMotionTrace):
    timestamps: np.ndarray
    qpos: np.ndarray
    source: Path
    fk: ForwardKinematics
    robot_state: np.ndarray | None = None
    source_type: str = "joint_fk"

    @classmethod
    def load_hdf5(
        cls,
        path: str | Path,
        *,
        arm: str,
        urdf_path: str | Path,
        tcp_frame: str | None = None,
        qpos_path: str | None = None,
        timestamp_path: str | None = None,
        video_timestamp_path: str = "/observations/timestamps",
        mount_xyz: tuple[float, float, float] | None = None,
        fk: ForwardKinematics | None = None,
    ) -> "JointMotionTrace":
        try:
            import h5py
        except ImportError as error:  # pragma: no cover - optional runtime dependency
            raise RuntimeError(
                "joint FK input requires h5py; install the 'fk' extra"
            ) from error

        source = Path(path).expanduser().resolve()
        with h5py.File(source, "r") as episode:
            if (qpos_path is None) != (timestamp_path is None):
                raise ValueError(
                    "qpos_path and timestamp_path must be supplied together"
                )
            if qpos_path is None:
                candidates = (
                    ("/observations/qpos", "/observations/state_timestamps"),
                    ("/observations/qpos", "/observations/timestamps"),
                    ("/observations/qpos_120hz", "/observations/timestamps_120hz"),
                )
                selected = next(
                    (
                        (candidate_qpos, candidate_time)
                        for candidate_qpos, candidate_time in candidates
                        if candidate_qpos in episode and candidate_time in episode
                    ),
                    None,
                )
                if selected is None:
                    expected = ", ".join(f"({q}, {t})" for q, t in candidates)
                    raise ValueError(
                        f"HDF5 contains no supported qpos/time pair; expected {expected}"
                    )
                qpos_path, timestamp_path = selected
            elif qpos_path not in episode or timestamp_path not in episode:
                raise ValueError(
                    f"HDF5 requires {qpos_path!r} and {timestamp_path!r} datasets"
                )

            all_qpos = np.asarray(episode[qpos_path], dtype=np.float64)
            timestamps = np.asarray(episode[timestamp_path], dtype=np.float64)
            video_time_origin = (
                float(
                    np.asarray(episode[video_timestamp_path], dtype=np.float64).reshape(
                        -1
                    )[0]
                )
                if video_timestamp_path in episode
                else None
            )
        if all_qpos.ndim != 2 or all_qpos.shape[0] != timestamps.shape[0]:
            raise ValueError(
                f"qpos/timestamp shape mismatch: {all_qpos.shape} and {timestamps.shape}"
            )
        if all_qpos.shape[1] == 16:
            if arm == "left":
                qpos = all_qpos[:, 0:7]
            elif arm == "right":
                qpos = all_qpos[:, 8:15]
            else:
                raise ValueError("arm must be 'left' or 'right'")
        elif all_qpos.shape[1] == 7:
            qpos = all_qpos
        else:
            raise ValueError(
                f"expected qpos shape (T, 16) or (T, 7), got {all_qpos.shape}"
            )
        if not np.isfinite(qpos).all():
            raise ValueError("qpos contains NaN or infinity")
        if fk is not None:
            kinematics = fk
        else:
            resolved_frame = tcp_frame or f"{arm}_wrist_x_link"
            base_fk = PinocchioCR1FK(
                urdf_path=urdf_path,
                tcp_frame=resolved_frame,
                mount_xyz=mount_xyz or default_mount_xyz(arm),
            )
            kinematics = (
                base_fk
                if tcp_frame is not None
                else LocalOffsetFK(
                    base_fk, (0.0, 0.0, TCP_LOCAL_Z_OFFSET_M)
                )
            )
        return cls(
            timestamps_in_seconds(
                timestamps, origin=video_time_origin, require_strict=True
            ),
            qpos,
            source,
            kinematics,
        )

    def _pose_at(self, timestamp: float) -> CartesianPose:
        q = np.asarray(
            [
                np.interp(timestamp, self.timestamps, self.qpos[:, dim])
                for dim in range(7)
            ]
        )
        return self.fk.pose(q)

    def prompt_summary(
        self, timestamps_s: list[float], max_rows: int | None = None
    ) -> str:
        summary = super().prompt_summary(timestamps_s, max_rows=max_rows)
        if self.robot_state is None:
            return summary
        if self.robot_state.shape != (len(self.timestamps), 8):
            raise ValueError(
                f"right robot_state must have shape ({len(self.timestamps)}, 8), "
                f"got {self.robot_state.shape}"
            )
        row_count = (
            len(timestamps_s) if max_rows is None else min(max_rows, len(timestamps_s))
        )
        indices = np.linspace(0, len(timestamps_s) - 1, row_count).astype(int)
        chosen = [timestamps_s[index] for index in sorted(set(indices.tolist()))]
        lines = summary.splitlines()
        lines[
            0
        ] += ",right_q0,right_q1,right_q2,right_q3,right_q4,right_q5,right_q6,right_gripper"
        for row_index, timestamp in enumerate(chosen, start=1):
            state = np.asarray(
                [
                    np.interp(timestamp, self.timestamps, self.robot_state[:, dim])
                    for dim in range(8)
                ]
            )
            lines[row_index] += "," + ",".join(f"{value:+.5f}" for value in state)
        return "\n".join(lines)


def default_mount_xyz(arm: str) -> tuple[float, float, float]:
    if arm == "left":
        return (-0.02, 0.2225, 0.235)
    if arm == "right":
        return (-0.02, -0.2225, 0.235)
    raise ValueError("arm must be 'left' or 'right'")


class PinocchioCR1FK:
    """CR1 arm-only URDF FK, with the shoulder mount translated into body axes."""

    def __init__(
        self,
        *,
        urdf_path: str | Path,
        tcp_frame: str = "tcp",
        mount_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        try:
            import pinocchio as pin
        except ImportError as error:  # pragma: no cover - optional runtime dependency
            raise RuntimeError(
                "joint FK input requires Pinocchio; install the 'fk' extra or run in the "
                "evo_remote_lerobot environment"
            ) from error
        source = Path(urdf_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        self._pin = pin
        self._model = pin.buildModelFromUrdf(str(source))
        if self._model.nq != 7:
            raise ValueError(f"expected a 7-DoF arm URDF, got nq={self._model.nq}")
        if not self._model.existFrame(tcp_frame):
            raise ValueError(f"TCP frame {tcp_frame!r} is absent from {source}")
        self._data = self._model.createData()
        self._frame_id = self._model.getFrameId(tcp_frame)
        self._mount_xyz = np.asarray(mount_xyz, dtype=np.float64)
        if self._mount_xyz.shape != (3,):
            raise ValueError("mount_xyz must contain exactly three values")

    def pose(self, q: np.ndarray) -> CartesianPose:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (7,):
            raise ValueError(f"expected q shape (7,), got {q.shape}")
        self._pin.framesForwardKinematics(self._model, self._data, q)
        placement = self._data.oMf[self._frame_id]
        translation = (
            np.asarray(placement.translation, dtype=np.float64) + self._mount_xyz
        )
        rotation = np.asarray(placement.rotation, dtype=np.float64)
        return CartesianPose(translation.copy(), rotation.copy())
