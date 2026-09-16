#!/usr/bin/env python3
"""Overlay paired fruit-prompt TCP trajectories on the initial posed CR1 URDF."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from yourdfpy import URDF


LEFT_JOINTS = (
    "left_shoulder_y_joint",
    "left_shoulder_x_joint",
    "left_shoulder_z_joint",
    "left_elbow_joint",
    "left_wrist_z_joint",
    "left_wrist_y_joint",
    "left_wrist_x_joint",
)
RIGHT_JOINTS = tuple(name.replace("left_", "right_", 1) for name in LEFT_JOINTS)
TCP_FRAMES = {"left": "left_atomic_tcp", "right": "right_atomic_tcp"}
PROMPT_COLORS = {
    "carrot": "#f97316",
    "orange": "#facc15",
    "red apple": "#dc2626",
    "green radish": "#16a34a",
    "red bell pepper": "#be123c",
    "red chili pepper": "#7f1d1d",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-npz", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--active-arm",
        choices=("both", "auto", "left", "right"),
        default="both",
        help="Both plots both TCPs; auto selects the longer recorded-GT TCP path.",
    )
    parser.add_argument("--robot-face-count", type=int, default=12_000)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configuration(state16: np.ndarray) -> dict[str, float]:
    state = np.asarray(state16, dtype=np.float64)
    if state.shape != (16,):
        raise ValueError(f"expected a 16-D state, got {state.shape}")
    q14 = np.concatenate((state[:7], state[8:15]))
    cfg = {
        name: float(value)
        for name, value in zip(LEFT_JOINTS + RIGHT_JOINTS, q14, strict=True)
    }
    cfg.update({"waist_z_joint": 0.0, "waist_x_joint": 0.0, "waist_y_joint": 0.0})
    return cfg


def _tcp_positions(robot: URDF, state_or_actions: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(state_or_actions, dtype=np.float64)
    if values.ndim == 1:
        values = values[None]
    if values.ndim != 2 or values.shape[1] != 16:
        raise ValueError(f"expected [N,16] joint values, got {values.shape}")
    result = {"left": [], "right": []}
    for row in values:
        robot.update_cfg(_configuration(row))
        for arm, frame in TCP_FRAMES.items():
            result[arm].append(robot.get_transform(frame)[:3, 3].copy())
    return {arm: np.asarray(points) for arm, points in result.items()}


def _robot_triangles(robot: URDF, state16: np.ndarray, face_count: int) -> np.ndarray:
    robot.update_cfg(_configuration(state16))
    mesh = robot.scene.dump(concatenate=True)
    faces = mesh.faces
    if len(faces) > face_count:
        stride = int(math.ceil(len(faces) / face_count))
        faces = faces[::stride]
    return np.asarray(mesh.vertices[faces], dtype=np.float64)


def _robot_skeleton(robot: URDF, state16: np.ndarray) -> np.ndarray:
    robot.update_cfg(_configuration(state16))
    joint_names = (
        ("waist_z_joint", "waist_x_joint", "waist_y_joint")
        + LEFT_JOINTS
        + RIGHT_JOINTS
        + ("left_atomic_tcp_joint", "right_atomic_tcp_joint")
    )
    segments = []
    for name in joint_names:
        joint = robot.joint_map[name]
        parent = robot.get_transform(joint.parent)[:3, 3]
        child = robot.get_transform(joint.child)[:3, 3]
        segments.append(np.stack((parent, child)))
    return np.asarray(segments)


def _path_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _equal_axes(ax: Any, points: np.ndarray) -> None:
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = (lower + upper) / 2
    radius = max(float(np.max(upper - lower)) / 2 * 1.04, 0.15)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _draw_panel(
    ax: Any,
    triangles: np.ndarray,
    skeleton: np.ndarray,
    initial_tcp: dict[str, np.ndarray],
    gt_tracks: dict[str, np.ndarray],
    prediction_tracks: dict[str, dict[str, np.ndarray]],
    active_arm: str,
    frame: int,
    original_prompt: str,
) -> None:
    ax.add_collection3d(
        Poly3DCollection(
            triangles,
            facecolor="#6b7280",
            edgecolor="none",
            alpha=0.20,
            zorder=1,
        )
    )
    for segment in skeleton:
        ax.plot(*segment.T, color="#374151", lw=3.2, alpha=0.92, zorder=4)
    joints = np.unique(skeleton.reshape(-1, 3), axis=0)
    ax.scatter(*joints.T, color="#4b5563", s=11, alpha=0.90, depthshade=False, zorder=4)
    display_arms = ("left", "right") if active_arm == "both" else (active_arm,)
    for arm, color in (("left", "#2563eb"), ("right", "#f97316")):
        marker = "*" if arm in display_arms else "o"
        alpha = 1.0 if arm in display_arms else 0.45
        ax.scatter(
            *initial_tcp[arm], color=color, marker=marker, s=75 if arm in display_arms else 28,
            alpha=alpha, depthshade=False, zorder=5,
        )

    for arm in display_arms:
        arm_style = ":" if arm == "left" else "-"
        gt_style = ":" if arm == "left" else "--"
        gt = gt_tracks[arm]
        ax.plot(*gt.T, color="#111827", lw=2.5, ls=gt_style, zorder=6)
        ax.scatter(*gt[-1], color="#111827", s=24, zorder=7)
        for name, tracks in prediction_tracks.items():
            points = tracks[arm]
            color = PROMPT_COLORS.get(name, "#7c3aed")
            ax.plot(*points.T, color=color, lw=1.9, ls=arm_style, zorder=6)
            ax.scatter(*points[-1], color=color, s=22, zorder=7)

    all_points = np.vstack(
        [triangles.reshape(-1, 3), skeleton.reshape(-1, 3)]
        + [gt_tracks[arm] for arm in display_arms]
        + [tracks[arm] for tracks in prediction_tracks.values() for arm in display_arms]
    )
    _equal_axes(ax, all_points)
    ax.set_xlabel("X / forward (m)", labelpad=3)
    ax.set_ylabel("Y / left (m)", labelpad=3)
    ax.set_zlabel("Z / up (m)", labelpad=3)
    ax.view_init(elev=24, azim=-58)
    ax.set_title(
        f"frame {frame} · {'both TCPs' if active_arm == 'both' else active_arm + ' TCP'}\n"
        f"{original_prompt}",
        fontsize=9,
    )
    ax.grid(True, alpha=0.18)


def main() -> None:
    args = _arguments()
    trajectory_path = args.trajectory_npz.expanduser().resolve()
    urdf_path = args.urdf.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    archive = np.load(trajectory_path, allow_pickle=False)
    episode = int(archive["episode"])
    frames = [int(value) for value in archive["frames"]]
    robot = URDF.load(str(urdf_path), build_scene_graph=True, load_meshes=True)
    required = set(LEFT_JOINTS + RIGHT_JOINTS + tuple(TCP_FRAMES.values()))
    missing = sorted(required - (set(robot.joint_map) | set(robot.link_map)))
    if missing:
        raise ValueError(f"URDF is missing required joints/links: {missing}")

    records: list[dict[str, Any]] = []
    prepared: list[dict[str, Any]] = []
    all_prompt_names: list[str] = []
    for frame in frames:
        prefix = f"frame_{frame:06d}"
        raw_state = np.asarray(archive[f"{prefix}_raw_state"], dtype=np.float64)
        gt_actions = np.asarray(archive[f"{prefix}_gt_actions"], dtype=np.float64)
        names = [str(value) for value in archive[f"{prefix}_prompt_names"]]
        texts = [str(value) for value in archive[f"{prefix}_prompt_texts"]]
        predictions = np.asarray(
            archive[f"{prefix}_prediction_actions"], dtype=np.float64
        )
        original_prompt = str(archive[f"{prefix}_original_subtask_prompt"])
        for name in names:
            if name not in all_prompt_names:
                all_prompt_names.append(name)

        initial = {arm: points[0] for arm, points in _tcp_positions(robot, raw_state).items()}
        gt_future = _tcp_positions(robot, gt_actions)
        gt_tracks = {
            arm: np.vstack((initial[arm], gt_future[arm])) for arm in TCP_FRAMES
        }
        prediction_tracks: dict[str, dict[str, np.ndarray]] = {}
        for name, actions in zip(names, predictions, strict=True):
            future = _tcp_positions(robot, actions)
            prediction_tracks[name] = {
                arm: np.vstack((initial[arm], future[arm])) for arm in TCP_FRAMES
            }
        path_lengths = {arm: _path_length(points) for arm, points in gt_tracks.items()}
        gt_active_arm = max(path_lengths, key=path_lengths.__getitem__)
        active_arm = gt_active_arm if args.active_arm == "auto" else args.active_arm
        triangles = _robot_triangles(robot, raw_state, args.robot_face_count)
        skeleton = _robot_skeleton(robot, raw_state)
        prepared.append(
            {
                "frame": frame,
                "raw_state": raw_state,
                "original_prompt": original_prompt,
                "initial": initial,
                "gt_tracks": gt_tracks,
                "prediction_tracks": prediction_tracks,
                "triangles": triangles,
                "skeleton": skeleton,
                "active_arm": active_arm,
                "gt_active_arm": gt_active_arm,
            }
        )
        records.append(
            {
                "frame": frame,
                "original_subtask_prompt": original_prompt,
                "prompts": dict(zip(names, texts, strict=True)),
                "active_arm": active_arm,
                "gt_active_arm": gt_active_arm,
                "gt_tcp_path_length_mm": {
                    arm: value * 1000 for arm, value in path_lengths.items()
                },
                "prediction_tcp_path_length_mm": {
                    name: {
                        arm: _path_length(tracks[arm]) * 1000
                        for arm in TCP_FRAMES
                    }
                    for name, tracks in prediction_tracks.items()
                },
                "prediction_tcp_endpoint_displacement_mm": {
                    name: {
                        arm: float(np.linalg.norm(tracks[arm][-1] - tracks[arm][0]) * 1000)
                        for arm in TCP_FRAMES
                    }
                    for name, tracks in prediction_tracks.items()
                },
                "initial_state_16d": raw_state.tolist(),
            }
        )

    rows = int(math.ceil(len(prepared) / 3))
    figure = plt.figure(figsize=(20, 7.2 * rows), constrained_layout=True)
    for index, row in enumerate(prepared):
        ax = figure.add_subplot(rows, 3, index + 1, projection="3d")
        _draw_panel(
            ax,
            row["triangles"],
            row["skeleton"],
            row["initial"],
            row["gt_tracks"],
            row["prediction_tracks"],
            row["active_arm"],
            row["frame"],
            row["original_prompt"],
        )
    handles = [
        Line2D([0], [0], color="#9ca3af", lw=7, alpha=0.35, label="CR1 initial URDF pose"),
        Line2D([0], [0], color="#111827", lw=2.5, ls="--", label="recorded GT"),
        Line2D([0], [0], color="#374151", lw=2.2, ls="-", label="right TCP (solid)"),
        Line2D([0], [0], color="#374151", lw=2.2, ls=":", label="left TCP (dotted)"),
    ] + [
        Line2D([0], [0], color=PROMPT_COLORS.get(name, "#7c3aed"), lw=2, label=name)
        for name in all_prompt_names
    ]
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=min(len(handles), 8),
        frameon=False,
    )
    subtitle = (
        "Image, state, and flow noise fixed within each frame; both left and right TCPs plotted"
        if args.active_arm == "both"
        else "Image, state, and flow noise fixed within each frame; arm selected from recorded GT motion"
    )
    figure.suptitle(
        f"Fruit episode {episode}: initial CR1 URDF pose + paired SUBtask-prompt TCP trajectories\n"
        f"{subtitle}",
        fontsize=16,
        fontweight="bold",
    )
    overview = output_dir / "fruit_prompt_tcp_trajectories_on_initial_cr1.png"
    figure.savefig(overview, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    for row in prepared:
        figure = plt.figure(figsize=(11.5, 9.5), constrained_layout=True)
        ax = figure.add_subplot(111, projection="3d")
        _draw_panel(
            ax,
            row["triangles"],
            row["skeleton"],
            row["initial"],
            row["gt_tracks"],
            row["prediction_tracks"],
            row["active_arm"],
            row["frame"],
            row["original_prompt"],
        )
        ax.legend(handles=handles, loc="upper left", framealpha=0.90, fontsize=8)
        figure.savefig(
            output_dir / f"fruit_frame_{row['frame']:06d}_urdf_tcp.png",
            dpi=args.dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)

    summary = {
        "contract": (
            "Initial CR1 URDF pose from raw 16-D state; recorded and predicted absolute "
            "joint actions projected to left/right atomic TCP frames; paired prompt variants "
            "share image, state, and flow noise."
        ),
        "episode": episode,
        "active_arm_mode": args.active_arm,
        "tcp_frames": TCP_FRAMES,
        "tcp_local_z_offset_m": 0.20,
        "trajectory_npz": str(trajectory_path),
        "trajectory_npz_sha256": _sha256(trajectory_path),
        "urdf": str(urdf_path),
        "urdf_sha256": _sha256(urdf_path),
        "overview": str(overview),
        "frames": records,
    }
    (output_dir / "fruit_prompt_urdf_tcp_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
