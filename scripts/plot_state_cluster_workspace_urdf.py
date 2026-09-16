#!/usr/bin/env python3
"""Plot a state-cluster TCP workspace projection over a posed CR1 URDF.

The selection manifest clusters full bimanual joint states (14 arm joints,
grippers excluded).  A three-dimensional figure cannot preserve that state
space, so this script visualizes its forward-kinematic projection into the
left/right atomic TCP workspace.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
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
ARM_COLORS = {"left": "#1677ff", "right": "#f97316"}
TCP_FRAMES = {"left": "left_atomic_tcp", "right": "right_atomic_tcp"}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robot-face-count", type=int, default=24_000)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _joint_values(row: dict[str, Any]) -> np.ndarray:
    state = np.asarray(row["full_state_16d"], dtype=np.float64)
    if state.shape != (16,):
        raise ValueError(f"expected full_state_16d, got {state.shape}")
    return np.concatenate((state[:7], state[8:15]))


def _configuration(q14: np.ndarray) -> dict[str, float]:
    names = LEFT_JOINTS + RIGHT_JOINTS
    return {name: float(value) for name, value in zip(names, q14, strict=True)}


def _fk_positions(robot: URDF, rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    left, right = [], []
    for row in rows:
        robot.update_cfg(_configuration(_joint_values(row)))
        left.append(robot.get_transform(TCP_FRAMES["left"])[:3, 3])
        right.append(robot.get_transform(TCP_FRAMES["right"])[:3, 3])
    return np.asarray(left), np.asarray(right)


def _medoid_index(states: np.ndarray) -> int:
    median = np.median(states, axis=0)
    return int(np.argmin(np.sqrt(np.mean((states - median) ** 2, axis=1))))


def _hull(points: np.ndarray) -> ConvexHull:
    if len(points) < 4:
        raise ValueError("at least four points are required for a 3D hull")
    return ConvexHull(points)


def _equal_axes(ax: Any, points: np.ndarray) -> None:
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = (lower + upper) / 2.0
    radius = float((upper - lower).max() / 2.0) * 1.05
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _pair_segments(
    rows: list[dict[str, Any]], target_positions: np.ndarray
) -> list[np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {}
    for row, position in zip(rows, target_positions, strict=True):
        grouped.setdefault(row["cluster_key"], []).append(position)
    return [np.stack(values) for values in grouped.values() if len(values) == 2]


def _static_plot(
    output: Path,
    rows: list[dict[str, Any]],
    left: np.ndarray,
    right: np.ndarray,
    robot_mesh: Any,
    representative_left: np.ndarray,
    representative_right: np.ndarray,
    dpi: int,
) -> None:
    left_hull, right_hull = _hull(left), _hull(right)

    fig = plt.figure(figsize=(14.5, 10), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    robot_triangles = robot_mesh.vertices[robot_mesh.faces]
    robot_surface = Poly3DCollection(
        robot_triangles,
        facecolor="#aeb6c2",
        edgecolor="none",
        alpha=0.38,
        zorder=1,
    )
    ax.add_collection3d(robot_surface)

    for arm, points, hull in (
        ("left", left, left_hull),
        ("right", right, right_hull),
    ):
        color = ARM_COLORS[arm]
        surface = Poly3DCollection(
            points[hull.simplices],
            facecolor=color,
            edgecolor=color,
            linewidth=0.25,
            alpha=0.10,
            zorder=2,
        )
        ax.add_collection3d(surface)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=18,
            c=color,
            alpha=0.72,
            depthshade=False,
            label=f"{arm.title()}-arm TCP ({len(points)} anchor states)",
            zorder=4,
        )

    ax.scatter(*representative_left, s=70, c=ARM_COLORS["left"], marker="*", edgecolors="white")
    ax.scatter(*representative_right, s=70, c=ARM_COLORS["right"], marker="*", edgecolors="white")
    all_points = np.vstack((robot_mesh.vertices, left, right))
    _equal_axes(ax, all_points)
    ax.set_xlabel("X / forward (m)", labelpad=10)
    ax.set_ylabel("Y / left (m)", labelpad=10)
    ax.set_zlabel("Z / up (m)", labelpad=10)
    ax.view_init(elev=23, azim=-58)
    ax.set_title(
        "CR1 state-cluster coverage — 3D atomic-TCP workspace projection\n"
        "250 full-bimanual clusters / 500 anchors; both arms included; TCP = wrist local +Z 0.20 m",
        pad=18,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.98), framealpha=0.92, fontsize=9)
    ax.grid(True, alpha=0.20)
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)


def _interactive_plot(
    output: Path,
    rows: list[dict[str, Any]],
    left: np.ndarray,
    right: np.ndarray,
    robot_mesh: Any,
) -> None:
    figure = go.Figure()

    figure.add_trace(
        go.Mesh3d(
            x=robot_mesh.vertices[:, 0], y=robot_mesh.vertices[:, 1], z=robot_mesh.vertices[:, 2],
            i=robot_mesh.faces[:, 0], j=robot_mesh.faces[:, 1], k=robot_mesh.faces[:, 2],
            color="#aeb6c2", opacity=0.40, name="CR1 URDF (representative pose)",
            hoverinfo="name", flatshading=True,
        )
    )

    for arm, points in (
        ("left", left),
        ("right", right),
    ):
        hull = _hull(points)
        hover = [
            "<br>".join(
                (
                    f"cluster={row['cluster_key']}",
                    f"cluster target arm={row['arm']}",
                    f"episode={row['episode']} frame={row['frame']}",
                    f"kind={row['kind']} atoms={'+'.join(row['atoms'])}",
                    f"prompt={row['atomic_prompt']}",
                )
            )
            for row in rows
        ]
        figure.add_trace(
            go.Mesh3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                i=hull.simplices[:, 0], j=hull.simplices[:, 1], k=hull.simplices[:, 2],
                color=ARM_COLORS[arm], opacity=0.11, name=f"{arm.title()}-arm coverage hull",
                hoverinfo="name",
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
                marker={"size": 4, "color": ARM_COLORS[arm], "opacity": 0.82},
                name=f"{arm.title()}-arm TCP ({len(points)} anchor states)",
                text=hover, hovertemplate="%{text}<extra></extra>",
            )
        )

    figure.update_layout(
        title=(
            "CR1 state-cluster coverage: 3D atomic-TCP projection"
            "<br><sup>250 full-bimanual clusters / 500 anchors; both arms included; TCP local +Z = 0.20 m</sup>"
        ),
        scene={
            "xaxis_title": "X / forward (m)",
            "yaxis_title": "Y / left (m)",
            "zaxis_title": "Z / up (m)",
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.55, "y": -1.75, "z": 1.15}},
        },
        legend={"x": 0.01, "y": 0.99},
        margin={"l": 0, "r": 0, "b": 0, "t": 75},
    )
    figure.write_html(output, include_plotlyjs=True, full_html=True)


def main() -> None:
    args = _arguments()
    manifest_path = args.manifest.expanduser().resolve()
    urdf_path = args.urdf.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["anchors"]
    if len(rows) != manifest.get("anchor_count"):
        raise ValueError("manifest anchor_count does not match anchors")
    if not manifest.get("full_state_clustering"):
        raise ValueError("this plot expects full bimanual state clustering")
    if float(manifest.get("gripper_distance_weight", -1.0)) != 0.0:
        raise ValueError("expected grippers to be excluded from cluster distance")

    # yourdfpy uses the filename string to resolve URDF-relative mesh paths.
    # Passing a pathlib.Path currently loses that base directory silently.
    robot = URDF.load(str(urdf_path), build_scene_graph=True, load_meshes=True)
    required = set(LEFT_JOINTS + RIGHT_JOINTS + tuple(TCP_FRAMES.values()))
    available = set(robot.joint_map) | set(robot.link_map)
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"URDF is missing required joints/links: {missing}")

    left, right = _fk_positions(robot, rows)
    states = np.stack([_joint_values(row) for row in rows])
    representative_index = _medoid_index(states)
    representative = rows[representative_index]
    robot.update_cfg(_configuration(states[representative_index]))
    representative_left = robot.get_transform(TCP_FRAMES["left"])[:3, 3]
    representative_right = robot.get_transform(TCP_FRAMES["right"])[:3, 3]

    robot_mesh = robot.scene.dump(concatenate=True)
    if len(robot_mesh.faces) > args.robot_face_count:
        robot_mesh = robot_mesh.simplify_quadric_decimation(face_count=args.robot_face_count)

    target_left_mask = np.asarray([row["arm"] == "left" for row in rows])
    target_right_mask = ~target_left_mask
    target_positions = np.where(target_left_mask[:, None], left, right)
    left_hull = _hull(left)
    right_hull = _hull(right)
    segments = _pair_segments(rows, target_positions)
    pair_distances_mm = np.asarray([np.linalg.norm(segment[1] - segment[0]) * 1000 for segment in segments])

    csv_path = output_dir / "state_cluster_tcp_workspace.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = (
            "anchor_key", "cluster_key", "episode", "frame", "target_arm",
            "left_x", "left_y", "left_z", "right_x", "right_y", "right_z",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row, left_position, right_position in zip(rows, left, right, strict=True):
            writer.writerow(
                {
                    "anchor_key": row["anchor_key"], "cluster_key": row["cluster_key"],
                    "episode": row["episode"], "frame": row["frame"], "target_arm": row["arm"],
                    "left_x": left_position[0], "left_y": left_position[1], "left_z": left_position[2],
                    "right_x": right_position[0], "right_y": right_position[1], "right_z": right_position[2],
                }
            )

    summary = {
        "contract": (
            "3D forward-kinematic projection of full 14D bimanual joint-state clusters; "
            "grippers excluded from clustering"
        ),
        "coordinate_frame": "CR1 URDF base_link; X forward, Y left, Z up",
        "tcp_frames": TCP_FRAMES,
        "tcp_local_z_offset_m": 0.20,
        "waist_configuration_rad": {"waist_z_joint": 0.0, "waist_x_joint": 0.0, "waist_y_joint": 0.0},
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "urdf": str(urdf_path),
        "urdf_sha256": _sha256(urdf_path),
        "anchor_count": len(rows),
        "cluster_count": len(manifest["clusters"]),
        "target_anchor_count": {"left": int(target_left_mask.sum()), "right": int(target_right_mask.sum())},
        "full_cluster_projection_convex_hull_volume_m3": {
            "left": left_hull.volume, "right": right_hull.volume
        },
        "full_cluster_projection_tcp_bounds_m": {
            "left": {"min": left.min(axis=0).tolist(), "max": left.max(axis=0).tolist()},
            "right": {"min": right.min(axis=0).tolist(), "max": right.max(axis=0).tolist()},
        },
        "within_cluster_target_tcp_distance_mm": {
            "count": len(pair_distances_mm),
            "median": float(np.median(pair_distances_mm)),
            "p90": float(np.percentile(pair_distances_mm, 90)),
            "max": float(pair_distances_mm.max()),
        },
        "representative_robot_pose": {
            "selection": "observed anchor nearest the coordinate-wise median in 14D joint RMS",
            "anchor_key": representative["anchor_key"],
            "episode": representative["episode"],
            "frame": representative["frame"],
            "q14_rad": states[representative_index].tolist(),
        },
        "robot_mesh_faces_rendered": int(len(robot_mesh.faces)),
    }
    (output_dir / "state_cluster_workspace_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _static_plot(
        output_dir / "state_cluster_workspace_with_cr1.png",
        rows, left, right, robot_mesh, representative_left, representative_right, args.dpi,
    )
    _interactive_plot(
        output_dir / "state_cluster_workspace_with_cr1_interactive.html",
        rows, left, right, robot_mesh,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
