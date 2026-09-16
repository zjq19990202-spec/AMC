#!/usr/bin/env python3
"""Plot per-cluster TCP envelopes from every member state over the CR1 URDF."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from yourdfpy import URDF


COLORS = {"left": "#1677ff", "right": "#f97316"}
JOINT_STATE_COLUMN = {
    "left_shoulder_y_joint": 0,
    "left_shoulder_x_joint": 1,
    "left_shoulder_z_joint": 2,
    "left_elbow_joint": 3,
    "left_wrist_z_joint": 4,
    "left_wrist_y_joint": 5,
    "left_wrist_x_joint": 6,
    "right_shoulder_y_joint": 8,
    "right_shoulder_x_joint": 9,
    "right_shoulder_z_joint": 10,
    "right_elbow_joint": 11,
    "right_wrist_z_joint": 12,
    "right_wrist_y_joint": 13,
    "right_wrist_x_joint": 14,
}
JOINT_COLUMNS_14D = np.asarray([*range(7), *range(8, 15)])
TCP_LINK = {"left": "left_atomic_tcp", "right": "right_atomic_tcp"}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--members", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robot-face-count", type=int, default=24_000)
    parser.add_argument("--member-points-per-cluster", type=int, default=48)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _vector(text: str | None) -> np.ndarray:
    return np.zeros(3) if not text else np.fromstring(text, sep=" ", dtype=np.float64)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _origin_transform(origin: ET.Element | None) -> np.ndarray:
    transform = np.eye(4)
    if origin is not None:
        transform[:3, :3] = _rpy_matrix(_vector(origin.get("rpy")))
        transform[:3, 3] = _vector(origin.get("xyz"))
    return transform


def _axis_rotations(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    outer = np.outer(axis, axis)
    cosine = np.cos(angles)[:, None, None]
    sine = np.sin(angles)[:, None, None]
    rotations = cosine * np.eye(3) + (1.0 - cosine) * outer + sine * skew
    transforms = np.zeros((len(angles), 4, 4), dtype=np.float64)
    transforms[:, :3, :3] = rotations
    transforms[:, 3, 3] = 1.0
    return transforms


class BatchUrdfFK:
    def __init__(self, urdf: Path) -> None:
        root = ET.parse(urdf).getroot()
        self.joints = {joint.get("name", ""): joint for joint in root.findall("joint")}
        self.parent_joint: dict[str, ET.Element] = {}
        for joint in self.joints.values():
            child = joint.find("child")
            if child is not None and child.get("link"):
                self.parent_joint[child.get("link", "")] = joint

    def _chain(self, target_link: str) -> list[ET.Element]:
        chain = []
        link = target_link
        while link in self.parent_joint:
            joint = self.parent_joint[link]
            chain.append(joint)
            parent = joint.find("parent")
            if parent is None:
                break
            link = parent.get("link", "")
        return list(reversed(chain))

    def positions(self, states: np.ndarray, target_link: str, chunk_size: int = 16384) -> np.ndarray:
        chain = self._chain(target_link)
        outputs = []
        for start in range(0, len(states), chunk_size):
            chunk = np.asarray(states[start : start + chunk_size], dtype=np.float64)
            transform = np.broadcast_to(np.eye(4), (len(chunk), 4, 4)).copy()
            for joint in chain:
                transform = transform @ _origin_transform(joint.find("origin"))
                if joint.get("type") not in {"revolute", "continuous"}:
                    continue
                joint_name = joint.get("name", "")
                angles = (
                    chunk[:, JOINT_STATE_COLUMN[joint_name]]
                    if joint_name in JOINT_STATE_COLUMN
                    else np.zeros(len(chunk))
                )
                axis_element = joint.find("axis")
                axis = _vector(axis_element.get("xyz") if axis_element is not None else "1 0 0")
                transform = transform @ _axis_rotations(axis, angles)
            outputs.append(transform[:, :3, 3])
        return np.concatenate(outputs)


def _configuration(state: np.ndarray) -> dict[str, float]:
    return {name: float(state[column]) for name, column in JOINT_STATE_COLUMN.items()}


def _ellipsoid(points: np.ndarray, rows: int = 8, columns: int = 12) -> dict[str, Any]:
    center = points.mean(axis=0)
    delta = points - center
    covariance = np.cov(delta, rowvar=False) + np.eye(3) * 1e-10
    inverse = np.linalg.inv(covariance)
    mahalanobis_squared = np.einsum("ni,ij,nj->n", delta, inverse, delta)
    scale = math.sqrt(float(np.quantile(mahalanobis_squared, 0.9)))
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    axes = np.sqrt(np.maximum(values[order], 1e-12)) * scale
    vectors = vectors[:, order]
    u = np.linspace(0, 2 * np.pi, columns, endpoint=False)
    v = np.linspace(0, np.pi, rows)
    unit = np.stack(
        (
            (np.sin(v)[:, None] * np.cos(u)[None, :]).ravel(),
            (np.sin(v)[:, None] * np.sin(u)[None, :]).ravel(),
            np.broadcast_to(np.cos(v)[:, None], (rows, columns)).ravel(),
        ),
        axis=1,
    )
    vertices = center + (unit * axes) @ vectors.T
    faces = []
    for row in range(rows - 1):
        for column in range(columns):
            next_column = (column + 1) % columns
            a, b = row * columns + column, row * columns + next_column
            c, d = (row + 1) * columns + column, (row + 1) * columns + next_column
            faces.extend(((a, c, b), (b, c, d)))
    return {
        "center": center,
        "axes": axes,
        "vertices": vertices,
        "faces": np.asarray(faces, dtype=np.int32),
        "tcp_radial_p90_m": float(np.quantile(np.linalg.norm(delta, axis=1), 0.9)),
    }


def _combine_meshes(records: list[dict[str, Any]], arm: str) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces, offset = [], [], 0
    for record in records:
        if record["arm"] != arm:
            continue
        mesh = record["ellipsoid"]
        vertices.append(mesh["vertices"])
        faces.append(mesh["faces"] + offset)
        offset += len(mesh["vertices"])
    return np.concatenate(vertices), np.concatenate(faces)


def _equal_axes(ax: Any, points: np.ndarray) -> None:
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = (lower + upper) / 2
    radius = float((upper - lower).max() / 2) * 1.04
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _static_plot(path: Path, records: list[dict[str, Any]], robot_mesh: Any, dpi: int) -> None:
    fig = plt.figure(figsize=(14.5, 10), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(
        Poly3DCollection(
            robot_mesh.vertices[robot_mesh.faces], facecolor="#aeb6c2", edgecolor="none", alpha=0.38
        )
    )
    all_vertices = [robot_mesh.vertices]
    for arm in ("left", "right"):
        vertices, faces = _combine_meshes(records, arm)
        all_vertices.append(vertices)
        ax.add_collection3d(
            Poly3DCollection(
                vertices[faces], facecolor=COLORS[arm], edgecolor=COLORS[arm],
                linewidth=0.12, alpha=0.055,
            )
        )
        centers = np.stack([row["ellipsoid"]["center"] for row in records if row["arm"] == arm])
        ax.scatter(
            centers[:, 0], centers[:, 1], centers[:, 2], s=13, c=COLORS[arm], alpha=0.88,
            depthshade=False, label=f"{arm.title()} target: 125 member-state clusters",
        )
    _equal_axes(ax, np.vstack(all_vertices))
    ax.set_xlabel("X / forward (m)", labelpad=10)
    ax.set_ylabel("Y / left (m)", labelpad=10)
    ax.set_zlabel("Z / up (m)", labelpad=10)
    ax.view_init(elev=23, azim=-58)
    ax.set_title(
        "CR1 per-cluster member-state coverage — 90% TCP ellipsoids\n"
        "250 selected target-arm clusters; envelopes computed from every assigned 3 Hz state",
        pad=18,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.98), framealpha=0.94)
    ax.grid(True, alpha=0.20)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def _static_arm_panels(path: Path, records: list[dict[str, Any]], robot_mesh: Any, dpi: int) -> None:
    figure = plt.figure(figsize=(17, 8.2), constrained_layout=True)
    bounds = np.vstack(
        [robot_mesh.vertices]
        + [row["ellipsoid"]["vertices"] for row in records]
    )
    for panel, arm in enumerate(("left", "right"), 1):
        ax = figure.add_subplot(1, 2, panel, projection="3d")
        ax.add_collection3d(
            Poly3DCollection(
                robot_mesh.vertices[robot_mesh.faces],
                facecolor="#aeb6c2", edgecolor="none", alpha=0.30,
            )
        )
        vertices, faces = _combine_meshes(records, arm)
        ax.add_collection3d(
            Poly3DCollection(
                vertices[faces], facecolor=COLORS[arm], edgecolor=COLORS[arm],
                linewidth=0.08, alpha=0.045,
            )
        )
        arm_records = [row for row in records if row["arm"] == arm]
        centers = np.stack([row["ellipsoid"]["center"] for row in arm_records])
        ax.scatter(
            centers[:, 0], centers[:, 1], centers[:, 2],
            s=11, c=COLORS[arm], alpha=0.92, depthshade=False,
        )
        _equal_axes(ax, bounds)
        ax.set_xlabel("X / forward (m)")
        ax.set_ylabel("Y / left (m)")
        ax.set_zlabel("Z / up (m)")
        ax.view_init(elev=23, azim=-58)
        ax.set_title(f"{arm.title()} target — 125 per-cluster 90% TCP envelopes")
        ax.grid(True, alpha=0.20)
    figure.suptitle(
        "CR1 member-state cluster radii (left/right separated)\n"
        "Each envelope is fitted from every 3 Hz state assigned to that cluster",
        fontsize=16,
    )
    figure.savefig(path, dpi=dpi, facecolor="white")
    plt.close(figure)


def _static_left_right_point_cloud(
    path: Path,
    positions: dict[str, np.ndarray],
    robot_mesh: Any,
    dpi: int,
) -> dict[str, int]:
    """Plot both TCPs for every selected full-state cluster member."""
    point_clouds = {arm: positions[arm] for arm in ("left", "right")}

    figure = plt.figure(figsize=(14.5, 10), constrained_layout=True)
    ax = figure.add_subplot(111, projection="3d")
    ax.add_collection3d(
        Poly3DCollection(
            robot_mesh.vertices[robot_mesh.faces],
            facecolor="#aeb6c2",
            edgecolor="none",
            alpha=0.34,
        )
    )
    for arm in ("left", "right"):
        points = point_clouds[arm]
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            s=2.0, c=COLORS[arm], alpha=0.10, linewidths=0,
            depthshade=False, rasterized=True,
            label=f"{arm.title()} TCP members ({len(points):,})",
        )

    bounds = np.vstack([robot_mesh.vertices, *point_clouds.values()])
    _equal_axes(ax, bounds)
    ax.set_xlabel("X / forward (m)", labelpad=10)
    ax.set_ylabel("Y / left (m)", labelpad=10)
    ax.set_zlabel("Z / up (m)", labelpad=10)
    ax.view_init(elev=23, azim=-58)
    ax.set_title(
        "CR1 selected-cluster TCP member point clouds\n"
        "Left arm = blue, right arm = orange; every assigned 3 Hz member state",
        pad=18,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.98), framealpha=0.94)
    ax.grid(True, alpha=0.20)
    figure.savefig(path, dpi=dpi, facecolor="white")
    plt.close(figure)
    return {arm: len(points) for arm, points in point_clouds.items()}


def _interactive_plot(
    path: Path,
    records: list[dict[str, Any]],
    robot_mesh: Any,
    sample_points: dict[str, np.ndarray],
) -> None:
    figure = go.Figure()
    figure.add_trace(
        go.Mesh3d(
            x=robot_mesh.vertices[:, 0], y=robot_mesh.vertices[:, 1], z=robot_mesh.vertices[:, 2],
            i=robot_mesh.faces[:, 0], j=robot_mesh.faces[:, 1], k=robot_mesh.faces[:, 2],
            color="#aeb6c2", opacity=0.40, flatshading=True,
            name="CR1 URDF (member-state medoid pose)", hoverinfo="name",
        )
    )
    for arm in ("left", "right"):
        vertices, faces = _combine_meshes(records, arm)
        arm_records = [row for row in records if row["arm"] == arm]
        centers = np.stack([row["ellipsoid"]["center"] for row in arm_records])
        hover = [
            "<br>".join(
                (
                    f"cluster={row['cluster_key']}",
                    f"members={row['member_count']}",
                    f"state RMS p90={row['state_rms_p90_rad']:.4f} rad",
                    f"TCP radial p90={row['ellipsoid']['tcp_radial_p90_m'] * 1000:.1f} mm",
                    "TCP ellipsoid axes=" + ", ".join(
                        f"{axis * 1000:.1f} mm" for axis in row["ellipsoid"]["axes"]
                    ),
                )
            )
            for row in arm_records
        ]
        figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color=COLORS[arm],
                opacity=0.075, name=f"{arm.title()} 90% member envelopes", hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=centers[:, 0], y=centers[:, 1], z=centers[:, 2], mode="markers",
                marker={"size": 4, "color": COLORS[arm], "opacity": 0.92},
                name=f"{arm.title()} cluster centers (125)", text=hover,
                hovertemplate="%{text}<extra></extra>",
            )
        )
        points = sample_points[arm]
        figure.add_trace(
            go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
                marker={"size": 1.5, "color": COLORS[arm], "opacity": 0.18},
                name=f"{arm.title()} sampled member TCPs", hoverinfo="skip", visible="legendonly",
            )
        )
    figure.update_layout(
        title=(
            "CR1 per-cluster member-state coverage: 90% TCP ellipsoids"
            "<br><sup>250 selected target-arm clusters; every assigned 3 Hz member state used</sup>"
        ),
        scene={
            "xaxis_title": "X / forward (m)", "yaxis_title": "Y / left (m)",
            "zaxis_title": "Z / up (m)", "aspectmode": "data",
            "camera": {"eye": {"x": 1.55, "y": -1.75, "z": 1.15}},
        },
        legend={"x": 0.01, "y": 0.99}, margin={"l": 0, "r": 0, "b": 0, "t": 75},
    )
    figure.write_html(path, include_plotlyjs=True, full_html=True)


def main() -> None:
    args = _arguments()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    members = np.load(args.members)
    states = np.asarray(members["states"], dtype=np.float64)
    labels = np.asarray(members["cluster_id"], dtype=np.int64)
    selected_ids = set(np.asarray(members["selected_cluster_ids"], dtype=int).tolist())
    manifest_ids = {
        int(row["cluster_key"].split("_c", 1)[1]) for row in manifest["clusters"]
    }
    if selected_ids != manifest_ids:
        raise ValueError("member archive and selection manifest contain different cluster IDs")

    fk = BatchUrdfFK(args.urdf)
    left_positions = fk.positions(states, TCP_LINK["left"])
    right_positions = fk.positions(states, TCP_LINK["right"])
    positions = {"left": left_positions, "right": right_positions}

    records = []
    maximum_p90_error = 0.0
    count_mismatches = 0
    rng = np.random.default_rng(20260821)
    sampled = {"left": [], "right": []}
    for row in manifest["clusters"]:
        cluster_id = int(row["cluster_key"].split("_c", 1)[1])
        mask = labels == cluster_id
        cluster_states = states[mask]
        center = cluster_states.mean(axis=0)
        state_rms = np.sqrt(
            np.mean(np.square(cluster_states[:, JOINT_COLUMNS_14D] - center[JOINT_COLUMNS_14D]), axis=1)
        )
        p90 = float(np.quantile(state_rms, 0.9))
        maximum_p90_error = max(maximum_p90_error, abs(p90 - row["state_rms_p90_rad"]))
        count_mismatches += int(len(cluster_states) != row["frame_count_3hz"])
        cluster_positions = positions[row["arm"]][mask]
        take = min(args.member_points_per_cluster, len(cluster_positions))
        sampled[row["arm"]].append(cluster_positions[rng.choice(len(cluster_positions), take, replace=False)])
        records.append(
            {
                "cluster_key": row["cluster_key"], "arm": row["arm"],
                "member_count": len(cluster_states), "state_rms_p90_rad": p90,
                "ellipsoid": _ellipsoid(cluster_positions),
            }
        )
    if count_mismatches or maximum_p90_error > 1e-6:
        raise RuntimeError(
            f"member validation failed: count_mismatches={count_mismatches}, "
            f"maximum_p90_error={maximum_p90_error}"
        )
    sampled_points = {arm: np.vstack(values) for arm, values in sampled.items()}

    q14 = states[:, JOINT_COLUMNS_14D]
    median = np.median(q14, axis=0)
    representative_index = int(np.argmin(np.mean(np.square(q14 - median), axis=1)))
    robot = URDF.load(str(args.urdf.resolve()), build_scene_graph=True, load_meshes=True)
    robot.update_cfg(_configuration(states[representative_index]))
    robot_mesh = robot.scene.dump(concatenate=True)
    if len(robot_mesh.faces) > args.robot_face_count:
        robot_mesh = robot_mesh.simplify_quadric_decimation(face_count=args.robot_face_count)

    _static_plot(output_dir / "cluster_member_radii_with_cr1.png", records, robot_mesh, args.dpi)
    _static_arm_panels(
        output_dir / "cluster_member_radii_left_right_panels.png",
        records, robot_mesh, args.dpi,
    )
    point_cloud_counts = _static_left_right_point_cloud(
        output_dir / "cluster_member_pointcloud_left_right_with_cr1.png",
        positions, robot_mesh, args.dpi,
    )
    _interactive_plot(
        output_dir / "cluster_member_radii_with_cr1_interactive.html",
        records, robot_mesh, sampled_points,
    )

    summary_records = []
    for row in records:
        summary_records.append(
            {
                "cluster_key": row["cluster_key"], "arm": row["arm"],
                "member_count": row["member_count"],
                "state_rms_p90_rad": row["state_rms_p90_rad"],
                "tcp_center_m": row["ellipsoid"]["center"].tolist(),
                "tcp_radial_p90_m": row["ellipsoid"]["tcp_radial_p90_m"],
                "tcp_ellipsoid_axes_m": row["ellipsoid"]["axes"].tolist(),
            }
        )
    summary = {
        "contract": "Per-cluster target-arm TCP envelopes from every assigned member state",
        "selection_manifest": str(args.manifest.resolve()),
        "member_archive": str(args.members.resolve()),
        "urdf": str(args.urdf.resolve()),
        "tcp_offset_m": 0.20,
        "selected_arm_prefixed_clusters": len(records),
        "unique_geometric_clusters": len(selected_ids),
        "member_state_count": len(states),
        "full_state_arm_point_cloud_counts": point_cloud_counts,
        "member_validation": {
            "frame_count_mismatches": count_mismatches,
            "maximum_state_p90_absolute_error": maximum_p90_error,
        },
        "clusters": summary_records,
    }
    (output_dir / "cluster_member_radii_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "clusters"}, indent=2))


if __name__ == "__main__":
    main()
