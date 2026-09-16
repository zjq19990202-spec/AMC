#!/usr/bin/env python3
"""Render the standalone CR1 right-arm URDF and mark its TCP in red."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


DEFAULT_URDF = (
    Path(__file__).resolve().parents[1]
    / "assets/cr1_right_arm_tcp_red/urdf/CR1ARMR_TCP_RED.urdf"
)
DEFAULT_Q = (0.89, -0.28, -0.19, 0.29, 0.0, -1.26, -0.17)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--q",
        type=float,
        nargs=7,
        default=DEFAULT_Q,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"),
        help="Right-arm joint positions in URDF order (radians)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/visualizations/cr1_right_arm_tcp_red.png"),
    )
    parser.add_argument("--max-faces-per-link", type=int, default=30000)
    return parser.parse_args()


def vector(text: str | None, length: int = 3) -> np.ndarray:
    if not text:
        return np.zeros(length, dtype=np.float64)
    values = np.fromstring(text, sep=" ", dtype=np.float64)
    if len(values) != length:
        raise ValueError(f"Expected {length} values, got {text!r}")
    return values


def origin_transform(element: ET.Element | None) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if element is None:
        return transform
    xyz = vector(element.get("xyz"))
    rpy = vector(element.get("rpy"))
    transform[:3, 3] = xyz
    transform[:3, :3] = trimesh.transformations.euler_matrix(*rpy, axes="sxyz")[:3, :3]
    return transform


def axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("Joint axis cannot be zero")
    return trimesh.transformations.rotation_matrix(angle, axis / norm)


def link_placements(root: ET.Element, q: list[float]) -> dict[str, np.ndarray]:
    children: dict[str, list[ET.Element]] = defaultdict(list)
    child_links: set[str] = set()
    revolute_names: list[str] = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.get("link")
        child_name = child.get("link")
        if not parent_name or not child_name:
            continue
        children[parent_name].append(joint)
        child_links.add(child_name)
        if joint.get("type") in {"revolute", "continuous"}:
            revolute_names.append(joint.get("name", ""))

    links = {link.get("name", "") for link in root.findall("link")}
    roots = sorted(links - child_links)
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root link, got {roots}")
    if len(q) != len(revolute_names):
        raise ValueError(f"URDF has {len(revolute_names)} revolute joints, received {len(q)} q values")
    q_by_joint = dict(zip(revolute_names, q, strict=True))

    placements = {roots[0]: np.eye(4, dtype=np.float64)}
    queue = deque([roots[0]])
    while queue:
        parent_name = queue.popleft()
        for joint in children[parent_name]:
            child = joint.find("child")
            assert child is not None
            child_name = child.get("link", "")
            transform = placements[parent_name] @ origin_transform(joint.find("origin"))
            if joint.get("type") in {"revolute", "continuous"}:
                axis_element = joint.find("axis")
                axis = vector(axis_element.get("xyz") if axis_element is not None else "1 0 0")
                transform = transform @ axis_rotation(axis, q_by_joint[joint.get("name", "")])
            placements[child_name] = transform
            queue.append(child_name)
    return placements


def load_visual_meshes(
    root: ET.Element, urdf: Path, placements: dict[str, np.ndarray]
) -> list[tuple[str, trimesh.Trimesh]]:
    meshes: list[tuple[str, trimesh.Trimesh]] = []
    for link in root.findall("link"):
        link_name = link.get("name", "")
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            if geometry is None:
                continue
            mesh_element = geometry.find("mesh")
            box_element = geometry.find("box")
            if mesh_element is not None:
                filename = mesh_element.get("filename")
                if not filename:
                    continue
                mesh_path = (urdf.parent / filename).resolve()
                mesh = trimesh.load_mesh(mesh_path, force="mesh", process=False)
                scale = vector(mesh_element.get("scale"), length=3) if mesh_element.get("scale") else None
                if scale is not None:
                    mesh.apply_scale(scale)
            elif box_element is not None:
                mesh = trimesh.creation.box(extents=vector(box_element.get("size")))
            else:
                continue
            mesh.apply_transform(placements[link_name] @ origin_transform(visual.find("origin")))
            meshes.append((link_name, mesh))
    return meshes


def set_equal_axes(axis: plt.Axes, points: np.ndarray) -> None:
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = (low + high) / 2.0
    radius = max((high - low).max() / 2.0, 0.1) * 1.08
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def render(
    meshes: list[tuple[str, trimesh.Trimesh]],
    placements: dict[str, np.ndarray],
    output: Path,
    max_faces_per_link: int,
) -> None:
    figure = plt.figure(figsize=(10, 10), facecolor="white")
    axis = figure.add_subplot(111, projection="3d")
    all_vertices = []
    gray_palette = ("#aeb7c4", "#d2d8e0")
    gray_index = 0

    for link_name, mesh in meshes:
        faces = mesh.faces
        if len(faces) > max_faces_per_link:
            indices = np.linspace(0, len(faces) - 1, max_faces_per_link, dtype=np.int64)
            faces = faces[indices]
        triangles = mesh.vertices[faces]
        if link_name == "tcp":
            face_color = "#ef1b1b"
            edge_color = "#8b0000"
            line_width = 0.75
            alpha = 1.0
        else:
            face_color = gray_palette[gray_index % len(gray_palette)]
            edge_color = face_color
            line_width = 0.0
            alpha = 1.0
            gray_index += 1
        collection = Poly3DCollection(
            triangles,
            facecolors=face_color,
            edgecolors=edge_color,
            linewidth=line_width,
            alpha=alpha,
            shade=True,
        )
        axis.add_collection3d(collection)
        all_vertices.append(mesh.vertices)

    points = np.concatenate(all_vertices, axis=0)
    set_equal_axes(axis, points)
    tcp = placements["tcp"][:3, 3]
    axis.scatter(*tcp, color="#ef1b1b", s=65, depthshade=False)
    axis.text(tcp[0], tcp[1], tcp[2] + 0.035, "TCP", color="#b91c1c", weight="bold")

    origin = np.zeros(3)
    axis.quiver(*origin, 0.10, 0, 0, color="#dc2626", arrow_length_ratio=0.12)
    axis.quiver(*origin, 0, 0.10, 0, color="#16a34a", arrow_length_ratio=0.12)
    axis.quiver(*origin, 0, 0, 0.10, color="#2563eb", arrow_length_ratio=0.12)
    axis.text(0.11, 0, 0, "+x", color="#dc2626")
    axis.text(0, 0.11, 0, "+y", color="#16a34a")
    axis.text(0, 0, 0.11, "+z", color="#2563eb")

    axis.set_xlabel("base x (m)")
    axis.set_ylabel("base y (m)")
    axis.set_zlabel("base z (m)")
    axis.set_title("CR1 standalone right arm — red TCP marker", pad=18, fontsize=14)
    axis.view_init(elev=22, azim=-52)
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    urdf = args.urdf.expanduser().resolve()
    root = ET.parse(urdf).getroot()
    placements = link_placements(root, list(args.q))
    meshes = load_visual_meshes(root, urdf, placements)
    render(meshes, placements, args.output, args.max_faces_per_link)
    tcp = placements["tcp"][:3, 3]
    print(args.output)
    print(f"tcp_xyz_m={tcp.tolist()}")


if __name__ == "__main__":
    main()
