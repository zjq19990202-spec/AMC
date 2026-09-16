#!/usr/bin/env python3
"""Compare 50-step FK trajectories under global, atomic and opposite prompts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from pathlib import Path
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.training import config as _training_config

from atomic_latent_vla.atomic import ATOMIC_NAMES
from atomic_latent_vla.annotation.motion import default_mount_xyz
from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import (
    _paligemma_tokenizer,
    atomic_collate,
    batch_to_observation,
    build_atomic_dataset,
)
from atomic_latent_vla.tcp import TCP_LOCAL_Z_OFFSET_M


VARIANT_COLORS = {
    "global": "#2563eb",
    "atomic": "#ea580c",
    "opposite": "#be185d",
    "gt": "#111827",
}
VARIANT_STYLES = {
    "global": "-",
    "atomic": "--",
    "opposite": ":",
    "gt": "-",
}


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return (
        np.eye(3)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def _vector(text: str | None) -> np.ndarray:
    if not text:
        return np.zeros(3, dtype=np.float64)
    value = np.fromstring(text, sep=" ", dtype=np.float64)
    if value.shape != (3,):
        raise ValueError(f"expected xyz/rpy/axis triplet, got {text!r}")
    return value


def _origin_transform(element: ET.Element | None) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if element is not None:
        result[:3, 3] = _vector(element.get("xyz"))
        result[:3, :3] = _rpy_matrix(_vector(element.get("rpy")))
    return result


class SimpleCR1FK:
    """Dependency-free URDF FK matching the dataset's wrist+0.19m TCP."""

    def __init__(self, urdf: Path):
        self.root = ET.parse(urdf).getroot()
        self.children: dict[str, list[ET.Element]] = defaultdict(list)
        child_links = set()
        links = {link.get("name", "") for link in self.root.findall("link")}
        self.revolute_names = []
        for joint in self.root.findall("joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            parent_name, child_name = parent.get("link"), child.get("link")
            if not parent_name or not child_name:
                continue
            self.children[parent_name].append(joint)
            child_links.add(child_name)
            if joint.get("type") in {"revolute", "continuous"}:
                self.revolute_names.append(joint.get("name", ""))
        roots = links - child_links
        if len(roots) != 1 or len(self.revolute_names) != 7:
            raise ValueError(
                f"expected one URDF root and seven joints, got {roots}, "
                f"{self.revolute_names}"
            )
        self.root_link = next(iter(roots))
        self.mount = np.asarray(default_mount_xyz("right"), dtype=np.float64)

    def pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=np.float64)
        q_by_joint = dict(zip(self.revolute_names, q, strict=True))
        placements = {self.root_link: np.eye(4, dtype=np.float64)}
        queue = deque([self.root_link])
        while queue:
            parent_name = queue.popleft()
            for joint in self.children[parent_name]:
                child = joint.find("child")
                assert child is not None
                child_name = child.get("link", "")
                transform = placements[parent_name] @ _origin_transform(
                    joint.find("origin")
                )
                if joint.get("type") in {"revolute", "continuous"}:
                    axis_element = joint.find("axis")
                    axis = _vector(
                        axis_element.get("xyz")
                        if axis_element is not None
                        else "1 0 0"
                    )
                    rotation = np.eye(4)
                    rotation[:3, :3] = _axis_rotation(
                        axis, q_by_joint[joint.get("name", "")]
                    )
                    transform = transform @ rotation
                placements[child_name] = transform
                queue.append(child_name)
        wrist = placements["right_wrist_x_link"]
        rotation = wrist[:3, :3]
        position = (
            wrist[:3, 3]
            + self.mount
            + rotation @ np.asarray([0.0, 0.0, TCP_LOCAL_Z_OFFSET_M])
        )
        return position, rotation

    def trajectory(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        poses = [self.pose(row) for row in np.asarray(q)]
        return (
            np.stack([pose[0] for pose in poses]),
            np.stack([pose[1] for pose in poses]),
        )


def _rotation_error_deg(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = np.einsum("...ji,...jk->...ik", prediction, target)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1, 1)
    return np.degrees(np.arccos(cosine))


def _opposite_prompt(prompt: str, labels: tuple[str, ...]) -> str:
    """Invert only the motion words in the three episode-485 prompt families."""

    result = prompt
    label_set = set(labels)
    if label_set == {"move_x_pos", "move_y_pos"}:
        result = result.replace("moves forward and left", "moves backward and right", 1)
    elif label_set == {"move_y_pos", "move_z_pos"}:
        result = result.replace("moves left and upward", "moves right and downward", 1)
    elif label_set == {"move_x_neg", "rotate_z_neg"}:
        result = result.replace(
            "rotates the door handle negative",
            "rotates the door handle positive",
            1,
        ).replace("moving backward", "moving forward", 1)
    if result == prompt:
        raise ValueError(f"no validated opposite rewrite for {labels}: {prompt}")
    return result


@nnx.jit
def _sample_three(
    model,
    observation,
    atomic_tokens,
    atomic_mask,
    opposite_tokens,
    opposite_mask,
    noise,
):
    observation = _model.preprocess_observation(None, observation, train=False)
    atomic_observation = model._with_prompt(  # noqa: SLF001
        observation, atomic_tokens, atomic_mask
    )
    opposite_observation = model._with_prompt(  # noqa: SLF001
        observation, opposite_tokens, opposite_mask
    )
    noise = model._mask_action_condition(noise)  # noqa: SLF001
    active_state = model._controlled_state(observation.state)  # noqa: SLF001

    def sample(prompted_observation):
        query_hidden, prefix_mask, kv_cache = model._prefix_forward(  # noqa: SLF001
            prompted_observation
        )
        _, _, z_model, _, _ = model._latent(query_hidden, active_state)  # noqa: SLF001

        def step(index, actions):
            time = jnp.asarray(1.0 - index / 10.0, dtype=actions.dtype)
            velocity = model._suffix_velocity(  # noqa: SLF001
                prefix_mask,
                kv_cache,
                actions,
                jnp.broadcast_to(time, (actions.shape[0],)),
                z_model,
            )
            return model._mask_action_condition(actions - 0.1 * velocity)  # noqa: SLF001

        result = jax.lax.fori_loop(0, 10, step, noise)
        return result[..., : model.config.active_action_dim]

    return (
        sample(observation),
        sample(atomic_observation),
        sample(opposite_observation),
    )


def _select_horizons(dataset, episode: int, count: int):
    raw = dataset._raw  # noqa: SLF001
    by_segment = defaultdict(list)
    for dataset_index, data_index in enumerate(raw.base._visible_indices):  # noqa: SLF001
        if int(raw.base._episode_index[data_index]) != episode:  # noqa: SLF001
            continue
        metadata = raw.metadata(dataset_index)
        if not metadata["atomic_supervision_mask"]:
            continue
        segment_id = int(raw._segment_id[data_index])  # noqa: SLF001
        by_segment[segment_id].append((dataset_index, metadata))
    if not by_segment:
        raise RuntimeError(f"episode {episode} has no strict atomic horizons")
    ordered = sorted(by_segment, key=lambda key: (-len(by_segment[key]), key))
    quotas = {key: count // len(ordered) for key in ordered}
    for key in ordered[: count % len(ordered)]:
        quotas[key] += 1
    selected = []
    for segment_id in sorted(by_segment):
        rows = by_segment[segment_id]
        indices = np.linspace(0, len(rows) - 1, quotas[segment_id]).round().astype(int)
        selected.extend(
            (segment_id, *rows[index]) for index in np.unique(indices)
        )
    return selected


def _output_transform(
    dataset_root: Path,
    config: AtomicPi05Config,
    *,
    norm_assets_dir: Path = Path("/mnt/cunchu/zjq/target"),
    norm_asset_id: str = "openpi_norm_compact_accepted_v3",
):
    """Build the training-faithful action decoder for the selected norm asset."""
    bridge_config = pi0_config.Pi0Config(pi05=True, max_token_len=config.max_token_len)
    factory = _training_config.LeRobotMarvinDataConfig(
        repo_id=str(dataset_root),
        prompt_from_task=True,
        adapt_to_pi=True,
        assets=_training_config.AssetsConfig(
            assets_dir=str(norm_assets_dir),
            asset_id=norm_asset_id,
        ),
    )
    data_config = factory.create(norm_assets_dir, bridge_config)
    unnormalize = _transforms.Unnormalize(
        data_config.norm_stats,
        use_quantiles=data_config.use_quantile_norm,
    )
    def decode(
        normalized_state: np.ndarray,
        raw_state: np.ndarray,
        normalized_actions: np.ndarray,
    ) -> dict[str, np.ndarray]:
        values = unnormalize(
            {
                "state": np.asarray(normalized_state),
                "actions": np.asarray(normalized_actions),
            }
        )
        for transform in data_config.data_transforms.outputs:
            values = transform(values)
        return {"actions": np.asarray(values["actions"])}

    return decode


def _equal_3d(axis, points: np.ndarray) -> None:
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    radius = max(float((high - low).max()) / 2, 0.005) * 1.12
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def _plot_pages(samples: list[dict], output_dir: Path) -> list[str]:
    pages = []
    for page_index, start in enumerate(range(0, len(samples), 5), 1):
        page = samples[start : start + 5]
        figure = plt.figure(figsize=(18, 4.0 * len(page)), constrained_layout=True)
        grid = figure.add_gridspec(len(page), 3, width_ratios=(1.15, 1, 1))
        for row_index, sample in enumerate(page):
            axis_3d = figure.add_subplot(grid[row_index, 0], projection="3d")
            all_positions = []
            for variant in ("gt", "global", "atomic", "opposite"):
                position = sample["poses"][variant][0]
                all_positions.append(position)
                axis_3d.plot(
                    position[:, 0], position[:, 1], position[:, 2],
                    color=VARIANT_COLORS[variant],
                    linestyle=VARIANT_STYLES[variant],
                    linewidth=2.2 if variant == "gt" else 1.6,
                    label=variant,
                )
                axis_3d.scatter(
                    *position[-1], color=VARIANT_COLORS[variant], s=18
                )
            _equal_3d(axis_3d, np.concatenate(all_positions))
            axis_3d.set_xlabel("x (m)")
            axis_3d.set_ylabel("y (m)")
            axis_3d.set_zlabel("z (m)")
            axis_3d.set_title(
                f"H{start + row_index + 1} seg={sample['segment_id']} "
                f"t={sample['timestamp_s']:.2f}s\n{sample['labels']}",
                fontsize=10,
            )
            if row_index == 0:
                axis_3d.legend(frameon=False, fontsize=8, loc="upper left")

            translation_axis = figure.add_subplot(grid[row_index, 1])
            rotation_axis = figure.add_subplot(grid[row_index, 2])
            steps = np.arange(50)
            for variant in ("global", "atomic", "opposite"):
                translation_axis.plot(
                    steps, sample["translation_error_mm"][variant],
                    color=VARIANT_COLORS[variant],
                    linestyle=VARIANT_STYLES[variant],
                    linewidth=1.6,
                    label=variant,
                )
                rotation_axis.plot(
                    steps, sample["rotation_error_deg"][variant],
                    color=VARIANT_COLORS[variant],
                    linestyle=VARIANT_STYLES[variant],
                    linewidth=1.6,
                    label=variant,
                )
            translation_axis.set_title("TCP translation error", fontsize=10)
            translation_axis.set_xlabel("horizon step")
            translation_axis.set_ylabel("mm")
            rotation_axis.set_title("TCP rotation error", fontsize=10)
            rotation_axis.set_xlabel("horizon step")
            rotation_axis.set_ylabel("degrees")
            for axis in (translation_axis, rotation_axis):
                axis.grid(color="#e5e7eb", linewidth=0.6)
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                if row_index == 0:
                    axis.legend(frameon=False, fontsize=8)
        figure.suptitle(
            "Episode 485 — 50-step FK trajectory prompt ablation "
            f"(page {page_index})",
            fontsize=15,
            fontweight="bold",
        )
        path = output_dir / f"fk_trajectory_ablation_page{page_index}.png"
        figure.savefig(path, dpi=190, bbox_inches="tight")
        plt.close(figure)
        pages.append(str(path))
    return pages


def _translation_target(labels: str) -> np.ndarray:
    target = np.zeros(3, dtype=np.float64)
    for label in labels.split("+"):
        parts = label.split("_")
        if len(parts) == 3 and parts[0] == "move":
            axis = {"x": 0, "y": 1, "z": 2}[parts[1]]
            target[axis] += 1.0 if parts[2] == "pos" else -1.0
    norm = np.linalg.norm(target)
    return target / norm if norm > 0 else target


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 1e-9 else 0.0


def _plot_direction_pages(samples: list[dict], output_dir: Path) -> list[str]:
    pages = []
    axis_names = ("Δx", "Δy", "Δz")
    colors = {"atomic": "#ea580c", "opposite": "#c2185b"}
    for page_index, start in enumerate(range(0, len(samples), 5), 1):
        page = samples[start : start + 5]
        figure, axes = plt.subplots(
            len(page), 2, figsize=(15, 3.2 * len(page)), constrained_layout=True
        )
        axes = np.atleast_2d(axes)
        for row_index, sample in enumerate(page):
            trace_axis, bar_axis = axes[row_index]
            for coordinate, coordinate_name in enumerate(axis_names):
                for variant, style in (("atomic", "-"), ("opposite", "--")):
                    displacement = (
                        sample["mean_positions"][variant]
                        - sample["mean_positions"][variant][0]
                    )[:, coordinate] * 1000
                    steps = np.arange(len(displacement))
                    trace_axis.plot(
                        steps,
                        displacement,
                        color=colors[variant],
                        linestyle=style,
                        alpha=1.0 - 0.22 * coordinate,
                        linewidth=1.6,
                        label=f"{variant} {coordinate_name}",
                    )
            trace_axis.axhline(0, color="#111827", linewidth=0.7)
            trace_axis.set_title(
                f"H{start + row_index + 1} seg={sample['segment_id']} "
                f"{sample['labels']}\n"
                f"atomic↔opposite displacement angle="
                f"{sample['direction']['atomic_opposite_angle_deg']:.1f}°"
            )
            trace_axis.set_xlabel("horizon step")
            trace_axis.set_ylabel("displacement from predicted start (mm)")
            trace_axis.grid(color="#e5e7eb", linewidth=0.6)
            if row_index == 0:
                trace_axis.legend(ncol=3, frameon=False, fontsize=8)

            x = np.arange(3)
            width = 0.34
            atomic_endpoint = sample["direction"]["atomic_endpoint_mm"]
            opposite_endpoint = sample["direction"]["opposite_endpoint_mm"]
            bar_axis.bar(
                x - width / 2, atomic_endpoint, width, color=colors["atomic"],
                label="atomic endpoint",
            )
            bar_axis.bar(
                x + width / 2, opposite_endpoint, width, color=colors["opposite"],
                label="opposite endpoint",
            )
            bar_axis.axhline(0, color="#111827", linewidth=0.7)
            bar_axis.set_xticks(x, ("x", "y", "z"))
            bar_axis.set_ylabel("endpoint displacement (mm)")
            bar_axis.set_title(
                "Expected-axis alignment: "
                f"atomic={sample['direction']['atomic_expected_cosine']:+.2f}, "
                f"opposite={sample['direction']['opposite_expected_cosine']:+.2f}"
            )
            bar_axis.grid(axis="y", color="#e5e7eb", linewidth=0.6)
            if row_index == 0:
                bar_axis.legend(frameon=False, fontsize=8)
            for axis in (trace_axis, bar_axis):
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
        figure.suptitle(
            "Direction test — trajectories are zeroed at their own predicted start "
            f"(page {page_index})",
            fontsize=15,
            fontweight="bold",
        )
        path = output_dir / f"prompt_direction_test_page{page_index}.png"
        figure.savefig(path, dpi=190, bbox_inches="tight")
        plt.close(figure)
        pages.append(str(path))
    return pages


def _matrix_to_rpy_deg(rotation: np.ndarray) -> np.ndarray:
    """Return base-frame ZYX roll, pitch, yaw angles in degrees."""
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1e-7:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def _world_rotation_vector_deg(
    start_rotation: np.ndarray, end_rotation: np.ndarray
) -> np.ndarray:
    """World/base-axis rotation vector for R_end R_start^T."""
    delta = end_rotation @ start_rotation.T
    cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)
    if abs(math.pi - angle) < 1e-5:
        eigenvalues, eigenvectors = np.linalg.eig(delta)
        axis = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
        axis /= np.linalg.norm(axis)
    else:
        axis = np.asarray(
            [
                delta[2, 1] - delta[1, 2],
                delta[0, 2] - delta[2, 0],
                delta[1, 0] - delta[0, 1],
            ]
        ) / (2.0 * math.sin(angle))
    return np.degrees(axis * angle)


def _draw_orientation(
    axis, position: np.ndarray, rotation: np.ndarray, length: float, alpha: float
) -> None:
    for index, color in enumerate(("#dc2626", "#16a34a", "#2563eb")):
        direction = rotation[:, index] * length
        axis.quiver(
            *position,
            *direction,
            color=color,
            linewidth=1.3,
            arrow_length_ratio=0.20,
            alpha=alpha,
        )


def _plot_pose_comparison(samples: list[dict], output_dir: Path) -> list[str]:
    """One readable figure per horizon: four poses plus world rotation bars."""
    paths = []
    variants = ("gt", "global", "atomic", "opposite")
    for sample_index, sample in enumerate(samples, 1):
        figure = plt.figure(figsize=(22, 5.4), constrained_layout=True)
        grid = figure.add_gridspec(1, 5, width_ratios=(1, 1, 1, 1, 1.25))
        all_positions = np.concatenate(
            [sample["poses"][variant][0] for variant in variants], axis=0
        )
        minimum = np.min(all_positions, axis=0)
        maximum = np.max(all_positions, axis=0)
        center = (minimum + maximum) / 2.0
        radius = max(float(np.max(maximum - minimum)) / 2.0, 0.025) * 1.15
        orientation_length = max(radius * 0.20, 0.008)
        rotation_vectors = {}
        for column, variant in enumerate(variants):
            axis = figure.add_subplot(grid[0, column], projection="3d")
            position, rotation = sample["poses"][variant]
            axis.plot(
                position[:, 0],
                position[:, 1],
                position[:, 2],
                color=VARIANT_COLORS[variant],
                linewidth=2.6,
            )
            axis.scatter(*position[0], color="#16a34a", s=42, label="start")
            axis.scatter(*position[-1], color="#dc2626", s=42, label="end")
            _draw_orientation(
                axis, position[0], rotation[0], orientation_length, alpha=0.45
            )
            _draw_orientation(
                axis, position[-1], rotation[-1], orientation_length, alpha=1.0
            )
            start_rpy = _matrix_to_rpy_deg(rotation[0])
            end_rpy = _matrix_to_rpy_deg(rotation[-1])
            rotation_vectors[variant] = _world_rotation_vector_deg(
                rotation[0], rotation[-1]
            )
            axis.set_xlim(center[0] - radius, center[0] + radius)
            axis.set_ylim(center[1] - radius, center[1] + radius)
            axis.set_zlim(center[2] - radius, center[2] + radius)
            axis.set_box_aspect((1, 1, 1))
            axis.set_xlabel("base x (m)")
            axis.set_ylabel("base y (m)")
            axis.set_zlabel("base z (m)")
            axis.set_title(
                f"{variant.upper()}\n"
                f"S RPY=({start_rpy[0]:.1f}, {start_rpy[1]:.1f}, "
                f"{start_rpy[2]:.1f})°\n"
                f"E RPY=({end_rpy[0]:.1f}, {end_rpy[1]:.1f}, "
                f"{end_rpy[2]:.1f})°",
                color=VARIANT_COLORS[variant],
                fontsize=10,
                fontweight="bold",
            )
            if column == 0:
                axis.legend(frameon=False, fontsize=8, loc="upper left")

        bar_axis = figure.add_subplot(grid[0, 4])
        x = np.arange(3)
        width = 0.19
        for variant_index, variant in enumerate(variants):
            offset = (variant_index - 1.5) * width
            bar_axis.bar(
                x + offset,
                rotation_vectors[variant],
                width,
                color=VARIANT_COLORS[variant],
                label=variant,
            )
        bar_axis.axhline(0, color="#111827", linewidth=0.8)
        bar_axis.set_xticks(x, ("Δrₓ", "Δrᵧ", "Δr_z"))
        bar_axis.set_ylabel("world/base rotation-vector component (degrees)")
        bar_axis.set_title(
            "Final incremental TCP rotation\n"
            r"$\log(R_{\rm end}R_{\rm start}^{T})$ in base axes",
            fontsize=11,
            fontweight="bold",
        )
        bar_axis.legend(frameon=False, fontsize=8, ncol=2)
        bar_axis.grid(axis="y", color="#e5e7eb", linewidth=0.6)
        bar_axis.spines["top"].set_visible(False)
        bar_axis.spines["right"].set_visible(False)
        figure.suptitle(
            f"Episode 485 · H{sample_index} · seg={sample['segment_id']} · "
            f"t={sample['timestamp_s']:.2f}s · {sample['labels']}\n"
            "TCP trajectory and orientation in base frame "
            "(paired-noise repeat 1; triad RGB = TCP xyz)",
            fontsize=15,
            fontweight="bold",
        )
        path = output_dir / f"base_pose_rotation_h{sample_index:02d}.png"
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        sample["world_rotation_delta_deg"] = {
            variant: rotation_vectors[variant].tolist() for variant in variants
        }
        paths.append(str(path))
    return paths


def _plot_right_action_dimensions(
    samples: list[dict], output_dir: Path
) -> list[str]:
    """Plot right J1-J7 in degrees and the native gripper channel."""
    paths = []
    variants = ("gt", "global", "atomic", "opposite")
    for sample_index, sample in enumerate(samples, 1):
        figure, axes = plt.subplots(
            4, 2, figsize=(15, 12), sharex=True, constrained_layout=True
        )
        steps = np.arange(51)
        for dimension, axis in enumerate(axes.flat):
            for variant in variants:
                values = sample["right_action_trajectories"][variant][:, dimension]
                if dimension < 7:
                    values = np.degrees(values)
                axis.plot(
                    steps,
                    values,
                    color=VARIANT_COLORS[variant],
                    linestyle=VARIANT_STYLES[variant],
                    linewidth=2.0 if variant == "gt" else 1.5,
                    label=variant,
                )
            axis.axvline(0, color="#6b7280", linewidth=0.7)
            axis.grid(color="#e5e7eb", linewidth=0.6)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if dimension < 7:
                axis.set_title(f"Right J{dimension + 1}")
                axis.set_ylabel("angle (degrees)")
            else:
                axis.set_title("Right gripper")
                axis.set_ylabel("native opening command")
            if dimension >= 6:
                axis.set_xlabel("horizon step (0 = current observed state)")
            if dimension == 0:
                axis.legend(frameon=False, ncol=4, fontsize=9)
        figure.suptitle(
            f"Episode 485 · H{sample_index} · seg={sample['segment_id']} · "
            f"t={sample['timestamp_s']:.2f}s · {sample['labels']}\n"
            "Right-arm 8-D action trajectory (paired-noise repeat 1)",
            fontsize=15,
            fontweight="bold",
        )
        path = output_dir / f"right_action_8d_h{sample_index:02d}.png"
        figure.savefig(path, dpi=190, bbox_inches="tight")
        plt.close(figure)
        paths.append(str(path))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=485)
    parser.add_argument("--horizons", type=int, default=10)
    parser.add_argument("--noise-repeats", type=int, default=3)
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path("assets/cr1_right_arm_tcp_red/urdf/CR1ARMR_TCP_RED.urdf"),
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = AtomicPi05Config(max_token_len=250, fast_action_ce_loss_weight=0.0)
    dataset = build_atomic_dataset(
        (args.dataset_root,),
        norm_assets_dir="/mnt/cunchu/zjq/target",
        norm_asset_id="openpi_norm_compact_accepted_v3",
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        include_fast=False,
    )
    selected = _select_horizons(dataset, args.episode, args.horizons)
    rows = [dataset[dataset_index] for _, dataset_index, _ in selected]
    batch = atomic_collate(rows)
    observation_np, _ = batch_to_observation(batch)
    observation = jax.tree.map(jnp.asarray, observation_np)
    tokenizer = _paligemma_tokenizer(config.max_token_len)

    raw = dataset._raw  # noqa: SLF001
    descriptions = []
    opposite_token_rows, opposite_mask_rows = [], []
    for segment_id, dataset_index, metadata in selected:
        weights = np.asarray(metadata["atomic_weights"])
        label_indices = np.flatnonzero(weights > 0)
        labels = tuple(ATOMIC_NAMES[index] for index in label_indices)
        opposite = _opposite_prompt(metadata["atomic_prompt"], labels)
        tokens, mask = tokenizer.tokenize(
            opposite, np.asarray(rows[len(descriptions)]["state"])
        )
        data_index = int(raw.base._visible_indices[dataset_index])  # noqa: SLF001
        episode_index = int(raw.base._episode_index[data_index])  # noqa: SLF001
        query_indices, _ = raw.base._get_query_indices(data_index, episode_index)  # noqa: SLF001
        action_indices = np.asarray(query_indices["action"], dtype=np.int64)
        descriptions.append(
            {
                "segment_id": segment_id,
                "dataset_index": dataset_index,
                "data_index": data_index,
                "timestamp_s": float(raw.base._timestamps[data_index]),  # noqa: SLF001
                "labels": "+".join(labels),
                "global_prompt": metadata["global_prompt"],
                "atomic_prompt": metadata["atomic_prompt"],
                "opposite_prompt": opposite,
                "raw_state": np.asarray(metadata["raw_state"]),
                "gt_actions": np.asarray(metadata["raw_actions"]),
                "action_indices": action_indices,
            }
        )
        opposite_token_rows.append(tokens)
        opposite_mask_rows.append(mask)

    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    normalize_back = _output_transform(args.dataset_root, config)
    fk = SimpleCR1FK(args.urdf)
    predictions = defaultdict(list)
    normalized_predictions = defaultdict(list)
    for repeat in range(args.noise_repeats):
        noise = jax.random.normal(
            jax.random.key(args.seed + repeat),
            (len(rows), config.action_horizon, config.action_dim),
        )
        sampled = _sample_three(
            model,
            observation,
            jnp.asarray(batch["atomic_prompt_tokens"]),
            jnp.asarray(batch["atomic_prompt_mask"]),
            jnp.asarray(np.stack(opposite_token_rows)),
            jnp.asarray(np.stack(opposite_mask_rows)),
            noise,
        )
        for variant, values in zip(
            ("global", "atomic", "opposite"), jax.device_get(sampled), strict=True
        ):
            normalized_predictions[variant].append(np.asarray(values))
            decoded = []
            for sample_index, prediction in enumerate(np.asarray(values)):
                output = normalize_back(
                    np.asarray(batch["state"][sample_index]),
                    descriptions[sample_index]["raw_state"],
                    prediction,
                )
                decoded.append(np.asarray(output["actions"]))
            predictions[variant].append(np.stack(decoded))
        print(f"sampled repeat {repeat + 1}/{args.noise_repeats}", flush=True)

    all_errors = {
        variant: {"translation_mm": [], "rotation_deg": [], "joint_deg": []}
        for variant in ("global", "atomic", "opposite")
    }
    plot_samples = []
    fk_validation_translation, fk_validation_rotation = [], []
    csv_rows = []
    for sample_index, description in enumerate(descriptions):
        gt_actions = description["gt_actions"]
        current_position, current_rotation = fk.trajectory(
            description["raw_state"][None, 8:15]
        )
        gt_future_position, gt_future_rotation = fk.trajectory(
            gt_actions[:, 8:15]
        )
        gt_plot_position = np.concatenate(
            [current_position, gt_future_position], axis=0
        )
        gt_plot_rotation = np.concatenate(
            [current_rotation, gt_future_rotation], axis=0
        )
        sidecar_rows = raw.tcp_pose[description["action_indices"], 12:]
        sidecar_position = sidecar_rows[:, :3]
        sidecar_rotation = sidecar_rows[:, 3:].reshape(-1, 3, 3)
        fk_validation_translation.extend(
            np.linalg.norm(gt_future_position - sidecar_position, axis=-1) * 1000
        )
        fk_validation_rotation.extend(
            _rotation_error_deg(gt_future_rotation, sidecar_rotation)
        )
        plot_entry = {
            **{
                key: description[key]
                for key in ("segment_id", "timestamp_s", "labels")
            },
            "poses": {"gt": (gt_plot_position, gt_plot_rotation)},
            "translation_error_mm": {},
            "rotation_error_deg": {},
            "mean_positions": {},
            "right_action_trajectories": {
                "gt": np.concatenate(
                    [
                        description["raw_state"][None, 8:16],
                        gt_actions[:, 8:16],
                    ],
                    axis=0,
                )
            },
        }
        for variant in ("global", "atomic", "opposite"):
            repeat_translation, repeat_rotation, repeat_joint = [], [], []
            first_pose = None
            repeat_positions = []
            for repeat_prediction in predictions[variant]:
                prediction = repeat_prediction[sample_index]
                future_position, future_rotation = fk.trajectory(
                    prediction[:, 8:15]
                )
                translation = (
                    np.linalg.norm(
                        future_position - gt_future_position, axis=-1
                    )
                    * 1000
                )
                rotation_error = _rotation_error_deg(
                    future_rotation, gt_future_rotation
                )
                joint_error = np.degrees(
                    np.mean(np.abs(prediction[:, 8:15] - gt_actions[:, 8:15]), axis=-1)
                )
                repeat_translation.append(translation)
                repeat_rotation.append(rotation_error)
                repeat_joint.append(joint_error)
                plot_position = np.concatenate(
                    [current_position, future_position], axis=0
                )
                plot_rotation = np.concatenate(
                    [current_rotation, future_rotation], axis=0
                )
                repeat_positions.append(plot_position)
                if first_pose is None:
                    first_pose = (plot_position, plot_rotation)
                    plot_entry["right_action_trajectories"][variant] = (
                        np.concatenate(
                            [
                                description["raw_state"][None, 8:16],
                                prediction[:, 8:16],
                            ],
                            axis=0,
                        )
                    )
            translation = np.mean(repeat_translation, axis=0)
            rotation_error = np.mean(repeat_rotation, axis=0)
            joint_error = np.mean(repeat_joint, axis=0)
            all_errors[variant]["translation_mm"].append(translation)
            all_errors[variant]["rotation_deg"].append(rotation_error)
            all_errors[variant]["joint_deg"].append(joint_error)
            plot_entry["poses"][variant] = first_pose
            plot_entry["mean_positions"][variant] = np.mean(
                repeat_positions, axis=0
            )
            plot_entry["translation_error_mm"][variant] = translation
            plot_entry["rotation_error_deg"][variant] = rotation_error
            csv_rows.append(
                {
                    "horizon": sample_index + 1,
                    "segment_id": description["segment_id"],
                    "timestamp_s": description["timestamp_s"],
                    "labels": description["labels"],
                    "variant": variant,
                    "translation_rmse_mm": float(
                        np.sqrt(np.mean(np.square(translation)))
                    ),
                    "translation_final_mm": float(translation[-1]),
                    "rotation_rmse_deg": float(
                        np.sqrt(np.mean(np.square(rotation_error)))
                    ),
                    "rotation_final_deg": float(rotation_error[-1]),
                    "joint_mae_deg": float(np.mean(joint_error)),
                }
            )
        expected = _translation_target(description["labels"])
        atomic_displacement = (
            plot_entry["mean_positions"]["atomic"][-1]
            - plot_entry["mean_positions"]["atomic"][0]
        )
        opposite_displacement = (
            plot_entry["mean_positions"]["opposite"][-1]
            - plot_entry["mean_positions"]["opposite"][0]
        )
        cosine = _cosine(atomic_displacement, opposite_displacement)
        plot_entry["direction"] = {
            "expected_translation_unit": expected.tolist(),
            "atomic_endpoint_mm": (atomic_displacement * 1000).tolist(),
            "opposite_endpoint_mm": (opposite_displacement * 1000).tolist(),
            "atomic_opposite_cosine": cosine,
            "atomic_opposite_angle_deg": float(
                np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            ),
            "atomic_expected_cosine": _cosine(atomic_displacement, expected),
            "opposite_expected_cosine": _cosine(
                opposite_displacement, -expected
            ),
        }
        plot_samples.append(plot_entry)

    metrics = {}
    for variant, values in all_errors.items():
        translation = np.concatenate(values["translation_mm"])
        rotation_error = np.concatenate(values["rotation_deg"])
        joint_error = np.concatenate(values["joint_deg"])
        metrics[variant] = {
            "translation_rmse_mm": float(
                np.sqrt(np.mean(np.square(translation)))
            ),
            "translation_mean_mm": float(np.mean(translation)),
            "translation_p90_mm": float(np.quantile(translation, 0.9)),
            "rotation_rmse_deg": float(
                np.sqrt(np.mean(np.square(rotation_error)))
            ),
            "rotation_mean_deg": float(np.mean(rotation_error)),
            "rotation_p90_deg": float(np.quantile(rotation_error, 0.9)),
            "joint_mae_deg": float(np.mean(joint_error)),
        }

    pages = _plot_pages(plot_samples, args.output_dir)
    direction_pages = _plot_direction_pages(plot_samples, args.output_dir)
    pose_comparison_figures = _plot_pose_comparison(
        plot_samples, args.output_dir
    )
    right_action_figures = _plot_right_action_dimensions(
        plot_samples, args.output_dir
    )
    direction_metrics = [sample["direction"] for sample in plot_samples]
    report = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "episode": args.episode,
        "horizon_count": len(selected),
        "action_horizon": config.action_horizon,
        "ode_steps": 10,
        "noise_repeats": args.noise_repeats,
        "paired_control": (
            "same image/state and initial flow noise for global, atomic and "
            "direction-inverted atomic prompts"
        ),
        "fk_contract": {
            "frame": "right_wrist_x_link",
            "tcp_local_z_offset_m": TCP_LOCAL_Z_OFFSET_M,
            "mount_xyz_m": list(default_mount_xyz("right")),
            "validation_max_translation_mm": float(
                np.max(fk_validation_translation)
            ),
            "validation_max_rotation_deg": float(np.max(fk_validation_rotation)),
        },
        "metrics": metrics,
        "direction_metrics": direction_metrics,
        "samples": [
            {
                key: value
                for key, value in description.items()
                if key
                in {
                    "segment_id",
                    "dataset_index",
                    "data_index",
                    "timestamp_s",
                    "labels",
                    "global_prompt",
                    "atomic_prompt",
                    "opposite_prompt",
                }
            }
            for description in descriptions
        ],
        "figures": pages,
        "direction_figures": direction_pages,
        "pose_comparison_figures": pose_comparison_figures,
        "right_action_figures": right_action_figures,
        "world_rotation_delta_deg": [
            sample["world_rotation_delta_deg"] for sample in plot_samples
        ],
    }
    report_path = args.output_dir / "fk_trajectory_ablation.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "action_pipeline_debug.npz",
        raw_state=np.stack(
            [description["raw_state"] for description in descriptions]
        ),
        raw_gt_actions=np.stack(
            [description["gt_actions"] for description in descriptions]
        ),
        normalized_state=np.asarray(batch["state"]),
        normalized_gt_actions=np.asarray(batch["actions"]),
        normalized_global=normalized_predictions["global"][0],
        normalized_atomic=normalized_predictions["atomic"][0],
        normalized_opposite=normalized_predictions["opposite"][0],
        decoded_global=predictions["global"][0],
        decoded_atomic=predictions["atomic"][0],
        decoded_opposite=predictions["opposite"][0],
    )
    with (args.output_dir / "fk_trajectory_ablation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
