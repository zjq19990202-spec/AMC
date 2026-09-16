#!/usr/bin/env python3
"""Measure whether an opposite atomic prompt reverses the generated TCP motion."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from openpi.models import model as _model

from atomic_latent_vla.atomic import ATOMIC_NAMES
from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import (
    _paligemma_tokenizer,
    atomic_collate,
    batch_to_observation,
    build_atomic_dataset,
)
from evaluate_zm_fk_trajectory_ablation import (
    SimpleCR1FK,
    _model as _restore_model,
    _output_transform,
    _sample_three,
    _select_horizons,
    _world_rotation_vector_deg,
)


def _swap_words(text: str, left: str, right: str, token: str) -> str:
    text = re.sub(rf"\b{re.escape(left)}\b", token, text, flags=re.IGNORECASE)
    text = re.sub(rf"\b{re.escape(right)}\b", left, text, flags=re.IGNORECASE)
    return text.replace(token, right)


def _semantic_opposite(prompt: str, labels: tuple[str, ...]) -> str:
    """Invert only label-relevant direction words while preserving task context."""
    text = prompt.replace("Right arm", "__RIGHT_ARM__")
    label_set = set(labels)
    if any(label.startswith("move_x_") for label in label_set):
        text = _swap_words(text, "forward", "backward", "__SWAP_X__")
    if any(label.startswith("move_y_") for label in label_set):
        text = _swap_words(text, "leftward", "rightward", "__SWAP_Y_ADV__")
        text = _swap_words(text, "left", "right", "__SWAP_Y__")
    if any(label.startswith("move_z_") for label in label_set):
        text = _swap_words(text, "upward", "downward", "__SWAP_Z_LONG__")
        text = _swap_words(text, "up", "down", "__SWAP_Z_SHORT__")
    if any(label.startswith("rotate_") for label in label_set):
        text = _swap_words(
            text, "positively", "negatively", "__SWAP_ROT_ADV__"
        )
        text = _swap_words(text, "positive", "negative", "__SWAP_ROT_ADJ__")
        text = _swap_words(
            text, "clockwise", "counterclockwise", "__SWAP_ROT_CLOCK__"
        )
    text = text.replace("__RIGHT_ARM__", "Right arm")
    if text == prompt:
        raise ValueError(
            f"could not invert label-relevant words for {labels}: {prompt}"
        )
    return text


_OPPOSITE_LABEL = {
    "move_x_pos": "move_x_neg",
    "move_x_neg": "move_x_pos",
    "move_y_pos": "move_y_neg",
    "move_y_neg": "move_y_pos",
    "move_z_pos": "move_z_neg",
    "move_z_neg": "move_z_pos",
    "rotate_x_pos": "rotate_x_neg",
    "rotate_x_neg": "rotate_x_pos",
    "rotate_y_pos": "rotate_y_neg",
    "rotate_y_neg": "rotate_y_pos",
    "rotate_z_pos": "rotate_z_neg",
    "rotate_z_neg": "rotate_z_pos",
}
_AXIS_PHRASE = {
    "move_x_pos": "move forward along base-frame +x",
    "move_x_neg": "move backward along base-frame -x",
    "move_y_pos": "move left along base-frame +y",
    "move_y_neg": "move right along base-frame -y",
    "move_z_pos": "move upward along base-frame +z",
    "move_z_neg": "move downward along base-frame -z",
    "rotate_x_pos": "rotate positively about the base-frame x axis",
    "rotate_x_neg": "rotate negatively about the base-frame x axis",
    "rotate_y_pos": "rotate positively about the base-frame y axis",
    "rotate_y_neg": "rotate negatively about the base-frame y axis",
    "rotate_z_pos": "rotate positively about the base-frame z axis",
    "rotate_z_neg": "rotate negatively about the base-frame z axis",
}


def _axis_prompt(labels: tuple[str, ...]) -> str:
    phrases = [_AXIS_PHRASE[label] for label in labels]
    return "Command the right-arm TCP to " + " and ".join(phrases) + "."


def _task_object(prompt: str) -> str:
    """Extract a stable task object without retaining directional relations."""
    lowered = prompt.lower()
    for needle, object_name in (
        ("circuit breaker", "circuit breaker"),
        ("blue lever", "blue lever"),
        ("door handle", "black door handle"),
        ("handle", "black door handle"),
        ("on button", "red ON button"),
        ("button", "red ON button"),
        ("switch", "ON/OFF switch"),
    ):
        if needle in lowered:
            return object_name
    return "task-relevant object visible in the scene"


def _object_axis_prompt(
    labels: tuple[str, ...], original_prompt: str
) -> str:
    phrases = [_AXIS_PHRASE[label] for label in labels]
    task_object = _task_object(original_prompt)
    return (
        f"To continue manipulating the {task_object}, command the right-arm TCP "
        f"to {' and '.join(phrases)}. Target a waypoint displaced or oriented "
        "from the current TCP pose in exactly those base-frame directions."
    )


def _reversal(atomic: np.ndarray, opposite: np.ndarray) -> dict[str, float]:
    atomic_norm = float(np.linalg.norm(atomic))
    opposite_norm = float(np.linalg.norm(opposite))
    if atomic_norm < 1e-8 or opposite_norm < 1e-8:
        return {
            "atomic_magnitude": atomic_norm,
            "opposite_magnitude": opposite_norm,
            "reverse_score": float("nan"),
            "reverse_gain": float("nan"),
        }
    cosine = float(np.dot(atomic, opposite) / (atomic_norm * opposite_norm))
    return {
        "atomic_magnitude": atomic_norm,
        "opposite_magnitude": opposite_norm,
        "reverse_score": -cosine,
        "reverse_gain": float(-np.dot(opposite, atomic) / np.dot(atomic, atomic)),
    }


def _summary(rows: list[dict], prefix: str, valid_threshold: float) -> dict:
    valid = [
        row for row in rows
        if row[f"{prefix}_atomic_magnitude"] >= valid_threshold
        and np.isfinite(row[f"{prefix}_reverse_score"])
    ]
    scores = np.asarray([row[f"{prefix}_reverse_score"] for row in valid])
    gains = np.asarray([row[f"{prefix}_reverse_gain"] for row in valid])
    strong = (scores >= 0.5) & (gains >= 0.2)
    return {
        "valid_samples": len(valid),
        "mean_reverse_score": float(np.mean(scores)) if len(valid) else None,
        "median_reverse_score": float(np.median(scores)) if len(valid) else None,
        "mean_reverse_gain": float(np.mean(gains)) if len(valid) else None,
        "median_reverse_gain": float(np.median(gains)) if len(valid) else None,
        "strong_reverse_rate": float(np.mean(strong)) if len(valid) else None,
        "definition": "strong means reverse_score>=0.5 and reverse_gain>=0.2",
    }


def _target_axis_summary(rows: list[dict]) -> dict:
    """Score only the motion component named by each atomic label."""
    result = {}
    for family, prefix, threshold in (
        ("move", "translation", 0.005),
        ("rotate", "rotation", 2.0),
    ):
        components = []
        for row in rows:
            scored_labels = row.get("flipped_labels", row["labels"])
            for label in scored_labels.split("+"):
                if not label.startswith(f"{family}_"):
                    continue
                _, axis, _ = label.split("_")
                suffix = axis if family == "move" else f"{axis}_deg"
                atomic = float(row[f"{prefix}_atomic_{suffix}"])
                opposite = float(row[f"{prefix}_opposite_{suffix}"])
                if abs(atomic) < threshold:
                    continue
                gain = -opposite / atomic
                components.append(
                    {
                        "episode": row["episode"],
                        "episode_horizon": row["episode_horizon"],
                        "label": label,
                        "atomic_component": atomic,
                        "opposite_component": opposite,
                        "reverse_gain": gain,
                        "sign_flipped": atomic * opposite < 0.0,
                    }
                )
        gains = np.asarray([item["reverse_gain"] for item in components])
        flipped = np.asarray(
            [item["sign_flipped"] for item in components], dtype=np.bool_
        )
        reversed_gains = gains[flipped]
        result[family] = {
            "valid_components": len(components),
            "sign_flip_rate": float(np.mean(flipped)) if len(flipped) else None,
            "strong_reverse_rate": (
                float(np.mean(flipped & (gains >= 0.2))) if len(flipped) else None
            ),
            "median_gain_among_flipped": (
                float(np.median(reversed_gains)) if len(reversed_gains) else None
            ),
            "threshold": threshold,
            "components": components,
        }
    return result


def _select_state_cluster(
    dataset,
    center: np.ndarray,
    radius: float,
    label_sets: tuple[tuple[str, ...], ...],
    samples_per_label: int,
    *,
    macro_translation_min_m: float | None = None,
    macro_rotation_min_rad: float | None = None,
):
    """Select tightly state-matched horizons, stratified by atomic label set."""
    raw = dataset._raw  # noqa: SLF001
    raw._ensure_annotations()  # noqa: SLF001
    groups = {labels: [] for labels in label_sets}
    for dataset_index, data_index in enumerate(raw.base._visible_indices):  # noqa: SLF001
        if int(raw._type[data_index]) <= 0:  # noqa: SLF001
            continue
        state = np.asarray(raw.base._states[data_index])[8:15]  # noqa: SLF001
        distance = float(np.sqrt(np.mean(np.square(state - center))))
        if distance > radius:
            continue
        # The deployed training loader may aggregate the full future horizon,
        # so validate the actual model-facing metadata after the cheap state
        # filter rather than assuming the current row annotation is sufficient.
        metadata = raw.metadata(dataset_index)
        if not metadata["atomic_supervision_mask"]:
            continue
        weights = np.asarray(metadata["atomic_weights"])
        labels = tuple(ATOMIC_NAMES[index] for index in np.flatnonzero(weights > 0))
        if labels not in groups:
            continue
        if macro_translation_min_m is not None:
            if macro_rotation_min_rad is None:
                raise ValueError(
                    "macro_rotation_min_rad is required with macro consistency"
                )
            macro_delta = np.asarray(metadata["tcp_twist_delta"])[-1]
            consistent = True
            for label_index in np.flatnonzero(weights > 0):
                axis = int(label_index) // 2
                expected_positive = int(label_index) % 2 == 0
                value = float(macro_delta[axis])
                threshold = (
                    macro_translation_min_m
                    if axis < 3
                    else macro_rotation_min_rad
                )
                consistent &= (
                    (value >= 0.0) == expected_positive
                    and abs(value) >= threshold
                )
            if not consistent:
                continue
        episode = int(raw.base._episode_index[data_index])  # noqa: SLF001
        segment_id = int(raw._segment_id[data_index])  # noqa: SLF001
        groups[labels].append(
            (distance, episode, segment_id, dataset_index, metadata)
        )

    selected = []
    for labels in label_sets:
        candidates = sorted(groups[labels])
        # Prefer different episodes; state distance is the tie-breaker.
        by_episode = {}
        for candidate in candidates:
            by_episode.setdefault(candidate[1], candidate)
        episode_distinct = sorted(by_episode.values())
        # Preserve episode diversity first, then fill from additional horizons
        # when a rare atomic combination only occurs in one/few episodes.
        chosen = episode_distinct[:samples_per_label]
        chosen_indices = {candidate[3] for candidate in chosen}
        chosen.extend(
            candidate
            for candidate in candidates
            if candidate[3] not in chosen_indices
        )
        chosen = chosen[:samples_per_label]
        if len(chosen) < samples_per_label:
            raise RuntimeError(
                f"state cluster has only {len(chosen)} matching "
                f"samples for {labels}, need {samples_per_label}"
            )
        for _, _, segment_id, dataset_index, metadata in chosen:
            selected.append((segment_id, dataset_index, metadata))
    return selected


def _plot(rows: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    x = np.arange(len(rows))
    episode_labels = [f"E{row['episode']}H{row['episode_horizon']}" for row in rows]
    for column, prefix in enumerate(("translation", "rotation")):
        score_axis = axes[0, column]
        gain_axis = axes[1, column]
        scores = [row[f"{prefix}_reverse_score"] for row in rows]
        gains = [row[f"{prefix}_reverse_gain"] for row in rows]
        colors = [
            "#16a34a" if score >= 0.5 and gain >= 0.2 else "#dc2626"
            for score, gain in zip(scores, gains, strict=True)
        ]
        score_axis.bar(x, scores, color=colors)
        score_axis.axhline(0, color="#111827", linewidth=0.8)
        score_axis.axhline(0.5, color="#16a34a", linestyle="--", linewidth=1.0)
        score_axis.set_ylim(-1.05, 1.05)
        score_axis.set_ylabel(r"$-\cos(\Delta_{\rm atomic},\Delta_{\rm opposite})$")
        score_axis.set_title(f"{prefix.title()} reversal direction score")
        gain_axis.bar(x, np.clip(gains, -2, 2), color=colors)
        gain_axis.axhline(0, color="#111827", linewidth=0.8)
        gain_axis.axhline(1, color="#16a34a", linestyle="--", linewidth=1.0)
        gain_axis.set_ylabel("opposite projection / atomic magnitude")
        gain_axis.set_title(f"{prefix.title()} signed reversal gain (clipped to ±2)")
        for axis in (score_axis, gain_axis):
            axis.set_xticks(x, episode_labels, rotation=70, fontsize=7)
            axis.grid(axis="y", color="#e5e7eb", linewidth=0.6)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
    figure.suptitle(
        "Atomic vs explicit opposite-prompt reversal across episodes\n"
        "green = direction score ≥0.5 and reverse gain ≥0.2",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_tcp_pages(plot_rows: list[dict], output_dir: Path) -> list[str]:
    """Plot TCP trajectories and endpoint translation/rotation bars only."""
    paths = []
    colors = {"atomic": "#ea580c", "opposite": "#be185d"}
    for episode in sorted({row["episode"] for row in plot_rows}):
        episode_rows = [row for row in plot_rows if row["episode"] == episode]
        figure = plt.figure(
            figsize=(16, 4.2 * len(episode_rows)), constrained_layout=True
        )
        grid = figure.add_gridspec(len(episode_rows), 3, width_ratios=(1.25, 1, 1))
        for row_index, row in enumerate(episode_rows):
            trajectory_axis = figure.add_subplot(
                grid[row_index, 0], projection="3d"
            )
            all_positions = np.concatenate(
                [row["positions"][variant] for variant in ("atomic", "opposite")]
            )
            for variant in ("atomic", "opposite"):
                positions = row["positions"][variant]
                trajectory_axis.plot(
                    positions[:, 0],
                    positions[:, 1],
                    positions[:, 2],
                    color=colors[variant],
                    linewidth=2.3,
                    label=variant,
                )
                trajectory_axis.scatter(
                    *positions[-1], color=colors[variant], s=30
                )
            trajectory_axis.scatter(
                *row["current_position"], color="#16a34a", s=36, label="start"
            )
            low, high = all_positions.min(axis=0), all_positions.max(axis=0)
            center = (low + high) / 2.0
            radius = max(float(np.max(high - low)) / 2.0, 0.025) * 1.15
            trajectory_axis.set_xlim(center[0] - radius, center[0] + radius)
            trajectory_axis.set_ylim(center[1] - radius, center[1] + radius)
            trajectory_axis.set_zlim(center[2] - radius, center[2] + radius)
            trajectory_axis.set_box_aspect((1, 1, 1))
            trajectory_axis.set_xlabel("base x (m)")
            trajectory_axis.set_ylabel("base y (m)")
            trajectory_axis.set_zlabel("base z (m)")
            trajectory_axis.set_title(
                f"H{row['episode_horizon']} · {row['labels']}\nTCP trajectory"
            )
            trajectory_axis.legend(frameon=False, fontsize=8)

            x = np.arange(3)
            width = 0.34
            translation_axis = figure.add_subplot(grid[row_index, 1])
            rotation_axis = figure.add_subplot(grid[row_index, 2])
            for variant_index, variant in enumerate(("atomic", "opposite")):
                offset = (variant_index - 0.5) * width
                translation_axis.bar(
                    x + offset,
                    row["translation_vectors"][variant] * 1000.0,
                    width,
                    color=colors[variant],
                    label=variant,
                )
                rotation_axis.bar(
                    x + offset,
                    row["rotation_vectors"][variant],
                    width,
                    color=colors[variant],
                    label=variant,
                )
            for axis, ylabel, title in (
                (
                    translation_axis,
                    "endpoint displacement (mm)",
                    "Final TCP translation",
                ),
                (
                    rotation_axis,
                    "axis-angle component (deg)",
                    r"Final TCP rotation: $\log(R_eR_s^T)$",
                ),
            ):
                axis.axhline(0.0, color="#111827", linewidth=0.8)
                axis.set_xticks(x, ("x", "y", "z"))
                axis.set_ylabel(ylabel)
                axis.set_title(title)
                axis.grid(axis="y", color="#e5e7eb", linewidth=0.6)
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                axis.legend(frameon=False, fontsize=8)
        figure.suptitle(
            f"Episode {episode}: opposite-prompt TCP response",
            fontsize=16,
            fontweight="bold",
        )
        path = output_dir / f"tcp_reversal_episode_{episode:06d}.png"
        figure.savefig(path, dpi=190, bbox_inches="tight")
        plt.close(figure)
        paths.append(str(path))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=[],
        help="episode ids for the ordinary selector; omitted with --state-center",
    )
    parser.add_argument("--horizons-per-episode", type=int, default=4)
    parser.add_argument("--noise-repeats", type=int, default=3)
    parser.add_argument(
        "--prompt-mode",
        choices=("semantic", "axis", "object_axis"),
        default="semantic",
        help="semantic preserves task wording; axis uses target-free base-axis commands",
    )
    parser.add_argument(
        "--flip-scope",
        choices=("all", "first"),
        default="all",
        help="invert every atomic direction or only the first labelled component",
    )
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--state-center",
        type=float,
        nargs=7,
        help="select right-arm horizons within RMS radius of this 7-D raw qpos",
    )
    parser.add_argument("--state-rms-radius", type=float, default=0.03)
    parser.add_argument("--state-samples-per-label", type=int, default=8)
    parser.add_argument(
        "--macro-consistent",
        action="store_true",
        help="require every atomic label to match the final 50-step GT TCP delta",
    )
    parser.add_argument("--macro-translation-min-m", type=float, default=0.01)
    parser.add_argument(
        "--macro-rotation-min-deg", type=float, default=2.0
    )
    parser.add_argument(
        "--state-label-set",
        action="append",
        default=[],
        help="plus-separated exact atomic label set; may be repeated",
    )
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
    selections = []
    if args.state_center is not None:
        if not args.state_label_set:
            raise ValueError("--state-center requires at least one --state-label-set")
        label_sets = tuple(
            tuple(value.split("+")) for value in args.state_label_set
        )
        state_rows = _select_state_cluster(
            dataset,
            np.asarray(args.state_center, dtype=np.float32),
            args.state_rms_radius,
            label_sets,
            args.state_samples_per_label,
            macro_translation_min_m=(
                args.macro_translation_min_m if args.macro_consistent else None
            ),
            macro_rotation_min_rad=(
                np.deg2rad(args.macro_rotation_min_deg)
                if args.macro_consistent
                else None
            ),
        )
        raw = dataset._raw  # noqa: SLF001
        for episode_horizon, (segment_id, dataset_index, metadata) in enumerate(
            state_rows, 1
        ):
            data_index = int(raw.base._visible_indices[dataset_index])  # noqa: SLF001
            episode = int(raw.base._episode_index[data_index])  # noqa: SLF001
            selections.append(
                (episode, episode_horizon, segment_id, dataset_index, metadata)
            )
    else:
        if not args.episodes:
            raise ValueError("provide --episodes or use --state-center")
        for episode in args.episodes:
            for episode_horizon, (segment_id, dataset_index, metadata) in enumerate(
                _select_horizons(dataset, episode, args.horizons_per_episode), 1
            ):
                selections.append(
                    (episode, episode_horizon, segment_id, dataset_index, metadata)
                )
    samples = [dataset[item[3]] for item in selections]
    batch = atomic_collate(samples)
    observation_np, _ = batch_to_observation(batch)
    observation = jax.tree.map(jnp.asarray, observation_np)
    tokenizer = _paligemma_tokenizer(config.max_token_len)
    descriptions = []
    atomic_tokens, atomic_masks = [], []
    opposite_tokens, opposite_masks = [], []
    for sample_index, (episode, episode_horizon, segment_id, dataset_index, metadata) in enumerate(
        selections
    ):
        weights = np.asarray(metadata["atomic_weights"])
        labels = tuple(ATOMIC_NAMES[index] for index in np.flatnonzero(weights > 0))
        flipped_labels = labels if args.flip_scope == "all" else labels[:1]
        counterfactual_labels = tuple(
            _OPPOSITE_LABEL[label] if label in flipped_labels else label
            for label in labels
        )
        if args.prompt_mode == "axis":
            atomic_prompt = _axis_prompt(labels)
            opposite_prompt = _axis_prompt(counterfactual_labels)
        elif args.prompt_mode == "object_axis":
            atomic_prompt = _object_axis_prompt(
                labels, metadata["atomic_prompt"]
            )
            opposite_prompt = _object_axis_prompt(
                counterfactual_labels,
                metadata["atomic_prompt"],
            )
        else:
            atomic_prompt = metadata["atomic_prompt"]
            opposite_prompt = _semantic_opposite(atomic_prompt, flipped_labels)
        atomic_token, atomic_mask = tokenizer.tokenize(
            atomic_prompt, np.asarray(samples[sample_index]["state"])
        )
        opposite_token, opposite_mask = tokenizer.tokenize(
            opposite_prompt, np.asarray(samples[sample_index]["state"])
        )
        descriptions.append(
            {
                "episode": episode,
                "episode_horizon": episode_horizon,
                "segment_id": segment_id,
                "labels": "+".join(labels),
                "flipped_labels": "+".join(flipped_labels),
                "atomic_prompt": atomic_prompt,
                "opposite_prompt": opposite_prompt,
                "raw_state": np.asarray(metadata["raw_state"]),
            }
        )
        atomic_tokens.append(atomic_token)
        atomic_masks.append(atomic_mask)
        opposite_tokens.append(opposite_token)
        opposite_masks.append(opposite_mask)

    params = _restore_model.restore_params(
        args.checkpoint / "params", dtype=jnp.bfloat16
    )
    model = config.load(params)
    model.eval()
    decode = _output_transform(args.dataset_root, config)
    fk = SimpleCR1FK(args.urdf)
    decoded = {variant: [] for variant in ("atomic", "opposite")}
    for repeat in range(args.noise_repeats):
        noise = jax.random.normal(
            jax.random.key(args.seed + repeat),
            (len(samples), config.action_horizon, config.action_dim),
        )
        _, atomic_values, opposite_values = jax.device_get(
            _sample_three(
                model,
                observation,
                jnp.asarray(np.stack(atomic_tokens)),
                jnp.asarray(np.stack(atomic_masks)),
                jnp.asarray(np.stack(opposite_tokens)),
                jnp.asarray(np.stack(opposite_masks)),
                noise,
            )
        )
        for variant, values in (
            ("atomic", atomic_values),
            ("opposite", opposite_values),
        ):
            rows = []
            for index, prediction in enumerate(np.asarray(values)):
                rows.append(
                    decode(
                        np.asarray(batch["state"][index]),
                        descriptions[index]["raw_state"],
                        prediction,
                    )["actions"]
                )
            decoded[variant].append(np.stack(rows))
        print(f"sampled repeat {repeat + 1}/{args.noise_repeats}", flush=True)

    result_rows = []
    plot_rows = []
    for index, description in enumerate(descriptions):
        _, current_rotation_rows = fk.trajectory(
            description["raw_state"][None, 8:15]
        )
        current_position_rows, _ = fk.trajectory(
            description["raw_state"][None, 8:15]
        )
        current_position = current_position_rows[0]
        current_rotation = current_rotation_rows[0]
        endpoint_positions = {}
        endpoint_rotations = {}
        plot_positions = {}
        for variant in ("atomic", "opposite"):
            positions, rotations = [], []
            for repeat_values in decoded[variant]:
                position_rows, rotation_rows = fk.trajectory(
                    repeat_values[index, :, 8:15]
                )
                positions.append(position_rows[-1])
                rotations.append(
                    _world_rotation_vector_deg(
                        current_rotation, rotation_rows[-1]
                    )
                )
                if variant not in plot_positions:
                    plot_positions[variant] = np.concatenate(
                        [current_position[None], position_rows], axis=0
                    )
            endpoint_positions[variant] = np.mean(positions, axis=0)
            endpoint_rotations[variant] = np.mean(rotations, axis=0)
        translation = _reversal(
            endpoint_positions["atomic"] - current_position,
            endpoint_positions["opposite"] - current_position,
        )
        translation_atomic_vector = endpoint_positions["atomic"] - current_position
        translation_opposite_vector = endpoint_positions["opposite"] - current_position
        rotation = _reversal(
            endpoint_rotations["atomic"], endpoint_rotations["opposite"]
        )
        row = {**description}
        row.pop("raw_state")
        for axis, axis_name in enumerate(("x", "y", "z")):
            row[f"translation_atomic_{axis_name}"] = float(
                translation_atomic_vector[axis]
            )
            row[f"translation_opposite_{axis_name}"] = float(
                translation_opposite_vector[axis]
            )
            row[f"rotation_atomic_{axis_name}_deg"] = float(
                endpoint_rotations["atomic"][axis]
            )
            row[f"rotation_opposite_{axis_name}_deg"] = float(
                endpoint_rotations["opposite"][axis]
            )
        for prefix, values in (("translation", translation), ("rotation", rotation)):
            for key, value in values.items():
                row[f"{prefix}_{key}"] = value
        result_rows.append(row)
        plot_rows.append(
            {
                "episode": row["episode"],
                "episode_horizon": row["episode_horizon"],
                "labels": row["labels"],
                "current_position": current_position,
                "positions": plot_positions,
                "translation_vectors": {
                    variant: endpoint_positions[variant] - current_position
                    for variant in ("atomic", "opposite")
                },
                "rotation_vectors": endpoint_rotations,
            }
        )

    summary = {
        "episodes": args.episodes,
        "horizons_per_episode": args.horizons_per_episode,
        "samples": len(result_rows),
        "noise_repeats": args.noise_repeats,
        "translation": _summary(result_rows, "translation", 0.005),
        "rotation": _summary(result_rows, "rotation", 2.0),
        "target_axes": _target_axis_summary(result_rows),
        "rows": result_rows,
    }
    if args.state_center is not None:
        summary["state_cluster"] = {
            "center_right_qpos": args.state_center,
            "rms_radius": args.state_rms_radius,
            "label_sets": args.state_label_set,
            "samples_per_label": args.state_samples_per_label,
        }
    (args.output_dir / "prompt_reversal_multi_episode.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "prompt_reversal_multi_episode.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    _plot(result_rows, args.output_dir / "prompt_reversal_multi_episode.png")
    _plot_tcp_pages(plot_rows, args.output_dir)
    print(json.dumps({key: summary[key] for key in ("translation", "rotation")}, indent=2))


if __name__ == "__main__":
    main()
