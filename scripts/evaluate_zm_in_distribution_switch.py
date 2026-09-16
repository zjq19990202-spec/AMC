#!/usr/bin/env python3
"""Test whether zM can switch between state-supported atomic motion modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flax import nnx
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
from evaluate_prompt_reversal_multi_episode import (
    _object_axis_prompt,
    _select_state_cluster,
)
from evaluate_zm_fk_trajectory_ablation import (
    SimpleCR1FK,
    _model as _restore_model,
    _output_transform,
    _world_rotation_vector_deg,
)


MODE_LABELS = (
    ("move_x_neg", "rotate_z_neg"),
    ("move_z_neg", "rotate_z_neg"),
    ("rotate_x_neg",),
)
SHORT_ATOM_NAMES = {
    "move_x_pos": "x+",
    "move_x_neg": "x-",
    "move_y_pos": "y+",
    "move_y_neg": "y-",
    "move_z_pos": "z+",
    "move_z_neg": "z-",
    "rotate_x_pos": "rx+",
    "rotate_x_neg": "rx-",
    "rotate_y_pos": "ry+",
    "rotate_y_neg": "ry-",
    "rotate_z_pos": "rz+",
    "rotate_z_neg": "rz-",
}


def _mode_name(labels: tuple[str, ...]) -> str:
    return " + ".join(SHORT_ATOM_NAMES[label] for label in labels)


@nnx.jit
def _sample_pair_with_z(
    model,
    observation,
    source_tokens,
    source_mask,
    target_tokens,
    target_mask,
    noise,
):
    observation = _model.preprocess_observation(None, observation, train=False)
    source_observation = model._with_prompt(  # noqa: SLF001
        observation, source_tokens, source_mask
    )
    target_observation = model._with_prompt(  # noqa: SLF001
        observation, target_tokens, target_mask
    )
    noise = model._mask_action_condition(noise)  # noqa: SLF001
    active_state = model._controlled_state(observation.state)  # noqa: SLF001

    def encode(prompted_observation):
        query_hidden, prefix_mask, kv_cache = model._prefix_forward(  # noqa: SLF001
            prompted_observation
        )
        _, _, z_model, _, _ = model._latent(  # noqa: SLF001
            query_hidden, active_state
        )
        return prefix_mask, kv_cache, z_model

    def sample(prefix_mask, kv_cache, z_model):
        def step(index, actions):
            time = jnp.asarray(1.0 - index / 10.0, dtype=actions.dtype)
            velocity = model._suffix_velocity(  # noqa: SLF001
                prefix_mask,
                kv_cache,
                actions,
                jnp.broadcast_to(time, (actions.shape[0],)),
                z_model,
            )
            return model._mask_action_condition(  # noqa: SLF001
                actions - 0.1 * velocity
            )

        return jax.lax.fori_loop(0, 10, step, noise)[
            ..., : model.config.active_action_dim
        ]

    source_prefix, source_cache, source_z = encode(source_observation)
    target_prefix, target_cache, target_z = encode(target_observation)
    return (
        sample(source_prefix, source_cache, source_z),
        sample(target_prefix, target_cache, target_z),
        source_z,
        target_z,
    )


def _labels(metadata: dict) -> tuple[str, ...]:
    weights = np.asarray(metadata["atomic_weights"])
    return tuple(ATOMIC_NAMES[index] for index in np.flatnonzero(weights > 0))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 1.0


def _feature(
    fk: SimpleCR1FK,
    raw_state: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    current_position, current_rotation = fk.trajectory(raw_state[None, 8:15])
    positions, rotations = fk.trajectory(actions[:, 8:15])
    translation_mm = (positions[-1] - current_position[0]) * 1000.0
    rotation_deg = _world_rotation_vector_deg(
        current_rotation[0], rotations[-1]
    )
    # Five millimetres and two degrees are the prior evaluation's meaningful
    # component thresholds, so this feature has comparable dimensionless axes.
    return np.concatenate([translation_mm / 5.0, rotation_deg / 2.0])


def _plot(
    summary: dict, path: Path, mode_labels: tuple[tuple[str, ...], ...]
) -> None:
    names = [_mode_name(labels) for labels in mode_labels]
    fields = (
        ("target_classification_rate", "Counterfactual classified as target"),
        ("target_approach_rate", "Counterfactual moves toward target prototype"),
        ("mean_z_cosine", "source/counterfactual zM cosine"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for axis, (field, title) in zip(axes, fields, strict=True):
        size = len(names)
        matrix = np.full((size, size), np.nan)
        for row in summary["switches"]:
            i = names.index(row["source"])
            j = names.index(row["target"])
            matrix[i, j] = row[field]
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
        axis.set_xticks(range(size), names, rotation=25, ha="right")
        axis.set_yticks(range(size), names)
        axis.set_xlabel("target prompt mode")
        axis.set_ylabel("source mode")
        axis.set_title(title)
        for i in range(size):
            for j in range(size):
                if np.isfinite(matrix[i, j]):
                    axis.text(
                        j,
                        i,
                        f"{matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white" if matrix[i, j] < 0.65 else "black",
                        fontweight="bold",
                    )
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle(
        "Macro-consistent state cluster: in-distribution zM prompt switching",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--state-center", type=float, nargs=7, required=True)
    parser.add_argument("--state-rms-radius", type=float, default=0.03)
    parser.add_argument("--samples-per-mode", type=int, default=8)
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        help="plus-separated exact atomic label set; repeat for each mode",
    )
    parser.add_argument("--macro-translation-min-m", type=float, default=0.01)
    parser.add_argument("--macro-rotation-min-deg", type=float, default=2.0)
    parser.add_argument("--noise-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mode_labels = (
        tuple(tuple(value.split("+")) for value in args.mode)
        if args.mode
        else MODE_LABELS
    )
    if len(mode_labels) < 2:
        raise ValueError("in-distribution switching requires at least two modes")

    config = AtomicPi05Config(max_token_len=250, fast_action_ce_loss_weight=0.0)
    dataset = build_atomic_dataset(
        (args.dataset_root,),
        norm_assets_dir="/mnt/cunchu/zjq/target",
        norm_asset_id="openpi_norm_compact_accepted_v3",
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        include_fast=False,
    )
    selected = _select_state_cluster(
        dataset,
        np.asarray(args.state_center, dtype=np.float32),
        args.state_rms_radius,
        mode_labels,
        args.samples_per_mode,
        macro_translation_min_m=args.macro_translation_min_m,
        macro_rotation_min_rad=np.deg2rad(args.macro_rotation_min_deg),
    )

    comparisons = []
    for segment_id, dataset_index, metadata in selected:
        source = _labels(metadata)
        for target in mode_labels:
            if target == source:
                continue
            comparisons.append(
                {
                    "segment_id": segment_id,
                    "dataset_index": dataset_index,
                    "metadata": metadata,
                    "source": source,
                    "target": target,
                }
            )

    samples = [dataset[row["dataset_index"]] for row in comparisons]
    batch = atomic_collate(samples)
    observation_np, _ = batch_to_observation(batch)
    observation = jax.tree.map(jnp.asarray, observation_np)
    tokenizer = _paligemma_tokenizer(config.max_token_len)
    source_tokens, source_masks, target_tokens, target_masks = [], [], [], []
    for sample, row in zip(samples, comparisons, strict=True):
        state = np.asarray(sample["state"])[:16]
        original = row["metadata"]["atomic_prompt"]
        source_prompt = _object_axis_prompt(row["source"], original)
        target_prompt = _object_axis_prompt(row["target"], original)
        source_token, source_mask = tokenizer.tokenize(source_prompt, state)
        target_token, target_mask = tokenizer.tokenize(target_prompt, state)
        row["source_prompt"] = source_prompt
        row["target_prompt"] = target_prompt
        source_tokens.append(source_token)
        source_masks.append(source_mask)
        target_tokens.append(target_token)
        target_masks.append(target_mask)

    params = _restore_model.restore_params(
        args.checkpoint / "params", dtype=jnp.bfloat16
    )
    model = config.load(params)
    model.eval()
    decode = _output_transform(args.dataset_root, config)
    fk = SimpleCR1FK(args.urdf)

    source_features = [[] for _ in comparisons]
    target_features = [[] for _ in comparisons]
    z_cosines = [[] for _ in comparisons]
    z_relative_changes = [[] for _ in comparisons]
    for repeat in range(args.noise_repeats):
        noise = jax.random.normal(
            jax.random.key(args.seed + repeat),
            (len(comparisons), config.action_horizon, config.action_dim),
        )
        source_actions, target_actions, source_z, target_z = jax.device_get(
            _sample_pair_with_z(
                model,
                observation,
                jnp.asarray(np.stack(source_tokens)),
                jnp.asarray(np.stack(source_masks)),
                jnp.asarray(np.stack(target_tokens)),
                jnp.asarray(np.stack(target_masks)),
                noise,
            )
        )
        for index, row in enumerate(comparisons):
            raw_state = np.asarray(row["metadata"]["raw_state"])
            decoded_source = decode(
                np.asarray(batch["state"][index]),
                raw_state,
                np.asarray(source_actions[index]),
            )["actions"]
            decoded_target = decode(
                np.asarray(batch["state"][index]),
                raw_state,
                np.asarray(target_actions[index]),
            )["actions"]
            source_features[index].append(_feature(fk, raw_state, decoded_source))
            target_features[index].append(_feature(fk, raw_state, decoded_target))
            source_vector = np.asarray(source_z[index]).reshape(-1).astype(np.float64)
            target_vector = np.asarray(target_z[index]).reshape(-1).astype(np.float64)
            z_cosines[index].append(_cosine(source_vector, target_vector))
            z_relative_changes[index].append(
                float(
                    np.linalg.norm(target_vector - source_vector)
                    / max(np.linalg.norm(source_vector), 1e-12)
                )
            )
        print(f"sampled repeat {repeat + 1}/{args.noise_repeats}", flush=True)

    mean_source = np.stack([np.mean(rows, axis=0) for rows in source_features])
    mean_target = np.stack([np.mean(rows, axis=0) for rows in target_features])
    prototypes = {
        labels: mean_source[
            [row["source"] == labels for row in comparisons]
        ].mean(axis=0)
        for labels in mode_labels
    }

    result_rows = []
    for index, row in enumerate(comparisons):
        source_name = _mode_name(row["source"])
        target_name = _mode_name(row["target"])
        source_feature = mean_source[index]
        target_feature = mean_target[index]
        distance_to_source = float(
            np.linalg.norm(target_feature - prototypes[row["source"]])
        )
        distance_to_target = float(
            np.linalg.norm(target_feature - prototypes[row["target"]])
        )
        before_target_distance = float(
            np.linalg.norm(source_feature - prototypes[row["target"]])
        )
        result_rows.append(
            {
                "source": source_name,
                "target": target_name,
                "segment_id": int(row["segment_id"]),
                "dataset_index": int(row["dataset_index"]),
                "source_prompt": row["source_prompt"],
                "target_prompt": row["target_prompt"],
                "mean_z_cosine": float(np.mean(z_cosines[index])),
                "mean_z_relative_change": float(
                    np.mean(z_relative_changes[index])
                ),
                "endpoint_feature_change": float(
                    np.linalg.norm(target_feature - source_feature)
                ),
                "before_target_distance": before_target_distance,
                "after_target_distance": distance_to_target,
                "after_source_distance": distance_to_source,
                "target_approach": distance_to_target < before_target_distance,
                "target_classification": distance_to_target < distance_to_source,
                "source_feature": source_feature.tolist(),
                "counterfactual_feature": target_feature.tolist(),
            }
        )

    switches = []
    for source in mode_labels:
        for target in mode_labels:
            if source == target:
                continue
            rows = [
                row
                for row in result_rows
                if row["source"] == _mode_name(source)
                and row["target"] == _mode_name(target)
            ]
            switches.append(
                {
                    "source": _mode_name(source),
                    "target": _mode_name(target),
                    "samples": len(rows),
                    "target_classification_rate": float(
                        np.mean([row["target_classification"] for row in rows])
                    ),
                    "target_approach_rate": float(
                        np.mean([row["target_approach"] for row in rows])
                    ),
                    "mean_z_cosine": float(
                        np.mean([row["mean_z_cosine"] for row in rows])
                    ),
                    "mean_z_relative_change": float(
                        np.mean([row["mean_z_relative_change"] for row in rows])
                    ),
                    "mean_endpoint_feature_change": float(
                        np.mean([row["endpoint_feature_change"] for row in rows])
                    ),
                }
            )

    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "state_center": args.state_center,
        "state_rms_radius": args.state_rms_radius,
        "samples_per_mode": args.samples_per_mode,
        "noise_repeats": args.noise_repeats,
        "mode_labels": [list(labels) for labels in mode_labels],
        "macro_translation_min_m": args.macro_translation_min_m,
        "macro_rotation_min_deg": args.macro_rotation_min_deg,
        "feature_units": "[dx,dy,dz]/5mm + [rx,ry,rz]/2deg",
        "switches": switches,
        "rows": result_rows,
    }
    result_path = args.output_dir / "zm_in_distribution_switch.json"
    result_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot(
        summary,
        args.output_dir / "zm_in_distribution_switch.png",
        mode_labels,
    )
    print(json.dumps(switches, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
