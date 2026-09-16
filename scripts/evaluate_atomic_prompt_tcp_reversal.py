#!/usr/bin/env python3
"""Plot paired TCP rollouts for correct and direction-reversed atomic prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from evaluate_global_episode_chunks import (
    _endpoint_tcp_from_actions,
    _sample_global,
    _tcp200_trajectory,
)
from evaluate_zm_fk_trajectory_ablation import _output_transform
from evaluate_zt_empty_vs_atomic import reverse_atomic_prompt


_TRANSLATION_LABELS = tuple(ATOMIC_NAMES[:6])
_AXIS = {
    "move_x_pos": np.asarray([1.0, 0.0, 0.0]),
    "move_x_neg": np.asarray([-1.0, 0.0, 0.0]),
    "move_y_pos": np.asarray([0.0, 1.0, 0.0]),
    "move_y_neg": np.asarray([0.0, -1.0, 0.0]),
    "move_z_pos": np.asarray([0.0, 0.0, 1.0]),
    "move_z_neg": np.asarray([0.0, 0.0, -1.0]),
}


def _select(dataset) -> list[dict]:
    """Pick one complete, single-translation horizon for each arm and direction."""
    raw = dataset._raw  # noqa: SLF001
    selected: dict[tuple[str, str], dict] = {}
    for dataset_index, data_index in enumerate(raw.base._visible_indices):  # noqa: SLF001
        metadata = raw.metadata(dataset_index)
        weights = np.asarray(metadata["atomic_weights"])
        masks = np.asarray(metadata["atomic_supervision_mask"], dtype=bool)
        if weights.shape != (2, 13):
            continue
        episode = int(raw.base._episode_index[data_index])  # noqa: SLF001
        query_indices, _ = raw.base._get_query_indices(int(data_index), episode)  # noqa: SLF001
        action_indices = np.asarray(query_indices["action"], dtype=np.int64)
        if len(action_indices) != 50 or len(np.unique(action_indices)) != 50:
            continue
        try:
            opposite = reverse_atomic_prompt(str(metadata["atomic_prompt"]))
        except ValueError:
            continue
        for arm_index, arm in enumerate(("right", "left")):
            active = np.flatnonzero(weights[arm_index] > 0)
            if not masks[arm_index] or len(active) != 1 or active[0] >= 6:
                continue
            label = ATOMIC_NAMES[int(active[0])]
            key = (arm, label)
            if key in selected:
                continue
            selected[key] = {
                "dataset_index": dataset_index,
                "data_index": int(data_index),
                "episode": episode,
                "frame": int(raw.base._frame_index[data_index]),  # noqa: SLF001
                "arm": arm,
                "label": label,
                "metadata": metadata,
                "action_indices": action_indices,
                "opposite_prompt": opposite,
            }
        if len(selected) == 12:
            break
    ordered = [
        selected[key]
        for arm in ("right", "left")
        for label in _TRANSLATION_LABELS
        if (key := (arm, label)) in selected
    ]
    if len(ordered) != 12:
        missing = [
            f"{arm}:{label}"
            for arm in ("right", "left")
            for label in _TRANSLATION_LABELS
            if (arm, label) not in selected
        ]
        raise RuntimeError(f"missing balanced atomic samples: {missing}")
    return ordered


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-9 else float("nan")


def _plot(rows: list[dict], output: Path) -> None:
    figure = plt.figure(figsize=(18, 12), constrained_layout=True)
    for index, row in enumerate(rows):
        axis = figure.add_subplot(3, 4, index + 1, projection="3d")
        for name, color, style in (
            ("gt", "#334155", "-"),
            ("atomic", "#0284c7", "-"),
            ("opposite", "#e11d48", "--"),
        ):
            xyz = np.asarray(row["relative_tcp_mm"][name])
            axis.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], style, color=color,
                      linewidth=2.1, label=name)
            axis.scatter(*xyz[-1], color=color, s=26)
        all_xyz = np.concatenate(
            [np.asarray(row["relative_tcp_mm"][name]) for name in ("gt", "atomic", "opposite")]
        )
        center = 0.5 * (all_xyz.min(axis=0) + all_xyz.max(axis=0))
        radius = max(float(np.ptp(all_xyz, axis=0).max()) * 0.58, 4.0)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_xlabel("base x / mm")
        axis.set_ylabel("base y / mm")
        axis.set_zlabel("base z / mm")
        axis.set_title(
            f"{row['arm']} · {row['label']}\n"
            f"ep{row['episode']} f{row['frame']} · flip={row['target_axis_sign_flipped']}",
            fontsize=10,
        )
        if index == 0:
            axis.legend(frameon=False, fontsize=9)
    figure.suptitle(
        "Current 60K checkpoint: TCP response to correct vs reversed atomic prompt\n"
        "same image, state and diffusion noise · TCP=0.20 m · trajectories aligned at t=0",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_components(rows: list[dict], output: Path) -> None:
    names = [f"{row['arm'][0].upper()}:{row['label'].replace('move_', '')}" for row in rows]
    atomic = [row["atomic_target_axis_mm"] for row in rows]
    opposite = [row["opposite_target_axis_mm"] for row in rows]
    x = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(16, 6), constrained_layout=True)
    axis.axhline(0, color="#475569", linewidth=1)
    axis.bar(x - 0.2, atomic, 0.4, label="correct atomic prompt", color="#0284c7")
    axis.bar(x + 0.2, opposite, 0.4, label="reversed atomic prompt", color="#e11d48")
    axis.set_xticks(x, names, rotation=35, ha="right")
    axis.set_ylabel("endpoint displacement along original target axis / mm")
    axis.set_title("Negative red bars mean the reversed prompt produced the requested opposite direction")
    axis.legend(frameon=False)
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--noise-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = AtomicPi05Config(max_token_len=250, fast_action_ce_loss_weight=0.0)
    dataset = build_atomic_dataset(
        (args.dataset_root,), norm_assets_dir="/mnt/cunchu/zjq/target",
        norm_asset_id="openpi_norm_compact_accepted_v3", action_horizon=config.action_horizon,
        max_token_len=config.max_token_len, include_fast=False,
    )
    selections = _select(dataset)
    samples = [dataset[row["dataset_index"]] for row in selections]
    batch = atomic_collate(samples)
    observation_np, _ = batch_to_observation(batch)
    observation = jax.tree.map(jnp.asarray, observation_np)
    tokenizer = _paligemma_tokenizer(config.max_token_len)
    opposite = [
        tokenizer.tokenize(row["opposite_prompt"], np.asarray(sample["state"]))
        for row, sample in zip(selections, samples, strict=True)
    ]

    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    decode = _output_transform(args.dataset_root, config)
    decoded = {"atomic": [], "opposite": []}
    for repeat in range(args.noise_repeats):
        noise = jax.random.normal(
            jax.random.key(args.seed + repeat),
            (len(samples), config.action_horizon, config.action_dim),
        )
        atomic_values = _sample_global(
            model, observation, noise,
            jnp.asarray(batch["atomic_prompt_tokens"]),
            jnp.asarray(batch["atomic_prompt_mask"]), None,
        )
        opposite_values = _sample_global(
            model, observation, noise,
            jnp.asarray(np.stack([item[0] for item in opposite])),
            jnp.asarray(np.stack([item[1] for item in opposite])), None,
        )
        for name, values in (("atomic", atomic_values), ("opposite", opposite_values)):
            actions = []
            for index, prediction in enumerate(np.asarray(jax.device_get(values))):
                actions.append(decode(
                    np.asarray(batch["state"][index]),
                    np.asarray(selections[index]["metadata"]["raw_state"]),
                    prediction,
                )["actions"])
            decoded[name].append(np.stack(actions))
        print(f"sampled repeat {repeat + 1}/{args.noise_repeats}", flush=True)

    raw = dataset._raw  # noqa: SLF001
    rows = []
    for index, selected in enumerate(selections):
        arm = selected["arm"]
        current = _endpoint_tcp_from_actions(
            raw, selected["data_index"],
            np.asarray(selected["metadata"]["raw_state"])[None], arm,
        )[0]
        trajectories = {
            "gt": np.concatenate([
                current[None], _tcp200_trajectory(raw, selected["action_indices"], arm)
            ])
        }
        for name in ("atomic", "opposite"):
            repeat_xyz = [
                np.concatenate([
                    current[None],
                    _endpoint_tcp_from_actions(raw, selected["data_index"], values[index], arm),
                ])
                for values in decoded[name]
            ]
            trajectories[name] = np.mean(repeat_xyz, axis=0)
        relative = {name: (xyz - current) * 1000.0 for name, xyz in trajectories.items()}
        expected = _AXIS[selected["label"]]
        atomic_delta = trajectories["atomic"][-1] - current
        opposite_delta = trajectories["opposite"][-1] - current
        atomic_component = float(np.dot(atomic_delta, expected) * 1000.0)
        opposite_component = float(np.dot(opposite_delta, expected) * 1000.0)
        rows.append({
            "episode": selected["episode"], "frame": selected["frame"],
            "arm": arm, "label": selected["label"],
            "atomic_prompt": selected["metadata"]["atomic_prompt"],
            "opposite_prompt": selected["opposite_prompt"],
            "atomic_target_axis_mm": atomic_component,
            "opposite_target_axis_mm": opposite_component,
            "target_axis_sign_flipped": bool(atomic_component * opposite_component < 0),
            "atomic_expected_cosine": _cosine(atomic_delta, expected),
            "opposite_expected_cosine": _cosine(opposite_delta, -expected),
            "atomic_opposite_cosine": _cosine(atomic_delta, opposite_delta),
            "relative_tcp_mm": {name: xyz.tolist() for name, xyz in relative.items()},
        })

    report = {
        "checkpoint": str(args.checkpoint),
        "tcp_offset_m": 0.20,
        "samples": len(rows),
        "noise_repeats": args.noise_repeats,
        "target_axis_sign_flip_rate": float(np.mean([row["target_axis_sign_flipped"] for row in rows])),
        "mean_atomic_expected_cosine": float(np.nanmean([row["atomic_expected_cosine"] for row in rows])),
        "mean_opposite_expected_cosine": float(np.nanmean([row["opposite_expected_cosine"] for row in rows])),
        "mean_atomic_opposite_cosine": float(np.nanmean([row["atomic_opposite_cosine"] for row in rows])),
        "rows": rows,
    }
    (args.output_dir / "tcp_prompt_reversal.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _plot(rows, args.output_dir / "tcp_prompt_reversal_12atoms.png")
    _plot_components(rows, args.output_dir / "tcp_prompt_reversal_components.png")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
