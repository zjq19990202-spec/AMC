#!/usr/bin/env python3
"""Compare old/new ZM checkpoints on paired 50-step 16-D action horizons."""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from openpi.models import model as _model

from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import (
    atomic_collate,
    batch_to_observation,
    build_atomic_dataset,
)

from evaluate_global_episode_chunks import _sample_global
from evaluate_zm_fk_trajectory_ablation import _output_transform


DEFAULT_SAMPLES = "1:0,1:500,1:1000,9:0,9:500,610:1000,610:2000"
COLORS = {"gt": "#111827", "old": "#f97316", "new": "#2563eb"}
STYLES = {"gt": "-", "old": "--", "new": "-"}


def _parse_samples(value: str) -> list[tuple[int, int]]:
    samples: list[tuple[int, int]] = []
    for item in value.split(","):
        episode, frame = item.strip().split(":", 1)
        samples.append((int(episode), int(frame)))
    if not samples:
        raise ValueError("at least one episode:frame sample is required")
    return samples


def _select_rows(dataset, requested: list[tuple[int, int]]):
    raw = dataset._raw  # noqa: SLF001
    wanted = set(requested)
    found = {}
    for dataset_index, data_index in enumerate(raw.base._visible_indices):  # noqa: SLF001
        key = (
            int(raw.base._episode_index[data_index]),  # noqa: SLF001
            int(raw.base._frame_index[data_index]),  # noqa: SLF001
        )
        if key not in wanted:
            continue
        metadata = raw.metadata(dataset_index)
        actions = np.asarray(metadata["raw_actions"])
        if actions.shape[0] != 50:
            raise ValueError(f"sample {key} has action shape {actions.shape}")
        query_indices, _ = raw.base._get_query_indices(  # noqa: SLF001
            data_index, key[0]
        )
        action_indices = np.asarray(query_indices["action"], dtype=np.int64)
        if len(np.unique(action_indices)) != 50:
            raise ValueError(f"sample {key} is a padded episode-tail horizon")
        found[key] = (dataset_index, metadata)
    missing = [key for key in requested if key not in found]
    if missing:
        raise KeyError(f"requested samples are absent: {missing}")
    return [(key, *found[key]) for key in requested]


def _infer(
    checkpoint: Path,
    config: AtomicPi05Config,
    observation,
    noise,
    prompt_tokens,
    prompt_mask,
) -> np.ndarray:
    print(f"RESTORE {checkpoint}", flush=True)
    params = _model.restore_params(checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    committed_mask = jnp.zeros(noise.shape[:2], dtype=jnp.bool_)
    prediction = _sample_global(
        model,
        observation,
        noise,
        prompt_tokens,
        prompt_mask,
        committed_mask,
    )
    prediction = np.asarray(jax.device_get(prediction))
    del model, params
    jax.clear_caches()
    return prediction


def _dimension_values(actions: np.ndarray, dimension: int) -> np.ndarray:
    values = np.asarray(actions[..., dimension])
    return np.degrees(values) if dimension not in (7, 15) else values


def _dimension_label(dimension: int) -> tuple[str, str]:
    arm = "Left" if dimension < 8 else "Right"
    local = dimension if dimension < 8 else dimension - 8
    if local == 7:
        return f"{arm} gripper", "native"
    return f"{arm} J{local + 1}", "degrees"


def _metrics(gt: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    result = {}
    for arm, joint_slice, gripper_index in (
        ("left", slice(0, 7), 7),
        ("right", slice(8, 15), 15),
    ):
        joint_error_deg = np.degrees(prediction[:, joint_slice] - gt[:, joint_slice])
        result[f"{arm}_joint_rmse_deg"] = float(
            np.sqrt(np.mean(np.square(joint_error_deg)))
        )
        result[f"{arm}_joint_endpoint_rmse_deg"] = float(
            np.sqrt(np.mean(np.square(joint_error_deg[-1])))
        )
        gripper_error = prediction[:, gripper_index] - gt[:, gripper_index]
        result[f"{arm}_gripper_rmse"] = float(
            np.sqrt(np.mean(np.square(gripper_error)))
        )
    return result


def _plot_sample(
    sample: dict,
    trajectories: dict[str, np.ndarray],
    output: Path,
) -> None:
    figure, axes = plt.subplots(
        4, 4, figsize=(19, 13), sharex=True, constrained_layout=True
    )
    steps = np.arange(50)
    for dimension, axis in enumerate(axes.flat):
        label, unit = _dimension_label(dimension)
        for variant in ("gt", "old", "new"):
            axis.plot(
                steps,
                _dimension_values(trajectories[variant], dimension),
                color=COLORS[variant],
                linestyle=STYLES[variant],
                linewidth=2.2 if variant == "gt" else 1.65,
                alpha=0.95,
                label={"gt": "GT", "old": "previous 60K", "new": "current 60K"}[
                    variant
                ],
            )
        axis.set_title(label)
        axis.set_ylabel(unit)
        axis.grid(color="#e5e7eb", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        if dimension >= 12:
            axis.set_xlabel("horizon step")
        if dimension == 0:
            axis.legend(frameon=False, fontsize=9, ncol=3)
    prompt = textwrap.fill(sample["subtask_prompt"], width=105)
    figure.suptitle(
        f"Episode {sample['episode']} · frame {sample['frame']} · paired 50-step horizon\n"
        f"{prompt}",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_summary(
    gt: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
    output: Path,
) -> dict[str, list[float]]:
    old_rmse, new_rmse = [], []
    for dimension in range(16):
        target = _dimension_values(gt, dimension)
        old_values = _dimension_values(old, dimension)
        new_values = _dimension_values(new, dimension)
        old_rmse.append(float(np.sqrt(np.mean(np.square(old_values - target)))))
        new_rmse.append(float(np.sqrt(np.mean(np.square(new_values - target)))))

    figure, (joint_axis, gripper_axis) = plt.subplots(
        2, 1, figsize=(17, 9), constrained_layout=True
    )
    joint_indices = list(range(7)) + list(range(8, 15))
    joint_labels = [_dimension_label(index)[0] for index in joint_indices]
    x = np.arange(len(joint_indices))
    width = 0.38
    joint_axis.bar(
        x - width / 2,
        [old_rmse[index] for index in joint_indices],
        width,
        color=COLORS["old"],
        label="previous 60K",
    )
    joint_axis.bar(
        x + width / 2,
        [new_rmse[index] for index in joint_indices],
        width,
        color=COLORS["new"],
        label="current 60K",
    )
    joint_axis.set_xticks(x, joint_labels, rotation=35, ha="right")
    joint_axis.set_ylabel("joint RMSE (degrees)")
    joint_axis.set_title("Paired horizon error by joint — all selected frames")
    joint_axis.legend(frameon=False)
    joint_axis.grid(axis="y", color="#e5e7eb", linewidth=0.7)

    gripper_indices = [7, 15]
    x = np.arange(2)
    gripper_axis.bar(
        x - width / 2,
        [old_rmse[index] for index in gripper_indices],
        width,
        color=COLORS["old"],
        label="previous 60K",
    )
    gripper_axis.bar(
        x + width / 2,
        [new_rmse[index] for index in gripper_indices],
        width,
        color=COLORS["new"],
        label="current 60K",
    )
    gripper_axis.set_xticks(x, ["Left gripper", "Right gripper"])
    gripper_axis.set_ylabel("RMSE (native units)")
    gripper_axis.set_title("Gripper error")
    gripper_axis.grid(axis="y", color="#e5e7eb", linewidth=0.7)
    for axis in (joint_axis, gripper_axis):
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return {"previous_60k": old_rmse, "current_60k": new_rmse}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-checkpoint", type=Path, required=True)
    parser.add_argument("--new-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--samples", default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=20260814)
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
    selected = _select_rows(dataset, _parse_samples(args.samples))
    rows = [dataset[dataset_index] for _, dataset_index, _ in selected]
    batch = atomic_collate(rows)
    observation_np, _ = batch_to_observation(batch)
    observation = jax.tree.map(jnp.asarray, observation_np)
    noise = jax.random.normal(
        jax.random.key(args.seed),
        (len(rows), config.action_horizon, config.action_dim),
    )
    prompt_tokens = jnp.asarray(batch["subtask_prompt_tokens"])
    prompt_mask = jnp.asarray(batch["subtask_prompt_mask"])

    old_normalized = _infer(
        args.old_checkpoint,
        config,
        observation,
        noise,
        prompt_tokens,
        prompt_mask,
    )
    new_normalized = _infer(
        args.new_checkpoint,
        config,
        observation,
        noise,
        prompt_tokens,
        prompt_mask,
    )

    decode = _output_transform(args.dataset_root, config)
    gt_rows, old_rows, new_rows = [], [], []
    summary_rows = []
    for index, ((episode_frame, _, metadata), old_pred, new_pred) in enumerate(
        zip(selected, old_normalized, new_normalized, strict=True)
    ):
        episode, frame = episode_frame
        gt = np.asarray(metadata["raw_actions"])[..., :16]
        old = np.asarray(
            decode(batch["state"][index], metadata["raw_state"], old_pred)["actions"]
        )
        new = np.asarray(
            decode(batch["state"][index], metadata["raw_state"], new_pred)["actions"]
        )
        gt_rows.append(gt)
        old_rows.append(old)
        new_rows.append(new)
        row = {
            "episode": episode,
            "frame": frame,
            "subtask_prompt": str(metadata["subtask_prompt"]),
        }
        for prefix, prediction in (("old", old), ("new", new)):
            row.update(
                {f"{prefix}_{key}": value for key, value in _metrics(gt, prediction).items()}
            )
        summary_rows.append(row)
        _plot_sample(
            row,
            {"gt": gt, "old": old, "new": new},
            args.output_dir / f"episode_{episode:06d}_frame_{frame:06d}_16d.png",
        )

    gt_all = np.concatenate(gt_rows, axis=0)
    old_all = np.concatenate(old_rows, axis=0)
    new_all = np.concatenate(new_rows, axis=0)
    per_dimension = _plot_summary(
        gt_all,
        old_all,
        new_all,
        args.output_dir / "joint_channel_rmse_comparison.png",
    )
    aggregate = {
        "old": _metrics(gt_all, old_all),
        "new": _metrics(gt_all, new_all),
    }
    report = {
        "old_checkpoint": str(args.old_checkpoint),
        "new_checkpoint": str(args.new_checkpoint),
        "dataset_root": str(args.dataset_root),
        "paired_noise_seed": args.seed,
        "prompt_source": "subtask",
        "samples": summary_rows,
        "aggregate": aggregate,
        "per_dimension_rmse": per_dimension,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    np.savez_compressed(
        args.output_dir / "paired_trajectories.npz",
        gt=np.stack(gt_rows),
        previous_60k=np.stack(old_rows),
        current_60k=np.stack(new_rows),
    )
    print(json.dumps({"aggregate": aggregate, "samples": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
