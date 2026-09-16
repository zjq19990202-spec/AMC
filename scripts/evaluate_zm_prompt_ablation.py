#!/usr/bin/env python3
"""Paired zM flow-loss ablation: global task prompt versus atomic subprompt.

Every A/B pair uses the same image/state/action horizon, preprocessing,
flow time and Gaussian noise.  Only the PaliGemma prompt tokens differ.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from openpi.models import model as _model

from atomic_latent_vla.atomic import ATOMIC_NAMES
from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import (
    atomic_collate,
    batch_to_observation,
    build_atomic_dataset,
)


@nnx.jit
def _paired_flow_mse(
    model,
    observation,
    actions,
    atomic_tokens,
    atomic_mask,
    rng,
):
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(
        preprocess_rng, observation, train=False
    )
    atomic_observation = model._with_prompt(  # noqa: SLF001
        observation, atomic_tokens, atomic_mask
    )
    actions = model._mask_action_condition(actions)  # noqa: SLF001
    noise = model._mask_action_condition(  # noqa: SLF001
        jax.random.normal(noise_rng, actions.shape)
    )
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    noisy = time[:, None, None] * noise + (1.0 - time[:, None, None]) * actions
    target = noise - actions
    active_state = model._controlled_state(observation.state)  # noqa: SLF001

    def predict(prompted_observation):
        query_hidden, prefix_mask, kv_cache = model._prefix_forward(  # noqa: SLF001
            prompted_observation
        )
        _, direction, z_model, _, _ = model._latent(  # noqa: SLF001
            query_hidden, active_state
        )
        velocity = model._suffix_velocity(  # noqa: SLF001
            prefix_mask, kv_cache, noisy, time, z_model
        )
        return model._controlled_actions(velocity), direction, z_model  # noqa: SLF001

    global_velocity, global_direction, global_zm = predict(observation)
    atomic_velocity, atomic_direction, atomic_zm = predict(atomic_observation)
    controlled_target = model._controlled_actions(target)  # noqa: SLF001
    reduce_axes = tuple(range(1, controlled_target.ndim))
    global_mse = jnp.mean(
        jnp.square(global_velocity - controlled_target), axis=reduce_axes
    )
    atomic_mse = jnp.mean(
        jnp.square(atomic_velocity - controlled_target), axis=reduce_axes
    )
    direction_cosine = jnp.sum(global_direction * atomic_direction, axis=-1)
    zm_cosine = jnp.sum(global_zm * atomic_zm, axis=-1) / jnp.maximum(
        jnp.linalg.norm(global_zm, axis=-1)
        * jnp.linalg.norm(atomic_zm, axis=-1),
        1e-8,
    )
    return global_mse, atomic_mse, direction_cosine, zm_cosine, time


def _episode_indices(dataset, episode_index: int) -> list[tuple[int, dict]]:
    raw = dataset._raw  # noqa: SLF001
    selected = []
    for dataset_index, data_index in enumerate(raw.base._visible_indices):  # noqa: SLF001
        if int(raw.base._episode_index[data_index]) != episode_index:  # noqa: SLF001
            continue
        metadata = raw.metadata(dataset_index)
        if not bool(metadata["atomic_supervision_mask"]):
            continue
        if metadata["atomic_prompt"].strip() == metadata["global_prompt"].strip():
            continue
        selected.append((dataset_index, metadata))
    return selected


def _summary(global_loss: np.ndarray, atomic_loss: np.ndarray) -> dict[str, float]:
    if len(global_loss) == 0:
        return {"count": 0}
    delta = atomic_loss - global_loss
    relative = delta / np.maximum(global_loss, 1e-8)
    return {
        "count": int(len(global_loss)),
        "global_flow_mse_mean": float(global_loss.mean()),
        "atomic_flow_mse_mean": float(atomic_loss.mean()),
        "atomic_minus_global_mean": float(delta.mean()),
        "atomic_minus_global_median": float(np.median(delta)),
        "relative_change_mean": float(relative.mean()),
        "relative_change_median": float(np.median(relative)),
        "atomic_better_fraction": float(np.mean(delta < 0)),
        "atomic_worse_fraction": float(np.mean(delta > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--norm-assets-dir", default="/mnt/cunchu/zjq/target")
    parser.add_argument("--norm-asset-id", default="openpi_norm_compact_accepted_v3")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--noise-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = AtomicPi05Config(max_token_len=250, fast_action_ce_loss_weight=0.0)
    dataset = build_atomic_dataset(
        (args.dataset_root,),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        include_fast=False,
    )
    selected = _episode_indices(dataset, args.episode)
    if args.max_samples > 0 and len(selected) > args.max_samples:
        keep = np.linspace(0, len(selected) - 1, args.max_samples).round().astype(int)
        selected = [selected[index] for index in keep]
    if not selected:
        raise RuntimeError(
            f"episode {args.episode} has no strict atomic rows with distinct prompts"
        )

    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()

    rows_out = []
    for batch_start in range(0, len(selected), args.batch_size):
        selected_batch = selected[batch_start : batch_start + args.batch_size]
        rows = [dataset[index] for index, _ in selected_batch]
        batch = atomic_collate(rows)
        observation_np, actions_np = batch_to_observation(batch)
        observation = jax.tree.map(jnp.asarray, observation_np)
        actions = jnp.asarray(actions_np)
        repeat_values = []
        for repeat in range(args.noise_repeats):
            values = _paired_flow_mse(
                model,
                observation,
                actions,
                jnp.asarray(batch["atomic_prompt_tokens"]),
                jnp.asarray(batch["atomic_prompt_mask"]),
                jax.random.key(args.seed + repeat + 1000 * batch_start),
            )
            repeat_values.append(jax.device_get(values))
        global_repeats = np.stack([value[0] for value in repeat_values])
        atomic_repeats = np.stack([value[1] for value in repeat_values])
        direction_repeats = np.stack([value[2] for value in repeat_values])
        zm_repeats = np.stack([value[3] for value in repeat_values])
        time_repeats = np.stack([value[4] for value in repeat_values])
        raw = dataset._raw  # noqa: SLF001
        for local_index, (dataset_index, metadata) in enumerate(selected_batch):
            data_index = int(raw.base._visible_indices[dataset_index])  # noqa: SLF001
            weights = np.asarray(metadata["atomic_weights"])
            labels = np.flatnonzero(weights > 0)
            global_loss = float(global_repeats[:, local_index].mean())
            atomic_loss = float(atomic_repeats[:, local_index].mean())
            rows_out.append(
                {
                    "dataset_index": dataset_index,
                    "frame_index": data_index,
                    "timestamp_s": float(raw.base._timestamps[data_index]),  # noqa: SLF001
                    "kind": "single" if len(labels) == 1 else "dual",
                    "labels": "+".join(ATOMIC_NAMES[index] for index in labels),
                    "weights": "+".join(f"{weights[index]:.4f}" for index in labels),
                    "global_prompt": metadata["global_prompt"],
                    "atomic_prompt": metadata["atomic_prompt"],
                    "global_flow_mse": global_loss,
                    "atomic_flow_mse": atomic_loss,
                    "atomic_minus_global": atomic_loss - global_loss,
                    "relative_change": (
                        atomic_loss - global_loss
                    ) / max(global_loss, 1e-8),
                    "direction_cosine": float(
                        direction_repeats[:, local_index].mean()
                    ),
                    "zm_cosine": float(zm_repeats[:, local_index].mean()),
                    "flow_time_mean": float(time_repeats[:, local_index].mean()),
                }
            )
        print(
            f"evaluated {min(batch_start + args.batch_size, len(selected))}/"
            f"{len(selected)}",
            flush=True,
        )

    global_loss = np.asarray([row["global_flow_mse"] for row in rows_out])
    atomic_loss = np.asarray([row["atomic_flow_mse"] for row in rows_out])
    kinds = np.asarray([row["kind"] for row in rows_out])
    report = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "episode": args.episode,
        "noise_repeats": args.noise_repeats,
        "paired_control": (
            "same image/state/action horizon, preprocessing, flow time and noise; "
            "only global versus atomic prompt tokens differ"
        ),
        "overall": _summary(global_loss, atomic_loss),
        "by_kind": {
            kind: _summary(global_loss[kinds == kind], atomic_loss[kinds == kind])
            for kind in sorted(set(kinds))
        },
        "mean_global_atomic_direction_cosine": float(
            np.mean([row["direction_cosine"] for row in rows_out])
        ),
        "mean_global_atomic_zm_cosine": float(
            np.mean([row["zm_cosine"] for row in rows_out])
        ),
        "label_counts": dict(Counter(row["labels"] for row in rows_out)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
