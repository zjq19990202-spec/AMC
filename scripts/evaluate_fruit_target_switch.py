#!/usr/bin/env python3
"""Counterfactual fruit target switch at one fixed image/state horizon."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
from openpi.models import model as _model

from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import (
    _paligemma_tokenizer,
    atomic_collate,
    batch_to_observation,
    build_atomic_dataset,
)
from evaluate_global_episode_chunks import (
    _endpoint_tcp_from_actions,
    _output_transform,
    _tcp200_trajectory,
)


PROMPTS = {
    "carrot": "Move the right hand toward the carrot and grasp it",
    "orange": "Move the right hand toward the orange and grasp it",
    "red apple": "Move the right hand toward the red apple and grasp it",
    "green radish": "Move the right hand toward the green radish and grasp it",
}
COLORS = {
    "recorded GT": "#111827",
    "carrot": "#f97316",
    "orange": "#facc15",
    "red apple": "#dc2626",
    "green radish": "#16a34a",
}


@nnx.jit
def _sample_variants(model, observation, tokens, masks, noise):
    observation = _model.preprocess_observation(None, observation, train=False)
    batch = tokens.shape[0]
    observation = jax.tree.map(
        lambda value: jnp.repeat(value, batch, axis=0), observation
    )
    observation = model._with_prompt(observation, tokens, masks)  # noqa: SLF001
    query_hidden, prefix_mask, kv_cache, _ = model._prefix_forward(observation)  # noqa: SLF001
    active_state = model._controlled_state(observation.state)  # noqa: SLF001
    _, direction, z_model, _, _ = model._latent(query_hidden, active_state)  # noqa: SLF001
    actions = model._mask_action_condition(jnp.repeat(noise, batch, axis=0))  # noqa: SLF001

    def step(index, current):
        time = jnp.asarray(1.0 - index / 10.0, current.dtype)
        velocity = model._suffix_velocity(  # noqa: SLF001
            prefix_mask,
            kv_cache,
            current,
            jnp.broadcast_to(time, (batch,)),
            z_model,
        )
        return model._mask_action_condition(current - 0.1 * velocity)  # noqa: SLF001

    return jax.lax.fori_loop(0, 10, step, actions)[..., :16], direction, z_model


@nnx.jit
def _sample_variants_layerwise(model, observation, tokens, masks, noise):
    """Current route: prompt-specific final and per-layer zM modulation."""

    observation = _model.preprocess_observation(None, observation, train=False)
    batch = tokens.shape[0]
    observation = jax.tree.map(
        lambda value: jnp.repeat(value, batch, axis=0), observation
    )
    observation = model._with_prompt(observation, tokens, masks)  # noqa: SLF001
    query_hidden, prefix_mask, kv_cache, _, layerwise_latents = model._prefix_forward(  # noqa: SLF001
        observation,
        return_layerwise_latents=True,
    )
    active_state = model._controlled_state(observation.state)  # noqa: SLF001
    _, direction, z_model, _, _ = model._latent(query_hidden, active_state)  # noqa: SLF001
    actions = model._mask_action_condition(jnp.repeat(noise, batch, axis=0))  # noqa: SLF001

    def step(index, current):
        time = jnp.asarray(1.0 - index / 10.0, current.dtype)
        velocity = model._suffix_velocity(  # noqa: SLF001
            prefix_mask,
            kv_cache,
            current,
            jnp.broadcast_to(time, (batch,)),
            z_model,
            layerwise_latents=layerwise_latents,
        )
        return model._mask_action_condition(current - 0.1 * velocity)  # noqa: SLF001

    return jax.lax.fori_loop(0, 10, step, actions)[..., :16], direction, z_model


@nnx.jit
def _sample_variants_no_zm(model, observation, tokens, masks, noise):
    """Ablation: retain prompt-specific Context KV but remove all zM FiLM."""

    observation = _model.preprocess_observation(None, observation, train=False)
    batch = tokens.shape[0]
    observation = jax.tree.map(
        lambda value: jnp.repeat(value, batch, axis=0), observation
    )
    observation = model._with_prompt(observation, tokens, masks)  # noqa: SLF001
    query_hidden, prefix_mask, kv_cache, _ = model._prefix_forward(observation)  # noqa: SLF001
    active_state = model._controlled_state(observation.state)  # noqa: SLF001
    _, direction, z_model, _, _ = model._latent(query_hidden, active_state)  # noqa: SLF001
    actions = model._mask_action_condition(jnp.repeat(noise, batch, axis=0))  # noqa: SLF001

    def step(index, current):
        time = jnp.asarray(1.0 - index / 10.0, current.dtype)
        velocity = model._suffix_velocity(  # noqa: SLF001
            prefix_mask,
            kv_cache,
            current,
            jnp.broadcast_to(time, (batch,)),
            None,
            layerwise_latents=None,
        )
        return model._mask_action_condition(current - 0.1 * velocity)  # noqa: SLF001

    return jax.lax.fori_loop(0, 10, step, actions)[..., :16], direction, z_model


def _find_row(dataset, episode: int, frame: int):
    raw = dataset._raw  # noqa: SLF001
    # ``raw`` may apply the reviewed start-of-episode inactive-arm mask on top
    # of the LeRobot base view.  Iterating the base view directly and then
    # passing that position to ``raw.metadata`` applies the mask a second time
    # and silently selects a different episode/frame.  Resolve every dataset
    # row through the raw view's authoritative mapping instead.
    for dataset_index in range(len(raw)):
        data_index = raw._data_index(dataset_index)  # noqa: SLF001
        if (
            int(raw.base._episode_index[data_index]) == episode  # noqa: SLF001
            and int(raw.base._frame_index[data_index]) == frame  # noqa: SLF001
        ):
            metadata = raw.metadata(dataset_index)
            query, _ = raw.base._get_query_indices(data_index, episode)  # noqa: SLF001
            return dataset_index, data_index, np.asarray(query["action"], dtype=np.int64), metadata
    raise RuntimeError(f"missing episode={episode} frame={frame}")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-9 else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=550)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = AtomicPi05Config(max_token_len=250, fast_action_ce_loss_weight=0.0)
    dataset = build_atomic_dataset(
        (args.dataset_root,),
        norm_assets_dir="/mnt/cunchu/zjq/target",
        norm_asset_id="openpi_norm_compact_accepted_v3",
        action_horizon=50,
        max_token_len=250,
        include_fast=False,
    )
    dataset_index, data_index, action_indices, metadata = _find_row(
        dataset, args.episode, args.frame
    )
    row = dataset[dataset_index]
    batch = atomic_collate([row])
    obs_np, _ = batch_to_observation(batch)
    observation = jax.tree.map(jnp.asarray, obs_np)
    tokenizer = _paligemma_tokenizer(250)
    prompt_names = list(PROMPTS)
    token_rows, mask_rows = zip(*[
        tokenizer.tokenize(PROMPTS[name], np.asarray(row["state"]))
        for name in prompt_names
    ], strict=True)

    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    noise = jax.random.normal(jax.random.key(args.seed), (1, 50, config.action_dim))
    normalized, directions, z_models = jax.device_get(_sample_variants(
        model,
        observation,
        jnp.asarray(np.stack(token_rows)),
        jnp.asarray(np.stack(mask_rows)),
        noise,
    ))
    decoder = _output_transform(args.dataset_root, config)
    raw = dataset._raw  # noqa: SLF001
    gt_actions = np.asarray(metadata["raw_actions"])
    gt_xyz = _tcp200_trajectory(raw, action_indices, "right")
    # Bimanual sidecar columns 24:36 are the right-arm state pose.
    current_xyz = np.asarray(raw.tcp_pose[data_index, 24:27])

    trajectories = {"recorded GT": np.concatenate([current_xyz[None], gt_xyz])}
    decoded = {}
    summary = {}
    for index, name in enumerate(prompt_names):
        actions = np.asarray(decoder(
            np.asarray(batch["state"][0]), metadata["raw_state"], normalized[index]
        )["actions"])
        decoded[name] = actions
        xyz = _endpoint_tcp_from_actions(raw, data_index, actions, "right")
        trajectory = np.concatenate([current_xyz[None], xyz])
        trajectories[name] = trajectory
        displacement = xyz[-1] - current_xyz
        summary[name] = {
            "prompt": PROMPTS[name],
            "endpoint_xyz_m": xyz[-1].tolist(),
            "endpoint_displacement_mm": (displacement * 1000).tolist(),
            "endpoint_magnitude_mm": float(np.linalg.norm(displacement) * 1000),
            "gt_endpoint_error_mm": float(np.linalg.norm(xyz[-1] - gt_xyz[-1]) * 1000),
            "q1_direction": np.asarray(directions[index]).tolist(),
            "zM_norm": float(np.linalg.norm(z_models[index])),
        }

    pairwise = {}
    for i, first in enumerate(prompt_names):
        for second in prompt_names[i + 1:]:
            a = trajectories[first][-1] - current_xyz
            b = trajectories[second][-1] - current_xyz
            pairwise[f"{first} vs {second}"] = {
                "endpoint_separation_mm": float(
                    np.linalg.norm(trajectories[first][-1] - trajectories[second][-1]) * 1000
                ),
                "displacement_cosine": _cosine(a, b),
                "normalized_action_rms_difference": float(
                    np.sqrt(np.mean(np.square(normalized[prompt_names.index(first)] - normalized[prompt_names.index(second)])))
                ),
            }

    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax3d = fig.add_subplot(grid[0, 0], projection="3d")
    for name, trajectory in trajectories.items():
        ax3d.plot(
            trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            color=COLORS[name], lw=2.4 if name == "recorded GT" else 1.8,
            ls="--" if name == "recorded GT" else "-", label=name,
        )
        ax3d.scatter(*trajectory[-1], color=COLORS[name], s=32)
    ax3d.scatter(*current_xyz, color="#7c3aed", marker="*", s=100, label="shared start")
    ax3d.set_xlabel("base x (m)"); ax3d.set_ylabel("base y (m)"); ax3d.set_zlabel("base z (m)")
    ax3d.set_title("Same scene/state/noise: right TCP 50-step trajectories")
    ax3d.legend(frameon=False, fontsize=8)

    coordinate = fig.add_subplot(grid[0, 1])
    for name in prompt_names:
        displacement = (trajectories[name] - current_xyz) * 1000
        coordinate.plot(displacement[:, 0], color=COLORS[name], lw=1.7, label=f"{name}: x")
    coordinate.set_title("Base-frame x displacement")
    coordinate.set_xlabel("horizon step"); coordinate.set_ylabel("mm"); coordinate.grid(alpha=.25)
    coordinate.legend(frameon=False, fontsize=8)

    endpoint = fig.add_subplot(grid[1, 0])
    x = np.arange(len(prompt_names)); width = .24
    values = np.stack([
        np.asarray(summary[name]["endpoint_displacement_mm"]) for name in prompt_names
    ])
    for dim, axis_name in enumerate("xyz"):
        endpoint.bar(x + (dim - 1) * width, values[:, dim], width, label=f"Δ{axis_name}")
    endpoint.axhline(0, color="#111827", lw=.7)
    endpoint.set_xticks(x, prompt_names, rotation=12)
    endpoint.set_ylabel("endpoint displacement (mm)")
    endpoint.set_title("Prompt-dependent TCP endpoint")
    endpoint.legend(frameon=False)

    joint = fig.add_subplot(grid[1, 1])
    for name in prompt_names:
        joint.plot(
            np.linalg.norm(decoded[name][:, 8:15] - metadata["raw_state"][None, 8:15], axis=-1),
            color=COLORS[name], label=name,
        )
    joint.set_title("Right joint-space distance from shared state")
    joint.set_xlabel("horizon step"); joint.set_ylabel("L2 radians"); joint.grid(alpha=.25)
    joint.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"Fruit target steering · episode {args.episode}, frame {args.frame}\n"
        "Only the target fruit in the subtask prompt changes",
        fontsize=15, fontweight="bold",
    )
    figure = args.output_dir / "fruit_target_switch_horizon.png"
    fig.savefig(figure, dpi=190, bbox_inches="tight")
    plt.close(fig)

    report = {
        "episode": args.episode,
        "frame": args.frame,
        "checkpoint": str(args.checkpoint),
        "original_global_prompt": metadata["global_prompt"],
        "original_subtask_prompt": metadata["subtask_prompt"],
        "paired_control": "identical three-camera images, robot state and flow noise; prompt target only changes",
        "targets": summary,
        "pairwise": pairwise,
        "figure": str(figure),
    }
    (args.output_dir / "fruit_target_switch_horizon.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "fruit_target_switch_horizon.npz",
        normalized_actions=normalized,
        decoded_actions=np.stack([decoded[name] for name in prompt_names]),
        gt_actions=gt_actions,
        trajectories=np.stack([trajectories[name] for name in prompt_names]),
        gt_trajectory=trajectories["recorded GT"],
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
