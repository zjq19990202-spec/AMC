#!/usr/bin/env python3
"""Cross-test real subtasks at frames with nearly identical robot state.

The image/state pair is fixed within each source row.  Only the subtask text is
changed, and every counterfactual text is copied from a real neighbouring row
in the same normalized joint-state cluster.
"""

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

from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import (
    _paligemma_tokenizer,
    atomic_collate,
    batch_to_observation,
    build_atomic_dataset,
)
from evaluate_fruit_target_switch import _cosine, _find_row, _sample_variants
from evaluate_global_episode_chunks import (
    _endpoint_tcp_from_actions,
    _output_transform,
    _tcp200_trajectory,
)


SOURCES = (
    (569, 1100, "green melon / right box image"),
    (610, 434, "green melon / center box image"),
    (653, 704, "orange pumpkin grasp image"),
)

PROMPTS = {
    "melon->right": "Lift the green bitter melon from the tabletop and place it into the right box",
    "melon->center": "Lift the green bitter melon from the tabletop and place it into the center box",
    "grasp pumpkin": "Move the right hand toward the orange pumpkin and grasp it",
    "carrot->right": "Lift the carrot from the tabletop and place it into the right box",
    "pear->center": "Lift the yellow pear from the tabletop and place it into the center box, release the gripper",
}

COLORS = {
    "GT": "#111827",
    "melon->right": "#dc2626",
    "melon->center": "#2563eb",
    "grasp pumpkin": "#f59e0b",
    "carrot->right": "#16a34a",
    "pear->center": "#9333ea",
}


@nnx.jit
def _flow_losses(model, observation, actions, tokens, masks, rng):
    """Paired FM loss: identical observation, GT, noise and flow time."""
    noise_rng, time_rng = jax.random.split(rng)
    observation = _model.preprocess_observation(None, observation, train=False)
    count = tokens.shape[0]
    observation = jax.tree.map(lambda value: jnp.repeat(value, count, axis=0), observation)
    observation = model._with_prompt(observation, tokens, masks)  # noqa: SLF001
    actions = jnp.repeat(actions, count, axis=0)
    actions = model._mask_action_condition(actions)  # noqa: SLF001
    noise = model._mask_action_condition(  # noqa: SLF001
        jnp.repeat(jax.random.normal(noise_rng, actions[:1].shape), count, axis=0)
    )
    shared_time = jax.random.beta(time_rng, 1.5, 1.0, (1,)) * 0.999 + 0.001
    time = jnp.repeat(shared_time, count, axis=0)
    noisy = time[:, None, None] * noise + (1.0 - time[:, None, None]) * actions
    target = model._controlled_actions(noise - actions)  # noqa: SLF001
    active_state = model._controlled_state(observation.state)  # noqa: SLF001
    query_hidden, prefix_mask, kv_cache, _ = model._prefix_forward(observation)  # noqa: SLF001
    _, _, z_model, _, _ = model._latent(query_hidden, active_state)  # noqa: SLF001
    velocity = model._suffix_velocity(prefix_mask, kv_cache, noisy, time, z_model)  # noqa: SLF001
    error = model._controlled_actions(velocity) - target  # noqa: SLF001
    return jnp.mean(jnp.square(error), axis=(1, 2)), time, z_model


def _equal_3d(ax, trajectories: list[np.ndarray]) -> None:
    points = np.concatenate(trajectories)
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo + hi) / 2
    radius = max(float(np.max(hi - lo)) / 2, 0.025)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _save_camera_montage(obs_np, title: str, path: Path) -> None:
    names = list(obs_np.images)
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.2), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, name in zip(axes, names, strict=True):
        image = np.asarray(obs_np.images[name][0])
        if image.dtype != np.uint8:
            image = np.clip((image + 1.0) * 127.5, 0, 255).astype(np.uint8)
        axis.imshow(image)
        axis.set_title(name)
        axis.axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--plot-steps", type=int, default=25)
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
    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    tokenizer = _paligemma_tokenizer(250)
    decoder = _output_transform(args.dataset_root, config)
    raw = dataset._raw  # noqa: SLF001
    prompt_names = list(PROMPTS)

    figure = plt.figure(figsize=(19, 6.2 * len(SOURCES)), constrained_layout=True)
    grid = figure.add_gridspec(len(SOURCES), 2, width_ratios=(1.15, 1.0))
    xy_figure, xy_axes = plt.subplots(
        1,
        len(SOURCES),
        figsize=(7.0 * len(SOURCES), 6.2),
        constrained_layout=True,
        squeeze=False,
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "control": "within each row, images/state/noise/flow-time fixed; only real subtask text changes",
        "prompt_provenance": "real prompts from normalized-state neighbours of episode 569 frame 1100",
        "plot_steps": args.plot_steps,
        "results": [],
    }

    for source_index, (episode, frame, label) in enumerate(SOURCES):
        dataset_index, data_index, action_indices, metadata = _find_row(dataset, episode, frame)
        row = dataset[dataset_index]
        batch = atomic_collate([row])
        obs_np, actions_np = batch_to_observation(batch)
        observation = jax.tree.map(jnp.asarray, obs_np)
        token_rows, mask_rows = zip(
            *[tokenizer.tokenize(PROMPTS[name], np.asarray(row["state"])) for name in prompt_names],
            strict=True,
        )
        tokens = jnp.asarray(np.stack(token_rows))
        masks = jnp.asarray(np.stack(mask_rows))
        noise = jax.random.normal(
            jax.random.fold_in(jax.random.key(args.seed), source_index),
            (1, 50, config.action_dim),
        )
        normalized, directions, sampled_z = jax.device_get(
            _sample_variants(model, observation, tokens, masks, noise)
        )
        flow_loss, flow_time, flow_z = jax.device_get(
            _flow_losses(
                model,
                observation,
                jnp.asarray(actions_np),
                tokens,
                masks,
                jax.random.fold_in(jax.random.key(args.seed + 1), source_index),
            )
        )

        current_xyz = np.asarray(raw.tcp_pose[data_index, 24:27])
        gt_xyz = _tcp200_trajectory(raw, action_indices, "right")
        trajectories = {"GT": np.concatenate([current_xyz[None], gt_xyz])}
        entries = {}
        for prompt_index, name in enumerate(prompt_names):
            actions = np.asarray(
                decoder(
                    np.asarray(batch["state"][0]),
                    metadata["raw_state"],
                    normalized[prompt_index],
                )["actions"]
            )
            xyz = _endpoint_tcp_from_actions(raw, data_index, actions, "right")
            trajectories[name] = np.concatenate([current_xyz[None], xyz])
            delta25 = xyz[min(args.plot_steps, len(xyz)) - 1] - current_xyz
            delta50 = xyz[-1] - current_xyz
            entries[name] = {
                "prompt": PROMPTS[name],
                "flow_mse": float(flow_loss[prompt_index]),
                "delta_step25_mm": (delta25 * 1000).tolist(),
                "delta_step50_mm": (delta50 * 1000).tolist(),
                "zM_norm": float(np.linalg.norm(sampled_z[prompt_index])),
                "q1_direction_norm": float(np.linalg.norm(directions[prompt_index])),
            }
        reference_z = sampled_z[0]
        for prompt_index, name in enumerate(prompt_names):
            candidate_z = sampled_z[prompt_index]
            entries[name]["zM_cosine_to_melon_right"] = _cosine(
                reference_z.reshape(-1), candidate_z.reshape(-1)
            )
            if reference_z.ndim == 2 and reference_z.shape[0] == 2:
                entries[name]["zM_cosine_to_melon_right_by_arm"] = [
                    _cosine(reference_z[arm], candidate_z[arm]) for arm in range(2)
                ]

        horizon = min(args.plot_steps, 50)
        ax3d = figure.add_subplot(grid[source_index, 0], projection="3d")
        for name, trajectory in trajectories.items():
            shown = trajectory[: horizon + 1]
            ax3d.plot(
                shown[:, 0], shown[:, 1], shown[:, 2],
                color=COLORS[name], lw=2.4 if name == "GT" else 1.7,
                ls="--" if name == "GT" else "-", label=name,
            )
            ax3d.scatter(*shown[-1], color=COLORS[name], s=28)
        ax3d.scatter(*current_xyz, color="#e11d48", marker="*", s=100)
        _equal_3d(ax3d, [trajectory[: horizon + 1] for trajectory in trajectories.values()])
        ax3d.set_xlabel("base x (m)")
        ax3d.set_ylabel("base y (m)")
        ax3d.set_zlabel("base z (m)")
        ax3d.set_title(f"{label}\nepisode {episode}, frame {frame}, first {horizon} steps")
        if source_index == 0:
            ax3d.legend(frameon=False, fontsize=8)

        axbar = figure.add_subplot(grid[source_index, 1])
        x = np.arange(len(prompt_names))
        width = 0.24
        displacement = np.asarray([entries[name]["delta_step25_mm"] for name in prompt_names])
        for dim, (axis_name, color) in enumerate(zip("xyz", ("#dc2626", "#16a34a", "#2563eb"), strict=True)):
            axbar.bar(x + (dim - 1) * width, displacement[:, dim], width, color=color, label=f"Delta {axis_name}")
        axbar.axhline(0, color="#6b7280", lw=0.8)
        axbar.set_xticks(x, prompt_names, rotation=18, ha="right")
        axbar.set_ylabel(f"right TCP displacement at step {horizon} (mm)")
        axbar.grid(axis="y", alpha=0.25)
        axbar.set_title("Same images/state, subtask-only endpoint response")
        if source_index == 0:
            axbar.legend(frameon=False, ncol=3, fontsize=8)

        # A base-frame X-Y projection is the most direct destination-steering
        # view: all curves share the origin, and their final circles are the
        # 50-step endpoints.  Plot displacement rather than absolute TCP pose
        # so different source scenes remain comparable.
        ax_xy = xy_axes[0, source_index]
        for name, trajectory in trajectories.items():
            delta_mm = (trajectory - current_xyz[None]) * 1000.0
            ax_xy.plot(
                delta_mm[:, 0],
                delta_mm[:, 1],
                color=COLORS[name],
                lw=2.5 if name == "GT" else 1.8,
                ls="--" if name == "GT" else "-",
                label=name,
            )
            ax_xy.scatter(delta_mm[-1, 0], delta_mm[-1, 1], color=COLORS[name], s=42)
        ax_xy.scatter(0.0, 0.0, color="#e11d48", marker="*", s=130, zorder=10, label="shared start")
        ax_xy.axhline(0.0, color="#9ca3af", lw=0.7)
        ax_xy.axvline(0.0, color="#9ca3af", lw=0.7)
        ax_xy.set_xlabel("base Delta X (mm), +X = forward")
        ax_xy.set_ylabel("base Delta Y (mm), +Y = left")
        ax_xy.set_title(f"{label}\nepisode {episode}, frame {frame}, 50-step endpoint")
        ax_xy.grid(alpha=0.25)
        ax_xy.set_aspect("equal", adjustable="datalim")
        if source_index == 0:
            ax_xy.legend(frameon=False, fontsize=8)

        camera_path = args.output_dir / f"ep{episode}_f{frame}_cameras.png"
        _save_camera_montage(obs_np, f"{label} | original: {metadata['subtask_prompt']}", camera_path)
        report["results"].append(
            {
                "episode": episode,
                "frame": frame,
                "state_cluster_label": label,
                "original_subtask_prompt": metadata["subtask_prompt"],
                "flow_time": float(flow_time[0]),
                "camera_figure": str(camera_path),
                "variants": entries,
            }
        )

    figure.suptitle(
        "Real-subtask counterfactuals at a shared robot-state cluster\n"
        "Each row fixes all visual/state inputs and changes only subtask text",
        fontsize=16,
        fontweight="bold",
    )
    plot_path = args.output_dir / "shared_state_subtask_switch.png"
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    xy_figure.suptitle(
        "X-Y endpoint steering under fixed images and state\n"
        "circle = step-50 endpoint; star = shared current TCP",
        fontsize=15,
        fontweight="bold",
    )
    xy_plot_path = args.output_dir / "shared_state_subtask_switch_xy.png"
    xy_figure.savefig(xy_plot_path, dpi=180, bbox_inches="tight")
    plt.close(xy_figure)
    report["plot"] = str(plot_path)
    report["xy_plot"] = str(xy_plot_path)
    report_path = args.output_dir / "shared_state_subtask_switch.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
