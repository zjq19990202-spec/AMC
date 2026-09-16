#!/usr/bin/env python3
"""Evaluate one complete episode as non-overlapping global-prompt chunks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from openpi.models import model as _model

from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import (
    atomic_collate,
    batch_to_observation,
    build_atomic_dataset,
)

# Reuse the already validated normalization/FK contract from the prompt-ablation
# evaluator. The compact target corpus stores 0.20 m bimanual FK sidecars, so
# these helpers are used only for normalization and the right-arm URDF parser;
# the TCP position below is read from the exact tcp200 sidecar.
import evaluate_zm_fk_trajectory_ablation as _fk_eval

from evaluate_zm_fk_trajectory_ablation import _output_transform


@nnx.jit
def _sample_global(
    model, observation, noise, override_tokens, override_mask, committed_mask
):
    observation = _model.preprocess_observation(None, observation, train=False)
    if override_tokens is not None:
        observation = model._with_prompt(observation, override_tokens, override_mask)  # noqa: SLF001
    query_hidden, prefix_mask, kv_cache, _ = model._prefix_forward(observation)  # noqa: SLF001
    active_state = model._controlled_state(observation.state)  # noqa: SLF001
    _, _, z_model, _, _ = model._latent(query_hidden, active_state)  # noqa: SLF001
    noise = model._mask_action_condition(noise)  # noqa: SLF001

    def step(index, actions):
        time = jnp.asarray(1.0 - index / 10.0, dtype=actions.dtype)
        velocity = model._suffix_velocity(  # noqa: SLF001
            prefix_mask,
            kv_cache,
            actions,
            jnp.broadcast_to(time, (actions.shape[0],)),
            z_model,
            committed_mask=committed_mask,
        )
        return model._mask_action_condition(actions - 0.1 * velocity)  # noqa: SLF001

    return jax.lax.fori_loop(0, 10, step, noise)[..., : model.config.active_action_dim]


@nnx.jit
def _triple_flow_mse(
    model,
    observation,
    actions,
    subtask_tokens,
    subtask_mask,
    atomic_tokens,
    atomic_mask,
    rng,
):
    """Paired per-row FM MSE; only prompt tokens differ across three branches."""
    noise_rng, time_rng = jax.random.split(rng)
    observation = _model.preprocess_observation(None, observation, train=False)
    prompted = (
        observation,
        model._with_prompt(observation, subtask_tokens, subtask_mask),  # noqa: SLF001
        model._with_prompt(observation, atomic_tokens, atomic_mask),  # noqa: SLF001
    )
    actions = model._mask_action_condition(actions)  # noqa: SLF001
    noise = model._mask_action_condition(  # noqa: SLF001
        jax.random.normal(noise_rng, actions.shape)
    )
    time = jax.random.beta(time_rng, 1.5, 1.0, actions.shape[:-2]) * 0.999 + 0.001
    noisy = time[:, None, None] * noise + (1.0 - time[:, None, None]) * actions
    controlled_target = model._controlled_actions(noise - actions)  # noqa: SLF001
    active_state = model._controlled_state(observation.state)  # noqa: SLF001

    def predict(item, disable_zm=False):
        query_hidden, prefix_mask, kv_cache, _ = model._prefix_forward(item)  # noqa: SLF001
        _, _, z_model, _, _ = model._latent(query_hidden, active_state)  # noqa: SLF001
        velocity = model._suffix_velocity(  # noqa: SLF001
            prefix_mask,
            kv_cache,
            noisy,
            time,
            z_model,
            committed_mask=(
                jnp.ones(noisy.shape[:2], dtype=jnp.bool_)
                if disable_zm else None
            ),
        )
        error = model._controlled_actions(velocity) - controlled_target  # noqa: SLF001
        return jnp.mean(jnp.square(error), axis=tuple(range(1, error.ndim)))

    return (
        predict(prompted[0]),
        predict(prompted[1]),
        predict(prompted[2]),
        predict(prompted[1], disable_zm=True),
    )


def _select_episode_chunks(dataset, episode: int, horizon: int):
    raw = dataset._raw  # noqa: SLF001
    candidates = []
    for dataset_index, data_index in enumerate(raw.base._visible_indices):  # noqa: SLF001
        if int(raw.base._episode_index[data_index]) != episode:  # noqa: SLF001
            continue
        frame = int(raw.base._frame_index[data_index])  # noqa: SLF001
        if frame % horizon:
            continue
        metadata = raw.metadata(dataset_index)
        actions = np.asarray(metadata["raw_actions"])
        if actions.shape[0] != horizon:
            continue
        # LeRobot pads at the episode edge. Retain only genuine full windows.
        query_indices, _ = raw.base._get_query_indices(data_index, episode)  # noqa: SLF001
        action_indices = np.asarray(query_indices["action"], dtype=np.int64)
        if len(np.unique(action_indices)) != horizon:
            continue
        candidates.append((frame, dataset_index, data_index, action_indices, metadata))
    candidates.sort(key=lambda row: row[0])
    return candidates


def _tcp200_trajectory(raw, indices: np.ndarray, arm: str) -> np.ndarray:
    # Sidecar layout: left(state, action), right(state, action), 12 values each.
    offset = 36 if arm == "right" else 12
    values = np.asarray(raw.tcp_pose[indices, offset : offset + 3])
    if values.shape != (len(indices), 3):
        raise ValueError(f"bad {arm} tcp200 sidecar shape {values.shape}")
    return values


def _endpoint_tcp_from_actions(raw, data_index: int, actions: np.ndarray, arm: str):
    """Use the model's decoded joint actions with the project's FK implementation."""
    urdf = Path("assets/cr1_right_arm_tcp_red/urdf/CR1ARMR_TCP_RED.urdf")
    # Both arms use the same arm URDF and raw seven joint coordinates. Their
    # only FK difference in sidecar generation is the shoulder mount position.
    # The dependency-free evaluator reads this module constant at pose time.
    # Override it to the exact tcp200 contract used by the target corpus.
    _fk_eval.TCP_LOCAL_Z_OFFSET_M = 0.20
    fk = _fk_eval.SimpleCR1FK(urdf)
    q_slice = slice(8, 15) if arm == "right" else slice(0, 7)
    q = np.asarray(actions[:, q_slice])
    positions = np.stack([fk.pose(row)[0] for row in q])
    if arm == "left":
        positions += np.asarray([0.0, 0.445, 0.0])
    return positions


def _plot(result: dict, output: Path) -> None:
    chunks = result["chunks"]
    gt_r = np.concatenate([np.asarray(row["right_gt_xyz_m"]) for row in chunks])
    pr_r = np.concatenate([np.asarray(row["right_pred_xyz_m"]) for row in chunks])
    gt_l = np.concatenate([np.asarray(row["left_gt_xyz_m"]) for row in chunks])
    pr_l = np.concatenate([np.asarray(row["left_pred_xyz_m"]) for row in chunks])
    step = np.arange(len(gt_r))

    fig = plt.figure(figsize=(18, 15), constrained_layout=True)
    grid = fig.add_gridspec(4, 2, height_ratios=(1.25, 1.25, 1, 1))
    for col, (arm, gt, pred) in enumerate((("right", gt_r, pr_r), ("left", gt_l, pr_l))):
        ax = fig.add_subplot(grid[0, col], projection="3d")
        ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="#111827", lw=2.2, label="GT")
        ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], color="#2563eb", lw=1.5, label="global prediction")
        ax.set_title(f"{arm.capitalize()} TCP trajectory")
        ax.set_xlabel("base x (m)"); ax.set_ylabel("base y (m)"); ax.set_zlabel("base z (m)")
        ax.legend(frameon=False)
        for boundary in range(0, len(gt), 50):
            ax.scatter(*gt[boundary], color="#111827", s=12)

        xyz_ax = fig.add_subplot(grid[1, col])
        for dim, (name, color) in enumerate(zip("xyz", ("#dc2626", "#16a34a", "#2563eb"), strict=True)):
            xyz_ax.plot(step, gt[:, dim], color=color, lw=1.8, label=f"GT {name}")
            xyz_ax.plot(step, pred[:, dim], color=color, lw=1.0, ls="--", alpha=.8, label=f"pred {name}")
        xyz_ax.set_title(f"{arm.capitalize()} TCP coordinates")
        xyz_ax.set_xlabel("episode step"); xyz_ax.set_ylabel("m")
        xyz_ax.grid(alpha=.25); xyz_ax.legend(ncol=3, fontsize=8, frameon=False)
        for boundary in range(0, len(gt) + 1, 50):
            xyz_ax.axvline(boundary, color="#9ca3af", lw=.5, alpha=.5)

        err_ax = fig.add_subplot(grid[2, col])
        err = np.linalg.norm(pred - gt, axis=-1) * 1000
        err_ax.plot(step, err, color="#7c3aed", lw=1.25)
        err_ax.set_title(f"{arm.capitalize()} TCP translation error")
        err_ax.set_xlabel("episode step"); err_ax.set_ylabel("mm"); err_ax.grid(alpha=.25)
        for boundary in range(0, len(gt) + 1, 50):
            err_ax.axvline(boundary, color="#9ca3af", lw=.5, alpha=.5)

        chunk_ax = fig.add_subplot(grid[3, col])
        rmse = [row[f"{arm}_translation_rmse_mm"] for row in chunks]
        endpoint = [row[f"{arm}_translation_final_mm"] for row in chunks]
        x = np.arange(len(chunks))
        chunk_ax.bar(x - .18, rmse, width=.36, color="#2563eb", label="50-step RMSE")
        chunk_ax.bar(x + .18, endpoint, width=.36, color="#f97316", label="endpoint error")
        chunk_ax.set_title(f"{arm.capitalize()} error by non-overlapping 50-step chunk")
        chunk_ax.set_xlabel("chunk index"); chunk_ax.set_ylabel("mm"); chunk_ax.grid(axis="y", alpha=.25)
        chunk_ax.legend(frameon=False)

    fig.suptitle(
        f"Episode {result['episode']} — {result['prompt_source']} prompt, every non-overlapping 50-step window\n"
        f"{result['prompt_description']}", fontsize=15, fontweight="bold"
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--prompt-source", choices=("global", "subtask", "atomic"), default="global"
    )
    parser.add_argument("--flow-repeats", type=int, default=3)
    parser.add_argument(
        "--disable-zm",
        action="store_true",
        help="zero every zM adapter residual while retaining VLM prefix context",
    )
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
    selected = _select_episode_chunks(dataset, args.episode, config.action_horizon)
    if not selected:
        raise RuntimeError(f"episode {args.episode} has no complete 50-step chunks")

    rows = [dataset[row[1]] for row in selected]
    batch = atomic_collate(rows)
    observation_np, actions_np = batch_to_observation(batch)
    observation = jax.tree.map(jnp.asarray, observation_np)
    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    noise = jax.random.normal(
        jax.random.key(args.seed),
        (len(rows), config.action_horizon, config.action_dim),
    )
    override_tokens = override_mask = None
    if args.prompt_source == "subtask":
        override_tokens = jnp.asarray(batch["subtask_prompt_tokens"])
        override_mask = jnp.asarray(batch["subtask_prompt_mask"])
    elif args.prompt_source == "atomic":
        override_tokens = jnp.asarray(batch["atomic_prompt_tokens"])
        override_mask = jnp.asarray(batch["atomic_prompt_mask"])
    committed_mask = jnp.full(
        (len(rows), config.action_horizon), args.disable_zm, dtype=jnp.bool_
    )
    normalized = np.asarray(jax.device_get(_sample_global(
        model, observation, noise, override_tokens, override_mask, committed_mask
    )))
    flow_repeats = []
    for repeat in range(args.flow_repeats):
        flow_repeats.append(jax.device_get(_triple_flow_mse(
            model,
            observation,
            jnp.asarray(actions_np),
            jnp.asarray(batch["subtask_prompt_tokens"]),
            jnp.asarray(batch["subtask_prompt_mask"]),
            jnp.asarray(batch["atomic_prompt_tokens"]),
            jnp.asarray(batch["atomic_prompt_mask"]),
            jax.random.key(args.seed + 100 + repeat),
        )))
    flow_by_prompt = {
        name: np.stack([values[index] for values in flow_repeats]).mean(axis=0)
        for index, name in enumerate(("global", "subtask", "atomic", "subtask_no_zm"))
    }
    decode = _output_transform(args.dataset_root, config)
    raw = dataset._raw  # noqa: SLF001

    chunks = []
    fk_validation_mm = {"right": [], "left": []}
    for i, ((frame, dataset_index, data_index, action_indices, metadata), prediction) in enumerate(zip(selected, normalized, strict=True)):
        decoded = np.asarray(decode(np.asarray(batch["state"][i]), metadata["raw_state"], prediction)["actions"])
        gt = np.asarray(metadata["raw_actions"])
        right_gt = _tcp200_trajectory(raw, action_indices, "right")
        left_gt = _tcp200_trajectory(raw, action_indices, "left")
        right_pred = _endpoint_tcp_from_actions(raw, data_index, decoded, "right")
        left_pred = _endpoint_tcp_from_actions(raw, data_index, decoded, "left")
        right_gt_fk = _endpoint_tcp_from_actions(raw, data_index, gt, "right")
        left_gt_fk = _endpoint_tcp_from_actions(raw, data_index, gt, "left")
        fk_validation_mm["right"].append(
            np.linalg.norm(right_gt_fk - right_gt, axis=-1) * 1000
        )
        fk_validation_mm["left"].append(
            np.linalg.norm(left_gt_fk - left_gt, axis=-1) * 1000
        )
        row = {"chunk": i, "start_frame": frame, "end_frame_exclusive": frame + 50}
        for arm, target, pred in (("right", right_gt, right_pred), ("left", left_gt, left_pred)):
            error = np.linalg.norm(pred - target, axis=-1) * 1000
            row[f"{arm}_translation_rmse_mm"] = float(np.sqrt(np.mean(error ** 2)))
            row[f"{arm}_translation_mean_mm"] = float(np.mean(error))
            row[f"{arm}_translation_final_mm"] = float(error[-1])
            row[f"{arm}_gt_xyz_m"] = target.tolist()
            row[f"{arm}_pred_xyz_m"] = pred.tolist()
        chunks.append(row)

    global_prompt = str(selected[0][4]["global_prompt"])
    subtask_prompts = [str(row[4]["subtask_prompt"]) for row in selected]
    atomic_prompts = [str(row[4]["atomic_prompt"]) for row in selected]
    atomic_valid = np.asarray(
        [bool(np.any(row[4]["atomic_supervision_mask"])) for row in selected], dtype=np.bool_
    )
    summary = {
        "episode": args.episode,
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "global_prompt": global_prompt,
        "prompt_source": args.prompt_source,
        "zm_adapter_enabled": not args.disable_zm,
        "prompt_description": (
            global_prompt if args.prompt_source == "global"
            else (
                "Per-window reviewed subtask prompts"
                if args.prompt_source == "subtask"
                else "Per-window atomic prompts (global fallback where unlabelled)"
            )
        ),
        "subtask_prompts": subtask_prompts,
        "atomic_prompts": atomic_prompts,
        "atomic_supervised_chunk_count": int(atomic_valid.sum()),
        "flow_mse": {
            name: {
                "all_chunks": float(values.mean()),
                "atomic_supervised_chunks": (
                    float(values[atomic_valid].mean()) if atomic_valid.any() else None
                ),
            }
            for name, values in flow_by_prompt.items()
        },
        "windowing": "non-overlapping complete 50-step chunks; GT state re-anchor per chunk",
        "chunk_count": len(chunks),
        "covered_steps": len(chunks) * 50,
        "fk_validation_max_mm": {
            arm: float(np.max(np.concatenate(values)))
            for arm, values in fk_validation_mm.items()
        },
        "right_translation_rmse_mm": float(np.sqrt(np.mean(np.concatenate([
            np.linalg.norm(np.asarray(row["right_pred_xyz_m"]) - np.asarray(row["right_gt_xyz_m"]), axis=-1) ** 2 for row in chunks
        ]))) * 1000),
        "left_translation_rmse_mm": float(np.sqrt(np.mean(np.concatenate([
            np.linalg.norm(np.asarray(row["left_pred_xyz_m"]) - np.asarray(row["left_gt_xyz_m"]), axis=-1) ** 2 for row in chunks
        ]))) * 1000),
        "chunks": chunks,
    }
    report = args.output_dir / f"episode_{args.episode:06d}_global_chunks.json"
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = args.output_dir / f"episode_{args.episode:06d}_global_chunks.csv"
    fields = [key for key in chunks[0] if not key.endswith("_xyz_m")]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader(); writer.writerows([{key: row[key] for key in fields} for row in chunks])
    figure = args.output_dir / f"episode_{args.episode:06d}_global_vs_gt.png"
    _plot(summary, figure)
    print(json.dumps({key: value for key, value in summary.items() if key != "chunks"}, ensure_ascii=False, indent=2))
    print(figure)


if __name__ == "__main__":
    main()
