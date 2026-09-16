#!/usr/bin/env python3
"""Trace the causal hidden-state effect of prompt-mismatched final zM.

For each reviewed fruit observation, all fruit prompts share the image, state,
flow noise, sampler state, and Context KV route.  The intervention cyclically
shifts the final two-arm zM pair across prompt variants.  Hidden comparisons
use the same noisy action input at every denoising step, so the reported
difference isolates zM rather than accumulated rollout divergence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from evaluate_fruit_target_switch import _find_row
from evaluate_fruit_target_switch_multiframe import _native_template_prompts


THRESHOLDS = (0.01, 0.05, 0.10)


@nnx.jit
def _trace_frame(model, observation, tokens, masks, noise):
    observation = _model.preprocess_observation(None, observation, train=False)
    batch = tokens.shape[0]
    observation = jax.tree.map(lambda value: jnp.repeat(value, batch, axis=0), observation)
    observation = model._with_prompt(observation, tokens, masks)  # noqa: SLF001
    query_hidden, prefix_mask, kv_cache, _ = model._prefix_forward(observation)  # noqa: SLF001
    active_state = model._controlled_state(observation.state)  # noqa: SLF001
    _, directions, z_model, _, _ = model._latent(query_hidden, active_state)  # noqa: SLF001
    wrong_z_model = jnp.roll(z_model, 1, axis=0)
    initial = model._mask_action_condition(jnp.repeat(noise, batch, axis=0))  # noqa: SLF001

    def hidden_metrics(correct_hidden, wrong_hidden, correct_velocity, wrong_velocity):
        delta = wrong_hidden - correct_hidden
        correct_ss_channel = jnp.mean(jnp.square(correct_hidden), axis=(1, 2))
        delta_ss_channel = jnp.mean(jnp.square(delta), axis=(1, 2))
        correct_norm = jnp.sqrt(jnp.sum(jnp.square(correct_hidden), axis=-1) + 1.0e-12)
        delta_norm = jnp.sqrt(jnp.sum(jnp.square(delta), axis=-1) + 1.0e-12)
        relative_token = delta_norm / correct_norm
        affected_token_counts = jnp.stack(
            [jnp.sum(relative_token > threshold, axis=(1, 2)) for threshold in THRESHOLDS],
            axis=-1,
        )
        cosine = jnp.sum(correct_hidden * wrong_hidden, axis=-1) / (
            jnp.sqrt(jnp.sum(jnp.square(correct_hidden), axis=-1) + 1.0e-12)
            * jnp.sqrt(jnp.sum(jnp.square(wrong_hidden), axis=-1) + 1.0e-12)
        )
        correct_velocity_f32 = correct_velocity.astype(jnp.float32)
        velocity_delta = wrong_velocity.astype(jnp.float32) - correct_velocity_f32
        velocity_relative_rms = jnp.sqrt(jnp.mean(jnp.square(velocity_delta))) / jnp.sqrt(
            jnp.mean(jnp.square(correct_velocity_f32)) + 1.0e-12
        )
        maximum_absolute_delta = jnp.max(jnp.abs(delta), axis=(1, 2, 3))
        return (
            correct_ss_channel,
            delta_ss_channel,
            affected_token_counts,
            jnp.mean(relative_token, axis=(1, 2)),
            jnp.mean(cosine, axis=(1, 2)),
            maximum_absolute_delta,
            velocity_relative_rms,
        )

    def scan_step(carry, index):
        correct_current, wrong_current = carry
        time = jnp.asarray(1.0 - index / 10.0, correct_current.dtype)
        timestep = jnp.broadcast_to(time, (batch,))
        correct_velocity, correct_layers, correct_final = model._suffix_velocity_with_hidden(  # noqa: SLF001
            prefix_mask,
            kv_cache,
            correct_current,
            timestep,
            z_model,
            layerwise_arm_latents=None,
        )
        matched_wrong_velocity, matched_wrong_layers, matched_wrong_final = (
            model._suffix_velocity_with_hidden(  # noqa: SLF001
                prefix_mask,
                kv_cache,
                correct_current,
                timestep,
                wrong_z_model,
                layerwise_arm_latents=None,
            )
        )
        rollout_wrong_velocity, rollout_wrong_layers, rollout_wrong_final = (
            model._suffix_velocity_with_hidden(  # noqa: SLF001
                prefix_mask,
                kv_cache,
                wrong_current,
                timestep,
                wrong_z_model,
                layerwise_arm_latents=None,
            )
        )
        correct_hidden = jnp.concatenate([correct_layers, correct_final[None]], axis=0).astype(
            jnp.float32
        )
        matched_wrong_hidden = jnp.concatenate(
            [matched_wrong_layers, matched_wrong_final[None]], axis=0
        ).astype(jnp.float32)
        rollout_wrong_hidden = jnp.concatenate(
            [rollout_wrong_layers, rollout_wrong_final[None]], axis=0
        ).astype(jnp.float32)
        matched_metrics = hidden_metrics(
            correct_hidden, matched_wrong_hidden, correct_velocity, matched_wrong_velocity
        )
        rollout_metrics = hidden_metrics(
            correct_hidden, rollout_wrong_hidden, correct_velocity, rollout_wrong_velocity
        )
        next_correct = model._mask_action_condition(  # noqa: SLF001
            correct_current - 0.1 * correct_velocity
        )
        next_wrong = model._mask_action_condition(  # noqa: SLF001
            wrong_current - 0.1 * rollout_wrong_velocity
        )
        return (next_correct, next_wrong), (matched_metrics, rollout_metrics)

    final_actions, metrics = jax.lax.scan(scan_step, (initial, initial), jnp.arange(10))
    return directions, z_model, wrong_z_model, metrics, final_actions


def _image_uint8(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if np.issubdtype(value.dtype, np.floating):
        if float(np.nanmin(value)) < -0.01:
            value = (value + 1.0) * 127.5
        elif float(np.nanmax(value)) <= 1.01:
            value = value * 255.0
    return np.clip(value, 0, 255).astype(np.uint8)


def _cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    numerator = np.sum(first * second, axis=-1)
    denominator = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
    return numerator / np.maximum(denominator, 1.0e-12)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--selection-dataset-name", default="target2058")
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--atomic-composition-sidecar", default=None)
    parser.add_argument(
        "--spherical-visual-latent",
        action="store_true",
        help="Load the bounded unit-sphere visual zM route used by spherical AFRO checkpoints.",
    )
    parser.add_argument("--fruit-target", action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260834)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.fruit_target)) != len(args.fruit_target):
        parser.error("--fruit-target values must be unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selection_payload = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    frame_specs = selection_payload["datasets"][args.selection_dataset_name]
    config = AtomicPi05Config(
        max_token_len=192,
        fast_action_ce_loss_weight=0.0,
        coefficient_target_kind="joint_delta",
        coefficient_target_dim=14,
        enable_layerwise_atomic_flow=False,
        spherical_visual_latent=args.spherical_visual_latent,
        visual_max_update_angle_deg=45.0 if args.spherical_visual_latent else 0.0,
    )
    dataset = build_atomic_dataset(
        (args.dataset_root,),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        action_horizon=50,
        max_token_len=192,
        include_fast=False,
        atomic_composition_sidecar=args.atomic_composition_sidecar,
        pad_subtask_horizon=True,
    )
    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    tokenizer = _paligemma_tokenizer(192)

    correct_ss_sum = None
    delta_ss_sum = None
    token_count_sum = None
    relative_token_sum = None
    cosine_sum = None
    maximum_delta = None
    velocity_relative = []
    matched_step_correct_ss_sum = None
    matched_step_delta_ss_sum = None
    matched_step_token_count_sum = None
    matched_step_velocity_sum = None
    rollout_step_correct_ss_sum = None
    rollout_step_delta_ss_sum = None
    rollout_step_token_count_sum = None
    rollout_step_velocity_sum = None
    final_action_relative = []
    z_rows = []
    source_images = []

    for specification in frame_specs:
        episode = int(specification["episode"])
        frame = int(specification["frame"])
        dataset_index, _, _, metadata = _find_row(dataset, episode, frame)
        row = dataset[dataset_index]
        batch = atomic_collate([row])
        obs_np, _ = batch_to_observation(batch)
        observation = jax.tree.map(jnp.asarray, obs_np)
        prompts, native_target = _native_template_prompts(
            metadata["subtask_prompt"], args.fruit_target
        )
        prompt_names = list(prompts)
        token_rows, mask_rows = zip(
            *[
                tokenizer.tokenize(prompts[name], np.asarray(row["state"]))
                for name in prompt_names
            ],
            strict=True,
        )
        noise_key = jax.random.fold_in(jax.random.key(args.seed), episode)
        noise = jax.random.normal(
            jax.random.fold_in(noise_key, frame), (1, 50, config.action_dim)
        )
        directions, z_model, wrong_z_model, metric_pair, final_actions = jax.device_get(
            _trace_frame(
                model,
                observation,
                jnp.asarray(np.stack(token_rows)),
                jnp.asarray(np.stack(mask_rows)),
                noise,
            )
        )
        metrics, rollout_metrics = metric_pair
        (
            correct_ss_channel,
            delta_ss_channel,
            affected_token_counts,
            relative_token_mean,
            hidden_cosine_mean,
            max_absolute_delta,
            velocity_relative_rms,
        ) = map(np.asarray, metrics)
        (
            rollout_correct_ss_channel,
            rollout_delta_ss_channel,
            rollout_affected_token_counts,
            _,
            _,
            _,
            rollout_velocity_relative_rms,
        ) = map(np.asarray, rollout_metrics)
        matched_step_correct_ss_sum = (
            correct_ss_channel
            if matched_step_correct_ss_sum is None
            else matched_step_correct_ss_sum + correct_ss_channel
        )
        matched_step_delta_ss_sum = (
            delta_ss_channel
            if matched_step_delta_ss_sum is None
            else matched_step_delta_ss_sum + delta_ss_channel
        )
        matched_step_token_count_sum = (
            affected_token_counts
            if matched_step_token_count_sum is None
            else matched_step_token_count_sum + affected_token_counts
        )
        matched_step_velocity_sum = (
            velocity_relative_rms
            if matched_step_velocity_sum is None
            else matched_step_velocity_sum + velocity_relative_rms
        )
        rollout_step_correct_ss_sum = (
            rollout_correct_ss_channel
            if rollout_step_correct_ss_sum is None
            else rollout_step_correct_ss_sum + rollout_correct_ss_channel
        )
        rollout_step_delta_ss_sum = (
            rollout_delta_ss_channel
            if rollout_step_delta_ss_sum is None
            else rollout_step_delta_ss_sum + rollout_delta_ss_channel
        )
        rollout_step_token_count_sum = (
            rollout_affected_token_counts
            if rollout_step_token_count_sum is None
            else rollout_step_token_count_sum + rollout_affected_token_counts
        )
        rollout_step_velocity_sum = (
            rollout_velocity_relative_rms
            if rollout_step_velocity_sum is None
            else rollout_step_velocity_sum + rollout_velocity_relative_rms
        )
        final_correct_actions, final_wrong_actions = map(np.asarray, final_actions)
        final_action_delta_rms = np.sqrt(
            np.mean(np.square(final_wrong_actions - final_correct_actions), axis=(1, 2))
        )
        final_action_correct_rms = np.sqrt(
            np.mean(np.square(final_correct_actions), axis=(1, 2))
        )
        final_action_relative.extend(
            (final_action_delta_rms / np.maximum(final_action_correct_rms, 1.0e-12)).tolist()
        )
        correct_ss_frame = correct_ss_channel.sum(axis=0)
        delta_ss_frame = delta_ss_channel.sum(axis=0)
        token_counts_frame = affected_token_counts.sum(axis=0)
        relative_frame = relative_token_mean.sum(axis=0)
        cosine_frame = hidden_cosine_mean.sum(axis=0)
        max_frame = max_absolute_delta.max(axis=0)
        correct_ss_sum = correct_ss_frame if correct_ss_sum is None else correct_ss_sum + correct_ss_frame
        delta_ss_sum = delta_ss_frame if delta_ss_sum is None else delta_ss_sum + delta_ss_frame
        token_count_sum = token_counts_frame if token_count_sum is None else token_count_sum + token_counts_frame
        relative_token_sum = relative_frame if relative_token_sum is None else relative_token_sum + relative_frame
        cosine_sum = cosine_frame if cosine_sum is None else cosine_sum + cosine_frame
        maximum_delta = max_frame if maximum_delta is None else np.maximum(maximum_delta, max_frame)
        velocity_relative.extend(np.asarray(velocity_relative_rms).tolist())

        active_index = 0 if specification["active_arm"] == "right" else 1
        z_cosine = _cosine(np.asarray(z_model), np.asarray(wrong_z_model))
        direction_cosine = _cosine(
            np.asarray(directions), np.roll(np.asarray(directions), 1, axis=0)
        )
        active_direction_cosine = (
            direction_cosine[:, active_index]
            if direction_cosine.ndim > 1
            else direction_cosine
        )
        z_rows.append(
            {
                "episode": episode,
                "frame": frame,
                "active_arm": specification["active_arm"],
                "native_target": native_target,
                "prompt_names": prompt_names,
                "mean_two_arm_zm_cosine_correct_vs_wrong": float(np.mean(z_cosine)),
                "mean_active_arm_zm_cosine_correct_vs_wrong": float(
                    np.mean(z_cosine[:, active_index])
                ),
                "mean_active_arm_direction_cosine_correct_vs_wrong": float(
                    np.mean(active_direction_cosine)
                ),
            }
        )
        source_images.append(
            (
                specification,
                _image_uint8(np.asarray(obs_np.images["base_0_rgb"])[0]),
            )
        )

    assert correct_ss_sum is not None
    assert delta_ss_sum is not None
    assert token_count_sum is not None
    assert relative_token_sum is not None
    assert cosine_sum is not None
    assert maximum_delta is not None
    assert matched_step_correct_ss_sum is not None
    assert matched_step_delta_ss_sum is not None
    assert matched_step_token_count_sum is not None
    assert matched_step_velocity_sum is not None
    assert rollout_step_correct_ss_sum is not None
    assert rollout_step_delta_ss_sum is not None
    assert rollout_step_token_count_sum is not None
    assert rollout_step_velocity_sum is not None
    trace_count = len(frame_specs) * 10
    token_total_per_position = len(frame_specs) * 10 * len(args.fruit_target) * 50
    channel_relative = np.sqrt(delta_ss_sum / np.maximum(correct_ss_sum, 1.0e-12))
    layer_relative = np.sqrt(
        delta_ss_sum.mean(axis=1) / np.maximum(correct_ss_sum.mean(axis=1), 1.0e-12)
    )
    matched_step_layer_relative = np.sqrt(
        matched_step_delta_ss_sum.mean(axis=2)
        / np.maximum(matched_step_correct_ss_sum.mean(axis=2), 1.0e-12)
    )
    rollout_step_layer_relative = np.sqrt(
        rollout_step_delta_ss_sum.mean(axis=2)
        / np.maximum(rollout_step_correct_ss_sum.mean(axis=2), 1.0e-12)
    )
    token_total_per_step_position = len(frame_specs) * len(args.fruit_target) * 50
    matched_step_token_fraction = (
        matched_step_token_count_sum / token_total_per_step_position
    )
    rollout_step_token_fraction = (
        rollout_step_token_count_sum / token_total_per_step_position
    )
    labels = [f"block_{index + 1:02d}" for index in range(channel_relative.shape[0] - 1)] + [
        "final_norm"
    ]
    per_layer = []
    for index, label in enumerate(labels):
        per_layer.append(
            {
                "position": label,
                "relative_hidden_rms": float(layer_relative[index]),
                "mean_token_relative_l2": float(relative_token_sum[index] / trace_count),
                "mean_hidden_cosine": float(cosine_sum[index] / trace_count),
                "max_absolute_hidden_delta": float(maximum_delta[index]),
                **{
                    f"token_fraction_relative_gt_{int(threshold * 100)}pct": float(
                        token_count_sum[index, threshold_index] / token_total_per_position
                    )
                    for threshold_index, threshold in enumerate(THRESHOLDS)
                },
                **{
                    f"channels_relative_gt_{int(threshold * 100)}pct": int(
                        np.sum(channel_relative[index] > threshold)
                    )
                    for threshold in THRESHOLDS
                },
                "hidden_width": int(channel_relative.shape[1]),
            }
        )

    block_rows = per_layer[:-1]
    block18_index = len(block_rows) - 1
    final_norm_index = len(per_layer) - 1
    per_denoising_step = []
    for step_index in range(10):
        per_denoising_step.append(
            {
                "step": step_index + 1,
                "time": float(1.0 - step_index / 10.0),
                "matched_current": {
                    "block18_relative_hidden_rms": float(
                        matched_step_layer_relative[step_index, block18_index]
                    ),
                    "final_norm_relative_hidden_rms": float(
                        matched_step_layer_relative[step_index, final_norm_index]
                    ),
                    "block18_token_fraction_gt_1pct": float(
                        matched_step_token_fraction[step_index, block18_index, 0]
                    ),
                    "final_norm_token_fraction_gt_1pct": float(
                        matched_step_token_fraction[step_index, final_norm_index, 0]
                    ),
                    "velocity_relative_rms": float(
                        matched_step_velocity_sum[step_index] / len(frame_specs)
                    ),
                },
                "accumulated_rollout": {
                    "block18_relative_hidden_rms": float(
                        rollout_step_layer_relative[step_index, block18_index]
                    ),
                    "final_norm_relative_hidden_rms": float(
                        rollout_step_layer_relative[step_index, final_norm_index]
                    ),
                    "block18_token_fraction_gt_1pct": float(
                        rollout_step_token_fraction[step_index, block18_index, 0]
                    ),
                    "final_norm_token_fraction_gt_1pct": float(
                        rollout_step_token_fraction[step_index, final_norm_index, 0]
                    ),
                    "velocity_relative_rms": float(
                        rollout_step_velocity_sum[step_index] / len(frame_specs)
                    ),
                },
            }
        )
    summary = {
        "contract": {
            "checkpoint": str(args.checkpoint.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "selection_manifest": str(args.selection_manifest.resolve()),
            "selection_manifest_sha256": _sha256(args.selection_manifest),
            "normalization": str(args.norm_assets_dir / args.norm_asset_id / "norm_stats.json"),
            "seed": args.seed,
            "frames": len(frame_specs),
            "prompt_variants_per_frame": len(args.fruit_target),
            "paired_prompt_cases": len(frame_specs) * len(args.fruit_target),
            "denoising_steps_traced": 10,
            "action_horizon": 50,
            "hidden_width": int(channel_relative.shape[1]),
            "action_expert_blocks": len(block_rows),
            "wrong_zm": "cyclic prompt-batch shift of the final two-arm zM pair",
            "spherical_visual_latent": bool(args.spherical_visual_latent),
            "hidden_comparison": "correct and wrong zM use identical Context KV and identical noisy action input at each traced step",
            "accumulated_rollout_comparison": "correct and wrong zM start from identical noise, then each route advances with its own velocity; the last-step hidden difference includes prior action-path divergence",
            "threshold_definition": "relative L2/RMS change with respect to the correct-zM hidden state",
        },
        "aggregate": {
            "blocks_with_nonzero_effect": int(
                sum(row["max_absolute_hidden_delta"] > 1.0e-6 for row in block_rows)
            ),
            "blocks_relative_rms_gt_1pct": int(
                sum(row["relative_hidden_rms"] > 0.01 for row in block_rows)
            ),
            "blocks_relative_rms_gt_5pct": int(
                sum(row["relative_hidden_rms"] > 0.05 for row in block_rows)
            ),
            "mean_block_relative_hidden_rms": float(
                np.mean([row["relative_hidden_rms"] for row in block_rows])
            ),
            "max_block_relative_hidden_rms": float(
                np.max([row["relative_hidden_rms"] for row in block_rows])
            ),
            "mean_block_token_fraction_gt_1pct": float(
                np.mean([row["token_fraction_relative_gt_1pct"] for row in block_rows])
            ),
            "mean_block_token_fraction_gt_5pct": float(
                np.mean([row["token_fraction_relative_gt_5pct"] for row in block_rows])
            ),
            "mean_block_channels_gt_1pct": float(
                np.mean([row["channels_relative_gt_1pct"] for row in block_rows])
            ),
            "mean_block_channels_gt_5pct": float(
                np.mean([row["channels_relative_gt_5pct"] for row in block_rows])
            ),
            "mean_velocity_relative_rms": float(np.mean(velocity_relative)),
            "mean_active_arm_zm_cosine_correct_vs_wrong": float(
                np.mean([row["mean_active_arm_zm_cosine_correct_vs_wrong"] for row in z_rows])
            ),
            "mean_active_arm_direction_cosine_correct_vs_wrong": float(
                np.mean(
                    [row["mean_active_arm_direction_cosine_correct_vs_wrong"] for row in z_rows]
                )
            ),
        },
        "last_denoising_step": per_denoising_step[-1],
        "mean_final_normalized_action_relative_rms": float(
            np.mean(final_action_relative)
        ),
        "per_denoising_step": per_denoising_step,
        "per_layer": per_layer,
        "per_frame_zm": z_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "per_layer.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_layer[0]))
        writer.writeheader()
        writer.writerows(per_layer)
    np.savez_compressed(
        args.output_dir / "hidden_channel_metrics.npz",
        labels=np.asarray(labels),
        channel_relative_rms=channel_relative.astype(np.float32),
        correct_channel_mean_square=correct_ss_sum.astype(np.float32) / trace_count,
        delta_channel_mean_square=delta_ss_sum.astype(np.float32) / trace_count,
    )

    figure, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), constrained_layout=True)
    x = np.arange(len(labels))
    axes[0].plot(x, layer_relative * 100, marker="o", color="#6d28d9", linewidth=2)
    axes[0].axhline(1, color="#94a3b8", linestyle="--", linewidth=1)
    axes[0].axhline(5, color="#64748b", linestyle=":", linewidth=1)
    axes[0].set_ylabel("relative hidden RMS change (%)")
    axes[0].set_title("Causal effect of prompt-mismatched zM on Action-Expert hidden states")
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        x,
        [row["token_fraction_relative_gt_1pct"] * 100 for row in per_layer],
        marker="o",
        label="tokens >1%",
    )
    axes[1].plot(
        x,
        [row["token_fraction_relative_gt_5pct"] * 100 for row in per_layer],
        marker="s",
        label="tokens >5%",
    )
    axes[1].set_ylabel("affected token vectors (%)")
    axes[1].set_xlabel("Action-Expert depth")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    axes[1].set_xticks(x, [str(index + 1) for index in range(len(labels) - 1)] + ["FN"])
    figure.savefig(args.output_dir / "hidden_effect_by_depth.png", dpi=210)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    step_axis = np.arange(1, 11)
    for axis, layer_name, title in (
        (axes[0], "block18_relative_hidden_rms", "Block 18"),
        (axes[1], "final_norm_relative_hidden_rms", "Final norm"),
    ):
        axis.plot(
            step_axis,
            [row["matched_current"][layer_name] * 100 for row in per_denoising_step],
            marker="o",
            label="zM-only (matched action input)",
        )
        axis.plot(
            step_axis,
            [row["accumulated_rollout"][layer_name] * 100 for row in per_denoising_step],
            marker="s",
            label="accumulated wrong-zM rollout",
        )
        axis.set_title(title)
        axis.set_xlabel("denoising step")
        axis.set_ylabel("relative hidden RMS change (%)")
        axis.set_xticks(step_axis)
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False)
    figure.savefig(args.output_dir / "last_hidden_over_denoising.png", dpi=210)
    plt.close(figure)

    columns = 4
    rows = int(np.ceil(len(source_images) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(14, 3.0 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for axis, (specification, image) in zip(axes, source_images, strict=False):
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(
            f"ep{specification['episode']} f{specification['frame']} | {specification['active_arm']}\n"
            f"{specification['prompt']}",
            fontsize=8,
        )
    for axis in axes[len(source_images) :]:
        axis.axis("off")
    figure.savefig(args.output_dir / "source_observations.png", dpi=190)
    plt.close(figure)
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
