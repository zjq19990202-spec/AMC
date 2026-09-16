#!/usr/bin/env python3
"""Measure the actual spherical zM rotation induced at each RTC offset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from atomic_latent_vla.pi05.force_training_data import (
    ForceNormalization,
    batch_to_force_inputs,
    build_force_dataset,
    force_collate,
)
from atomic_latent_vla.pi05.model import l2_normalize, spherical_tangent_update
from evaluate_force_b2_episode50_ablation import _candidate_indices, _choose_episode
from evaluate_force_rtc_offsets import (
    OFFSETS,
    _force_metadata_at_offset,
    _load_model,
    _prepare_force_context,
)


@nnx.jit
def _rotation(
    model,
    context,
    current_force,
    current_state,
    current_mask,
    update_offset,
):
    modulation = model._require_force_conditioner().modulate(  # noqa: SLF001
        context.z_model,
        context.force_latent,
        current_force,
        current_state,
        current_mask,
        update_offset,
        slow_history_tokens=context.slow_history_tokens,
        slow_history_token_mask=context.slow_history_token_mask,
    )
    updated = spherical_tangent_update(
        context.z_model,
        modulation.delta_z,
        jnp.deg2rad(model.config.force_max_update_angle_deg),
    )
    base = l2_normalize(context.z_model.astype(jnp.float32))
    updated = l2_normalize(updated.astype(jnp.float32))
    cosine = jnp.clip(jnp.sum(base * updated, axis=-1), -1.0, 1.0)
    tangent = updated - cosine[..., None] * base
    sine = jnp.linalg.norm(tangent, axis=-1)
    angle_deg = jnp.rad2deg(jnp.arctan2(sine, cosine))
    raw_delta_norm = jnp.linalg.norm(modulation.delta_z.astype(jnp.float32), axis=-1)
    tangent_delta = modulation.delta_z.astype(jnp.float32)
    tangent_delta = tangent_delta - jnp.sum(tangent_delta * base, axis=-1, keepdims=True) * base
    tangent_delta_norm = jnp.linalg.norm(tangent_delta, axis=-1)
    return angle_deg, raw_delta_norm, tangent_delta_norm


def _stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--subtask-sidecar", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--force-norm", type=Path, required=True)
    parser.add_argument("--episode", default="0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-windows", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder-width", type=int, default=512)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--encoder-mlp-dim", type=int, default=1024)
    parser.add_argument("--force-latent-dim", type=int, default=512)
    parser.add_argument("--full-token-force-adapter", action="store_true")
    parser.add_argument("--full-token-force-adapter-heads", type=int, default=2)
    args = parser.parse_args()

    config, model = _load_model("b2", args.checkpoint, args)
    dataset = build_force_dataset(
        (args.dataset_root,),
        subtask_sidecars=(args.subtask_sidecar,),
        pad_subtask_horizon=True,
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        force_norm_path=args.force_norm,
        max_token_len=config.max_token_len,
        force_update_offsets=OFFSETS,
        load_future_force_targets=False,
        seed=0,
    )
    candidates = _candidate_indices(dataset)
    episode = _choose_episode(candidates, args.episode)
    raw = dataset._raw  # noqa: SLF001
    selected = sorted(
        candidates[episode],
        key=lambda index: int(raw.base._frame_index[int(raw.anchors[index])]),  # noqa: SLF001
    )
    if args.max_windows:
        selected = selected[: args.max_windows]
    force_norm = ForceNormalization.load(args.force_norm)
    values = {
        offset: {"angle": [], "raw": [], "tangent": []} for offset in OFFSETS
    }
    phase0_angles = {offset: [] for offset in OFFSETS}
    slow_only_angles = {offset: [] for offset in OFFSETS}
    phase0_slow_only_angles = {offset: [] for offset in OFFSETS}
    selection = []

    for start in range(0, len(selected), args.batch_size):
        indices = selected[start : start + args.batch_size]
        real_count = len(indices)
        if real_count < args.batch_size:
            indices += [indices[-1]] * (args.batch_size - real_count)
        batch = force_collate([dataset[index] for index in indices])
        observation_np, _, initial_force = batch_to_force_inputs(batch)
        observation = jax.tree.map(jnp.asarray, observation_np)
        context = _prepare_force_context(
            model,
            observation,
            jnp.asarray(initial_force["slow_force_history"]),
            jnp.asarray(initial_force["slow_state_history"]),
            jnp.asarray(initial_force["slow_history_mask"]),
        )
        for offset in OFFSETS:
            force = _force_metadata_at_offset(raw, force_norm, indices, offset)
            angle, raw_norm, tangent_norm = _rotation(
                model,
                context,
                jnp.asarray(force["current_force_history"]),
                jnp.asarray(force["current_state_history"]),
                jnp.asarray(force["current_history_mask"]),
                jnp.asarray(force["update_offset"]),
            )
            zero_offset = jnp.zeros_like(jnp.asarray(force["update_offset"]))
            zero_fast_mask = jnp.zeros_like(
                jnp.asarray(force["current_history_mask"])
            )
            phase0_angle, _, _ = _rotation(
                model,
                context,
                jnp.asarray(force["current_force_history"]),
                jnp.asarray(force["current_state_history"]),
                jnp.asarray(force["current_history_mask"]),
                zero_offset,
            )
            slow_only_angle, _, _ = _rotation(
                model,
                context,
                jnp.asarray(force["current_force_history"]),
                jnp.asarray(force["current_state_history"]),
                zero_fast_mask,
                jnp.asarray(force["update_offset"]),
            )
            phase0_slow_only_angle, _, _ = _rotation(
                model,
                context,
                jnp.asarray(force["current_force_history"]),
                jnp.asarray(force["current_state_history"]),
                zero_fast_mask,
                zero_offset,
            )
            values[offset]["angle"].append(np.asarray(jax.device_get(angle))[:real_count])
            values[offset]["raw"].append(np.asarray(jax.device_get(raw_norm))[:real_count])
            values[offset]["tangent"].append(
                np.asarray(jax.device_get(tangent_norm))[:real_count]
            )
            phase0_angles[offset].append(
                np.asarray(jax.device_get(phase0_angle))[:real_count]
            )
            slow_only_angles[offset].append(
                np.asarray(jax.device_get(slow_only_angle))[:real_count]
            )
            phase0_slow_only_angles[offset].append(
                np.asarray(jax.device_get(phase0_slow_only_angle))[:real_count]
            )
        for index in indices[:real_count]:
            anchor = int(raw.anchors[index])
            selection.append(
                {
                    "dataset_index": int(index),
                    "episode": episode,
                    "frame": int(raw.base._frame_index[anchor]),  # noqa: SLF001
                }
            )

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "episode": episode,
        "window_count": len(selected),
        "arm_order": ["right", "left"],
        "contract": "exact spherical geodesic angle between zM and force-updated zM; correct slow+fast tokens; current aligned subtask with boundary padding",
        "offsets": {},
        "selection": selection,
    }
    for offset in OFFSETS:
        angle = np.concatenate(values[offset]["angle"], axis=0)
        raw_norm = np.concatenate(values[offset]["raw"], axis=0)
        tangent_norm = np.concatenate(values[offset]["tangent"], axis=0)
        phase0_angle = np.concatenate(phase0_angles[offset], axis=0)
        slow_only_angle = np.concatenate(slow_only_angles[offset], axis=0)
        phase0_slow_only_angle = np.concatenate(
            phase0_slow_only_angles[offset], axis=0
        )
        summary["offsets"][str(offset)] = {
            "angle_deg_all_arms": _stats(angle.reshape(-1)),
            "angle_deg_right": _stats(angle[:, 0]),
            "angle_deg_left": _stats(angle[:, 1]),
            "raw_delta_norm_all_arms": _stats(raw_norm.reshape(-1)),
            "tangent_delta_norm_all_arms": _stats(tangent_norm.reshape(-1)),
            "angle_deg_phase0_correct_tokens": _stats(phase0_angle.reshape(-1)),
            "angle_deg_real_phase_slow_only": _stats(slow_only_angle.reshape(-1)),
            "angle_deg_phase0_slow_only": _stats(
                phase0_slow_only_angle.reshape(-1)
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"offsets": summary["offsets"]}, indent=2))


if __name__ == "__main__":
    main()
