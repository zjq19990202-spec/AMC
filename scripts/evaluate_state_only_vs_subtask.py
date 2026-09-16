#!/usr/bin/env python3
"""Paired no-image Flow-loss ablation: empty language versus subtask text.

Both branches retain the exact same normalized robot state.  The state-only
branch tokenizes an empty task with pi0.5's normal state serialization, while
the subtask branch tokenizes the annotated subtask with that same state.  The
image dictionary is empty in both branches, and actions, noise and flow time
are paired exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
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


def _text_only(observation, tokens, mask):
    return _model.Observation(
        images={},
        image_masks={},
        state=observation.state,
        tokenized_prompt=tokens,
        tokenized_prompt_mask=mask,
        token_ar_mask=None,
        token_loss_mask=None,
    )


@nnx.jit
def _paired_metrics(model, observation, actions, empty_tokens, empty_mask, sub_tokens, sub_mask, rng):
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    # OpenPI's standard preprocessor validates/normalizes the configured camera
    # keys.  Run it once, then remove every image before either prefix forward.
    observation = _model.preprocess_observation(
        preprocess_rng, observation, train=False
    )
    state_observation = _text_only(observation, empty_tokens, empty_mask)
    sub_observation = _text_only(observation, sub_tokens, sub_mask)
    actions = model._mask_action_condition(actions)  # noqa: SLF001
    noise = model._mask_action_condition(  # noqa: SLF001
        jax.random.normal(noise_rng, actions.shape)
    )
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    noisy = time[:, None, None] * noise + (1.0 - time[:, None, None]) * actions
    target = noise - actions

    def predict(prompted_observation):
        active_state = model._controlled_state(prompted_observation.state)  # noqa: SLF001
        query_hidden, prefix_mask, kv_cache, _ = model._prefix_forward(  # noqa: SLF001
            prompted_observation
        )
        _, right_direction, z_model, left_direction, _ = model._latent(  # noqa: SLF001
            query_hidden, active_state
        )
        velocity = model._suffix_velocity(  # noqa: SLF001
            prefix_mask, kv_cache, noisy, time, z_model
        )
        return (
            velocity,
            jnp.stack([right_direction, left_direction], axis=1),
            z_model,
        )

    state_velocity, state_direction, state_zm = predict(state_observation)
    sub_velocity, sub_direction, sub_zm = predict(sub_observation)

    def losses(velocity):
        squared = jnp.square(velocity - target)
        total = jnp.mean(squared, axis=(1, 2))
        left = jnp.mean(squared[..., :8], axis=(1, 2))
        right = jnp.mean(squared[..., 8:16], axis=(1, 2))
        return total, left, right

    state_loss, state_left, state_right = losses(state_velocity)
    sub_loss, sub_left, sub_right = losses(sub_velocity)
    direction_cosine = jnp.sum(state_direction * sub_direction, axis=-1)
    zm_cosine = jnp.sum(state_zm * sub_zm, axis=-1) / jnp.maximum(
        jnp.linalg.norm(state_zm, axis=-1) * jnp.linalg.norm(sub_zm, axis=-1),
        1e-8,
    )
    zm_relative_delta = jnp.linalg.norm(sub_zm - state_zm, axis=-1) / jnp.maximum(
        jnp.linalg.norm(state_zm, axis=-1), 1e-8
    )
    state_zm_norm = jnp.linalg.norm(state_zm, axis=-1)
    sub_zm_norm = jnp.linalg.norm(sub_zm, axis=-1)
    return (
        state_loss,
        sub_loss,
        state_left,
        sub_left,
        state_right,
        sub_right,
        direction_cosine,
        zm_cosine,
        zm_relative_delta,
        state_zm_norm,
        sub_zm_norm,
        time,
    )


def _select_indices(dataset, max_samples: int, seed: int) -> list[int]:
    raw = dataset._raw  # noqa: SLF001
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dataset))
    selected: list[int] = []
    for index in order:
        metadata = raw.metadata(int(index))
        if not bool(metadata.get("subtask_supervision_mask", False)):
            continue
        if not str(metadata.get("subtask_prompt", "")).strip():
            continue
        selected.append(int(index))
        if len(selected) >= max_samples:
            break
    if not selected:
        raise RuntimeError("no samples with valid subtask supervision were found")
    return selected


def _mean(values):
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", default="/mnt/cunchu/zjq/target")
    parser.add_argument("--norm-asset-id", default="openpi_norm_compact_accepted_v3")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--noise-repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260813)
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
    selected = _select_indices(dataset, args.max_samples, args.seed)
    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    tokenizer = _paligemma_tokenizer(config.max_token_len)

    accumulated = [[] for _ in range(12)]
    for batch_start in range(0, len(selected), args.batch_size):
        indices = selected[batch_start : batch_start + args.batch_size]
        batch = atomic_collate([dataset[index] for index in indices])
        observation_np, actions_np = batch_to_observation(batch)
        empty = [tokenizer.tokenize("", state) for state in batch["state"]]
        empty_tokens = np.stack([value[0] for value in empty])
        empty_mask = np.stack([value[1] for value in empty])
        observation = jax.tree.map(jnp.asarray, observation_np)
        actions = jnp.asarray(actions_np)
        repeat_values = []
        for repeat in range(args.noise_repeats):
            repeat_values.append(
                jax.device_get(
                    _paired_metrics(
                        model,
                        observation,
                        actions,
                        jnp.asarray(empty_tokens),
                        jnp.asarray(empty_mask),
                        jnp.asarray(batch["subtask_prompt_tokens"]),
                        jnp.asarray(batch["subtask_prompt_mask"]),
                        jax.random.key(args.seed + repeat + 1000 * batch_start),
                    )
                )
            )
        for metric_index in range(12):
            # Average paired noise repeats first, retaining one value per sample.
            accumulated[metric_index].extend(
                np.mean(
                    np.stack([value[metric_index] for value in repeat_values]), axis=0
                ).tolist()
            )
        print(f"evaluated {min(batch_start + args.batch_size, len(selected))}/{len(selected)}", flush=True)

    arrays = [np.asarray(values) for values in accumulated]
    (
        state_loss,
        sub_loss,
        state_left,
        sub_left,
        state_right,
        sub_right,
        direction_cosine,
        zm_cosine,
        zm_relative_delta,
        state_zm_norm,
        sub_zm_norm,
        flow_time,
    ) = arrays
    delta = sub_loss - state_loss
    report = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "samples": len(selected),
        "noise_repeats": args.noise_repeats,
        "comparison": (
            "no images in either branch; state-only uses tokenize('', state); "
            "subtask uses tokenize(subtask, same state); paired action/noise/flow-time"
        ),
        "state_only_flow_mse": _mean(state_loss),
        "subtask_plus_state_flow_mse": _mean(sub_loss),
        "subtask_minus_state_only": _mean(delta),
        "relative_change": _mean(delta / np.maximum(state_loss, 1e-8)),
        "subtask_better_fraction": _mean(delta < 0),
        "state_only_left_mse": _mean(state_left),
        "subtask_plus_state_left_mse": _mean(sub_left),
        "state_only_right_mse": _mean(state_right),
        "subtask_plus_state_right_mse": _mean(sub_right),
        "mean_direction_cosine_right_left": np.mean(direction_cosine, axis=0).tolist(),
        "mean_zm_cosine_right_left": np.mean(zm_cosine, axis=0).tolist(),
        "mean_zm_relative_delta_right_left": np.mean(zm_relative_delta, axis=0).tolist(),
        "state_only_zm_norm_right_left": np.mean(state_zm_norm, axis=0).tolist(),
        "state_only_zm_norm_std_right_left": np.std(state_zm_norm, axis=0).tolist(),
        "subtask_plus_state_zm_norm_right_left": np.mean(sub_zm_norm, axis=0).tolist(),
        "mean_flow_time": _mean(flow_time),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
