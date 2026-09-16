#!/usr/bin/env python3
"""Paired full-observation zM flow loss for empty/correct/reversed atomic text."""

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
from evaluate_zt_empty_vs_atomic import reverse_atomic_prompt


@nnx.jit
def _paired(model, observation, actions, atomic_tokens, atomic_mask,
            empty_tokens, empty_mask, reverse_tokens, reverse_mask, rng):
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(preprocess_rng, observation, train=False)
    variants = (
        model._with_prompt(observation, atomic_tokens, atomic_mask),
        model._with_prompt(observation, empty_tokens, empty_mask),
        model._with_prompt(observation, reverse_tokens, reverse_mask),
    )
    actions = model._mask_action_condition(actions)
    noise = model._mask_action_condition(jax.random.normal(noise_rng, actions.shape))
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    noisy = time[:, None, None] * noise + (1.0 - time[:, None, None]) * actions
    target = noise - actions

    def predict(obs):
        active_state = model._controlled_state(obs.state)
        query_hidden, prefix_mask, kv_cache, _ = model._prefix_forward(obs)
        _, right_direction, z_model, left_direction, _ = model._latent(
            query_hidden, active_state
        )
        velocity = model._suffix_velocity(prefix_mask, kv_cache, noisy, time, z_model)
        return (
            velocity,
            jnp.stack((right_direction, left_direction), axis=1),
            z_model,
        )

    def training_flow_loss(velocity):
        squared = jnp.square(velocity - target)
        return jnp.mean(squared, axis=(1, 2))

    outputs = tuple(predict(obs) for obs in variants)
    losses = tuple(training_flow_loss(output[0]) for output in outputs)
    directions = tuple(output[1] for output in outputs)
    z_models = tuple(output[2] for output in outputs)
    return (
        *losses,
        jnp.sum(directions[0] * directions[1], axis=-1),
        jnp.sum(directions[0] * directions[2], axis=-1),
        jnp.sum(z_models[0] * z_models[1], axis=-1)
        / jnp.maximum(jnp.linalg.norm(z_models[0], axis=-1) * jnp.linalg.norm(z_models[1], axis=-1), 1e-8),
        jnp.sum(z_models[0] * z_models[2], axis=-1)
        / jnp.maximum(jnp.linalg.norm(z_models[0], axis=-1) * jnp.linalg.norm(z_models[2], axis=-1), 1e-8),
        time,
    )


def _select(dataset, count: int, seed: int) -> list[int]:
    raw = dataset._raw
    selected = []
    for index in np.random.default_rng(seed).permutation(len(dataset)):
        metadata = raw.metadata(int(index))
        if np.any(metadata["atomic_supervision_mask"]):
            try:
                reverse_atomic_prompt(metadata["atomic_prompt"])
            except ValueError:
                continue
            selected.append(int(index))
            if len(selected) >= count:
                break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", default="/mnt/cunchu/zjq/target")
    parser.add_argument("--norm-asset-id", default="openpi_norm_compact_accepted_v3")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--noise-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = AtomicPi05Config(max_token_len=250, fast_action_ce_loss_weight=0.0)
    dataset = build_atomic_dataset(
        (args.dataset_root,), norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id, action_horizon=config.action_horizon,
        max_token_len=config.max_token_len, include_fast=False,
    )
    selected = _select(dataset, args.max_samples, args.seed)
    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    tokenizer = _paligemma_tokenizer(config.max_token_len)
    collected = [[] for _ in range(8)]

    for start in range(0, len(selected), args.batch_size):
        rows = [dataset[index] for index in selected[start:start + args.batch_size]]
        batch = atomic_collate(rows)
        observation_np, actions_np = batch_to_observation(batch)
        empty = [tokenizer.tokenize("", state) for state in batch["state"]]
        reverse = [
            tokenizer.tokenize(reverse_atomic_prompt(row["atomic_prompt"]), state)
            for row, state in zip(rows, batch["state"], strict=True)
        ]
        arguments = (
            model,
            jax.tree.map(jnp.asarray, observation_np),
            jnp.asarray(actions_np),
            jnp.asarray(batch["atomic_prompt_tokens"]),
            jnp.asarray(batch["atomic_prompt_mask"]),
            jnp.asarray(np.stack([item[0] for item in empty])),
            jnp.asarray(np.stack([item[1] for item in empty])),
            jnp.asarray(np.stack([item[0] for item in reverse])),
            jnp.asarray(np.stack([item[1] for item in reverse])),
        )
        repeats = [
            jax.device_get(_paired(*arguments, jax.random.key(args.seed + 1000 * start + repeat)))
            for repeat in range(args.noise_repeats)
        ]
        for metric in range(8):
            collected[metric].extend(np.mean(np.stack([x[metric] for x in repeats]), axis=0).tolist())
        print(f"evaluated {min(start + args.batch_size, len(selected))}/{len(selected)}", flush=True)

    arrays = [np.asarray(values) for values in collected]
    atomic, empty, reverse = arrays[:3]
    report = {
        "samples": len(selected),
        "noise_repeats": args.noise_repeats,
        "comparison": "same full images/state/actions/noise/time; only prompt changes",
        "atomic_plus_full_observation_flow_mse": float(atomic.mean()),
        "empty_text_plus_full_observation_flow_mse": float(empty.mean()),
        "reverse_atomic_plus_full_observation_flow_mse": float(reverse.mean()),
        "atomic_better_than_empty_fraction": float(np.mean(atomic < empty)),
        "atomic_better_than_reverse_fraction": float(np.mean(atomic < reverse)),
        "empty_relative_to_atomic": float(empty.mean() / atomic.mean()),
        "reverse_relative_to_atomic": float(reverse.mean() / atomic.mean()),
        "atomic_empty_direction_cosine_right_left": np.mean(arrays[3], axis=0).tolist(),
        "atomic_reverse_direction_cosine_right_left": np.mean(arrays[4], axis=0).tolist(),
        "atomic_empty_zm_cosine_right_left": np.mean(arrays[5], axis=0).tolist(),
        "atomic_reverse_zm_cosine_right_left": np.mean(arrays[6], axis=0).tolist(),
        "mean_flow_time": float(arrays[7].mean()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
