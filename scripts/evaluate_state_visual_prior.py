#!/usr/bin/env python3
"""Measure state/image shortcut strength in the 70K atomic policy.

For the same action target, flow noise, and flow time, compare the factual
observation with distribution-preserving batch permutations of state and/or
images.  State permutation is applied to both Pi0.5 state channels: the
discretized state tokens in the prompt and the continuous Action Expert state.
"""

from __future__ import annotations

import argparse
import dataclasses
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


def _concat_observations(a, b):
    return jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), a, b)


@nnx.jit
def _paired_metrics(
    model,
    factual_observation,
    perturbed_observation,
    factual_latent_state,
    perturbed_latent_state,
    actions,
    factual_tokens,
    factual_mask,
    perturbed_tokens,
    perturbed_mask,
    rng,
):
    factual_observation = _model.preprocess_observation(
        None, factual_observation, train=False
    )
    perturbed_observation = _model.preprocess_observation(
        None, perturbed_observation, train=False
    )
    factual_observation = model._with_prompt(  # noqa: SLF001
        factual_observation, factual_tokens, factual_mask
    )
    perturbed_observation = model._with_prompt(  # noqa: SLF001
        perturbed_observation, perturbed_tokens, perturbed_mask
    )
    paired = _concat_observations(factual_observation, perturbed_observation)
    query_hidden, prefix_mask, kv_cache, _ = model._prefix_forward(paired)  # noqa: SLF001
    active_state = jnp.concatenate(
        [
            model._controlled_state(factual_latent_state),  # noqa: SLF001
            model._controlled_state(perturbed_latent_state),  # noqa: SLF001
        ],
        axis=0,
    )
    _, _, z_model, _, _ = model._latent(query_hidden, active_state)  # noqa: SLF001

    noise_rng, time_rng = jax.random.split(rng)
    actions = model._mask_action_condition(actions)  # noqa: SLF001
    noise = model._mask_action_condition(  # noqa: SLF001
        jax.random.normal(noise_rng, actions.shape)
    )
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    noisy = time[:, None, None] * noise + (1.0 - time[:, None, None]) * actions
    target = model._controlled_actions(noise - actions)  # noqa: SLF001
    paired_noisy = jnp.concatenate([noisy, noisy], axis=0)
    paired_time = jnp.concatenate([time, time], axis=0)
    velocity = model._suffix_velocity(  # noqa: SLF001
        prefix_mask, kv_cache, paired_noisy, paired_time, z_model
    )
    velocity = model._controlled_actions(velocity)  # noqa: SLF001
    batch_size = actions.shape[0]
    factual_velocity = velocity[:batch_size]
    perturbed_velocity = velocity[batch_size:]
    reduce_axes = tuple(range(1, target.ndim))
    factual_mse = jnp.mean(jnp.square(factual_velocity - target), axis=reduce_axes)
    perturbed_mse = jnp.mean(jnp.square(perturbed_velocity - target), axis=reduce_axes)
    dot = jnp.sum(factual_velocity * perturbed_velocity, axis=reduce_axes)
    factual_norm = jnp.sqrt(jnp.sum(jnp.square(factual_velocity), axis=reduce_axes))
    perturbed_norm = jnp.sqrt(jnp.sum(jnp.square(perturbed_velocity), axis=reduce_axes))
    cosine = dot / jnp.maximum(factual_norm * perturbed_norm, 1.0e-8)
    relative_change = jnp.sqrt(
        jnp.sum(jnp.square(perturbed_velocity - factual_velocity), axis=reduce_axes)
    ) / jnp.maximum(factual_norm, 1.0e-8)
    return factual_mse, perturbed_mse, cosine, relative_change, time


def _permute_observation(observation, permutation, *, state: bool, images: bool):
    return dataclasses.replace(
        observation,
        state=observation.state[permutation] if state else observation.state,
        images={
            key: value[permutation] if images else value
            for key, value in observation.images.items()
        },
        image_masks={
            key: value[permutation] if images else value
            for key, value in observation.image_masks.items()
        },
    )


def _summary(rows):
    result = {}
    for prompt_mode in ("empty", "atomic"):
        result[prompt_mode] = {}
        for perturbation in (
            "context_state_shuffle",
            "latent_state_shuffle",
            "both_state_shuffle",
            "image_shuffle",
        ):
            selected = [
                row
                for row in rows
                if row["prompt_mode"] == prompt_mode
                and row["perturbation"] == perturbation
            ]
            factual = np.asarray([row["factual_mse"] for row in selected])
            perturbed = np.asarray([row["perturbed_mse"] for row in selected])
            result[prompt_mode][perturbation] = {
                "n": len(selected),
                "factual_mse": float(factual.mean()),
                "perturbed_mse": float(perturbed.mean()),
                "mse_increase_fraction": float((perturbed / np.maximum(factual, 1e-8) - 1).mean()),
                "perturbed_worse_rate": float(np.mean(perturbed > factual)),
                "velocity_cosine": float(np.mean([row["velocity_cosine"] for row in selected])),
                "relative_velocity_change": float(
                    np.mean([row["relative_velocity_change"] for row in selected])
                ),
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--noise-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    anchors = [
        anchor
        for index, anchor in enumerate(manifest["anchors"])
        if index % args.num_shards == args.shard_index
    ]
    config = AtomicPi05Config(max_token_len=250, fast_action_ce_loss_weight=0.0)
    dataset = build_atomic_dataset(
        (args.dataset_root,),
        norm_assets_dir="/mnt/cunchu/zjq/target",
        norm_asset_id="openpi_norm_compact_accepted_v3",
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        include_fast=False,
    )
    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    tokenizer = _paligemma_tokenizer(config.max_token_len)

    rows_out = []
    for batch_start in range(0, len(anchors), args.batch_size):
        chunk = anchors[batch_start : batch_start + args.batch_size]
        real_count = len(chunk)
        if real_count < args.batch_size:
            chunk = chunk + [chunk[-1]] * (args.batch_size - real_count)
        samples = [dataset[row["dataset_index"]] for row in chunk]
        batch = atomic_collate(samples)
        observation_np, actions_np = batch_to_observation(batch)
        observation = jax.tree.map(jnp.asarray, observation_np)
        actions = jnp.asarray(actions_np)
        # A nontrivial cyclic shift preserves each marginal distribution while
        # breaking the sample-level state/image/action association.
        permutation = np.roll(np.arange(args.batch_size), args.batch_size // 2)
        variants = {
            # Pi0.5 state tokens are supplied separately below.  Keeping the
            # Observation state intact here lets us intervene independently on
            # the discrete context route and the continuous AtomicQueries route.
            "context_state_shuffle": observation,
            "latent_state_shuffle": observation,
            "both_state_shuffle": observation,
            "image_shuffle": _permute_observation(
                observation, permutation, state=False, images=True
            ),
        }
        states = np.asarray(batch["state"])
        shuffled_states = states[permutation]
        for prompt_mode in ("empty", "atomic"):
            texts = ["" if prompt_mode == "empty" else row["atomic_prompt"] for row in chunk]
            factual_tokenized = [
                tokenizer.tokenize(text, state)
                for text, state in zip(texts, states, strict=True)
            ]
            factual_tokens = jnp.asarray(np.stack([value[0] for value in factual_tokenized]))
            factual_masks = jnp.asarray(np.stack([value[1] for value in factual_tokenized]))
            for perturbation, perturbed_observation in variants.items():
                shuffle_context = perturbation in {
                    "context_state_shuffle", "both_state_shuffle"
                }
                shuffle_latent = perturbation in {
                    "latent_state_shuffle", "both_state_shuffle"
                }
                prompt_states = shuffled_states if shuffle_context else states
                perturbed_latent_state = shuffled_states if shuffle_latent else states
                perturbed_tokenized = [
                    tokenizer.tokenize(text, state)
                    for text, state in zip(texts, prompt_states, strict=True)
                ]
                perturbed_tokens = jnp.asarray(np.stack([value[0] for value in perturbed_tokenized]))
                perturbed_masks = jnp.asarray(np.stack([value[1] for value in perturbed_tokenized]))
                repeats = []
                for repeat in range(args.noise_repeats):
                    repeats.append(
                        jax.device_get(
                            _paired_metrics(
                                model,
                                observation,
                                perturbed_observation,
                                jnp.asarray(states),
                                jnp.asarray(perturbed_latent_state),
                                actions,
                                factual_tokens,
                                factual_masks,
                                perturbed_tokens,
                                perturbed_masks,
                                jax.random.key(args.seed + 100000 * args.shard_index + 1000 * batch_start + repeat),
                            )
                        )
                    )
                values = [np.stack([repeat[index] for repeat in repeats]) for index in range(5)]
                for local_index, anchor in enumerate(chunk[:real_count]):
                    rows_out.append(
                        {
                            "anchor_key": anchor["anchor_key"],
                            "arm": anchor["arm"],
                            "kind": anchor["kind"],
                            "prompt_mode": prompt_mode,
                            "perturbation": perturbation,
                            "factual_mse": float(values[0][:, local_index].mean()),
                            "perturbed_mse": float(values[1][:, local_index].mean()),
                            "velocity_cosine": float(values[2][:, local_index].mean()),
                            "relative_velocity_change": float(values[3][:, local_index].mean()),
                            "flow_time": float(values[4][:, local_index].mean()),
                        }
                    )
        print(
            f"state-prior shard {args.shard_index}: "
            f"{min(batch_start + args.batch_size, len(anchors))}/{len(anchors)}",
            flush=True,
        )

    payload = {
        "checkpoint": str(args.checkpoint),
        "anchor_count": len(anchors),
        "noise_repeats": args.noise_repeats,
        "contract": (
            "same target/noise/time; context-state shuffle changes only Pi0.5 discretized "
            "prompt-state tokens; latent-state shuffle changes only the continuous state "
            "added inside AtomicQueries; image shuffle preserves image marginals"
        ),
        "summary": _summary(rows_out),
        "rows": rows_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(rows_out)}, indent=2))


if __name__ == "__main__":
    main()
