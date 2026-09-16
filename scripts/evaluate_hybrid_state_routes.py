#!/usr/bin/env python3
"""Exact two-VLM-pass state-route intervention for AtomicPi05.

One prefix pass supplies only the factual/context KV cache.  A second prefix
pass supplies only Q1--Q4 and therefore zM.  Real-state and batch-shuffled-state
passes are then cross-composed at the Action Expert:

  real KV + real zM       baseline
  shuffled KV + real zM   context-state route only
  real KV + shuffled zM   atomic-state route only
  shuffled KV + shuffled zM

The shuffled atomic pass changes both ways state reaches the atomic latent:
discrete Pi0.5 state tokens read by Q1--Q4 and the continuous state MLP added
inside AtomicQueries.
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


@nnx.jit
def _hybrid_route_metrics(
    model,
    observation,
    actions,
    real_tokens,
    real_mask,
    shuffled_tokens,
    shuffled_mask,
    real_state,
    shuffled_state,
    rng,
):
    observation = _model.preprocess_observation(None, observation, train=False)
    real_observation = model._with_prompt(observation, real_tokens, real_mask)  # noqa: SLF001
    shuffled_observation = model._with_prompt(  # noqa: SLF001
        observation, shuffled_tokens, shuffled_mask
    )
    real_query, real_prefix_mask, real_kv, _ = model._prefix_forward(real_observation)  # noqa: SLF001
    shuffled_query, shuffled_prefix_mask, shuffled_kv, _ = model._prefix_forward(  # noqa: SLF001
        shuffled_observation
    )
    real_active_state = model._controlled_state(real_state)  # noqa: SLF001
    shuffled_active_state = model._controlled_state(shuffled_state)  # noqa: SLF001
    _, _, real_zm, _, _ = model._latent(real_query, real_active_state)  # noqa: SLF001
    _, _, shuffled_zm, _, _ = model._latent(  # noqa: SLF001
        shuffled_query, shuffled_active_state
    )

    noise_rng, time_rng = jax.random.split(rng)
    actions = model._mask_action_condition(actions)  # noqa: SLF001
    noise = model._mask_action_condition(  # noqa: SLF001
        jax.random.normal(noise_rng, actions.shape)
    )
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    noisy = time[:, None, None] * noise + (1.0 - time[:, None, None]) * actions
    target = model._controlled_actions(noise - actions)  # noqa: SLF001

    def predict(prefix_mask, kv_cache, z_model):
        return model._controlled_actions(  # noqa: SLF001
            model._suffix_velocity(prefix_mask, kv_cache, noisy, time, z_model)  # noqa: SLF001
        )

    baseline = predict(real_prefix_mask, real_kv, real_zm)
    context_state = predict(shuffled_prefix_mask, shuffled_kv, real_zm)
    atomic_state = predict(real_prefix_mask, real_kv, shuffled_zm)
    both_state = predict(shuffled_prefix_mask, shuffled_kv, shuffled_zm)
    variants = jnp.stack([baseline, context_state, atomic_state, both_state], axis=0)
    reduce_axes = tuple(range(2, variants.ndim))
    mse = jnp.mean(jnp.square(variants - target[None]), axis=reduce_axes)
    base = variants[0]
    dot = jnp.sum(variants * base[None], axis=reduce_axes)
    base_norm = jnp.sqrt(jnp.sum(jnp.square(base), axis=tuple(range(1, base.ndim))))
    variant_norm = jnp.sqrt(jnp.sum(jnp.square(variants), axis=reduce_axes))
    cosine = dot / jnp.maximum(variant_norm * base_norm[None], 1.0e-8)
    relative_change = jnp.sqrt(
        jnp.sum(jnp.square(variants - base[None]), axis=reduce_axes)
    ) / jnp.maximum(base_norm[None], 1.0e-8)
    return mse, cosine, relative_change, time


def _aggregate(rows):
    result = {}
    for prompt_mode in ("empty", "atomic"):
        selected = [row for row in rows if row["prompt_mode"] == prompt_mode]
        baseline = np.asarray([row["baseline_mse"] for row in selected])
        result[prompt_mode] = {"n": len(selected), "baseline_mse": float(baseline.mean())}
        for route in ("context_state", "atomic_state", "both_state"):
            perturbed = np.asarray([row[f"{route}_mse"] for row in selected])
            result[prompt_mode][route] = {
                "mse": float(perturbed.mean()),
                "aggregate_ratio": float(perturbed.mean() / baseline.mean()),
                "median_ratio": float(np.median(perturbed / np.maximum(baseline, 1e-8))),
                "worse_rate": float(np.mean(perturbed > baseline)),
                "velocity_cosine": float(np.mean([row[f"{route}_cosine"] for row in selected])),
                "relative_velocity_change": float(
                    np.mean([row[f"{route}_relative_change"] for row in selected])
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
        row for index, row in enumerate(manifest["anchors"])
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
        real_states = np.asarray(batch["state"])
        permutation = np.roll(np.arange(args.batch_size), args.batch_size // 2)
        shuffled_states = real_states[permutation]

        for prompt_mode in ("empty", "atomic"):
            texts = ["" if prompt_mode == "empty" else row["atomic_prompt"] for row in chunk]
            real_tokenized = [
                tokenizer.tokenize(text, state)
                for text, state in zip(texts, real_states, strict=True)
            ]
            shuffled_tokenized = [
                tokenizer.tokenize(text, state)
                for text, state in zip(texts, shuffled_states, strict=True)
            ]
            real_tokens = jnp.asarray(np.stack([value[0] for value in real_tokenized]))
            real_masks = jnp.asarray(np.stack([value[1] for value in real_tokenized]))
            shuffled_tokens = jnp.asarray(np.stack([value[0] for value in shuffled_tokenized]))
            shuffled_masks = jnp.asarray(np.stack([value[1] for value in shuffled_tokenized]))
            repeats = []
            for repeat in range(args.noise_repeats):
                repeats.append(
                    jax.device_get(
                        _hybrid_route_metrics(
                            model,
                            observation,
                            actions,
                            real_tokens,
                            real_masks,
                            shuffled_tokens,
                            shuffled_masks,
                            jnp.asarray(real_states),
                            jnp.asarray(shuffled_states),
                            jax.random.key(
                                args.seed + 100000 * args.shard_index
                                + 1000 * batch_start + repeat
                            ),
                        )
                    )
                )
            mse = np.stack([value[0] for value in repeats]).mean(axis=0)
            cosine = np.stack([value[1] for value in repeats]).mean(axis=0)
            relative = np.stack([value[2] for value in repeats]).mean(axis=0)
            times = np.stack([value[3] for value in repeats]).mean(axis=0)
            for local_index, anchor in enumerate(chunk[:real_count]):
                rows_out.append(
                    {
                        "anchor_key": anchor["anchor_key"],
                        "arm": anchor["arm"],
                        "kind": anchor["kind"],
                        "prompt_mode": prompt_mode,
                        "baseline_mse": float(mse[0, local_index]),
                        "context_state_mse": float(mse[1, local_index]),
                        "atomic_state_mse": float(mse[2, local_index]),
                        "both_state_mse": float(mse[3, local_index]),
                        "context_state_cosine": float(cosine[1, local_index]),
                        "atomic_state_cosine": float(cosine[2, local_index]),
                        "both_state_cosine": float(cosine[3, local_index]),
                        "context_state_relative_change": float(relative[1, local_index]),
                        "atomic_state_relative_change": float(relative[2, local_index]),
                        "both_state_relative_change": float(relative[3, local_index]),
                        "flow_time": float(times[local_index]),
                    }
                )
        print(
            f"hybrid-state shard {args.shard_index}: "
            f"{min(batch_start + args.batch_size, len(anchors))}/{len(anchors)}",
            flush=True,
        )

    payload = {
        "checkpoint": str(args.checkpoint),
        "anchor_count": len(anchors),
        "noise_repeats": args.noise_repeats,
        "contract": (
            "two independent prefix passes; cross-compose KV context and Q1-Q4/zM; "
            "atomic-state intervention changes both Q-visible state tokens and continuous state MLP"
        ),
        "summary": _aggregate(rows_out),
        "rows": rows_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(rows_out)}, indent=2))


if __name__ == "__main__":
    main()
