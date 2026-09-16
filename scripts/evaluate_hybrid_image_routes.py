#!/usr/bin/env python3
"""Exact two-VLM-pass image-route intervention for AtomicPi05.

The reviewed manifest contains two nearby-state anchors per cluster.  Each
anchor keeps its own state, prompt, action target, and flow noise while all
three camera images are replaced by the other anchor from the same cluster.
Real and donor-image prefix passes are then cross-composed at the Action
Expert:

  real-image KV + real-image zM       baseline
  donor-image KV + real-image zM      context image route only
  real-image KV + donor-image zM      atomic image route only
  donor-image KV + donor-image zM     both image routes
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
def _hybrid_image_metrics(
    model,
    real_observation,
    donor_observation,
    actions,
    tokens,
    token_mask,
    rng,
):
    real_observation = _model.preprocess_observation(None, real_observation, train=False)
    donor_observation = _model.preprocess_observation(None, donor_observation, train=False)
    real_observation = model._with_prompt(real_observation, tokens, token_mask)  # noqa: SLF001
    donor_observation = model._with_prompt(donor_observation, tokens, token_mask)  # noqa: SLF001

    real_query, real_prefix_mask, real_kv, _ = model._prefix_forward(real_observation)  # noqa: SLF001
    donor_query, donor_prefix_mask, donor_kv, _ = model._prefix_forward(  # noqa: SLF001
        donor_observation
    )
    # Both passes deliberately use the factual state.  Only image pixels and
    # image masks differ between real_observation and donor_observation.
    active_state = model._controlled_state(real_observation.state)  # noqa: SLF001
    _, _, real_zm, _, _ = model._latent(real_query, active_state)  # noqa: SLF001
    _, _, donor_zm, _, _ = model._latent(donor_query, active_state)  # noqa: SLF001

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
    context_image = predict(donor_prefix_mask, donor_kv, real_zm)
    atomic_image = predict(real_prefix_mask, real_kv, donor_zm)
    both_image = predict(donor_prefix_mask, donor_kv, donor_zm)
    variants = jnp.stack([baseline, context_image, atomic_image, both_image], axis=0)
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


def _aggregate(rows: list[dict]) -> dict:
    result = {}
    for prompt_mode in ("empty", "atomic"):
        selected = [row for row in rows if row["prompt_mode"] == prompt_mode]
        baseline = np.asarray([row["baseline_mse"] for row in selected])
        result[prompt_mode] = {"n": len(selected), "baseline_mse": float(baseline.mean())}
        for route in ("context_image", "atomic_image", "both_image"):
            perturbed = np.asarray([row[f"{route}_mse"] for row in selected])
            result[prompt_mode][route] = {
                "mse": float(perturbed.mean()),
                "aggregate_ratio": float(perturbed.mean() / baseline.mean()),
                "median_ratio": float(np.median(perturbed / np.maximum(baseline, 1.0e-8))),
                "worse_rate": float(np.mean(perturbed > baseline)),
                "velocity_cosine": float(
                    np.mean([row[f"{route}_cosine"] for row in selected])
                ),
                "relative_velocity_change": float(
                    np.mean([row[f"{route}_relative_change"] for row in selected])
                ),
            }
    return result


def _donor_map(anchors: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for anchor in anchors:
        grouped[anchor["cluster_key"]].append(anchor)
    result = {}
    for cluster, rows in grouped.items():
        if len(rows) < 2:
            raise ValueError(f"cluster {cluster} has no paired donor image")
        ordered = sorted(rows, key=lambda row: (int(row["pair_index"]), row["anchor_key"]))
        for index, row in enumerate(ordered):
            result[row["anchor_key"]] = ordered[(index + 1) % len(ordered)]
    return result


def main() -> None:
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
    all_anchors = manifest["anchors"]
    donors = _donor_map(all_anchors)
    anchors = [
        row for index, row in enumerate(all_anchors)
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
        real_samples = [dataset[row["dataset_index"]] for row in chunk]
        donor_rows = [donors[row["anchor_key"]] for row in chunk]
        donor_samples = [dataset[row["dataset_index"]] for row in donor_rows]
        real_batch = atomic_collate(real_samples)
        donor_batch = atomic_collate(donor_samples)
        real_np, actions_np = batch_to_observation(real_batch)
        donor_np, _ = batch_to_observation(donor_batch)
        real_observation = jax.tree.map(jnp.asarray, real_np)
        # Construct an observation that differs only in its image fields.
        donor_observation = real_observation.replace(
            images=jax.tree.map(jnp.asarray, donor_np.images),
            image_masks=jax.tree.map(jnp.asarray, donor_np.image_masks),
        )
        actions = jnp.asarray(actions_np)
        states = np.asarray(real_batch["state"])

        for prompt_mode in ("empty", "atomic"):
            texts = ["" if prompt_mode == "empty" else row["atomic_prompt"] for row in chunk]
            tokenized = [
                tokenizer.tokenize(text, state)
                for text, state in zip(texts, states, strict=True)
            ]
            tokens = jnp.asarray(np.stack([value[0] for value in tokenized]))
            masks = jnp.asarray(np.stack([value[1] for value in tokenized]))
            repeats = []
            for repeat in range(args.noise_repeats):
                repeats.append(
                    jax.device_get(
                        _hybrid_image_metrics(
                            model,
                            real_observation,
                            donor_observation,
                            actions,
                            tokens,
                            masks,
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
                donor = donor_rows[local_index]
                rows_out.append(
                    {
                        "anchor_key": anchor["anchor_key"],
                        "donor_anchor_key": donor["anchor_key"],
                        "cluster_key": anchor["cluster_key"],
                        "arm": anchor["arm"],
                        "kind": anchor["kind"],
                        "prompt_mode": prompt_mode,
                        "baseline_mse": float(mse[0, local_index]),
                        "context_image_mse": float(mse[1, local_index]),
                        "atomic_image_mse": float(mse[2, local_index]),
                        "both_image_mse": float(mse[3, local_index]),
                        "context_image_cosine": float(cosine[1, local_index]),
                        "atomic_image_cosine": float(cosine[2, local_index]),
                        "both_image_cosine": float(cosine[3, local_index]),
                        "context_image_relative_change": float(relative[1, local_index]),
                        "atomic_image_relative_change": float(relative[2, local_index]),
                        "both_image_relative_change": float(relative[3, local_index]),
                        "flow_time": float(times[local_index]),
                    }
                )
        print(
            f"hybrid-image shard {args.shard_index}: "
            f"{min(batch_start + args.batch_size, len(anchors))}/{len(anchors)}",
            flush=True,
        )

    payload = {
        "checkpoint": str(args.checkpoint),
        "anchor_count": len(anchors),
        "noise_repeats": args.noise_repeats,
        "contract": (
            "paired nearby-state donor image; factual state/prompt/action/noise fixed; "
            "cross-compose image-conditioned KV context and Q1-Q3/zM"
        ),
        "summary": _aggregate(rows_out),
        "rows": rows_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(rows_out)}, indent=2))


if __name__ == "__main__":
    main()
