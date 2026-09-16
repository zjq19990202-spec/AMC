#!/usr/bin/env python3
"""Paired Flow-MSE evaluation on the reviewed 500 bimanual state anchors.

For every anchor, image, state, ground-truth action, flow time and Gaussian
noise are shared.  Only the prompt changes among empty, subtask and the
reviewed native atomic instruction.  This measures conditioning quality; it
is deliberately separate from free-rollout TCP steering.
"""

from __future__ import annotations

import argparse
import csv
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
def _three_prompt_flow_mse(
    model,
    observation,
    actions,
    prompt_tokens,
    prompt_masks,
    rng,
):
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(
        preprocess_rng, observation, train=False
    )
    actions = model._mask_action_condition(actions)  # noqa: SLF001
    noise = model._mask_action_condition(  # noqa: SLF001
        jax.random.normal(noise_rng, actions.shape)
    )
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    noisy = time[:, None, None] * noise + (1.0 - time[:, None, None]) * actions
    target = model._controlled_actions(noise - actions)  # noqa: SLF001
    active_state = model._controlled_state(observation.state)  # noqa: SLF001

    def predict(tokens, masks):
        prompted = model._with_prompt(observation, tokens, masks)  # noqa: SLF001
        query_hidden, prefix_mask, kv_cache = model._prefix_forward(prompted)  # noqa: SLF001
        _, direction, z_model, _, _ = model._latent(  # noqa: SLF001
            query_hidden, active_state
        )
        velocity = model._suffix_velocity(  # noqa: SLF001
            prefix_mask, kv_cache, noisy, time, z_model
        )
        velocity = model._controlled_actions(velocity)  # noqa: SLF001
        mse = jnp.mean(jnp.square(velocity - target), axis=tuple(range(1, target.ndim)))
        return mse, direction, z_model

    outputs = [
        predict(prompt_tokens[:, variant], prompt_masks[:, variant])
        for variant in range(prompt_tokens.shape[1])
    ]
    return (
        jnp.stack([output[0] for output in outputs], axis=1),
        jnp.stack([output[1] for output in outputs], axis=1),
        jnp.stack([output[2] for output in outputs], axis=1),
        time,
    )


def _summary(rows: list[dict]) -> dict:
    empty = np.asarray([row["empty_flow_mse"] for row in rows])
    subtask = np.asarray([row["subtask_flow_mse"] for row in rows])
    atomic = np.asarray([row["atomic_flow_mse"] for row in rows])
    return {
        "count": len(rows),
        "empty_flow_mse_mean": float(empty.mean()),
        "subtask_flow_mse_mean": float(subtask.mean()),
        "atomic_flow_mse_mean": float(atomic.mean()),
        "atomic_minus_empty_mean": float((atomic - empty).mean()),
        "atomic_minus_subtask_mean": float((atomic - subtask).mean()),
        "atomic_better_than_empty_fraction": float(np.mean(atomic < empty)),
        "atomic_better_than_subtask_fraction": float(np.mean(atomic < subtask)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--noise-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--max-anchors", type=int, default=0)
    args = parser.parse_args()

    manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    anchors = manifest["anchors"]
    if args.max_anchors:
        anchors = anchors[: args.max_anchors]
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

    output_rows = []
    for start in range(0, len(anchors), args.batch_size):
        anchor_batch = anchors[start : start + args.batch_size]
        real_count = len(anchor_batch)
        if real_count < args.batch_size:
            anchor_batch = anchor_batch + [anchor_batch[-1]] * (
                args.batch_size - real_count
            )
        samples = [dataset[int(anchor["dataset_index"])] for anchor in anchor_batch]
        batch = atomic_collate(samples)
        observation_np, actions_np = batch_to_observation(batch)
        observation = jax.tree.map(jnp.asarray, observation_np)
        actions = jnp.asarray(actions_np)
        prompts = [
            ("", str(anchor["subtask_prompt"]), str(anchor["atomic_prompt"]))
            for anchor in anchor_batch
        ]
        tokenized = [
            [tokenizer.tokenize(prompt, np.asarray(sample["state"])) for prompt in row]
            for row, sample in zip(prompts, samples, strict=True)
        ]
        tokens = jnp.asarray(
            np.stack([[item[0] for item in row] for row in tokenized])
        )
        masks = jnp.asarray(
            np.stack([[item[1] for item in row] for row in tokenized])
        )
        repeat_outputs = []
        for repeat in range(args.noise_repeats):
            repeat_outputs.append(
                jax.device_get(
                    _three_prompt_flow_mse(
                        model,
                        observation,
                        actions,
                        tokens,
                        masks,
                        jax.random.key(args.seed + 1009 * start + repeat),
                    )
                )
            )
        losses = np.mean(np.stack([row[0] for row in repeat_outputs]), axis=0)
        directions = np.mean(np.stack([row[1] for row in repeat_outputs]), axis=0)
        z_models = np.mean(np.stack([row[2] for row in repeat_outputs]), axis=0)
        times = np.mean(np.stack([row[3] for row in repeat_outputs]), axis=0)
        for local, anchor in enumerate(anchor_batch[:real_count]):
            direction_cosine = float(
                np.sum(directions[local, 1] * directions[local, 2])
            )
            zm_cosine = float(
                np.sum(z_models[local, 1] * z_models[local, 2])
                / max(
                    np.linalg.norm(z_models[local, 1])
                    * np.linalg.norm(z_models[local, 2]),
                    1e-8,
                )
            )
            output_rows.append(
                {
                    "anchor_key": anchor["anchor_key"],
                    "cluster_key": anchor["cluster_key"],
                    "episode": anchor["episode"],
                    "frame": anchor["frame"],
                    "arm": anchor["arm"],
                    "kind": anchor["kind"],
                    "other_status": anchor["other_status"],
                    "atoms": "+".join(anchor["atoms"]),
                    "empty_flow_mse": float(losses[local, 0]),
                    "subtask_flow_mse": float(losses[local, 1]),
                    "atomic_flow_mse": float(losses[local, 2]),
                    "subtask_atomic_direction_cosine": direction_cosine,
                    "subtask_atomic_zm_cosine": zm_cosine,
                    "flow_time_mean": float(times[local]),
                }
            )
        print(f"evaluated {min(start + args.batch_size, len(anchors))}/{len(anchors)}", flush=True)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in output_rows:
        grouped[(row["arm"], row["kind"])].append(row)
        grouped[(row["arm"], "all")].append(row)
        grouped[("both", row["kind"])].append(row)
        grouped[("both", "all")].append(row)
    summaries = {
        f"{arm}/{kind}": _summary(rows)
        for (arm, kind), rows in sorted(grouped.items())
    }
    report = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "selection_manifest": str(args.selection_manifest),
        "paired_control": (
            "same image/state/GT action/flow time/noise; only empty, subtask or native atomic prompt differs"
        ),
        "noise_repeats": args.noise_repeats,
        "summary": summaries,
        "rows": output_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
