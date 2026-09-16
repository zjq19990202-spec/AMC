#!/usr/bin/env python3
"""Evaluate counterfactual orthogonal projection guidance on the 70K policy.

The factual atomic branch and language-masked branch share images, state,
initial flow noise, and every intermediate ODE action.  At each flow step we
compute

    v_perp = v_atom - proj_{v_empty}(v_atom)
    v_opg  = v_atom + gamma * v_perp

without changing model parameters.  The script deliberately consumes the
reviewed 500-anchor manifest so prompt formatting and TCP=0.20 m evaluation
match the existing atomic steering audit.
"""

from __future__ import annotations

import argparse
import gc
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
import evaluate_zm_fk_trajectory_ablation as _fk_eval
from evaluate_zm_fk_trajectory_ablation import _output_transform
import evaluate_many_cluster_single_dual_steering as base


@nnx.jit
def _sample_opg(
    model,
    observation,
    noise,
    factual_tokens,
    factual_mask,
    empty_tokens,
    empty_mask,
    gamma,
):
    """Run one OPG trajectory; factual and empty velocities share each x_t."""

    observation = _model.preprocess_observation(None, observation, train=False)
    factual = model._with_prompt(observation, factual_tokens, factual_mask)  # noqa: SLF001
    empty = model._with_prompt(observation, empty_tokens, empty_mask)  # noqa: SLF001
    paired = jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=0), factual, empty)
    query_hidden, prefix_mask, kv_cache, _ = model._prefix_forward(paired)  # noqa: SLF001
    active_state = model._controlled_state(paired.state)  # noqa: SLF001
    _, _, z_model, _, _ = model._latent(query_hidden, active_state)  # noqa: SLF001
    noise = model._mask_action_condition(noise)  # noqa: SLF001
    batch_size = noise.shape[0]

    def step(index, actions):
        time = jnp.asarray(1.0 - index / 10.0, dtype=actions.dtype)
        paired_actions = jnp.concatenate([actions, actions], axis=0)
        velocity = model._suffix_velocity(  # noqa: SLF001
            prefix_mask,
            kv_cache,
            paired_actions,
            jnp.broadcast_to(time, (2 * batch_size,)),
            z_model,
        )
        velocity = model._mask_action_condition(velocity)  # noqa: SLF001
        factual_velocity = velocity[:batch_size]
        empty_velocity = velocity[batch_size:]
        reduce_axes = tuple(range(1, factual_velocity.ndim))
        numerator = jnp.sum(factual_velocity * empty_velocity, axis=reduce_axes)
        denominator = jnp.sum(jnp.square(empty_velocity), axis=reduce_axes) + 1.0e-8
        projection = (numerator / denominator)[:, None, None] * empty_velocity
        perpendicular = factual_velocity - projection
        guided_velocity = factual_velocity + gamma[:, None, None] * perpendicular
        return model._mask_action_condition(actions - 0.1 * guided_velocity)  # noqa: SLF001

    return jax.lax.fori_loop(0, 10, step, noise)[..., : model.config.active_action_dim]


def _trajectory(fk, raw_state: np.ndarray, actions: np.ndarray, arm: str) -> np.ndarray:
    return base._trajectory(fk, raw_state, actions, arm)  # noqa: SLF001


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--gammas", type=float, nargs="+", default=(0.0, 0.5, 1.0, 2.0))
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")

    manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    all_anchors = manifest["anchors"]
    anchors = [
        anchor
        for index, anchor in enumerate(all_anchors)
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
    decode = _output_transform(args.dataset_root, config)
    _fk_eval.TCP_LOCAL_Z_OFFSET_M = 0.20
    fk = _fk_eval.SimpleCR1FK(Path("assets/cr1_right_arm_tcp_red/urdf/CR1ARMR_TCP_RED.urdf"))
    tokenizer = _paligemma_tokenizer(config.max_token_len)

    rng = np.random.default_rng(args.seed)
    noise_by_anchor = {
        anchor["anchor_key"]: rng.standard_normal(
            (config.action_horizon, config.action_dim), dtype=np.float32
        )
        for anchor in all_anchors
    }
    rows: list[dict] = []
    for anchor in anchors:
        for variant, prompt, atoms in (
            ("native", anchor["atomic_prompt"], anchor["atoms"]),
            ("reverse", anchor["reverse_prompt"], [base._opposite(atom) for atom in anchor["atoms"]]),  # noqa: SLF001
        ):
            for gamma in args.gammas:
                rows.append(
                    {
                        **{key: value for key, value in anchor.items() if key != "state_7d"},
                        "variant": variant,
                        "prompt": prompt,
                        "target_atoms": atoms,
                        "gamma": float(gamma),
                    }
                )

    sample_cache = {index: dataset[index] for index in {row["dataset_index"] for row in rows}}
    metadata_cache = {
        index: dataset._raw.metadata(index)  # noqa: SLF001
        for index in sample_cache
    }
    outputs: list[dict] = []
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start : start + args.batch_size]
        real_count = len(chunk)
        if real_count < args.batch_size:
            chunk = chunk + [chunk[-1]] * (args.batch_size - real_count)
        samples = [sample_cache[row["dataset_index"]] for row in chunk]
        batch = atomic_collate(samples)
        observation_np, _ = batch_to_observation(batch)
        observation = jax.tree.map(jnp.asarray, observation_np)
        tokenized = [
            tokenizer.tokenize(row["prompt"], np.asarray(sample["state"]))
            for row, sample in zip(chunk, samples, strict=True)
        ]
        empty_tokenized = [
            tokenizer.tokenize("", np.asarray(sample["state"]))
            for sample in samples
        ]
        tokens = jnp.asarray(np.stack([item[0] for item in tokenized]))
        masks = jnp.asarray(np.stack([item[1] for item in tokenized]))
        empty_tokens = jnp.asarray(np.stack([item[0] for item in empty_tokenized]))
        empty_masks = jnp.asarray(np.stack([item[1] for item in empty_tokenized]))
        noises = np.stack([noise_by_anchor[row["anchor_key"]] for row in chunk])
        gammas = jnp.asarray([row["gamma"] for row in chunk], dtype=jnp.float32)
        predictions = np.asarray(
            jax.device_get(
                _sample_opg(
                    model,
                    observation,
                    jnp.asarray(noises),
                    tokens,
                    masks,
                    empty_tokens,
                    empty_masks,
                    gammas,
                )
            )
        )
        for local_index, (row, prediction) in enumerate(
            zip(chunk[:real_count], predictions[:real_count], strict=True)
        ):
            metadata = metadata_cache[row["dataset_index"]]
            decoded = decode(
                np.asarray(batch["state"][local_index]),
                np.asarray(metadata["raw_state"]),
                prediction,
            )["actions"]
            trajectory = _trajectory(
                fk, np.asarray(metadata["raw_state"]), decoded, row["arm"]
            )
            outputs.append(
                {
                    **{key: value for key, value in row.items() if key != "prompt"},
                    "prompt": row["prompt"],
                    "trajectory_twist": {
                        "25": trajectory[25].tolist(),
                        "50": trajectory[50].tolist(),
                    },
                }
            )
        print(f"OPG shard {args.shard_index}: {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)

    payload = {
        "checkpoint": str(args.checkpoint),
        "selection_manifest": str(args.selection_manifest),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "anchor_count": len(anchors),
        "gammas": list(args.gammas),
        "tcp_offset_m": 0.20,
        "projection_scope": "whole controlled 50x16 velocity field per sample",
        "outputs": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"rows": len(outputs), "output": str(args.output)}, indent=2))
    del model, params
    gc.collect()
    jax.clear_caches()


if __name__ == "__main__":
    main()
