#!/usr/bin/env python3
"""Split atomic-prompt steering into Context and Q->z_M paths.

For exact sign-reversal modes that were observed in the same bimanual-14D
state cluster, hold image, state, initial TCP and flow noise fixed.  Compare:

* empty Context + empty z_M;
* atomic Context + empty z_M;
* empty Context + atomic z_M;
* atomic Context + atomic z_M.

The script also measures whether the reverse prompt moves each arm's Q1 unit
direction toward the supervised atom code.  Codes are diagnostic only: the
production inference path never looks them up or inserts them into z_M.
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

from atomic_latent_vla.atomic import ATOMIC_NAMES
from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import (
    _paligemma_tokenizer,
    atomic_collate,
    batch_to_observation,
    build_atomic_dataset,
)
import evaluate_many_cluster_single_dual_steering as base
import evaluate_zm_fk_trajectory_ablation as fk_eval
from evaluate_zm_fk_trajectory_ablation import _output_transform


ROUTES = ("empty", "context_only", "zm_only", "full_prompt")
ARM_INDEX = {"right": 0, "left": 1}


def _reverse_atom(atom: str) -> str:
    if atom.endswith("_pos"):
        return atom[:-3] + "neg"
    if atom.endswith("_neg"):
        return atom[:-3] + "pos"
    raise ValueError(f"not a signed motion atom: {atom}")


@nnx.jit
def _sample_routes(
    model,
    observation,
    noise,
    empty_tokens,
    empty_mask,
    prompt_tokens,
    prompt_mask,
    state,
):
    observation = _model.preprocess_observation(None, observation, train=False)

    def prefix(tokens, mask):
        conditioned = model._with_prompt(observation, tokens, mask)  # noqa: SLF001
        return model._prefix_forward(conditioned)  # noqa: SLF001

    q_empty, mask_empty, kv_empty, _ = prefix(empty_tokens, empty_mask)
    q_prompt, mask_prompt, kv_prompt, _ = prefix(prompt_tokens, prompt_mask)
    active_state = model._controlled_state(state)  # noqa: SLF001
    _, right_empty, z_empty, left_empty, _ = model._latent(q_empty, active_state)  # noqa: SLF001
    _, right_prompt, z_prompt, left_prompt, _ = model._latent(q_prompt, active_state)  # noqa: SLF001

    noise = model._mask_action_condition(noise)  # noqa: SLF001
    batch_size = noise.shape[0]

    def sample(prefix_mask, kv_cache, z_model):
        def step(index, current):
            time = jnp.asarray(1.0 - index / 10.0, dtype=current.dtype)
            velocity = model._suffix_velocity(  # noqa: SLF001
                prefix_mask,
                kv_cache,
                current,
                jnp.broadcast_to(time, (batch_size,)),
                z_model,
            )
            velocity = model._mask_action_condition(velocity)  # noqa: SLF001
            return model._mask_action_condition(current - 0.1 * velocity)  # noqa: SLF001

        return jax.lax.fori_loop(0, 10, step, noise)[..., : model.config.active_action_dim]

    actions = jnp.stack(
        [
            sample(mask_empty, kv_empty, z_empty),
            sample(mask_prompt, kv_prompt, z_empty),
            sample(mask_empty, kv_empty, z_prompt),
            sample(mask_prompt, kv_prompt, z_prompt),
        ],
        axis=0,
    )
    directions = jnp.stack(
        [
            jnp.stack([right_empty, left_empty], axis=1),
            jnp.stack([right_prompt, left_prompt], axis=1),
        ],
        axis=0,
    )
    return actions, directions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    eligible = []
    for anchor in selection["anchors"]:
        reverse_atoms = tuple(_reverse_atom(atom) for atom in anchor["atoms"])
        seen_modes = {tuple(mode) for mode in anchor["cluster_isolated_modes"]}
        if reverse_atoms not in seen_modes:
            continue
        eligible.append({**anchor, "reverse_atoms": list(reverse_atoms)})
    anchors = [
        anchor for index, anchor in enumerate(eligible)
        if index % args.num_shards == args.shard_index
    ]
    if not anchors:
        raise RuntimeError("no exact cluster-seen reverse anchors in this shard")

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
    codes = np.asarray(jax.device_get(model.codebook.value), dtype=np.float64)
    codes /= np.maximum(np.linalg.norm(codes, axis=-1, keepdims=True), 1.0e-8)
    tokenizer = _paligemma_tokenizer(config.max_token_len)
    decode = _output_transform(args.dataset_root, config)
    fk_eval.TCP_LOCAL_Z_OFFSET_M = 0.20
    fk = fk_eval.SimpleCR1FK(Path("assets/cr1_right_arm_tcp_red/urdf/CR1ARMR_TCP_RED.urdf"))

    rng = np.random.default_rng(args.seed)
    noise_by_key = {
        anchor["anchor_key"]: rng.standard_normal(
            (config.action_horizon, config.action_dim), dtype=np.float32
        )
        for anchor in eligible
    }
    sample_cache = {anchor["dataset_index"]: dataset[anchor["dataset_index"]] for anchor in anchors}
    metadata_cache = {
        index: dataset._raw.metadata(index)  # noqa: SLF001
        for index in sample_cache
    }
    rows = []
    for start in range(0, len(anchors), args.batch_size):
        chunk = anchors[start : start + args.batch_size]
        real_count = len(chunk)
        if real_count < args.batch_size:
            chunk = chunk + [chunk[-1]] * (args.batch_size - real_count)
        samples = [sample_cache[anchor["dataset_index"]] for anchor in chunk]
        batch = atomic_collate(samples)
        observation_np, _ = batch_to_observation(batch)
        observation = jax.tree.map(jnp.asarray, observation_np)
        states = np.asarray(batch["state"])
        empty = [tokenizer.tokenize("", state) for state in states]
        prompted = [
            tokenizer.tokenize(anchor["reverse_prompt"], state)
            for anchor, state in zip(chunk, states, strict=True)
        ]
        actions, directions = jax.device_get(
            _sample_routes(
                model,
                observation,
                jnp.asarray(np.stack([noise_by_key[a["anchor_key"]] for a in chunk])),
                jnp.asarray(np.stack([x[0] for x in empty])),
                jnp.asarray(np.stack([x[1] for x in empty])),
                jnp.asarray(np.stack([x[0] for x in prompted])),
                jnp.asarray(np.stack([x[1] for x in prompted])),
                jnp.asarray(states),
            )
        )
        actions = np.asarray(actions)
        directions = np.asarray(directions, dtype=np.float64)
        for local_index, anchor in enumerate(chunk[:real_count]):
            metadata = metadata_cache[anchor["dataset_index"]]
            trajectories = {}
            for route_index, route in enumerate(ROUTES):
                decoded = decode(
                    states[local_index],
                    np.asarray(metadata["raw_state"]),
                    actions[route_index, local_index],
                )["actions"]
                trajectories[route] = base._trajectory(  # noqa: SLF001
                    fk,
                    np.asarray(metadata["raw_state"]),
                    decoded,
                    anchor["arm"],
                )
            arm_index = ARM_INDEX[anchor["arm"]]
            target_indices = [ATOMIC_NAMES.index(atom) for atom in anchor["reverse_atoms"]]
            direction_empty = directions[0, local_index, arm_index]
            direction_prompt = directions[1, local_index, arm_index]
            similarity_empty = codes[arm_index] @ direction_empty
            similarity_prompt = codes[arm_index] @ direction_prompt
            rows.append(
                {
                    "anchor_key": anchor["anchor_key"],
                    "arm": anchor["arm"],
                    "kind": anchor["kind"],
                    "native_atoms": anchor["atoms"],
                    "reverse_atoms": anchor["reverse_atoms"],
                    "twists_t25": {route: trajectories[route][25].tolist() for route in ROUTES},
                    "twists_t50": {route: trajectories[route][50].tolist() for route in ROUTES},
                    "q1": {
                        "direction_cosine_empty_prompt": float(
                            np.dot(direction_empty, direction_prompt)
                        ),
                        "target_similarity_empty": [float(similarity_empty[i]) for i in target_indices],
                        "target_similarity_prompt": [float(similarity_prompt[i]) for i in target_indices],
                        "top1_empty": int(np.argmax(similarity_empty)),
                        "top1_prompt": int(np.argmax(similarity_prompt)),
                        "target_indices": target_indices,
                    },
                }
            )
        print(f"prompt-route shard {args.shard_index}: {min(start + args.batch_size, len(anchors))}/{len(anchors)}", flush=True)

    payload = {
        "checkpoint": str(args.checkpoint),
        "eligible_total": len(eligible),
        "anchor_count": len(anchors),
        "routes": list(ROUTES),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    del model, params
    gc.collect()
    jax.clear_caches()


if __name__ == "__main__":
    main()
