#!/usr/bin/env python3
"""Causal route ablation for the non-random empty-prompt action prior.

For each pair of nearby robot states with different recorded atomic motions,
keep the evaluated sample A's physical start state and diffusion noise fixed.
The textual instruction is empty in every condition; Pi0.5's discretized state
tokens remain in the prefix.  Cross-composed prefix passes isolate:

* Context state-token prior;
* Q1--Q4 state-token prior;
* the direct continuous-state MLP prior;
* Context image prior;
* Q1--Q4 image prior;
* the complete donor observation prior.

Predicted T25/T50 TCP twists are compared with the recorded A and donor-B TCP
twists.  A positive donor-shift means that changing only the named route moved
the generated endpoint from A's behavior toward B's behavior.
"""

from __future__ import annotations

import argparse
import gc
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
import evaluate_many_cluster_single_dual_steering as base
import evaluate_zm_fk_trajectory_ablation as fk_eval
from evaluate_zm_fk_trajectory_ablation import _output_transform


ROUTES = (
    "baseline",
    "context_state_tokens",
    "atomic_state_tokens",
    "atomic_state_mlp",
    "atomic_state_both",
    "both_state_routes",
    "context_image",
    "atomic_image",
    "both_image_routes",
    "full_donor_condition",
)
TWIST_SCALE = np.asarray([5.0, 5.0, 5.0, 1.0, 1.0, 1.0], dtype=np.float64)


def _donor_map(anchors: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for anchor in anchors:
        grouped[anchor["cluster_key"]].append(anchor)
    result: dict[str, dict] = {}
    for cluster, rows in grouped.items():
        if len(rows) != 2:
            raise ValueError(f"expected two anchors in {cluster}, got {len(rows)}")
        first, second = sorted(rows, key=lambda row: int(row["pair_index"]))
        result[first["anchor_key"]] = second
        result[second["anchor_key"]] = first
    return result


@nnx.jit
def _sample_empty_prior_routes(
    model,
    observation_a,
    observation_image_b,
    noise,
    tokens_state_a,
    mask_state_a,
    tokens_state_b,
    mask_state_b,
    state_a,
    state_b,
):
    observation_a = _model.preprocess_observation(None, observation_a, train=False)
    observation_image_b = _model.preprocess_observation(
        None, observation_image_b, train=False
    )

    def prefix(observation, tokens, mask):
        conditioned = model._with_prompt(observation, tokens, mask)  # noqa: SLF001
        return model._prefix_forward(conditioned)  # noqa: SLF001

    query_aa, mask_aa, kv_aa, _ = prefix(
        observation_a, tokens_state_a, mask_state_a
    )
    query_as, mask_as, kv_as, _ = prefix(
        observation_a, tokens_state_b, mask_state_b
    )
    query_ia, mask_ia, kv_ia, _ = prefix(
        observation_image_b, tokens_state_a, mask_state_a
    )
    query_ib, mask_ib, kv_ib, _ = prefix(
        observation_image_b, tokens_state_b, mask_state_b
    )

    active_a = model._controlled_state(state_a)  # noqa: SLF001
    active_b = model._controlled_state(state_b)  # noqa: SLF001
    _, _, z_aa, _, _ = model._latent(query_aa, active_a)  # noqa: SLF001
    # Only the state tokens visible to Q1--Q4 change.
    _, _, z_as_token, _, _ = model._latent(query_as, active_a)  # noqa: SLF001
    # Only the direct continuous-state MLP input changes.
    _, _, z_as_mlp, _, _ = model._latent(query_aa, active_b)  # noqa: SLF001
    # Both atomic-side state routes change.
    _, _, z_as_both, _, _ = model._latent(query_as, active_b)  # noqa: SLF001
    # Only image pixels visible to Q1--Q4 change.
    _, _, z_ia, _, _ = model._latent(query_ia, active_a)  # noqa: SLF001
    # Complete donor image/state atomic latent.
    _, _, z_ib, _, _ = model._latent(query_ib, active_b)  # noqa: SLF001

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

        return jax.lax.fori_loop(0, 10, step, noise)[
            ..., : model.config.active_action_dim
        ]

    return jnp.stack(
        [
            sample(mask_aa, kv_aa, z_aa),
            sample(mask_as, kv_as, z_aa),
            sample(mask_aa, kv_aa, z_as_token),
            sample(mask_aa, kv_aa, z_as_mlp),
            sample(mask_aa, kv_aa, z_as_both),
            sample(mask_as, kv_as, z_as_both),
            sample(mask_ia, kv_ia, z_aa),
            sample(mask_aa, kv_aa, z_ia),
            sample(mask_ia, kv_ia, z_ia),
            sample(mask_ib, kv_ib, z_ib),
        ],
        axis=0,
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1.0e-8 else 0.0


def _route_metrics(
    prediction: np.ndarray,
    baseline: np.ndarray,
    gt_a: np.ndarray,
    gt_b: np.ndarray,
) -> dict:
    p = prediction / TWIST_SCALE
    p0 = baseline / TWIST_SCALE
    a = gt_a / TWIST_SCALE
    b = gt_b / TWIST_SCALE
    donor_axis = b - a
    donor_norm = float(np.linalg.norm(donor_axis))
    donor_unit = donor_axis / max(donor_norm, 1.0e-8)
    return {
        "cosine_gt_a": _cosine(p, a),
        "cosine_gt_b": _cosine(p, b),
        "closer_to_gt_a": bool(np.linalg.norm(p - a) < np.linalg.norm(p - b)),
        "donor_shift_from_baseline": float(np.dot(p - p0, donor_unit)),
        "route_change_norm": float(np.linalg.norm(p - p0)),
        "gt_pair_separation": donor_norm,
    }


def _aggregate(rows: list[dict]) -> dict:
    result: dict[str, dict] = {}
    for step in (25, 50):
        key = f"t{step}"
        baseline_rows = [row["metrics"][key]["baseline"] for row in rows]
        result[key] = {
            "n": len(rows),
            "baseline_cosine_gt_a": float(
                np.mean([value["cosine_gt_a"] for value in baseline_rows])
            ),
            "baseline_cosine_gt_b": float(
                np.mean([value["cosine_gt_b"] for value in baseline_rows])
            ),
            "baseline_closer_to_gt_a_rate": float(
                np.mean([value["closer_to_gt_a"] for value in baseline_rows])
            ),
            "routes": {},
        }
        for route in ROUTES[1:]:
            values = [row["metrics"][key][route] for row in rows]
            result[key]["routes"][route] = {
                "mean_donor_shift": float(
                    np.mean([value["donor_shift_from_baseline"] for value in values])
                ),
                "median_donor_shift": float(
                    np.median([value["donor_shift_from_baseline"] for value in values])
                ),
                "donor_shift_positive_rate": float(
                    np.mean([value["donor_shift_from_baseline"] > 0 for value in values])
                ),
                "mean_route_change_norm": float(
                    np.mean([value["route_change_norm"] for value in values])
                ),
                "closer_to_gt_a_rate": float(
                    np.mean([value["closer_to_gt_a"] for value in values])
                ),
                "cosine_gt_a": float(
                    np.mean([value["cosine_gt_a"] for value in values])
                ),
                "cosine_gt_b": float(
                    np.mean([value["cosine_gt_b"] for value in values])
                ),
            }
    return result


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

    manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    all_anchors = manifest["anchors"]
    donors = _donor_map(all_anchors)
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
    tokenizer = _paligemma_tokenizer(config.max_token_len)
    decode = _output_transform(args.dataset_root, config)
    fk_eval.TCP_LOCAL_Z_OFFSET_M = 0.20
    fk = fk_eval.SimpleCR1FK(
        Path("assets/cr1_right_arm_tcp_red/urdf/CR1ARMR_TCP_RED.urdf")
    )
    rng = np.random.default_rng(args.seed)
    noises = {
        anchor["anchor_key"]: rng.standard_normal(
            (config.action_horizon, config.action_dim), dtype=np.float32
        )
        for anchor in all_anchors
    }
    # Each process loads only its own shard and paired donors. Loading all 500
    # decoded image samples in every process creates severe CPU oversubscription
    # before the first GPU batch.
    shard_indices = {
        index
        for anchor in anchors
        for index in (
            anchor["dataset_index"],
            donors[anchor["anchor_key"]]["dataset_index"],
        )
    }
    sample_cache = {index: dataset[index] for index in shard_indices}
    metadata_cache = {
        index: dataset._raw.metadata(index)  # noqa: SLF001
        for index in sample_cache
    }
    rows: list[dict] = []

    for start in range(0, len(anchors), args.batch_size):
        chunk = anchors[start : start + args.batch_size]
        real_count = len(chunk)
        if real_count < args.batch_size:
            chunk = chunk + [chunk[-1]] * (args.batch_size - real_count)
        donor_rows = [donors[anchor["anchor_key"]] for anchor in chunk]
        samples_a = [sample_cache[anchor["dataset_index"]] for anchor in chunk]
        samples_b = [sample_cache[donor["dataset_index"]] for donor in donor_rows]
        batch_a = atomic_collate(samples_a)
        batch_b = atomic_collate(samples_b)
        obs_a_np, _ = batch_to_observation(batch_a)
        obs_b_np, _ = batch_to_observation(batch_b)
        obs_a = jax.tree.map(jnp.asarray, obs_a_np)
        obs_image_b = obs_a.replace(
            images=jax.tree.map(jnp.asarray, obs_b_np.images),
            image_masks=jax.tree.map(jnp.asarray, obs_b_np.image_masks),
        )
        states_a = np.asarray(batch_a["state"])
        states_b = np.asarray(batch_b["state"])
        empty_a = [tokenizer.tokenize("", state) for state in states_a]
        empty_b = [tokenizer.tokenize("", state) for state in states_b]
        sampled = np.asarray(
            jax.device_get(
                _sample_empty_prior_routes(
                    model,
                    obs_a,
                    obs_image_b,
                    jnp.asarray(
                        np.stack([noises[anchor["anchor_key"]] for anchor in chunk])
                    ),
                    jnp.asarray(np.stack([value[0] for value in empty_a])),
                    jnp.asarray(np.stack([value[1] for value in empty_a])),
                    jnp.asarray(np.stack([value[0] for value in empty_b])),
                    jnp.asarray(np.stack([value[1] for value in empty_b])),
                    jnp.asarray(states_a),
                    jnp.asarray(states_b),
                )
            )
        )
        for local_index, anchor in enumerate(chunk[:real_count]):
            donor = donor_rows[local_index]
            metadata_a = metadata_cache[anchor["dataset_index"]]
            metadata_b = metadata_cache[donor["dataset_index"]]
            decoded_gt_a = decode(
                states_a[local_index],
                np.asarray(metadata_a["raw_state"]),
                np.asarray(batch_a["actions"][local_index]),
            )["actions"]
            decoded_gt_b = decode(
                states_b[local_index],
                np.asarray(metadata_b["raw_state"]),
                np.asarray(batch_b["actions"][local_index]),
            )["actions"]
            gt_a = base._trajectory(  # noqa: SLF001
                fk, np.asarray(metadata_a["raw_state"]), decoded_gt_a, anchor["arm"]
            )
            gt_b = base._trajectory(  # noqa: SLF001
                fk, np.asarray(metadata_b["raw_state"]), decoded_gt_b, anchor["arm"]
            )
            predicted: dict[str, np.ndarray] = {}
            for route_index, route in enumerate(ROUTES):
                decoded = decode(
                    states_a[local_index],
                    np.asarray(metadata_a["raw_state"]),
                    sampled[route_index, local_index],
                )["actions"]
                predicted[route] = base._trajectory(  # noqa: SLF001
                    fk, np.asarray(metadata_a["raw_state"]), decoded, anchor["arm"]
                )
            metrics: dict[str, dict] = {}
            for step in (25, 50):
                metrics[f"t{step}"] = {
                    route: _route_metrics(
                        predicted[route][step],
                        predicted["baseline"][step],
                        gt_a[step],
                        gt_b[step],
                    )
                    for route in ROUTES
                }
            rows.append(
                {
                    "anchor_key": anchor["anchor_key"],
                    "donor_anchor_key": donor["anchor_key"],
                    "cluster_key": anchor["cluster_key"],
                    "arm": anchor["arm"],
                    "kind": anchor["kind"],
                    "atoms_a": anchor["atoms"],
                    "atoms_b": donor["atoms"],
                    "metrics": metrics,
                    "twists_t25": {
                        "gt_a": gt_a[25].tolist(),
                        "gt_b": gt_b[25].tolist(),
                        **{route: predicted[route][25].tolist() for route in ROUTES},
                    },
                    "twists_t50": {
                        "gt_a": gt_a[50].tolist(),
                        "gt_b": gt_b[50].tolist(),
                        **{route: predicted[route][50].tolist() for route in ROUTES},
                    },
                }
            )
        print(
            f"empty-prior shard {args.shard_index}: "
            f"{min(start + args.batch_size, len(anchors))}/{len(anchors)}",
            flush=True,
        )

    payload = {
        "checkpoint": str(args.checkpoint),
        "anchor_count": len(anchors),
        "routes": list(ROUTES),
        "contract": (
            "empty instruction in all routes; source physical start and diffusion noise fixed; "
            "Pi0.5 state tokens retained; cross-compose Context KV, Q-visible inputs, and direct state MLP"
        ),
        "summary": _aggregate(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    del model, params
    gc.collect()
    jax.clear_caches()


if __name__ == "__main__":
    main()
