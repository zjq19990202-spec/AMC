#!/usr/bin/env python3
"""Attribute counterfactual atomic-prompt steering to context KV versus zM.

For every reviewed isolated-arm anchor, the image, robot state, and initial
flow noise remain fixed.  Empty and steering prompts are encoded in two
independent prefix passes, then cross-composed as

    empty KV + empty zM       (00, neutral baseline)
    steer KV + empty zM       (10, context-only prompt route)
    empty KV + steer zM       (01, atomic-only prompt route)
    steer KV + steer zM       (11, complete steering prompt)

The endpoint target-axis response is additionally split with the exact
two-factor Shapley decomposition, so interaction between the two routes is
not double-counted.  Two counterfactual prompt families are evaluated:

* reverse: the reviewed semantic sign-reversal prompt;
* other_axis: preserve motion family/sign and rotate x->y->z->x.
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
import evaluate_500_anchor_atomic_sweep as sweep
import evaluate_many_cluster_single_dual_steering as base
import evaluate_zm_fk_trajectory_ablation as fk_eval
from evaluate_zm_fk_trajectory_ablation import _output_transform


ROUTES = ("empty", "context", "atomic", "both")
TRANSLATION_THRESHOLD_MM = 5.0
ROTATION_THRESHOLD_DEG = 1.0


def _other_axis(atom: str) -> str:
    family, axis, sign = atom.split("_")
    next_axis = {"x": "y", "y": "z", "z": "x"}[axis]
    return f"{family}_{next_axis}_{sign}"


def _component(atom: str) -> tuple[int, float, float]:
    family, axis, sign_name = atom.split("_")
    index = "xyz".index(axis) + (3 if family == "rotate" else 0)
    sign = 1.0 if sign_name == "pos" else -1.0
    threshold = ROTATION_THRESHOLD_DEG if family == "rotate" else TRANSLATION_THRESHOLD_MM
    return index, sign, threshold


@nnx.jit
def _sample_prompt_routes(
    model,
    observation,
    noise,
    empty_tokens,
    empty_mask,
    steer_tokens,
    steer_mask,
):
    observation = _model.preprocess_observation(None, observation, train=False)
    empty_observation = model._with_prompt(observation, empty_tokens, empty_mask)  # noqa: SLF001
    steer_observation = model._with_prompt(observation, steer_tokens, steer_mask)  # noqa: SLF001

    empty_query, empty_prefix_mask, empty_kv, _ = model._prefix_forward(  # noqa: SLF001
        empty_observation
    )
    steer_query, steer_prefix_mask, steer_kv, _ = model._prefix_forward(  # noqa: SLF001
        steer_observation
    )
    active_state = model._controlled_state(observation.state)  # noqa: SLF001
    _, _, empty_zm, _, _ = model._latent(empty_query, active_state)  # noqa: SLF001
    _, _, steer_zm, _, _ = model._latent(steer_query, active_state)  # noqa: SLF001

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

        result = jax.lax.fori_loop(0, 10, step, noise)
        return result[..., : model.config.active_action_dim]

    return jnp.stack(
        [
            sample(empty_prefix_mask, empty_kv, empty_zm),
            sample(steer_prefix_mask, steer_kv, empty_zm),
            sample(empty_prefix_mask, empty_kv, steer_zm),
            sample(steer_prefix_mask, steer_kv, steer_zm),
        ],
        axis=0,
    )


def _prompt_cases(anchor: dict) -> list[dict]:
    reverse_atoms = tuple(base._opposite(atom) for atom in anchor["atoms"])  # noqa: SLF001
    other_atoms = tuple(_other_axis(atom) for atom in anchor["atoms"])
    return [
        {
            "variant": "reverse",
            "prompt": anchor["reverse_prompt"],
            "target_atoms": list(reverse_atoms),
        },
        {
            "variant": "other_axis",
            "prompt": sweep._canonical_prompt(anchor, other_atoms),  # noqa: SLF001
            "target_atoms": list(other_atoms),
        },
    ]


def _route_metrics(endpoints: dict[str, np.ndarray], atoms: list[str]) -> dict:
    components = [_component(atom) for atom in atoms]
    route_scores: dict[str, list[float]] = {}
    for route in ("context", "atomic", "both"):
        delta = endpoints[route] - endpoints["empty"]
        route_scores[route] = [
            float(sign * delta[index] / threshold)
            for index, sign, threshold in components
        ]

    context_shapley = []
    atomic_shapley = []
    for index, sign, threshold in components:
        y00 = endpoints["empty"][index]
        y10 = endpoints["context"][index]
        y01 = endpoints["atomic"][index]
        y11 = endpoints["both"][index]
        context_shapley.append(
            float(sign * 0.5 * ((y10 - y00) + (y11 - y01)) / threshold)
        )
        atomic_shapley.append(
            float(sign * 0.5 * ((y01 - y00) + (y11 - y10)) / threshold)
        )
    return {
        "normalized_target_scores": route_scores,
        "soft_success": {
            route: bool(all(value > 0.0 for value in scores))
            for route, scores in route_scores.items()
        },
        "strict_success": {
            route: bool(all(value > 1.0 for value in scores))
            for route, scores in route_scores.items()
        },
        "mean_context_shapley": float(np.mean(context_shapley)),
        "mean_atomic_shapley": float(np.mean(atomic_shapley)),
        "context_shapley": context_shapley,
        "atomic_shapley": atomic_shapley,
    }


def _aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        for arm in (row["arm"], "both"):
            for kind in (row["kind"], "all"):
                groups[(row["variant"], arm, kind)].append(row)

    summaries = []
    for (variant, arm, kind), selected in sorted(groups.items()):
        context_shapley = np.asarray([row["mean_context_shapley"] for row in selected])
        atomic_shapley = np.asarray([row["mean_atomic_shapley"] for row in selected])
        total = context_shapley + atomic_shapley
        positive_context = np.maximum(context_shapley, 0.0).sum()
        positive_atomic = np.maximum(atomic_shapley, 0.0).sum()
        positive_sum = positive_context + positive_atomic
        summaries.append(
            {
                "variant": variant,
                "arm": arm,
                "kind": kind,
                "n": len(selected),
                "context_soft_success_rate": float(
                    np.mean([row["soft_success"]["context"] for row in selected])
                ),
                "atomic_soft_success_rate": float(
                    np.mean([row["soft_success"]["atomic"] for row in selected])
                ),
                "both_soft_success_rate": float(
                    np.mean([row["soft_success"]["both"] for row in selected])
                ),
                "context_strict_success_rate": float(
                    np.mean([row["strict_success"]["context"] for row in selected])
                ),
                "atomic_strict_success_rate": float(
                    np.mean([row["strict_success"]["atomic"] for row in selected])
                ),
                "both_strict_success_rate": float(
                    np.mean([row["strict_success"]["both"] for row in selected])
                ),
                "mean_context_shapley": float(context_shapley.mean()),
                "mean_atomic_shapley": float(atomic_shapley.mean()),
                "mean_complete_target_effect": float(total.mean()),
                "context_shapley_dominance_rate": float(np.mean(context_shapley > atomic_shapley)),
                "positive_shapley_context_share": (
                    float(positive_context / positive_sum) if positive_sum > 0 else None
                ),
            }
        )
    return summaries


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
    args = parser.parse_args()

    manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    all_anchors = manifest["anchors"]
    anchors = [
        anchor for index, anchor in enumerate(all_anchors)
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
    fk = fk_eval.SimpleCR1FK(Path("assets/cr1_right_arm_tcp_red/urdf/CR1ARMR_TCP_RED.urdf"))

    rng = np.random.default_rng(args.seed)
    noise_by_anchor = {
        anchor["anchor_key"]: rng.standard_normal(
            (config.action_horizon, config.action_dim), dtype=np.float32
        )
        for anchor in all_anchors
    }
    specs = [
        {**{key: value for key, value in anchor.items() if key != "state_7d"}, **case}
        for anchor in anchors
        for case in _prompt_cases(anchor)
    ]
    sample_cache = {index: dataset[index] for index in {row["dataset_index"] for row in specs}}
    metadata_cache = {
        index: dataset._raw.metadata(index)  # noqa: SLF001
        for index in sample_cache
    }

    rows = []
    for start in range(0, len(specs), args.batch_size):
        chunk = specs[start : start + args.batch_size]
        real_count = len(chunk)
        if real_count < args.batch_size:
            chunk = chunk + [chunk[-1]] * (args.batch_size - real_count)
        samples = [sample_cache[row["dataset_index"]] for row in chunk]
        batch = atomic_collate(samples)
        observation_np, _ = batch_to_observation(batch)
        observation = jax.tree.map(jnp.asarray, observation_np)
        steer = [
            tokenizer.tokenize(row["prompt"], np.asarray(sample["state"]))
            for row, sample in zip(chunk, samples, strict=True)
        ]
        empty = [
            tokenizer.tokenize("", np.asarray(sample["state"]))
            for sample in samples
        ]
        noise = jnp.asarray(np.stack([noise_by_anchor[row["anchor_key"]] for row in chunk]))
        sampled = np.asarray(
            jax.device_get(
                _sample_prompt_routes(
                    model,
                    observation,
                    noise,
                    jnp.asarray(np.stack([value[0] for value in empty])),
                    jnp.asarray(np.stack([value[1] for value in empty])),
                    jnp.asarray(np.stack([value[0] for value in steer])),
                    jnp.asarray(np.stack([value[1] for value in steer])),
                )
            )
        )
        for local_index, row in enumerate(chunk[:real_count]):
            metadata = metadata_cache[row["dataset_index"]]
            endpoints = {}
            for route_index, route in enumerate(ROUTES):
                decoded = decode(
                    np.asarray(batch["state"][local_index]),
                    np.asarray(metadata["raw_state"]),
                    sampled[route_index, local_index],
                )["actions"]
                trajectory = base._trajectory(  # noqa: SLF001
                    fk, np.asarray(metadata["raw_state"]), decoded, row["arm"]
                )
                endpoints[route] = trajectory[50]
            metrics = _route_metrics(endpoints, row["target_atoms"])
            rows.append(
                {
                    **{key: value for key, value in row.items() if key != "prompt"},
                    "prompt": row["prompt"],
                    "endpoints_t50": {key: value.tolist() for key, value in endpoints.items()},
                    **metrics,
                }
            )
        print(
            f"prompt-route shard {args.shard_index}: "
            f"{min(start + args.batch_size, len(specs))}/{len(specs)}",
            flush=True,
        )

    payload = {
        "checkpoint": str(args.checkpoint),
        "selection_manifest": str(args.selection_manifest),
        "anchor_count": len(anchors),
        "case_count": len(rows),
        "tcp_offset_m": 0.20,
        "thresholds": {
            "translation_mm": TRANSLATION_THRESHOLD_MM,
            "rotation_deg": ROTATION_THRESHOLD_DEG,
        },
        "contract": "00 empty; 10 steer KV only; 01 steer zM only; 11 steer both; same image/state/noise",
        "summary": _aggregate(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, indent=2))
    del model, params
    gc.collect()
    jax.clear_caches()


if __name__ == "__main__":
    main()
