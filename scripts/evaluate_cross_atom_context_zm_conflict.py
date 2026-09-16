#!/usr/bin/env python3
"""Direct context-versus-zM conflict test on nearby-state opposite atoms.

For every same-cluster pair that contains an opposite sign on the same TCP
twist component, encode two factual prefixes while keeping the evaluated
state fixed:

  A = image A + atomic prompt A + state A
  B = image B + atomic prompt B + state A

Then sample AA, context(A)+zM(B), context(B)+zM(A), and BB with the same flow
noise.  On a sign-conflicting component, the mixed trajectory unambiguously
votes for either its context or its zM condition.
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
import evaluate_many_cluster_single_dual_steering as base
import evaluate_zm_fk_trajectory_ablation as fk_eval
from evaluate_zm_fk_trajectory_ablation import _output_transform


ROUTES = ("aa", "context_a_z_b", "context_b_z_a", "bb")


def _atom_component(atom: str) -> tuple[str, int, float, float]:
    family, axis, sign_name = atom.split("_")
    key = f"{family}_{axis}"
    index = "xyz".index(axis) + (3 if family == "rotate" else 0)
    sign = 1.0 if sign_name == "pos" else -1.0
    threshold = 1.0 if family == "rotate" else 5.0
    return key, index, sign, threshold


def _conflicts(a_atoms: list[str], b_atoms: list[str]) -> list[dict]:
    a = {_atom_component(atom)[0]: (atom, *_atom_component(atom)[1:]) for atom in a_atoms}
    b = {_atom_component(atom)[0]: (atom, *_atom_component(atom)[1:]) for atom in b_atoms}
    rows = []
    for key in sorted(a.keys() & b.keys()):
        a_atom, index, a_sign, threshold = a[key]
        b_atom, b_index, b_sign, b_threshold = b[key]
        if a_sign == b_sign:
            continue
        assert index == b_index and threshold == b_threshold
        rows.append(
            {
                "component": key,
                "index": index,
                "threshold": threshold,
                "a_atom": a_atom,
                "b_atom": b_atom,
                "a_sign": a_sign,
                "b_sign": b_sign,
            }
        )
    return rows


@nnx.jit
def _sample_cross(model, observation_a, observation_b, noise, tokens_a, mask_a, tokens_b, mask_b):
    observation_a = _model.preprocess_observation(None, observation_a, train=False)
    observation_b = _model.preprocess_observation(None, observation_b, train=False)
    prefix_a = model._with_prompt(observation_a, tokens_a, mask_a)  # noqa: SLF001
    prefix_b = model._with_prompt(observation_b, tokens_b, mask_b)  # noqa: SLF001
    query_a, mask_prefix_a, kv_a, _ = model._prefix_forward(prefix_a)  # noqa: SLF001
    query_b, mask_prefix_b, kv_b, _ = model._prefix_forward(prefix_b)  # noqa: SLF001
    active_state = model._controlled_state(observation_a.state)  # noqa: SLF001
    _, _, z_a, _, _ = model._latent(query_a, active_state)  # noqa: SLF001
    _, _, z_b, _, _ = model._latent(query_b, active_state)  # noqa: SLF001
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
            sample(mask_prefix_a, kv_a, z_a),
            sample(mask_prefix_a, kv_a, z_b),
            sample(mask_prefix_b, kv_b, z_a),
            sample(mask_prefix_b, kv_b, z_b),
        ],
        axis=0,
    )


def _winner(value: float, context_sign: float, z_sign: float, threshold: float) -> str:
    if context_sign * value > threshold:
        return "context"
    if z_sign * value > threshold:
        return "zm"
    return "neither"


def _summary(rows: list[dict]) -> dict:
    votes = [vote for row in rows for vote in row["votes"]]
    result = {"pair_orientations": len(rows), "component_votes": len(votes)}
    for route in ("context_a_z_b", "context_b_z_a"):
        values = [vote[f"{route}_winner"] for vote in votes]
        result[route] = {
            key: float(np.mean([value == key for value in values]))
            for key in ("context", "zm", "neither")
        }
    baseline = []
    for vote in votes:
        baseline.append(vote["aa_matches_a"])
        baseline.append(vote["bb_matches_b"])
    result["matched_baseline_requested_sign_rate"] = float(np.mean(baseline))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for anchor in manifest["anchors"]:
        grouped[anchor["cluster_key"]].append(anchor)
    specs = []
    for cluster, anchors in sorted(grouped.items()):
        if len(anchors) != 2:
            continue
        first, second = sorted(anchors, key=lambda row: int(row["pair_index"]))
        for a, b in ((first, second), (second, first)):
            conflicts = _conflicts(a["atoms"], b["atoms"])
            if conflicts:
                specs.append({"cluster_key": cluster, "a": a, "b": b, "conflicts": conflicts})

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
    noises = {
        spec["a"]["anchor_key"]: rng.standard_normal(
            (config.action_horizon, config.action_dim), dtype=np.float32
        )
        for spec in specs
    }
    sample_cache = {
        index: dataset[index]
        for spec in specs
        for index in (spec["a"]["dataset_index"], spec["b"]["dataset_index"])
    }
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
        samples_a = [sample_cache[spec["a"]["dataset_index"]] for spec in chunk]
        samples_b = [sample_cache[spec["b"]["dataset_index"]] for spec in chunk]
        batch_a = atomic_collate(samples_a)
        batch_b = atomic_collate(samples_b)
        observation_a_np, _ = batch_to_observation(batch_a)
        observation_b_np, _ = batch_to_observation(batch_b)
        observation_a = jax.tree.map(jnp.asarray, observation_a_np)
        donor_images = jax.tree.map(jnp.asarray, observation_b_np.images)
        donor_masks = jax.tree.map(jnp.asarray, observation_b_np.image_masks)
        observation_b = observation_a.replace(images=donor_images, image_masks=donor_masks)
        states = np.asarray(batch_a["state"])
        token_a = [
            tokenizer.tokenize(spec["a"]["atomic_prompt"], state)
            for spec, state in zip(chunk, states, strict=True)
        ]
        token_b = [
            tokenizer.tokenize(spec["b"]["atomic_prompt"], state)
            for spec, state in zip(chunk, states, strict=True)
        ]
        sampled = np.asarray(
            jax.device_get(
                _sample_cross(
                    model,
                    observation_a,
                    observation_b,
                    jnp.asarray(np.stack([noises[spec["a"]["anchor_key"]] for spec in chunk])),
                    jnp.asarray(np.stack([value[0] for value in token_a])),
                    jnp.asarray(np.stack([value[1] for value in token_a])),
                    jnp.asarray(np.stack([value[0] for value in token_b])),
                    jnp.asarray(np.stack([value[1] for value in token_b])),
                )
            )
        )
        for local_index, spec in enumerate(chunk[:real_count]):
            a = spec["a"]
            metadata = metadata_cache[a["dataset_index"]]
            endpoints = {}
            for route_index, route in enumerate(ROUTES):
                decoded = decode(
                    np.asarray(batch_a["state"][local_index]),
                    np.asarray(metadata["raw_state"]),
                    sampled[route_index, local_index],
                )["actions"]
                trajectory = base._trajectory(  # noqa: SLF001
                    fk, np.asarray(metadata["raw_state"]), decoded, a["arm"]
                )
                endpoints[route] = trajectory[50]
            votes = []
            for conflict in spec["conflicts"]:
                index = conflict["index"]
                threshold = conflict["threshold"]
                a_sign = conflict["a_sign"]
                b_sign = conflict["b_sign"]
                votes.append(
                    {
                        **conflict,
                        "aa_value": float(endpoints["aa"][index]),
                        "context_a_z_b_value": float(endpoints["context_a_z_b"][index]),
                        "context_b_z_a_value": float(endpoints["context_b_z_a"][index]),
                        "bb_value": float(endpoints["bb"][index]),
                        "aa_matches_a": bool(a_sign * endpoints["aa"][index] > threshold),
                        "bb_matches_b": bool(b_sign * endpoints["bb"][index] > threshold),
                        "context_a_z_b_winner": _winner(
                            float(endpoints["context_a_z_b"][index]), a_sign, b_sign, threshold
                        ),
                        "context_b_z_a_winner": _winner(
                            float(endpoints["context_b_z_a"][index]), b_sign, a_sign, threshold
                        ),
                    }
                )
            rows.append(
                {
                    "cluster_key": spec["cluster_key"],
                    "arm": a["arm"],
                    "a_anchor": a["anchor_key"],
                    "b_anchor": spec["b"]["anchor_key"],
                    "a_atoms": a["atoms"],
                    "b_atoms": spec["b"]["atoms"],
                    "endpoints_t50": {key: value.tolist() for key, value in endpoints.items()},
                    "votes": votes,
                }
            )
        print(f"cross-atom conflict: {min(start + args.batch_size, len(specs))}/{len(specs)}", flush=True)

    payload = {
        "checkpoint": str(args.checkpoint),
        "state_cluster_pairs_with_sign_conflict": len(specs) // 2,
        "pair_orientations": len(specs),
        "tcp_offset_m": 0.20,
        "contract": "same evaluated state/noise; A/B differ in nearby-cluster images and atomic prompts; direct opposite-component vote",
        "summary": _summary(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
