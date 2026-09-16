#!/usr/bin/env python3
"""Zero-shot ablation: remove discrete state numbers from Context, keep zM factual.

The normal prefix pass uses ``Task + prompt + State numbers`` and supplies the
Q1--Q3 atomic latent.  A second pass keeps the same Pi0.5 textual wrapper but
uses an empty ``State:`` field; only its KV cache is substituted into the
Action Expert.  Thus the evaluated policy is

    Context(image, prompt, no state numbers) + zM(image, prompt, real state).

The script reports paired flow MSE on factual actions and relative-to-empty TCP
steering for reverse and orthogonal-axis prompts.  This is intentionally a
zero-shot diagnostic; the checkpoint was not trained on an empty State field.
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
import evaluate_500_anchor_atomic_sweep as sweep
import evaluate_many_cluster_single_dual_steering as base
import evaluate_zm_fk_trajectory_ablation as fk_eval
from evaluate_zm_fk_trajectory_ablation import _output_transform


def _other_axis(atom: str) -> str:
    family, axis, sign = atom.split("_")
    return f"{family}_{dict(x='y', y='z', z='x')[axis]}_{sign}"


def _component(atom: str) -> tuple[int, float, float]:
    family, axis, sign_name = atom.split("_")
    index = "xyz".index(axis) + (3 if family == "rotate" else 0)
    sign = 1.0 if sign_name == "pos" else -1.0
    threshold = 1.0 if family == "rotate" else 5.0
    return index, sign, threshold


def _pad(tokens: list[int], max_len: int) -> tuple[np.ndarray, np.ndarray]:
    tokens = tokens[:max_len]
    mask = [True] * len(tokens)
    if len(tokens) < max_len:
        padding = max_len - len(tokens)
        tokens = tokens + [0] * padding
        mask = mask + [False] * padding
    return np.asarray(tokens), np.asarray(mask)


def _tokenize_context_without_state(tokenizer, prompt: str, max_len: int):
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    text = f"Task: {cleaned}, State: ;\nAction: "
    return _pad(tokenizer._tokenizer.encode(text, add_bos=True), max_len)  # noqa: SLF001


@nnx.jit
def _paired_flow_metrics(
    model,
    observation,
    actions,
    normal_tokens,
    normal_mask,
    no_state_tokens,
    no_state_mask,
    rng,
):
    observation = _model.preprocess_observation(None, observation, train=False)
    normal_observation = model._with_prompt(observation, normal_tokens, normal_mask)  # noqa: SLF001
    no_state_observation = model._with_prompt(  # noqa: SLF001
        observation, no_state_tokens, no_state_mask
    )
    query, normal_prefix_mask, normal_kv, _ = model._prefix_forward(normal_observation)  # noqa: SLF001
    _, no_state_prefix_mask, no_state_kv, _ = model._prefix_forward(  # noqa: SLF001
        no_state_observation
    )
    active_state = model._controlled_state(observation.state)  # noqa: SLF001
    _, _, z_model, _, _ = model._latent(query, active_state)  # noqa: SLF001

    noise_rng, time_rng = jax.random.split(rng)
    actions = model._mask_action_condition(actions)  # noqa: SLF001
    noise = model._mask_action_condition(jax.random.normal(noise_rng, actions.shape))  # noqa: SLF001
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    noisy = time[:, None, None] * noise + (1.0 - time[:, None, None]) * actions
    target = model._controlled_actions(noise - actions)  # noqa: SLF001

    def predict(prefix_mask, kv_cache):
        return model._controlled_actions(  # noqa: SLF001
            model._suffix_velocity(prefix_mask, kv_cache, noisy, time, z_model)  # noqa: SLF001
        )

    normal = predict(normal_prefix_mask, normal_kv)
    no_state = predict(no_state_prefix_mask, no_state_kv)
    return (
        jnp.mean(jnp.square(normal - target), axis=(1, 2)),
        jnp.mean(jnp.square(no_state - target), axis=(1, 2)),
        jnp.sqrt(jnp.sum(jnp.square(no_state - normal), axis=(1, 2)))
        / jnp.maximum(jnp.sqrt(jnp.sum(jnp.square(normal), axis=(1, 2))), 1.0e-8),
    )


@nnx.jit
def _sample_pair(
    model,
    observation,
    noise,
    normal_tokens,
    normal_mask,
    no_state_tokens,
    no_state_mask,
):
    observation = _model.preprocess_observation(None, observation, train=False)
    normal_observation = model._with_prompt(observation, normal_tokens, normal_mask)  # noqa: SLF001
    no_state_observation = model._with_prompt(  # noqa: SLF001
        observation, no_state_tokens, no_state_mask
    )
    query, normal_prefix_mask, normal_kv, _ = model._prefix_forward(normal_observation)  # noqa: SLF001
    _, no_state_prefix_mask, no_state_kv, _ = model._prefix_forward(  # noqa: SLF001
        no_state_observation
    )
    active_state = model._controlled_state(observation.state)  # noqa: SLF001
    _, _, z_model, _, _ = model._latent(query, active_state)  # noqa: SLF001
    noise = model._mask_action_condition(noise)  # noqa: SLF001
    batch_size = noise.shape[0]

    def sample(prefix_mask, kv_cache):
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
        [sample(normal_prefix_mask, normal_kv), sample(no_state_prefix_mask, no_state_kv)],
        axis=0,
    )


def _aggregate_flow(rows: list[dict]) -> dict:
    result = {}
    for mode in ("empty", "atomic"):
        selected = [row for row in rows if row["prompt_mode"] == mode]
        normal = np.asarray([row["normal_mse"] for row in selected])
        ablated = np.asarray([row["no_context_state_mse"] for row in selected])
        result[mode] = {
            "n": len(selected),
            "normal_mse": float(normal.mean()),
            "no_context_state_mse": float(ablated.mean()),
            "mse_ratio": float(ablated.mean() / normal.mean()),
            "worse_rate": float(np.mean(ablated > normal)),
            "relative_velocity_change": float(
                np.mean([row["relative_velocity_change"] for row in selected])
            ),
        }
    return result


def _aggregate_steering(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["arm"], row["kind"])].append(row)
        grouped[(row["variant"], "both", "all")].append(row)
    result = []
    for (variant, arm, kind), selected in sorted(grouped.items()):
        result.append(
            {
                "variant": variant,
                "arm": arm,
                "kind": kind,
                "n": len(selected),
                **{
                    f"{condition}_{level}_success_rate": float(
                        np.mean([row[f"{condition}_{level}_success"] for row in selected])
                    )
                    for condition in ("normal", "no_context_state")
                    for level in ("soft", "strict")
                },
            }
        )
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
    noise = {
        anchor["anchor_key"]: rng.standard_normal(
            (config.action_horizon, config.action_dim), dtype=np.float32
        )
        for anchor in all_anchors
    }
    flow_rows = []
    steering_rows = []

    for start in range(0, len(anchors), args.batch_size):
        chunk = anchors[start : start + args.batch_size]
        real_count = len(chunk)
        if real_count < args.batch_size:
            chunk = chunk + [chunk[-1]] * (args.batch_size - real_count)
        samples = [dataset[row["dataset_index"]] for row in chunk]
        batch = atomic_collate(samples)
        observation_np, actions_np = batch_to_observation(batch)
        observation = jax.tree.map(jnp.asarray, observation_np)
        actions = jnp.asarray(actions_np)
        states = np.asarray(batch["state"])

        # Paired factual-action MSE for empty and correct atomic prompts.
        for prompt_mode in ("empty", "atomic"):
            texts = ["" if prompt_mode == "empty" else row["atomic_prompt"] for row in chunk]
            normal = [
                tokenizer.tokenize(text, state)
                for text, state in zip(texts, states, strict=True)
            ]
            no_state = [
                _tokenize_context_without_state(tokenizer, text, config.max_token_len)
                for text in texts
            ]
            normal_mse, no_state_mse, relative = jax.device_get(
                _paired_flow_metrics(
                    model,
                    observation,
                    actions,
                    jnp.asarray(np.stack([value[0] for value in normal])),
                    jnp.asarray(np.stack([value[1] for value in normal])),
                    jnp.asarray(np.stack([value[0] for value in no_state])),
                    jnp.asarray(np.stack([value[1] for value in no_state])),
                    jax.random.key(args.seed + 100000 * args.shard_index + start),
                )
            )
            for local_index, anchor in enumerate(chunk[:real_count]):
                flow_rows.append(
                    {
                        "anchor_key": anchor["anchor_key"],
                        "prompt_mode": prompt_mode,
                        "normal_mse": float(normal_mse[local_index]),
                        "no_context_state_mse": float(no_state_mse[local_index]),
                        "relative_velocity_change": float(relative[local_index]),
                    }
                )

        # Roll out empty, reverse and other-axis prompts under both Context choices.
        prompt_specs = []
        for anchor in chunk:
            reverse_atoms = tuple(base._opposite(atom) for atom in anchor["atoms"])  # noqa: SLF001
            other_atoms = tuple(_other_axis(atom) for atom in anchor["atoms"])
            prompt_specs.extend(
                [
                    ("empty", "", ()),
                    ("reverse", anchor["reverse_prompt"], reverse_atoms),
                    ("other_axis", sweep._canonical_prompt(anchor, other_atoms), other_atoms),  # noqa: SLF001
                ]
            )
        endpoints: dict[tuple[int, str], dict[str, np.ndarray]] = {}
        # Prompt batches are kept shape-stable by evaluating one variant at a time.
        for variant_index, variant in enumerate(("empty", "reverse", "other_axis")):
            selected = [prompt_specs[3 * index + variant_index] for index in range(len(chunk))]
            texts = [item[1] for item in selected]
            normal = [
                tokenizer.tokenize(text, state)
                for text, state in zip(texts, states, strict=True)
            ]
            no_state = [
                _tokenize_context_without_state(tokenizer, text, config.max_token_len)
                for text in texts
            ]
            predictions = np.asarray(
                jax.device_get(
                    _sample_pair(
                        model,
                        observation,
                        jnp.asarray(np.stack([noise[row["anchor_key"]] for row in chunk])),
                        jnp.asarray(np.stack([value[0] for value in normal])),
                        jnp.asarray(np.stack([value[1] for value in normal])),
                        jnp.asarray(np.stack([value[0] for value in no_state])),
                        jnp.asarray(np.stack([value[1] for value in no_state])),
                    )
                )
            )
            for local_index, anchor in enumerate(chunk[:real_count]):
                metadata = dataset._raw.metadata(anchor["dataset_index"])  # noqa: SLF001
                route_endpoints = {}
                for route_index, route in enumerate(("normal", "no_context_state")):
                    decoded = decode(
                        np.asarray(batch["state"][local_index]),
                        np.asarray(metadata["raw_state"]),
                        predictions[route_index, local_index],
                    )["actions"]
                    trajectory = base._trajectory(  # noqa: SLF001
                        fk, np.asarray(metadata["raw_state"]), decoded, anchor["arm"]
                    )
                    route_endpoints[route] = trajectory[50]
                endpoints[(local_index, variant)] = route_endpoints

        for local_index, anchor in enumerate(chunk[:real_count]):
            empty = endpoints[(local_index, "empty")]
            for variant, atoms in (
                ("reverse", tuple(base._opposite(atom) for atom in anchor["atoms"])),  # noqa: SLF001
                ("other_axis", tuple(_other_axis(atom) for atom in anchor["atoms"])),
            ):
                row = {
                    "anchor_key": anchor["anchor_key"],
                    "arm": anchor["arm"],
                    "kind": anchor["kind"],
                    "variant": variant,
                    "target_atoms": list(atoms),
                }
                for condition in ("normal", "no_context_state"):
                    delta = endpoints[(local_index, variant)][condition] - empty[condition]
                    scores = [
                        sign * delta[index] / threshold
                        for index, sign, threshold in map(_component, atoms)
                    ]
                    row[f"{condition}_soft_success"] = bool(all(score > 0 for score in scores))
                    row[f"{condition}_strict_success"] = bool(all(score > 1 for score in scores))
                    row[f"{condition}_scores"] = [float(score) for score in scores]
                steering_rows.append(row)
        print(
            f"context-no-state shard {args.shard_index}: "
            f"{min(start + args.batch_size, len(anchors))}/{len(anchors)}",
            flush=True,
        )

    payload = {
        "checkpoint": str(args.checkpoint),
        "anchor_count": len(anchors),
        "contract": "normal zM always uses image+prompt+real state; only Action Expert Context KV uses an empty State field",
        "flow_summary": _aggregate_flow(flow_rows),
        "steering_summary": _aggregate_steering(steering_rows),
        "flow_rows": flow_rows,
        "steering_rows": steering_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "flow": len(flow_rows), "steering": len(steering_rows)}, indent=2))


if __name__ == "__main__":
    main()
