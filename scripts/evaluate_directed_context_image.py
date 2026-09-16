#!/usr/bin/env python3
"""Replace only Context images with target-atom donor images; keep zM factual."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

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
from evaluate_context_without_state import _component
from evaluate_directed_context_state import _directed_pairs
from evaluate_hybrid_image_routes import _hybrid_image_metrics
from evaluate_state_image_prior_conflict import _sample_pair
from evaluate_zm_fk_trajectory_ablation import _output_transform


def _aggregate(flow: list[dict], steer: list[dict]) -> tuple[dict, list[dict]]:
    normal = np.asarray([row["normal_mse"] for row in flow])
    directed = np.asarray([row["directed_context_image_mse"] for row in flow])
    flow_summary = {
        "n": len(flow),
        "normal_mse": float(normal.mean()),
        "directed_context_image_mse": float(directed.mean()),
        "mse_ratio": float(directed.mean() / normal.mean()),
        "paired_ratio_median": float(np.median(directed / np.maximum(normal, 1.0e-8))),
        "worse_rate": float(np.mean(directed > normal)),
        "relative_velocity_change": float(
            np.mean([row["relative_velocity_change"] for row in flow])
        ),
    }
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in steer:
        groups[(row["arm"], row["target_kind"])].append(row)
        groups[("both", "all")].append(row)
    steering_summary = []
    for (arm, kind), rows in sorted(groups.items()):
        result = {"arm": arm, "target_kind": kind, "n": len(rows)}
        for route in ("normal", "directed_context_image"):
            for level in ("soft", "strict"):
                result[f"{route}_{level}_success_rate"] = float(
                    np.mean([row[f"{route}_{level}_success"] for row in rows])
                )
            scores = np.asarray([min(row[f"{route}_scores"]) for row in rows])
            result[f"{route}_min_score_mean"] = float(scores.mean())
            result[f"{route}_min_score_median"] = float(np.median(scores))
        steering_summary.append(result)
    return flow_summary, steering_summary


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
    all_pairs = _directed_pairs(manifest["anchors"])
    pairs = [
        pair for index, pair in enumerate(all_pairs)
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
    noises = {
        pair["source"]["anchor_key"]: rng.standard_normal(
            (config.action_horizon, config.action_dim), dtype=np.float32
        )
        for pair in all_pairs
    }
    flow_rows = []
    steering_rows = []

    for start in range(0, len(pairs), args.batch_size):
        chunk = pairs[start : start + args.batch_size]
        real_count = len(chunk)
        if real_count < args.batch_size:
            chunk = chunk + [chunk[-1]] * (args.batch_size - real_count)
        source_samples = [dataset[pair["source"]["dataset_index"]] for pair in chunk]
        donor_samples = [dataset[pair["target"]["dataset_index"]] for pair in chunk]
        source_batch = atomic_collate(source_samples)
        donor_batch = atomic_collate(donor_samples)
        source_np, actions_np = batch_to_observation(source_batch)
        donor_np, _ = batch_to_observation(donor_batch)
        source_observation = jax.tree.map(jnp.asarray, source_np)
        donor_observation = source_observation.replace(
            images=jax.tree.map(jnp.asarray, donor_np.images),
            image_masks=jax.tree.map(jnp.asarray, donor_np.image_masks),
        )
        actions = jnp.asarray(actions_np)
        source_states = np.asarray(source_batch["state"])
        target_texts = [pair["target"]["atomic_prompt"] for pair in chunk]
        tokenized = [
            tokenizer.tokenize(text, state)
            for text, state in zip(target_texts, source_states, strict=True)
        ]
        mse, _, relative, _ = jax.device_get(
            _hybrid_image_metrics(
                model,
                source_observation,
                donor_observation,
                actions,
                jnp.asarray(np.stack([value[0] for value in tokenized])),
                jnp.asarray(np.stack([value[1] for value in tokenized])),
                jax.random.key(args.seed + 100000 * args.shard_index + start),
            )
        )
        for index, pair in enumerate(chunk[:real_count]):
            flow_rows.append(
                {
                    "source_anchor": pair["source"]["anchor_key"],
                    "target_anchor": pair["target"]["anchor_key"],
                    "arm": pair["source"]["arm"],
                    "target_kind": pair["target"]["kind"],
                    "normal_mse": float(mse[0, index]),
                    "directed_context_image_mse": float(mse[1, index]),
                    "relative_velocity_change": float(relative[1, index]),
                }
            )

        endpoints: dict[tuple[int, str], dict[str, np.ndarray]] = {}
        for prompt_mode in ("empty", "target"):
            texts = ["" for _ in chunk] if prompt_mode == "empty" else target_texts
            tokenized = [
                tokenizer.tokenize(text, state)
                for text, state in zip(texts, source_states, strict=True)
            ]
            predictions = np.asarray(
                jax.device_get(
                    _sample_pair(
                        model,
                        source_observation,
                        donor_observation,
                        jnp.asarray(
                            np.stack([noises[pair["source"]["anchor_key"]] for pair in chunk])
                        ),
                        jnp.asarray(np.stack([value[0] for value in tokenized])),
                        jnp.asarray(np.stack([value[1] for value in tokenized])),
                    )
                )
            )
            for index, pair in enumerate(chunk[:real_count]):
                metadata = dataset._raw.metadata(pair["source"]["dataset_index"])  # noqa: SLF001
                route_endpoints = {}
                for route_index, route in ((0, "normal"), (1, "directed_context_image")):
                    decoded = decode(
                        np.asarray(source_batch["state"][index]),
                        np.asarray(metadata["raw_state"]),
                        predictions[route_index, index],
                    )["actions"]
                    trajectory = base._trajectory(  # noqa: SLF001
                        fk, np.asarray(metadata["raw_state"]), decoded, pair["source"]["arm"]
                    )
                    route_endpoints[route] = trajectory[50]
                endpoints[(index, prompt_mode)] = route_endpoints

        for index, pair in enumerate(chunk[:real_count]):
            row = {
                "source_anchor": pair["source"]["anchor_key"],
                "target_anchor": pair["target"]["anchor_key"],
                "arm": pair["source"]["arm"],
                "source_atoms": pair["source"]["atoms"],
                "target_atoms": pair["target"]["atoms"],
                "target_kind": pair["target"]["kind"],
            }
            for route in ("normal", "directed_context_image"):
                delta = endpoints[(index, "target")][route] - endpoints[(index, "empty")][route]
                scores = [
                    sign * delta[component] / threshold
                    for component, sign, threshold in map(_component, pair["target"]["atoms"])
                ]
                row[f"{route}_soft_success"] = bool(all(score > 0 for score in scores))
                row[f"{route}_strict_success"] = bool(all(score > 1 for score in scores))
                row[f"{route}_scores"] = [float(score) for score in scores]
            steering_rows.append(row)
        print(
            f"directed-context-image shard {args.shard_index}: "
            f"{min(start + args.batch_size, len(pairs))}/{len(pairs)}",
            flush=True,
        )

    flow_summary, steering_summary = _aggregate(flow_rows, steering_rows)
    payload = {
        "checkpoint": str(args.checkpoint),
        "pair_count": len(pairs),
        "contract": (
            "source state/noise fixed; zM uses source images/state plus target prompt; "
            "only Context KV images switch source->nearby target-atom donor; "
            "steering is target-prompt minus route-matched empty prompt"
        ),
        "flow_summary": flow_summary,
        "steering_summary": steering_summary,
        "flow_rows": flow_rows,
        "steering_rows": steering_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "flow": len(flow_rows)}, indent=2))


if __name__ == "__main__":
    main()
