#!/usr/bin/env python3
"""Compare zT for dual-atom horizons with different temporal organization."""

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

from atomic_latent_vla.atomic import ATOMIC_NAMES
from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import (
    build_atomic_text_loader,
    text_batch_to_observation,
)


@nnx.jit
def _encode(model, observation):
    query_hidden, _, _ = model._prefix_forward(observation)  # noqa: SLF001
    state = model._controlled_state(observation.state)  # noqa: SLF001
    zt = model.queries.text_latent(query_hidden, state)
    direction = model.queries.direction(zt)
    direction = direction / jnp.maximum(jnp.linalg.norm(direction, axis=-1, keepdims=True), 1e-8)
    codes = model.codebook.value
    codes = codes / jnp.maximum(jnp.linalg.norm(codes, axis=-1, keepdims=True), 1e-8)
    return direction, direction @ codes.T


def _temporal_class(delta: np.ndarray, labels: tuple[int, int]) -> tuple[str, dict]:
    """Classify clear sequential/simultaneous cases from cumulative TCP deltas."""

    # Labels are (+x,-x,+y,-y,+z,-z,+rx,-rx,+ry,-ry,+rz,-rz).
    increments = np.diff(
        np.concatenate([np.zeros((1, delta.shape[-1]), dtype=delta.dtype), delta], axis=0),
        axis=0,
    )
    profiles = []
    centroids = []
    for label in labels:
        axis = label // 2
        sign = 1.0 if label % 2 == 0 else -1.0
        profile = np.maximum(sign * increments[:, axis], 0.0)
        mass = float(profile.sum())
        if mass <= 1e-8:
            return "unclear", {}
        profile = profile / mass
        profiles.append(profile)
        centroids.append(float(np.sum(profile * np.linspace(0.0, 1.0, len(profile)))))

    overlap = float(np.minimum(profiles[0], profiles[1]).sum())
    gap = centroids[1] - centroids[0]
    if abs(gap) <= 0.10 and overlap >= 0.25:
        kind = "simultaneous"
    elif gap >= 0.18:
        kind = f"{labels[0]}_then_{labels[1]}"
    elif gap <= -0.18:
        kind = f"{labels[1]}_then_{labels[0]}"
    else:
        kind = "unclear"
    return kind, {
        "centroid": [round(value, 4) for value in centroids],
        "centroid_gap": round(gap, 4),
        "profile_overlap": round(overlap, 4),
    }


def _centroid(rows: list[dict]) -> np.ndarray:
    value = np.mean(np.stack([row["direction"] for row in rows]), axis=0)
    return value / max(float(np.linalg.norm(value)), 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", action="append", required=True)
    parser.add_argument("--norm-assets-dir", default="/mnt/cunchu/zjq/target")
    parser.add_argument("--norm-asset-id", default="openpi_norm_compact_accepted_v3")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = AtomicPi05Config(max_token_len=250, fast_action_ce_loss_weight=0.0)
    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    loader = build_atomic_text_loader(
        tuple(args.dataset_root),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=20260801,
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        include_fast=False,
    )

    examples: list[dict] = []
    iterator = iter(loader)
    for _ in range(args.num_batches):
        batch = next(iterator)
        weights = np.asarray(batch["atomic_weights"], dtype=np.float32)
        supervised = np.asarray(batch["atomic_supervision_mask"], dtype=bool)
        dual = supervised & (np.sum(weights > 0, axis=-1) == 2)
        if not np.any(dual):
            continue
        observation = jax.tree.map(jnp.asarray, text_batch_to_observation(batch))
        direction, similarity = jax.device_get(_encode(model, observation))
        deltas = np.asarray(batch["tcp_twist_delta"], dtype=np.float32)
        for index in np.flatnonzero(dual):
            labels = tuple(int(value) for value in np.flatnonzero(weights[index] > 0))
            kind, temporal = _temporal_class(deltas[index], labels)
            if kind == "unclear":
                continue
            top = np.argsort(similarity[index])[::-1][:4]
            examples.append(
                {
                    "pair": list(labels),
                    "pair_names": [ATOMIC_NAMES[value] for value in labels],
                    "target_weights": [round(float(weights[index, value]), 4) for value in labels],
                    "temporal_class": kind,
                    **temporal,
                    "direction": np.asarray(direction[index], dtype=np.float32),
                    "top_code_similarity": [
                        [ATOMIC_NAMES[value], round(float(similarity[index, value]), 4)]
                        for value in top
                    ],
                }
            )

    groups: dict[tuple[tuple[int, int], str], list[dict]] = defaultdict(list)
    for row in examples:
        groups[(tuple(row["pair"]), row["temporal_class"])].append(row)

    comparisons = []
    by_pair: dict[tuple[int, int], list[tuple[str, list[dict]]]] = defaultdict(list)
    for (pair, kind), rows in groups.items():
        if len(rows) >= 2:
            by_pair[pair].append((kind, rows))
    for pair, categories in by_pair.items():
        for i, (kind_a, rows_a) in enumerate(categories):
            for kind_b, rows_b in categories[i + 1 :]:
                ca, cb = _centroid(rows_a), _centroid(rows_b)
                comparisons.append(
                    {
                        "pair": list(pair),
                        "pair_names": [ATOMIC_NAMES[value] for value in pair],
                        "class_a": kind_a,
                        "count_a": len(rows_a),
                        "class_b": kind_b,
                        "count_b": len(rows_b),
                        "zt_centroid_cosine": round(float(ca @ cb), 6),
                        "zt_centroid_angle_deg": round(
                            float(np.degrees(np.arccos(np.clip(ca @ cb, -1.0, 1.0)))), 4
                        ),
                    }
                )

    # Keep a few concrete rows from the most useful groups.
    printable = []
    for key in sorted(groups, key=lambda item: (-len(groups[item]), item))[:8]:
        printable.extend(groups[key][:2])
    printable = printable[: args.examples]
    for row in printable:
        row["direction"] = [round(float(value), 6) for value in row["direction"][:12]]
        row["direction_note"] = "first 12 of 512 dimensions"

    report = {
        "checkpoint": str(args.checkpoint),
        "scanned_samples": args.batch_size * args.num_batches,
        "clear_dual_samples": len(examples),
        "group_counts": {
            f"{ATOMIC_NAMES[pair[0]]}+{ATOMIC_NAMES[pair[1]]}/{kind}": len(rows)
            for (pair, kind), rows in sorted(groups.items())
        },
        "same_pair_different_timing": comparisons,
        "examples": printable,
        "interpretation": (
            "Atomic ratio supervision discards order. Any order sensitivity in zT can only "
            "come from prompt/state and the Q1-only DCT trajectory objective."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
