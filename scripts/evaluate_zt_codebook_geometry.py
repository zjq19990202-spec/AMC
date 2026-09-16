#!/usr/bin/env python3
"""Measure zT Q1 class compactness and alignment to a frozen atomic codebook."""

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
from atomic_latent_vla.pi05.model import l2_normalize
from atomic_latent_vla.pi05.training_data import (
    atomic_text_collate,
    build_atomic_text_dataset,
    text_batch_to_observation,
)


ARM_NAMES = ("right", "left")


@nnx.jit
def _encode(model, observation):
    active_state = model._controlled_state(observation.state)  # noqa: SLF001
    query_hidden, _, _, _ = model._prefix_forward(observation)  # noqa: SLF001
    arm_latents = model.queries.text_arm_latents(query_hidden, active_state)
    directions = l2_normalize(model.queries.direction(arm_latents))
    codes = l2_normalize(model.codebook.value)
    return directions, codes


def _angle(cosine: np.ndarray | float) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _single_label(row: dict, arm: int) -> int | None:
    if not bool(np.asarray(row["atomic_supervision_mask"])[arm]):
        return None
    weights = np.asarray(row["atomic_weights"])[arm]
    labels = np.flatnonzero(weights > 1.0e-6)
    if len(labels) != 1:
        return None
    return int(labels[0])


def _select(dataset, per_class: int, scan_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    selected: dict[tuple[int, int], list[int]] = defaultdict(list)
    permutation = rng.permutation(len(dataset))[: min(scan_samples, len(dataset))]
    for raw_index in permutation:
        index = int(raw_index)
        row = dataset[index]
        for arm in range(2):
            label = _single_label(row, arm)
            if label is not None and len(selected[(arm, label)]) < per_class:
                selected[(arm, label)].append(index)
        if all(len(selected[(arm, label)]) >= per_class for arm in range(2) for label in range(13)):
            break
    # One row can supervise both arms. Encode it once, then fan it back out.
    unique_indices = sorted({index for values in selected.values() for index in values})
    return selected, unique_indices


def _arm_report(vectors: np.ndarray, labels: np.ndarray, codes: np.ndarray) -> dict:
    similarities = vectors @ codes.T
    target_cosine = similarities[np.arange(len(labels)), labels]
    masked = similarities.copy()
    masked[np.arange(len(labels)), labels] = -np.inf
    wrong_cosine = masked.max(axis=1)
    prediction = similarities.argmax(axis=1)

    target_angle = _angle(target_cosine)
    wrong_angle = _angle(wrong_cosine)
    angular_margin = wrong_angle - target_angle

    per_class = {}
    centroids = []
    centroid_labels = []
    all_within_centroid = []
    all_pairwise = []
    all_centroid_to_code = []
    rng = np.random.default_rng(20260816)
    for label in sorted(set(labels.tolist())):
        hit = labels == label
        rows = vectors[hit]
        centroid = rows.mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1.0e-8)
        within_centroid = _angle(rows @ centroid)
        # Exact all-pairs is small at the default quota, but cap it for larger runs.
        first, second = np.triu_indices(len(rows), k=1)
        if len(first) > 20000:
            keep = rng.choice(len(first), size=20000, replace=False)
            first, second = first[keep], second[keep]
        pairwise = _angle(np.sum(rows[first] * rows[second], axis=1))
        centroid_to_code = float(_angle(centroid @ codes[label]))
        class_target = target_angle[hit]
        class_margin = angular_margin[hit]
        per_class[ATOMIC_NAMES[label]] = {
            "count": int(hit.sum()),
            "q1_to_target_code_deg": _stats(class_target),
            "q1_to_class_centroid_deg": _stats(within_centroid),
            "same_class_pairwise_deg": _stats(pairwise),
            "class_centroid_to_code_deg": centroid_to_code,
            "target_top1_accuracy": float(np.mean(prediction[hit] == label)),
            "angular_margin_to_nearest_wrong_code_deg": _stats(class_margin),
        }
        centroids.append(centroid)
        centroid_labels.append(label)
        all_within_centroid.append(within_centroid)
        all_pairwise.append(pairwise)
        all_centroid_to_code.append(centroid_to_code)

    centroids = np.stack(centroids)
    centroid_labels = np.asarray(centroid_labels)
    centroid_similarity = vectors @ centroids.T
    centroid_prediction = centroid_labels[centroid_similarity.argmax(axis=1)]

    code_pairs = codes @ codes.T
    upper = code_pairs[np.triu_indices(len(codes), k=1)]
    return {
        "sample_count": int(len(labels)),
        "class_count": int(len(per_class)),
        "target_code_top1_accuracy": float(np.mean(prediction == labels)),
        "class_centroid_top1_accuracy": float(np.mean(centroid_prediction == labels)),
        "q1_to_target_code_deg": _stats(target_angle),
        "q1_to_nearest_wrong_code_deg": _stats(wrong_angle),
        "angular_margin_to_nearest_wrong_code_deg": _stats(angular_margin),
        "q1_to_class_centroid_deg": _stats(np.concatenate(all_within_centroid)),
        "same_class_pairwise_deg": _stats(np.concatenate(all_pairwise)),
        "class_centroid_to_code_deg": _stats(np.asarray(all_centroid_to_code)),
        "codebook_pairwise_angle_deg": _stats(_angle(upper)),
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", default="/mnt/cunchu/zjq/target")
    parser.add_argument("--norm-asset-id", default="openpi_norm_compact_accepted_v3")
    parser.add_argument("--per-class", type=int, default=64)
    parser.add_argument("--scan-samples", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = AtomicPi05Config(max_token_len=250, fast_action_ce_loss_weight=0.0)
    dataset = build_atomic_text_dataset(
        (args.dataset_root,),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        include_fast=False,
        reliable_atomic_only=True,
    )
    selected, unique_indices = _select(dataset, args.per_class, args.scan_samples, args.seed)
    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()

    encoded: dict[int, np.ndarray] = {}
    codes_np = None
    for start in range(0, len(unique_indices), args.batch_size):
        indices = unique_indices[start : start + args.batch_size]
        rows = [dataset[index] for index in indices]
        batch = atomic_text_collate(rows)
        observation = jax.tree.map(jnp.asarray, text_batch_to_observation(batch))
        directions, codes = jax.device_get(_encode(model, observation))
        directions = np.asarray(directions, dtype=np.float32)
        codes_np = np.asarray(codes, dtype=np.float32)
        for index, direction in zip(indices, directions, strict=True):
            encoded[index] = direction
        print(f"encoded {min(start + args.batch_size, len(unique_indices))}/{len(unique_indices)}", flush=True)
    assert codes_np is not None

    report = {
        "checkpoint": str(args.checkpoint),
        "selection": {
            "per_class_requested": args.per_class,
            "unique_rows": len(unique_indices),
            "counts": {
                ARM_NAMES[arm]: {
                    ATOMIC_NAMES[label]: len(selected[(arm, label)]) for label in range(13)
                }
                for arm in range(2)
            },
        },
        "arms": {},
    }
    for arm in range(2):
        vectors, labels = [], []
        for label in range(13):
            for index in selected[(arm, label)]:
                vectors.append(encoded[index][arm])
                labels.append(label)
        report["arms"][ARM_NAMES[arm]] = _arm_report(
            np.asarray(vectors), np.asarray(labels), codes_np[arm]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
