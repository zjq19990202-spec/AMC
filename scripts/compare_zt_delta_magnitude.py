#!/usr/bin/env python3
"""Compare zT on identical atomic prompts/labels but different FK amplitudes."""

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
    atomic_text_collate,
    build_atomic_text_dataset,
    text_batch_to_observation,
)


@nnx.jit
def _encode(model, observation):
    query_hidden, _, _ = model._prefix_forward(observation)  # noqa: SLF001
    state = model._controlled_state(observation.state)  # noqa: SLF001
    raw_zt = model.queries.text_latent(query_hidden, state)
    direction = model.queries.direction(raw_zt)
    direction = direction / jnp.maximum(jnp.linalg.norm(direction, axis=-1, keepdims=True), 1e-8)
    codes = model.codebook.value
    codes = codes / jnp.maximum(jnp.linalg.norm(codes, axis=-1, keepdims=True), 1e-8)
    return raw_zt, direction, direction @ codes.T


def _labels(row: dict) -> tuple[int, ...]:
    if not bool(row["atomic_supervision_mask"]):
        return ()
    return tuple(int(value) for value in np.flatnonzero(np.asarray(row["atomic_weights"]) > 0))


def _amplitude(row: dict, labels: tuple[int, ...]) -> tuple[float, list[float]]:
    delta = np.asarray(row["tcp_twist_delta"], dtype=np.float64)
    increments = np.diff(
        np.concatenate([np.zeros((1, delta.shape[-1])), delta], axis=0), axis=0
    )
    masses = []
    scaled = []
    for label in labels:
        axis = label // 2
        sign = 1.0 if label % 2 == 0 else -1.0
        mass = float(np.maximum(sign * increments[:, axis], 0.0).sum())
        masses.append(mass)
        scaled.append(mass / (0.02 if axis < 3 else 0.15))
    return float(np.linalg.norm(scaled)), masses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", action="append", required=True)
    parser.add_argument("--norm-assets-dir", default="/mnt/cunchu/zjq/target")
    parser.add_argument("--norm-asset-id", default="openpi_norm_compact_accepted_v3")
    parser.add_argument("--scan-samples", type=int, default=12000)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--min-low-amplitude", type=float, default=0.25)
    parser.add_argument("--min-amplitude-ratio", type=float, default=1.5)
    parser.add_argument("--max-amplitude-ratio", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = AtomicPi05Config(max_token_len=250, fast_action_ce_loss_weight=0.0)
    dataset = build_atomic_text_dataset(
        tuple(args.dataset_root),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        include_fast=False,
    )
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=min(args.scan_samples, len(dataset)), replace=False)

    grouped: dict[tuple[str, tuple[int, ...]], list[tuple[int, dict, float, list[float]]]] = (
        defaultdict(list)
    )
    for index in indices:
        row = dataset[int(index)]
        labels = _labels(row)
        if not labels:
            continue
        amplitude, masses = _amplitude(row, labels)
        if amplitude > 1e-6:
            grouped[(str(row["atomic_prompt"]), labels)].append(
                (int(index), row, amplitude, masses)
            )

    candidates = []
    for (prompt, labels), rows in grouped.items():
        if len(rows) < 2:
            continue
        rows.sort(key=lambda item: item[2])
        # Prefer equal/near-equal atomic mixtures so amplitude is the main change.
        best = None
        for low in rows[: min(5, len(rows))]:
            for high in rows[-min(5, len(rows)) :]:
                low_w = np.asarray(low[1]["atomic_weights"])[list(labels)]
                high_w = np.asarray(high[1]["atomic_weights"])[list(labels)]
                ratio_delta = float(np.abs(low_w - high_w).sum())
                amplitude_ratio = high[2] / max(low[2], 1e-8)
                if (
                    low[2] < args.min_low_amplitude
                    or amplitude_ratio < args.min_amplitude_ratio
                    or amplitude_ratio > args.max_amplitude_ratio
                    or ratio_delta > 0.10
                ):
                    continue
                score = np.log(amplitude_ratio) - ratio_delta
                if best is None or score > best[0]:
                    best = (score, low, high, ratio_delta)
        if best is not None:
            candidates.append((best[0], prompt, labels, best[1], best[2], best[3]))
    candidates.sort(reverse=True, key=lambda item: item[0])
    selected = candidates[: args.examples]
    if not selected:
        raise RuntimeError("no same-prompt/same-label amplitude pairs found")

    rows_for_model = []
    for _, _, _, low, high, _ in selected:
        rows_for_model.extend([low[1], high[1]])
    batch = atomic_text_collate(rows_for_model)

    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    observation = jax.tree.map(jnp.asarray, text_batch_to_observation(batch))
    raw_zt, direction, similarity = map(np.asarray, jax.device_get(_encode(model, observation)))

    output = []
    for pair_index, (_, prompt, labels, low, high, ratio_delta) in enumerate(selected):
        low_i, high_i = 2 * pair_index, 2 * pair_index + 1
        cosine = float(direction[low_i] @ direction[high_i])
        top_low = np.argsort(similarity[low_i])[::-1][:3]
        top_high = np.argsort(similarity[high_i])[::-1][:3]
        output.append(
            {
                "prompt": prompt,
                "labels": [ATOMIC_NAMES[label] for label in labels],
                "atomic_weight_l1_difference": round(ratio_delta, 5),
                "low": {
                    "scaled_amplitude": round(low[2], 5),
                    "per_atom_physical_path": [round(value, 6) for value in low[3]],
                    "target_weights": [
                        round(float(np.asarray(low[1]["atomic_weights"])[label]), 5)
                        for label in labels
                    ],
                    "zt_raw_norm": round(float(np.linalg.norm(raw_zt[low_i])), 5),
                    "top_codes": [
                        [ATOMIC_NAMES[label], round(float(similarity[low_i, label]), 5)]
                        for label in top_low
                    ],
                },
                "high": {
                    "scaled_amplitude": round(high[2], 5),
                    "per_atom_physical_path": [round(value, 6) for value in high[3]],
                    "target_weights": [
                        round(float(np.asarray(high[1]["atomic_weights"])[label]), 5)
                        for label in labels
                    ],
                    "zt_raw_norm": round(float(np.linalg.norm(raw_zt[high_i])), 5),
                    "top_codes": [
                        [ATOMIC_NAMES[label], round(float(similarity[high_i, label]), 5)]
                        for label in top_high
                    ],
                },
                "amplitude_ratio": round(high[2] / low[2], 5),
                "zt_direction_cosine": round(cosine, 6),
                "zt_direction_angle_deg": round(
                    float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))), 5
                ),
            }
        )

    report = {
        "checkpoint": str(args.checkpoint),
        "scanned_samples": len(indices),
        "control": "exact same prompt text and same one/two atomic labels",
        "amplitude_definition": (
            "positive signed path length on active FK axes, translation/0.02 m and "
            "rotation/0.15 rad before joint L2 norm"
        ),
        "examples": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
