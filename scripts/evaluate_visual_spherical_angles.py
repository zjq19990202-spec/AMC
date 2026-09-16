#!/usr/bin/env python3
"""Measure the training-faithful final-layer visual zT->zM sphere angle."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from torch.utils.data import DataLoader, Subset

from openpi.models import model as openpi_model

from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import (
    atomic_collate,
    batch_to_observation,
    build_atomic_dataset,
)


@nnx.jit
def _encode_angles(model, observation, prompt_tokens, prompt_masks):
    observation = openpi_model.preprocess_observation(None, observation, train=False)
    observation = model._with_prompt(observation, prompt_tokens, prompt_masks)  # noqa: SLF001
    prefix_output = model._prefix_forward(observation)  # noqa: SLF001
    query_hidden = prefix_output[0]
    active_state = model._controlled_state(observation.state)  # noqa: SLF001
    _, right_direction, z_model, left_direction, _ = model._latent(  # noqa: SLF001
        query_hidden, active_state
    )
    # The checkpoint runs in bfloat16, but angular diagnostics require a
    # float32 dot product; otherwise arccos is visibly quantized in ~0.25 deg
    # increments near the observed 38-degree operating point.
    directions = jnp.stack([right_direction, left_direction], axis=1).astype(jnp.float32)
    z_model = z_model.astype(jnp.float32)
    directions = directions / jnp.maximum(
        jnp.linalg.norm(directions, axis=-1, keepdims=True), 1.0e-8
    )
    z_model = z_model / jnp.maximum(jnp.linalg.norm(z_model, axis=-1, keepdims=True), 1.0e-8)
    cosine = jnp.clip(jnp.sum(directions * z_model, axis=-1), -1.0, 1.0)
    angles = jnp.rad2deg(jnp.arccos(cosine))
    return angles, cosine, jnp.linalg.norm(z_model, axis=-1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_deg": float(array.mean()),
        "std_deg": float(array.std()),
        "median_deg": float(np.median(array)),
        "p90_deg": float(np.quantile(array, 0.90)),
        "p95_deg": float(np.quantile(array, 0.95)),
        "p99_deg": float(np.quantile(array, 0.99)),
        "max_deg": float(array.max()),
        "fraction_ge_30deg": float(np.mean(array >= 30.0)),
        "fraction_ge_35deg": float(np.mean(array >= 35.0)),
        "fraction_ge_38deg": float(np.mean(array >= 38.0)),
    }


def _category(weights: np.ndarray, supervised: bool) -> str:
    if not supervised:
        return "drop"
    active = np.flatnonzero(weights > 0)
    if active.size == 1 and int(active[0]) == 12:
        return "stay"
    if active.size == 1:
        return "single"
    if active.size == 2:
        return "dual"
    return f"multi_{active.size}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, action="append", required=True)
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--composition-sidecar", default="fk_horizon_3hz_gate_top5_stay_v2")
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    config = AtomicPi05Config(
        max_token_len=192,
        coefficient_target_kind="joint_delta",
        coefficient_target_dim=14,
        enable_layerwise_atomic_flow=True,
        spherical_visual_latent=True,
        visual_max_update_angle_deg=45.0,
        atomic_loss_weight=0.05,
        atomic_composition_loss_weight=0.015,
        freeze_vision_encoder=False,
        fast_action_ce_loss_weight=0.0,
        subtask_ce_loss_weight=0.0,
    )
    dataset = build_atomic_dataset(
        args.dataset_root,
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        include_fast=False,
        atomic_composition_sidecar=args.composition_sidecar,
        pad_subtask_horizon=True,
    )
    if args.sample_count > len(dataset):
        raise ValueError(f"sample-count {args.sample_count} exceeds dataset size {len(dataset)}")
    rng = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(len(dataset), size=args.sample_count, replace=False))

    # Fork video-decoding workers before JAX initializes its GPU runtime. This
    # is the same ordering required by the training loader.
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False,
        collate_fn=atomic_collate,
    )
    data_iter = iter(loader)

    params = openpi_model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()

    grouped: dict[str, list[float]] = defaultdict(list)
    selections = []
    cumulative = list(getattr(dataset, "cumulative_sizes", [len(dataset)]))
    roots = [str(path) for path in args.dataset_root]
    norm_path = args.norm_assets_dir / args.norm_asset_id / "norm_stats.json"

    for start, batch in zip(range(0, len(indices), args.batch_size), data_iter, strict=True):
        selected = indices[start : start + args.batch_size].tolist()
        real_count = len(selected)
        observation_np, _ = batch_to_observation(batch)
        observation = jax.tree.map(jnp.asarray, observation_np)
        angles, _, norms = jax.device_get(
            _encode_angles(
                model,
                observation,
                jnp.asarray(batch["subtask_prompt_tokens"]),
                jnp.asarray(batch["subtask_prompt_mask"]),
            )
        )
        weights = np.asarray(batch["atomic_weights"])
        supervision = np.asarray(batch["atomic_supervision_mask"])
        for local, dataset_index in enumerate(selected[:real_count]):
            source_index = bisect.bisect_right(cumulative, dataset_index)
            source_start = 0 if source_index == 0 else cumulative[source_index - 1]
            source_dataset = (
                dataset.datasets[source_index]
                if hasattr(dataset, "datasets")
                else dataset
            )
            metadata = source_dataset._raw.metadata(int(dataset_index - source_start))  # noqa: SLF001
            record = {
                "dataset_index": int(dataset_index),
                "dataset_root": roots[source_index],
                "episode": int(metadata.get("episode_index", metadata.get("episode", -1))),
                "frame": int(metadata.get("frame_index", metadata.get("frame", -1))),
                "subtask_prompt": str(metadata.get("subtask_prompt", "")),
                "arms": {},
            }
            for arm_index, arm in enumerate(("right", "left")):
                category = _category(weights[local, arm_index], bool(supervision[local, arm_index]))
                angle = float(angles[local, arm_index])
                grouped["both/all"].append(angle)
                grouped[f"{arm}/all"].append(angle)
                grouped[f"both/{category}"].append(angle)
                grouped[f"{arm}/{category}"].append(angle)
                record["arms"][arm] = {
                    "category": category,
                    "angle_deg": angle,
                    "zm_norm": float(norms[local, arm_index]),
                }
            selections.append(record)
        print(f"encoded {min(start + args.batch_size, len(indices))}/{len(indices)}", flush=True)

    summary = {name: _stats(values) for name, values in sorted(grouped.items())}
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(args.checkpoint.name),
        "dataset_roots": roots,
        "dataset_length": len(dataset),
        "sample_count": len(indices),
        "seed": args.seed,
        "prompt_contract": "100% native SUBtask prompt; padded single-subtask 50-step horizon",
        "norm_assets_dir": str(args.norm_assets_dir),
        "norm_asset_id": args.norm_asset_id,
        "norm_stats_sha256": _sha256(norm_path),
        "metric": "final-layer geodesic angle in degrees between unit zT direction and spherical visual zM",
        "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "selection.json").write_text(
        json.dumps(selections, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "run_contract.md").write_text(
        "\n".join(
            [
                "# Visual spherical-angle evaluation contract",
                "",
                f"- checkpoint: `{args.checkpoint}`",
                f"- step: `{args.checkpoint.name}`",
                f"- dataset roots: `{len(roots)}` union2375 roots",
                f"- samples: `{len(indices)}` random rows without replacement, seed `{args.seed}`",
                "- prompt: `100% native SUBtask`, matching the active spherical-ZM run",
                "- horizon: `50`, padded within one subtask",
                f"- norm: `{args.norm_assets_dir / args.norm_asset_id}`",
                f"- norm_stats sha256: `{_sha256(norm_path)}`",
                "- architecture: layerwise visual spherical zM, 4-D tangent basis, 45-degree cap",
                "- reported metric: final-layer angle(zT, zM); intermediate-layer angles are not included",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
