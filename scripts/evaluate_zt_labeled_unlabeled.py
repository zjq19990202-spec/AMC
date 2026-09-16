#!/usr/bin/env python3
"""Compare zT coefficient-DiT behavior on atomic and unlabeled horizons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from openpi.models import model as _model

from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.model import dct_prefix, l2_normalize, recover_flow_endpoint
from atomic_latent_vla.pi05.training_data import (
    atomic_text_collate,
    build_atomic_text_dataset,
    text_batch_to_observation,
)


@nnx.jit
def _evaluate(model, observation, tcp_twist_delta, rng, global_step):
    active_state = model._controlled_state(observation.state)  # noqa: SLF001
    query_hidden, _, _, _ = model._prefix_forward(observation)  # noqa: SLF001
    arm_latents = model.queries.text_arm_latents(query_hidden, active_state)
    directions = l2_normalize(model.queries.direction(arm_latents))
    fused = model._fuse_text_arm_latents(directions)  # noqa: SLF001

    noise_rng, time_rng = jax.random.split(rng)
    target = dct_prefix(tcp_twist_delta, model.config.coefficient_count)
    noise = jax.random.normal(noise_rng, target.shape)
    time = jax.random.uniform(time_rng, target.shape[:-2], minval=0.001, maxval=1.0)
    noisy = time[:, None, None] * noise + (1.0 - time[:, None, None]) * target
    velocity = model._coefficient_velocity(noisy, time, fused)  # noqa: SLF001
    recovered = recover_flow_endpoint(noisy, time, velocity)
    squared_error = jnp.square(recovered - target)

    codes = l2_normalize(model.codebook.value)
    similarities = jnp.einsum("bad,aed->bae", directions, codes)
    ordered = jnp.sort(similarities, axis=-1)
    step = jnp.asarray(global_step, squared_error.dtype)
    warmup = jnp.asarray(model.config.coefficient_velocity_warmup_steps, squared_error.dtype)
    transition = jnp.asarray(
        max(model.config.coefficient_wall_transition_steps, 1), squared_error.dtype
    )
    return {
        "sample_loss": jnp.mean(squared_error, axis=(1, 2)),
        "arm_loss": jnp.stack(
            [
                jnp.mean(squared_error[..., :6], axis=(1, 2)),
                jnp.mean(squared_error[..., 6:12], axis=(1, 2)),
            ],
            axis=1,
        ),
        "direction_norm": jnp.linalg.norm(directions, axis=-1),
        "top1_cosine": ordered[..., -1],
        "top1_margin": ordered[..., -1] - ordered[..., -2],
        "wall_weight": jnp.clip((step - warmup) / transition, 0.0, 1.0),
    }


def _summary(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    selected = values[mask]
    if selected.size == 0:
        return {"count": 0}
    return {
        "count": int(selected.size),
        "mean": float(selected.mean()),
        "median": float(np.median(selected)),
        "p90": float(np.quantile(selected, 0.9)),
    }


def _select(dataset, per_group: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    labeled: list[int] = []
    unlabeled: list[int] = []
    for index in rng.permutation(len(dataset)):
        row = dataset[int(index)]
        mask = np.asarray(row["atomic_supervision_mask"], dtype=bool)
        target = labeled if np.any(mask) else unlabeled
        if len(target) < per_group:
            target.append(int(index))
        if len(labeled) == per_group and len(unlabeled) == per_group:
            break
    if len(labeled) != per_group or len(unlabeled) != per_group:
        raise RuntimeError(
            f"could not select balanced groups: labeled={len(labeled)} unlabeled={len(unlabeled)}"
        )
    interleaved = np.stack([labeled, unlabeled], axis=1).reshape(-1)
    return interleaved.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", default="/mnt/cunchu/zjq/target")
    parser.add_argument("--norm-asset-id", default="openpi_norm_compact_accepted_v3")
    parser.add_argument("--tcp-twist-norm", type=Path)
    parser.add_argument("--samples-per-group", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--noise-repeats", type=int, default=2)
    parser.add_argument("--global-step", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (2 * args.samples_per_group) % args.batch_size:
        raise ValueError("2*samples-per-group must be divisible by batch-size")

    config = AtomicPi05Config(max_token_len=250, fast_action_ce_loss_weight=0.0)
    tcp_norm_path = args.tcp_twist_norm or (
        args.dataset_root / "meta" / "tcp_twist_norm_bimanual_tcp200.json"
    )
    norm_payload = json.loads(tcp_norm_path.read_text(encoding="utf-8"))
    norm_stats = norm_payload["norm_stats"]["tcp_twist_delta"]
    tcp_range = np.asarray(norm_stats["q99"], dtype=np.float32) - np.asarray(
        norm_stats["q01"], dtype=np.float32
    )

    dataset = build_atomic_text_dataset(
        (args.dataset_root,),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        include_fast=False,
    )
    selected = _select(dataset, args.samples_per_group, args.seed)
    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()

    collected: dict[str, list[np.ndarray]] = {}
    supervision: list[np.ndarray] = []
    for start in range(0, len(selected), args.batch_size):
        rows = [dataset[index] for index in selected[start : start + args.batch_size]]
        batch = atomic_text_collate(rows)
        observation = jax.tree.map(jnp.asarray, text_batch_to_observation(batch))
        normalized_tcp_delta = (
            2.0
            * np.asarray(batch["tcp_twist_delta"], dtype=np.float32)
            / tcp_range[None, None, :]
        )
        repeated = [
            jax.device_get(
                _evaluate(
                    model,
                    observation,
                    jnp.asarray(normalized_tcp_delta),
                    jax.random.key(args.seed + start * 100 + repeat),
                    jnp.asarray(args.global_step),
                )
            )
            for repeat in range(args.noise_repeats)
        ]
        for key in repeated[0]:
            values = np.stack([np.asarray(item[key]) for item in repeated])
            collected.setdefault(key, []).append(np.atleast_1d(values.mean(axis=0)))
        supervision.append(np.asarray(batch["atomic_supervision_mask"], dtype=bool))
        print(f"evaluated {start + args.batch_size}/{len(selected)}", flush=True)

    values = {key: np.concatenate(parts, axis=0) for key, parts in collected.items()}
    arm_mask = np.concatenate(supervision, axis=0)
    sample_mask = np.any(arm_mask, axis=1)
    sample_loss = values["sample_loss"]
    arm_loss = values["arm_loss"]
    total_error = float(sample_loss.sum())
    unlabeled_error = float(sample_loss[~sample_mask].sum())
    report = {
        "checkpoint": str(args.checkpoint),
        "samples": int(len(sample_mask)),
        "noise_repeats": args.noise_repeats,
        "wall_weight": float(values["wall_weight"].mean()),
        "sample_dct_loss": {
            "labeled": _summary(sample_loss, sample_mask),
            "unlabeled": _summary(sample_loss, ~sample_mask),
            "unlabeled_over_labeled_mean": float(
                sample_loss[~sample_mask].mean() / max(sample_loss[sample_mask].mean(), 1e-12)
            ),
            "unlabeled_error_mass_fraction_balanced_sample": float(
                unlabeled_error / max(total_error, 1e-12)
            ),
        },
        "arm_dct_loss": {
            "labeled": _summary(arm_loss, arm_mask),
            "unlabeled": _summary(arm_loss, ~arm_mask),
        },
        "direction_norm": {
            "labeled": _summary(values["direction_norm"], arm_mask),
            "unlabeled": _summary(values["direction_norm"], ~arm_mask),
            "max_abs_deviation_from_one": float(
                np.max(np.abs(values["direction_norm"] - 1.0))
            ),
        },
        "nearest_code_cosine": {
            "labeled": _summary(values["top1_cosine"], arm_mask),
            "unlabeled": _summary(values["top1_cosine"], ~arm_mask),
        },
        "nearest_code_margin": {
            "labeled": _summary(values["top1_margin"], arm_mask),
            "unlabeled": _summary(values["top1_margin"], ~arm_mask),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
