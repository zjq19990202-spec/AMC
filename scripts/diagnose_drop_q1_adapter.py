#!/usr/bin/env python3
"""Measure unsupervised/drop Q1 geometry and the effective zM adapter output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from openpi.models import model as _model

from atomic_latent_vla.atomic import ATOMIC_NAMES
from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import batch_to_observation, build_atomic_loader


def _text_observation(
    observation: _model.Observation,
    prompt_tokens: jax.Array,
    prompt_mask: jax.Array,
) -> _model.Observation:
    return _model.Observation(
        images={},
        image_masks={},
        state=observation.state,
        tokenized_prompt=prompt_tokens,
        tokenized_prompt_mask=prompt_mask,
        token_ar_mask=None,
        token_loss_mask=None,
    )


@nnx.jit
def _evaluate(model, observation, text_observation, actions, rng):
    preprocess_full, noise_rng = jax.random.split(rng)
    observation = _model.preprocess_observation(preprocess_full, observation, train=False)
    # The joint trainer's zT branch intentionally consumes the loader's
    # already-normalized text/state observation without generic preprocessing;
    # that generic path requires all three camera keys.
    actions = model._mask_action_condition(actions)  # noqa: SLF001
    state = model._controlled_state(observation.state)  # noqa: SLF001

    query_hidden, prefix_mask, kv_cache = model._prefix_forward(observation)  # noqa: SLF001
    _, full_direction, z_model, _, _ = model._latent(query_hidden, state)  # noqa: SLF001

    text_query_hidden, _, _ = model._prefix_forward(text_observation)  # noqa: SLF001
    _, text_direction, _, _, _ = model._latent(text_query_hidden, state)  # noqa: SLF001

    noise = model._mask_action_condition(  # noqa: SLF001
        jax.random.normal(noise_rng, actions.shape)
    )
    time = jnp.full(actions.shape[0], 0.5, dtype=actions.dtype)
    noisy = time[:, None, None] * noise + (1.0 - time[:, None, None]) * actions
    target = noise - actions

    velocity = model._suffix_velocity(  # noqa: SLF001
        prefix_mask, kv_cache, noisy, time, z_model
    )
    parallel = jnp.sum(z_model * full_direction, axis=-1)
    z_model_no_q3 = parallel[:, None] * full_direction
    velocity_no_q3 = model._suffix_velocity(  # noqa: SLF001
        prefix_mask, kv_cache, noisy, time, z_model_no_q3
    )
    # None bypasses every per-block atomic adapter, while preserving the
    # released attention, FFN and time-AdaRMS computation.
    velocity_no_adapter = model._suffix_velocity(  # noqa: SLF001
        prefix_mask, kv_cache, noisy, time, None
    )
    velocity_shuffled = model._suffix_velocity(  # noqa: SLF001
        prefix_mask, kv_cache, noisy, time, jnp.roll(z_model, 1, axis=0)
    )

    codes = model.codebook.value
    codes = codes / jnp.maximum(jnp.linalg.norm(codes, axis=-1, keepdims=True), 1e-8)
    full_similarity = full_direction @ codes.T
    text_similarity = text_direction @ codes.T
    detail = z_model - parallel[:, None] * full_direction
    controlled = slice(
        model.config.controlled_action_start,
        model.config.controlled_action_start + model.config.controlled_action_dim,
    )
    return {
        "full_direction": full_direction,
        "text_direction": text_direction,
        "full_similarity": full_similarity,
        "text_similarity": text_similarity,
        "z_model": z_model,
        "z_parallel": parallel,
        "z_detail_norm": jnp.linalg.norm(detail, axis=-1),
        "velocity": velocity[..., controlled],
        "velocity_no_q3": velocity_no_q3[..., controlled],
        "velocity_no_adapter": velocity_no_adapter[..., controlled],
        "velocity_shuffled": velocity_shuffled[..., controlled],
        "target": target[..., controlled],
    }


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value))))


def _atomic_condition_stats(params, z_model: np.ndarray) -> dict[str, object]:
    leaves = {
        jax.tree_util.keystr(path): np.asarray(value, dtype=np.float32)
        for path, value in jax.tree_util.tree_flatten_with_path(params)[0]
        if "atomic_condition" in jax.tree_util.keystr(path)
    }
    def leaf(fragment: str) -> np.ndarray:
        matches = [value for path, value in leaves.items() if fragment in path]
        if len(matches) != 1:
            raise KeyError(f"expected one parameter containing {fragment!r}, got {len(matches)}")
        return matches[0]

    kernel_in = leaf("atomic_condition_in']['kernel")
    bias_in = leaf("atomic_condition_in']['bias")
    kernel_out = leaf("atomic_condition_out']['kernel")
    bias_out = leaf("atomic_condition_out']['bias")
    z = np.asarray(z_model, dtype=np.float32)
    pre = np.einsum("nd,ldh->lnh", z, kernel_in) + bias_in[:, None, :]
    # Numerically stable SiLU / swish.
    hidden = pre / (1.0 + np.exp(-np.clip(pre, -60.0, 60.0)))
    scale_shift = np.einsum("lnh,lhq->lnq", hidden, kernel_out) + bias_out[:, None, :]
    scale, shift = np.split(scale_shift, 2, axis=-1)

    def stats(value: np.ndarray) -> dict[str, object]:
        return {
            "rms": _rms(value),
            "mean_abs": float(np.mean(np.abs(value))),
            "p99_abs": float(np.quantile(np.abs(value), 0.99)),
            "max_abs": float(np.max(np.abs(value))),
            "per_layer_rms": [
                float(np.sqrt(np.mean(np.square(layer)))) for layer in value
            ],
        }
    return {
        "input_zm": stats(z),
        "condition_in_pre_swish": stats(pre),
        "condition_hidden": stats(hidden),
        "film_scale": stats(scale),
        "film_shift": stats(shift),
    }


def _direction_summary(similarity: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    selected = similarity[mask]
    if selected.shape[0] == 0:
        return {"count": 0}
    top_order = np.argsort(selected, axis=-1)[:, ::-1]
    top1 = np.take_along_axis(selected, top_order[:, :1], axis=-1)[:, 0]
    top2 = np.take_along_axis(selected, top_order[:, 1:2], axis=-1)[:, 0]
    nearest = top_order[:, 0]
    counts = np.bincount(nearest, minlength=len(ATOMIC_NAMES))
    distribution = {
        name: {"count": int(count), "fraction": float(count / max(len(nearest), 1))}
        for name, count in zip(ATOMIC_NAMES, counts, strict=True)
    }
    return {
        "count": int(selected.shape[0]),
        "top1_cosine_mean": float(top1.mean()),
        "top1_cosine_std": float(top1.std()),
        "top1_cosine_p10": float(np.quantile(top1, 0.1)),
        "top1_cosine_p90": float(np.quantile(top1, 0.9)),
        "top1_minus_top2_mean": float((top1 - top2).mean()),
        "nearest_code_distribution": distribution,
    }


def _group_summary(values: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, object]:
    if not np.any(mask):
        return {"count": 0}
    velocity = values["velocity"][mask]
    no_q3 = values["velocity_no_q3"][mask]
    no_adapter = values["velocity_no_adapter"][mask]
    shuffled = values["velocity_shuffled"][mask]
    target = values["target"][mask]
    parallel = values["z_parallel"][mask]
    delta_rms = _rms(velocity - no_adapter)
    output_rms = _rms(velocity)
    return {
        "count": int(mask.sum()),
        "full_q1": _direction_summary(values["full_similarity"], mask),
        "text_q1": _direction_summary(values["text_similarity"], mask),
        "q1_text_full_cosine_mean": float(
            np.sum(
                values["full_direction"][mask] * values["text_direction"][mask],
                axis=-1,
            ).mean()
        ),
        "zm_norm_mean": float(np.linalg.norm(values["z_model"][mask], axis=-1).mean()),
        "q2_parallel_abs_mean": float(np.abs(values["z_parallel"][mask]).mean()),
        "q2_parallel_mean": float(parallel.mean()),
        "q2_negative_fraction": float(np.mean(parallel < 0)),
        "q2_min": float(parallel.min()),
        "q2_p10": float(np.quantile(parallel, 0.10)),
        "q2_median": float(np.median(parallel)),
        "q2_p90": float(np.quantile(parallel, 0.90)),
        "q2_max": float(parallel.max()),
        "q3_detail_norm_mean": float(values["z_detail_norm"][mask].mean()),
        "action_velocity_rms": output_rms,
        "adapter_induced_velocity_delta_rms": delta_rms,
        "delta_over_output_rms": float(delta_rms / max(output_rms, 1e-8)),
        "flow_mse_full": float(np.mean(np.square(velocity - target))),
        "flow_mse_no_q3": float(np.mean(np.square(no_q3 - target))),
        "q3_induced_velocity_delta_rms": _rms(velocity - no_q3),
        "flow_mse_no_adapter": float(np.mean(np.square(no_adapter - target))),
        "flow_mse_shuffled_zm": float(np.mean(np.square(shuffled - target))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", action="append", required=True)
    parser.add_argument("--norm-assets-dir", default="/mnt/cunchu/zjq/target")
    parser.add_argument("--norm-asset-id", default="openpi_norm_compact_accepted_v3")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = AtomicPi05Config(max_token_len=250, fast_action_ce_loss_weight=0.0)
    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    loader = build_atomic_loader(
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

    collected: dict[str, list[np.ndarray]] = {}
    supervised_parts: list[np.ndarray] = []
    atomic_weight_parts: list[np.ndarray] = []
    iterator = iter(loader)
    for batch_index in range(args.num_batches):
        batch = next(iterator)
        observation_np, actions_np = batch_to_observation(batch)
        observation = jax.tree.map(jnp.asarray, observation_np)
        actions = jnp.asarray(actions_np)
        text_observation = _text_observation(
            observation,
            jnp.asarray(batch["atomic_prompt_tokens"]),
            jnp.asarray(batch["atomic_prompt_mask"]),
        )
        result = _evaluate(
            model,
            observation,
            text_observation,
            actions,
            jax.random.key(20260801 + batch_index),
        )
        result = jax.device_get(result)
        for key, value in result.items():
            collected.setdefault(key, []).append(np.asarray(value))
        supervised_parts.append(np.asarray(batch["atomic_supervision_mask"], dtype=bool))
        atomic_weight_parts.append(np.asarray(batch["atomic_weights"], dtype=np.float32))

    values = {key: np.concatenate(parts, axis=0) for key, parts in collected.items()}
    supervised = np.concatenate(supervised_parts)
    atomic_weights = np.concatenate(atomic_weight_parts)
    drop = ~supervised
    positive_count = np.sum(atomic_weights > 0, axis=-1)
    single = supervised & (positive_count == 1)
    dual = supervised & (positive_count >= 2)
    if not np.any(drop):
        raise RuntimeError("sampled batches contain no drop/unsupervised rows")

    velocity = values["velocity"]
    no_adapter = values["velocity_no_adapter"]
    shuffled = values["velocity_shuffled"]
    target = values["target"]
    delta = velocity - no_adapter
    output_rms = _rms(velocity[drop])
    delta_rms = _rms(delta[drop])
    report = {
        "checkpoint": str(args.checkpoint),
        "sample_count": int(len(supervised)),
        "drop_count": int(drop.sum()),
        "supervised_count": int(supervised.sum()),
        "single_count": int(single.sum()),
        "dual_count": int(dual.sum()),
        "groups": {
            "drop": _group_summary(values, drop),
            "single": _group_summary(values, single),
            "dual": _group_summary(values, dual),
        },
        "adapter_condition": _atomic_condition_stats(params, values["z_model"]),
        "drop_full_q1": _direction_summary(values["full_similarity"], drop),
        "drop_text_q1": _direction_summary(values["text_similarity"], drop),
        "drop_q1_text_full_cosine": {
            "mean": float(
                np.sum(
                    values["full_direction"][drop] * values["text_direction"][drop],
                    axis=-1,
                ).mean()
            )
        },
        "drop_zm": {
            "norm_mean": float(np.linalg.norm(values["z_model"][drop], axis=-1).mean()),
            "q2_parallel_abs_mean": float(np.abs(values["z_parallel"][drop]).mean()),
            "q3_detail_norm_mean": float(values["z_detail_norm"][drop].mean()),
        },
        "drop_adapter_effect": {
            "action_velocity_rms": output_rms,
            "adapter_induced_velocity_delta_rms": delta_rms,
            "delta_over_output_rms": float(delta_rms / max(output_rms, 1e-8)),
            "flow_mse_full": float(np.mean(np.square(velocity[drop] - target[drop]))),
            "flow_mse_no_adapter": float(
                np.mean(np.square(no_adapter[drop] - target[drop]))
            ),
            "flow_mse_shuffled_zm": float(
                np.mean(np.square(shuffled[drop] - target[drop]))
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
