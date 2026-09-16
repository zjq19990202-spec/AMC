#!/usr/bin/env python3
"""Compare training-time Flow MSE with full ODE sampling on one exact batch."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from openpi.models import model as openpi_model
from openpi.training import sharding

from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import batch_to_observation, build_atomic_loader


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@nnx.jit
def _velocity_error(
    model,
    observation,
    actions,
    subtask_tokens,
    subtask_masks,
    times,
    rng,
):
    """Run the exact subtask-only layerwise velocity path used by ZM training."""

    preprocess_rng, noise_rng, _ = jax.random.split(rng, 3)
    observation = model._with_prompt(  # noqa: SLF001
        observation, subtask_tokens, subtask_masks
    )
    observation = openpi_model.preprocess_observation(
        preprocess_rng, observation, train=True
    )
    actions = model._mask_action_condition(actions)  # noqa: SLF001
    noise = model._mask_action_condition(  # noqa: SLF001
        jax.random.normal(noise_rng, actions.shape)
    )
    noisy = times[:, None, None] * noise + (1.0 - times[:, None, None]) * actions
    target_velocity = noise - actions
    if not model.config.enable_layerwise_atomic_flow:
        raise ValueError("this diagnostic requires the layerwise ZM training path")
    _, predicted_velocity, _ = model._joint_layerwise_velocity(  # noqa: SLF001
        observation, noisy, times
    )
    squared_error = jnp.square(predicted_velocity - target_velocity)
    return squared_error, jnp.square(target_velocity), actions


@nnx.jit
def _sample_subtask(
    model,
    observation,
    subtask_tokens,
    subtask_masks,
    noise,
    rng,
):
    observation = model._with_prompt(  # noqa: SLF001
        observation, subtask_tokens, subtask_masks
    )
    return model.sample_actions(rng, observation, num_steps=10, noise=noise)


def _active_loss(squared_error: np.ndarray, actions: np.ndarray) -> float:
    per_left = np.mean(squared_error[..., :8], axis=(1, 2))
    per_right = np.mean(squared_error[..., 8:16], axis=(1, 2))
    left_motion = np.mean(np.square(actions[..., :7]), axis=(1, 2))
    right_motion = np.mean(np.square(actions[..., 8:15]), axis=(1, 2))
    motion = np.stack([left_motion, right_motion], axis=-1)
    total = motion.sum(axis=-1, keepdims=True)
    shares = np.divide(
        motion,
        np.maximum(total, 1e-8),
        out=np.full_like(motion, 0.5),
        where=total > 1e-8,
    )
    return float(np.mean(np.sum(shares * np.stack([per_left, per_right], axis=-1), axis=-1)))


def _loss_summary(
    squared_error: np.ndarray,
    target_squared: np.ndarray,
    actions: np.ndarray,
) -> dict[str, float]:
    joint_indices = list(range(7)) + list(range(8, 15))
    per_sample_flow = np.mean(squared_error, axis=(1, 2))
    per_sample_action_max = np.max(np.abs(actions[..., :16]), axis=(1, 2))
    result = {
        "flow_mse_32d": float(np.mean(squared_error)),
        "flow_mse_real16": float(np.mean(squared_error[..., :16])),
        "flow_mse_joint14": float(np.mean(squared_error[..., joint_indices])),
        "flow_mse_gripper2": float(np.mean(squared_error[..., [7, 15]])),
        "flow_mse_padding16": float(np.mean(squared_error[..., 16:])),
        "flow_active_metric": _active_loss(squared_error, actions),
        "target_velocity_mse_32d": float(np.mean(target_squared)),
        "normalized_action_abs_mean": float(np.mean(np.abs(actions[..., :16]))),
        "normalized_action_abs_max": float(np.max(np.abs(actions[..., :16]))),
        "normalized_action_mse": float(np.mean(np.square(actions[..., :16]))),
        "normalized_action_outlier_fraction": float(np.mean(np.abs(actions[..., :16]) > 1.0)),
        "per_sample_flow_mse_p50": float(np.quantile(per_sample_flow, 0.50)),
        "per_sample_flow_mse_p90": float(np.quantile(per_sample_flow, 0.90)),
        "per_sample_flow_mse_p99": float(np.quantile(per_sample_flow, 0.99)),
        "per_sample_flow_mse_max": float(np.max(per_sample_flow)),
        "per_sample_action_abs_max_p50": float(np.quantile(per_sample_action_max, 0.50)),
        "per_sample_action_abs_max_p90": float(np.quantile(per_sample_action_max, 0.90)),
        "per_sample_action_abs_max_p99": float(np.quantile(per_sample_action_max, 0.99)),
    }
    result["padding_share_of_error_sum"] = float(
        np.sum(squared_error[..., 16:]) / np.maximum(np.sum(squared_error), 1e-12)
    )
    return result


def _time_bins(times: np.ndarray, squared_error: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (times >= lower) & (times < upper if upper < 1.0 else times <= upper)
        rows.append(
            {
                "lower": round(float(lower), 1),
                "upper": round(float(upper), 1),
                "count": int(mask.sum()),
                "flow_mse_32d": float(np.mean(squared_error[mask])) if mask.any() else None,
                "flow_mse_real16": (
                    float(np.mean(squared_error[mask, ..., :16])) if mask.any() else None
                ),
                "flow_mse_padding16": (
                    float(np.mean(squared_error[mask, ..., 16:])) if mask.any() else None
                ),
            }
        )
    return rows


def _put_batch(tree, data_sharding):
    return jax.tree.map(
        lambda value: jax.device_put(jnp.asarray(value), data_sharding), tree
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--atomic-composition-sidecar", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--skip-ode", action="store_true")
    parser.add_argument("--clip-state-action", action="store_true")
    args = parser.parse_args()

    if args.batch_size % args.devices:
        raise ValueError("batch size must be divisible by devices")
    norm_file = args.norm_assets_dir / args.norm_asset_id / "norm_stats.json"
    if not norm_file.is_file():
        raise FileNotFoundError(norm_file)
    if not (args.checkpoint / "params").is_dir():
        raise FileNotFoundError(args.checkpoint / "params")

    config = AtomicPi05Config(
        max_token_len=200,
        fast_action_ce_loss_weight=0.0,
        coefficient_target_kind="joint_delta",
        coefficient_target_dim=14,
        subtask_ce_loss_weight=0.0,
        enable_layerwise_atomic_flow=True,
        freeze_vision_encoder=False,
    )
    loader = build_atomic_loader(
        (args.dataset_root,),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        batch_size=args.batch_size,
        num_workers=32,
        seed=args.seed,
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        include_fast=False,
        subtask_max_token_len=config.subtask_max_token_len,
        atomic_composition_sidecar=args.atomic_composition_sidecar,
    )
    batch_np = next(iter(loader))
    observation_np, actions_np = batch_to_observation(batch_np)
    if args.clip_state_action:
        observation_np = dataclasses.replace(
            observation_np, state=np.clip(observation_np.state, -1.0, 1.0)
        )
        actions_np = np.clip(actions_np, -1.0, 1.0)

    # Fork DataLoader workers before initializing the JAX GPU runtime.
    if args.devices != jax.device_count():
        raise ValueError(
            f"expected exactly {args.devices} visible devices, got {jax.device_count()}"
        )
    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
    )
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Preserve the checkpoint's stored dtype. These fine-tuned checkpoints are
    # float32; forcing every parameter to bfloat16 can quantize away small
    # low-learning-rate updates before the model is evaluated.
    params = openpi_model.restore_params(args.checkpoint / "params")
    model = config.load(params)
    model_state = jax.device_put(nnx.state(model), replicated)
    model = nnx.merge(nnx.graphdef(model), model_state)

    observation = _put_batch(observation_np, data_sharding)
    actions = jax.device_put(jnp.asarray(actions_np), data_sharding)
    subtask_tokens = jax.device_put(
        jnp.asarray(batch_np["subtask_prompt_tokens"]), data_sharding
    )
    subtask_masks = jax.device_put(
        jnp.asarray(batch_np["subtask_prompt_mask"]), data_sharding
    )

    rng = jax.random.key(args.seed)
    _, _, time_rng = jax.random.split(rng, 3)
    random_times = jax.random.beta(time_rng, 1.5, 1, (args.batch_size,)) * 0.999 + 0.001
    random_times = jax.device_put(random_times, data_sharding)
    random_error, random_target, normalized_actions = jax.device_get(
        _velocity_error(
            model,
            observation,
            actions,
            subtask_tokens,
            subtask_masks,
            random_times,
            rng,
        )
    )
    random_times_np = np.asarray(jax.device_get(random_times))

    stratified_times_np = 0.001 + 0.999 * (
        (np.arange(args.batch_size, dtype=np.float32) + 0.5) / args.batch_size
    )
    stratified_times = jax.device_put(jnp.asarray(stratified_times_np), data_sharding)
    stratified_error, stratified_target, _ = jax.device_get(
        _velocity_error(
            model,
            observation,
            actions,
            subtask_tokens,
            subtask_masks,
            stratified_times,
            rng,
        )
    )

    actions_host = np.asarray(normalized_actions)
    ode_report = None
    if not args.skip_ode:
        sample_noise = jax.random.normal(
            jax.random.fold_in(rng, 9001),
            (args.batch_size, config.action_horizon, config.action_dim),
        )
        sample_noise = jax.device_put(sample_noise, data_sharding)
        sampled = np.asarray(
            jax.device_get(
                _sample_subtask(
                    model,
                    observation,
                    subtask_tokens,
                    subtask_masks,
                    sample_noise,
                    jax.random.fold_in(rng, 9002),
                )
            )
        )
        rollout_squared_error = np.square(sampled - actions_host[..., :16])
        joint_indices = list(range(7)) + list(range(8, 15))
        ode_report = {
            "normalized_mse_real16": float(np.mean(rollout_squared_error)),
            "normalized_mse_joint14": float(
                np.mean(rollout_squared_error[..., joint_indices])
            ),
            "normalized_mse_gripper2": float(
                np.mean(rollout_squared_error[..., [7, 15]])
            ),
        }

    report = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "batch_size": args.batch_size,
        "devices": args.devices,
        "per_device_batch": args.batch_size // args.devices,
        "seed": args.seed,
        "parameter_restore_dtype": "checkpoint_native",
        "state_action_clip": args.clip_state_action,
        "command": shlex.join(sys.argv),
        "prompt_contract": {
            "route": "subtask_only",
            "source": "training dataset sidecar subtask_prompt",
            "max_token_len": config.max_token_len,
            "tokenizer": "PaligemmaTokenizer",
        },
        "normalization_contract": {
            "asset_id": args.norm_asset_id,
            "norm_stats_sha256": _sha256(norm_file),
            "quantile": True,
            "state_action_clip": False,
        },
        "flow_contract": {
            "time_distribution": "Beta(1.5, 1) * 0.999 + 0.001",
            "target_velocity": "masked Gaussian noise - normalized actions",
            "objective": "mean squared velocity error over [B,50,32]",
            "train_augmentation": True,
            "layerwise_joint_forward": True,
        },
        "random_beta_t": {
            "time_mean": float(np.mean(random_times_np)),
            "time_std": float(np.std(random_times_np)),
            "summary": _loss_summary(
                np.asarray(random_error),
                np.asarray(random_target),
                actions_host,
            ),
            "time_bins": _time_bins(random_times_np, np.asarray(random_error)),
        },
        "stratified_uniform_t": {
            "summary": _loss_summary(
                np.asarray(stratified_error),
                np.asarray(stratified_target),
                actions_host,
            ),
            "time_bins": _time_bins(stratified_times_np, np.asarray(stratified_error)),
        },
        "ode10_sampling": ode_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    contract = f"""# Run contract

- Checkpoint: `{args.checkpoint}`
- Dataset: `{args.dataset_root}`
- Prompt: subtask-only, dataset-sidecar `subtask_prompt`, max length {config.max_token_len}
- Norm: `{args.norm_asset_id}` (`{_sha256(norm_file)}`)
- Normalization: q01/q99 affine, state/action clipping disabled
- Composition sidecar: `{args.atomic_composition_sidecar}`
- Batch: {args.batch_size} total, {args.devices} devices, {args.batch_size // args.devices} per device
- Flow time: `Beta(1.5, 1) * 0.999 + 0.001`
- Flow target: `masked Gaussian noise - normalized actions`
- Training-path preprocessing: enabled (`train=True`)
- Sampling: 10-step Euler ODE
- Command: `{shlex.join(sys.argv)}`
"""
    (args.output.parent / "run_contract.md").write_text(contract, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
