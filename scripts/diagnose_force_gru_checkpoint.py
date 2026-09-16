#!/usr/bin/env python3
"""Ablate the trained B1 GRU recurrence on episode-diverse held-out anchors."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from torch.utils.data import DataLoader, Subset

from openpi.models import model as openpi_model

from atomic_latent_vla.pi05.config import AtomicPi05Config
from atomic_latent_vla.pi05.force import (
    future_force_delta_target,
    future_force_forecast_metrics,
    sincos_embedding,
    temporal_masked_mean,
)
from atomic_latent_vla.pi05.force_training_data import (
    ForceNormalization,
    _ForceProcessedDataset,
    adapt_force_state_to_pi,
    batch_to_force_inputs,
    build_force_dataset,
    force_collate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--force-norm", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--anchors-per-episode", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def _episode_diverse_loader(args: argparse.Namespace) -> tuple[DataLoader, list[int]]:
    dataset = build_force_dataset(
        (args.dataset_root,),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        force_norm_path=args.force_norm,
        max_token_len=200,
        seed=0,
    )
    if not isinstance(dataset, _ForceProcessedDataset):
        raise TypeError("diagnostic expects exactly one force dataset")
    episodes = dataset._raw.anchor_episodes  # noqa: SLF001
    validation_episodes = sorted(np.unique(episodes[episodes % 10 == 0]).tolist())
    selected_by_episode: dict[int, list[int]] = {}
    for episode in validation_episodes:
        indices = np.flatnonzero(episodes == episode)
        if not len(indices):
            continue
        positions = np.linspace(
            0, len(indices) - 1, min(args.anchors_per_episode, len(indices)), dtype=np.int64
        )
        selected_by_episode[int(episode)] = indices[positions].tolist()

    # Round-robin ordering makes each batch contain different episodes, which
    # makes the shuffled-z_F control meaningful rather than a local-time shift.
    selected: list[int] = []
    for rank in range(args.anchors_per_episode):
        for episode in validation_episodes:
            values = selected_by_episode.get(int(episode), [])
            if rank < len(values):
                selected.append(values[rank])
    maximum = args.batch_size * args.max_batches
    selected = selected[:maximum]
    selected_episodes = episodes[np.asarray(selected)].astype(int).tolist()
    loader = DataLoader(
        Subset(dataset, selected),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        drop_last=True,
        collate_fn=force_collate,
    )
    return loader, selected_episodes


def _decode_with_reset_hidden(conditioner, force_latent: jax.Array) -> jax.Array:
    """Use trained GRU cells but reset their carry to z_F at every step."""

    config = conditioner.config
    batch = force_latent.shape[0]
    joint_latent = force_latent.reshape(
        batch, config.arm_count * config.force_latent_dim
    )
    stride = config.force_future_decoder_stride
    horizon = config.force_future_samples // stride
    future_positions = jnp.arange(1, horizon + 1, dtype=jnp.float32)
    decoder_inputs = (
        conditioner.future_latent_step(joint_latent)[:, None]
        + sincos_embedding(
            future_positions,
            config.force_encoder_width,
            base=config.force_position_base,
        )[None]
    )
    initial = conditioner.future_initial(joint_latent).reshape(
        batch,
        config.force_future_decoder_depth,
        config.force_encoder_width,
    )
    initial = jnp.swapaxes(initial, 0, 1)

    def decode_step(carry: jax.Array, step_input: jax.Array):
        del carry
        output = step_input
        for index in range(config.force_future_decoder_depth):
            _, output = conditioner.future_gru_cells[f"layer_{index}"](
                initial[index], output
            )
        return initial, output

    _, decoded = jax.lax.scan(
        decode_step,
        initial,
        jnp.swapaxes(decoder_inputs, 0, 1),
    )
    decoded = jnp.swapaxes(decoded, 0, 1)
    phase_hidden = (
        decoded[:, :, None, :]
        + conditioner.future_phase_embedding.value[None, None, :, :]
    )
    phase_hidden = nnx.swish(
        conditioner.future_phase_hidden(conditioner.future_phase_norm(phase_hidden))
    )
    prediction = conditioner.future_phase_out(phase_hidden)
    prediction = prediction.reshape(
        batch,
        horizon,
        stride,
        config.arm_count,
        config.force_dim,
    )
    prediction = jnp.transpose(prediction, (0, 3, 1, 2, 4))
    return prediction.reshape(
        batch,
        config.arm_count,
        config.force_future_samples,
        config.force_dim,
    )


def _loss_metrics(
    prediction: jax.Array,
    target: jax.Array,
    mask: jax.Array,
    *,
    config: AtomicPi05Config,
) -> dict[str, jax.Array]:
    def smooth_l1(values: jax.Array, valid: jax.Array) -> jax.Array:
        absolute = jnp.abs(values)
        point_loss = jnp.where(
            absolute < 1,
            0.5 * jnp.square(absolute),
            absolute - 0.5,
        )
        weights = valid[..., None].astype(point_loss.dtype)
        return jnp.sum(point_loss * weights) / jnp.maximum(
            jnp.sum(weights) * point_loss.shape[-1], 1
        )

    raw_loss = smooth_l1(prediction - target, mask)
    coarse_prediction, coarse_mask = temporal_masked_mean(
        prediction, mask, stride=config.force_temporal_stride
    )
    coarse_target, _ = temporal_masked_mean(
        target, mask, stride=config.force_temporal_stride
    )
    coarse_loss = smooth_l1(coarse_prediction - coarse_target, coarse_mask)
    metrics = future_force_forecast_metrics(
        prediction,
        target,
        mask,
        coarse_stride=config.force_temporal_stride,
        sample_rate_hz=config.force_sample_rate_hz,
    )
    coarse_step_change = jnp.mean(jnp.abs(jnp.diff(coarse_prediction, axis=-2)))
    return {
        "loss": raw_loss + config.force_future_coarse_loss_weight * coarse_loss,
        "raw_rmse": metrics["raw_rmse"],
        "coarse_rmse": metrics["coarse_rmse"],
        "peak_amplitude_mae": metrics["peak_amplitude_mae"],
        "peak_timing_mae_ms": metrics["peak_timing_mae_ms"],
        "coarse_step_change": coarse_step_change,
    }


@nnx.jit
def _diagnose_batch(
    model,
    observation,
    actions,
    force,
    normalized_physical_zero_force,
    normalized_raw_zero_state,
    rng,
):
    output = model.compute_force_stage_loss(
        rng,
        observation,
        actions,
        **force,
        train_fast=False,
        train=False,
        return_output=True,
    )
    target, mask = future_force_delta_target(
        force["slow_force_history"],
        force["future_force"],
        force["future_force_mask"],
    )
    conditioner = model._require_force_conditioner()  # noqa: SLF001
    normal = output.predicted_future_force_delta
    # Keep the paired batch fixed and intervene on only one conditioning route.
    # Batch order is episode-diverse, so roll(1) is an out-of-sample condition.
    query_hidden, prefix_mask, _, _, prefix_hidden = model._prefix_forward(  # noqa: SLF001
        observation,
        return_prefix_hidden=True,
    )
    _, _, z_model, _, _ = model._latent(  # noqa: SLF001
        query_hidden, model._controlled_state(observation.state)  # noqa: SLF001
    )
    context_mask = prefix_mask[:, : prefix_hidden.shape[1]]

    def predict_from_slow_inputs(history_force, history_state):
        context = conditioner.encode_context(
            prefix_hidden,
            context_mask,
            z_model,
            history_force,
            history_state,
            force["slow_history_mask"],
        )
        return conditioner.predict_future_force_delta(context.latent, z_model)

    # Input-level controls. "tensor_zero" is the usual ablation: zero after
    # normalization, which is approximately the center of the training range.
    # "physical_zero" passes a zero wrench through the force normalization.
    # For state, raw all-zero joint coordinates are adapted to PI coordinates
    # before normalization; this is deliberately reported separately because
    # it is an out-of-distribution robot pose, not a neutral missing-state token.
    force_tensor_zero = predict_from_slow_inputs(
        jnp.zeros_like(force["slow_force_history"]),
        force["slow_state_history"],
    )
    force_physical_zero = predict_from_slow_inputs(
        jnp.broadcast_to(
            normalized_physical_zero_force,
            force["slow_force_history"].shape,
        ),
        force["slow_state_history"],
    )
    state_tensor_zero = predict_from_slow_inputs(
        force["slow_force_history"],
        jnp.zeros_like(force["slow_state_history"]),
    )
    state_raw_zero = predict_from_slow_inputs(
        force["slow_force_history"],
        jnp.broadcast_to(
            normalized_raw_zero_state,
            force["slow_state_history"].shape,
        ),
    )
    both_tensor_zero = predict_from_slow_inputs(
        jnp.zeros_like(force["slow_force_history"]),
        jnp.zeros_like(force["slow_state_history"]),
    )
    shuffled = conditioner.predict_future_force_delta(
        jnp.roll(output.force_latent, 1, axis=0), z_model
    )
    zero = conditioner.predict_future_force_delta(
        jnp.zeros_like(output.force_latent), z_model
    )
    shuffled_zm = conditioner.predict_future_force_delta(
        output.force_latent, jnp.roll(z_model, 1, axis=0)
    )
    zero_zm = conditioner.predict_future_force_delta(
        output.force_latent, jnp.zeros_like(z_model)
    )
    both_zero = conditioner.predict_future_force_delta(
        jnp.zeros_like(output.force_latent), jnp.zeros_like(z_model)
    )
    variants = {
        "normal": normal,
        "shuffled_zf": shuffled,
        "zero_zf": zero,
        "shuffled_zm": shuffled_zm,
        "zero_zm": zero_zm,
        "both_zero": both_zero,
        "slow_force_tensor_zero": force_tensor_zero,
        "slow_force_physical_zero": force_physical_zero,
        "slow_state_tensor_zero": state_tensor_zero,
        "slow_state_raw_zero": state_raw_zero,
        "slow_force_state_tensor_zero": both_tensor_zero,
    }
    result = {
        name: _loss_metrics(prediction, target, mask, config=model.config)
        for name, prediction in variants.items()
    }
    result["effects"] = {
        "shuffle_prediction_mae": jnp.mean(jnp.abs(normal - shuffled)),
        "zero_zf_prediction_mae": jnp.mean(jnp.abs(normal - zero)),
        "shuffle_zm_prediction_mae": jnp.mean(jnp.abs(normal - shuffled_zm)),
        "zero_zm_prediction_mae": jnp.mean(jnp.abs(normal - zero_zm)),
        "slow_force_tensor_zero_prediction_mae": jnp.mean(
            jnp.abs(normal - force_tensor_zero)
        ),
        "slow_force_physical_zero_prediction_mae": jnp.mean(
            jnp.abs(normal - force_physical_zero)
        ),
        "slow_state_tensor_zero_prediction_mae": jnp.mean(
            jnp.abs(normal - state_tensor_zero)
        ),
        "slow_state_raw_zero_prediction_mae": jnp.mean(
            jnp.abs(normal - state_raw_zero)
        ),
        "slow_force_state_tensor_zero_prediction_mae": jnp.mean(
            jnp.abs(normal - both_tensor_zero)
        ),
        "force_latent_batch_std": jnp.mean(jnp.std(output.force_latent, axis=0)),
    }
    return result


def main() -> None:
    args = parse_args()
    loader, episodes = _episode_diverse_loader(args)
    if not episodes:
        raise RuntimeError("no held-out anchors selected")
    # Decode/prefetch before initializing JAX. Forking DataLoader workers after
    # the GPU runtime starts is unsafe and can deadlock on some hosts.
    batches_np = []
    for batch_np in loader:
        batches_np.append(batch_np)
        if len(batches_np) >= args.max_batches:
            break
    if not batches_np:
        raise RuntimeError("held-out loader produced no full batches")
    params_path = args.checkpoint / "params"
    if not params_path.is_dir():
        raise FileNotFoundError(params_path)
    config = AtomicPi05Config(
        max_token_len=200,
        coefficient_target_kind="joint_delta",
        coefficient_target_dim=14,
        fast_action_ce_loss_weight=0.0,
        enable_force_stage=True,
        force_future_decoder_stride=4,
        force_future_decoder_kind="phase_mlp",
        force_encoder_depth=2,
        force_position_base=10_000.0,
        force_history_train_lengths=(120,),
        force_future_loss_weight=1.0,
        force_flow_loss_weight=1.0,
        force_stop_gradient_backbone=True,
        force_encoder_width=512,
        force_encoder_num_heads=8,
        force_encoder_mlp_dim=1024,
        force_latent_dim=512,
        force_context_from_prefix=False,
        force_future_condition_on_zm=True,
    )
    params = openpi_model.restore_params(params_path, dtype=jnp.bfloat16)
    # The formal B1 source used a bias-free recurrent hidden projection. The
    # currently installed NNX GRUCell materializes an additional zero bias by
    # default. Merge the checkpoint as a parameter subset so that this exact
    # zero is retained; mathematically it is identical to the trained graph's
    # missing bias and avoids weakening the checkpoint structure check for any
    # parameter that is actually present.
    model = config.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(params)
    model = nnx.merge(graphdef, state)
    model.eval()

    force_norm = ForceNormalization.load(args.force_norm)
    normalized_physical_zero_force = jnp.asarray(
        force_norm.normalize_force(np.zeros((6,), dtype=np.float32))
    )
    normalized_raw_zero_state = jnp.asarray(
        force_norm.normalize_state(
            adapt_force_state_to_pi(np.zeros((16,), dtype=np.float32))
        )
    )

    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    count = 0
    for batch_index, batch_np in enumerate(batches_np):
        observation_np, actions_np, force_np = batch_to_force_inputs(batch_np)
        observation = jax.tree.map(jnp.asarray, observation_np)
        actions = jnp.asarray(actions_np)
        force = {key: jnp.asarray(value) for key, value in force_np.items()}
        result = jax.device_get(
            _diagnose_batch(
                model,
                observation,
                actions,
                force,
                normalized_physical_zero_force,
                normalized_raw_zero_state,
                jax.random.key(20260823 + batch_index),
            )
        )
        for group, values in result.items():
            for key, value in values.items():
                sums[group][key] += float(value)
        count += 1
        print(f"evaluated_batches={count}", flush=True)

    print(
        f"anchors={count * args.batch_size} "
        f"heldout_episodes={sorted(set(episodes[: count * args.batch_size]))}"
    )
    for group in (
        "normal", "shuffled_zf", "zero_zf",
        "shuffled_zm", "zero_zm", "both_zero", "effects",
        "slow_force_tensor_zero", "slow_force_physical_zero",
        "slow_state_tensor_zero", "slow_state_raw_zero",
        "slow_force_state_tensor_zero",
    ):
        values = " ".join(
            f"{key}={value / count:.6f}" for key, value in sorted(sums[group].items())
        )
        print(f"{group} {values}")


if __name__ == "__main__":
    main()
