#!/usr/bin/env python3
"""Pretrain the shared slow/fast force encoder with slow future-force prediction."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from openpi.shared import array_typing as at
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import sharding
from openpi.training import utils as training_utils

from atomic_latent_vla.pi05.config import AtomicPi05Config
from atomic_latent_vla.pi05.force import (
    future_force_delta_target,
    future_force_forecast_metrics,
)
from atomic_latent_vla.pi05.force_training_data import (
    batch_to_force_inputs,
    build_force_train_validation_loaders,
)
from atomic_latent_vla.pi05.weights import AtomicPi05CheckpointLoader


DEFAULT_ROOTS = (
    "/mnt/cunchu/zjq/target/lerobot_v3_4to1_plug_force",
    "/mnt/cunchu/zjq/target/lerobot_v3_4to1_vase_force",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", action="append", default=[])
    parser.add_argument("--norm-assets-dir", required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--force-norm", required=True)
    parser.add_argument("--base-params", required=True)
    parser.add_argument("--checkpoint-base-dir", default="/mnt/cunchu/zjq/atomic_pi05_runs")
    parser.add_argument("--run-name", default=time.strftime("force_encoder_b1_%Y%m%d_%H%M%S"))
    parser.add_argument(
        "--coefficient-target", choices=("tcp_twist", "joint_delta"), default="joint_delta"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--max-token-len", type=int, default=200)
    parser.add_argument("--future-decoder-stride", type=int, choices=(1, 4), default=4)
    parser.add_argument(
        "--future-decoder-kind",
        choices=("linear_chunk", "phase_mlp"),
        default="phase_mlp",
    )
    parser.add_argument("--encoder-depth", type=int, choices=(1, 2), default=2)
    parser.add_argument("--encoder-width", type=int, default=256)
    parser.add_argument("--encoder-heads", type=int, default=4)
    parser.add_argument("--encoder-mlp-dim", type=int, default=1024)
    parser.add_argument("--force-latent-dim", type=int, default=256)
    parser.add_argument(
        "--force-only-zf",
        action="store_true",
        help="build zF only from slow force/state; omit prefix and zM semantic shortcuts",
    )
    parser.add_argument(
        "--future-condition-on-zm",
        action="store_true",
        help="condition only the training-time future-force decoder on stop-gradient zM",
    )
    parser.add_argument("--position-base", type=float, default=10_000.0)
    parser.add_argument(
        "--history-train-lengths",
        default="120",
        help="comma-separated end-aligned B1 history lengths, e.g. 120 or 40,80,120",
    )
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--peak-lr", type=float, default=1.0e-4)
    parser.add_argument("--decay-steps", type=int, default=20_000)
    parser.add_argument("--decay-lr", type=float, default=1.0e-5)
    parser.add_argument("--save-interval", type=int, default=1_000)
    parser.add_argument("--keep-period", type=int, default=5_000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-modulus", type=int, default=10)
    parser.add_argument("--validation-remainder", type=int, default=0)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--skip-checkpoint", action="store_true")
    return parser.parse_args()


def _base_train_module():
    """Load the training utilities from this repository's Marvin/OpenPI copy."""

    train_path = Path(__file__).resolve().parents[1] / "vendor" / "pi0.5" / "scripts" / "train.py"
    if not train_path.is_file():
        train_path = Path("/home/admin123/zjq/ws/pi0.5/scripts/train.py")
    spec = importlib.util.spec_from_file_location("atomic_force_base_train", train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import OpenPI trainer from {train_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_jax_batch(batch_np: dict[str, object], data_sharding: jax.sharding.NamedSharding):
    observation, actions, force = batch_to_force_inputs(batch_np)
    observation = jax.tree.map(
        lambda value: jax.make_array_from_process_local_data(data_sharding, value),
        observation,
    )
    actions = jax.make_array_from_process_local_data(data_sharding, actions)
    force = {
        key: jax.make_array_from_process_local_data(data_sharding, value)
        for key, value in force.items()
    }
    return observation, actions, force


def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch,
):
    model = nnx.merge(state.model_def, state.params)
    model.train()
    observation, actions, force = batch

    def loss_fn(model, step_rng):
        output = model.compute_force_stage_loss(
            step_rng,
            observation,
            actions,
            **force,
            train_fast=False,
            train=True,
            return_output=True,
        )
        return output.total_loss, {
            "loss": output.total_loss,
            "future_force_loss": output.future_force_loss,
            "future_force_raw_loss": output.future_force_raw_loss,
            "future_force_coarse_loss": output.future_force_coarse_loss,
            "force_latent_norm": jnp.mean(jnp.linalg.norm(output.force_latent, axis=-1)),
            # B1 must retain the fast residual's exact zero initialization.
            "delta_z_norm": jnp.mean(jnp.linalg.norm(output.delta_z, axis=-1)),
        }

    train_rng = jax.random.fold_in(rng, state.step)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng
    )
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    nnx.update(model, optax.apply_updates(params, updates))
    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=nnx.state(model),
        opt_state=new_opt_state,
    )
    return new_state, info | {"grad_norm": optax.global_norm(grads), "objective": loss}


def eval_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch,
):
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    observation, actions, force = batch
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
    metrics = future_force_forecast_metrics(
        output.predicted_future_force_delta,
        target,
        mask,
        coarse_stride=config.model.force_temporal_stride,
        sample_rate_hz=config.model.force_sample_rate_hz,
    )
    return {
        "future_force_loss": output.future_force_loss,
        "future_force_raw_loss": output.future_force_raw_loss,
        "future_force_coarse_loss": output.future_force_coarse_loss,
        **metrics,
    }


def _write_contract(
    args: argparse.Namespace, config: _config.TrainConfig, roots: tuple[str, ...]
) -> None:
    directory = Path(config.checkpoint_dir) / "run_contract"
    directory.mkdir(parents=True, exist_ok=True)
    force_norm = Path(args.force_norm).resolve()
    base_norm = Path(args.norm_assets_dir, args.norm_asset_id, "norm_stats.json").resolve()
    files = {
        "force.py": Path(__file__).resolve().parents[1] / "src/atomic_latent_vla/pi05/force.py",
        "model.py": Path(__file__).resolve().parents[1] / "src/atomic_latent_vla/pi05/model.py",
        "gemma_adapter.py": Path(__file__).resolve().parents[1]
        / "src/atomic_latent_vla/pi05/gemma_adapter.py",
        "config.py": Path(__file__).resolve().parents[1]
        / "src/atomic_latent_vla/pi05/config.py",
        "force_training_data.py": Path(__file__).resolve().parents[1]
        / "src/atomic_latent_vla/pi05/force_training_data.py",
        "train_force_encoder_b1.py": Path(__file__).resolve(),
        "force_norm": force_norm,
        "base_norm": base_norm,
    }
    contract = {
        "stage": "B1 shared force encoder pretraining",
        "datasets": list(roots),
        "base_params": str(Path(args.base_params).resolve()),
        "base_norm": str(base_norm),
        "force_norm": str(force_norm),
        "prompt": {
            "route": "PromptFromLeRobotTask",
            "source": "each dataset's meta/tasks.parquet task string",
            "mixing": "none",
            "max_token_length": args.max_token_len,
        },
        "normalization": {
            "force_and_force_state": "separate per-channel q01/q99 affine, no clipping",
            "load_compensation": False,
            "static_sensor_bias_already_removed": True,
            "constant_range": "zero",
            "force_left_right": (
                "both sensors retained in fixed [right,left] association; pooled "
                "statistics and exactly shared encoder/projection parameters"
            ),
            "state_adapt_to_pi": True,
            "state_encoder_views": (
                "normalized native [left8,right8] becomes right:[right8,left8], "
                "left:[left8,right8]"
            ),
            "state_clip": False,
            "action_clip": False,
            "inference_output_clip": False,
        },
        "time": {
            "force_hz": 120,
            "token_hz": 30,
            "slow_samples": 120,
            "fast_samples": 40,
            "position": (
                "end-aligned integer token lags with Transformer sincos; "
                f"base={config.model.force_position_base:g}"
            ),
            "b1_history_lengths": list(config.model.force_history_train_lengths),
        },
        "episode_holdout": {
            "rule": "episode_index % modulus == remainder",
            "validation_modulus": args.validation_modulus,
            "validation_remainder": args.validation_remainder,
            "eval_batches": args.eval_batches,
        },
        "objective": {
            "train_fast": False,
            "future_force": (
                "both slow arm latents concatenated; a 50-step GRU emits four "
                "raw-rate samples per step, preserving all 200 normalized "
                "deltas for both 6D wrenches"
            ),
            "future_decoder_stride": config.model.force_future_decoder_stride,
            "future_decoder_kind": config.model.force_future_decoder_kind,
            "encoder_depth": config.model.force_encoder_depth,
            "encoder_width": config.model.force_encoder_width,
            "encoder_heads": config.model.force_encoder_num_heads,
            "encoder_mlp_dim": config.model.force_encoder_mlp_dim,
            "force_latent_dim": config.model.force_latent_dim,
            "future_decoder_steps": (
                config.model.force_future_samples
                // config.model.force_future_decoder_stride
            ),
            "future_force_coarse_auxiliary": "mean4 to 50 points; weight 0.25",
            "future_force_loss_weight": 1.0,
            "fast_prediction_loss": False,
        },
        "data_loading": {
            "load_future_force_targets": True,
            "future_force_target_shape_per_sample": [2, 200, 6],
        },
        "optimizer_gradient_clip_norm": getattr(config.optimizer, "clip_gradient_norm", None),
        "batch_size": args.batch_size,
        "devices": args.devices,
        "fsdp_devices": args.fsdp_devices,
        "schedule": {
            "warmup_steps": args.warmup_steps,
            "peak_lr": args.peak_lr,
            "decay_steps": args.decay_steps,
            "decay_lr": args.decay_lr,
        },
        "checkpointing": {
            "save_interval": args.save_interval,
            "keep_period": args.keep_period,
            "orbax_max_to_keep": 1,
        },
        "sha256": {name: _sha256(path) for name, path in files.items()},
    }
    (directory / "contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    shutil.copy2(force_norm, directory / "force_norm_stats.json")
    shutil.copy2(base_norm, directory / "base_norm_stats.json")


def main() -> None:
    args = parse_args()
    if args.max_token_len < 200:
        raise ValueError("full-32-D PI0.5 state tokenization requires max-token-len >= 200")
    history_train_lengths = tuple(
        int(value.strip()) for value in args.history_train_lengths.split(",") if value.strip()
    )
    if not history_train_lengths:
        raise ValueError("history-train-lengths must contain at least one length")
    if args.eval_batches <= 0:
        raise ValueError("eval-batches must be positive")
    roots = tuple(args.dataset_root) if args.dataset_root else DEFAULT_ROOTS
    if args.batch_size % args.devices:
        raise ValueError("batch size must be divisible by devices")
    if args.fsdp_devices <= 0 or args.devices % args.fsdp_devices:
        raise ValueError("fsdp-devices must be positive and divide devices")
    for root in roots:
        if not Path(root, "meta", "info.json").is_file():
            raise FileNotFoundError(f"force dataset not found: {root}")
    for path in (
        Path(args.force_norm),
        Path(args.norm_assets_dir, args.norm_asset_id, "norm_stats.json"),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not Path(args.base_params).is_dir():
        raise FileNotFoundError(args.base_params)

    # Construct the loader before initializing JAX so forked video workers do
    # not inherit a live multithreaded GPU runtime.
    loader, validation_loader = build_force_train_validation_loaders(
        roots,
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        force_norm_path=args.force_norm,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_token_len=args.max_token_len,
        validation_modulus=args.validation_modulus,
        validation_remainder=args.validation_remainder,
        load_future_force_targets=True,
    )
    data_iter = iter(loader)
    first_np = next(data_iter)
    # Spawn/decode every fixed held-out batch before JAX starts any runtime
    # threads. Forking DataLoader workers after JAX initialization can deadlock.
    validation_batches_np = []
    for validation_np in validation_loader:
        validation_batches_np.append(validation_np)
        if len(validation_batches_np) >= args.eval_batches:
            break
    if not validation_batches_np:
        raise RuntimeError("validation loader produced no full batches")

    if args.devices > jax.device_count():
        raise ValueError(f"requested {args.devices} devices, found {jax.device_count()}")
    coefficient_dim = 14 if args.coefficient_target == "joint_delta" else 12
    model_config = AtomicPi05Config(
        max_token_len=args.max_token_len,
        coefficient_target_kind=args.coefficient_target,
        coefficient_target_dim=coefficient_dim,
        fast_action_ce_loss_weight=0.0,
        enable_force_stage=True,
        force_future_decoder_stride=args.future_decoder_stride,
        force_future_decoder_kind=args.future_decoder_kind,
        force_encoder_depth=args.encoder_depth,
        force_encoder_width=args.encoder_width,
        force_encoder_num_heads=args.encoder_heads,
        force_encoder_mlp_dim=args.encoder_mlp_dim,
        force_latent_dim=args.force_latent_dim,
        force_context_from_prefix=not args.force_only_zf,
        force_future_condition_on_zm=args.future_condition_on_zm,
        force_position_base=args.position_base,
        force_history_train_lengths=history_train_lengths,
        # B1 has no competing policy loss, so report/optimize the reconstruction
        # objective at its natural scale. B2 may restore the documented 0.2.
        force_future_loss_weight=1.0,
        force_flow_loss_weight=1.0,
        force_stop_gradient_backbone=True,
    )
    base_config = _config.get_config("pi05_hdf5_dscrew_v3_jax")
    lr_schedule = dataclasses.replace(
        base_config.lr_schedule,
        warmup_steps=args.warmup_steps,
        peak_lr=args.peak_lr,
        decay_steps=args.decay_steps,
        decay_lr=args.decay_lr,
    )
    config = dataclasses.replace(
        base_config,
        name=args.run_name,
        exp_name=args.run_name,
        model=model_config,
        weight_loader=AtomicPi05CheckpointLoader(args.base_params),
        freeze_filter=model_config.get_force_freeze_filter(train_atomic_adapters=False),
        lr_schedule=lr_schedule,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.steps,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
        checkpoint_base_dir=args.checkpoint_base_dir,
        wandb_enabled=False,
        ema_decay=None,
        fsdp_devices=args.fsdp_devices,
        overwrite=False,
        resume=False,
    )
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    logging.info(
        "first batch actions=%s slow_force=%s slow_state=%s future_force=%s",
        first_np["actions"].shape,
        first_np["slow_force_history"].shape,
        first_np["slow_state_history"].shape,
        first_np["future_force"].shape,
    )
    logging.info(
        "B1 batch=%d (%d/GPU) devices=%d fsdp=%d encoder_depth=%d "
        "decoder=%s:%dx%d position_base=%g "
        "history_lengths=%s force_future_weight=1.0",
        args.batch_size,
        args.batch_size // args.devices,
        args.devices,
        args.fsdp_devices,
        model_config.force_encoder_depth,
        model_config.force_future_decoder_kind,
        model_config.force_future_samples // model_config.force_future_decoder_stride,
        model_config.force_future_decoder_stride,
        model_config.force_position_base,
        model_config.force_history_train_lengths,
    )

    base_train = _base_train_module()
    rng = jax.random.key(args.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=False,
        resume=False,
    )
    assert not resuming
    _write_contract(args, config, roots)
    state, state_sharding = base_train.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    checkpoint_loader_view = SimpleNamespace(
        data_config=lambda: SimpleNamespace(norm_stats=None, asset_id=None)
    )
    ptrain = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    peval = jax.jit(
        functools.partial(eval_step, config),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=replicated,
    )
    current_np = first_np
    while int(state.step) < config.num_train_steps:
        batch = _to_jax_batch(current_np, data_sharding)
        state, info = ptrain(train_rng, state, batch)
        step = int(state.step)
        if step == 1 or step % args.log_interval == 0:
            values = {key: float(value) for key, value in jax.device_get(info).items()}
            logging.info(
                "step=%d %s", step, " ".join(f"{key}={value:.6f}" for key, value in values.items())
            )
        if not args.skip_checkpoint and (
            step % config.save_interval == 0 or step == config.num_train_steps
        ):
            _checkpoints.save_state(checkpoint_manager, state, checkpoint_loader_view, step)
            logging.info("submitted checkpoint step=%d", step)
        try:
            current_np = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            current_np = next(data_iter)
    if not args.skip_checkpoint:
        checkpoint_manager.wait_until_finished()

    validation_sums: dict[str, float] = {}
    validation_count = 0
    for validation_count, validation_np in enumerate(validation_batches_np, start=1):
        validation_batch = _to_jax_batch(validation_np, data_sharding)
        validation_info = peval(
            jax.random.fold_in(train_rng, 1_000_000 + validation_count),
            state,
            validation_batch,
        )
        values = {
            key: float(value) for key, value in jax.device_get(validation_info).items()
        }
        for key, value in values.items():
            validation_sums[key] = validation_sums.get(key, 0.0) + value
    logging.info(
        "heldout batches=%d %s",
        validation_count,
        " ".join(
            f"{key}={value / validation_count:.6f}"
            for key, value in sorted(validation_sums.items())
        ),
    )


if __name__ == "__main__":
    main()
