#!/usr/bin/env python3
"""Stage-A text/state z_T training with the shared PI0.5 Action Expert.

This entrypoint deliberately has no image/video input and never executes the
z_M visual-detail route. It uses the released π0.5 base checkpoint and trains
the language prior through the same Action Expert that z_M will later reuse:

* per-arm Two-Way supervision, plus Dual Top-2 weighted-cos;
* standard 50-by-32 PI0.5 flow through the same Action Expert used by z_M;
* a hard codebook freeze after the configured semantic warm-up;
The legacy DCT head remains checkpoint-compatible but is disabled by default;
there is no Wall/t-squared weighting, FAST action-token CE, or video decode.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.training import config as _config
from openpi.training import checkpoints as _checkpoints
from openpi.training import sharding
from openpi.training import utils as training_utils

from atomic_latent_vla.pi05 import (
    AtomicCompositionTargets,
    AtomicPi05Config,
    AtomicTargets,
)
from atomic_latent_vla.pi05.training_data import build_atomic_text_loader
from atomic_latent_vla.pi05.weights import AtomicPi05CheckpointLoader
from atomic_latent_vla.tcp import BIMANUAL_TCP_POSE_SIDECAR


DEFAULT_ROOTS = (
    "/mnt/cunchu/admin123/atom/dscrew_native",
    "/mnt/cunchu/admin123/atom/dscrew_mirror",
    "/mnt/cunchu/admin123/atom/record_data_gap_split_native_v2",
    "/mnt/cunchu/admin123/atom/record_data_gap_split_mirror_v2",
    "/mnt/cunchu/admin123/atom/cabinet_native",
    "/mnt/cunchu/admin123/atom/cabinet_mirror",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", action="append", default=[])
    parser.add_argument("--norm-assets-dir", required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument(
        "--atomic-composition-sidecar",
        default=None,
        help=(
            "Dataset meta sidecar containing Dual Top-2 weighted-cos targets. "
            "Pass the reviewed stay-aware sidecar explicitly for target data."
        ),
    )
    parser.add_argument(
        "--tcp-twist-norm",
        required=True,
        help="TCP relative-delta q01/q99 stats from compute_tcp_twist_norm_fast.py.",
    )
    parser.add_argument(
        "--coefficient-target",
        choices=("tcp_twist", "joint_delta"),
        default="tcp_twist",
        help=(
            "Compact DiT target: normalized TCP200 twist or OpenPI-normalized "
            "right7+left7 relative joint actions."
        ),
    )
    parser.add_argument(
        "--base-params",
        default="/mnt/cunchu/zjq/.cache/openpi/openpi-assets/checkpoints/pi05_base/params",
    )
    parser.add_argument("--checkpoint-base-dir", default="/mnt/cunchu/zjq/atomic_pi05_runs")
    parser.add_argument("--run-name", default=time.strftime("atomic_zt_fast_%Y%m%d_%H%M%S"))
    parser.add_argument("--batch-size", type=int, default=32, help="global FSDP batch size")
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument(
        "--max-token-len",
        type=int,
        default=200,
        help="Original PI0.5 padded prompt/full-32-D-state prefix length.",
    )
    parser.add_argument(
        "--fsdp-devices",
        type=int,
        default=None,
        help=(
            "Devices per FSDP group. Defaults to --devices; use 1 for "
            "replicated JAX data parallel when one GPU can hold the model."
        ),
    )
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--save-interval", type=int, default=1_000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1_000,
        help="LR warmup updates; defaults to the pi05 multi-task recipe.",
    )
    parser.add_argument(
        "--peak-lr",
        type=float,
        default=2.5e-5,
        help="Peak learning rate; defaults to the pi05 multi-task recipe.",
    )
    parser.add_argument(
        "--decay-lr",
        type=float,
        default=2.5e-6,
        help="Final learning rate; defaults to the pi05 multi-task recipe.",
    )
    parser.add_argument(
        "--lr-decay-steps",
        type=int,
        default=30_000,
        help="LR decay updates; defaults to the pi05 multi-task recipe.",
    )
    parser.add_argument(
        "--codebook-freeze-step",
        type=int,
        default=25_000,
        help="Freeze the 2x13 semantic codebook at this global optimizer step.",
    )
    parser.add_argument(
        "--codebook-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional Stage-A single-code angular Huber weight. The production "
            "default is zero: Two-Way/dual-ratio already learn Q1 and codebook."
        ),
    )
    parser.add_argument(
        "--flow-loss-weight",
        type=float,
        default=1.0,
        help="Weight of the standard shared PI0.5 50x32 velocity-flow loss.",
    )
    parser.add_argument(
        "--coefficient-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional legacy DCT auxiliary weight. The shared-PI0.5 recipe "
            "keeps this at zero and does not execute the coefficient DiT."
        ),
    )
    parser.add_argument(
        "--atomic-loss-weight",
        type=float,
        default=0.1,
        help="Weight of Q1/codebook Two-Way supervision.",
    )
    parser.add_argument(
        "--atomic-projection-loss-weight",
        type=float,
        default=0.025,
        help="Weight of the Dual-only Top-2 weighted-cos direction target.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help="Run compilation/data smoke tests without writing multi-GB checkpoints.",
    )
    parser.add_argument(
        "--resume-trainable",
        default=None,
        help="Legacy params.pkl initialization; restores weights but starts a fresh optimizer state.",
    )
    parser.add_argument(
        "--initial-step",
        type=int,
        default=0,
        help="Global step represented by --resume-trainable, used for loss/LR schedules and checkpoint numbering.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume model, optimizer and global step from this run's latest full OpenPI checkpoint.",
    )
    return parser.parse_args()


def _base_train_module():
    root = "/mnt/cunchu/yc/pi05/scripts"
    if root not in sys.path:
        sys.path.insert(0, root)
    import train as base_train  # noqa: PLC0415

    return base_train


def _atomic_targets(weights: jax.Array, supervised: jax.Array) -> AtomicTargets:
    order = jnp.flip(jnp.argsort(weights, axis=-1)[..., -2:], axis=-1)
    top_weights = jnp.take_along_axis(weights, order, axis=-1)
    has_second = top_weights[..., 1] > 0
    labels = jnp.where(supervised[..., None], order, -jnp.ones_like(order))
    labels = labels.at[..., 1].set(
        jnp.where(supervised & has_second, labels[..., 1], -1)
    )
    top_weights = jnp.where(supervised[..., None], top_weights, jnp.zeros_like(top_weights))
    return AtomicTargets(
        labels=labels.astype(jnp.int32), weights=top_weights, supervision_mask=supervised
    )


def _dual_composition_targets(
    weights: jax.Array,
    confidence: jax.Array,
    available: jax.Array,
    strict_supervision: jax.Array,
    dual_supervision: jax.Array,
) -> AtomicCompositionTargets:
    mask = (
        available
        & strict_supervision
        & dual_supervision
        & (jnp.sum(weights, axis=-1) > 0.0)
    )
    normalized = weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1e-8)
    return AtomicCompositionTargets(
        weights=normalized,
        confidence=jnp.where(mask, confidence, 0.0),
        supervision_mask=mask,
    )


def train_step(
    config: _config.TrainConfig,
    coefficient_target: str,
    tcp_twist_quantile_range: jax.Array,
    update_codes: bool,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, dict[str, jax.Array]],
) -> tuple[training_utils.TrainState, dict[str, jax.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()
    frozen_codebook = None if update_codes else model.codebook.value
    observation, actions, extra = batch
    def loss_fn(model, loss_rng):
        atomic_weights = extra["atomic_weights"]
        atomic_mask = extra["atomic_supervision_mask"]
        atomic_count = jnp.sum(atomic_weights > 0.0, axis=-1)
        stay_mask = atomic_mask & (atomic_weights[..., 12] > 0.0)
        single_motion_mask = atomic_mask & (atomic_count == 1) & ~stay_mask
        dual_motion_mask = atomic_mask & (atomic_count == 2)
        # Match π0.5 relative-action quantile normalization: a zero relative
        # TCP delta stays zero, and q99-q01 maps to a scale of two. This is a
        # data statistic, not the FK atomic-gate threshold.
        if coefficient_target == "tcp_twist":
            coefficient_delta = (
                2.0 * extra["tcp_twist_delta"]
                / tcp_twist_quantile_range.astype(extra["tcp_twist_delta"].dtype)[None, None, :]
            )
        elif coefficient_target == "joint_delta":
            # The text loader has already applied the exact OpenPI sequence:
            # DeltaActions -> quantile Normalize. Do not normalize twice.
            coefficient_delta = extra["joint_delta"]
        else:
            raise ValueError(f"unsupported coefficient target {coefficient_target!r}")
        output = model.compute_text_stage_loss(
            loss_rng,
            observation,
            actions=actions,
            atomic_targets=_atomic_targets(
                extra["atomic_weights"], extra["atomic_supervision_mask"]
            ),
            atomic_composition_targets=_dual_composition_targets(
                extra["atomic_composition_weights"],
                extra["atomic_composition_confidence"],
                extra["atomic_composition_mask"],
                atomic_mask,
                dual_motion_mask,
            ),
            coefficient_tcp_twist_delta=coefficient_delta,
            coefficient_arm_mask=atomic_mask,
            global_step=state.step,
            update_codes=update_codes,
            train=True,
            return_output=True,
        )
        codes = model.codebook.value / jnp.maximum(
            jnp.linalg.norm(model.codebook.value, axis=-1, keepdims=True), 1e-8
        )
        similarities = jnp.einsum("bad,aed->bae", output.direction, codes)
        positive_mask = (atomic_weights > 0.0) & atomic_mask[..., None]
        negative_mask = ~positive_mask
        weakest_positive = jnp.min(
            jnp.where(positive_mask, similarities, jnp.inf), axis=-1
        )
        strongest_negative = jnp.max(
            jnp.where(negative_mask, similarities, -jnp.inf), axis=-1
        )
        strongest_negative = jnp.where(atomic_mask, strongest_negative, 0.0)
        margin_values = jnp.where(
            atomic_mask, weakest_positive - strongest_negative, 0.0
        )
        supervised_count = jnp.maximum(jnp.sum(atomic_mask), 1)
        target_cosine = jnp.sum(
            jnp.sum(similarities * atomic_weights, axis=-1) * atomic_mask
        ) / supervised_count
        negative_cosine = jnp.sum(strongest_negative * atomic_mask) / supervised_count
        margin = jnp.sum(margin_values) / supervised_count

        def arm_mean(value: jax.Array, arm: int) -> jax.Array:
            arm_count = jnp.maximum(jnp.sum(atomic_mask[:, arm]), 1)
            return jnp.sum(value[:, arm] * atomic_mask[:, arm]) / arm_count

        return output.total_loss, {
            "flow_loss": output.flow_loss,
            "flow_unweighted_loss": output.flow_unweighted_loss,
            "flow_left_loss": output.flow_left_loss,
            "flow_right_loss": output.flow_right_loss,
            "flow_active_loss": output.flow_active_loss,
            "flow_left_motion_share": output.flow_left_motion_share,
            "flow_right_motion_share": output.flow_right_motion_share,
            "coefficient_loss": output.coefficient_loss,
            "coefficient_velocity_loss": output.coefficient_losses.velocity,
            "coefficient_wall_loss": output.coefficient_losses.wall,
            "coefficient_wall_weight": output.coefficient_losses.wall_weight,
            "coefficient_right_loss": output.coefficient_losses.right,
            "coefficient_left_loss": output.coefficient_losses.left,
            "text_samplewise_two_way_ranking_loss": output.atomic_losses.ranking,
            "text_ratio_kl_loss": output.atomic_losses.ratio_kl,
            "text_weighted_ratio_kl_loss": (
                model.config.atomic_ratio_loss_weight * output.atomic_losses.ratio_kl
            ),
            "text_codebook_loss": output.atomic_losses.codebook,
            "text_atomic_total_loss": output.atomic_total_loss,
            "text_projection_loss": output.projection_loss,
            "text_weighted_projection_loss": (
                model.config.atomic_composition_loss_weight * output.projection_loss
            ),
            "perplexity": output.atomic_losses.perplexity,
            "right_perplexity": output.atomic_losses.right_perplexity,
            "left_perplexity": output.atomic_losses.left_perplexity,
            "atomic_fraction": jnp.mean(
                atomic_mask.astype(jnp.float32)
            ),
            "single_motion_fraction": jnp.mean(single_motion_mask.astype(jnp.float32)),
            "dual_motion_fraction": jnp.mean(dual_motion_mask.astype(jnp.float32)),
            "stay_fraction": jnp.mean(stay_mask.astype(jnp.float32)),
            "right_atomic_fraction": jnp.mean(
                atomic_mask[:, 0].astype(jnp.float32)
            ),
            "left_atomic_fraction": jnp.mean(
                atomic_mask[:, 1].astype(jnp.float32)
            ),
            "right_stay_fraction": jnp.mean(stay_mask[:, 0].astype(jnp.float32)),
            "left_stay_fraction": jnp.mean(stay_mask[:, 1].astype(jnp.float32)),
            "codebook_frozen": jnp.asarray(not update_codes, dtype=jnp.float32),
            "target_code_cosine": target_cosine,
            "strongest_negative_cosine": negative_cosine,
            "target_negative_margin": margin,
            "right_target_negative_margin": arm_mean(
                margin_values, 0
            ),
            "left_target_negative_margin": arm_mean(
                margin_values, 1
            ),
        }

    train_rng = jax.random.fold_in(rng, state.step)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model, train_rng)
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    nnx.update(model, optax.apply_updates(params, updates))
    if frozen_codebook is not None:
        # Stop-gradient prevents loss updates; restoring additionally blocks
        # optimizer weight decay/momentum from moving the frozen coordinates.
        model.codebook.value = frozen_codebook
    new_state = dataclasses.replace(
        state, step=state.step + 1, params=nnx.state(model), opt_state=new_opt_state
    )
    info |= {"loss": loss, "grad_norm": optax.global_norm(grads)}
    return new_state, info


def _load_tcp_twist_quantile_range(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"TCP norm file not found: {path}; run scripts/compute_tcp_twist_norm_fast.py first"
        )
    payload = json.loads(path.read_text())
    actual_sidecar = payload.get("_meta", {}).get("sidecar")
    if actual_sidecar != BIMANUAL_TCP_POSE_SIDECAR:
        raise ValueError(
            f"{path}: expected tcp200 norm from {BIMANUAL_TCP_POSE_SIDECAR}, "
            f"got {actual_sidecar!r}"
        )
    try:
        stats = payload["norm_stats"]["tcp_twist_delta"]
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid TCP norm file: {path}") from error
    if q01.shape != (12,) or q99.shape != (12,):
        raise ValueError(f"{path}: expected bimanual 12D q01/q99, got {q01.shape}/{q99.shape}")
    scale = q99 - q01
    if not np.all(np.isfinite(scale)) or np.any(scale <= 1e-6):
        raise ValueError(f"{path}: invalid q99-q01 TCP scales {scale.tolist()}")
    return scale


def _to_jax_batch(
    batch_np: dict[str, np.ndarray], data_sharding: jax.sharding.NamedSharding
) -> tuple[_model.Observation, jax.Array, dict[str, jax.Array]]:
    to_jax = lambda value: jax.make_array_from_process_local_data(data_sharding, value)
    observation = _model.Observation(
        images={},
        image_masks={},
        state=to_jax(batch_np["state"]),
        tokenized_prompt=to_jax(batch_np["atomic_prompt_tokens"]),
        tokenized_prompt_mask=to_jax(batch_np["atomic_prompt_mask"]),
        token_ar_mask=None,
        token_loss_mask=None,
    )
    actions = to_jax(batch_np["actions"])
    keys = (
        "atomic_prompt_tokens",
        "atomic_prompt_mask",
        "atomic_weights",
        "atomic_supervision_mask",
        "atomic_composition_weights",
        "atomic_composition_confidence",
        "atomic_composition_mask",
        "tcp_twist_delta",
        "joint_delta",
    )
    extra = {key: jax.make_array_from_process_local_data(data_sharding, batch_np[key]) for key in keys}
    return observation, actions, extra


def main() -> None:
    args = parse_args()
    if args.resume and (args.resume_trainable is not None or args.initial_step != 0):
        raise ValueError("--resume is mutually exclusive with --resume-trainable/--initial-step")
    if args.batch_size % args.devices:
        raise ValueError("batch-size must be divisible by devices")
    if args.max_token_len < 200:
        raise ValueError(
            "original PI0.5 Marvin tokenization serializes the full padded 32-D state; "
            "--max-token-len must be at least 200"
        )
    fsdp_devices = args.devices if args.fsdp_devices is None else args.fsdp_devices
    if fsdp_devices <= 0 or args.devices % fsdp_devices:
        raise ValueError("fsdp-devices must be positive and divide devices")
    if min(
        args.flow_loss_weight,
        args.coefficient_loss_weight,
        args.atomic_loss_weight,
        args.atomic_projection_loss_weight,
    ) < 0:
        raise ValueError("flow/coefficient/atomic/projection loss weights must be non-negative")
    if not 0 <= args.codebook_freeze_step <= args.steps:
        raise ValueError("codebook-freeze-step must lie in [0, steps]")
    roots = tuple(args.dataset_root) if args.dataset_root else DEFAULT_ROOTS
    if missing := [root for root in roots if not Path(root, "meta", "info.json").is_file()]:
        raise FileNotFoundError(f"missing dataset roots: {missing}")
    if not Path(args.norm_assets_dir, args.norm_asset_id, "norm_stats.json").is_file():
        raise FileNotFoundError("norm_stats.json not found under norm-assets-dir/norm-asset-id")
    if not Path(args.base_params).is_dir():
        raise FileNotFoundError(f"π0.5 base params not found: {args.base_params}")
    tcp_twist_quantile_range = (
        _load_tcp_twist_quantile_range(args.tcp_twist_norm)
        if args.coefficient_target == "tcp_twist"
        else np.ones(14, dtype=np.float32)
    )

    # The local π0.5 vendor exposes the standard Marvin/JAX recipe under this
    # name. Dataset, model, optimizer schedule, checkpointing, and freezing
    # fields used by Atomic training are replaced explicitly below.
    base_config = _config.get_config("pi05_hdf5_dscrew_v3_jax")
    model_config = AtomicPi05Config(
        max_token_len=args.max_token_len,
        fast_action_ce_loss_weight=0.0,
        text_flow_loss_weight=args.flow_loss_weight,
        coefficient_loss_weight=args.coefficient_loss_weight,
        text_atomic_loss_weight=args.atomic_loss_weight,
        atomic_composition_loss_weight=args.atomic_projection_loss_weight,
        codebook_loss_weight=args.codebook_loss_weight,
        coefficient_target_kind=args.coefficient_target,
        coefficient_target_dim=12 if args.coefficient_target == "tcp_twist" else 14,
    )
    lr_schedule = dataclasses.replace(
        base_config.lr_schedule,
        warmup_steps=args.warmup_steps,
        peak_lr=args.peak_lr,
        decay_steps=args.lr_decay_steps,
        decay_lr=args.decay_lr,
    )
    config = dataclasses.replace(
        base_config,
        name=args.run_name,
        exp_name=args.run_name,
        model=model_config,
        weight_loader=AtomicPi05CheckpointLoader(
            args.base_params, trainable_params_path=args.resume_trainable
        ),
        lr_schedule=lr_schedule,
        # This is the same verified slash-path filter as joint training: the
        # released SigLIP ViT remains frozen; Q1/LLM/FAST heads may learn.
        freeze_filter=model_config.get_freeze_filter(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.smoke_steps or args.steps,
        save_interval=args.save_interval,
        checkpoint_base_dir=args.checkpoint_base_dir,
        wandb_enabled=False,
        ema_decay=None,
        fsdp_devices=fsdp_devices,
        overwrite=not args.resume,
        resume=args.resume,
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    logging.info("config=%s", config)
    logging.info(
        "global batch=%d (%d/GPU), shared text-flow weight=%.3f, "
        "atomic weight=%.3f, dual projection weight=%.3f, "
        "DCT weight=%.3f, fsdp_devices=%d",
        args.batch_size,
        args.batch_size // args.devices,
        args.flow_loss_weight,
        args.atomic_loss_weight,
        args.atomic_projection_loss_weight,
        args.coefficient_loss_weight,
        fsdp_devices,
    )
    if args.coefficient_target == "tcp_twist":
        logging.info(
            "TCP relative-delta quantile range q99-q01=%s; normalization=2*delta/range",
            tcp_twist_quantile_range.tolist(),
        )
    else:
        logging.info(
            "joint coefficient target uses OpenPI-normalized right7+left7 relative actions"
        )

    loader = build_atomic_text_loader(
        roots,
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        atomic_composition_sidecar=args.atomic_composition_sidecar,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        action_horizon=model_config.action_horizon,
        max_token_len=model_config.max_token_len,
        include_fast=False,
        reliable_atomic_only=True,
    )
    data_iter = iter(loader)
    first_np = next(data_iter)
    first_weights = first_np["atomic_weights"]
    first_mask = first_np["atomic_supervision_mask"]
    first_count = (first_weights > 0.0).sum(axis=-1)
    first_stay = first_mask & (first_weights[..., 12] > 0.0)
    if not np.all(np.any(first_mask, axis=1)):
        raise AssertionError("reliable-only zT loader emitted an unlabeled row")
    logging.info(
        "reliable-only zT rows=%d; first batch state=%s atomic=%.3f "
        "single=%.3f dual=%.3f stay=%.3f right_dct=%.3f left_dct=%.3f",
        len(loader.dataset),
        first_np["state"].shape,
        float(first_mask.mean()),
        float((first_mask & (first_count == 1) & ~first_stay).mean()),
        float((first_mask & (first_count == 2)).mean()),
        float(first_stay.mean()),
        float(first_mask[:, 0].mean()),
        float(first_mask[:, 1].mean()),
    )
    # Create the no-video text loader before touching JAX's GPU backend.
    if args.devices > jax.device_count():
        raise ValueError(f"requested {args.devices} devices but only {jax.device_count()} are visible")

    base_train = _base_train_module()
    rng = jax.random.key(args.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    tcp_twist_quantile_range_jax = jax.device_put(tcp_twist_quantile_range, replicated)
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    state, state_sharding = base_train.init_train_state(config, init_rng, mesh, resume=resuming)
    initialized_model = nnx.merge(state.model_def, state.params)
    expected_codebook_shape = (
        model_config.arm_count,
        model_config.num_atomic_codes,
        model_config.latent_dim,
    )
    if initialized_model.codebook.value.shape != expected_codebook_shape:
        raise ValueError(
            f"expected arm-specific codebook {expected_codebook_shape}, "
            f"got {initialized_model.codebook.value.shape}"
        )
    logging.info("verified atomic codebook shape=%s", expected_codebook_shape)
    del initialized_model
    # The Stage-A PyTorch loader has no OpenPI DataLoader wrapper. OpenPI's
    # checkpoint helper uses data_config only to copy norm assets; those are
    # already immutable external inputs for this run, so a no-assets view is
    # sufficient. Model params, optimizer state and global step are still
    # saved and restored by the native checkpoint implementation.
    checkpoint_loader_view = SimpleNamespace(
        data_config=lambda: SimpleNamespace(norm_stats=None, asset_id=None)
    )
    if resuming:
        state = _checkpoints.restore_state(checkpoint_manager, state, checkpoint_loader_view)
    elif args.initial_step:
        state = dataclasses.replace(
            state, step=jnp.asarray(args.initial_step, dtype=state.step.dtype)
        )
    jax.block_until_ready(state)
    ptrain_update_codes = jax.jit(
        functools.partial(
            train_step, config, args.coefficient_target, tcp_twist_quantile_range_jax, True
        ),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    ptrain_frozen_codes = jax.jit(
        functools.partial(
            train_step, config, args.coefficient_target, tcp_twist_quantile_range_jax, False
        ),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    logging.info(
        "initialized text/state zT/shared-PI0.5-flow Stage A on %d devices",
        args.devices,
    )

    current_np = first_np
    final_step = args.initial_step + config.num_train_steps
    remaining_steps = max(final_step - int(state.step), 0)
    for _ in range(remaining_steps):
        ptrain = (
            ptrain_update_codes
            if int(state.step) < args.codebook_freeze_step
            else ptrain_frozen_codes
        )
        state, info = ptrain(train_rng, state, _to_jax_batch(current_np, data_sharding))
        step = int(state.step)
        if step % args.log_interval == 0 or step == 1:
            values = {key: float(value) for key, value in jax.device_get(info).items()}
            logging.info("step=%d %s", step, " ".join(f"{key}={value:.5f}" for key, value in values.items()))
        if not args.skip_checkpoint and (step % config.save_interval == 0 or step == final_step):
            _checkpoints.save_state(checkpoint_manager, state, checkpoint_loader_view, step)
            logging.info("submitted full OpenPI checkpoint step=%d", step)
        try:
            current_np = next(data_iter)
        except StopIteration:
            # A finite LeRobot epoch is not a training terminal condition.
            # Rebuild the shuffled iterator and keep the global JAX step.
            data_iter = iter(loader)
            current_np = next(data_iter)
    if not args.skip_checkpoint:
        logging.info("waiting for full checkpoint writes to finish")
        checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main()
