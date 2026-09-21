#!/usr/bin/env python3
"""Train the force-reactive B2 policy from a completed B1 parameter checkpoint."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
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
from openpi.shared import nnx_utils
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import sharding
from openpi.training import utils as training_utils

from atomic_latent_vla.pi05.config import AtomicPi05Config
from atomic_latent_vla.pi05.force_training_data import (
    batch_to_force_inputs,
    build_force_train_validation_loaders,
)
from atomic_latent_vla.pi05.weights import (
    AtomicPi05CheckpointLoader,
    AtomicPi05ForceOverlayCheckpointLoader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", action="append", required=True)
    parser.add_argument(
        "--subtask-sidecar",
        action="append",
        help=(
            "aligned episode_subtasks.jsonl for the corresponding --dataset-root; "
            "repeat once per root. Episodes absent from a sidecar are excluded."
        ),
    )
    parser.add_argument(
        "--pad-subtask-horizon",
        action="store_true",
        help=(
            "match Atomic ZM training: when the current subtask ends inside the "
            "50-step horizon, repeat that subtask's final absolute action"
        ),
    )
    parser.add_argument(
        "--truncate-to-sidecar-end",
        action="store_true",
        help=(
            "treat each sidecar's final semantic-segment end as the effective "
            "episode boundary; required for explicitly truncated task contracts"
        ),
    )
    parser.add_argument("--norm-assets-dir", required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--force-norm", required=True)
    parser.add_argument("--b1-params", required=True)
    parser.add_argument(
        "--force-init-params",
        help="optional checkpoint whose force_conditioner subtree overlays the clean B1/base tree",
    )
    parser.add_argument("--checkpoint-base-dir", default="/mnt/cunchu/zjq/atomic_pi05_force_runs")
    parser.add_argument("--run-name", default=time.strftime("force_stage_b2_%Y%m%d_%H%M%S"))
    parser.add_argument(
        "--coefficient-target", choices=("tcp_twist", "joint_delta"), default="joint_delta"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--max-token-len", type=int, default=200)
    parser.add_argument("--encoder-width", type=int, default=256)
    parser.add_argument("--encoder-depth", type=int, default=2)
    parser.add_argument("--encoder-heads", type=int, default=4)
    parser.add_argument("--encoder-mlp-dim", type=int, default=1024)
    parser.add_argument("--force-latent-dim", type=int, default=256)
    parser.add_argument(
        "--force-only-zf",
        action="store_true",
        help="build zF only from slow force/state; must match the B1 architecture",
    )
    parser.add_argument(
        "--future-condition-on-zm",
        action="store_true",
        help="condition only the training-time future-force decoder on stop-gradient zM",
    )
    parser.add_argument("--steps", type=int, default=6_000)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--peak-lr", type=float, default=3.0e-5)
    parser.add_argument("--decay-steps", type=int, default=6_000)
    parser.add_argument("--decay-lr", type=float, default=3.0e-6)
    parser.add_argument("--future-force-loss-weight", type=float, default=0.0)
    parser.add_argument("--flow-loss-weight", type=float, default=1.0)
    parser.add_argument("--delta-regularization-weight", type=float, default=1.0e-4)
    parser.add_argument("--force-improvement-weight", type=float, default=0.0)
    parser.add_argument("--force-improvement-margin", type=float, default=0.0)
    parser.add_argument("--force-hidden-cross-attention", action="store_true")
    parser.add_argument("--force-hidden-cross-attention-heads", type=int, default=2)
    parser.add_argument(
        "--layerwise-atomic-flow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "use depth-specific zM values; pass --no-layerwise-atomic-flow for "
            "one final zM pair reused by all 18 Action-Expert blocks"
        ),
    )
    parser.add_argument(
        "--full-token-force-adapter",
        action="store_true",
        help=(
            "freeze completed B1 and the whole VLA; train a per-arm zM+phase query "
            "over 30 SlowProj + 10 FastProj tokens to produce zero-init dZ_M"
        ),
    )
    parser.add_argument("--full-token-force-adapter-heads", type=int, default=2)
    parser.add_argument(
        "--spherical-visual-latent",
        action="store_true",
        help="match a spherical zM base checkpoint",
    )
    parser.add_argument("--visual-max-update-angle-deg", type=float, default=45.0)
    parser.add_argument(
        "--spherical-force-update",
        action="store_true",
        help="project direct force dZ onto every zM tangent plane and sphere-map it",
    )
    parser.add_argument("--force-max-update-angle-deg", type=float, default=15.0)
    parser.add_argument("--force-rotation-loss-weight", type=float, default=0.0)
    parser.add_argument("--force-rotation-free-angle-deg", type=float, default=0.0)
    parser.add_argument("--save-interval", type=int, default=1_000)
    parser.add_argument("--keep-period", type=int, default=5_000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-modulus", type=int, default=10)
    parser.add_argument("--validation-remainder", type=int, default=0)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument(
        "--force-update-action-steps",
        type=int,
        default=10,
        help="fast-history span in action steps; 20 consumes at most 80 new 120 Hz samples",
    )
    parser.add_argument(
        "--force-update-offsets",
        default="0,10,20,30,40",
        help="comma-separated fixed-horizon RTC offsets; production uses 0,10,20,30,40",
    )
    parser.add_argument(
        "--freeze-action-path",
        action="store_true",
        help="train force conditioning only; freeze future decoder and Action Expert path",
    )
    parser.add_argument(
        "--unfreeze-action-path",
        action="store_true",
        help=(
            "with --full-token-force-adapter, also train the Action Expert and "
            "action/time projections for Training-Time RTC suffix continuation"
        ),
    )
    parser.add_argument(
        "--unfreeze-zf-path",
        action="store_true",
        help=(
            "with --full-token-force-adapter, train the complete slow/fast force "
            "encoder and zF path while keeping the future-force decoder frozen"
        ),
    )
    parser.add_argument("--skip-checkpoint", action="store_true")
    return parser.parse_args()


def _base_train_module():
    train_path = Path(__file__).resolve().parents[1] / "vendor" / "pi0.5" / "scripts" / "train.py"
    if not train_path.is_file():
        raise FileNotFoundError(f"Marvin/OpenPI trainer not found: {train_path}")
    import importlib.util

    spec = importlib.util.spec_from_file_location("atomic_force_b2_base_train", train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import OpenPI trainer from {train_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_update_offsets(value: str, horizon: int) -> tuple[int, ...]:
    offsets = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not offsets:
        raise ValueError("--force-update-offsets cannot be empty")
    if tuple(sorted(set(offsets))) != offsets:
        raise ValueError("--force-update-offsets must be unique and sorted")
    if offsets[0] < 0 or offsets[-1] >= horizon:
        raise ValueError("--force-update-offsets must lie in [0, action_horizon)")
    return offsets


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_jax_batch(batch_np: dict[str, object], data_sharding: jax.sharding.NamedSharding):
    observation, actions, force = batch_to_force_inputs(batch_np)
    observation = jax.tree.map(
        lambda value: jax.make_array_from_process_local_data(data_sharding, value), observation
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
            train_fast=True,
            train=True,
            return_output=True,
        )
        return output.total_loss, {
            "loss": output.total_loss,
            "flow_loss": output.flow_loss,
            "future_force_loss": output.future_force_loss,
            "future_force_raw_loss": output.future_force_raw_loss,
            "future_force_coarse_loss": output.future_force_coarse_loss,
            "delta_z_regularization": output.delta_z_regularization,
            "base_flow_loss": output.base_flow_loss,
            "improvement_loss": output.improvement_loss,
            "force_rotation_loss": output.force_rotation_loss,
            "force_rotation_angle_deg": jnp.rad2deg(
                output.force_rotation_mean_angle_rad
            ),
            "force_flow_gain": output.base_flow_loss - output.flow_loss,
            "delta_z_norm": jnp.mean(jnp.linalg.norm(output.delta_z, axis=-1)),
            "force_latent_norm": jnp.mean(jnp.linalg.norm(output.force_latent, axis=-1)),
            "layer_gate_mean": jnp.mean(model._require_force_conditioner().layer_gates()),  # noqa: SLF001
            "offset0_fraction": jnp.mean((force["update_offset"] == 0).astype(jnp.float32)),
        }

    train_rng = jax.random.fold_in(rng, state.step)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng
    )
    adapter_out_filter = nnx_utils.PathRegex(
        r".*(?:force_cross_out|full_token_adapter_out)(?:/.*)?"
    )
    adapter_out_grad_norm = optax.global_norm(grads.filter(adapter_out_filter))
    atomic_adapter_grad_norm = optax.global_norm(
        grads.filter(nnx_utils.PathRegex(r".*atomic_adapter(?:/.*)?"))
    )
    atomic_up_grad_norm = optax.global_norm(
        grads.filter(
            nnx_utils.PathRegex(r".*atomic_adapter/atomic_up(?:/.*)?")
        )
    )
    differentiable_params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, differentiable_params)
    nnx.update(model, optax.apply_updates(differentiable_params, updates))
    updated_params = nnx.state(model)
    adapter_out_param_norm = optax.global_norm(
        updated_params.filter(adapter_out_filter)
    )
    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=updated_params,
        opt_state=new_opt_state,
    )
    return new_state, info | {
        "grad_norm": optax.global_norm(grads),
        "adapter_out_grad_norm": adapter_out_grad_norm,
        "adapter_out_param_norm": adapter_out_param_norm,
        "atomic_adapter_grad_norm": atomic_adapter_grad_norm,
        "atomic_up_grad_norm": atomic_up_grad_norm,
        "objective": loss,
    }


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
        train_fast=True,
        train=False,
        return_output=True,
    )
    # Offset zero is the deployment-critical initial full-chunk prediction.  It
    # has no post-anchor fast samples and therefore isolates z_F conditioning.
    offset0_force = dict(force)
    offset0_force["update_offset"] = jnp.zeros_like(force["update_offset"])
    offset0_force["current_history_mask"] = jnp.zeros_like(force["current_history_mask"])
    offset0 = model.compute_force_stage_loss(
        jax.random.fold_in(rng, 1),
        observation,
        actions,
        **offset0_force,
        train_fast=True,
        train=False,
        return_output=True,
    )
    return {
        "total_loss": output.total_loss,
        "flow_loss": output.flow_loss,
        "offset0_flow_loss": offset0.flow_loss,
        "future_force_loss": output.future_force_loss,
        "delta_z_norm": jnp.mean(jnp.linalg.norm(output.delta_z, axis=-1)),
        "offset0_delta_z_norm": jnp.mean(jnp.linalg.norm(offset0.delta_z, axis=-1)),
        "layer_gate_mean": jnp.mean(model._require_force_conditioner().layer_gates()),  # noqa: SLF001
        "force_rotation_loss": output.force_rotation_loss,
        "force_rotation_angle_deg": jnp.rad2deg(
            output.force_rotation_mean_angle_rad
        ),
    }


def _write_contract(
    args: argparse.Namespace, config: _config.TrainConfig, roots: tuple[str, ...]
) -> None:
    directory = Path(config.checkpoint_dir) / "run_contract"
    directory.mkdir(parents=True, exist_ok=True)
    force_norm = Path(args.force_norm).resolve()
    base_norm = Path(args.norm_assets_dir, args.norm_asset_id, "norm_stats.json").resolve()
    source_root = Path(__file__).resolve().parents[1]
    files = {
        "force.py": source_root / "src/atomic_latent_vla/pi05/force.py",
        "model.py": source_root / "src/atomic_latent_vla/pi05/model.py",
        "gemma_adapter.py": source_root / "src/atomic_latent_vla/pi05/gemma_adapter.py",
        "config.py": source_root / "src/atomic_latent_vla/pi05/config.py",
        "force_training_data.py": source_root / "src/atomic_latent_vla/pi05/force_training_data.py",
        "train_force_stage_b2.py": Path(__file__).resolve(),
        "force_norm": force_norm,
        "base_norm": base_norm,
        "b1_metadata": Path(args.b1_params).resolve() / "_METADATA",
    }
    sidecars = tuple(Path(path).resolve() for path in (args.subtask_sidecar or ()))
    for index, sidecar in enumerate(sidecars):
        files[f"subtask_sidecar_{index}"] = sidecar
    update_offsets = _parse_update_offsets(args.force_update_offsets, 50)
    contract = {
        "stage": "B2 force-conditioned action flow",
        "datasets": list(roots),
        "b1_params": str(Path(args.b1_params).resolve()),
        "force_init_params": (
            str(Path(args.force_init_params).resolve()) if args.force_init_params else None
        ),
        "merge_contract": (
            "B1 params is a complete matching base-policy parameter tree; restore it once, "
            "including trained slow force tensors and untouched zero-init fast tensors; "
            "start a fresh B2 Adam state for the expanded trainable set"
        ),
        "prompt": {
            "route": (
                "current_subtask_at_anchor" if sidecars else "PromptFromLeRobotTask"
            ),
            "source": (
                [str(path) for path in sidecars]
                if sidecars
                else "each dataset meta/tasks.parquet task string"
            ),
            "mixing": "none",
            "horizon_composition": (
                "hold anchor frame current_subtask for the entire 50-step target; "
                "repeat its final absolute action after the boundary; no concatenation"
                if sidecars and args.pad_subtask_horizon
                else "current subtask prompt with the unmodified dataset action horizon"
                if sidecars
                else "dataset task prompt"
            ),
            "missing_episode_policy": (
                "exclude every episode absent from its sidecar"
                if sidecars
                else "not applicable"
            ),
            "truncate_to_sidecar_end": args.truncate_to_sidecar_end,
            "max_token_length": args.max_token_len,
        },
        "normalization": {
            "base_norm": str(base_norm),
            "force_norm": str(force_norm),
            "quantile_mode": "q01/q99 affine",
            "force_state_separate": True,
            "state_adapt_to_pi": True,
            "state_clip": False,
            "action_clip": False,
            "inference_output_clip": False,
            "load_compensation": False,
            "static_sensor_bias_already_removed": True,
        },
        "objective": {
            "train_fast": True,
            "flow_loss_weight": args.flow_loss_weight,
            "future_force_loss_weight": args.future_force_loss_weight,
            "delta_z_regularization_weight": args.delta_regularization_weight,
            "force_improvement_weight": args.force_improvement_weight,
            "force_improvement_margin": args.force_improvement_margin,
            "force_rotation_loss_weight": args.force_rotation_loss_weight,
            "force_rotation_free_angle_deg": args.force_rotation_free_angle_deg,
            "force_max_update_angle_deg": args.force_max_update_angle_deg,
            "force_hidden_cross_attention": args.force_hidden_cross_attention,
            "force_hidden_cross_attention_heads": args.force_hidden_cross_attention_heads,
            "full_token_force_adapter": args.full_token_force_adapter,
            "full_token_force_adapter_heads": args.full_token_force_adapter_heads,
            "force_memory": (
                "per-arm token-axis concat: 30 SlowProj + 10 FastProj; "
                "Query=Proj(zM)+sincos(update_offset); output=zero-init dZ_M"
                if args.full_token_force_adapter
                else "legacy compressed zF + fast route"
            ),
            "offsets": list(update_offsets),
            "offset0": "z_M queries z_F with every fast token masked; full 0:50 flow target",
            "later_offsets": "z_M queries z_F plus acquired fast tokens; committed prefix masked",
            "suffix_weight": 1.0,
            "committed_prefix_weight": 0.0,
            "rtc_role_signal": "per-token flow time only; no learned committed embedding",
        },
        "data_loading": {
            "load_future_force_targets": args.future_force_loss_weight > 0,
            "future_force_disabled_placeholder": "masked [2,1,6] neutral tensor",
        },
        "trainable": (
            (
                (
                    "complete slow/fast force encoder and zF path (future decoder excluded) + "
                    "full-token 2-head adapter + dZ_M output + layer gates + "
                    "Action Expert + action/time projections"
                    if args.unfreeze_zf_path
                    else "FastProj + full-token 2-head adapter + dZ_M output + layer gates + "
                    "Action Expert + action/time projections"
                )
                if args.unfreeze_action_path
                else (
                    "complete slow/fast force encoder and zF path (future decoder excluded)"
                    if args.unfreeze_zf_path
                    else "FastProj + full-token 2-head adapter + dZ_M output + layer gates only"
                )
            )
            if args.full_token_force_adapter
            else (
                "force_conditioner excluding future decoder"
                if args.freeze_action_path
                else (
                    "force_conditioner including future decoder, Action Expert, action/time projections, atomic adapters"
                    if args.future_force_loss_weight > 0
                    else "force_conditioner excluding future decoder, Action Expert, action/time projections, atomic adapters"
                )
            )
        ),
        "frozen": (
            (
                (
                    "future-force decoder; SigLIP/VLM; AFRO atomic adapters; Q heads; codebook"
                    if args.unfreeze_zf_path
                    else "completed B1 encoder/SlowProj/zF/future decoder; SigLIP/VLM; "
                    "AFRO atomic adapters; Q heads; codebook"
                )
                if args.unfreeze_action_path
                else "completed B1 encoder/SlowProj/zF/future decoder; VLM; Action Expert; "
                "action/time projections; AFRO atomic adapters; Q heads; codebook"
            )
            if args.full_token_force_adapter
            else (
                "VLA, Action Expert, action/time projections, atomic adapters, Q heads, codebook, future-force decoder"
                if args.freeze_action_path
                else (
                    "SigLIP, VLM stream, Q heads, codebook"
                    if args.future_force_loss_weight > 0
                    else "SigLIP, VLM stream, Q heads, codebook, future-force decoder"
                )
            )
        ),
        "optimizer": {
            "kind": type(config.optimizer).__name__,
            "gradient_clip_norm": getattr(config.optimizer, "clip_gradient_norm", None),
            "fresh_state": True,
        },
        "batch_size": args.batch_size,
        "devices": args.devices,
        "fsdp_devices": args.fsdp_devices,
        "mesh": [args.devices // args.fsdp_devices, args.fsdp_devices],
        "schedule": {
            "steps": args.steps,
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
    if args.max_token_len < 192:
        raise ValueError("full-32-D PI0.5 state tokenization requires max-token-len >= 192")
    roots = tuple(args.dataset_root)
    sidecars = tuple(args.subtask_sidecar or ())
    if sidecars and len(sidecars) != len(roots):
        raise ValueError("repeat --subtask-sidecar exactly once per --dataset-root")
    if args.batch_size % args.devices:
        raise ValueError("batch size must be divisible by devices")
    if args.fsdp_devices <= 0 or args.devices % args.fsdp_devices:
        raise ValueError("fsdp-devices must be positive and divide devices")
    if args.steps <= 0 or args.eval_batches <= 0:
        raise ValueError("steps and eval-batches must be positive")
    if args.future_force_loss_weight < 0:
        raise ValueError("--future-force-loss-weight must be non-negative")
    if args.force_improvement_weight < 0 or args.force_improvement_margin < 0:
        raise ValueError("force improvement weight and margin must be non-negative")
    if args.force_rotation_loss_weight < 0:
        raise ValueError("force rotation loss weight must be non-negative")
    if not 0 <= args.force_rotation_free_angle_deg < args.force_max_update_angle_deg:
        raise ValueError(
            "force rotation free angle must lie in [0, force max update angle)"
        )
    if args.force_hidden_cross_attention_heads <= 0:
        raise ValueError("force hidden cross-attention heads must be positive")
    if args.full_token_force_adapter_heads <= 0:
        raise ValueError("full-token force adapter heads must be positive")
    if args.full_token_force_adapter and args.force_hidden_cross_attention:
        raise ValueError(
            "full-token dZ_M adapter and Action-Expert hidden cross-attention are mutually exclusive"
        )
    if args.unfreeze_action_path and not args.full_token_force_adapter:
        raise ValueError("--unfreeze-action-path requires --full-token-force-adapter")
    if args.unfreeze_zf_path and not args.full_token_force_adapter:
        raise ValueError("--unfreeze-zf-path requires --full-token-force-adapter")
    if args.unfreeze_action_path and args.freeze_action_path:
        raise ValueError("--unfreeze-action-path and --freeze-action-path are mutually exclusive")
    if args.full_token_force_adapter and args.future_force_loss_weight > 0:
        raise ValueError(
            "full-token B2 freezes the completed B1/future decoder; set future-force loss to zero"
        )
    if args.freeze_action_path and args.future_force_loss_weight > 0:
        raise ValueError("future-force loss requires the future decoder to remain trainable")
    for root in roots:
        if not Path(root, "meta", "info.json").is_file():
            raise FileNotFoundError(f"force dataset not found: {root}")
    for sidecar in sidecars:
        if not Path(sidecar).is_file():
            raise FileNotFoundError(sidecar)
    for path in (
        Path(args.force_norm),
        Path(args.norm_assets_dir, args.norm_asset_id, "norm_stats.json"),
        Path(args.b1_params, "_METADATA"),
        Path(args.b1_params, "manifest.ocdbt"),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.force_init_params:
        for name in ("_METADATA", "manifest.ocdbt"):
            path = Path(args.force_init_params, name)
            if not path.is_file():
                raise FileNotFoundError(path)

    update_offsets = _parse_update_offsets(args.force_update_offsets, 50)
    load_future_force_targets = args.future_force_loss_weight > 0
    loader, validation_loader = build_force_train_validation_loaders(
        roots,
        subtask_sidecars=sidecars or None,
        pad_subtask_horizon=args.pad_subtask_horizon,
        truncate_to_sidecar_end=args.truncate_to_sidecar_end,
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        force_norm_path=args.force_norm,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_token_len=args.max_token_len,
        validation_modulus=args.validation_modulus,
        validation_remainder=args.validation_remainder,
        force_update_action_steps=args.force_update_action_steps,
        force_update_offsets=update_offsets,
        load_future_force_targets=load_future_force_targets,
    )
    data_iter = iter(loader)
    first_np = next(data_iter)
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
        force_fast_history_samples=args.force_update_action_steps * 4,
        force_update_action_steps=args.force_update_action_steps,
        force_future_decoder_stride=4,
        force_future_decoder_kind="phase_mlp",
        force_encoder_width=args.encoder_width,
        force_encoder_depth=args.encoder_depth,
        force_encoder_num_heads=args.encoder_heads,
        force_encoder_mlp_dim=args.encoder_mlp_dim,
        force_latent_dim=args.force_latent_dim,
        force_context_from_prefix=not args.force_only_zf,
        force_future_condition_on_zm=args.future_condition_on_zm,
        force_position_base=10_000.0,
        force_history_train_lengths=(120,),
        force_future_loss_weight=args.future_force_loss_weight,
        force_flow_loss_weight=args.flow_loss_weight,
        force_delta_regularization_weight=args.delta_regularization_weight,
        force_improvement_loss_weight=args.force_improvement_weight,
        force_improvement_margin=args.force_improvement_margin,
        enable_force_hidden_cross_attention=args.force_hidden_cross_attention,
        force_hidden_cross_attention_heads=args.force_hidden_cross_attention_heads,
        enable_layerwise_atomic_flow=args.layerwise_atomic_flow,
        force_full_token_adapter=args.full_token_force_adapter,
        force_full_token_adapter_heads=args.full_token_force_adapter_heads,
        spherical_visual_latent=args.spherical_visual_latent,
        visual_max_update_angle_deg=args.visual_max_update_angle_deg,
        spherical_force_update=args.spherical_force_update,
        force_max_update_angle_deg=args.force_max_update_angle_deg,
        force_rotation_loss_weight=args.force_rotation_loss_weight,
        force_rotation_free_angle_deg=args.force_rotation_free_angle_deg,
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
        weight_loader=(
            AtomicPi05ForceOverlayCheckpointLoader(args.b1_params, args.force_init_params)
            if args.force_init_params
            else AtomicPi05CheckpointLoader(args.b1_params)
        ),
        freeze_filter=(
            model_config.get_force_full_token_adapter_freeze_filter(
                train_action_path=args.unfreeze_action_path,
                train_zf_path=args.unfreeze_zf_path,
            )
            if args.full_token_force_adapter
            else model_config.get_force_freeze_filter(
                train_atomic_adapters=True,
                train_future_decoder=args.future_force_loss_weight > 0,
            )
            if not args.freeze_action_path
            else model_config.get_force_action_only_freeze_filter()
        ),
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
        "B2 restore=%s batch=%d (%d/GPU) devices=%d fsdp=%d mesh=[%d,%d] "
        "loss=flow*%g+future*%g+delta_reg*%g+improve*%g(margin=%g)"
        "+force_rotation*%g(free=%gdeg,max=%gdeg) "
        "offsets=%s suffix_weight=1 committed_weight=0 "
        "rtc_phase_query=sincos(update_offset) future_targets_loaded=%s freeze_action_path=%s "
        "unfreeze_action_path=%s unfreeze_zf_path=%s "
        "force_hidden_cross_attention=%s heads=%d layerwise_atomic_flow=%s "
        "full_token_adapter=%s heads=%d "
        "fresh_adam=true",
        args.b1_params,
        args.batch_size,
        args.batch_size // args.devices,
        args.devices,
        args.fsdp_devices,
        args.devices // args.fsdp_devices,
        args.fsdp_devices,
        args.flow_loss_weight,
        args.future_force_loss_weight,
        args.delta_regularization_weight,
        args.force_improvement_weight,
        args.force_improvement_margin,
        args.force_rotation_loss_weight,
        args.force_rotation_free_angle_deg,
        args.force_max_update_angle_deg,
        update_offsets,
        load_future_force_targets,
        args.freeze_action_path,
        args.unfreeze_action_path,
        args.unfreeze_zf_path,
        args.force_hidden_cross_attention,
        args.force_hidden_cross_attention_heads,
        args.layerwise_atomic_flow,
        args.full_token_force_adapter,
        args.full_token_force_adapter_heads,
    )
    logging.info(
        "first batch actions=%s slow_force=%s current_force=%s future_force=%s "
        "future_mask_any=%s offsets=%s",
        first_np["actions"].shape,
        first_np["slow_force_history"].shape,
        first_np["current_force_history"].shape,
        first_np["future_force"].shape,
        bool(first_np["future_force_mask"].any()),
        sorted(set(int(value) for value in first_np["update_offset"])),
    )

    base_train = _base_train_module()
    rng = jax.random.key(args.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
    )
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
    logging.info("restored B1 parameter tree and initialized fresh B2 optimizer state")
    if args.full_token_force_adapter:
        atomic_cross_attention_params = state.params.filter(
            nnx_utils.PathRegex(r".*atomic_cross_attention(?:/.*)?")
        )
        zm_residual_proj_params = state.params.filter(
            nnx_utils.PathRegex(
                r".*atomic_cross_attention/zm_residual_proj(?:/.*)?"
            )
        )
        full_adapter_params = state.params.filter(
            nnx.All(
                nnx.Param,
                nnx.Not(
                    model_config.get_force_full_token_adapter_freeze_filter(
                        train_action_path=args.unfreeze_action_path,
                        train_zf_path=args.unfreeze_zf_path,
                    )
                ),
            )
        )
        logging.info(
            "restored parameter audit atomic_cross_attention_norm=%.9g "
            "zm_residual_proj_norm=%.9g "
            "full_token_trainable_norm=%.9g full_token_trainable_leaves=%d",
            float(optax.global_norm(atomic_cross_attention_params)),
            float(optax.global_norm(zm_residual_proj_params)),
            float(optax.global_norm(full_adapter_params)),
            len(full_adapter_params.flat_state()),
        )
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
                "step=%d %s", step, " ".join(f"{key}={value:.9g}" for key, value in values.items())
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
            jax.random.fold_in(train_rng, 1_000_000 + validation_count), state, validation_batch
        )
        values = {key: float(value) for key, value in jax.device_get(validation_info).items()}
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
