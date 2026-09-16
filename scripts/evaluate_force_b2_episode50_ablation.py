#!/usr/bin/env python3
"""Compare B2 force modulation with a same-checkpoint zero-delta ablation.

The evaluator uses one complete held-out episode and evaluates every genuine
50-frame anchor.  Both branches share the B2 backbone/Action Expert, prompt,
observation, normalized GT target and diffusion noise; only the force-produced
``delta_z`` is replaced by zero in the ablation branch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from openpi.models import model as _model

from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.force_training_data import (
    ForceNormalization,
    batch_to_force_inputs,
    build_force_dataset,
    force_collate,
)
from atomic_latent_vla.pi05.weights import AtomicPi05CheckpointLoader

import evaluate_zm_fk_trajectory_ablation as _fk_eval
from evaluate_zm_fk_trajectory_ablation import _output_transform
from evaluate_global_episode_chunks import _sample_global


OFFSETS = (0,)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config(args: argparse.Namespace | None = None) -> AtomicPi05Config:
    """Recreate the exact architecture used by ``train_force_stage_b2.py``."""

    return AtomicPi05Config(
        max_token_len=192,
        coefficient_target_kind="joint_delta",
        coefficient_target_dim=14,
        fast_action_ce_loss_weight=0.0,
        enable_force_stage=True,
        force_fast_history_samples=40,
        force_update_action_steps=10,
        force_future_decoder_stride=4,
        force_future_decoder_kind="phase_mlp",
        force_encoder_depth=2,
        force_encoder_width=getattr(args, "encoder_width", 512),
        force_encoder_num_heads=getattr(args, "encoder_heads", 8),
        force_encoder_mlp_dim=getattr(args, "encoder_mlp_dim", 1024),
        force_latent_dim=getattr(args, "force_latent_dim", 512),
        force_position_base=10_000.0,
        force_history_train_lengths=(120,),
        force_future_loss_weight=0.0,
        force_flow_loss_weight=1.0,
        force_delta_regularization_weight=1.0e-4,
        force_improvement_loss_weight=1.0,
        force_improvement_margin=0.001,
        force_stop_gradient_backbone=True,
        force_context_from_prefix=False,
        force_future_condition_on_zm=True,
        enable_force_hidden_cross_attention=getattr(
            args, "force_hidden_cross_attention", False
        ),
        force_hidden_cross_attention_heads=getattr(
            args, "force_hidden_cross_attention_heads", 2
        ),
        force_full_token_adapter=getattr(args, "full_token_force_adapter", False),
        force_full_token_adapter_heads=getattr(
            args, "full_token_force_adapter_heads", 2
        ),
        enable_layerwise_atomic_flow=False,
        spherical_visual_latent=True,
        visual_max_update_angle_deg=45.0,
        spherical_force_update=True,
        force_max_update_angle_deg=45.0,
        force_rotation_loss_weight=0.005,
        force_rotation_free_angle_deg=20.0,
    )


def _candidate_indices(
    dataset, *, require_validation_episode: bool = True
) -> dict[int, list[int]]:
    raw = dataset._raw  # noqa: SLF001
    result: dict[int, list[int]] = {}
    for dataset_index, anchor in enumerate(raw.anchors):
        episode = int(raw.anchor_episodes[dataset_index])
        frame = int(raw.base._frame_index[int(anchor)])  # noqa: SLF001
        if (require_validation_episode and episode % 10) or frame % 50:
            continue
        result.setdefault(episode, []).append(dataset_index)
    return result


def _choose_episode(candidates: dict[int, list[int]], requested: str) -> int:
    if requested != "auto":
        episode = int(requested)
        if episode not in candidates:
            raise ValueError(f"episode {episode} has no held-out complete 50-frame anchors")
        return episode
    # Choose the episode nearest the median window count. Selection is based
    # only on coverage, never on a model prediction or target error.
    counts = np.asarray([len(rows) for rows in candidates.values()])
    median = float(np.median(counts))
    return min(candidates, key=lambda episode: (abs(len(candidates[episode]) - median), episode))


def _anchor_prompt(raw, anchor: int) -> str:
    episode = int(raw.base._episode_index[anchor])  # noqa: SLF001
    frame = int(raw.base._frame_index[anchor])  # noqa: SLF001
    if raw.subtasks is not None:
        for start, end, text in raw.subtasks[episode]:
            if start <= frame < end:
                return text
        raise ValueError(f"no subtask covers episode={episode}, frame={frame}")
    task_index = int(raw.base._task_index[anchor])  # noqa: SLF001
    return str(raw.base.tasks[task_index])


@nnx.jit
def _prepare_context(model, observation, slow_force, slow_state, slow_mask):
    return model.prepare_force_policy_context(
        observation,
        slow_force_history=slow_force,
        slow_state_history=slow_state,
        slow_history_mask=slow_mask,
    )


@nnx.jit
def _sample_offset0(model, context, current_force, current_state, current_mask, noise):
    """Run the deployed spherical B2 offset-0 sampler."""

    offset = jnp.zeros((noise.shape[0],), dtype=jnp.int32)
    return model.sample_actions_force_update(
        jax.random.key(0),
        context,
        current_force_history=current_force,
        current_state_history=current_state,
        current_history_mask=current_mask,
        update_offset=offset,
        num_steps=10,
        noise=noise,
        executed_actions=None,
    )


@nnx.jit
def _sample_offset0_zero_full_token_memory(
    model, context, current_force, current_state, current_mask, noise
):
    """Set the final SlowProj+FastProj attention memory tokens to exact zero."""

    offset = jnp.zeros((noise.shape[0],), dtype=jnp.int32)
    return model.sample_actions_force_update(
        jax.random.key(0),
        context,
        current_force_history=current_force,
        current_state_history=current_state,
        current_history_mask=current_mask,
        update_offset=offset,
        num_steps=10,
        noise=noise,
        executed_actions=None,
        full_token_memory_scale=jnp.asarray(0.0, dtype=noise.dtype),
    )


def _sample_offset0_cross_impl(
    model,
    context,
    current_force,
    current_state,
    current_mask,
    noise,
    *,
    use_force: bool,
):
    """Sample with the hidden force residual exactly present or absent."""

    conditioner = model._require_force_conditioner()  # noqa: SLF001
    offset = jnp.zeros((noise.shape[0],), dtype=jnp.int32)
    modulation = conditioner.modulate(
        context.z_model,
        context.force_latent,
        current_force,
        current_state,
        current_mask,
        offset,
    )
    # The hidden-cross-attention architecture leaves the original zM route
    # untouched. A zero legacy delta is therefore the exact layerwise zM path.
    layerwise_latents = model._force_layerwise_latents(  # noqa: SLF001
        context.layerwise_arm_latents,
        jnp.zeros_like(modulation.delta_z),
    )
    force_tokens = (
        conditioner.compose_force_tokens(
            context.force_latent,
            modulation.recent_tokens,
            current_mask,
            offset,
        )
        if use_force
        else None
    )
    action_chunk = model._mask_action_condition(noise)  # noqa: SLF001
    committed = jnp.zeros(action_chunk.shape[:2], dtype=jnp.bool_)

    def step(index, actions):
        time = jnp.asarray(1.0 - index / 10.0, dtype=actions.dtype)
        velocity = model._suffix_velocity(  # noqa: SLF001
            context.prefix_mask,
            context.kv_cache,
            actions,
            jnp.broadcast_to(time, committed.shape),
            context.z_model,
            committed,
            layerwise_latents=layerwise_latents,
            force_tokens=force_tokens,
        )
        return model._mask_action_condition(actions - 0.1 * velocity)  # noqa: SLF001

    result = jax.lax.fori_loop(0, 10, step, action_chunk)
    return result[..., : model.config.active_action_dim], modulation.delta_z


@nnx.jit
def _sample_offset0_cross_force(
    model, context, current_force, current_state, current_mask, noise
):
    return _sample_offset0_cross_impl(
        model,
        context,
        current_force,
        current_state,
        current_mask,
        noise,
        use_force=True,
    )


@nnx.jit
def _sample_offset0_cross_zero(
    model, context, current_force, current_state, current_mask, noise
):
    # Passing None is important: zero tokens would retain learned projection
    # biases and would not be a true force-free control.
    return _sample_offset0_cross_impl(
        model,
        context,
        current_force,
        current_state,
        current_mask,
        noise,
        use_force=False,
    )


def _tcp(actions: np.ndarray, arm: str) -> np.ndarray:
    _fk_eval.TCP_LOCAL_Z_OFFSET_M = 0.20
    fk = _fk_eval.SimpleCR1FK(Path("assets/cr1_right_arm_tcp_red/urdf/CR1ARMR_TCP_RED.urdf"))
    joint_slice = slice(8, 15) if arm == "right" else slice(0, 7)
    points = np.stack([fk.pose(q)[0] for q in np.asarray(actions)[:, joint_slice]])
    if arm == "left":
        points += np.asarray([0.0, 0.445, 0.0])
    return points


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _metrics(pred_norm: np.ndarray, gt_norm: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> dict:
    result = {
        "normalized_action_rmse": _rmse(pred_norm[..., :16] - gt_norm[..., :16]),
        "joint_rmse_rad": _rmse(pred[..., np.r_[0:7, 8:15]] - gt[..., np.r_[0:7, 8:15]]),
        "left_joint_rmse_rad": _rmse(pred[..., :7] - gt[..., :7]),
        "right_joint_rmse_rad": _rmse(pred[..., 8:15] - gt[..., 8:15]),
        "left_gripper_rmse": _rmse(pred[..., 7] - gt[..., 7]),
        "right_gripper_rmse": _rmse(pred[..., 15] - gt[..., 15]),
    }
    for arm in ("left", "right"):
        error = (_tcp(pred, arm) - _tcp(gt, arm)) * 1000.0
        result[f"{arm}_tcp_xyz_rmse_mm"] = _rmse(error)
        result[f"{arm}_tcp_translation_rmse_mm"] = _rmse(np.linalg.norm(error, axis=-1))
    return result


def _aggregate(per_window: list[dict], method: str) -> dict:
    norm_pred = np.concatenate([row[f"{method}_normalized"] for row in per_window])
    norm_gt = np.concatenate([row["gt_normalized"] for row in per_window])
    pred = np.concatenate([row[f"{method}_decoded"] for row in per_window])
    gt = np.concatenate([row["gt_decoded"] for row in per_window])
    return _metrics(norm_pred, norm_gt, pred, gt)


def _plot_aggregate(rows: list[dict], output: Path) -> None:
    frames = np.asarray([row["frame"] for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    specs = (
        ("normalized_action_rmse", "Normalized 16D action RMSE"),
        ("joint_rmse_rad", "Decoded 14-joint RMSE (rad)"),
        ("left_tcp_translation_rmse_mm", "Left TCP translation RMSE (mm)"),
        ("right_tcp_translation_rmse_mm", "Right TCP translation RMSE (mm)"),
    )
    for axis, (key, title) in zip(axes.flat, specs, strict=True):
        axis.plot(frames, [row["force_metrics"][key] for row in rows], "o-", ms=3, label="B2 + force")
        axis.plot(frames, [row["zero_metrics"][key] for row in rows], "o-", ms=3, label="same B2, Δz=0")
        axis.set_title(title)
        axis.set_xlabel("episode frame (50-step anchors)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_horizon(row: dict, output: Path) -> None:
    gt = row["gt_decoded"]
    force = row["force_decoded"]
    zero = row["zero_decoded"]
    fig, axes = plt.subplots(4, 4, figsize=(18, 12), sharex=True, constrained_layout=True)
    labels = [*[f"L q{i + 1}" for i in range(7)], "L grip", *[f"R q{i + 1}" for i in range(7)], "R grip"]
    for dim, axis in enumerate(axes.flat):
        axis.plot(gt[:, dim], color="#111827", lw=2.0, label="GT")
        axis.plot(force[:, dim], color="#dc2626", lw=1.3, label="B2 + force")
        axis.plot(zero[:, dim], color="#2563eb", lw=1.2, ls="--", label="same B2, Δz=0")
        axis.set_title(labels[dim])
        axis.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Episode {row['episode']} frame {row['frame']} — {row['prompt']}")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--subtask-sidecar", type=Path)
    parser.add_argument("--pad-subtask-horizon", action="store_true")
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--force-norm", type=Path, required=True)
    parser.add_argument("--episode", default="auto")
    parser.add_argument(
        "--allow-explicit-excluded-episode",
        action="store_true",
        help=(
            "when --episode is explicit, allow an episode outside the normal "
            "episode%%10 validation split (for auditing episodes excluded from training)"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="optional leading-window cap for smoke tests; zero evaluates the full episode",
    )
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder-width", type=int, default=256)
    parser.add_argument("--encoder-heads", type=int, default=4)
    parser.add_argument("--encoder-mlp-dim", type=int, default=1024)
    parser.add_argument("--force-latent-dim", type=int, default=256)
    parser.add_argument("--force-hidden-cross-attention", action="store_true")
    parser.add_argument("--force-hidden-cross-attention-heads", type=int, default=2)
    parser.add_argument("--full-token-force-adapter", action="store_true")
    parser.add_argument("--full-token-force-adapter-heads", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = _config(args)
    dataset = build_force_dataset(
        (args.dataset_root,),
        subtask_sidecars=(args.subtask_sidecar,) if args.subtask_sidecar else None,
        pad_subtask_horizon=args.pad_subtask_horizon,
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        force_norm_path=args.force_norm,
        max_token_len=config.max_token_len,
        seed=0,
    )
    candidates = _candidate_indices(
        dataset,
        require_validation_episode=not (
            args.allow_explicit_excluded_episode and args.episode != "auto"
        ),
    )
    episode = _choose_episode(candidates, args.episode)
    selected = sorted(
        candidates[episode],
        key=lambda index: int(dataset._raw.base._frame_index[int(dataset._raw.anchors[index])]),  # noqa: SLF001
    )
    if args.max_windows > 0:
        selected = selected[: args.max_windows]
    raw = dataset._raw  # noqa: SLF001
    selection = []
    for dataset_index in selected:
        anchor = int(raw.anchors[dataset_index])
        selection.append(
            {
                "dataset_index": int(dataset_index),
                "anchor_data_index": anchor,
                "episode": episode,
                "frame": int(raw.base._frame_index[anchor]),  # noqa: SLF001
                "prompt": _anchor_prompt(raw, anchor),
            }
        )
    selection_path = args.output_dir / "selection.json"
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "command.txt").write_text(
        " ".join(shlex.quote(value) for value in sys.argv) + "\n", encoding="utf-8"
    )
    norm = ForceNormalization.load(args.force_norm)
    decode = _output_transform(
        args.dataset_root,
        config,
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
    )

    # B1/B2 checkpoints made with the training host's NNX version can omit an
    # optional GRU dense bias represented as an explicit zero array by the
    # inference host. Use the same compatibility merge as training instead of
    # demanding a byte-identical raw pytree.
    initialized = config.create(jax.random.key(0))
    _, initialized_state = nnx.split(initialized)
    params = AtomicPi05CheckpointLoader(str(args.checkpoint / "params")).load(
        initialized_state.to_pure_dict()
    )
    del initialized, initialized_state
    # ``intersect_trees`` drops explicit ``None`` leaves, but NNX still counts
    # the disabled GRU bias in its graph structure. The compatibility loader
    # has already removed every genuinely extra parameter, so skip that second
    # intersection here and retain the matching ``None`` leaf.
    model = config.load(params, remove_extra_params=False)
    model.eval()
    base_model = None
    if args.base_checkpoint is not None:
        base_initialized = config.create(jax.random.key(1))
        _, base_initialized_state = nnx.split(base_initialized)
        base_params = AtomicPi05CheckpointLoader(str(args.base_checkpoint / "params")).load(
            base_initialized_state.to_pure_dict()
        )
        del base_initialized, base_initialized_state
        base_model = config.load(base_params, remove_extra_params=False)
        base_model.eval()
    rows: list[dict] = []
    for start in range(0, len(selected), args.batch_size):
        indices = selected[start : start + args.batch_size]
        real_count = len(indices)
        if real_count < args.batch_size:
            indices += [indices[-1]] * (args.batch_size - real_count)
        samples = [dataset[index] for index in indices]
        batch = force_collate(samples)
        observation_np, gt_normalized, force = batch_to_force_inputs(batch)
        observation = jax.tree.map(jnp.asarray, observation_np)
        context = _prepare_context(
            model,
            observation,
            jnp.asarray(force["slow_force_history"]),
            jnp.asarray(force["slow_state_history"]),
            jnp.asarray(force["slow_history_mask"]),
        )
        # At a new 50-step chunk no post-anchor fast samples exist. This is the
        # deployment offset-0 contract: zF is active, fast tokens are masked.
        current_mask = jnp.zeros_like(jnp.asarray(force["current_history_mask"]))
        noise = jax.random.normal(
            jax.random.fold_in(jax.random.key(args.seed), start),
            (args.batch_size, config.action_horizon, config.action_dim),
        )
        sample_args = (
            model,
            context,
            jnp.asarray(force["current_force_history"]),
            jnp.asarray(force["current_state_history"]),
            current_mask,
            noise,
        )
        if config.enable_force_hidden_cross_attention:
            force_pred, delta_z = _sample_offset0_cross_force(*sample_args)
            zero_pred, _ = _sample_offset0_cross_zero(*sample_args)
            zero_token_pred = None
        else:
            force_pred, modulation = _sample_offset0(*sample_args)
            delta_z = modulation.delta_z
            zero_pred = _sample_global(
                model,
                observation,
                noise,
                None,
                None,
                jnp.zeros(noise.shape[:2], dtype=jnp.bool_),
            )
            if not config.force_full_token_adapter:
                raise ValueError("zero-token control requires the full-token force adapter")
            # Preserve zM, phase, masks and all learned adapter parameters, but
            # zero the final 30 SlowProj + 10 FastProj memory after type
            # embeddings and immediately before cross-attention.
            zero_token_pred, _ = _sample_offset0_zero_full_token_memory(*sample_args)
        # Same-episode wrong-zF control: preserve each row's image/state/prompt,
        # KV cache and noise, but cyclically assign the neighboring window's
        # force latent. Use a batch size that divides the selected window count.
        wrong_context = context.replace(
            force_latent=jnp.roll(context.force_latent, 1, axis=0),
            slow_history_tokens=jnp.roll(context.slow_history_tokens, 1, axis=0),
            slow_history_token_mask=jnp.roll(
                context.slow_history_token_mask, 1, axis=0
            ),
        )
        wrong_args = (
            model,
            wrong_context,
            jnp.asarray(force["current_force_history"]),
            jnp.asarray(force["current_state_history"]),
            current_mask,
            noise,
        )
        if config.enable_force_hidden_cross_attention:
            wrong_zf_pred, _ = _sample_offset0_cross_force(*wrong_args)
        else:
            wrong_zf_pred, _ = _sample_offset0(*wrong_args)
        base_pred = None
        if base_model is not None:
            base_pred = _sample_global(
                base_model,
                observation,
                noise,
                None,
                None,
                jnp.zeros(noise.shape[:2], dtype=jnp.bool_),
            )
        force_pred = np.asarray(jax.device_get(force_pred))[:real_count]
        zero_pred = np.asarray(jax.device_get(zero_pred))[:real_count]
        wrong_zf_pred = np.asarray(jax.device_get(wrong_zf_pred))[:real_count]
        if zero_token_pred is not None:
            zero_token_pred = np.asarray(jax.device_get(zero_token_pred))[:real_count]
        if base_pred is not None:
            base_pred = np.asarray(jax.device_get(base_pred))[:real_count]
        delta_z = np.asarray(jax.device_get(delta_z))[:real_count]
        for local, dataset_index in enumerate(indices[:real_count]):
            anchor = int(raw.anchors[dataset_index])
            frame = int(raw.base._frame_index[anchor])  # noqa: SLF001
            prompt = _anchor_prompt(raw, anchor)
            raw_state = np.asarray(raw.base._states[anchor])  # noqa: SLF001
            gt_norm = np.asarray(gt_normalized[local])[..., :16]
            state_norm = np.asarray(batch["state"][local])
            gt_decoded = np.asarray(decode(state_norm, raw_state, gt_norm)["actions"])
            force_decoded = np.asarray(decode(state_norm, raw_state, force_pred[local])["actions"])
            zero_decoded = np.asarray(decode(state_norm, raw_state, zero_pred[local])["actions"])
            wrong_zf_decoded = np.asarray(
                decode(state_norm, raw_state, wrong_zf_pred[local])["actions"]
            )
            zero_token_decoded = (
                np.asarray(
                    decode(state_norm, raw_state, zero_token_pred[local])["actions"]
                )
                if zero_token_pred is not None
                else None
            )
            base_decoded = (
                np.asarray(decode(state_norm, raw_state, base_pred[local])["actions"])
                if base_pred is not None
                else None
            )
            rows.append(
                {
                    "episode": episode,
                    "frame": frame,
                    "dataset_index": int(dataset_index),
                    "anchor_data_index": anchor,
                    "prompt": prompt,
                    "delta_z_norm": float(np.mean(np.linalg.norm(delta_z[local], axis=-1))),
                    "gt_normalized": gt_norm,
                    "force_normalized": force_pred[local],
                    "zero_normalized": zero_pred[local],
                    "gt_decoded": gt_decoded,
                    "force_decoded": force_decoded,
                    "zero_decoded": zero_decoded,
                    "force_metrics": _metrics(force_pred[local], gt_norm, force_decoded, gt_decoded),
                    "zero_metrics": _metrics(zero_pred[local], gt_norm, zero_decoded, gt_decoded),
                    "wrong_zf_normalized": wrong_zf_pred[local],
                    "wrong_zf_decoded": wrong_zf_decoded,
                    "wrong_zf_metrics": _metrics(
                        wrong_zf_pred[local], gt_norm, wrong_zf_decoded, gt_decoded
                    ),
                    **(
                        {
                            "zero_token_normalized": zero_token_pred[local],
                            "zero_token_decoded": zero_token_decoded,
                            "zero_token_metrics": _metrics(
                                zero_token_pred[local],
                                gt_norm,
                                zero_token_decoded,
                                gt_decoded,
                            ),
                        }
                        if zero_token_pred is not None
                        else {}
                    ),
                    **(
                        {
                            "base_normalized": base_pred[local],
                            "base_decoded": base_decoded,
                            "base_metrics": _metrics(base_pred[local], gt_norm, base_decoded, gt_decoded),
                        }
                        if base_pred is not None
                        else {}
                    ),
                }
            )

    force_summary = _aggregate(rows, "force")
    zero_summary = _aggregate(rows, "zero")
    wrong_zf_summary = _aggregate(rows, "wrong_zf")
    zero_token_summary = _aggregate(rows, "zero_token")
    base_summary = _aggregate(rows, "base") if base_model is not None else None
    primary = "normalized_action_rmse"
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "episode": episode,
        "selection": (
            "explicit episode excluded from training; every genuine frame%50==0 anchor"
            if args.allow_explicit_excluded_episode and args.episode != "auto"
            else "held-out episode; every genuine frame%50==0 anchor; auto episode nearest median window count"
        ),
        "window_count": len(rows),
        "covered_gt_steps": len(rows) * 50,
        "offset": 0,
        "prompt_contract": (
            "current subtask from aligned sidecar; padded at semantic boundary; no prompt mixing"
            if args.subtask_sidecar and args.pad_subtask_horizon
            else "current subtask from aligned sidecar; unmodified horizon; no prompt mixing"
            if args.subtask_sidecar
            else "PromptFromLeRobotTask using meta/tasks.parquet; no prompt mixing"
        ),
        "comparison_contract": "same B2 checkpoint/observation/prompt/GT/noise; only learned force delta_z versus delta_z=0",
        "force": force_summary,
        "same_b2_delta_z_zero": zero_summary,
        "same_episode_wrong_zf": wrong_zf_summary,
        "slow_fast_attention_memory_zero": zero_token_summary,
        "zero_token_normalized_rmse_change_pct": 100.0
        * (zero_token_summary[primary] / force_summary[primary] - 1.0),
        "wrong_zf_normalized_rmse_change_pct": 100.0
        * (wrong_zf_summary[primary] / force_summary[primary] - 1.0),
        "winner_by_normalized_action_rmse": "force" if force_summary[primary] < zero_summary[primary] else "delta_z_zero",
        "normalized_action_rmse_relative_change_pct": 100.0 * (force_summary[primary] / zero_summary[primary] - 1.0),
        "mean_delta_z_norm": float(np.mean([row["delta_z_norm"] for row in rows])),
        **(
            {
                "afro50k": base_summary,
                "b2_vs_afro50k_normalized_rmse_change_pct": 100.0
                * (force_summary[primary] / base_summary[primary] - 1.0),
                "winner_b2_vs_afro50k": (
                    "b2_force" if force_summary[primary] < base_summary[primary] else "afro50k"
                ),
            }
            if base_summary is not None
            else {}
        ),
    }
    serializable_rows = []
    for row in rows:
        serializable_rows.append({
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in row.items()
        })
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary | {"windows": serializable_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (args.output_dir / "windows.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["episode", "frame", "prompt", "delta_z_norm"] + [
            f"{method}_{key}"
            for method in (("force", "zero", "wrong_zf", "zero_token", "base") if base_model is not None else ("force", "zero", "wrong_zf", "zero_token"))
            for key in rows[0][f"{method}_metrics"]
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in ("episode", "frame", "prompt", "delta_z_norm")}
            for method in (("force", "zero", "wrong_zf", "zero_token", "base") if base_model is not None else ("force", "zero", "wrong_zf", "zero_token")):
                flat.update({f"{method}_{key}": value for key, value in row[f"{method}_metrics"].items()})
            writer.writerow(flat)
    _plot_aggregate(rows, args.output_dir / "episode50_error_comparison.png")
    _plot_horizon(rows[len(rows) // 2], args.output_dir / "representative_16d_horizon.png")
    base_manifest_line = (
        f"- AFRO manifest sha256: `{_sha256(args.base_checkpoint / 'params' / 'manifest.ocdbt')}`\n"
        if args.base_checkpoint
        else ""
    )
    contract = (
        f"# Evaluation contract\n\n"
        f"- checkpoint: `{args.checkpoint.resolve()}`\n"
        f"- checkpoint manifest sha256: `{_sha256(args.checkpoint / 'params' / 'manifest.ocdbt')}`\n"
        f"- AFRO base checkpoint: `{args.base_checkpoint.resolve() if args.base_checkpoint else 'not evaluated'}`\n"
        + base_manifest_line
        + f"- dataset: `{args.dataset_root.resolve()}`\n"
        f"- subtask sidecar: `{args.subtask_sidecar.resolve() if args.subtask_sidecar else 'none'}`\n"
        f"- subtask sidecar sha256: `{_sha256(args.subtask_sidecar) if args.subtask_sidecar else 'none'}`\n"
        f"- base norm: `{(args.norm_assets_dir / args.norm_asset_id / 'norm_stats.json').resolve()}`\n"
        f"- base norm sha256: `{_sha256(args.norm_assets_dir / args.norm_asset_id / 'norm_stats.json')}`\n"
        f"- force norm: `{args.force_norm.resolve()}`\n"
        f"- force norm sha256: `{_sha256(args.force_norm)}`\n"
        f"- prompt: {'current subtask from aligned sidecar' if args.subtask_sidecar else 'task string from meta/tasks.parquet'}\n"
        f"- subtask boundary padding: `{args.pad_subtask_horizon}`\n"
        f"- selection: episode {episode}, every genuine 50-frame anchor; "
        f"allow_explicit_excluded_episode={args.allow_explicit_excluded_episode}\n"
        f"- selection manifest: `{selection_path.resolve()}` sha256 `{_sha256(selection_path)}`\n"
        f"- paired control: identical B2 weights, input, target and noise; only `delta_z` is zeroed\n"
        f"- B2 versus AFRO50K: identical observation/state/prompt/noise/normalization/sampler\n"
        f"- architecture: final ordered right/left zM through restored 2-head AFRO cross-attention; spherical Force correction max45deg\n"
        f"- output: normalized 16D, decoded joints/grippers, CR1 FK with 0.20 m TCP\n"
    )
    (args.output_dir / "run_contract.md").write_text(contract, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
