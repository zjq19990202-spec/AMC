#!/usr/bin/env python3
"""Train AtomicPi05 on annotated local LeRobot v3 datasets.

This is deliberately a standalone entrypoint: the released OpenPI trainer
only consumes ``Observation, actions`` and would silently drop our atomic
prompt/label/raw-qpos sidecar.  It reuses OpenPI's FSDP state, optimizer and
checkpoint loader without modifying the π0.5 repository.
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
import orbax.checkpoint as ocp
from flax import nnx

from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.training import config as _config
from openpi.training import checkpoints as _checkpoints
from openpi.training import sharding
from openpi.training import utils as training_utils

from atomic_latent_vla.pi05 import AtomicCompositionTargets, AtomicPi05Config, AtomicTargets
from atomic_latent_vla.pi05.training_data import batch_to_observation, build_atomic_loader
from atomic_latent_vla.pi05.weights import AtomicPi05CheckpointLoader
from atomic_latent_vla.tcp import BIMANUAL_TCP_POSE_SIDECAR


@dataclasses.dataclass(frozen=True)
class GradientAccumulationOptimizer:
    """Wrap an OpenPI optimizer with mean-gradient accumulation.

    ``TrainState.step`` is advanced only when ``MultiSteps`` applies an inner
    optimizer update, so schedules, checkpoint numbers, and prompt curricula
    remain expressed in optimizer updates rather than micro-steps.
    """

    inner: object
    accumulation_steps: int

    def create(self, lr, weight_decay_mask=None):
        tx = self.inner.create(lr, weight_decay_mask=weight_decay_mask)
        multisteps = optax.MultiSteps(
            tx,
            every_k_schedule=self.accumulation_steps,
            use_grad_mean=True,
        )
        # OpenPI's TrainState is runtime type-checked against the narrow
        # GradientTransformation named tuple. MultiSteps implements the same
        # init/update protocol but is a class, so expose it through that tuple.
        return optax.GradientTransformation(multisteps.init, multisteps.update)


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
        "--zt-teacher-sidecar",
        type=Path,
        action="append",
        default=[],
        help=(
            "Frozen atomic-prompt zT Q1/Q3 sidecar. Repeat once per dataset "
            "root in the same order."
        ),
    )
    parser.add_argument(
        "--atomic-composition-sidecar",
        default="fk_horizon_3hz_gate_top5_v1",
        help=(
            "Dataset-meta-relative directory (or absolute path) containing "
            "the materialized Dual/Drop composition targets."
        ),
    )
    parser.add_argument(
        "--pad-subtask-horizon",
        action="store_true",
        help=(
            "Keep one subtask prompt per 50-step sample. If its reviewed segment "
            "ends inside the horizon, repeat that segment's final absolute action "
            "instead of crossing into and concatenating the next subtask."
        ),
    )
    parser.add_argument(
        "--tcp-twist-norm",
        required=True,
        help="TCP relative-delta q01/q99 stats used by Stage-A zT training.",
    )
    parser.add_argument(
        "--coefficient-target",
        choices=("tcp_twist", "joint_delta"),
        default="tcp_twist",
        help="Stage-A coefficient-head layout required by the loaded checkpoint.",
    )
    parser.add_argument(
        "--base-params",
        default="/mnt/cunchu/zjq/.cache/openpi/openpi-assets/checkpoints/pi05_base/params",
    )
    parser.add_argument("--checkpoint-base-dir", default="/mnt/cunchu/zjq/atomic_pi05_runs")
    parser.add_argument("--run-name", default=time.strftime("atomic_pi05_bs32_%Y%m%d_%H%M%S"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument(
        "--max-token-len",
        type=int,
        default=200,
        help=(
            "Padded PI0.5 prompt length. This may be reduced only after an "
            "exhaustive prompt audit proves that no valid prompt is truncated."
        ),
    )
    parser.add_argument(
        "--fsdp-devices",
        type=int,
        default=None,
        help=(
            "Number of devices in each FSDP group. Defaults to --devices "
            "for backward compatibility; use 1 for replicated data parallel."
        ),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help=(
            "Number of global micro-batches averaged per optimizer update. "
            "Effective batch = batch-size * this value."
        ),
    )
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--save-interval", type=int, default=1_000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-steps", type=int, default=0, help="override steps for a quick launch check")
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help="Compile/run smoke updates without writing multi-GB checkpoints.",
    )
    parser.add_argument("--initial-step", type=int, default=0)
    parser.add_argument(
        "--phase-start-step",
        type=int,
        default=None,
        help="Global step at which joint zT/zM training began; anchors its prompt curriculum.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume the latest full checkpoint in this run.")
    parser.add_argument(
        "--restore-full-state",
        type=Path,
        default=None,
        help=(
            "Restore params, optimizer moments, and global step from this exact "
            "OpenPI checkpoint step into a new run. When a zT teacher sidecar is "
            "provided, its matching frozen codebook replaces the restored one."
        ),
    )
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
        "--zt-fraction",
        type=float,
        default=0.5,
        help="Fraction of each global batch assigned to text/state zT; the remainder trains full zM.",
    )
    parser.add_argument(
        "--paired-zt-zm",
        action="store_true",
        help=(
            "Run text-only ZT and full ZM on the same physical batch. A batch "
            "of 128 therefore supplies 128 ZT plus 128 ZM routes, and ZM can "
            "reuse stop-gradient ZT directions without a third prefix pass."
        ),
    )
    parser.add_argument("--zt-loss-weight", type=float, default=0.5)
    parser.add_argument("--zm-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--text-flow-loss-weight",
        type=float,
        default=0.0,
        help="Shared PI0.5 flow loss on the text/state ZT shard.",
    )
    parser.add_argument("--coefficient-loss-weight", type=float, default=1.0)
    parser.add_argument("--text-atomic-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--full-atomic-loss-weight",
        type=float,
        default=1.0,
        help="ZM cosine alignment to the same-state atomic-prompt ZT teacher.",
    )
    parser.add_argument(
        "--full-atomic-huber-delta-deg",
        type=float,
        default=0.0,
        help="Deprecated compatibility option; ZM no longer evaluates angular/Huber loss.",
    )
    parser.add_argument(
        "--codebook-loss-weight",
        type=float,
        default=1.0,
        help="Code-only geodesic update weight; set to 0 after the codebook is frozen.",
    )
    parser.add_argument(
        "--freeze-codebook",
        action="store_true",
        help=(
            "Keep all codebook parameters bitwise fixed while still training "
            "Q1/Q3 with the frozen zT teacher, strict Two-Way, and gate projection."
        ),
    )
    parser.add_argument("--subtask-ce-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--atomic-composition-loss-weight",
        "--atomic-projection-loss-weight",
        dest="atomic_composition_loss_weight",
        type=float,
        default=0.10,
        help=(
            "Cosine weight for normalized frozen-codebook directions: Dual "
            "Top-2 in zT or Drop Top-5 in zM. The old option name remains an alias."
        ),
    )
    parser.add_argument(
        "--atomic-prompt-probability",
        type=float,
        default=1.0,
        help=(
            "Among ZM rows where both arms have strict atoms, probability of "
            "replacing the composed subtask prompt with both arm atomic prompts."
        ),
    )
    parser.add_argument(
        "--atomic-text-ce-probability",
        type=float,
        default=1.0 / 3.0,
        help=(
            "Conditional probability of AR atomic-text CE among the 60%% "
            "both-arm-strict rows that use a subtask prompt. 1/3 gives 20%% "
            "of all both-arm-strict rows."
        ),
    )
    parser.add_argument(
        "--subtask-ce-batch-size",
        type=int,
        default=0,
        help=(
            "If positive, compact label-selected atomic-text CE rows into this "
            "fixed-size sub-batch; 0 retains the full-batch masked CE path."
        ),
    )
    parser.add_argument(
        "--unfreeze-vision",
        action="store_true",
        help="Train the SigLIP vision encoder instead of freezing it.",
    )
    parser.add_argument(
        "--layerwise-atomic-flow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use a distinct zM composed at every transformer depth. Pass "
            "--no-layerwise-atomic-flow to compose one final zM and reuse that "
            "same condition in every Action Expert FiLM adapter. This does not "
            "disable zM or reduce modulation to a single layer."
        ),
    )
    parser.add_argument(
        "--spherical-visual-latent",
        action="store_true",
        help="retract each four-basis visual zM update to the unit sphere",
    )
    parser.add_argument("--visual-max-update-angle-deg", type=float, default=45.0)
    parser.add_argument("--visual-rotation-loss-weight", type=float, default=0.0)
    parser.add_argument("--visual-rotation-free-angle-deg", type=float, default=20.0)
    parser.add_argument("--visual-rotation-loss-warmup-steps", type=int, default=0)
    parser.add_argument("--subprompt-warmup-steps", type=int, default=5_000)
    parser.add_argument(
        "--subprompt-probability-after-warmup",
        type=float,
        default=0.40,
        help="Probability that a zM row receives its direct subtask after warmup; remaining rows use global.",
    )
    return parser.parse_args()


def _base_train_module():
    """Use OpenPI's tested FSDP initialization / checkpoint merge routine."""

    root = "/mnt/cunchu/yc/pi05/scripts"
    if root not in sys.path:
        sys.path.insert(0, root)
    import train as base_train  # noqa: PLC0415

    return base_train


def _restore_state_to_requested_sharding(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    state_sharding: training_utils.TrainState,
    step: int | None = None,
) -> training_utils.TrainState:
    """Restore directly into the current mesh without a second full-state copy."""

    with at.disable_typechecking():
        train_state, params = _checkpoints._split_params(state)
        train_state_sharding, params_sharding = _checkpoints._split_params(
            state_sharding
        )
    restored = checkpoint_manager.restore(
        step,
        args=ocp.args.Composite(
            train_state=ocp.args.PyTreeRestore(
                item=train_state,
                restore_args=ocp.checkpoint_utils.construct_restore_args(
                    train_state, train_state_sharding
                ),
            ),
            params=ocp.args.PyTreeRestore(
                item={"params": params},
                restore_args=ocp.checkpoint_utils.construct_restore_args(
                    {"params": params}, {"params": params_sharding}
                ),
            ),
        ),
    )
    with at.disable_typechecking():
        return _checkpoints._merge_params(
            restored["train_state"], restored["params"]
        )


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


def _zm_drop_composition_targets(
    weights: jax.Array,
    confidence: jax.Array,
    available: jax.Array,
    strict_supervision: jax.Array,
) -> AtomicCompositionTargets:
    """Use Top-5 code-direction projection only for dropped arms."""

    mask = (
        available
        & ~strict_supervision
        & (jnp.sum(weights, axis=-1) > 0.0)
    )
    normalized = weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1e-8)
    return AtomicCompositionTargets(
        weights=normalized,
        confidence=jnp.where(mask, confidence, 0.0),
        supervision_mask=mask,
    )


def _zt_dual_composition_targets(
    weights: jax.Array,
    confidence: jax.Array,
    available: jax.Array,
    strict_supervision: jax.Array,
    dual_supervision: jax.Array,
) -> AtomicCompositionTargets:
    """Use Top-2 code-direction projection only for strict Dual arms in zT."""

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


def _load_tcp_twist_quantile_range(path: str) -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    actual_sidecar = payload.get("_meta", {}).get("sidecar")
    if actual_sidecar != BIMANUAL_TCP_POSE_SIDECAR:
        raise ValueError(
            f"{path}: expected tcp200 norm from {BIMANUAL_TCP_POSE_SIDECAR}, "
            f"got {actual_sidecar!r}"
        )
    stats = payload["norm_stats"]["tcp_twist_delta"]
    value = np.asarray(stats["q99"], dtype=np.float32) - np.asarray(stats["q01"], dtype=np.float32)
    if value.shape != (12,) or not np.all(np.isfinite(value)) or np.any(value <= 0):
        raise ValueError(f"invalid TCP twist q99-q01 range in {path}: {value}")
    return value


def _subprompt_probability(
    phase_step: jax.Array,
    warmup_steps: int,
    probability_after_warmup: float,
) -> jax.Array:
    """Full z_M prompt curriculum; z_T always receives its local/fallback text."""

    # First establish the direct subtask-to-action mapping without asking the
    # model to infer the active subtask.  After 5k joint updates, introduce the
    # global task on at most 60% of rows while retaining 40% direct subtask
    # conditioning throughout the rest of training.
    return jnp.where(phase_step < warmup_steps, 1.0, probability_after_warmup)


def _slice_observation(observation: _model.Observation, start: int, end: int) -> _model.Observation:
    return jax.tree.map(lambda value: value[start:end], observation)


def _text_observation(
    observation: _model.Observation,
    prompt_tokens: jax.Array,
    prompt_mask: jax.Array,
) -> _model.Observation:
    """Remove cameras and install the local/fallback prompt for the zT shard."""

    return _model.Observation(
        images={},
        image_masks={},
        state=observation.state,
        tokenized_prompt=prompt_tokens,
        tokenized_prompt_mask=prompt_mask,
        token_ar_mask=None,
        token_loss_mask=None,
    )


def train_step(
    config: _config.TrainConfig,
    tcp_twist_quantile_range: jax.Array,
    phase_start_step: int,
    zt_batch_size: int,
    paired_zt_zm: bool,
    zt_loss_weight: float,
    zm_loss_weight: float,
    freeze_codebook: bool,
    subtask_ce_batch_size: int,
    subprompt_warmup_steps: int,
    subprompt_probability_after_warmup: float,
    atomic_prompt_probability: float,
    atomic_text_ce_probability: float,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, jax.Array, dict[str, jax.Array]],
    rng_fold_value: jax.Array,
    advance_step: jax.Array,
) -> tuple[training_utils.TrainState, dict[str, jax.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()
    frozen_codebook = model.codebook.value if freeze_codebook else None
    observation, actions, extra = batch
    def loss_fn(model, rng):
        atomic_weights = extra["atomic_weights"]
        atomic_mask = extra["atomic_supervision_mask"]
        atomic_count = jnp.sum(atomic_weights > 0.0, axis=-1)
        stay_mask = atomic_mask & (atomic_weights[..., 12] > 0.0)
        single_motion_mask = atomic_mask & (atomic_count == 1) & ~stay_mask
        dual_motion_mask = atomic_mask & (atomic_count == 2)
        zt_rng, zm_rng, prompt_rng, ce_rng = jax.random.split(rng, 4)
        zt_slice = slice(0, zt_batch_size)
        zm_start = 0 if paired_zt_zm else zt_batch_size
        zm_slice = slice(zm_start, actions.shape[0])
        zm_observation = _slice_observation(observation, zm_start, actions.shape[0])
        zm_targets = _atomic_targets(
            extra["atomic_weights"][zm_slice], extra["atomic_supervision_mask"][zm_slice]
        )
        zm_composition_targets = _zm_drop_composition_targets(
            extra["atomic_composition_weights"][zm_slice],
            extra["atomic_composition_confidence"][zm_slice],
            extra["atomic_composition_mask"][zm_slice],
            extra["atomic_supervision_mask"][zm_slice],
        )
        zm_batch_size = actions.shape[0] - zm_start
        both_arms_strict = jnp.all(
            extra["atomic_supervision_mask"][zm_slice], axis=-1
        )
        any_arm_strict = jnp.any(
            extra["atomic_supervision_mask"][zm_slice], axis=-1
        )
        # ZM normally consumes the horizon subtask (already composed with
        # "then" when it crosses segments). Only a both-arm strict row may
        # replace it with the left+right atomic prompt.
        full_use_atomic_prompt = both_arms_strict & jax.random.bernoulli(
            jax.random.fold_in(prompt_rng, state.step),
            atomic_prompt_probability,
            (zm_batch_size,),
        )
        full_use_subprompt = ~full_use_atomic_prompt

        # The ZM teacher is always the no-image atomic-prompt ZT encoding for
        # this exact state, even when ZM itself receives a subtask prompt. Its
        # output is frozen at the branch boundary: ZM updates Q1--Q4/flow but
        # cannot move the teacher path or the codebook. The independent ZT
        # shard below remains fully trainable.
        if paired_zt_zm:
            # Filled immediately after the ZT forward below. Keeping this
            # branch static avoids compiling the unused third prefix pass.
            text_teacher_directions = None
        elif "zt_teacher_directions" in extra:
            text_teacher_directions = extra["zt_teacher_directions"][zm_slice]
        else:
            teacher_observation = _text_observation(
                _slice_observation(observation, zm_start, actions.shape[0]),
                extra["atomic_prompt_tokens"][zm_slice],
                extra["atomic_prompt_mask"][zm_slice],
            )
            text_teacher_directions = jax.lax.stop_gradient(
                model.encode_text_arm_directions(teacher_observation)
            )

        # The optional AR auxiliary is fully disabled in the current recipe,
        # but keep its legacy gate for checkpoint/config compatibility.
        ce_gate_rng = ce_rng
        atomic_ce_rows = (
            any_arm_strict
            & full_use_subprompt
            & jax.random.bernoulli(
                jax.random.fold_in(ce_gate_rng, state.step),
                atomic_text_ce_probability,
                (zm_batch_size,),
            )
        )
        selected_atomic_ce_count = jnp.sum(atomic_ce_rows.astype(jnp.int32))
        subtask_target_tokens = extra.get("atomic_target_tokens")
        subtask_target_mask = extra.get("atomic_target_mask")
        if subtask_target_mask is not None:
            subtask_target_mask = subtask_target_mask[zm_slice] & atomic_ce_rows[:, None]
        zt_output = None
        if zt_batch_size:
            zt_observation = _text_observation(
                _slice_observation(observation, 0, zt_batch_size),
                extra["atomic_prompt_tokens"][zt_slice],
                extra["atomic_prompt_mask"][zt_slice],
            )
            # Keep the Stage-A target exactly unchanged: q01/q99 relative-TCP
            # statistics map a range of q99-q01 to two normalized units.
            coeff_delta = (
                2.0 * extra["tcp_twist_delta"][zt_slice]
                / tcp_twist_quantile_range.astype(extra["tcp_twist_delta"].dtype)[None, None, :]
            )
            zt_targets = _atomic_targets(
                extra["atomic_weights"][zt_slice], extra["atomic_supervision_mask"][zt_slice]
            )
            zt_composition_targets = _zt_dual_composition_targets(
                extra["atomic_composition_weights"][zt_slice],
                extra["atomic_composition_confidence"][zt_slice],
                extra["atomic_composition_mask"][zt_slice],
                extra["atomic_supervision_mask"][zt_slice],
                dual_motion_mask[zt_slice],
            )
            zt_output = model.compute_text_stage_loss(
                zt_rng,
                zt_observation,
                actions=actions[zt_slice],
                atomic_targets=zt_targets,
                atomic_composition_targets=zt_composition_targets,
                coefficient_tcp_twist_delta=coeff_delta,
                global_step=state.step,
                train=True,
                return_output=True,
            )
        if paired_zt_zm:
            if zt_output is None:
                raise ValueError("paired ZT/ZM requires an active ZT branch")
            # Same rows, same normalized state, and the ZT pass always used
            # the atomic prompt. ZM may use subtask text, but its strict arms
            # consistently chase this atomic ZT target without backpropagating
            # through the teacher Q path.
            text_teacher_directions = jax.lax.stop_gradient(zt_output.direction)
        zm_output = model.compute_full_stage_loss(
            zm_rng,
            zm_observation,
            actions[zm_slice],
            train=True,
            atomic_targets=zm_targets,
            atomic_composition_targets=zm_composition_targets,
            text_teacher_directions=text_teacher_directions,
            global_step=state.step,
            visual_rotation_phase_step=jnp.maximum(
                state.step - phase_start_step, 0
            ),
            subprompt_tokens=extra["subtask_prompt_tokens"][zm_slice],
            subprompt_mask=extra["subtask_prompt_mask"][zm_slice],
            atomic_prompt_tokens=extra["atomic_prompt_tokens"][zm_slice],
            atomic_prompt_mask=extra["atomic_prompt_mask"][zm_slice],
            full_use_atomic_prompt=full_use_atomic_prompt,
            subtask_target_tokens=(
                None if subtask_target_tokens is None else subtask_target_tokens[zm_slice]
            ),
            subtask_target_mask=subtask_target_mask,
            return_output=True,
        )
        zero = jnp.zeros((), zm_output.total_loss.dtype)
        total = zm_loss_weight * zm_output.total_loss
        if zt_output is not None:
            total = total + zt_loss_weight * zt_output.total_loss
        return total, {
            "flow_loss": zm_output.flow_loss,
            "flow_unweighted_loss": zm_output.flow_unweighted_loss,
            "flow_left_loss": zm_output.flow_left_loss,
            "flow_right_loss": zm_output.flow_right_loss,
            "flow_active_loss": zm_output.flow_active_loss,
            "flow_left_motion_share": zm_output.flow_left_motion_share,
            "flow_right_motion_share": zm_output.flow_right_motion_share,
            "coefficient_loss": zero if zt_output is None else zt_output.coefficient_loss,
            "coefficient_velocity_loss": zero if zt_output is None else zt_output.coefficient_losses.velocity,
            "coefficient_wall_loss": zero if zt_output is None else zt_output.coefficient_losses.wall,
            "coefficient_wall_weight": zero if zt_output is None else zt_output.coefficient_losses.wall_weight,
            # Keep every VQ term separately visible.  The full z_M branch is
            # deliberately read-only w.r.t. the codebook, while z_T updates it.
            "full_atomic_total_loss": zm_output.atomic_total_loss,
            "full_weighted_atomic_total_loss": (
                model.config.atomic_loss_weight * zm_output.atomic_total_loss
            ),
            "full_samplewise_two_way_ranking_loss": zm_output.atomic_losses.ranking,
            "full_ratio_kl_loss": zm_output.atomic_losses.ratio_kl,
            "full_weighted_ratio_kl_loss": (
                model.config.atomic_ratio_loss_weight * zm_output.atomic_losses.ratio_kl
            ),
            "full_composition_kl_loss": zm_output.composition_kl_loss,
            "full_weighted_composition_kl_loss": (
                model.config.atomic_composition_loss_weight
                * zm_output.composition_kl_loss
            ),
            "full_projection_loss": zm_output.projection_loss,
            "full_weighted_projection_loss": (
                model.config.atomic_composition_loss_weight
                * zm_output.projection_loss
            ),
            "full_codebook_loss": zm_output.atomic_losses.codebook,
            "full_text_teacher_cosine_loss": zm_output.text_teacher_cosine_loss,
            "full_text_teacher_cosine_similarity": (
                zm_output.text_teacher_cosine_similarity
            ),
            "full_strict_atomic_huber_loss": zm_output.strict_atomic_huber_loss,
            "full_visual_rotation_loss": zm_output.visual_rotation_loss,
            "full_visual_rotation_mean_angle_deg": (
                jnp.rad2deg(zm_output.visual_rotation_mean_angle_rad)
            ),
            "full_visual_rotation_loss_scale": zm_output.visual_rotation_loss_scale,
            "full_weighted_visual_rotation_loss": (
                model.config.visual_rotation_loss_weight
                * zm_output.visual_rotation_loss_scale
                * zm_output.visual_rotation_loss
            ),
            "text_samplewise_two_way_ranking_loss": zero if zt_output is None else zt_output.atomic_losses.ranking,
            "text_ratio_kl_loss": zero if zt_output is None else zt_output.atomic_losses.ratio_kl,
            "text_weighted_ratio_kl_loss": (
                zero
                if zt_output is None
                else model.config.atomic_ratio_loss_weight * zt_output.atomic_losses.ratio_kl
            ),
            "text_codebook_loss": zero if zt_output is None else zt_output.atomic_losses.codebook,
            "text_atomic_total_loss": zero if zt_output is None else zt_output.atomic_total_loss,
            "text_projection_loss": zero if zt_output is None else zt_output.projection_loss,
            "text_weighted_projection_loss": (
                zero
                if zt_output is None
                else model.config.atomic_composition_loss_weight * zt_output.projection_loss
            ),
            "quantity_loss": zm_output.quantity_loss,
            "subtask_ce_loss": zm_output.subtask_ce_loss,
            "atomic_text_ce_loss": zm_output.subtask_ce_loss,
            "perplexity": zm_output.atomic_losses.perplexity,
            "text_perplexity": zero if zt_output is None else zt_output.atomic_losses.perplexity,
            "full_right_perplexity": zm_output.atomic_losses.right_perplexity,
            "full_left_perplexity": zm_output.atomic_losses.left_perplexity,
            "text_right_perplexity": zero if zt_output is None else zt_output.atomic_losses.right_perplexity,
            "text_left_perplexity": zero if zt_output is None else zt_output.atomic_losses.left_perplexity,
            "atomic_fraction": jnp.mean(extra["atomic_supervision_mask"].astype(jnp.float32)),
            "single_motion_fraction": jnp.mean(single_motion_mask.astype(jnp.float32)),
            "dual_motion_fraction": jnp.mean(dual_motion_mask.astype(jnp.float32)),
            "stay_fraction": jnp.mean(stay_mask.astype(jnp.float32)),
            "right_atomic_fraction": jnp.mean(
                extra["atomic_supervision_mask"][:, 0].astype(jnp.float32)
            ),
            "left_atomic_fraction": jnp.mean(
                extra["atomic_supervision_mask"][:, 1].astype(jnp.float32)
            ),
            "right_stay_fraction": jnp.mean(stay_mask[:, 0].astype(jnp.float32)),
            "left_stay_fraction": jnp.mean(stay_mask[:, 1].astype(jnp.float32)),
            "zt_atomic_fraction": (
                zero
                if not zt_batch_size
                else jnp.mean(extra["atomic_supervision_mask"][zt_slice].astype(jnp.float32))
            ),
            "zm_atomic_fraction": jnp.mean(
                extra["atomic_supervision_mask"][zm_slice].astype(jnp.float32)
            ),
            "zt_right_atomic_fraction": (
                zero
                if not zt_batch_size
                else jnp.mean(extra["atomic_supervision_mask"][zt_slice, 0].astype(jnp.float32))
            ),
            "zt_left_atomic_fraction": (
                zero
                if not zt_batch_size
                else jnp.mean(extra["atomic_supervision_mask"][zt_slice, 1].astype(jnp.float32))
            ),
            "zm_right_atomic_fraction": jnp.mean(
                extra["atomic_supervision_mask"][zm_slice, 0].astype(jnp.float32)
            ),
            "zm_left_atomic_fraction": jnp.mean(
                extra["atomic_supervision_mask"][zm_slice, 1].astype(jnp.float32)
            ),
            "full_subprompt_fraction": jnp.mean(full_use_subprompt.astype(jnp.float32)),
            "full_atomic_prompt_fraction": jnp.mean(
                full_use_atomic_prompt.astype(jnp.float32)
            ),
            "both_arms_strict_fraction": jnp.mean(
                both_arms_strict.astype(jnp.float32)
            ),
            "composition_supervision_fraction": jnp.mean(
                zm_composition_targets.supervision_mask.astype(jnp.float32)
            ),
            "subtask_ce_fraction": (
                jnp.zeros((), dtype=jnp.float32)
                if subtask_target_mask is None
                else selected_atomic_ce_count.astype(jnp.float32) / zm_batch_size
            ),
            "subtask_ce_candidate_fraction": jnp.mean(
                atomic_ce_rows.astype(jnp.float32)
            ),
            "subtask_ce_compact_utilization": (
                selected_atomic_ce_count.astype(jnp.float32) / subtask_ce_batch_size
                if subtask_ce_batch_size
                else jnp.mean(atomic_ce_rows.astype(jnp.float32))
            ),
            "zt_weighted_loss": zero if zt_output is None else zt_loss_weight * zt_output.total_loss,
            "zm_weighted_loss": zm_loss_weight * zm_output.total_loss,
        }

    # state.step is intentionally unchanged between accumulation micro-steps;
    # use an independent monotonic fold value so noise/time samples still differ.
    train_rng = jax.random.fold_in(rng, rng_fold_value)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model, train_rng)
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    nnx.update(model, optax.apply_updates(params, updates))
    if frozen_codebook is not None:
        # The full zM loss already stop-gradients code directions. Restoring
        # the value here additionally blocks optimizer weight decay without
        # changing the optimizer tree, so a 60k train_state remains resumable.
        model.codebook.value = frozen_codebook
    new_params = nnx.state(model)
    new_state = dataclasses.replace(
        state,
        step=state.step + advance_step.astype(state.step.dtype),
        params=new_params,
        opt_state=new_opt_state,
    )
    info |= {"loss": loss, "grad_norm": optax.global_norm(grads)}
    return new_state, info


def _to_jax_batch(
    batch_np: dict,
    data_sharding: jax.sharding.NamedSharding,
) -> tuple[_model.Observation, jax.Array, dict[str, jax.Array]]:
    observation, actions = batch_to_observation(batch_np)
    observation = jax.tree.map(lambda x: jax.make_array_from_process_local_data(data_sharding, x), observation)
    actions = jax.make_array_from_process_local_data(data_sharding, actions)
    extras = {
        key: jax.make_array_from_process_local_data(data_sharding, batch_np[key])
        for key in (
            "atomic_prompt_tokens",
            "atomic_prompt_mask",
            "atomic_weights",
            "atomic_supervision_mask",
            "atomic_composition_weights",
            "atomic_composition_confidence",
            "atomic_composition_mask",
            "tcp_twist_delta",
            "subtask_prompt_tokens",
            "subtask_prompt_mask",
            "subtask_target_tokens",
            "subtask_target_mask",
            "atomic_target_tokens",
            "atomic_target_mask",
            "zt_teacher_directions",
            "zt_teacher_valid",
        )
        if key in batch_np
    }
    return observation, actions, extras


def main() -> None:
    args = parse_args()
    if args.resume and args.restore_full_state is not None:
        raise ValueError("--resume and --restore-full-state are mutually exclusive")
    restore_step = None
    if args.restore_full_state is not None:
        if not args.restore_full_state.is_dir():
            raise FileNotFoundError(
                f"full-state checkpoint step not found: {args.restore_full_state}"
            )
        if not (args.restore_full_state / "train_state" / "_METADATA").is_file():
            raise FileNotFoundError(
                f"checkpoint has no complete train_state: {args.restore_full_state}"
            )
        if not (args.restore_full_state / "params" / "_METADATA").is_file():
            raise FileNotFoundError(
                f"checkpoint has no complete params: {args.restore_full_state}"
            )
        try:
            restore_step = int(args.restore_full_state.name)
        except ValueError as exc:
            raise ValueError(
                "--restore-full-state must name a numeric checkpoint step directory"
            ) from exc
        if not args.zt_teacher_sidecar:
            raise ValueError(
                "--restore-full-state requires --zt-teacher-sidecar so the frozen "
                "teacher codebook can be installed and verified"
            )
    fsdp_devices = args.devices if args.fsdp_devices is None else args.fsdp_devices
    if args.batch_size % args.devices:
        raise ValueError("batch-size must be divisible by devices")
    if fsdp_devices <= 0 or args.devices % fsdp_devices:
        raise ValueError("fsdp-devices must be positive and divide devices")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient-accumulation-steps must be positive")
    if not 0 <= args.zt_fraction < 1:
        raise ValueError("zt-fraction must be in [0, 1)")
    if args.zt_loss_weight < 0 or args.zm_loss_weight < 0:
        raise ValueError("branch loss weights must be non-negative")
    if min(
        args.coefficient_loss_weight,
        args.text_flow_loss_weight,
        args.text_atomic_loss_weight,
        args.full_atomic_loss_weight,
        args.codebook_loss_weight,
        args.subtask_ce_loss_weight,
        args.atomic_composition_loss_weight,
    ) < 0:
        raise ValueError("loss weights must be non-negative")
    if args.subprompt_warmup_steps < 0:
        raise ValueError("subprompt-warmup-steps must be non-negative")
    if not 0 <= args.subprompt_probability_after_warmup <= 1:
        raise ValueError("subprompt probability must be in [0, 1]")
    if not 0 <= args.atomic_prompt_probability <= 1:
        raise ValueError("atomic-prompt-probability must be in [0, 1]")
    if not 0 <= args.atomic_text_ce_probability <= 1:
        raise ValueError("atomic-text-ce-probability must be in [0, 1]")
    zt_batch_size = (
        args.batch_size if args.paired_zt_zm else round(args.batch_size * args.zt_fraction)
    )
    if args.paired_zt_zm:
        if args.zt_fraction != 0.5:
            raise ValueError("paired ZT/ZM uses two equal routes; set --zt-fraction 0.5")
        if args.zt_loss_weight <= 0 or args.zm_loss_weight <= 0:
            raise ValueError("paired ZT/ZM requires positive weights for both routes")
        if args.freeze_codebook:
            raise ValueError("paired ZT/ZM must leave the codebook trainable for the ZT route")
        if args.zt_teacher_sidecar:
            raise ValueError("paired ZT/ZM reuses its online ZT output; do not pass a sidecar")
    elif zt_batch_size < 0 or zt_batch_size >= args.batch_size:
        raise ValueError("zt-fraction leaves an invalid zT or empty zM shard")
    if zt_batch_size == 0 and args.zt_loss_weight != 0:
        raise ValueError("zt-loss-weight must be 0 when zt-fraction is 0")
    zm_batch_size = args.batch_size if args.paired_zt_zm else args.batch_size - zt_batch_size
    if args.subtask_ce_batch_size < 0 or args.subtask_ce_batch_size > zm_batch_size:
        raise ValueError("subtask-ce-batch-size must be in [0, zM batch size]")
    if args.subtask_ce_batch_size and args.subtask_ce_batch_size % args.devices:
        raise ValueError("subtask-ce-batch-size must be divisible by --devices")
    roots = tuple(args.dataset_root) if args.dataset_root else DEFAULT_ROOTS
    if missing := [root for root in roots if not Path(root, "meta", "info.json").is_file()]:
        raise FileNotFoundError(f"missing dataset roots: {missing}")
    if not Path(args.norm_assets_dir, args.norm_asset_id, "norm_stats.json").is_file():
        raise FileNotFoundError("norm_stats.json not found under norm-assets-dir/norm-asset-id")
    if not Path(args.base_params).is_dir():
        raise FileNotFoundError(f"π0.5 base params not found: {args.base_params}")
    tcp_twist_quantile_range = _load_tcp_twist_quantile_range(args.tcp_twist_norm)
    phase_start_step = args.initial_step if args.phase_start_step is None else args.phase_start_step
    if phase_start_step < 0 or args.initial_step < 0:
        raise ValueError("initial/phase-start steps must be non-negative")
    if args.max_token_len < 192:
        raise ValueError(
            "original PI0.5 Marvin tokenization serializes the full padded 32-D state; "
            "the audited union2375 corpus requires --max-token-len at least 192"
        )
    if args.zt_teacher_sidecar:
        if len(args.zt_teacher_sidecar) != len(roots):
            raise ValueError(
                "repeat --zt-teacher-sidecar once per dataset root: "
                f"{len(args.zt_teacher_sidecar)} != {len(roots)}"
            )
        expected_teacher_contract = {
            "version": 1,
            "prompt": "atomic",
            "norm_asset_id": args.norm_asset_id,
            "max_token_len": args.max_token_len,
            "coefficient_target": args.coefficient_target,
        }
        reference_codebook = None
        for root, teacher_sidecar in zip(roots, args.zt_teacher_sidecar, strict=True):
            if not teacher_sidecar.is_dir():
                raise FileNotFoundError(f"zT teacher sidecar not found: {teacher_sidecar}")
            manifest_path = teacher_sidecar / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"zT teacher manifest not found: {manifest_path}")
            teacher_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            mismatches = {
                key: (teacher_manifest.get(key), expected)
                for key, expected in expected_teacher_contract.items()
                if teacher_manifest.get(key) != expected
            }
            teacher_dataset = Path(str(teacher_manifest.get("dataset_root", "")))
            if teacher_dataset.resolve() != Path(root).resolve():
                mismatches["dataset_root"] = (
                    teacher_manifest.get("dataset_root"),
                    str(Path(root).resolve()),
                )
            teacher_norm_root = Path(str(teacher_manifest.get("norm_assets_dir", "")))
            if teacher_norm_root.resolve() != Path(args.norm_assets_dir).resolve():
                mismatches["norm_assets_dir"] = (
                    teacher_manifest.get("norm_assets_dir"),
                    str(Path(args.norm_assets_dir).resolve()),
                )
            if mismatches:
                raise ValueError(f"zT teacher sidecar contract mismatch: {mismatches}")
            codebook = np.load(teacher_sidecar / "codebook.npy")
            if reference_codebook is None:
                reference_codebook = codebook
            elif codebook.shape != reference_codebook.shape or not np.allclose(
                codebook, reference_codebook, atol=1.0e-6
            ):
                raise ValueError("zT teacher sidecars do not share one identical codebook")

    # The local π0.5 vendor exposes the standard Marvin/JAX recipe under this
    # name. Dataset, model, optimizer schedule, checkpointing, and freezing
    # fields used by Atomic training are replaced explicitly below.
    base_config = _config.get_config("pi05_hdf5_dscrew_v3_jax")
    model_config = AtomicPi05Config(
        max_token_len=args.max_token_len,
        coefficient_target_kind=args.coefficient_target,
        coefficient_target_dim=14 if args.coefficient_target == "joint_delta" else 12,
        enable_layerwise_atomic_flow=args.layerwise_atomic_flow,
        spherical_visual_latent=args.spherical_visual_latent,
        visual_max_update_angle_deg=args.visual_max_update_angle_deg,
        visual_rotation_loss_weight=args.visual_rotation_loss_weight,
        visual_rotation_free_angle_deg=args.visual_rotation_free_angle_deg,
        visual_rotation_loss_warmup_steps=args.visual_rotation_loss_warmup_steps,
        fast_action_ce_loss_weight=0.0,
        text_flow_loss_weight=args.text_flow_loss_weight,
        coefficient_loss_weight=args.coefficient_loss_weight,
        text_atomic_loss_weight=args.text_atomic_loss_weight,
        atomic_loss_weight=args.full_atomic_loss_weight,
        full_atomic_huber_delta_deg=args.full_atomic_huber_delta_deg,
        codebook_loss_weight=args.codebook_loss_weight,
        subtask_ce_loss_weight=args.subtask_ce_loss_weight,
        atomic_composition_loss_weight=args.atomic_composition_loss_weight,
        freeze_vision_encoder=not args.unfreeze_vision,
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
        weight_loader=AtomicPi05CheckpointLoader(args.base_params),
        lr_schedule=lr_schedule,
        optimizer=(
            base_config.optimizer
            if args.gradient_accumulation_steps == 1
            else GradientAccumulationOptimizer(
                base_config.optimizer,
                args.gradient_accumulation_steps,
            )
        ),
        # The Atomic model owns the freezing policy: keep the pretrained
        # SigLIP visual encoder fixed while allowing PaliGemma language,
        # Action Expert, and the atomic heads/adapters to learn.
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
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    # OpenPI may configure logging while being imported; force our training
    # logger so a nohup run always records step metrics and checkpoints.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    logging.info("config=%s", config)
    logging.info(
        "micro global batch=%d (%d/GPU), accumulation=%d, effective batch/update=%d, "
        "vision_frozen=%s, fsdp_devices=%d",
        args.batch_size,
        args.batch_size // args.devices,
        args.gradient_accumulation_steps,
        args.batch_size * args.gradient_accumulation_steps,
        model_config.freeze_vision_encoder,
        fsdp_devices,
    )
    logging.info(
        "TCP relative-delta quantile range q99-q01=%s; normalization=2*delta/range",
        tcp_twist_quantile_range.tolist(),
    )

    loader = build_atomic_loader(
        roots,
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        action_horizon=model_config.action_horizon,
        max_token_len=model_config.max_token_len,
        # The joint z_T/z_M objective has no FAST token CE. Avoid preparing a
        # 250-token FAST sequence in every worker merely as unused side data.
        include_fast=False,
        subtask_max_token_len=model_config.subtask_max_token_len,
        zt_teacher_sidecar=args.zt_teacher_sidecar or None,
        atomic_composition_sidecar=args.atomic_composition_sidecar,
        pad_subtask_horizon=args.pad_subtask_horizon,
    )
    data_iter = iter(loader)
    first_np = next(data_iter)
    first_weights = first_np["atomic_weights"]
    first_mask = first_np["atomic_supervision_mask"]
    first_count = (first_weights > 0.0).sum(axis=-1)
    first_stay = first_mask & (first_weights[..., 12] > 0.0)
    logging.info(
        "first batch actions=%s state=%s atomic=%.3f single=%.3f dual=%.3f stay=%.3f",
        first_np["actions"].shape,
        first_np["state"].shape,
        float(first_mask.mean()),
        float((first_mask & (first_count == 1) & ~first_stay).mean()),
        float((first_mask & (first_count == 2)).mean()),
        float(first_stay.mean()),
    )
    # Do not initialize JAX's multithreaded GPU runtime before PyTorch has
    # forked its DataLoader workers. Forking after jax.device_count() can
    # deadlock while constructing the first large video batch.
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
    # Both an in-place resume and an external full-state restore need only an
    # abstract params/optimizer template here. Materializing base parameters
    # first would occupy the complete GPU pool and then duplicate it while
    # Orbax restores the requested checkpoint.
    state, state_sharding = base_train.init_train_state(
        config,
        init_rng,
        mesh,
        resume=resuming or args.restore_full_state is not None,
    )
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
    checkpoint_loader_view = SimpleNamespace(
        data_config=lambda: SimpleNamespace(norm_stats=None, asset_id=None)
    )
    if resuming:
        state = _restore_state_to_requested_sharding(
            checkpoint_manager, state, state_sharding
        )
        # This resume path intentionally requires the same device/FSDP
        # topology as the saved run. Orbax restores that sharding directly.
        # Re-sharding the complete params+optimizer tree through an identity
        # JIT duplicates roughly 100 GiB of state and can OOM before training.
    elif args.restore_full_state is not None:
        source_manager, source_available = _checkpoints.initialize_checkpoint_dir(
            args.restore_full_state.parent,
            keep_period=None,
            overwrite=False,
            resume=True,
        )
        if not source_available or restore_step not in source_manager.all_steps():
            raise FileNotFoundError(
                f"checkpoint manager cannot see requested step {restore_step} under "
                f"{args.restore_full_state.parent}"
            )
        state = _restore_state_to_requested_sharding(
            source_manager, state, state_sharding, step=restore_step
        )
        logging.info(
            "restored complete external train_state step=%d from %s",
            restore_step,
            args.restore_full_state,
        )
        restored_model = nnx.merge(state.model_def, state.params)
        teacher_codebook = np.load(args.zt_teacher_sidecar[0] / "codebook.npy")
        if teacher_codebook.shape != restored_model.codebook.value.shape:
            raise ValueError(
                "teacher codebook shape does not match restored model: "
                f"{teacher_codebook.shape} != {restored_model.codebook.value.shape}"
            )
        codebook_sharding = restored_model.codebook.value.sharding
        restored_model.codebook.value = jax.device_put(
            teacher_codebook.astype(np.float32), codebook_sharding
        )
        state = dataclasses.replace(state, params=nnx.state(restored_model))
        del restored_model
        logging.info(
            "installed teacher codebook from %s while preserving restored Adam state",
            args.zt_teacher_sidecar[0] / "codebook.npy",
        )
    elif args.initial_step:
        state = dataclasses.replace(state, step=jnp.asarray(args.initial_step, dtype=state.step.dtype))
    jax.block_until_ready(state)
    # Resume initialization is abstract: params contain ShapeDtypeStructs until
    # Orbax restore completes. Validate checkpoint values only after the state
    # is materialized, while keeping the early structural shape check above.
    if args.zt_teacher_sidecar:
        restored_model = nnx.merge(state.model_def, state.params)
        teacher_codebook = np.load(args.zt_teacher_sidecar[0] / "codebook.npy")
        current_codebook = np.asarray(jax.device_get(restored_model.codebook.value))
        current_codebook = current_codebook / np.maximum(
            np.linalg.norm(current_codebook, axis=-1, keepdims=True), 1e-8
        )
        if teacher_codebook.shape != current_codebook.shape or not np.allclose(
            teacher_codebook, current_codebook, atol=2e-3
        ):
            raise ValueError(
                "training checkpoint codebook does not match frozen zT teacher sidecar"
            )
        logging.info("verified frozen zT teacher codebook matches training checkpoint")
        del restored_model
    # ``phase_start_step`` anchors the prompt curriculum only.  A resumed run
    # must execute ``--steps`` additional optimizer updates from the restored
    # checkpoint instead of reusing that curriculum anchor as its terminal
    # step (which made a 60k resume with a 40k anchor exit immediately).
    start_step = int(jax.device_get(state.step))
    final_step = start_step + config.num_train_steps
    ptrain = jax.jit(
        functools.partial(
            train_step,
            config,
            tcp_twist_quantile_range_jax,
            phase_start_step,
            zt_batch_size,
            args.paired_zt_zm,
            args.zt_loss_weight,
            args.zm_loss_weight,
            args.freeze_codebook,
            args.subtask_ce_batch_size,
            args.subprompt_warmup_steps,
            args.subprompt_probability_after_warmup,
            args.atomic_prompt_probability,
            args.atomic_text_ce_probability,
        ),
        in_shardings=(replicated, state_sharding, data_sharding, replicated, replicated),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    logging.info("initialized AtomicPi05 on %d devices", args.devices)

    current_np = first_np
    micro_step = 0
    accumulated_info = None
    while int(state.step) < final_step:
        batch = _to_jax_batch(current_np, data_sharding)
        boundary = (micro_step + 1) % args.gradient_accumulation_steps == 0
        state, info = ptrain(
            train_rng,
            state,
            batch,
            jnp.asarray(micro_step, dtype=jnp.int32),
            jnp.asarray(boundary),
        )
        accumulated_info = (
            info
            if accumulated_info is None
            else jax.tree.map(lambda total, value: total + value, accumulated_info, info)
        )
        micro_step += 1
        if not boundary:
            try:
                current_np = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                current_np = next(data_iter)
            continue

        step = int(state.step)
        if step % args.log_interval == 0 or step == 1:
            mean_info = jax.tree.map(
                lambda value: value / args.gradient_accumulation_steps,
                accumulated_info,
            )
            values = {key: float(value) for key, value in jax.device_get(mean_info).items()}
            logging.info("step=%d %s", step, " ".join(f"{key}={value:.5f}" for key, value in values.items()))
        accumulated_info = None
        if not args.skip_checkpoint and (step % config.save_interval == 0 or step == final_step):
            _checkpoints.save_state(checkpoint_manager, state, checkpoint_loader_view, step)
            logging.info("submitted full OpenPI checkpoint step=%d", step)
        try:
            current_np = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            current_np = next(data_iter)
    if not args.skip_checkpoint:
        logging.info("waiting for full checkpoint writes to finish")
        checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main()
