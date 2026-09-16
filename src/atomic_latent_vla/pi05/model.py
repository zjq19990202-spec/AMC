"""Clean native-JAX π0.5 model with atomic and force-conditioned latents.

The model keeps exactly two pretrained streams from π0.5:

* PaliGemma/SigLIP encodes image + prompt + discretized robot state;
* Gemma-300M Flow Action Expert denoises a 50 x 32 action chunk and receives
  flow time through its original AdaRMS condition.

Everything else in this file is new: Q1--Q4, two arm-specific 13-atom
codebooks (26 learned prototypes total), a Q1-only
DCT-coefficient decoder, and a z_M-only deep Action-Expert side adapter. The
optional force second stage adds one shared-attention correction per arm before
each layer's arm fusion, using a learned scalar gate at every depth;
subtask text is retained only as an auxiliary autoregressive continuation from
the clean VLM prefix. There is no event memory, transition planner, MoE, CFG,
PyTorch model, or second action expert.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
from flax import struct
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import siglip as _siglip
from openpi.shared import array_typing as at

from .config import AtomicPi05Config
from .force import (
    ForceConditioner,
    ForceModulationOutput,
    future_force_delta_target,
    temporal_masked_mean,
)
from .gemma_adapter import (
    AtomicGemmaModule,
    CoefficientDiTModule,
    fuse_intermediate_arm_latents,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def make_attn_mask(input_mask: jax.Array, mask_ar: jax.Array) -> jax.Array:
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumulative = jnp.cumsum(mask_ar, axis=1)
    causal = cumulative[:, None, :] <= cumulative[:, :, None]
    valid = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(causal, valid)


def pack_sequence_by_mask(
    values: jax.Array, mask: jax.Array, max_len: int
) -> tuple[jax.Array, jax.Array]:
    """Left-pack valid sequence entries while preserving their order.

    This is intentionally the same operation used by OpenPI's π0.5 FAST
    auxiliary path.  FAST has a variable length language/state prefix and a
    causal action-token suffix, while the JAX training step needs static
    tensor shapes for FSDP compilation.
    """

    if values.ndim not in (2, 3):
        raise ValueError(f"expected rank-2 or rank-3 sequence values, got {values.shape}")
    mask_i = mask.astype(jnp.int32)
    positions = jnp.maximum(jnp.cumsum(mask_i, axis=1) - 1, 0)
    keep_mask = jnp.logical_and(mask, positions < max_len)
    positions = jnp.clip(positions, 0, max_len - 1)
    weights = jax.nn.one_hot(positions, max_len, dtype=jnp.float32)
    weights = weights * keep_mask[:, :, None].astype(jnp.float32)
    if values.ndim == 3:
        packed = jnp.einsum("blo,bld->bod", weights.astype(values.dtype), values)
    else:
        packed = jnp.sum(weights.astype(values.dtype) * values[:, :, None], axis=1)
    packed_mask = jnp.arange(max_len)[None, :] < jnp.sum(
        keep_mask.astype(jnp.int32), axis=1, keepdims=True
    )
    return packed, packed_mask


def take_batch_rows(values: jax.Array, indices: jax.Array) -> jax.Array:
    """Gather batch rows with an FSDP-safe differentiable primitive.

    ``jnp.take`` currently emits an invalid bounds-check broadcast when its
    gather is differentiated across OpenPI's joint batch/FSDP mesh.  The
    selected indices come from ``top_k`` over the same batch and are therefore
    known in bounds; using ``promise_in_bounds`` avoids that broken verifier
    path while retaining the correct gather/scatter gradient.
    """

    if values.ndim < 1 or indices.ndim != 1:
        raise ValueError("take_batch_rows expects values[B,...] and indices[K]")
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=tuple(range(1, values.ndim)),
        collapsed_slice_dims=(0,),
        start_index_map=(0,),
    )
    return jax.lax.gather(
        values,
        indices[:, None],
        dimension_numbers,
        slice_sizes=(1, *values.shape[1:]),
        mode="promise_in_bounds",
    )


def posemb_sincos(position: jax.Array, dim: int) -> jax.Array:
    if dim % 2:
        raise ValueError("π0.5 time embedding width must be even")
    fraction = jnp.linspace(0.0, 1.0, dim // 2)
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    angles = position[..., None] * (2 * jnp.pi / period)
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


def l2_normalize(value: jax.Array) -> jax.Array:
    return value / jnp.maximum(jnp.linalg.norm(value, axis=-1, keepdims=True), 1e-8)


def spherical_tangent_update(
    base: jax.Array,
    tangent: jax.Array,
    max_angle_rad: float,
) -> jax.Array:
    """Apply a smooth bounded tangent update and return a unit vector."""

    output_dtype = base.dtype
    base = l2_normalize(base.astype(jnp.float32))
    tangent = tangent.astype(jnp.float32)
    tangent = tangent - jnp.sum(tangent * base, axis=-1, keepdims=True) * base
    max_angle = jnp.asarray(max_angle_rad, dtype=jnp.float32)
    # Express sin(angle) * tangent/||tangent|| as a smooth scalar times
    # tangent. The epsilon is inside the squared norm so the zero-initialized
    # force output retains a finite identity Jacobian on its first update.
    normalized_sq_norm = (
        jnp.sum(jnp.square(tangent), axis=-1, keepdims=True) / jnp.square(max_angle)
    )
    safe_normalized_norm = jnp.sqrt(normalized_sq_norm + 1.0e-12)
    angle = max_angle * jnp.tanh(safe_normalized_norm)
    tangent_scale = jnp.sin(angle) / (max_angle * safe_normalized_norm)
    updated = jnp.cos(angle) * base + tangent_scale * tangent
    return l2_normalize(updated).astype(output_dtype)


def visual_rotation_hinge_loss(
    directions: jax.Array,
    visual_latents: jax.Array,
    *,
    free_angle_rad: float,
    max_angle_rad: float,
) -> tuple[jax.Array, jax.Array]:
    """Return normalized squared geodesic excess and the mean angle."""

    directions_fp32 = l2_normalize(directions.astype(jnp.float32))
    visual_fp32 = l2_normalize(visual_latents.astype(jnp.float32))
    cosine = jnp.clip(
        jnp.sum(directions_fp32 * visual_fp32, axis=-1), -1.0, 1.0
    )
    tangent = visual_fp32 - cosine[..., None] * directions_fp32
    # The epsilon gives the exactly aligned path a finite autodiff Jacobian.
    sine = jnp.sqrt(jnp.sum(jnp.square(tangent), axis=-1) + 1.0e-12)
    angle = jnp.arctan2(sine, cosine)
    free_angle = jnp.asarray(free_angle_rad, dtype=angle.dtype)
    max_angle = jnp.asarray(max_angle_rad, dtype=angle.dtype)
    normalized_excess = jax.nn.relu(angle - free_angle) / max_angle
    return jnp.mean(jnp.square(normalized_excess)), jnp.mean(angle)


def huber_angular_distance(
    left: jax.Array,
    right: jax.Array,
    *,
    delta_rad: float,
) -> jax.Array:
    """Stable geodesic pull, optionally Huber-smoothed below ``delta``.

    Both operands must already be unit vectors. ``atan2(sin(theta),
    cos(theta))`` avoids ``acos``' poorly conditioned derivative near aligned
    vectors. The quadratic cap makes the optimum smooth instead of retaining a
    unit angular gradient all the way to exactly zero.
    """

    cosine = jnp.clip(jnp.sum(left * right, axis=-1), -1.0, 1.0)
    tangent = left - cosine[..., None] * right
    sine = jnp.linalg.norm(tangent, axis=-1)
    angle = jnp.arctan2(sine, cosine)
    # A zero transition angle disables Huber smoothing and yields the plain
    # geodesic angle. Keep this as a Python branch so autodiff never traces
    # the otherwise inactive division by zero below.
    if delta_rad <= 0.0:
        return angle
    delta = jnp.asarray(delta_rad, dtype=angle.dtype)
    return jnp.where(
        angle < delta,
        0.5 * jnp.square(angle) / delta,
        angle - 0.5 * delta,
    )


def dct_prefix(trajectory: jax.Array, coefficients: int) -> jax.Array:
    """Return the first orthonormal DCT-II coefficients of ``[B,H,D]``.

    The coefficient DiT is supervised directly in this basis.  There is no
    inverse-DCT trajectory reconstruction in its loss: the prefix is a compact
    4 x 6 Cartesian action code, not a hand-defined smooth trajectory target.
    """

    horizon = trajectory.shape[-2]
    if not 0 < coefficients <= horizon:
        raise ValueError("coefficients must be in [1, trajectory horizon]")
    time = jnp.arange(horizon, dtype=trajectory.dtype)
    frequency = jnp.arange(horizon, dtype=trajectory.dtype)
    basis = jnp.cos(jnp.pi / horizon * (time[None, :] + 0.5) * frequency[:, None])
    normalizer = jnp.sqrt(jnp.asarray(2.0 / horizon, trajectory.dtype))
    basis = basis * normalizer
    basis = basis.at[0].set(jnp.asarray(1.0 / jnp.sqrt(horizon), trajectory.dtype))
    dct = jnp.einsum("bhd,kh->bkd", trajectory, basis)
    return dct[:, :coefficients, :]


def recover_flow_endpoint(noisy: jax.Array, timestep: jax.Array, velocity: jax.Array) -> jax.Array:
    """Recover the clean endpoint for ``x_t=t*noise+(1-t)*data``.

    We retain velocity parameterization for π0.5 compatibility, but supervise
    this recovered coefficient action directly, as in Wall-OSS-0.5's
    Action-Space Supervision.  Its squared error is exactly ``t**2`` times
    the velocity-field squared error, so high-noise examples carry the global
    action-code supervision without an ad-hoc loss weight.
    """

    return noisy - timestep[:, None, None] * velocity


@struct.dataclass
class AtomicTargets:
    """Up to two atomic labels; ``-1`` marks an absent second label."""

    labels: jax.Array  # [B, 2] int32
    weights: jax.Array  # [B, 2] float32
    supervision_mask: jax.Array  # [B, 2] bool in [right, left] order

    def valid(self) -> jax.Array:
        return (self.labels >= 0) & (self.weights > 0)

    def count(self) -> jax.Array:
        return jnp.sum(self.valid(), axis=-1)


@struct.dataclass
class AtomicCompositionTargets:
    """Gate mixtures routed as Dual Top-2 in zT or Drop Top-5 in zM."""

    weights: jax.Array  # [B, 2, 13] float32
    confidence: jax.Array  # [B, 2] float32, entropy-derived
    supervision_mask: jax.Array  # [B, 2] bool


@struct.dataclass
class FastActionTokens:
    """Released π0.5 FAST token contract for an action-token CE target."""

    tokens: jax.Array  # [B, T] int32
    mask: jax.Array  # [B, T] bool
    ar_mask: jax.Array  # [B, T] bool
    loss_mask: jax.Array  # [B, T] bool


@struct.dataclass
class AtomicLosses:
    ranking: jax.Array
    ratio_kl: jax.Array
    codebook: jax.Array
    perplexity: jax.Array
    right_perplexity: jax.Array
    left_perplexity: jax.Array


@struct.dataclass
class CoefficientLosses:
    """All Q1 DCT supervision terms, logged independently."""

    velocity: jax.Array
    wall: jax.Array
    wall_weight: jax.Array
    total: jax.Array
    right: jax.Array
    left: jax.Array


@struct.dataclass
class AtomicPi05Output:
    flow_loss: jax.Array
    coefficient_loss: jax.Array
    coefficient_losses: CoefficientLosses
    total_loss: jax.Array
    text_atomic_loss: jax.Array
    atomic_losses: AtomicLosses
    text_atomic_losses: AtomicLosses
    quantity_loss: jax.Array
    subtask_ce_loss: jax.Array
    direction: jax.Array
    z_text: jax.Array
    z_model: jax.Array


@struct.dataclass
class AtomicTextStageOutput:
    """Stage-A terms from one text/state prefix and the shared Action Expert."""

    flow_loss: jax.Array
    flow_unweighted_loss: jax.Array
    flow_left_loss: jax.Array
    flow_right_loss: jax.Array
    flow_active_loss: jax.Array
    flow_left_motion_share: jax.Array
    flow_right_motion_share: jax.Array
    coefficient_loss: jax.Array
    coefficient_losses: CoefficientLosses
    fast_action_ce_loss: jax.Array
    atomic_total_loss: jax.Array
    projection_loss: jax.Array
    total_loss: jax.Array
    atomic_losses: AtomicLosses
    direction: jax.Array
    z_text: jax.Array


@struct.dataclass
class AtomicFullStageOutput:
    """Full-observation z_M losses with a stop-gradient text/state teacher."""

    flow_loss: jax.Array
    flow_unweighted_loss: jax.Array
    flow_left_loss: jax.Array
    flow_right_loss: jax.Array
    flow_active_loss: jax.Array
    flow_left_motion_share: jax.Array
    flow_right_motion_share: jax.Array
    atomic_total_loss: jax.Array
    text_teacher_cosine_loss: jax.Array
    text_teacher_cosine_similarity: jax.Array
    # Legacy dashboard slot. ZM no longer evaluates the angular/Huber target.
    strict_atomic_huber_loss: jax.Array
    visual_rotation_loss: jax.Array
    visual_rotation_mean_angle_rad: jax.Array
    visual_rotation_loss_scale: jax.Array
    total_loss: jax.Array
    atomic_losses: AtomicLosses
    projection_loss: jax.Array
    # Legacy dashboard slot. Composition KL is no longer evaluated.
    composition_kl_loss: jax.Array
    quantity_loss: jax.Array
    subtask_ce_loss: jax.Array
    direction: jax.Array
    z_model: jax.Array


@struct.dataclass
class ForceStageOutput:
    """Losses and latent diagnostics for the optional force second stage."""

    flow_loss: jax.Array
    future_force_loss: jax.Array
    future_force_raw_loss: jax.Array
    future_force_coarse_loss: jax.Array
    delta_z_regularization: jax.Array
    base_flow_loss: jax.Array
    improvement_loss: jax.Array
    force_rotation_loss: jax.Array
    force_rotation_mean_angle_rad: jax.Array
    total_loss: jax.Array
    force_latent: jax.Array
    delta_z: jax.Array
    z_exec: jax.Array
    predicted_future_force_delta: jax.Array


@struct.dataclass
class ForcePolicyContext:
    """Cached slow VLA state reused by each ten-step force update."""

    prefix_mask: jax.Array
    kv_cache: object
    z_model: jax.Array
    force_latent: jax.Array
    slow_history_tokens: jax.Array
    slow_history_token_mask: jax.Array
    layerwise_arm_latents: jax.Array


def rtc_committed_mask(update_offset: jax.Array, horizon: int) -> jax.Array:
    """Return ``[B,H]`` mask for the fixed-horizon already-executed prefix."""

    if update_offset.ndim != 1:
        raise ValueError("update_offset must have shape [batch]")
    return jnp.arange(horizon)[None, :] < update_offset[:, None]


def rtc_flow_batch(
    actions: jax.Array,
    noise: jax.Array,
    flow_time: jax.Array,
    update_offset: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build fixed-0:50 RTC inputs without reading a moving action target.

    Returns noisy actions, per-token flow time, committed mask and target
    velocity. Prefix actions are exactly clean demonstration actions; only
    the suffix is interpolated toward Gaussian noise.
    """

    if actions.shape != noise.shape or actions.ndim != 3:
        raise ValueError("actions and noise must share shape [B,H,D]")
    if flow_time.shape != actions.shape[:1]:
        raise ValueError("flow_time must have shape [batch]")
    if update_offset.shape != actions.shape[:1]:
        raise ValueError("update_offset must have shape [batch]")
    committed = rtc_committed_mask(update_offset, actions.shape[1])
    token_time = jnp.where(committed, 0.0, flow_time[:, None])
    noisy = token_time[..., None] * noise + (1.0 - token_time[..., None]) * actions
    return noisy, token_time, committed, noise - actions


def rtc_flow_loss(
    error: jax.Array,
    committed_mask: jax.Array,
) -> jax.Array:
    """Uniform suffix-only flow MSE from the Training-Time RTC objective."""

    if error.ndim != 3 or committed_mask.shape != error.shape[:2]:
        raise ValueError("RTC error/mask must have shapes [B,H,D] and [B,H]")
    weights = jnp.logical_not(committed_mask).astype(error.dtype)
    return jnp.sum(jnp.square(error) * weights[..., None]) / jnp.maximum(
        jnp.sum(weights) * error.shape[-1], 1
    )


def bimanual_flow_losses(
    predicted_velocity: jax.Array,
    target_velocity: jax.Array,
    normalized_actions: jax.Array,
    *,
    atomic_arm_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return PI0.5 Flow loss plus real-arm diagnostics.

    The native action contract is ``[left q(7), grip, right q(7), grip]``.
    Dimensions 16:32 are zero-padded data coordinates, but stock pi0.5 still
    gives them Gaussian noise and trains their Flow endpoint back to zero. If
    ``atomic_arm_mask`` is supplied in [right,left] order, the optimization
    objective instead covers only the seven joints of each supervised arm;
    both grippers, the other arm, and padding are excluded. ZM omits this mask
    and therefore keeps the exact stock 32-D objective.
    """

    if predicted_velocity.shape != target_velocity.shape:
        raise ValueError("predicted and target velocity shapes must match")
    if predicted_velocity.ndim != 3 or predicted_velocity.shape[-1] != 32:
        raise ValueError("pi0.5 flow expects [B,H,32] velocity tensors")
    if normalized_actions.shape != predicted_velocity.shape:
        raise ValueError("normalized_actions must match the controlled velocity shape")
    squared_error = jnp.square(predicted_velocity - target_velocity)
    stock_flow_loss = jnp.mean(squared_error)
    real_error = squared_error[..., :16]
    if atomic_arm_mask is None:
        # Match openpi.models.pi0.Pi0.compute_loss exactly: mean over all 32
        # action coordinates, followed by the trainer's batch/horizon reduction.
        flow_loss = stock_flow_loss
        per_sample_left = jnp.mean(real_error[..., :8], axis=(1, 2))
        per_sample_right = jnp.mean(real_error[..., 8:16], axis=(1, 2))
        left_loss = jnp.mean(per_sample_left)
        right_loss = jnp.mean(per_sample_right)
    else:
        if atomic_arm_mask.shape != predicted_velocity.shape[:1] + (2,):
            raise ValueError("atomic_arm_mask must have shape [B,2] in [right,left] order")
        arm_mask = atomic_arm_mask.astype(squared_error.dtype)
        right_mask = arm_mask[:, 0]
        left_mask = arm_mask[:, 1]
        left_joint_error = squared_error[..., :7]
        right_joint_error = squared_error[..., 8:15]
        horizon = predicted_velocity.shape[1]
        left_denominator = jnp.maximum(jnp.sum(left_mask) * horizon * 7, 1.0)
        right_denominator = jnp.maximum(jnp.sum(right_mask) * horizon * 7, 1.0)
        left_loss = jnp.sum(left_joint_error * left_mask[:, None, None]) / left_denominator
        right_loss = jnp.sum(right_joint_error * right_mask[:, None, None]) / right_denominator
        denominator = jnp.maximum(
            (jnp.sum(left_mask) + jnp.sum(right_mask)) * horizon * 7,
            1.0,
        )
        flow_loss = (
            jnp.sum(left_joint_error * left_mask[:, None, None])
            + jnp.sum(right_joint_error * right_mask[:, None, None])
        ) / denominator

    if atomic_arm_mask is None:
        left_motion = jnp.mean(jnp.square(normalized_actions[..., :7]), axis=(1, 2))
        right_motion = jnp.mean(jnp.square(normalized_actions[..., 8:15]), axis=(1, 2))
        motion = jnp.stack([left_motion, right_motion], axis=-1)
        motion_total = jnp.sum(motion, axis=-1, keepdims=True)
        motion_share = jnp.where(
            motion_total > 1.0e-8,
            motion / jnp.maximum(motion_total, 1.0e-8),
            jnp.full_like(motion, 0.5),
        )
        per_sample_arm_loss = jnp.stack([per_sample_left, per_sample_right], axis=-1)
        active_loss = jnp.mean(jnp.sum(motion_share * per_sample_arm_loss, axis=-1))
    else:
        # For masked zT flow, report shares from the actual supervised arms,
        # independent of whether a labelled horizon happens to be stationary.
        supervised = jnp.stack([left_mask, right_mask], axis=-1)
        supervised_count = jnp.sum(supervised, axis=-1, keepdims=True)
        motion_share = jnp.where(
            supervised_count > 0,
            supervised / jnp.maximum(supervised_count, 1.0),
            jnp.full_like(supervised, 0.5),
        )
        active_loss = flow_loss
    motion_share = jax.lax.stop_gradient(motion_share)
    mean_share = jnp.mean(motion_share, axis=0)
    return (
        flow_loss,
        stock_flow_loss,
        left_loss,
        right_loss,
        active_loss,
        mean_share[0],
        mean_share[1],
    )


def rtc_clamp_prefix(
    actions: jax.Array,
    executed_actions: jax.Array,
    committed_mask: jax.Array,
) -> jax.Array:
    """Hard inpainting constraint applied after every inference Euler step."""

    if actions.shape != executed_actions.shape or committed_mask.shape != actions.shape[:2]:
        raise ValueError("RTC clamp expects matching actions and [B,H] mask")
    return jnp.where(committed_mask[..., None], executed_actions, actions)


class AtomicQueries(nnx.Module):
    """Two query pairs producing right/left latents in one shared code space.

    Q1/Q2 form the right direction/detail latent; Q3/Q4 form the left one.
    Projection heads, tangent bases and state encoder are shared across arms.
    """

    num_queries: int = 4

    def __init__(self, prefix_dim: int, config: AtomicPi05Config, *, rngs: nnx.Rngs):
        self.config = config
        d = config.latent_dim
        if config.active_state_dim % 2:
            raise ValueError("active_state_dim must split evenly across two arms")
        self.arm_state_dim = config.active_state_dim // 2
        self.query_tokens = nnx.Param(
            0.02 * jax.random.normal(rngs(), (self.num_queries, prefix_dim))
        )
        self.q1_in = nnx.Linear(prefix_dim, d, rngs=rngs)
        self.q2_in = nnx.Linear(prefix_dim, d, rngs=rngs)
        self.q3_in = nnx.Linear(prefix_dim, d, rngs=rngs)
        self.q4_in = nnx.Linear(prefix_dim, d, rngs=rngs)
        # The 16-D robot state has heterogeneous units. Give it one small
        # nonlinear projection before it meets VLM-derived query vectors.
        # Keep the released/right-arm state projection width checkpoint
        # compatible.  Each arm receives a role-centred full state: the
        # non-acting arm first and the acting arm second.  For the right arm
        # this is exactly the original CR1 [left,right] ordering; the left-arm
        # view swaps the halves to [right,left].
        self.state_in = nnx.Linear(
            config.active_state_dim, config.state_encoder_hidden_dim, rngs=rngs
        )
        self.state_out = nnx.Linear(config.state_encoder_hidden_dim, d, rngs=rngs)
        # The raw Q1 feature is projected to the unit-sphere z_T used by both
        # atomic alignment and the compact DCT decoder. No coarse-trajectory
        # information can bypass this normalized bottleneck.
        self.direction = nnx.Linear(d, d, rngs=rngs)
        # The even query in each arm pair emits coefficients over a shared
        # learned tangent basis, so visual detail can change atomic composition
        # without an unconstrained dense shift.  The historical q2_scale and
        # quantity heads stay as dormant checkpoint-compatibility placeholders;
        # Q3/Q4 are now exclusively the left-arm direction/detail pair.
        self.q2_scale = nnx.Linear(d, 1, rngs=rngs)
        self.q3_detail = nnx.Linear(d, config.detail_dim, rngs=rngs)
        self.basis_seed = nnx.Param(0.02 * jax.random.normal(rngs(), (config.detail_dim, d)))
        self.quantity_goal = nnx.Linear(d, d, rngs=rngs)
        self.quantity_value = nnx.Linear(d, 1, rngs=rngs)

    def _encode_state(self, state: jax.Array) -> jax.Array:
        return self.state_out(nnx.swish(self.state_in(state)))

    def _arm_state_views(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        if state.shape[-1] != 2 * self.arm_state_dim:
            raise ValueError(
                f"expected {2 * self.arm_state_dim} state values, got {state.shape[-1]}"
            )
        left = state[..., : self.arm_state_dim]
        right = state[..., self.arm_state_dim :]
        right_view = state
        left_view = jnp.concatenate([right, left], axis=-1)
        return right_view, left_view

    def tokens(self, batch_size: int, dtype: jnp.dtype) -> jax.Array:
        return jnp.broadcast_to(
            self.query_tokens.value[None],
            (batch_size, self.num_queries, self.query_tokens.value.shape[-1]),
        ).astype(dtype)

    def layerwise_composer_params(self) -> dict[str, object]:
        """Expose the shared query heads to the scanned layerwise composer.

        These are references to the existing NNX parameters, not copied heads.
        Flow gradients from every intermediate depth therefore update the same
        Q1--Q4 projections used by the final supervised ``z_M``.
        """

        def linear_params(layer: nnx.Linear) -> dict[str, jax.Array]:
            params = {"kernel": layer.kernel.value}
            if layer.bias is not None:
                params["bias"] = layer.bias.value
            return params

        return {
            "q1_in": linear_params(self.q1_in),
            "q3_in": linear_params(self.q3_in),
            "direction": linear_params(self.direction),
            "q3_detail": linear_params(self.q3_detail),
            "basis_seed": self.basis_seed.value,
        }

    def text_latent(self, query_hidden: jax.Array, state: jax.Array) -> jax.Array:
        return self.text_arm_latents(query_hidden, state)[:, 0]

    def text_arm_latents(self, query_hidden: jax.Array, state: jax.Array) -> jax.Array:
        """Return ordered [right, left] coarse latents from Q1 and Q3."""

        # State already enters PaliGemma as discretized prefix tokens.  The
        # historical continuous-state MLP duplicated that path and let Q1
        # infer dataset priors without using the atomic text.  Keep its
        # parameters checkpoint-compatible, but make them dormant.
        del state
        right = self.q1_in(query_hidden[:, 0])
        left = self.q1_in(query_hidden[:, 2])
        return jnp.stack([right, left], axis=1)

    def text_direction(self, query_hidden: jax.Array, state: jax.Array) -> jax.Array:
        return l2_normalize(self.direction(self.text_latent(query_hidden, state)))

    def text_arm_directions(self, query_hidden: jax.Array, state: jax.Array) -> jax.Array:
        return l2_normalize(self.direction(self.text_arm_latents(query_hidden, state)))

    def _tangent_basis(self, direction: jax.Array) -> jax.Array:
        """Project learned seeds to Q1's tangent plane and orthonormalize."""

        seeds = jnp.broadcast_to(
            self.basis_seed.value[None],
            (direction.shape[0], self.config.detail_dim, direction.shape[-1]),
        )
        projected = (
            seeds - jnp.einsum("bkd,bd->bk", seeds, direction)[..., None] * direction[:, None, :]
        )
        # QR acts on [B,D,K]. Use fp32 for a stable differentiable basis even
        # when the PaliGemma query stream itself is bfloat16.
        basis, _ = jnp.linalg.qr(
            jnp.swapaxes(projected, -1, -2).astype(jnp.float32), mode="reduced"
        )
        return jnp.swapaxes(basis, -1, -2).astype(direction.dtype)

    def _arm_latent(
        self,
        direction_hidden: jax.Array,
        detail_hidden: jax.Array,
        arm_state: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        del arm_state
        raw_direction = self.q1_in(direction_hidden)
        detail_feature = self.q3_in(detail_hidden)
        direction = l2_normalize(self.direction(raw_direction))
        detail = jnp.tanh(self.q3_detail(detail_feature))
        tangent_shift = jnp.einsum("bkd,bk->bd", self._tangent_basis(direction), detail)
        latent = direction + self.config.max_shift_scale * tangent_shift
        if self.config.spherical_visual_latent:
            latent = spherical_tangent_update(
                direction,
                self.config.max_shift_scale * tangent_shift,
                jnp.deg2rad(self.config.visual_max_update_angle_deg),
            )
        return raw_direction, direction, latent

    def __call__(self, query_hidden: jax.Array, state: jax.Array) -> tuple[jax.Array, ...]:
        if query_hidden.shape[1] != self.num_queries:
            raise ValueError(
                f"expected {self.num_queries} query tokens, got {query_hidden.shape[1]}"
            )
        del state
        z_q1_full, direction, z_right = self._arm_latent(
            query_hidden[:, 0], query_hidden[:, 1], None
        )
        _, left_direction, z_left = self._arm_latent(query_hidden[:, 2], query_hidden[:, 3], None)
        z_model = jnp.stack([z_right, z_left], axis=1)
        quantity_value = jnp.zeros((query_hidden.shape[0],), dtype=query_hidden.dtype)
        return z_q1_full, direction, z_model, left_direction, quantity_value


class AtomicPi05(_model.BaseModel):
    """New π0.5-derived model; it does not instantiate or subclass ``Pi0``."""

    def __init__(self, config: AtomicPi05Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        paligemma = _gemma.get_config(config.paligemma_variant)
        action_expert = _gemma.get_config(config.action_expert_variant)
        self.PaliGemma = nnx.Dict(
            llm=nnx_bridge.ToNNX(
                AtomicGemmaModule(
                    configs=[paligemma, action_expert],
                    embed_dtype=config.dtype,
                    latent_dim=config.latent_dim,
                    adapter_condition_hidden_dim=config.adapter_condition_hidden_dim,
                    adapter_bottleneck_dim=config.adapter_dim,
                    arm_mlp_hidden_dim=config.arm_mlp_hidden_dim,
                    arm_fusion_hidden_dim=config.arm_fusion_hidden_dim,
                    atomic_cross_attention_num_heads=(
                        config.atomic_cross_attention_num_heads
                    ),
                    atomic_cross_attention_head_dim=(
                        config.atomic_cross_attention_head_dim
                    ),
                    adarms=True,
                )
            ),
            img=nnx_bridge.ToNNX(
                _siglip.Module(
                    num_classes=paligemma.width,
                    variant="So400m/14",
                    pool_type="none",
                    scan=True,
                    dtype_mm=config.dtype,
                )
            ),
        )
        self.PaliGemma.llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])
        self.PaliGemma.img.lazy_init(
            next(iter(config.fake_obs().images.values())), train=False, rngs=rngs
        )
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert.width, rngs=rngs)
        self.time_mlp_in = nnx.Linear(action_expert.width, action_expert.width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(action_expert.width, action_expert.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert.width, config.action_dim, rngs=rngs)
        self.queries = AtomicQueries(paligemma.width, config, rngs=rngs)
        coefficient_dit = config.coefficient_dit_config()
        self.coefficient_in_proj = nnx.Linear(
            config.coefficient_target_dim, coefficient_dit.width, rngs=rngs
        )
        self.coefficient_time_in = nnx.Linear(
            coefficient_dit.width, coefficient_dit.width, rngs=rngs
        )
        self.coefficient_time_out = nnx.Linear(
            coefficient_dit.width, coefficient_dit.width, rngs=rngs
        )
        self.coefficient_arm_in = nnx.Linear(
            config.latent_dim, config.arm_mlp_hidden_dim, rngs=rngs
        )
        self.coefficient_arm_out = nnx.Linear(
            config.arm_mlp_hidden_dim, config.latent_dim, rngs=rngs
        )
        self.coefficient_arm_embedding = nnx.Param(
            0.02 * jax.random.normal(rngs(), (2, config.latent_dim))
        )
        self.coefficient_fusion_norm = nnx.LayerNorm(
            2 * config.latent_dim, use_bias=True, rngs=rngs
        )
        self.coefficient_fusion_in = nnx.Linear(
            2 * config.latent_dim, config.arm_fusion_hidden_dim, rngs=rngs
        )
        self.coefficient_fusion_out = nnx.Linear(
            config.arm_fusion_hidden_dim, config.latent_dim, rngs=rngs
        )
        self.coefficient_z_in = nnx.Linear(config.latent_dim, coefficient_dit.width, rngs=rngs)
        self.coefficient_out_proj = nnx.Linear(
            coefficient_dit.width, config.coefficient_target_dim, rngs=rngs
        )
        self.coefficient_dit = nnx_bridge.ToNNX(CoefficientDiTModule(coefficient_dit, config.dtype))
        self.coefficient_dit.lazy_init(rngs=rngs, method="init")
        # Arm identity is part of the atomic action. Keep independent right
        # and left spherical prototypes instead of collapsing both arms onto
        # one shared 13-way codebook.
        self.codebook = nnx.Param(
            0.02
            * jax.random.normal(
                rngs(), (config.arm_count, config.num_atomic_codes, config.latent_dim)
            )
        )
        if config.enable_force_stage:
            self.force_conditioner = ForceConditioner(
                prefix_dim=paligemma.width,
                action_dim=config.action_dim,
                latent_dim=config.latent_dim,
                num_layers=paligemma.depth,
                config=config,
                rngs=rngs,
            )
        self.deterministic = True

    def _controlled_state(self, state: jax.Array) -> jax.Array:
        """Keep pi0.5's 16-D state; the loader holds left state at episode start."""

        return state[:, : self.config.active_state_dim]

    @staticmethod
    def _unwrap_nnx_params(tree: object) -> object:
        """Convert a nested ToNNX parameter mapping to an array pytree."""

        if hasattr(tree, "items"):
            return {key: AtomicPi05._unwrap_nnx_params(value) for key, value in tree.items()}
        return tree.value if hasattr(tree, "value") else tree

    def _layerwise_atomic_inputs(self) -> tuple[dict[str, object], dict[str, object], jax.Array]:
        """Return shared Q-composer, arm-fusion and final-norm parameters."""

        # ``ToNNX`` exposes each Linen parameter subtree as a direct NNX
        # attribute (``arm_fusion``, ``final_norm``, ...), not under a common
        # ``params`` mapping.
        fusion_params = self._unwrap_nnx_params(self.PaliGemma.llm.arm_fusion)
        final_norm_scale = self._unwrap_nnx_params(self.PaliGemma.llm.final_norm["scale"])
        return (
            self.queries.layerwise_composer_params(),
            fusion_params,
            final_norm_scale,
        )

    def _mask_action_condition(self, actions: jax.Array) -> jax.Array:
        """Keep stock pi0.5's full 32-D Flow path for the bimanual model.

        The conditional legacy branch preserves old right-only evaluators;
        production bimanual configs use start=0/dim=16 and therefore return
        the full padded tensor unchanged.
        """

        if (
            self.config.controlled_action_start == 0
            and self.config.controlled_action_dim == self.config.active_action_dim
        ):
            return actions

        indices = jnp.arange(self.config.action_dim)
        keep = (indices >= self.config.controlled_action_start) & (
            indices < self.config.controlled_action_start + self.config.controlled_action_dim
        )
        return actions * keep[None, None, :]

    def _controlled_actions(self, actions: jax.Array) -> jax.Array:
        start = self.config.controlled_action_start
        return actions[..., start : start + self.config.controlled_action_dim]

    def embed_prefix(
        self, observation: _model.Observation
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        tokens: list[jax.Array] = []
        masks: list[jax.Array] = []
        ar: list[bool] = []
        for name in observation.images:
            image_tokens, _ = self.PaliGemma.img(observation.images[name], train=False)
            tokens.append(image_tokens)
            masks.append(
                einops.repeat(observation.image_masks[name], "b -> b n", n=image_tokens.shape[1])
            )
            ar.extend([False] * image_tokens.shape[1])
        if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
            raise ValueError("AtomicPi05 requires pi0.5 tokenized prompt and prompt mask")
        text_tokens = self.PaliGemma.llm(observation.tokenized_prompt, method="embed")
        tokens.append(text_tokens)
        masks.append(observation.tokenized_prompt_mask)
        ar.extend([False] * text_tokens.shape[1])
        return jnp.concatenate(tokens, axis=1), jnp.concatenate(masks, axis=1), jnp.asarray(ar)

    def _query_prefix_mask(self, prefix_mask: jax.Array, prefix_ar: jax.Array) -> jax.Array:
        """Prefix cannot read queries; each arm has an isolated query pair."""

        batch_size, prefix_len = prefix_mask.shape
        query_len = self.queries.num_queries
        prefix_attention = make_attn_mask(prefix_mask, prefix_ar)
        full = jnp.zeros(
            (batch_size, prefix_len + query_len, prefix_len + query_len), dtype=jnp.bool_
        )
        full = full.at[:, :prefix_len, :prefix_len].set(prefix_attention)
        # Query rows can read every valid prefix token, but no prefix row can
        # read any query. This keeps the pretrained prefix/KV cache unchanged.
        full = full.at[:, prefix_len:, :prefix_len].set(prefix_mask[:, None, :])
        if query_len != 4:
            raise ValueError("bimanual query mask expects Q1--Q4")
        branch_mask = jnp.asarray(
            [
                [True, False, False, False],
                [True, True, False, False],
                [False, False, True, False],
                [False, False, True, True],
            ],
            dtype=jnp.bool_,
        )
        return full.at[:, prefix_len:, prefix_len:].set(branch_mask)

    def _subtask_prefix_mask(
        self,
        prefix_mask: jax.Array,
        prefix_ar: jax.Array,
        subtask_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Build two independent tails after the clean VLM prefix.

        Q1--Q4 read the current VLM prefix and their same-arm query history.
        The optional teacher-forced subtask suffix reads the VLM prefix and its
        own causal history. Neither tail can write back into the prefix or read
        the other tail, and neither is visible to the Action Expert.
        """

        batch_size, prefix_len = prefix_mask.shape
        query_len = self.queries.num_queries
        subtask_len = 0 if subtask_mask is None else subtask_mask.shape[1]
        atomic_start = prefix_len
        subtask_start = atomic_start + query_len
        total_len = subtask_start + subtask_len
        full = jnp.zeros((batch_size, total_len, total_len), dtype=jnp.bool_)
        full = full.at[:, :subtask_start, :subtask_start].set(
            self._query_prefix_mask(prefix_mask, prefix_ar)
        )
        if subtask_mask is not None:
            full = full.at[:, subtask_start:, :prefix_len].set(
                subtask_mask[:, :, None] & prefix_mask[:, None, :]
            )
            subtask_causal = jnp.tril(jnp.ones((subtask_len, subtask_len), dtype=jnp.bool_))
            full = full.at[:, subtask_start:, subtask_start:].set(
                subtask_mask[:, :, None] & subtask_mask[:, None, :] & subtask_causal[None]
            )
        return full

    def _subtask_ce_loss(
        self,
        prefix_hidden: jax.Array,
        prefix_mask: jax.Array,
        subtask_hidden: jax.Array,
        target_tokens: jax.Array,
        target_mask: jax.Array,
    ) -> jax.Array:
        """Teacher-forced next-token CE rooted at the clean VLM prefix.

        The last valid prefix hidden predicts target token 0; each teacher token
        hidden predicts the following token. The final teacher hidden has no
        next target and is intentionally unused.
        """

        if subtask_hidden.shape[:2] != target_tokens.shape:
            raise ValueError("subtask hidden and target tokens must share [batch, length]")
        if prefix_hidden.shape[:2] != prefix_mask.shape:
            raise ValueError("prefix hidden and mask must share [batch, length]")
        prefix_indices = jnp.arange(prefix_mask.shape[1])[None, :]
        last_valid = jnp.max(jnp.where(prefix_mask, prefix_indices, -1), axis=1)
        last_valid = jnp.maximum(last_valid, 0)
        root_hidden = jnp.take_along_axis(prefix_hidden, last_valid[:, None, None], axis=1)
        previous_hidden = jnp.concatenate([root_hidden, subtask_hidden[:, :-1]], axis=1)
        token_logp: list[jax.Array] = []
        chunk_size = self.config.subtask_ce_decode_chunk_size
        for start in range(0, previous_hidden.shape[1], chunk_size):
            end = min(start + chunk_size, previous_hidden.shape[1])
            logits = self.PaliGemma.llm(previous_hidden[:, start:end], method="decode")
            logp = jax.nn.log_softmax(logits, axis=-1)
            selected = jnp.take_along_axis(logp, target_tokens[:, start:end, None], axis=-1)[..., 0]
            token_logp.append(selected)
        token_logp = jnp.concatenate(token_logp, axis=1)
        mask = target_mask.astype(token_logp.dtype)
        return -jnp.sum(token_logp * mask) / jnp.maximum(jnp.sum(mask), 1)

    def _with_prompt(
        self, observation: _model.Observation, tokens: jax.Array, mask: jax.Array
    ) -> _model.Observation:
        return _model.Observation(
            images=observation.images,
            image_masks=observation.image_masks,
            state=observation.state,
            tokenized_prompt=tokens,
            tokenized_prompt_mask=mask,
            token_ar_mask=observation.token_ar_mask,
            token_loss_mask=observation.token_loss_mask,
        )

    def _text_only(
        self,
        observation: _model.Observation,
        tokens: jax.Array | None = None,
        mask: jax.Array | None = None,
    ) -> _model.Observation:
        return _model.Observation(
            images={},
            image_masks={},
            state=observation.state,
            tokenized_prompt=observation.tokenized_prompt if tokens is None else tokens,
            tokenized_prompt_mask=observation.tokenized_prompt_mask if mask is None else mask,
            token_ar_mask=None,
            token_loss_mask=None,
        )

    def _prefix_forward(
        self,
        observation: _model.Observation,
        *,
        subtask_tokens: jax.Array | None = None,
        subtask_mask: jax.Array | None = None,
        return_prefix_hidden: bool = False,
        return_layerwise_latents: bool = False,
        return_layerwise_arm_latents: bool = False,
    ) -> tuple:
        if (subtask_tokens is None) != (subtask_mask is None):
            raise ValueError("subtask_tokens and subtask_mask must be provided together")
        prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix(observation)
        query_tokens = self.queries.tokens(prefix_tokens.shape[0], prefix_tokens.dtype)
        query_mask = jnp.ones(query_tokens.shape[:2], dtype=jnp.bool_)
        if subtask_tokens is None:
            subtask_embeddings = jnp.zeros(
                (prefix_tokens.shape[0], 0, prefix_tokens.shape[-1]),
                dtype=prefix_tokens.dtype,
            )
            subtask_input_mask = jnp.zeros((prefix_tokens.shape[0], 0), dtype=jnp.bool_)
        else:
            if subtask_tokens.ndim != 2 or subtask_mask.shape != subtask_tokens.shape:
                raise ValueError("subtask tokens/mask must both have shape [batch, length]")
            if subtask_tokens.shape[0] != prefix_tokens.shape[0]:
                raise ValueError("subtask token batch must match observation batch")
            if subtask_tokens.shape[1] > self.config.subtask_max_token_len:
                raise ValueError(
                    f"subtask target exceeds {self.config.subtask_max_token_len} tokens"
                )
            subtask_embeddings = self.PaliGemma.llm(subtask_tokens, method="embed").astype(
                prefix_tokens.dtype
            )
            subtask_input_mask = subtask_mask.astype(jnp.bool_)
        combined_tokens = jnp.concatenate(
            [prefix_tokens, query_tokens, subtask_embeddings],
            axis=1,
        )
        # Atomic and subtask tails are parallel children of the clean prefix.
        # They reuse post-prefix positions while masks keep them isolated.
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        prefix_count = jnp.sum(prefix_mask, axis=1, keepdims=True)
        query_positions = (
            prefix_count + jnp.arange(query_tokens.shape[1], dtype=prefix_count.dtype)[None]
        )
        subtask_positions = prefix_count + jnp.maximum(
            jnp.cumsum(subtask_input_mask, axis=1) - 1, 0
        )
        positions = jnp.concatenate(
            [prefix_positions, query_positions, subtask_positions],
            axis=1,
        )
        layerwise_kwargs: dict[str, object] = {}
        if return_layerwise_arm_latents:
            return_layerwise_latents = True
        if return_layerwise_latents:
            query_params, fusion_params, final_norm_scale = self._layerwise_atomic_inputs()
            layerwise_kwargs = {
                "query_start": prefix_tokens.shape[1],
                "query_params": query_params,
                "fusion_params": fusion_params,
                "query_final_norm_scale": final_norm_scale,
                "max_shift_scale": self.config.max_shift_scale,
                "spherical_visual_latent": self.config.spherical_visual_latent,
                "visual_max_update_angle_rad": jnp.deg2rad(
                    self.config.visual_max_update_angle_deg
                ),
                "return_layerwise_latents": True,
                "return_layerwise_arm_latents": return_layerwise_arm_latents,
            }
        (combined_out, _), cache_output = self.PaliGemma.llm(
            [combined_tokens, None],
            mask=self._subtask_prefix_mask(
                prefix_mask,
                prefix_ar,
                subtask_input_mask if subtask_tokens is not None else None,
            ),
            positions=positions,
            **layerwise_kwargs,
        )
        if return_layerwise_arm_latents:
            kv_cache, layerwise_latents, layerwise_arm_latents = cache_output
        elif return_layerwise_latents:
            kv_cache, layerwise_latents = cache_output
            layerwise_arm_latents = None
        else:
            kv_cache = cache_output
            layerwise_latents = None
            layerwise_arm_latents = None
        atomic_start = prefix_tokens.shape[1]
        subtask_start = atomic_start + query_tokens.shape[1]
        subtask_ce_loss = jnp.zeros((), combined_out.dtype)
        if subtask_tokens is not None:
            subtask_hidden = combined_out[:, subtask_start:]
            subtask_ce_loss = self._subtask_ce_loss(
                combined_out[:, : prefix_tokens.shape[1]],
                prefix_mask,
                subtask_hidden,
                subtask_tokens,
                subtask_input_mask,
            )
        # Q1--Q4 and teacher-forced text remain physically present in the cache,
        # but the Action Expert can read only the pretrained VLM prefix.
        hidden_tail_mask = jnp.zeros(
            (
                prefix_tokens.shape[0],
                query_tokens.shape[1] + subtask_embeddings.shape[1],
            ),
            dtype=jnp.bool_,
        )
        action_prefix_mask = jnp.concatenate([prefix_mask, hidden_tail_mask], axis=1)
        result = (
            combined_out[:, atomic_start:subtask_start],
            action_prefix_mask,
            kv_cache,
            subtask_ce_loss,
        )
        if return_prefix_hidden:
            result = (*result, combined_out[:, : prefix_tokens.shape[1]])
        if return_layerwise_latents:
            result = (*result, layerwise_latents)
        if return_layerwise_arm_latents:
            result = (*result, layerwise_arm_latents)
        return result

    def _atomic_losses(
        self,
        direction: jax.Array,
        targets: AtomicTargets,
        *,
        update_codes: bool,
        global_step: jax.Array | None = None,
        arm_index: int = 0,
    ) -> AtomicLosses:
        """Align Q1 with one or two simultaneous semantic atom prototypes.

        The old soft-target InfoNCE treated a dual label as mutually
        exclusive class probability, making its two positive atoms compete.
        Two-Way multi-label ranking instead makes the weakest positive outrank
        the strongest negative. A separate positive-only KL retains the FK
        top-2 strength ratio without imposing a linear combination on Q1.
        """

        codebook = self.codebook.value
        # Rank-2 is retained for focused unit tests and legacy right-only
        # utilities. Production bimanual models always store [2,13,D].
        if codebook.ndim == 2:
            codes = l2_normalize(codebook)
        elif codebook.ndim == 3 and codebook.shape[0] == self.config.arm_count:
            codes = l2_normalize(codebook[arm_index])
        else:
            raise ValueError(f"codebook must have shape [13,D] or [2,13,D], got {codebook.shape}")
        valid = targets.valid() & targets.supervision_mask[:, None]
        safe_labels = jnp.clip(targets.labels, 0, self.config.num_atomic_codes - 1)
        similarities = jnp.einsum(
            "bd,ed->be",
            direction,
            jax.lax.stop_gradient(codes) if not update_codes else codes,
        )

        positive_mask = jnp.any(
            jax.nn.one_hot(safe_labels, self.config.num_atomic_codes, dtype=jnp.bool_)
            & valid[..., None],
            axis=1,
        )
        # Keep unsupervised rows numerically finite before masking their loss.
        safe_positive_mask = positive_mask.at[:, 0].set(
            positive_mask[:, 0] | ~targets.supervision_mask
        )
        negative_mask = ~safe_positive_mask
        scores = similarities / self.config.atomic_temperature
        positive_lse = self.config.atomic_two_way_positive_temperature * jax.nn.logsumexp(
            jnp.where(
                safe_positive_mask,
                -scores / self.config.atomic_two_way_positive_temperature,
                -jnp.inf,
            ),
            axis=-1,
        )
        negative_lse = self.config.atomic_two_way_negative_temperature * jax.nn.logsumexp(
            jnp.where(
                negative_mask,
                scores / self.config.atomic_two_way_negative_temperature,
                -jnp.inf,
            ),
            axis=-1,
        )
        ranking_row = jax.nn.softplus(positive_lse + negative_lse)

        target_weight = targets.weights * valid
        target_weight = target_weight / jnp.maximum(target_weight.sum(axis=-1, keepdims=True), 1e-8)
        positive_similarities = jnp.take_along_axis(similarities, safe_labels, axis=1)
        # The first slot is a harmless dummy only on unsupervised rows. Every
        # supervised row has a real first atom by the AtomicTargets contract.
        safe_valid = valid.at[:, 0].set(valid[:, 0] | ~targets.supervision_mask)
        positive_log_prob = jax.nn.log_softmax(
            jnp.where(
                safe_valid,
                positive_similarities / self.config.atomic_ratio_temperature,
                -jnp.inf,
            ),
            axis=-1,
        )
        log_target = jnp.log(jnp.maximum(target_weight, 1e-8))
        ratio_row = jnp.sum(
            jnp.where(valid, target_weight * (log_target - positive_log_prob), 0.0),
            axis=-1,
        )
        denominator = jnp.maximum(jnp.sum(targets.supervision_mask), 1)
        ranking = jnp.sum(ranking_row * targets.supervision_mask) / denominator
        single = targets.supervision_mask & (targets.count() == 1)
        dual = targets.supervision_mask & (targets.count() == 2)
        # Ratio is defined only for simultaneous dual atoms. Averaging it over
        # singles would silently halve its configured weight on this dataset.
        dual_denominator = jnp.maximum(jnp.sum(dual), 1)
        ratio_kl = jnp.sum(ratio_row * dual) / dual_denominator
        selected = codes[safe_labels[:, 0]]
        single_denominator = jnp.maximum(jnp.sum(single), 1)
        legacy_mse = (
            jnp.sum(
                jnp.mean(jnp.square(jax.lax.stop_gradient(direction) - selected), axis=-1) * single
            )
            / single_denominator
        )
        # Direct geodesic optimization removes the sin(theta) attenuation of
        # 1-cos(theta). Huber smoothing is active only inside the configured
        # tiny terminal angle, and stop_gradient keeps this a code update only
        # rather than a Q1 commitment loss. The explicit scale preserves the
        # former codebook-loss magnitude in the overall objective.
        huber_delta_rad = self.config.codebook_huber_delta_deg * jnp.pi / 180.0
        angular = (
            jnp.sum(
                self.config.codebook_huber_angle_scale
                * huber_angular_distance(
                    jax.lax.stop_gradient(direction),
                    selected,
                    delta_rad=huber_delta_rad,
                )
                * single
            )
            / single_denominator
        )
        if global_step is None:
            codebook = angular
        else:
            use_angular = global_step >= self.config.codebook_angular_start_step
            codebook = jnp.where(use_angular, angular, legacy_mse)
        if not update_codes:
            codebook = jnp.zeros((), direction.dtype)
        # Monitor the same temperature-scaled scores optimized by the atomic
        # loss; the old unscaled metric stayed near 12 even after alignment.
        probabilities = jax.nn.softmax(scores, axis=-1)
        supervised_count = jnp.sum(targets.supervision_mask)
        average = jnp.sum(probabilities * targets.supervision_mask[:, None], axis=0) / jnp.maximum(
            supervised_count, 1
        )
        perplexity = jnp.where(
            supervised_count > 0,
            jnp.exp(-jnp.sum(average * jnp.log(jnp.maximum(average, 1e-8)))),
            jnp.zeros((), average.dtype),
        )
        zero = jnp.zeros((), perplexity.dtype)
        return AtomicLosses(
            ranking,
            ratio_kl,
            codebook,
            perplexity,
            perplexity if arm_index == 0 else zero,
            perplexity if arm_index == 1 else zero,
        )

    def _arm_atomic_losses(
        self,
        directions: jax.Array,
        targets: AtomicTargets,
        *,
        update_codes: bool,
        global_step: jax.Array | None,
    ) -> AtomicLosses:
        """Apply independent right/left 13-way codebooks and combine losses."""

        if directions.ndim == 2:
            return self._atomic_losses(
                directions,
                targets,
                update_codes=update_codes,
                global_step=global_step,
                arm_index=0,
            )
        if directions.ndim != 3 or directions.shape[1] != 2:
            raise ValueError(f"arm directions must have shape [B,2,D], got {directions.shape}")
        if targets.labels.ndim == 2:
            # Backward compatibility for old right-only unit tests/checkpoints.
            return self._atomic_losses(
                directions[:, 0],
                targets,
                update_codes=update_codes,
                global_step=global_step,
                arm_index=0,
            )
        if targets.labels.shape[:2] != directions.shape[:2]:
            raise ValueError(
                "bimanual atomic targets must share [B,2] with directions; "
                f"got {targets.labels.shape} and {directions.shape}"
            )
        arm_targets = [
            AtomicTargets(
                labels=targets.labels[:, arm],
                weights=targets.weights[:, arm],
                supervision_mask=targets.supervision_mask[:, arm],
            )
            for arm in range(2)
        ]
        arm_losses = [
            self._atomic_losses(
                directions[:, arm],
                arm_targets[arm],
                update_codes=update_codes,
                global_step=global_step,
                arm_index=arm,
            )
            for arm in range(2)
        ]

        def weighted(field: str, counts: jax.Array) -> jax.Array:
            values = jnp.stack([getattr(loss, field) for loss in arm_losses])
            return jnp.sum(values * counts) / jnp.maximum(jnp.sum(counts), 1)

        supervised_counts = jnp.stack([jnp.sum(target.supervision_mask) for target in arm_targets])
        dual_counts = jnp.stack(
            [jnp.sum(target.supervision_mask & (target.count() == 2)) for target in arm_targets]
        )
        single_counts = jnp.stack(
            [jnp.sum(target.supervision_mask & (target.count() == 1)) for target in arm_targets]
        )
        right_perplexity = arm_losses[0].right_perplexity
        left_perplexity = arm_losses[1].left_perplexity
        return AtomicLosses(
            ranking=weighted("ranking", supervised_counts),
            ratio_kl=weighted("ratio_kl", dual_counts),
            codebook=weighted("codebook", single_counts),
            perplexity=right_perplexity + left_perplexity,
            right_perplexity=right_perplexity,
            left_perplexity=left_perplexity,
        )

    def _frozen_codebook_projection_loss(
        self,
        directions: jax.Array,
        gate_weights: jax.Array,
        supervision_mask: jax.Array,
    ) -> jax.Array:
        """Align Q to a normalized gate-weighted code direction.

        This directly minimizes ``1-cos(Q, norm(sum_i w_i code_i))``. It
        preserves the gate-weighted compromise direction without interpreting
        bounded cosine logits as a categorical probability distribution.
        """

        if directions.ndim != 3 or directions.shape[1:] != (
            self.config.arm_count,
            self.config.latent_dim,
        ):
            raise ValueError(
                "projection directions must have shape "
                f"[B,{self.config.arm_count},{self.config.latent_dim}]"
            )
        if gate_weights.shape != (
            directions.shape[0],
            self.config.arm_count,
            self.config.num_atomic_codes,
        ):
            raise ValueError("gate weights must have shape [B,arm,code]")
        if supervision_mask.shape != directions.shape[:2]:
            raise ValueError("projection mask must have shape [B,arm]")
        codes = jax.lax.stop_gradient(l2_normalize(self.codebook.value))
        target = jnp.maximum(gate_weights, 0.0)
        target = target / jnp.maximum(target.sum(axis=-1, keepdims=True), 1e-8)
        target_direction = jnp.einsum("bac,acd->bad", target, codes)
        target_norm = jnp.linalg.norm(target_direction, axis=-1)
        valid = supervision_mask & (target_norm > 1e-6)
        fallback = jnp.broadcast_to(codes[None, :, 0, :], target_direction.shape)
        target_direction = l2_normalize(jnp.where(valid[..., None], target_direction, fallback))
        similarity = jnp.clip(
            jnp.sum(l2_normalize(directions) * target_direction, axis=-1),
            -1.0,
            1.0,
        )
        row_loss = 1.0 - similarity
        mask = valid.astype(row_loss.dtype)
        return jnp.sum(row_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    def _frozen_composition_projection_loss(
        self,
        directions: jax.Array,
        targets: AtomicCompositionTargets,
    ) -> jax.Array:
        """Match Q to a routed Dual-Top2 or Drop-Top5 code direction."""

        return self._frozen_codebook_projection_loss(
            directions,
            targets.weights,
            targets.supervision_mask,
        )

    @staticmethod
    def _text_teacher_cosine_alignment(
        student_directions: jax.Array,
        teacher_directions: jax.Array,
        supervision_mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Align visual Q1/Q3 to same-prompt/state text-only Q1/Q3.

        The teacher is stop-gradient by contract. The mask is arm-wise, so a
        mixed block can distill a strict arm while its dropped arm receives
        gate-mixture projection supervision instead.
        """

        if student_directions.ndim != 3 or student_directions.shape[1] != 2:
            raise ValueError("student directions must have shape [B,2,D]")
        if teacher_directions.shape != student_directions.shape:
            raise ValueError("teacher directions must match student [B,2,D]")
        if supervision_mask.shape != student_directions.shape[:2]:
            raise ValueError("teacher supervision mask must have shape [B,2]")
        student = l2_normalize(student_directions)
        teacher = jax.lax.stop_gradient(l2_normalize(teacher_directions))
        cosine = jnp.clip(jnp.sum(student * teacher, axis=-1), -1.0, 1.0)
        mask = supervision_mask.astype(cosine.dtype)
        denominator = jnp.maximum(jnp.sum(mask), 1.0)
        similarity = jnp.sum(cosine * mask) / denominator
        loss = jnp.sum((1.0 - cosine) * mask) / denominator
        return loss, similarity

    def _frozen_full_atomic_huber(
        self,
        directions: jax.Array,
        targets: AtomicTargets,
    ) -> jax.Array:
        """Align strict zM directions to frozen single/dual code mixtures.

        Stage A already established the semantic prototypes. Stage B therefore
        never updates the codebook and does not reuse its discriminative
        Two-Way/ratio objective. A dual target is represented by the normalized
        weighted sum of its two frozen codes, so the same geometric objective
        covers both strict single and strict dual arms.
        """

        if directions.ndim != 3 or directions.shape[1:] != (
            self.config.arm_count,
            self.config.latent_dim,
        ):
            raise ValueError(
                "full atomic directions must have shape "
                f"[B,{self.config.arm_count},{self.config.latent_dim}]"
            )
        if targets.labels.ndim != 3 or targets.labels.shape[:2] != directions.shape[:2]:
            raise ValueError("full atomic labels must have shape [B,arm,slot]")
        if targets.weights.shape != targets.labels.shape:
            raise ValueError("full atomic weights must match labels")
        if targets.supervision_mask.shape != directions.shape[:2]:
            raise ValueError("full atomic supervision mask must have shape [B,arm]")

        codes = jax.lax.stop_gradient(l2_normalize(self.codebook.value))
        valid = targets.valid() & targets.supervision_mask[..., None]
        safe_labels = jnp.clip(targets.labels, 0, self.config.num_atomic_codes - 1)
        selected = jnp.einsum(
            "bakc,acd->bakd",
            jax.nn.one_hot(safe_labels, self.config.num_atomic_codes, dtype=directions.dtype),
            codes.astype(directions.dtype),
        )
        weights = jnp.where(valid, jnp.maximum(targets.weights, 0.0), 0.0)
        weights = weights / jnp.maximum(weights.sum(axis=-1, keepdims=True), 1e-8)
        target_direction = jnp.sum(weights[..., None] * selected, axis=2)
        # Keep masked rows finite; their value is removed below.
        fallback = jnp.broadcast_to(codes[None, :, 0, :], target_direction.shape)
        target_direction = l2_normalize(
            jnp.where(targets.supervision_mask[..., None], target_direction, fallback)
        )
        delta_rad = self.config.full_atomic_huber_delta_deg * jnp.pi / 180.0
        row_loss = self.config.full_atomic_huber_angle_scale * huber_angular_distance(
            l2_normalize(directions),
            target_direction,
            delta_rad=delta_rad,
        )
        mask = targets.supervision_mask.astype(row_loss.dtype)
        return jnp.sum(row_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    def _frozen_full_atomic_perplexities(
        self,
        directions: jax.Array,
        targets: AtomicTargets,
    ) -> tuple[jax.Array, jax.Array]:
        """Read-only Stage-B code usage without evaluating obsolete losses."""

        codes = jax.lax.stop_gradient(l2_normalize(self.codebook.value))
        scores = (
            jnp.einsum("bad,acd->bac", l2_normalize(directions), codes)
            / self.config.atomic_temperature
        )
        probabilities = jax.nn.softmax(scores, axis=-1)
        values = []
        for arm in range(self.config.arm_count):
            mask = targets.supervision_mask[:, arm].astype(probabilities.dtype)
            average = jnp.sum(probabilities[:, arm] * mask[:, None], axis=0) / jnp.maximum(
                jnp.sum(mask), 1.0
            )
            values.append(
                jnp.where(
                    jnp.sum(mask) > 0,
                    jnp.exp(-jnp.sum(average * jnp.log(jnp.maximum(average, 1e-8)))),
                    jnp.zeros((), probabilities.dtype),
                )
            )
        return values[0], values[1]

    def _suffix_velocity(
        self,
        prefix_mask: jax.Array,
        kv_cache: object,
        noisy_actions: jax.Array,
        timestep: jax.Array,
        z_model: jax.Array,
        committed_mask: jax.Array | None = None,
        layerwise_latents: jax.Array | None = None,
        force_tokens: jax.Array | None = None,
    ) -> CoefficientLosses:
        if force_tokens is not None:
            raise ValueError(
                "the matched AFRO route applies force through spherical z_M; "
                "independent Action-Expert force tokens are disabled"
            )
        action_tokens = self.action_in_proj(noisy_actions)
        if timestep.ndim not in (1, 2):
            raise ValueError("timestep must have shape [B] or [B,H]")
        if timestep.ndim == 2 and timestep.shape != action_tokens.shape[:2]:
            raise ValueError(
                f"token timestep must have shape {action_tokens.shape[:2]}, got {timestep.shape}"
            )
        if committed_mask is None:
            committed_mask = jnp.zeros(action_tokens.shape[:2], dtype=jnp.bool_)
        if committed_mask.shape != action_tokens.shape[:2]:
            raise ValueError(
                f"committed mask must have shape {action_tokens.shape[:2]}, got {committed_mask.shape}"
            )
        time = nnx.swish(
            self.time_mlp_out(
                nnx.swish(self.time_mlp_in(posemb_sincos(timestep, action_tokens.shape[-1])))
            )
        )
        suffix_mask = jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_)
        suffix_ar = jnp.asarray([True] + [False] * (self.config.action_horizon - 1))
        suffix_attention = make_attn_mask(suffix_mask, suffix_ar)
        prefix_attention = einops.repeat(prefix_mask, "b p -> b s p", s=action_tokens.shape[1])
        full_attention = jnp.concatenate([prefix_attention, suffix_attention], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        _, fusion_params, _ = self._layerwise_atomic_inputs()
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, action_tokens],
            mask=full_attention,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, time],
            latent_condition=z_model,
            layerwise_arm_latent_condition=layerwise_latents,
            latent_update_mask=~committed_mask,
            fusion_params=fusion_params,
        )
        velocity = self.action_out_proj(suffix_out[:, -self.config.action_horizon :])
        return velocity

    def _joint_layerwise_velocity(
        self,
        observation: _model.Observation,
        noisy_actions: jax.Array,
        timestep: jax.Array,
        committed_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Run Prefix/Q and Action Expert together, one transformer layer at a time.

        Each block first updates the clean Prefix and Q1--Q4 stream, composes
        that depth's ``z_M^(l)`` with the shared final query heads, and applies
        it to the same depth's Action-Expert adapter. No all-layer KV cache
        crosses a second large forward call during training.

        Returns ``(query_hidden_final, velocity, layerwise_fused_latents)``.
        The final query hidden still goes through ``AtomicQueries`` outside
        this function and retains all existing semantic supervision.
        """

        prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix(observation)
        query_tokens = self.queries.tokens(prefix_tokens.shape[0], prefix_tokens.dtype)
        combined_tokens = jnp.concatenate([prefix_tokens, query_tokens], axis=1)

        action_tokens = self.action_in_proj(noisy_actions)
        if timestep.ndim not in (1, 2):
            raise ValueError("timestep must have shape [B] or [B,H]")
        if timestep.ndim == 2 and timestep.shape != action_tokens.shape[:2]:
            raise ValueError(
                f"token timestep must have shape {action_tokens.shape[:2]}, got {timestep.shape}"
            )
        if committed_mask is None:
            committed_mask = jnp.zeros(action_tokens.shape[:2], dtype=jnp.bool_)
        if committed_mask.shape != action_tokens.shape[:2]:
            raise ValueError(
                f"committed mask must have shape {action_tokens.shape[:2]}, got {committed_mask.shape}"
            )
        time = nnx.swish(
            self.time_mlp_out(
                nnx.swish(self.time_mlp_in(posemb_sincos(timestep, action_tokens.shape[-1])))
            )
        )

        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        prefix_count = jnp.sum(prefix_mask, axis=1, keepdims=True)
        query_positions = (
            prefix_count + jnp.arange(query_tokens.shape[1], dtype=prefix_count.dtype)[None]
        )
        action_positions = (
            prefix_count + jnp.arange(action_tokens.shape[1], dtype=prefix_count.dtype)[None]
        )
        positions = jnp.concatenate([prefix_positions, query_positions, action_positions], axis=1)

        prefix_query_attention = self._query_prefix_mask(prefix_mask, prefix_ar)
        prefix_rows = jnp.concatenate(
            [
                prefix_query_attention,
                jnp.zeros(
                    (
                        prefix_tokens.shape[0],
                        combined_tokens.shape[1],
                        action_tokens.shape[1],
                    ),
                    dtype=jnp.bool_,
                ),
            ],
            axis=-1,
        )
        action_prefix_mask = jnp.concatenate(
            [
                prefix_mask,
                jnp.zeros(query_tokens.shape[:2], dtype=jnp.bool_),
            ],
            axis=1,
        )
        action_prefix_attention = einops.repeat(
            action_prefix_mask, "b p -> b s p", s=action_tokens.shape[1]
        )
        suffix_mask = jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_)
        suffix_ar = jnp.asarray([True] + [False] * (self.config.action_horizon - 1))
        action_rows = jnp.concatenate(
            [
                action_prefix_attention,
                make_attn_mask(suffix_mask, suffix_ar),
            ],
            axis=-1,
        )
        full_attention = jnp.concatenate([prefix_rows, action_rows], axis=1)

        query_params, fusion_params, final_norm_scale = self._layerwise_atomic_inputs()
        (combined_out, suffix_out), (_, layerwise_latents) = self.PaliGemma.llm(
            [combined_tokens, action_tokens],
            mask=full_attention,
            positions=positions,
            adarms_cond=[None, time],
            latent_update_mask=~committed_mask,
            query_start=prefix_tokens.shape[1],
            query_params=query_params,
            fusion_params=fusion_params,
            query_final_norm_scale=final_norm_scale,
            max_shift_scale=self.config.max_shift_scale,
            spherical_visual_latent=self.config.spherical_visual_latent,
            visual_max_update_angle_rad=jnp.deg2rad(
                self.config.visual_max_update_angle_deg
            ),
            return_layerwise_latents=True,
        )
        query_hidden = combined_out[:, prefix_tokens.shape[1] :]
        velocity = self.action_out_proj(suffix_out[:, -self.config.action_horizon :])
        return query_hidden, velocity, layerwise_latents

    def _latent(
        self,
        query_hidden: jax.Array,
        active_state: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        return self.queries(query_hidden, active_state)

    def _coefficient_velocity(
        self,
        noisy_coefficients: jax.Array,
        timestep: jax.Array,
        z_text: jax.Array,
    ) -> jax.Array:
        """Q1-only DiT velocity field for the compact ``K x 6`` TCP DCT code."""

        coefficient_tokens = self.coefficient_in_proj(noisy_coefficients)
        time = nnx.swish(
            self.coefficient_time_out(
                nnx.swish(
                    self.coefficient_time_in(posemb_sincos(timestep, coefficient_tokens.shape[-1]))
                )
            )
        )
        # This is the sole context channel: no raw prompt hidden, image token,
        # or raw state token enters the coefficient DiT. Time and z_T meet
        # only in its standard AdaRMS/DiT condition vector.
        condition = time + self.coefficient_z_in(z_text)
        positions = jnp.broadcast_to(
            jnp.arange(coefficient_tokens.shape[1], dtype=jnp.int32)[None],
            coefficient_tokens.shape[:2],
        )
        hidden = self.coefficient_dit(coefficient_tokens, positions, condition)
        return self.coefficient_out_proj(hidden)

    def _fuse_text_arm_latents(self, arm_latents: jax.Array) -> jax.Array:
        """Use the z_M-style ordered shared-MLP fusion for coarse z_T."""

        if arm_latents.ndim != 3 or arm_latents.shape[1:] != (2, self.config.latent_dim):
            raise ValueError(
                "text arm latents must have shape "
                f"[B,2,{self.config.latent_dim}], got {arm_latents.shape}"
            )
        hidden = nnx.swish(self.coefficient_arm_in(arm_latents))
        hidden = self.coefficient_arm_out(hidden)
        hidden = hidden + self.coefficient_arm_embedding.value[None]
        hidden = hidden.reshape(hidden.shape[0], 2 * hidden.shape[-1])
        hidden = self.coefficient_fusion_norm(hidden)
        hidden = nnx.swish(self.coefficient_fusion_in(hidden))
        return self.coefficient_fusion_out(hidden)

    def _coefficient_loss(
        self,
        rng: jax.Array,
        z_text: jax.Array,
        normalized_tcp_twist_delta: jax.Array,
        global_step: jax.Array | None,
        arm_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Masked bimanual Flow loss on normalized [right 6D, left 6D] modes."""

        noise_rng, time_rng = jax.random.split(rng)
        if arm_mask is None:
            arm_mask = jnp.ones(z_text.shape[:2], dtype=jnp.bool_)
        if arm_mask.shape != z_text.shape[:2]:
            raise ValueError(
                f"coefficient arm mask must have shape {z_text.shape[:2]}, got {arm_mask.shape}"
            )
        # A missing arm contributes a fixed zero token and receives no DCT
        # gradient through the joint fusion trunk.
        masked_z_text = jnp.where(arm_mask[..., None], z_text, jnp.zeros_like(z_text))
        fused_z_text = self._fuse_text_arm_latents(masked_z_text)
        expected_shape = (
            fused_z_text.shape[0],
            self.config.action_horizon,
            self.config.coefficient_target_dim,
        )
        if normalized_tcp_twist_delta.shape != expected_shape:
            raise ValueError(
                "coefficient_tcp_twist_delta must be normalized base-frame "
                "[B, action_horizon, right6+left6]; "
                f"expected {expected_shape}, got {normalized_tcp_twist_delta.shape}"
            )
        if self.config.coefficient_target_dim % self.config.arm_count:
            raise ValueError("coefficient target dimensions must split evenly across arms")
        arm_dim = self.config.coefficient_target_dim // self.config.arm_count
        dimension_mask = jnp.repeat(arm_mask, arm_dim, axis=-1)[:, None, :]

        def masked_mse(error: jax.Array, mask: jax.Array) -> jax.Array:
            broadcast_mask = jnp.broadcast_to(mask, error.shape)
            return jnp.sum(jnp.square(error) * broadcast_mask) / jnp.maximum(
                jnp.sum(broadcast_mask), 1
            )

        def arm_mse(error: jax.Array, arm: int) -> jax.Array:
            start = arm * arm_dim
            stop = start + arm_dim
            mask = arm_mask[:, arm, None, None]
            return masked_mse(error[..., start:stop], mask)

        target = dct_prefix(normalized_tcp_twist_delta, self.config.coefficient_count)
        noise = jax.random.normal(noise_rng, target.shape)
        time = jax.random.uniform(time_rng, target.shape[:-2], minval=0.001, maxval=1.0)
        noisy = time[:, None, None] * noise + (1.0 - time[:, None, None]) * target
        velocity = self._coefficient_velocity(noisy, time, fused_z_text)
        recovered_coefficients = recover_flow_endpoint(noisy, time, velocity)
        velocity_error = velocity - (noise - target)
        velocity_loss = masked_mse(velocity_error, dimension_mask)
        velocity_right = arm_mse(velocity_error, 0)
        velocity_left = arm_mse(velocity_error, 1)
        # For this flow convention Wall action-space supervision is exactly
        # t**2-weighted velocity error, emphasizing high-noise/global code
        # recovery. Warm up as an ordinary vector field first, then blend.
        wall_error = recovered_coefficients - target
        wall_loss = masked_mse(wall_error, dimension_mask)
        wall_right = arm_mse(wall_error, 0)
        wall_left = arm_mse(wall_error, 1)
        if global_step is None:
            one = jnp.ones((), wall_loss.dtype)
            return CoefficientLosses(
                velocity_loss, wall_loss, one, wall_loss, wall_right, wall_left
            )
        step = jnp.asarray(global_step, wall_loss.dtype)
        warmup = jnp.asarray(self.config.coefficient_velocity_warmup_steps, wall_loss.dtype)
        transition = jnp.asarray(
            max(self.config.coefficient_wall_transition_steps, 1), wall_loss.dtype
        )
        wall_weight = jnp.clip((step - warmup) / transition, 0.0, 1.0)
        total = (1.0 - wall_weight) * velocity_loss + wall_weight * wall_loss
        right = (1.0 - wall_weight) * velocity_right + wall_weight * wall_right
        left = (1.0 - wall_weight) * velocity_left + wall_weight * wall_left
        return CoefficientLosses(velocity_loss, wall_loss, wall_weight, total, right, left)

    def _fast_action_ce_loss(
        self, fast: FastActionTokens, observation: _model.Observation | None = None
    ) -> jax.Array:
        """π0.5 FAST action-token CE, optionally conditioned on frozen ViT tokens.

        The action suffix is strictly causal and receives loss; the prompt and
        discretized state prefix is bidirectional and receives no loss.  No
        image token, Q token, flow suffix, or Action Expert token is present
        here, so FAST cannot bypass the z_T/Q1 backbone via z_M.  When a
        full observation is supplied, its frozen SigLIP tokens are included
        exactly as in the released π0.5 FAST auxiliary path; this is the
        visual stabilization objective used by Stage A.
        """

        token_embeddings = self.PaliGemma.llm(fast.tokens, method="embed")
        prefix_mask = jnp.logical_and(fast.mask, jnp.logical_not(fast.loss_mask))
        suffix_mask_raw = jnp.logical_and(fast.mask, fast.loss_mask)
        prefix_embeddings, prefix_mask = pack_sequence_by_mask(
            token_embeddings, prefix_mask, self.config.fast_action_ce_prefix_len
        )
        suffix_embeddings, suffix_mask = pack_sequence_by_mask(
            token_embeddings, suffix_mask_raw, self.config.fast_action_ce_suffix_len
        )
        suffix_targets, _ = pack_sequence_by_mask(
            fast.tokens, suffix_mask_raw, self.config.fast_action_ce_suffix_len
        )
        token_blocks: list[jax.Array] = []
        mask_blocks: list[jax.Array] = []
        ar_blocks: list[jax.Array] = []
        if observation is not None:
            for name in observation.images:
                image_tokens, _ = self.PaliGemma.img(observation.images[name], train=False)
                token_blocks.append(image_tokens)
                mask_blocks.append(
                    einops.repeat(
                        observation.image_masks[name], "b -> b t", t=image_tokens.shape[1]
                    )
                )
                ar_blocks.append(jnp.zeros(image_tokens.shape[:2], dtype=jnp.bool_))
        token_blocks.append(prefix_embeddings)
        mask_blocks.append(prefix_mask)
        ar_blocks.append(jnp.zeros(prefix_mask.shape, dtype=jnp.bool_))
        suffix_start = sum(block.shape[1] for block in token_blocks)
        token_blocks.append(suffix_embeddings)
        mask_blocks.append(suffix_mask)
        ar_blocks.append(suffix_mask)
        ce_tokens = jnp.concatenate(token_blocks, axis=1)
        ce_input_mask = jnp.concatenate(mask_blocks, axis=1)
        ce_ar_mask = jnp.concatenate(ar_blocks, axis=1)
        ce_mask = make_attn_mask(ce_input_mask, ce_ar_mask)
        ce_positions = jnp.cumsum(ce_input_mask, axis=1) - 1
        (ce_out, _), _ = self.PaliGemma.llm(
            [ce_tokens[:, :-1], None],
            mask=ce_mask[:, :-1, :-1],
            positions=ce_positions[:, :-1],
        )
        # Predict the first suffix token from the preceding final prefix
        # token, followed by next-token predictions inside the suffix.
        pre_logits = ce_out[:, suffix_start - 1 : suffix_start + suffix_targets.shape[1] - 1]
        return self._fast_action_ce_from_hidden(pre_logits, suffix_targets, suffix_mask)

    def _fast_action_ce_from_hidden(
        self,
        pre_logits: jax.Array,
        suffix_targets: jax.Array,
        suffix_mask: jax.Array,
    ) -> jax.Array:
        """Vocabulary CE after a shared PaliGemma forward pass."""

        token_logp: list[jax.Array] = []
        for start in range(0, pre_logits.shape[1], self.config.fast_action_ce_decode_chunk_size):
            end = min(start + self.config.fast_action_ce_decode_chunk_size, pre_logits.shape[1])
            logits = self.PaliGemma.llm(pre_logits[:, start:end], method="decode")
            logp = jax.nn.log_softmax(logits, axis=-1)
            target = jax.nn.one_hot(suffix_targets[:, start:end], logp.shape[-1], dtype=logp.dtype)
            token_logp.append(jnp.sum(target * logp, axis=-1))
        token_logp = jnp.concatenate(token_logp, axis=-1)
        per_row = -jnp.sum(token_logp * suffix_mask, axis=-1) / jnp.maximum(
            jnp.sum(suffix_mask, axis=-1), 1
        )
        return jnp.mean(per_row)

    def _visual_q1_fast_forward(
        self,
        observation: _model.Observation,
        fast: FastActionTokens,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """One shared visual/LLM pass for Q1 and released FAST action CE.

        FAST's prompt/state prefix is also Q1's Stage-A context.  The physical
        order is ``[visual + FAST-prefix, FAST-action, Q1..Q4]`` so the first
        FAST action token is still predicted from the final FAST-prefix token.
        The block mask prevents FAST actions from reading queries, and queries
        from reading target actions.  Thus this is equivalent to two semantic
        branches but has one image encode and one LLM forward/backward.
        """

        token_embeddings = self.PaliGemma.llm(fast.tokens, method="embed")
        fast_prefix_raw = jnp.logical_and(fast.mask, jnp.logical_not(fast.loss_mask))
        suffix_raw = jnp.logical_and(fast.mask, fast.loss_mask)
        fast_prefix, fast_prefix_mask = pack_sequence_by_mask(
            token_embeddings, fast_prefix_raw, self.config.fast_action_ce_prefix_len
        )
        suffix_tokens, suffix_mask = pack_sequence_by_mask(
            token_embeddings, suffix_raw, self.config.fast_action_ce_suffix_len
        )
        suffix_targets, _ = pack_sequence_by_mask(
            fast.tokens, suffix_raw, self.config.fast_action_ce_suffix_len
        )

        image_blocks: list[jax.Array] = []
        image_masks: list[jax.Array] = []
        for name in observation.images:
            image_tokens, _ = self.PaliGemma.img(observation.images[name], train=False)
            image_blocks.append(image_tokens)
            image_masks.append(
                einops.repeat(observation.image_masks[name], "b -> b t", t=image_tokens.shape[1])
            )
        visual_tokens = jnp.concatenate(image_blocks, axis=1)
        visual_mask = jnp.concatenate(image_masks, axis=1)
        prefix_tokens = jnp.concatenate([visual_tokens, fast_prefix], axis=1)
        prefix_mask = jnp.concatenate([visual_mask, fast_prefix_mask], axis=1)
        prefix_len = prefix_tokens.shape[1]
        suffix_len = suffix_tokens.shape[1]

        query_tokens = self.queries.tokens(prefix_tokens.shape[0], prefix_tokens.dtype)
        query_mask = jnp.ones(query_tokens.shape[:2], dtype=jnp.bool_)
        combined_tokens = jnp.concatenate([prefix_tokens, suffix_tokens, query_tokens], axis=1)

        # Both semantic branches retain their original relative RoPE positions:
        # queries begin after the FAST prefix; action tokens also begin there.
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        prefix_count = jnp.sum(prefix_mask, axis=1, keepdims=True)
        suffix_positions = prefix_count + jnp.cumsum(suffix_mask, axis=1) - 1
        query_positions = prefix_count + jnp.arange(query_tokens.shape[1])[None, :]
        positions = jnp.concatenate([prefix_positions, suffix_positions, query_positions], axis=1)

        batch_size = prefix_tokens.shape[0]
        total_len = combined_tokens.shape[1]
        query_start = prefix_len + suffix_len
        full_mask = jnp.zeros((batch_size, total_len, total_len), dtype=jnp.bool_)
        # Frozen π0.5 visual + FAST prompt/state prefix is bidirectional.
        full_mask = full_mask.at[:, :prefix_len, :prefix_len].set(
            make_attn_mask(prefix_mask, jnp.zeros(prefix_mask.shape[1], dtype=jnp.bool_))
        )
        # FAST suffix sees its prefix and its causal action history, never Qs.
        full_mask = full_mask.at[:, prefix_len:query_start, :prefix_len].set(
            prefix_mask[:, None, :]
        )
        full_mask = full_mask.at[:, prefix_len:query_start, prefix_len:query_start].set(
            make_attn_mask(suffix_mask, suffix_mask)
        )
        # Ordered Q tokens read the same visual/FAST context, never actions.
        full_mask = full_mask.at[:, query_start:, :prefix_len].set(prefix_mask[:, None, :])
        query_causal = jnp.asarray(
            [
                [True, False, False, False],
                [True, True, False, False],
                [False, False, True, False],
                [False, False, True, True],
            ],
            dtype=jnp.bool_,
        )
        full_mask = full_mask.at[:, query_start:, query_start:].set(query_causal)

        (combined_out, _), _ = self.PaliGemma.llm(
            [combined_tokens, None], mask=full_mask, positions=positions
        )
        pre_logits = combined_out[:, prefix_len - 1 : query_start - 1]
        return combined_out[:, query_start:], pre_logits, suffix_targets, suffix_mask

    def encode_text_arm_directions(
        self,
        observation: _model.Observation,
    ) -> jax.Array:
        """Encode an atomic-prompt text/state teacher without auxiliary losses."""

        if observation.images or observation.image_masks:
            raise ValueError("text teacher observations must not contain images")
        query_hidden, _, _, _ = self._prefix_forward(observation)
        active_state = self._controlled_state(observation.state)
        return self.queries.text_arm_directions(query_hidden, active_state)

    def compute_text_stage_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        actions: _model.Actions | None = None,
        atomic_targets: AtomicTargets | None,
        atomic_composition_targets: AtomicCompositionTargets | None = None,
        coefficient_tcp_twist_delta: jax.Array | None = None,
        coefficient_arm_mask: jax.Array | None = None,
        fast_action_tokens: FastActionTokens | None = None,
        global_step: jax.Array | None = None,
        update_codes: bool = True,
        train: bool = False,
        return_output: bool = False,
    ) -> jax.Array | AtomicTextStageOutput:
        """Stage A: train text/state z_T and optionally the shared PI0.5 flow.

        ``observation`` deliberately has no image fields. The route is text +
        state -> Q1/Q3 atomic directions. When ``text_flow_loss_weight`` is
        non-zero, those directions condition the same 50x32 PI0.5 Action
        Expert used by z_M; the compact coefficient DiT can remain disabled.
        """

        if observation.images or observation.image_masks:
            raise ValueError("compute_text_stage_loss accepts text/state observations only")
        # The text-only loader already applies π0.5's state/action
        # normalization and PaliGemma prompt tokenization.  OpenPI's generic
        # preprocessor insists on all camera keys even when images are unused,
        # so invoking it here would turn this no-ViT stage into an invalid
        # pseudo-visual observation.
        del train
        coefficient_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        query_hidden, prefix_mask, kv_cache, _ = self._prefix_forward(observation)
        active_state = self._controlled_state(observation.state)
        z_text_arms = self.queries.text_arm_latents(query_hidden, active_state)
        arm_directions = l2_normalize(self.queries.direction(z_text_arms))
        z_text = z_text_arms[:, 0]
        zero = jnp.zeros((), arm_directions.dtype)
        if self.config.text_flow_loss_weight > 0.0:
            if actions is None:
                raise ValueError("actions are required when text_flow_loss_weight is non-zero")
            if atomic_targets is None:
                raise ValueError("zT shared flow requires per-arm atomic targets for loss masking")
            actions = self._mask_action_condition(actions)
            arm_mask = atomic_targets.supervision_mask.astype(actions.dtype)
            if arm_mask.shape != actions.shape[:1] + (2,):
                raise ValueError("zT atomic supervision mask must have shape [B,2]")
            # Native action order is [left7, left_grip, right7, right_grip,
            # padding16], while atomic target order is [right,left]. ZT sees
            # and learns only supervised arm joints; grippers are always zero.
            coordinate_mask = jnp.concatenate(
                [
                    jnp.broadcast_to(arm_mask[:, 1:2], (actions.shape[0], 7)),
                    jnp.zeros((actions.shape[0], 1), dtype=actions.dtype),
                    jnp.broadcast_to(arm_mask[:, 0:1], (actions.shape[0], 7)),
                    jnp.zeros((actions.shape[0], 17), dtype=actions.dtype),
                ],
                axis=-1,
            )[:, None, :]
            actions = actions * coordinate_mask
            noise = (
                self._mask_action_condition(jax.random.normal(noise_rng, actions.shape))
                * coordinate_mask
            )
            time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
            noisy = time[:, None, None] * noise + (1.0 - time[:, None, None]) * actions
            target_velocity = noise - actions
            velocity = self._suffix_velocity(
                prefix_mask,
                kv_cache,
                noisy,
                time,
                # z_T has no visual tangent residual. Its per-arm unit
                # directions enter the shared arm-fusion/FiLM adapters.
                arm_directions,
            )
            (
                flow_loss,
                flow_unweighted_loss,
                flow_left_loss,
                flow_right_loss,
                flow_active_loss,
                flow_left_motion_share,
                flow_right_motion_share,
            ) = bimanual_flow_losses(
                velocity,
                target_velocity,
                actions,
                atomic_arm_mask=atomic_targets.supervision_mask,
            )
        else:
            flow_loss = zero
            flow_unweighted_loss = zero
            flow_left_loss = zero
            flow_right_loss = zero
            flow_active_loss = zero
            flow_left_motion_share = zero
            flow_right_motion_share = zero
        if atomic_targets is None:
            atomic_losses = AtomicLosses(zero, zero, zero, zero, zero, zero)
            atomic_loss = zero
        else:
            atomic_losses = self._arm_atomic_losses(
                arm_directions,
                atomic_targets,
                update_codes=update_codes,
                global_step=global_step,
            )
            atomic_loss = (
                atomic_losses.ranking
                + self.config.atomic_ratio_loss_weight * atomic_losses.ratio_kl
                + self.config.codebook_loss_weight * atomic_losses.codebook
            )
        projection_loss = (
            self._frozen_composition_projection_loss(arm_directions, atomic_composition_targets)
            if atomic_composition_targets is not None
            else zero
        )
        if self.config.coefficient_loss_weight > 0.0:
            if coefficient_tcp_twist_delta is None:
                raise ValueError(
                    "coefficient target is required when coefficient_loss_weight is non-zero"
                )
            coefficient_losses = self._coefficient_loss(
                coefficient_rng,
                arm_directions,
                coefficient_tcp_twist_delta,
                global_step,
                coefficient_arm_mask,
            )
        else:
            coefficient_losses = CoefficientLosses(
                velocity=zero,
                wall=zero,
                wall_weight=zero,
                total=zero,
                right=zero,
                left=zero,
            )
        coefficient_loss = coefficient_losses.total
        fast_action_ce_loss = (
            self._fast_action_ce_loss(fast_action_tokens)
            if fast_action_tokens is not None
            else jnp.zeros((), z_text.dtype)
        )
        total = (
            self.config.text_flow_loss_weight * flow_loss
            + self.config.coefficient_loss_weight * coefficient_loss
            + self.config.text_atomic_loss_weight * atomic_loss
            + self.config.atomic_composition_loss_weight * projection_loss
            + self.config.fast_action_ce_loss_weight * fast_action_ce_loss
        )
        if not return_output:
            return total
        return AtomicTextStageOutput(
            flow_loss=flow_loss,
            flow_unweighted_loss=flow_unweighted_loss,
            flow_left_loss=flow_left_loss,
            flow_right_loss=flow_right_loss,
            flow_active_loss=flow_active_loss,
            flow_left_motion_share=flow_left_motion_share,
            flow_right_motion_share=flow_right_motion_share,
            coefficient_loss=coefficient_loss,
            coefficient_losses=coefficient_losses,
            fast_action_ce_loss=fast_action_ce_loss,
            atomic_total_loss=atomic_loss,
            projection_loss=projection_loss,
            total_loss=total,
            atomic_losses=atomic_losses,
            direction=arm_directions,
            z_text=z_text_arms,
        )

    def compute_visual_atomic_stage_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        atomic_prompt_tokens: jax.Array,
        atomic_prompt_mask: jax.Array,
        atomic_targets: AtomicTargets | None,
        coefficient_tcp_twist_delta: jax.Array,
        global_step: jax.Array | None = None,
        train: bool = False,
        return_output: bool = False,
    ) -> jax.Array | AtomicTextStageOutput:
        """Stage A with images: frozen ViT -> VLM/Q1 -> atomic code and DCT.

        Unlike the later z_M action-flow phase, this intentionally has no Q2/
        Q3 conditioning path and does not run the Gemma Action Expert. Q1 is
        read after image, atomic-subprompt and state tokens, so its codebook
        direction and DCT trajectory code are grounded in the current scene.
        The released FAST CE branch is deliberately disabled for this run: it
        remains available as an auxiliary objective in later training, but it
        must not consume an action suffix or provide a competing shortcut here.
        """

        preprocess_rng, coefficient_rng = jax.random.split(rng)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        # On clean atomic chunks the loader supplies the local atomic prompt;
        # elsewhere it supplies the global task prompt.  Q1 therefore always
        # receives the same instruction that owns its DCT target.
        observation = self._with_prompt(observation, atomic_prompt_tokens, atomic_prompt_mask)
        query_hidden, _, _, _ = self._prefix_forward(observation)
        active_state = self._controlled_state(observation.state)
        z_q1 = self.queries.text_latent(query_hidden, active_state)
        direction = l2_normalize(self.queries.direction(z_q1))
        if atomic_targets is None:
            zero = jnp.zeros((), z_q1.dtype)
            atomic_losses = AtomicLosses(zero, zero, zero, zero, zero, zero)
            atomic_loss = zero
        else:
            atomic_losses = self._atomic_losses(
                direction, atomic_targets, update_codes=True, global_step=global_step
            )
            atomic_loss = (
                atomic_losses.ranking
                + self.config.atomic_ratio_loss_weight * atomic_losses.ratio_kl
                + self.config.codebook_loss_weight * atomic_losses.codebook
            )
        coefficient_losses = self._coefficient_loss(
            coefficient_rng, direction, coefficient_tcp_twist_delta, global_step
        )
        coefficient_loss = coefficient_losses.total
        fast_action_ce_loss = jnp.zeros((), z_q1.dtype)
        total = (
            self.config.coefficient_loss_weight * coefficient_loss
            + self.config.text_atomic_loss_weight * atomic_loss
        )
        if not return_output:
            return total
        return AtomicTextStageOutput(
            flow_loss=jnp.zeros((), z_q1.dtype),
            flow_unweighted_loss=jnp.zeros((), z_q1.dtype),
            flow_left_loss=jnp.zeros((), z_q1.dtype),
            flow_right_loss=jnp.zeros((), z_q1.dtype),
            flow_active_loss=jnp.zeros((), z_q1.dtype),
            flow_left_motion_share=jnp.zeros((), z_q1.dtype),
            flow_right_motion_share=jnp.zeros((), z_q1.dtype),
            coefficient_loss=coefficient_loss,
            coefficient_losses=coefficient_losses,
            fast_action_ce_loss=fast_action_ce_loss,
            atomic_total_loss=atomic_loss,
            projection_loss=jnp.zeros((), z_q1.dtype),
            total_loss=total,
            atomic_losses=atomic_losses,
            direction=direction,
            z_text=z_q1,
        )

    def _auxiliary_loss(
        self,
        full_direction: jax.Array,
        targets: AtomicTargets | None,
        quantity_value: jax.Array,
        quantity_target: jax.Array | None,
        quantity_valid: jax.Array | None,
        text_direction: jax.Array | None,
        global_step: jax.Array | None,
    ) -> tuple[jax.Array, AtomicLosses, jax.Array, AtomicLosses, jax.Array]:
        zero = jnp.zeros((), full_direction.dtype)
        if targets is None:
            full_losses = AtomicLosses(zero, zero, zero, zero, zero, zero)
            text_losses = AtomicLosses(zero, zero, zero, zero, zero, zero)
            text_loss = zero
        else:
            # Full z_M is only pulled to stop-gradient code directions. The
            # language-only Q1/z_T pass is the only branch allowed to update
            # the codebook on clean single-atom rows.
            full_losses = self._atomic_losses(
                full_direction, targets, update_codes=False, global_step=global_step
            )
            if text_direction is None:
                raise ValueError("atomic targets require the text-only Q1 direction")
            text_losses = self._atomic_losses(
                text_direction, targets, update_codes=True, global_step=global_step
            )
            text_loss = (
                text_losses.ranking
                + self.config.atomic_ratio_loss_weight * text_losses.ratio_kl
                + self.config.codebook_loss_weight * text_losses.codebook
            )
        if quantity_target is None or quantity_valid is None:
            quantity_loss = zero
        else:
            valid = quantity_valid.astype(quantity_value.dtype)
            smooth_l1 = jnp.where(
                jnp.abs(quantity_value - quantity_target) < 1,
                0.5 * jnp.square(quantity_value - quantity_target),
                jnp.abs(quantity_value - quantity_target) - 0.5,
            )
            quantity_loss = jnp.sum(smooth_l1 * valid) / jnp.maximum(jnp.sum(valid), 1)
        full_loss = (
            full_losses.ranking + self.config.atomic_ratio_loss_weight * full_losses.ratio_kl
        )
        return full_loss, full_losses, text_loss, text_losses, quantity_loss

    def compute_full_stage_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        atomic_targets: AtomicTargets | None = None,
        atomic_composition_targets: AtomicCompositionTargets | None = None,
        text_teacher_directions: jax.Array | None = None,
        quantity_target: jax.Array | None = None,
        quantity_valid: jax.Array | None = None,
        subprompt_tokens: jax.Array | None = None,
        subprompt_mask: jax.Array | None = None,
        full_use_subprompt: jax.Array | None = None,
        atomic_prompt_tokens: jax.Array | None = None,
        atomic_prompt_mask: jax.Array | None = None,
        full_use_atomic_prompt: jax.Array | None = None,
        subtask_target_tokens: jax.Array | None = None,
        subtask_target_mask: jax.Array | None = None,
        subtask_ce_indices: jax.Array | None = None,
        global_step: jax.Array | None = None,
        visual_rotation_phase_step: jax.Array | None = None,
        train: bool = False,
        return_output: bool = False,
    ) -> jax.Array | AtomicFullStageOutput:
        """Train the full-observation z_M route on a mixed-batch shard.

        Strict single/dual/stay arms align visual Q1/Q3 only to a stop-gradient
        atomic-prompt zT target computed by the caller for the same normalized
        state. Dropped arms, which have no strict zT-Q1 target, use direct
        Drop-Top5 code-direction regression. Stage-A Two-Way and KL objectives
        do not execute in zM.
        """

        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        if (subprompt_tokens is None) != (subprompt_mask is None):
            raise ValueError("subprompt_tokens and subprompt_mask must be provided together")
        if (atomic_prompt_tokens is None) != (atomic_prompt_mask is None):
            raise ValueError("atomic prompt tokens and mask must be provided together")
        if atomic_prompt_tokens is not None:
            if subprompt_tokens is None or subprompt_mask is None:
                raise ValueError("atomic/subtask prompt mixing requires both prompt types")
            if full_use_atomic_prompt is None:
                raise ValueError("atomic/subtask prompt mixing requires a selection mask")
            if full_use_atomic_prompt.shape != observation.state.shape[:1]:
                raise ValueError("full_use_atomic_prompt must have shape [batch]")
            observation = self._with_prompt(
                observation,
                jnp.where(full_use_atomic_prompt[:, None], atomic_prompt_tokens, subprompt_tokens),
                jnp.where(full_use_atomic_prompt[:, None], atomic_prompt_mask, subprompt_mask),
            )
        elif subprompt_tokens is not None:
            if full_use_subprompt is None:
                full_use_subprompt = jnp.zeros(observation.state.shape[0], dtype=jnp.bool_)
            if full_use_subprompt.shape != observation.state.shape[:1]:
                raise ValueError("full_use_subprompt must have shape [batch]")
            if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
                raise ValueError("AtomicPi05 requires a global tokenized prompt")
            observation = self._with_prompt(
                observation,
                jnp.where(
                    full_use_subprompt[:, None],
                    subprompt_tokens,
                    observation.tokenized_prompt,
                ),
                jnp.where(
                    full_use_subprompt[:, None],
                    subprompt_mask,
                    observation.tokenized_prompt_mask,
                ),
            )
        elif full_use_subprompt is not None:
            raise ValueError("full_use_subprompt requires tokenized subprompts")
        if atomic_prompt_tokens is None and full_use_atomic_prompt is not None:
            raise ValueError("full_use_atomic_prompt requires tokenized atomic prompts")

        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        actions = self._mask_action_condition(actions)
        noise = self._mask_action_condition(jax.random.normal(noise_rng, actions.shape))
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
        noisy = time[:, None, None] * noise + (1.0 - time[:, None, None]) * actions
        target_velocity = noise - actions
        active_state = self._controlled_state(observation.state)
        use_joint_layerwise = (
            self.config.enable_layerwise_atomic_flow and self.config.subtask_ce_loss_weight == 0.0
        )
        if use_joint_layerwise:
            # The production target-60K recipe has CE disabled. Prefix/Q and
            # Action Expert can therefore advance together, and each Action
            # block receives z_M composed from Q1--Q4 at the same depth.
            query_hidden, velocity, _ = self._joint_layerwise_velocity(
                observation,
                noisy,
                time,
            )
            subtask_ce_loss = jnp.zeros((), velocity.dtype)
            prefix_mask = None
            kv_cache = None
        elif self.config.subtask_ce_loss_weight == 0.0:
            # CE is disabled for this phase. Do not append or decode the
            # teacher-forced text suffix; a zero loss weight alone would
            # otherwise still pay the full vocabulary-decoding cost.
            query_hidden, prefix_mask, kv_cache, subtask_ce_loss = self._prefix_forward(observation)
        elif subtask_ce_indices is None:
            # Backward-compatible path: teacher-force every row in the main
            # prefix call. Prefer the explicit CE sub-batch path below for
            # mixed subtask/global training.
            query_hidden, prefix_mask, kv_cache, subtask_ce_loss = self._prefix_forward(
                observation,
                subtask_tokens=subtask_target_tokens,
                subtask_mask=subtask_target_mask,
            )
        else:
            if subtask_ce_indices.ndim != 1:
                raise ValueError("subtask_ce_indices must have shape [compact_batch]")
            compact_batch_size = subtask_ce_indices.shape[0]
            if compact_batch_size <= 0 or compact_batch_size > actions.shape[0]:
                raise ValueError("compact CE batch must be in [1, full batch]")
            if subtask_target_tokens is None or subtask_target_mask is None:
                raise ValueError("subtask CE sub-batch requires target tokens and mask")
            # Flow/atomic training uses the complete zM batch without carrying
            # a 64-token teacher-forced suffix on rows whose CE is masked out.
            query_hidden, prefix_mask, kv_cache, _ = self._prefix_forward(observation)
            # Gather only the label-dependent CE rows into a static compact
            # sub-batch instead of making all 256 rows carry the text suffix.
            ce_observation = jax.tree.map(
                lambda value: take_batch_rows(value, subtask_ce_indices), observation
            )
            _, _, _, subtask_ce_loss = self._prefix_forward(
                ce_observation,
                subtask_tokens=take_batch_rows(subtask_target_tokens, subtask_ce_indices),
                subtask_mask=take_batch_rows(subtask_target_mask, subtask_ce_indices),
            )
            # _subtask_ce_loss normalizes by the number of valid target tokens,
            # not by batch size. Gathering therefore preserves loss/gradient
            # scale exactly; invalid padding rows add neither numerator nor
            # denominator.
        _, direction, z_model, left_direction, quantity_value = self._latent(
            query_hidden, active_state
        )
        if not use_joint_layerwise:
            velocity = self._suffix_velocity(prefix_mask, kv_cache, noisy, time, z_model)
        (
            flow_loss,
            flow_unweighted_loss,
            flow_left_loss,
            flow_right_loss,
            flow_active_loss,
            flow_left_motion_share,
            flow_right_motion_share,
        ) = bimanual_flow_losses(
            velocity,
            target_velocity,
            actions,
        )

        zero = jnp.zeros((), flow_loss.dtype)
        full_directions = jnp.stack([direction, left_direction], axis=1)
        if atomic_targets is None:
            atomic_losses = AtomicLosses(zero, zero, zero, zero, zero, zero)
            text_teacher_cosine_loss = zero
            text_teacher_cosine_similarity = zero
            strict_atomic_huber_loss = zero
        else:
            if text_teacher_directions is None:
                raise ValueError("strict ZM atomic targets require zT teacher directions")
            (
                text_teacher_cosine_loss,
                text_teacher_cosine_similarity,
            ) = self._text_teacher_cosine_alignment(
                full_directions,
                text_teacher_directions,
                atomic_targets.supervision_mask,
            )
            # zT already learned code identity/ranking. zM only distills its
            # frozen Q1/Q3 directions for strict Single/Dual/Stay arms; it must
            # not relearn Two-Way, ratio KL, or codebook updates.
            atomic_losses = AtomicLosses(zero, zero, zero, zero, zero, zero)
            # Keep the old metric present as an exact zero for dashboard
            # compatibility; no angular/Huber target executes in ZM.
            strict_atomic_huber_loss = zero
        atomic_loss = text_teacher_cosine_loss
        projection_loss = (
            self._frozen_composition_projection_loss(full_directions, atomic_composition_targets)
            if atomic_composition_targets is not None
            else zero
        )
        composition_kl_loss = zero
        if self.config.visual_rotation_loss_weight > 0.0:
            if not self.config.spherical_visual_latent:
                raise ValueError("visual rotation loss requires spherical_visual_latent")
            visual_rotation_loss, visual_rotation_mean_angle_rad = (
                visual_rotation_hinge_loss(
                    full_directions,
                    z_model,
                    free_angle_rad=jnp.deg2rad(
                        self.config.visual_rotation_free_angle_deg
                    ),
                    max_angle_rad=jnp.deg2rad(
                        self.config.visual_max_update_angle_deg
                    ),
                )
            )
            if self.config.visual_rotation_loss_warmup_steps > 0:
                if visual_rotation_phase_step is None:
                    raise ValueError(
                        "visual rotation warmup requires visual_rotation_phase_step"
                    )
                visual_rotation_loss_scale = jnp.clip(
                    (visual_rotation_phase_step.astype(jnp.float32) + 1.0)
                    / self.config.visual_rotation_loss_warmup_steps,
                    0.0,
                    1.0,
                )
            else:
                visual_rotation_loss_scale = jnp.ones((), dtype=flow_loss.dtype)
        else:
            visual_rotation_loss = zero
            visual_rotation_mean_angle_rad = zero
            visual_rotation_loss_scale = zero
        if quantity_target is None or quantity_valid is None:
            quantity_loss = zero
        else:
            valid = quantity_valid.astype(quantity_value.dtype)
            error = jnp.abs(quantity_value - quantity_target)
            smooth_l1 = jnp.where(error < 1, 0.5 * jnp.square(error), error - 0.5)
            quantity_loss = jnp.sum(smooth_l1 * valid) / jnp.maximum(jnp.sum(valid), 1)
        total = (
            flow_loss
            + self.config.atomic_loss_weight * atomic_loss
            + self.config.atomic_composition_loss_weight * projection_loss
            + self.config.visual_rotation_loss_weight
            * visual_rotation_loss_scale
            * visual_rotation_loss
            + self.config.quantity_loss_weight * quantity_loss
            + self.config.subtask_ce_loss_weight * subtask_ce_loss
        )
        if not return_output:
            return total
        return AtomicFullStageOutput(
            flow_loss=flow_loss,
            flow_unweighted_loss=flow_unweighted_loss,
            flow_left_loss=flow_left_loss,
            flow_right_loss=flow_right_loss,
            flow_active_loss=flow_active_loss,
            flow_left_motion_share=flow_left_motion_share,
            flow_right_motion_share=flow_right_motion_share,
            atomic_total_loss=atomic_loss,
            text_teacher_cosine_loss=text_teacher_cosine_loss,
            text_teacher_cosine_similarity=text_teacher_cosine_similarity,
            strict_atomic_huber_loss=strict_atomic_huber_loss,
            visual_rotation_loss=visual_rotation_loss,
            visual_rotation_mean_angle_rad=visual_rotation_mean_angle_rad,
            visual_rotation_loss_scale=visual_rotation_loss_scale,
            total_loss=total,
            atomic_losses=atomic_losses,
            projection_loss=projection_loss,
            composition_kl_loss=composition_kl_loss,
            quantity_loss=quantity_loss,
            subtask_ce_loss=subtask_ce_loss,
            direction=direction,
            z_model=z_model,
        )

    def compute_subtask_ce_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        prompt_tokens: jax.Array,
        prompt_mask: jax.Array,
        target_tokens: jax.Array,
        target_mask: jax.Array,
        train: bool = False,
    ) -> jax.Array:
        """Compute only the teacher-forced atomic-text CE branch.

        This excludes z_M, Flow Matching, and the Action Expert. The trainer
        can therefore compile the full batch and compact CE batch as separate
        FSDP programs, accumulate both gradients, and apply one update.
        """

        if prompt_tokens.shape != prompt_mask.shape:
            raise ValueError("CE prompt tokens/mask must share [batch, length]")
        if target_tokens.shape != target_mask.shape:
            raise ValueError("CE target tokens/mask must share [batch, length]")
        if prompt_tokens.shape[0] != observation.state.shape[0]:
            raise ValueError("CE prompt batch must match observation batch")
        if target_tokens.shape[0] != observation.state.shape[0]:
            raise ValueError("CE target batch must match observation batch")
        observation = self._with_prompt(observation, prompt_tokens, prompt_mask)
        observation = _model.preprocess_observation(rng, observation, train=train)
        _, _, _, subtask_ce_loss = self._prefix_forward(
            observation,
            subtask_tokens=target_tokens,
            subtask_mask=target_mask,
        )
        return subtask_ce_loss

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        atomic_targets: AtomicTargets | None = None,
        quantity_target: jax.Array | None = None,
        quantity_valid: jax.Array | None = None,
        coefficient_tcp_twist_delta: jax.Array | None = None,
        global_step: jax.Array | None = None,
        subprompt_tokens: jax.Array | None = None,
        subprompt_mask: jax.Array | None = None,
        full_use_subprompt: jax.Array | None = None,
        subtask_target_tokens: jax.Array | None = None,
        subtask_target_mask: jax.Array | None = None,
        return_output: bool = False,
    ) -> jax.Array | AtomicPi05Output:
        preprocess_rng, noise_rng, time_rng, coefficient_rng = jax.random.split(rng, 4)
        if (subprompt_tokens is None) != (subprompt_mask is None):
            raise ValueError("subprompt_tokens and subprompt_mask must be provided together")
        # Keep the original episode-level prompt for z_T fallback. The full
        # z_M pass below may replace individual rows with a subtask prompt.
        global_prompt_tokens = observation.tokenized_prompt
        global_prompt_mask = observation.tokenized_prompt_mask
        if global_prompt_tokens is None or global_prompt_mask is None:
            raise ValueError("AtomicPi05 requires a global tokenized prompt")
        if subprompt_tokens is not None:
            if full_use_subprompt is None:
                full_use_subprompt = jnp.zeros(observation.state.shape[0], dtype=jnp.bool_)
            if full_use_subprompt.shape != observation.state.shape[:1]:
                raise ValueError("full_use_subprompt must have shape [batch]")
            observation = self._with_prompt(
                observation,
                jnp.where(full_use_subprompt[:, None], subprompt_tokens, global_prompt_tokens),
                jnp.where(full_use_subprompt[:, None], subprompt_mask, global_prompt_mask),
            )
        elif full_use_subprompt is not None:
            raise ValueError("full_use_subprompt requires tokenized subprompts")
        if atomic_targets is not None and (subprompt_tokens is None or subprompt_mask is None):
            raise ValueError("atomic training requires segment subprompt_tokens and subprompt_mask")
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        actions = self._mask_action_condition(actions)
        noise = self._mask_action_condition(jax.random.normal(noise_rng, actions.shape))
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
        noisy = time[:, None, None] * noise + (1.0 - time[:, None, None]) * actions
        target_velocity = noise - actions
        active_state = self._controlled_state(observation.state)
        # Every training row has a z_T branch. For labelled atomic rows the
        # loader provides the local subtask prompt. For an unlabelled complex
        # row it may be absent, in which case z_T receives the global prompt
        # and is supervised only by the coefficient code, never the codebook.
        # Q2/Q3/full observation never enter this path.
        if subprompt_tokens is None:
            text_prompt_tokens, text_prompt_mask = global_prompt_tokens, global_prompt_mask
        else:
            text_prompt_tokens, text_prompt_mask = subprompt_tokens, subprompt_mask
        text_query_hidden, _, _, _ = self._prefix_forward(
            self._text_only(observation, text_prompt_tokens, text_prompt_mask)
        )
        z_text = self.queries.text_latent(text_query_hidden, active_state)
        text_direction = l2_normalize(self.queries.direction(z_text))
        query_hidden, prefix_mask, kv_cache, subtask_ce_loss = self._prefix_forward(
            observation,
            subtask_tokens=subtask_target_tokens,
            subtask_mask=subtask_target_mask,
        )
        z_q1_full, direction, z_model, _, quantity_value = self._latent(query_hidden, active_state)
        velocity = self._suffix_velocity(prefix_mask, kv_cache, noisy, time, z_model)
        flow_loss, _, _, _, _, _, _ = bimanual_flow_losses(
            velocity,
            target_velocity,
            actions,
        )
        atomic_loss, atomic_losses, text_atomic_loss, text_atomic_losses, quantity_loss = (
            self._auxiliary_loss(
                direction,
                atomic_targets,
                quantity_value,
                quantity_target,
                quantity_valid,
                text_direction,
                global_step,
            )
        )
        # z_T is always supervised by a compact action code. Atomic labels
        # merely add its semantic/codebook losses; they never turn DCT off.
        if coefficient_tcp_twist_delta is None:
            raise ValueError(
                "every training row requires coefficient_tcp_twist_delta: FK the current and "
                "50 future right-arm states, then express their TCP deltas in base axes"
            )
        coefficient_losses = self._coefficient_loss(
            coefficient_rng, text_direction, coefficient_tcp_twist_delta, global_step
        )
        coefficient_loss = coefficient_losses.total
        total = (
            flow_loss
            + self.config.coefficient_loss_weight * coefficient_loss
            + self.config.atomic_loss_weight * atomic_loss
            + self.config.text_atomic_loss_weight * text_atomic_loss
            + self.config.quantity_loss_weight * quantity_loss
            + self.config.subtask_ce_loss_weight * subtask_ce_loss
        )
        if not return_output:
            return total
        return AtomicPi05Output(
            flow_loss=flow_loss,
            coefficient_loss=coefficient_loss,
            coefficient_losses=coefficient_losses,
            total_loss=total,
            text_atomic_loss=text_atomic_loss,
            atomic_losses=atomic_losses,
            text_atomic_losses=text_atomic_losses,
            quantity_loss=quantity_loss,
            subtask_ce_loss=subtask_ce_loss,
            direction=direction,
            z_text=z_text,
            z_model=z_model,
        )

    def _require_force_conditioner(self) -> ForceConditioner:
        if not self.config.enable_force_stage or not hasattr(self, "force_conditioner"):
            raise RuntimeError(
                "force stage is disabled; construct AtomicPi05Config(enable_force_stage=True)"
            )
        return self.force_conditioner

    def _force_layerwise_latents(
        self,
        layerwise_arm_latents: jax.Array,
        delta_z: jax.Array,
    ) -> jax.Array:
        """Inject one bimanual force correction into every atomic depth.

        Cross-attention has already produced ``delta_z`` once. This helper is
        intentionally only a broadcast, scalar gate and the checkpointed
        right/left fusion; it never reruns the temporal encoder or attention.
        """

        conditioner = self._require_force_conditioner()
        expected = (
            conditioner.num_layers,
            delta_z.shape[0],
            self.config.arm_count,
            self.config.latent_dim,
        )
        if layerwise_arm_latents.shape != expected:
            raise ValueError(
                f"layerwise arm latents must have shape {expected}, got "
                f"{layerwise_arm_latents.shape}"
            )
        if delta_z.shape[1:] != (
            self.config.arm_count,
            self.config.latent_dim,
        ):
            raise ValueError("delta_z must have shape [B,2,latent_dim]")
        gates = conditioner.layer_gates().astype(layerwise_arm_latents.dtype)
        scaled_delta = gates[:, None, None, None] * delta_z[None].astype(
            layerwise_arm_latents.dtype
        )
        corrected_arms = layerwise_arm_latents + scaled_delta
        if self.config.spherical_force_update:
            corrected_arms = spherical_tangent_update(
                layerwise_arm_latents,
                scaled_delta,
                jnp.deg2rad(self.config.force_max_update_angle_deg),
            )
        _, fusion_params, _ = self._layerwise_atomic_inputs()
        return jax.vmap(
            lambda arm_latents: fuse_intermediate_arm_latents(arm_latents, fusion_params)
        )(corrected_arms)

    def compute_force_stage_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        slow_force_history: jax.Array,
        slow_state_history: jax.Array,
        slow_history_mask: jax.Array,
        current_force_history: jax.Array,
        current_state_history: jax.Array,
        current_history_mask: jax.Array,
        future_force: jax.Array,
        future_force_mask: jax.Array,
        update_offset: jax.Array,
        base_actions: jax.Array | None = None,
        train_fast: bool = True,
        train: bool = False,
        return_output: bool = False,
    ) -> jax.Array | ForceStageOutput:
        """Train future-force representation and ten-step latent correction.

        ``observation`` is the slow VLA observation at the beginning of the
        cached 50-step horizon. ``actions`` is always that anchor's fixed
        ``0:50`` target. ``update_offset`` selects a clean teacher-forced
        prefix and a suffix-only RTC loss; it never shifts the target to
        ``10:60`` or beyond. The caller may sample one offset per row without
        running all five fast updates.

        Force and state histories, as well as the future force target, must be
        normalized before entering this method. Future force remains at 120 Hz
        for the main 200-point delta objective; an action-rate pooled objective
        is added only as a light long-horizon auxiliary term.
        """

        conditioner = self._require_force_conditioner()
        preprocess_rng, noise_rng, time_rng, history_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        actions = self._mask_action_condition(actions)
        # Kept in the public signature for old callers, but deliberately not
        # consumed: future-force prediction must pass through z_F_slow.
        del base_actions
        batch_size = actions.shape[0]
        if update_offset.shape != (batch_size,):
            raise ValueError("update_offset must have shape [batch]")
        valid_offset = (
            (update_offset >= 0)
            & (update_offset < self.config.action_horizon)
            & (update_offset % self.config.force_update_action_steps == 0)
        )
        if not isinstance(update_offset, jax.core.Tracer) and not bool(jnp.all(valid_offset)):
            raise ValueError("update_offset must be one of the fixed-horizon fast update offsets")

        # B1 normally uses the complete one-second slow window. The configurable
        # suffix mechanism is retained for controlled history-length ablations;
        # every variant keeps the same newest timestamp and positional meaning.
        if train and not train_fast:
            train_lengths = jnp.asarray(
                self.config.force_history_train_lengths, dtype=jnp.int32
            )
            length_index = jax.random.randint(
                history_rng, (batch_size,), 0, train_lengths.shape[0]
            )
            retained = train_lengths[length_index]
            sample_index = jnp.arange(slow_force_history.shape[-2])[None]
            suffix_mask = sample_index >= (
                slow_force_history.shape[-2] - retained[:, None]
            )
            slow_history_mask = slow_history_mask & suffix_mask[:, None, :]

        if train_fast and self.config.enable_layerwise_atomic_flow:
            prefix_result = self._prefix_forward(
                observation,
                return_prefix_hidden=True,
                return_layerwise_arm_latents=True,
            )
            (
                query_hidden,
                prefix_mask,
                kv_cache,
                _,
                prefix_hidden,
                _,
                layerwise_arm_latents,
            ) = prefix_result
        else:
            query_hidden, prefix_mask, kv_cache, _, prefix_hidden = self._prefix_forward(
                observation,
                return_prefix_hidden=True,
            )
            layerwise_arm_latents = None
        active_state = self._controlled_state(observation.state)
        _, _, z_model, _, _ = self._latent(query_hidden, active_state)
        context_mask = prefix_mask[:, : prefix_hidden.shape[1]]
        if self.config.force_stop_gradient_backbone:
            prefix_hidden = jax.lax.stop_gradient(prefix_hidden)
            z_model = jax.lax.stop_gradient(z_model)
            if layerwise_arm_latents is not None:
                layerwise_arm_latents = jax.lax.stop_gradient(layerwise_arm_latents)
            kv_cache = jax.tree.map(jax.lax.stop_gradient, kv_cache)

        slow_context = conditioner.encode_context(
            prefix_hidden,
            context_mask,
            z_model,
            slow_force_history,
            slow_state_history,
            slow_history_mask,
        )
        if self.config.force_future_loss_weight > 0:
            predicted_future = conditioner.predict_future_force_delta(
                slow_context.latent,
                z_model,
            )
            future_target, raw_future_mask = future_force_delta_target(
                slow_force_history,
                future_force,
                future_force_mask,
            )
            if predicted_future.shape != future_target.shape:
                raise ValueError(
                    "future-force decoder/target mismatch: "
                    f"{predicted_future.shape} vs {future_target.shape}"
                )

            def masked_smooth_l1(
                prediction: jax.Array, target: jax.Array, mask: jax.Array
            ) -> jax.Array:
                error = jnp.abs(prediction - target)
                smooth_l1 = jnp.where(error < 1, 0.5 * jnp.square(error), error - 0.5)
                weights = mask[..., None].astype(smooth_l1.dtype)
                return jnp.sum(smooth_l1 * weights) / jnp.maximum(
                    jnp.sum(weights) * self.config.force_dim, 1
                )

            future_force_raw_loss = masked_smooth_l1(
                predicted_future, future_target, raw_future_mask
            )
            coarse_prediction, coarse_future_mask = temporal_masked_mean(
                predicted_future,
                raw_future_mask,
                stride=self.config.force_temporal_stride,
            )
            coarse_target, _ = temporal_masked_mean(
                future_target,
                raw_future_mask,
                stride=self.config.force_temporal_stride,
            )
            future_force_coarse_loss = masked_smooth_l1(
                coarse_prediction, coarse_target, coarse_future_mask
            )
            future_force_loss = (
                future_force_raw_loss
                + self.config.force_future_coarse_loss_weight * future_force_coarse_loss
            )
        else:
            # A zero objective is an actual decoder-off contract, not merely a
            # zero multiplier after paying for the 200-step prediction graph.
            predicted_future = jnp.zeros_like(future_force)
            future_force_raw_loss = jnp.zeros((), dtype=future_force.dtype)
            future_force_coarse_loss = jnp.zeros((), dtype=future_force.dtype)
            future_force_loss = jnp.zeros((), dtype=future_force.dtype)

        if train_fast:
            modulation = conditioner.modulate(
                z_model,
                slow_context.latent,
                current_force_history,
                current_state_history,
                current_history_mask,
                update_offset,
                slow_history_tokens=slow_context.history_tokens,
                slow_history_token_mask=slow_context.history_token_mask,
            )
            if (
                self.config.spherical_force_update
                and not self.config.enable_layerwise_atomic_flow
            ):
                modulation = modulation.replace(
                    z_exec=spherical_tangent_update(
                        z_model,
                        modulation.delta_z,
                        jnp.deg2rad(self.config.force_max_update_angle_deg),
                    )
                )
            noise = self._mask_action_condition(jax.random.normal(noise_rng, actions.shape))
            time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
            noisy, token_time, committed_mask, target_velocity = rtc_flow_batch(
                actions, noise, time, update_offset
            )
            force_tokens = None
            if self.config.enable_force_hidden_cross_attention:
                force_tokens = conditioner.compose_force_tokens(
                    slow_context.latent,
                    modulation.recent_tokens,
                    current_history_mask,
                    update_offset,
                )
                _, fusion_params, _ = self._layerwise_atomic_inputs()
                # zM remains exactly on its existing FiLM residual path.  The
                # force token gets an independent hidden-state cross-attention
                # residual and never modifies zM.
                force_layerwise_latents = jax.vmap(
                    lambda arm_latents: fuse_intermediate_arm_latents(
                        arm_latents, fusion_params
                    )
                )(layerwise_arm_latents)
            else:
                force_layerwise_latents = (
                    self._force_layerwise_latents(
                        layerwise_arm_latents, modulation.delta_z
                    )
                    if self.config.enable_layerwise_atomic_flow
                    else None
                )
            velocity = self._suffix_velocity(
                prefix_mask,
                kv_cache,
                noisy,
                token_time,
                modulation.z_exec if not self.config.enable_layerwise_atomic_flow else z_model,
                committed_mask,
                layerwise_latents=(
                    force_layerwise_latents
                    if self.config.enable_layerwise_atomic_flow
                    else None
                ),
                force_tokens=force_tokens,
            )
            flow_loss = rtc_flow_loss(
                velocity - target_velocity,
                committed_mask,
            )
            delta_z_regularization = jnp.mean(jnp.square(modulation.delta_z))
            if self.config.force_improvement_loss_weight > 0:
                # Exact zM-only control: reuse the same observation prefix,
                # teacher-forced committed prefix, noise and flow time, and
                # remove only the zF+fast correction.  Stop-gradient prevents
                # the optimizer from satisfying the margin by degrading zM.
                if self.config.enable_layerwise_atomic_flow:
                    _, fusion_params, _ = self._layerwise_atomic_inputs()
                    base_layerwise_latents = jax.vmap(
                        lambda arm_latents: fuse_intermediate_arm_latents(
                            arm_latents, fusion_params
                        )
                    )(layerwise_arm_latents)
                else:
                    base_layerwise_latents = None
                base_velocity = self._suffix_velocity(
                    prefix_mask,
                    kv_cache,
                    noisy,
                    token_time,
                    z_model,
                    committed_mask,
                    layerwise_latents=base_layerwise_latents,
                )
                base_flow_loss = rtc_flow_loss(
                    base_velocity - target_velocity,
                    committed_mask,
                )
                improvement_loss = jax.nn.relu(
                    flow_loss
                    - jax.lax.stop_gradient(base_flow_loss)
                    + self.config.force_improvement_margin
                )
            else:
                base_flow_loss = jnp.zeros((), flow_loss.dtype)
                improvement_loss = jnp.zeros((), flow_loss.dtype)
        else:
            # Stage B1 does not run the 300M Action Expert. Its sole purpose is
            # to give z_F a future-force meaning before delta_z is introduced.
            zero_delta = jnp.zeros_like(z_model)
            modulation = ForceModulationOutput(
                delta_z=zero_delta,
                z_exec=z_model,
                recent_tokens=jnp.zeros(
                    (
                        batch_size,
                        self.config.arm_count,
                        self.config.force_update_action_steps,
                        self.config.force_encoder_width,
                    ),
                    dtype=z_model.dtype,
                ),
            )
            flow_loss = jnp.zeros((), future_force_loss.dtype)
            delta_z_regularization = jnp.zeros((), future_force_loss.dtype)
            base_flow_loss = jnp.zeros((), future_force_loss.dtype)
            improvement_loss = jnp.zeros((), future_force_loss.dtype)
        if train_fast and self.config.force_rotation_loss_weight > 0.0:
            force_rotation_loss, force_rotation_mean_angle_rad = (
                visual_rotation_hinge_loss(
                    z_model,
                    modulation.z_exec,
                    free_angle_rad=jnp.deg2rad(
                        self.config.force_rotation_free_angle_deg
                    ),
                    max_angle_rad=jnp.deg2rad(
                        self.config.force_max_update_angle_deg
                    ),
                )
            )
        else:
            force_rotation_loss = jnp.zeros((), future_force_loss.dtype)
            force_rotation_mean_angle_rad = jnp.zeros((), future_force_loss.dtype)
        total = (
            self.config.force_flow_loss_weight * flow_loss
            + self.config.force_future_loss_weight * future_force_loss
            + self.config.force_delta_regularization_weight * delta_z_regularization
            + self.config.force_improvement_loss_weight * improvement_loss
            + self.config.force_rotation_loss_weight * force_rotation_loss
        )
        if not return_output:
            return total
        return ForceStageOutput(
            flow_loss=flow_loss,
            future_force_loss=future_force_loss,
            future_force_raw_loss=future_force_raw_loss,
            future_force_coarse_loss=future_force_coarse_loss,
            delta_z_regularization=delta_z_regularization,
            base_flow_loss=base_flow_loss,
            improvement_loss=improvement_loss,
            force_rotation_loss=force_rotation_loss,
            force_rotation_mean_angle_rad=force_rotation_mean_angle_rad,
            total_loss=total,
            force_latent=slow_context.latent,
            delta_z=modulation.delta_z,
            z_exec=modulation.z_exec,
            predicted_future_force_delta=predicted_future,
        )

    def prepare_force_policy_context(
        self,
        observation: _model.Observation,
        *,
        slow_force_history: jax.Array,
        slow_state_history: jax.Array,
        slow_history_mask: jax.Array,
    ) -> ForcePolicyContext:
        """Run the slow VLA once and cache everything needed by fast updates."""

        conditioner = self._require_force_conditioner()
        observation = _model.preprocess_observation(None, observation, train=False)
        if self.config.enable_layerwise_atomic_flow:
            (
                query_hidden,
                prefix_mask,
                kv_cache,
                _,
                prefix_hidden,
                _,
                layerwise_arm_latents,
            ) = self._prefix_forward(
                observation,
                return_prefix_hidden=True,
                return_layerwise_arm_latents=True,
            )
        else:
            query_hidden, prefix_mask, kv_cache, _, prefix_hidden = self._prefix_forward(
                observation,
                return_prefix_hidden=True,
            )
            layerwise_arm_latents = None
        active_state = self._controlled_state(observation.state)
        _, _, z_model, _, _ = self._latent(query_hidden, active_state)
        context = conditioner.encode_context(
            prefix_hidden,
            prefix_mask[:, : prefix_hidden.shape[1]],
            z_model,
            slow_force_history,
            slow_state_history,
            slow_history_mask,
        )
        return ForcePolicyContext(
            prefix_mask=prefix_mask,
            kv_cache=kv_cache,
            z_model=z_model,
            force_latent=context.latent,
            slow_history_tokens=context.history_tokens,
            slow_history_token_mask=context.history_token_mask,
            layerwise_arm_latents=layerwise_arm_latents,
        )

    def sample_actions_force_update(
        self,
        rng: at.KeyArrayLike,
        context: ForcePolicyContext,
        *,
        current_force_history: jax.Array,
        current_state_history: jax.Array,
        current_history_mask: jax.Array,
        update_offset: jax.Array,
        num_steps: int = 10,
        noise: jax.Array | None = None,
        executed_actions: jax.Array | None = None,
        full_token_memory_scale: jax.Array | None = None,
        force_update_scale: jax.Array | float = 1.0,
    ) -> tuple[_model.Actions, ForceModulationOutput]:
        """RTC-inpaint the unexecuted suffix without rerunning the VLM.

        ``noise`` should be sampled once and reused for every update in one
        slow cycle. For non-zero offsets ``executed_actions`` must contain the
        normalized actions actually sent to the robot on the fixed 0:50 time
        axis; those positions are clamped after every Euler step.
        """

        conditioner = self._require_force_conditioner()
        modulation = conditioner.modulate(
            context.z_model,
            context.force_latent,
            current_force_history,
            current_state_history,
            current_history_mask,
            update_offset,
            slow_history_tokens=context.slow_history_tokens,
            slow_history_token_mask=context.slow_history_token_mask,
            full_token_memory_scale=full_token_memory_scale,
        )
        update_scale = jnp.asarray(force_update_scale, dtype=modulation.delta_z.dtype)
        if (
            self.config.spherical_force_update
            and not self.config.enable_layerwise_atomic_flow
        ):
            modulation = modulation.replace(
                z_exec=spherical_tangent_update(
                    context.z_model,
                    modulation.delta_z * update_scale,
                    jnp.deg2rad(self.config.force_max_update_angle_deg),
                )
            )
        elif not self.config.enable_layerwise_atomic_flow:
            modulation = modulation.replace(
                z_exec=context.z_model
                + update_scale * (modulation.z_exec - context.z_model)
            )
        force_tokens = None
        if self.config.enable_force_hidden_cross_attention:
            force_tokens = conditioner.compose_force_tokens(
                context.force_latent,
                modulation.recent_tokens,
                current_history_mask,
                update_offset,
            )
            _, fusion_params, _ = self._layerwise_atomic_inputs()
            force_layerwise_latents = jax.vmap(
                lambda arm_latents: fuse_intermediate_arm_latents(
                    arm_latents, fusion_params
                )
            )(context.layerwise_arm_latents)
        else:
            force_layerwise_latents = (
                self._force_layerwise_latents(
                    context.layerwise_arm_latents, modulation.delta_z
                )
                if self.config.enable_layerwise_atomic_flow
                else None
            )
        batch_size = context.z_model.shape[0]
        if update_offset.shape != (batch_size,):
            raise ValueError("update_offset must have shape [batch]")
        committed_mask = rtc_committed_mask(update_offset, self.action_horizon)
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        if noise.shape != (batch_size, self.action_horizon, self.action_dim):
            raise ValueError(
                f"noise must have shape {(batch_size, self.action_horizon, self.action_dim)}"
            )
        noise = self._mask_action_condition(noise)
        if executed_actions is None:
            if not isinstance(update_offset, jax.core.Tracer) and bool(jnp.any(update_offset > 0)):
                raise ValueError("non-zero RTC offset requires executed_actions")
            executed_actions = jnp.zeros_like(noise)
        elif executed_actions.shape == (
            batch_size,
            self.action_horizon,
            self.config.active_action_dim,
        ):
            executed_actions = jnp.pad(
                executed_actions,
                ((0, 0), (0, 0), (0, self.action_dim - self.config.active_action_dim)),
            )
        elif executed_actions.shape != noise.shape:
            raise ValueError(
                "executed_actions must have full or active action dimension; "
                f"got {executed_actions.shape}"
            )
        executed_actions = self._mask_action_condition(executed_actions)
        initial_actions = rtc_clamp_prefix(noise, executed_actions, committed_mask)
        dt = -1.0 / num_steps

        def step(carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            action_chunk, time = carry
            token_time = jnp.where(
                committed_mask,
                0.0,
                jnp.broadcast_to(time, committed_mask.shape),
            )
            velocity = self._suffix_velocity(
                context.prefix_mask,
                context.kv_cache,
                action_chunk,
                token_time,
                (
                    context.z_model
                    if self.config.enable_layerwise_atomic_flow
                    else modulation.z_exec
                ),
                committed_mask,
                layerwise_latents=(
                    force_layerwise_latents
                    if self.config.enable_layerwise_atomic_flow
                    else None
                ),
                force_tokens=force_tokens,
            )
            updated = self._mask_action_condition(action_chunk + dt * velocity)
            updated = rtc_clamp_prefix(updated, executed_actions, committed_mask)
            return updated, time + dt

        def condition(carry: tuple[jax.Array, jax.Array]) -> jax.Array:
            return carry[1] >= -dt / 2

        result, _ = jax.lax.while_loop(condition, step, (initial_actions, jnp.asarray(1.0)))
        result = rtc_clamp_prefix(result, executed_actions, committed_mask)
        return result[..., : self.config.active_action_dim], modulation

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: jax.Array | None = None,
        **kwargs,
    ) -> _model.Actions:
        # The shared OpenPI Policy always forwards its optional CFG fields.
        # Atomic checkpoints do not use CFG, but accepting disabled values
        # keeps the stock initial-sample/RTC state machine intact.
        if kwargs.get("cfg_observation") is not None:
            raise ValueError("Atomic PI0.5 RTC does not support CFG")
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        noise = self._mask_action_condition(noise)
        if self.config.enable_layerwise_atomic_flow:
            (
                query_hidden,
                prefix_mask,
                kv_cache,
                _,
                layerwise_latents,
            ) = self._prefix_forward(
                observation,
                return_layerwise_latents=True,
            )
        else:
            query_hidden, prefix_mask, kv_cache, _ = self._prefix_forward(observation)
            layerwise_latents = None
        active_state = self._controlled_state(observation.state)
        _, _, z_model, _, _ = self._latent(query_hidden, active_state)
        dt = -1.0 / num_steps

        def step(carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            actions, time = carry
            velocity = self._suffix_velocity(
                prefix_mask,
                kv_cache,
                actions,
                jnp.broadcast_to(time, (batch_size,)),
                z_model,
                layerwise_latents=layerwise_latents,
            )
            return self._mask_action_condition(actions + dt * velocity), time + dt

        def condition(carry: tuple[jax.Array, jax.Array]) -> jax.Array:
            return carry[1] >= -dt / 2

        result, _ = jax.lax.while_loop(condition, step, (noise, jnp.asarray(1.0)))
        return result[..., : self.config.active_action_dim]

    @override
    def sample_actions_rtc(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        prefix_actions: jax.Array,
        inference_delay: int,
        execution_horizon: int,
        *args,
        num_steps: int | at.Int[at.Array, ""] = 10,
        use_correction: bool = False,
        use_subtraction: bool = True,
        use_mask: bool = True,
        rtc_max_guidance_weight: float | None = None,
        state_delta: jax.Array | None = None,
        **kwargs,
    ) -> _model.Actions:
        """Guide a fresh Atomic flow sample toward the rolled previous chunk.

        This is the stock PI0.5 inference-time RTC contract adapted to the
        Atomic layerwise velocity field.  The checkpoint still predicts its
        trained 50-step horizon; ``execution_horizon`` only controls how the
        previous prediction is rolled and attended at the hand-off.
        """

        del args, kwargs, use_correction
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        if prefix_actions.shape[-1] == self.config.active_action_dim:
            prefix_actions = jnp.pad(
                prefix_actions,
                ((0, 0), (0, 0), (0, self.action_dim - self.config.active_action_dim)),
            )
        if prefix_actions.shape != (batch_size, self.action_horizon, self.action_dim):
            raise ValueError(
                "prefix_actions must match the fixed 50-step action tensor; "
                f"got {prefix_actions.shape}"
            )
        if state_delta is not None:
            aligned = state_delta[:, : self.config.active_action_dim]
            aligned = jnp.pad(
                aligned,
                ((0, 0), (0, self.action_dim - self.config.active_action_dim)),
            )
            # Arm joints are delta actions. Grippers (7 and 15) are absolute.
            delta_mask = jnp.zeros((self.action_dim,), dtype=prefix_actions.dtype)
            delta_mask = delta_mask.at[jnp.asarray(list(range(7)) + list(range(8, 15)))].set(1)
            prefix_actions = prefix_actions - aligned[:, None, :] * delta_mask[None, None, :]

        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        noise = self._mask_action_condition(noise)
        if self.config.enable_layerwise_atomic_flow:
            (
                query_hidden,
                prefix_mask,
                kv_cache,
                _,
                layerwise_latents,
            ) = self._prefix_forward(
                observation, return_layerwise_latents=True
            )
        else:
            query_hidden, prefix_mask, kv_cache, _ = self._prefix_forward(observation)
            layerwise_latents = None
        _, _, z_model, _, _ = self._latent(
            query_hidden, self._controlled_state(observation.state)
        )

        positions = jnp.arange(self.action_horizon)
        start = jnp.minimum(inference_delay, execution_horizon)
        denominator = jnp.maximum(execution_horizon - start + 1, 1)
        weights = jnp.clip((start - 1 - positions) / denominator + 1, 0, 1)
        weights = jnp.where(positions >= execution_horizon, 0, weights)
        mask_flag = jnp.asarray(use_mask, dtype=weights.dtype)
        weights = mask_flag * weights + (1.0 - mask_flag) * jnp.ones_like(weights)
        max_guidance = 10.0 if rtc_max_guidance_weight is None else rtc_max_guidance_weight
        dt = -1.0 / num_steps

        def step(carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            actions, time = carry
            velocity = self._suffix_velocity(
                prefix_mask,
                kv_cache,
                actions,
                jnp.broadcast_to(time, (batch_size,)),
                z_model,
                layerwise_latents=layerwise_latents,
            )
            denoised = actions - velocity * time
            residual = (prefix_actions - denoised) * weights[None, :, None]
            tau = 1.0 - time
            square = time * time
            inv_r2 = (square + tau * tau) / jnp.maximum(square, 1e-6)
            guidance = jnp.minimum(
                jnp.nan_to_num(time / jnp.maximum(tau, 1e-6), posinf=max_guidance)
                * inv_r2,
                max_guidance,
            )
            sign = 1.0 - 2.0 * jnp.asarray(use_subtraction, dtype=velocity.dtype)
            guided_velocity = velocity + sign * guidance * residual
            updated = self._mask_action_condition(actions + dt * guided_velocity)
            return updated, time + dt

        def condition(carry: tuple[jax.Array, jax.Array]) -> jax.Array:
            return carry[1] >= -dt / 2

        result, _ = jax.lax.while_loop(condition, step, (noise, jnp.asarray(1.0)))
        return result[..., : self.config.active_action_dim]

    def sample_actions_rtc_with_cutoffs(self, *args, **kwargs) -> dict[str, _model.Actions]:
        """Compatibility hook used only when diagnostic noise plots are requested."""

        return {"t0.0": self.sample_actions_rtc(*args, **kwargs)}
