"""Force-history representation and fast latent correction for AtomicPi05.

The force-free VLA remains the nominal policy. This module is an optional
second-stage side path:

* a shared 120 Hz force/state encoder emits position-aware 30 Hz tokens;
* independent slow/fast projection heads adapt the shared tokens without
  duplicating the temporal encoder;
* an external learned Qforce first reads frozen VLM prefix memory, then uses
  that semantic query to read slow force tokens, producing the slow contact
  latent without adding tokens to the VLM;
* a training-only GRU decoder consumes the concatenated right/left slow
  latents and predicts both future force sequences without an action shortcut;
* every ten action steps, the two nominal arm latents independently query
  their corresponding force memory through shared weights, producing one
  zero-initialized ``delta_z`` per arm;
* one learned scalar per Action-Expert depth broadcasts those two corrections
  into the corresponding arm latents before the existing arm fusion/adapter.

No recurrent state is carried between calls. Online state is the explicit
rolling force/state window, preventing a stale GRU hidden state from drifting.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.nnx as nnx
from flax import struct

from .config import AtomicPi05Config


# NNX stores initializer callables in graph metadata.  Factory initializers
# such as ``orthogonal()`` return a fresh closure on every invocation, which
# makes two otherwise identical model constructions have unequal GraphDefs and
# breaks ``jax.eval_shape`` -> sharded initialization.  Reuse one callable so
# graph metadata is deterministic across constructions.
_STABLE_GRU_RECURRENT_INIT = nnx.initializers.orthogonal()


class _StableGRUCell(nnx.Module):
    """NNX GRU without non-array or retained-RNG checkpoint leaves.

    ``nnx.GRUCell`` stores both ``dense_h.bias=Param(None)`` and its
    initialization ``rngs`` object. Those leaves vary across NNX/Orbax
    versions and cannot be bf16-cast by the frozen-parameter path. This cell
    implements the identical equations using only numeric Linear parameters.
    """

    def __init__(self, in_features: int, hidden_features: int, *, rngs: nnx.Rngs):
        self.dense_i = nnx.Linear(
            in_features,
            3 * hidden_features,
            use_bias=True,
            rngs=rngs,
        )
        self.dense_h = nnx.Linear(
            hidden_features,
            3 * hidden_features,
            use_bias=True,
            kernel_init=_STABLE_GRU_RECURRENT_INIT,
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def __call__(self, carry: jax.Array, inputs: jax.Array) -> tuple[jax.Array, jax.Array]:
        x_transformed = self.dense_i(inputs)
        h_transformed = self.dense_h(carry)
        xi_r, xi_z, xi_n = jnp.split(x_transformed, 3, axis=-1)
        hh_r, hh_z, hh_n = jnp.split(h_transformed, 3, axis=-1)
        reset = jax.nn.sigmoid(xi_r + hh_r)
        update = jax.nn.sigmoid(xi_z + hh_z)
        candidate = jnp.tanh(xi_n + reset * hh_n)
        next_hidden = (1.0 - update) * candidate + update * carry
        return next_hidden, next_hidden


def sincos_embedding(
    position: jax.Array,
    dim: int,
    *,
    base: float = 10_000.0,
) -> jax.Array:
    """Transformer sinusoidal embedding with an explicit frequency base."""

    if dim % 2:
        raise ValueError("force phase embedding width must be even")
    if base <= 1:
        raise ValueError("force position base must be greater than one")
    frequency = 1.0 / (
        float(base) ** (2.0 * jnp.arange(dim // 2, dtype=jnp.float32) / float(dim))
    )
    angles = jnp.einsum("b,d->bd", position.astype(jnp.float32), frequency)
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


def force_token_lag_positions(token_count: int) -> jax.Array:
    """Return end-aligned token indices for a force-history window.

    Slow and fast histories use the same sampling clock and temporal encoder;
    they differ only in how far into the past they extend.  Encoding every
    window independently on ``[0, 1]`` would therefore make a 10-token fast
    window appear to span the same duration as a 30-token slow window.  Use
    integer lags relative to the newest token instead, so the fast positions
    ``[-9,...,0]`` are exactly the suffix of slow ``[-29,...,0]``.
    """

    if token_count <= 0:
        raise ValueError("force token count must be positive")
    return jnp.arange(token_count, dtype=jnp.float32) - (token_count - 1)


def canonicalize_force_state_views(state: jax.Array) -> jax.Array:
    """Build right/left encoder views with the local arm state first.

    Recorded CR1 force-side state is ordered ``[left, right]``, while AtomicPi05
    arm latents and force tensors are ordered ``[right, left]``.  The temporal
    encoder shares every parameter across arms, so each branch must receive the
    same semantic layout: ``[local_arm, opposite_arm]``.

    Args:
        state: Shared normalized state with shape ``[B,T,2*S]`` and recorded
            order ``[left_S, right_S]``.

    Returns:
        State views with shape ``[B,2,T,2*S]`` and arm order ``[right,left]``.
    """

    if state.ndim != 3:
        raise ValueError("shared force state must have shape [B,T,S]")
    state_dim = state.shape[-1]
    if state_dim <= 0 or state_dim % 2:
        raise ValueError("shared force state width must split evenly across two arms")
    half = state_dim // 2
    left_state = state[..., :half]
    right_state = state[..., half:]
    right_view = jnp.concatenate([right_state, left_state], axis=-1)
    left_view = jnp.concatenate([left_state, right_state], axis=-1)
    return jnp.stack([right_view, left_view], axis=1)


def future_force_delta_target(
    history_force: jax.Array,
    future_force: jax.Array,
    future_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return every 120 Hz future wrench relative to the current wrench.

    Inputs must already use the training force normalization. The final
    history sample is the force at the current action timestamp. Keeping all
    200 future samples prevents brief contact transients from disappearing in
    a four-sample average.
    """

    if future_force.ndim not in (3, 4) or future_mask.shape != future_force.shape[:-1]:
        raise ValueError(
            "future force/mask must have shapes [B,T,F]/[B,T] or [B,arm,T,F]/[B,arm,T]"
        )
    if history_force.shape[:-2] != future_force.shape[:-2]:
        raise ValueError("history and future force leading dimensions must match")
    if history_force.shape[-1] != future_force.shape[-1]:
        raise ValueError("history and future force widths must match")
    current = history_force[..., -1, :]
    return future_force - current[..., None, :], future_mask


def temporal_masked_mean(
    values: jax.Array,
    mask: jax.Array,
    *,
    stride: int,
) -> tuple[jax.Array, jax.Array]:
    """Pool consecutive temporal samples without treating padding as zero."""

    if values.ndim not in (3, 4) or mask.shape != values.shape[:-1]:
        raise ValueError("values/mask must be [B,T,D]/[B,T] or [B,arm,T,D]/[B,arm,T]")
    if stride <= 0 or values.shape[-2] % stride:
        raise ValueError("temporal length must be divisible by a positive stride")
    steps, width = values.shape[-2:]
    leading = values.shape[:-2]
    grouped_values = values.reshape(*leading, steps // stride, stride, width)
    grouped_mask = mask.reshape(*leading, steps // stride, stride)
    weights = grouped_mask.astype(values.dtype)
    pooled = jnp.sum(grouped_values * weights[..., None], axis=-2) / jnp.maximum(
        jnp.sum(weights, axis=-1, keepdims=True), 1
    )
    return pooled, jnp.any(grouped_mask, axis=-1)


def future_force_forecast_metrics(
    prediction: jax.Array,
    target: jax.Array,
    mask: jax.Array,
    *,
    coarse_stride: int = 4,
    sample_rate_hz: float = 120.0,
) -> dict[str, jax.Array]:
    """Held-out metrics sensitive to smoothness and contact-peak timing."""

    if prediction.shape != target.shape or mask.shape != target.shape[:-1]:
        raise ValueError("prediction/target/mask shapes must align")
    if prediction.ndim != 4:
        raise ValueError("force forecasts must have shape [B,arm,T,D]")

    def masked_rmse(error: jax.Array, valid: jax.Array) -> jax.Array:
        weights = valid[..., None].astype(error.dtype)
        denominator = jnp.maximum(jnp.sum(weights) * error.shape[-1], 1)
        return jnp.sqrt(jnp.sum(jnp.square(error) * weights) / denominator)

    raw_rmse = masked_rmse(prediction - target, mask)
    coarse_prediction, coarse_mask = temporal_masked_mean(
        prediction, mask, stride=coarse_stride
    )
    coarse_target, _ = temporal_masked_mean(target, mask, stride=coarse_stride)
    coarse_rmse = masked_rmse(coarse_prediction - coarse_target, coarse_mask)

    derivative_mask = mask[..., 1:] & mask[..., :-1]
    prediction_rate = jnp.diff(prediction, axis=-2) * sample_rate_hz
    target_rate = jnp.diff(target, axis=-2) * sample_rate_hz
    derivative_rmse = masked_rmse(prediction_rate - target_rate, derivative_mask)

    valid_channel = jnp.any(mask, axis=-1)[..., None]
    masked_prediction_abs = jnp.where(mask[..., None], jnp.abs(prediction), -jnp.inf)
    masked_target_abs = jnp.where(mask[..., None], jnp.abs(target), -jnp.inf)
    prediction_peak = jnp.max(masked_prediction_abs, axis=-2)
    target_peak = jnp.max(masked_target_abs, axis=-2)
    peak_weights = valid_channel.astype(prediction.dtype)
    peak_denominator = jnp.maximum(jnp.sum(peak_weights) * prediction.shape[-1], 1)
    peak_amplitude_mae = jnp.sum(
        jnp.abs(prediction_peak - target_peak) * peak_weights
    ) / peak_denominator

    prediction_peak_index = jnp.argmax(masked_prediction_abs, axis=-2)
    target_peak_index = jnp.argmax(masked_target_abs, axis=-2)
    peak_timing_error_samples = jnp.sum(
        jnp.abs(prediction_peak_index - target_peak_index) * peak_weights
    ) / peak_denominator
    return {
        "raw_rmse": raw_rmse,
        "coarse_rmse": coarse_rmse,
        "derivative_rmse_per_s": derivative_rmse,
        "peak_amplitude_mae": peak_amplitude_mae,
        "peak_timing_mae_samples": peak_timing_error_samples,
        "peak_timing_mae_ms": peak_timing_error_samples * (1000.0 / sample_rate_hz),
    }


@struct.dataclass
class ForceContextOutput:
    latent: jax.Array  # [B,2,force_latent_dim], ordered [right,left]
    history_tokens: jax.Array  # [B,2,history/4,force_encoder_width]
    history_token_mask: jax.Array  # [B,2,history/4]


@struct.dataclass
class ForceModulationOutput:
    delta_z: jax.Array  # [B,2,latent_dim], ordered [right,left]
    z_exec: jax.Array  # [B,2,latent_dim]
    recent_tokens: jax.Array  # [B,2,update_action_steps,force_encoder_width]


class ForceTemporalBlock(nnx.Module):
    """Small pre-norm temporal Transformer block."""

    def __init__(self, config: AtomicPi05Config, *, rngs: nnx.Rngs):
        width = config.force_encoder_width
        # Keep explicit zero-initialized biases. NNX represents a disabled bias
        # as Param(None), which the base trainer cannot cast when this branch is
        # frozen during B1/B2 partitioning.
        self.attn_norm = nnx.LayerNorm(width, use_bias=True, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=config.force_encoder_num_heads,
            in_features=width,
            qkv_features=width,
            out_features=width,
            deterministic=True,
            decode=False,
            rngs=rngs,
        )
        self.ffn_norm = nnx.LayerNorm(width, use_bias=True, rngs=rngs)
        self.ffn_in = nnx.Linear(width, config.force_encoder_mlp_dim, rngs=rngs)
        self.ffn_out = nnx.Linear(config.force_encoder_mlp_dim, width, rngs=rngs)

    def __call__(self, hidden: jax.Array, token_mask: jax.Array) -> jax.Array:
        # All tokens are historical at the time of the call, so bidirectional
        # attention inside this explicit rolling window cannot see the future.
        valid_attention = token_mask[:, None, :, None] & token_mask[:, None, None, :]
        normalized = self.attn_norm(hidden)
        hidden = hidden + self.attn(normalized, mask=valid_attention)
        normalized = self.ffn_norm(hidden)
        hidden = hidden + self.ffn_out(nnx.swish(self.ffn_in(normalized)))
        return jnp.where(token_mask[..., None], hidden, jnp.zeros_like(hidden))


class ForceProjectionHead(nnx.Module):
    """Small residual projection specialized for the slow or fast use case."""

    def __init__(self, width: int, *, rngs: nnx.Rngs):
        self.norm = nnx.LayerNorm(width, use_bias=True, rngs=rngs)
        self.hidden = nnx.Linear(width, width, rngs=rngs)
        self.out = nnx.Linear(width, width, rngs=rngs)

    def __call__(self, tokens: jax.Array, token_mask: jax.Array) -> jax.Array:
        projected = tokens + self.out(nnx.swish(self.hidden(self.norm(tokens))))
        return jnp.where(token_mask[..., None], projected, jnp.zeros_like(projected))


class ForceConditioner(nnx.Module):
    """Shared slow/fast force encoder and z_M residual generator."""

    def __init__(
        self,
        prefix_dim: int,
        action_dim: int,
        latent_dim: int,
        config: AtomicPi05Config,
        *,
        num_layers: int = 18,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.action_dim = action_dim
        self.num_layers = int(num_layers)
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        width = config.force_encoder_width
        grouped_input_dim = config.force_temporal_stride * (
            config.force_dim + config.force_state_dim
        )
        self.history_in = nnx.Linear(grouped_input_dim, width, rngs=rngs)
        self.history_norm = nnx.LayerNorm(width, use_bias=True, rngs=rngs)
        self.history_blocks = nnx.Dict(
            **{
                f"block_{index}": ForceTemporalBlock(config, rngs=rngs)
                for index in range(config.force_encoder_depth)
            }
        )
        # The sensor statistics and temporal structure are shared, while the
        # slow head may retain predictive context and the fast head may retain
        # abrupt contact changes. They deliberately do not share parameters.
        self.slow_projection = ForceProjectionHead(width, rngs=rngs)
        self.fast_projection = ForceProjectionHead(width, rngs=rngs)

        # Legacy checkpoints form zF from prefix/zM plus force.  New
        # force-only checkpoints deliberately omit those semantic modules so
        # zF cannot become a second action/prefix pathway.
        self.force_query = nnx.Param(0.02 * jax.random.normal(rngs(), (width,)))
        self.force_query_norm = nnx.LayerNorm(width, use_bias=True, rngs=rngs)
        self.force_attention = nnx.MultiHeadAttention(
            num_heads=config.force_encoder_num_heads,
            in_features=width,
            qkv_features=width,
            out_features=width,
            deterministic=True,
            decode=False,
            rngs=rngs,
        )
        if config.force_context_from_prefix:
            self.force_query_from_z = nnx.Linear(latent_dim, width, rngs=rngs)
            self.semantic_query_norm = nnx.LayerNorm(width, use_bias=True, rngs=rngs)
            self.semantic_memory = nnx.Linear(prefix_dim, width, rngs=rngs)
            self.semantic_attention = nnx.MultiHeadAttention(
                num_heads=config.force_encoder_num_heads,
                in_features=width,
                qkv_features=width,
                out_features=width,
                deterministic=True,
                decode=False,
                rngs=rngs,
            )
            self.latent_in = nnx.Linear(latent_dim + 2 * width, 2 * width, rngs=rngs)
        else:
            self.latent_in = nnx.Linear(width, 2 * width, rngs=rngs)
        self.latent_out = nnx.Linear(2 * width, config.force_latent_dim, rngs=rngs)
        self.latent_norm = nnx.LayerNorm(config.force_latent_dim, use_bias=True, rngs=rngs)

        # Training-only future-force GRU. z_F initializes every recurrent
        # layer and is injected at every step. It deliberately receives no
        # nominal/ground-truth action, so future-force supervision must shape
        # the slow force latent rather than use an action-only shortcut.
        joint_force_latent_dim = config.arm_count * config.force_latent_dim
        future_condition_dim = joint_force_latent_dim
        if config.force_future_condition_on_zm:
            future_condition_dim += config.arm_count * latent_dim
        self.future_latent_step = nnx.Linear(future_condition_dim, width, rngs=rngs)
        self.future_initial = nnx.Linear(
            future_condition_dim,
            config.force_future_decoder_depth * width,
            rngs=rngs,
        )
        future_gru_cells = {}
        for index in range(config.force_future_decoder_depth):
            cell = _StableGRUCell(width, width, rngs=rngs)
            future_gru_cells[f"layer_{index}"] = cell
        self.future_gru_cells = nnx.Dict(**future_gru_cells)
        if config.force_future_decoder_kind == "linear_chunk":
            self.future_out = nnx.Linear(
                width,
                config.arm_count * config.force_future_decoder_stride * config.force_dim,
                rngs=rngs,
            )
        elif config.force_future_decoder_kind == "phase_mlp":
            self.future_phase_embedding = nnx.Param(
                0.02
                * jax.random.normal(
                    rngs(),
                    (config.force_future_decoder_stride, width),
                )
            )
            self.future_phase_norm = nnx.LayerNorm(width, use_bias=True, rngs=rngs)
            self.future_phase_hidden = nnx.Linear(width, width, rngs=rngs)
            self.future_phase_out = nnx.Linear(
                width,
                config.arm_count * config.force_dim,
                rngs=rngs,
            )
        else:  # AtomicPi05Config validates this before module construction.
            raise ValueError(f"unknown future decoder kind: {config.force_future_decoder_kind}")

        # Fast correction. z_M alone supplies the semantic/action query. The
        # force side is represented only in K/V: one slow token followed by
        # the latest projected force tokens. A learned type embedding marks
        # which temporal scale produced each memory token.
        self.fast_query = nnx.Linear(latent_dim, width, rngs=rngs)
        self.fast_query_norm = nnx.LayerNorm(width, use_bias=True, rngs=rngs)
        self.slow_memory_from_latent = nnx.Linear(config.force_latent_dim, width, rngs=rngs)
        self.force_scale_embedding = nnx.Param(0.02 * jax.random.normal(rngs(), (2, width)))
        self.fast_attention = nnx.MultiHeadAttention(
            num_heads=config.force_encoder_num_heads,
            in_features=width,
            qkv_features=width,
            out_features=width,
            deterministic=True,
            decode=False,
            rngs=rngs,
        )
        self.fast_hidden = nnx.Linear(width, 2 * width, rngs=rngs)
        if config.force_full_token_adapter:
            # B2-only adapter.  The completed B1 checkpoint supplies the
            # temporal encoder and SlowProj weights; these new parameters are
            # initialized when the larger graph restores that checkpoint.
            # Slow and fast tokens remain separate K/V entries so zM+phase can
            # select the force history relevant to the current action update.
            self.full_token_adapter_query = nnx.Linear(latent_dim, width, rngs=rngs)
            self.full_token_adapter_query_norm = nnx.LayerNorm(
                width, use_bias=True, rngs=rngs
            )
            self.full_token_adapter_scale_embedding = nnx.Param(
                0.02 * jax.random.normal(rngs(), (2, width))
            )
            self.full_token_adapter_attention = nnx.MultiHeadAttention(
                num_heads=config.force_full_token_adapter_heads,
                in_features=width,
                qkv_features=width,
                out_features=width,
                deterministic=True,
                decode=False,
                rngs=rngs,
            )
            self.full_token_adapter_hidden = nnx.Linear(width, 2 * width, rngs=rngs)
            # Exact zero initialization makes step zero bitwise-equivalent to
            # the frozen AFRO policy while leaving a non-zero gradient into
            # this projection on the first B2 update.
            self.full_token_adapter_out = nnx.Linear(
                2 * width,
                latent_dim,
                kernel_init=nnx.initializers.zeros_init(),
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )
        if config.enable_force_hidden_cross_attention:
            # Per-arm force token used directly by the Action Expert.  Slow
            # context and the pooled newest fast tokens are concatenated and
            # reduced by an MLP; no zM feature enters this force-only token.
            self.force_token_in = nnx.Linear(
                config.force_latent_dim + width, 2 * width, rngs=rngs
            )
            self.force_token_out = nnx.Linear(2 * width, width, rngs=rngs)
            self.force_token_norm = nnx.LayerNorm(width, use_bias=True, rngs=rngs)
        # The final projection is exactly zero at initialization, so enabling
        # the force module initially reproduces the force-free VLA output.
        self.delta_out = nnx.Linear(
            2 * width,
            latent_dim,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        # One shared scalar per Action-Expert depth distributes each arm's
        # single fast cross-attention correction.  delta_out is exactly zero
        # at initialization, so gates must start non-zero or the bilinear
        # gate*delta path would receive no gradient.  2*sigmoid(0)=1 while
        # keeping every learned gate positive and bounded by two.
        self.layer_gate_logits = nnx.Param(jnp.zeros((self.num_layers,), dtype=jnp.float32))

    def layer_gates(self) -> jax.Array:
        """Return shared right/left force injection strengths for all layers."""

        return 2.0 * jax.nn.sigmoid(self.layer_gate_logits.value)

    def compose_force_tokens(
        self,
        force_latent: jax.Array,
        recent_tokens: jax.Array,
        current_history_mask: jax.Array,
        update_offset: jax.Array,
    ) -> jax.Array:
        """Return ordered right/left slow+fast+RTC-phase tokens ``[B,2,D]``."""

        if not self.config.enable_force_hidden_cross_attention:
            raise RuntimeError("force hidden cross-attention is disabled")
        batch, arms, token_count, width = recent_tokens.shape
        if force_latent.shape != (batch, arms, self.config.force_latent_dim):
            raise ValueError("force_latent must have shape [B,2,force_latent_dim]")
        if update_offset.shape != (batch,):
            raise ValueError("update_offset must have shape [B]")
        fast_samples = self.config.force_fast_history_samples
        raw_mask = current_history_mask[:, :, -fast_samples:]
        if raw_mask.shape[:2] != (batch, arms):
            raise ValueError("current_history_mask must have shape [B,2,T]")
        stride = self.config.force_temporal_stride
        token_mask = jnp.any(
            raw_mask.reshape(batch, arms, fast_samples // stride, stride), axis=-1
        )
        if token_mask.shape[-1] != token_count:
            raise ValueError("fast token/mask length mismatch")
        weights = token_mask[..., None].astype(recent_tokens.dtype)
        pooled_fast = jnp.sum(recent_tokens * weights, axis=-2) / jnp.maximum(
            jnp.sum(weights, axis=-2), 1
        )
        combined = jnp.concatenate([force_latent, pooled_fast], axis=-1)
        token = self.force_token_out(nnx.swish(self.force_token_in(combined)))
        # Preserve the Training-Time RTC phase used by the original fast
        # correction.  Both arm tokens receive the same fixed-horizon offset
        # (0/10/20/30/40), while their force/state contents remain independent.
        phase = sincos_embedding(
            update_offset.astype(jnp.float32),
            width,
            base=self.config.force_position_base,
        )
        token = token + phase[:, None, :].astype(token.dtype)
        return self.force_token_norm(token)

    def encode_history(
        self,
        force: jax.Array,
        state: jax.Array,
        sample_mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        config = self.config
        if force.ndim != 3 or force.shape[-1] != config.force_dim:
            raise ValueError(f"force must have shape [B,T,{config.force_dim}]")
        if state.shape != (*force.shape[:2], config.force_state_dim):
            raise ValueError(f"force state must have shape [B,T,{config.force_state_dim}]")
        if sample_mask.shape != force.shape[:2]:
            raise ValueError("force sample mask must have shape [B,T]")
        if force.shape[1] % config.force_temporal_stride:
            raise ValueError("force history length must be divisible by temporal stride")

        values = jnp.concatenate([force, state], axis=-1)
        values = jnp.where(sample_mask[..., None], values, jnp.zeros_like(values))
        batch, samples, channels = values.shape
        stride = config.force_temporal_stride
        grouped = values.reshape(batch, samples // stride, stride * channels)
        grouped_mask = jnp.any(sample_mask.reshape(batch, samples // stride, stride), axis=2)
        hidden = self.history_norm(nnx.swish(self.history_in(grouped)))
        # A non-overlapping Linear over four flattened samples is exactly a
        # Conv1D(kernel=4,stride=4) tokenization. Add explicit relative time so
        # the following attention is not permutation-invariant.
        token_count = hidden.shape[1]
        token_lag_positions = force_token_lag_positions(token_count)
        hidden = hidden + sincos_embedding(
            token_lag_positions,
            hidden.shape[-1],
            base=config.force_position_base,
        )[None].astype(hidden.dtype)
        hidden = jnp.where(grouped_mask[..., None], hidden, jnp.zeros_like(hidden))

        # Multi-head attention cannot consume an entirely masked row. Expose a
        # zero placeholder internally; retain the real mask for pooling/output.
        safe_mask = grouped_mask.at[:, -1].set(grouped_mask[:, -1] | ~jnp.any(grouped_mask, axis=1))
        for index in range(config.force_encoder_depth):
            hidden = self.history_blocks[f"block_{index}"](hidden, safe_mask)
        hidden = jnp.where(grouped_mask[..., None], hidden, jnp.zeros_like(hidden))
        return hidden, grouped_mask

    def encode_context(
        self,
        prefix_hidden: jax.Array,
        prefix_mask: jax.Array,
        z_model: jax.Array,
        history_force: jax.Array,
        history_state: jax.Array,
        history_mask: jax.Array,
    ) -> ForceContextOutput:
        if history_force.ndim != 4 or history_force.shape[1] != self.config.arm_count:
            raise ValueError(
                f"history_force must have shape [B,{self.config.arm_count},T,{self.config.force_dim}]"
            )
        batch, arms, samples, _ = history_force.shape
        if z_model.shape != (batch, arms, self.config.latent_dim):
            raise ValueError(f"z_model must have shape {(batch, arms, self.config.latent_dim)}")
        if history_state.shape == (batch, samples, self.config.force_state_dim):
            history_state = canonicalize_force_state_views(history_state)
        if history_state.shape != (
            batch,
            arms,
            samples,
            self.config.force_state_dim,
        ):
            raise ValueError(
                "history_state must be recorded-order shared [B,T,S] or "
                "canonical [local,opposite] bimanual [B,2,T,S]"
            )
        if history_mask.shape != (batch, arms, samples):
            raise ValueError("history_mask must have shape [B,2,T]")
        flat_force = history_force.reshape(batch * arms, samples, self.config.force_dim)
        flat_state = history_state.reshape(batch * arms, samples, self.config.force_state_dim)
        flat_mask = history_mask.reshape(batch * arms, samples)
        flat_z = z_model.reshape(batch * arms, self.config.latent_dim)
        history_tokens, token_mask = self.encode_history(flat_force, flat_state, flat_mask)
        slow_tokens = self.slow_projection(history_tokens, token_mask)
        if self.config.force_context_from_prefix:
            query = (self.force_query.value[None] + self.force_query_from_z(flat_z))[:, None]
            repeated_prefix = jnp.broadcast_to(
                prefix_hidden[:, None],
                (batch, arms, *prefix_hidden.shape[1:]),
            ).reshape(batch * arms, *prefix_hidden.shape[1:])
            repeated_prefix_mask = jnp.broadcast_to(
                prefix_mask[:, None], (batch, arms, prefix_mask.shape[1])
            ).reshape(batch * arms, prefix_mask.shape[1])
            memory = self.semantic_memory(repeated_prefix)
            semantic_mask = repeated_prefix_mask[:, None, None, :]
            semantic = self.semantic_attention(
                self.semantic_query_norm(query), memory, memory, mask=semantic_mask
            )
            query_after_semantic = query + semantic
        else:
            query_after_semantic = jnp.broadcast_to(
                self.force_query.value[None, None], (batch * arms, 1, slow_tokens.shape[-1])
            )
        has_force = jnp.any(token_mask, axis=1)
        safe_token_mask = token_mask.at[:, -1].set(token_mask[:, -1] | ~has_force)
        force_context = self.force_attention(
            self.force_query_norm(query_after_semantic),
            slow_tokens,
            slow_tokens,
            mask=safe_token_mask[:, None, None, :],
        )
        force_context = jnp.where(
            has_force[:, None, None], force_context, jnp.zeros_like(force_context)
        )
        query_final = query_after_semantic + force_context
        latent_input = (
            jnp.concatenate(
                [flat_z, query_after_semantic[:, 0], query_final[:, 0]], axis=-1
            )
            if self.config.force_context_from_prefix
            else force_context[:, 0]
        )
        latent = self.latent_norm(self.latent_out(nnx.swish(self.latent_in(latent_input))))
        return ForceContextOutput(
            latent.reshape(batch, arms, self.config.force_latent_dim),
            slow_tokens.reshape(batch, arms, slow_tokens.shape[1], slow_tokens.shape[2]),
            token_mask.reshape(batch, arms, token_mask.shape[1]),
        )

    def predict_future_force_delta(
        self,
        force_latent: jax.Array,
        z_model: jax.Array | None = None,
    ) -> jax.Array:
        if force_latent.ndim != 3 or force_latent.shape[1:] != (
            self.config.arm_count,
            self.config.force_latent_dim,
        ):
            raise ValueError("force_latent must have shape [B,2,force_latent_dim]")
        batch = force_latent.shape[0]
        joint_latent = force_latent.reshape(
            batch, self.config.arm_count * self.config.force_latent_dim
        )
        if self.config.force_future_condition_on_zm:
            if z_model is None or z_model.shape != (
                batch,
                self.config.arm_count,
                self.config.latent_dim,
            ):
                raise ValueError("z_model must have shape [B,2,latent_dim] for conditioned forecast")
            frozen_z_model = jax.lax.stop_gradient(z_model)
            future_condition = jnp.concatenate(
                [joint_latent, frozen_z_model.reshape(batch, -1)], axis=-1
            )
        else:
            future_condition = joint_latent
        stride = self.config.force_future_decoder_stride
        horizon = self.config.force_future_samples // stride
        future_positions = jnp.arange(1, horizon + 1, dtype=jnp.float32)
        decoder_inputs = (
            self.future_latent_step(future_condition)[:, None]
            + sincos_embedding(
                future_positions,
                self.config.force_encoder_width,
                base=self.config.force_position_base,
            )[None]
        )
        initial = self.future_initial(future_condition).reshape(
            batch,
            self.config.force_future_decoder_depth,
            self.config.force_encoder_width,
        )
        initial = jnp.swapaxes(initial, 0, 1)

        def decode_step(carry: jax.Array, step_input: jax.Array) -> tuple[jax.Array, jax.Array]:
            output = step_input
            next_layers = []
            for index in range(self.config.force_future_decoder_depth):
                next_hidden, output = self.future_gru_cells[f"layer_{index}"](carry[index], output)
                next_layers.append(next_hidden)
            return jnp.stack(next_layers, axis=0), output

        _, decoded = jax.lax.scan(
            decode_step,
            initial,
            jnp.swapaxes(decoder_inputs, 0, 1),
        )
        decoded = jnp.swapaxes(decoded, 0, 1)
        if self.config.force_future_decoder_kind == "linear_chunk":
            prediction = self.future_out(decoded)
            prediction = prediction.reshape(
                batch,
                horizon,
                self.config.arm_count,
                stride,
                self.config.force_dim,
            )
            prediction = jnp.transpose(prediction, (0, 2, 1, 3, 4))
        else:
            # Every 30 Hz GRU state is expanded into four explicitly ordered
            # 120 Hz phases. The same MLP decodes every phase and time step;
            # only the learned phase token distinguishes the within-block slot.
            phase_hidden = (
                decoded[:, :, None, :]
                + self.future_phase_embedding.value[None, None, :, :]
            )
            phase_hidden = nnx.swish(
                self.future_phase_hidden(self.future_phase_norm(phase_hidden))
            )
            prediction = self.future_phase_out(phase_hidden)
            prediction = prediction.reshape(
                batch,
                horizon,
                stride,
                self.config.arm_count,
                self.config.force_dim,
            )
            prediction = jnp.transpose(prediction, (0, 3, 1, 2, 4))
        return prediction.reshape(
            batch,
            self.config.arm_count,
            self.config.force_future_samples,
            self.config.force_dim,
        )

    def modulate(
        self,
        z_model: jax.Array,
        force_latent: jax.Array,
        current_force: jax.Array,
        current_state: jax.Array,
        current_mask: jax.Array,
        update_offset: jax.Array,
        *,
        slow_history_tokens: jax.Array | None = None,
        slow_history_token_mask: jax.Array | None = None,
        full_token_memory_scale: jax.Array | None = None,
    ) -> ForceModulationOutput:
        if current_force.ndim != 4 or current_force.shape[1] != self.config.arm_count:
            raise ValueError("current_force must have shape [B,2,T,force_dim]")
        batch, arms, samples, _ = current_force.shape
        if z_model.shape != (batch, arms, self.config.latent_dim):
            raise ValueError("z_model must have shape [B,2,latent_dim]")
        if force_latent.shape != (batch, arms, self.config.force_latent_dim):
            raise ValueError("force_latent must have shape [B,2,force_latent_dim]")
        if current_state.shape == (batch, samples, self.config.force_state_dim):
            current_state = canonicalize_force_state_views(current_state)
        if current_state.shape != (batch, arms, samples, self.config.force_state_dim):
            raise ValueError(
                "current_state must be recorded-order shared [B,T,S] or "
                "canonical [local,opposite] bimanual [B,2,T,S]"
            )
        if current_mask.shape != (batch, arms, samples):
            raise ValueError("current_mask must have shape [B,2,T]")
        # Slice the raw 120 Hz window *before* temporal self-attention. Taking
        # the last ten tokens after encoding a full second would leak the
        # preceding 0.67 s into every fast token through bidirectional attention.
        fast_samples = self.config.force_fast_history_samples
        current_force = current_force[:, :, -fast_samples:]
        current_state = current_state[:, :, -fast_samples:]
        current_mask = current_mask[:, :, -fast_samples:]
        current_force = current_force.reshape(batch * arms, fast_samples, self.config.force_dim)
        current_state = current_state.reshape(
            batch * arms, fast_samples, self.config.force_state_dim
        )
        current_mask = current_mask.reshape(batch * arms, fast_samples)
        shared_tokens, token_mask = self.encode_history(current_force, current_state, current_mask)
        tokens = self.fast_projection(shared_tokens, token_mask)
        recent_tokens = tokens
        recent_mask = token_mask
        repeated_offset = jnp.broadcast_to(update_offset[:, None], (batch, arms)).reshape(-1)
        phase_hidden = sincos_embedding(
            repeated_offset.astype(jnp.float32),
            self.config.force_encoder_width,
            base=self.config.force_position_base,
        )
        flat_z = z_model.reshape(batch * arms, self.config.latent_dim)
        flat_slow = force_latent.reshape(batch * arms, self.config.force_latent_dim)
        if self.config.force_full_token_adapter:
            expected_slow_tokens = (
                batch,
                arms,
                self.config.force_history_samples // self.config.force_temporal_stride,
                self.config.force_encoder_width,
            )
            expected_slow_mask = expected_slow_tokens[:-1]
            if slow_history_tokens is None or slow_history_tokens.shape != expected_slow_tokens:
                raise ValueError(
                    "full-token adapter requires slow_history_tokens with shape "
                    f"{expected_slow_tokens}"
                )
            if (
                slow_history_token_mask is None
                or slow_history_token_mask.shape != expected_slow_mask
            ):
                raise ValueError(
                    "full-token adapter requires slow_history_token_mask with shape "
                    f"{expected_slow_mask}"
                )
            flat_slow_tokens = slow_history_tokens.reshape(
                batch * arms,
                expected_slow_tokens[2],
                self.config.force_encoder_width,
            )
            flat_slow_mask = slow_history_token_mask.reshape(
                batch * arms, expected_slow_tokens[2]
            )
            query = self.full_token_adapter_query_norm(
                self.full_token_adapter_query(flat_z) + phase_hidden
            )[:, None]
            slow_memory = (
                flat_slow_tokens
                + self.full_token_adapter_scale_embedding.value[0][None, None]
            )
            fast_memory = (
                recent_tokens
                + self.full_token_adapter_scale_embedding.value[1][None, None]
            )
            force_memory = jnp.concatenate([slow_memory, fast_memory], axis=1)
            force_memory_mask = jnp.concatenate([flat_slow_mask, recent_mask], axis=1)
            if full_token_memory_scale is not None:
                force_memory = force_memory * jnp.asarray(
                    full_token_memory_scale, dtype=force_memory.dtype
                )
            attended = self.full_token_adapter_attention(
                query,
                force_memory,
                force_memory,
                mask=force_memory_mask[:, None, None, :],
            )[:, 0]
            delta_z = self.full_token_adapter_out(
                nnx.swish(self.full_token_adapter_hidden(attended))
            )
        else:
            query = self.fast_query_norm(self.fast_query(flat_z) + phase_hidden)[:, None]
            slow_token = self.slow_memory_from_latent(flat_slow)[:, None]
            slow_token = slow_token + self.force_scale_embedding.value[0][None, None]
            fast_tokens = recent_tokens + self.force_scale_embedding.value[1][None, None]
            force_memory = jnp.concatenate([slow_token, fast_tokens], axis=1)
            force_memory_mask = jnp.concatenate(
                [
                    jnp.ones((recent_mask.shape[0], 1), dtype=jnp.bool_),
                    recent_mask,
                ],
                axis=1,
            )
            attended = self.fast_attention(
                query,
                force_memory,
                force_memory,
                mask=force_memory_mask[:, None, None, :],
            )[:, 0]
            delta_z = self.delta_out(nnx.swish(self.fast_hidden(attended)))
        # The slow z_F memory is valid at offset zero even though no post-anchor
        # fast samples have arrived yet. Mask only the empty fast tokens above;
        # do not suppress the residual produced from z_F itself.
        delta_z = delta_z.reshape(batch, arms, self.config.latent_dim)
        recent_tokens = recent_tokens.reshape(
            batch, arms, recent_tokens.shape[1], recent_tokens.shape[2]
        )
        return ForceModulationOutput(delta_z, z_model + delta_z, recent_tokens)
