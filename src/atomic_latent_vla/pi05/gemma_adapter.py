"""π0.5 Gemma execution with a latent-only Action-Expert side attention.

The implementation deliberately copies the *execution shell* of OpenPI's
``gemma.Module``/``gemma.Block``.  The pretrained attention, FFN, RMSNorm,
AdaRMS and parameter names are unchanged. The new
``layers/atomic_cross_attention`` subtree lets Action tokens read exactly two
ordered arm z_M tokens through a zero-initialized residual at the end of each
block. This side path never consumes or modifies Gemma's self-attention mask.
"""

from __future__ import annotations

from collections.abc import Sequence

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.models import gemma as _gemma
from openpi.models import lora as _lora
from openpi.training import sharding as _sharding


class TokenwiseRMSNorm(nn.Module):
    """Checkpoint-compatible Gemma RMSNorm with token-wise AdaRMS support.

    OpenPI's released RMSNorm assumes ``cond`` is ``[B,D]`` and inserts a
    singleton sequence axis. RTC needs ``[B,H,D]`` so committed clean tokens
    can use flow time zero while the suffix uses the sampled flow time. The
    parameter names and shapes remain identical to Gemma RMSNorm: regular
    normalization owns ``scale`` and adaptive normalization owns ``Dense_0``.
    """

    @nn.compact
    def __call__(self, x: jax.Array, cond: jax.Array | None):
        dtype = x.dtype
        variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normalized = jnp.asarray(x * jax.lax.rsqrt(variance + 1e-6))
        if cond is None:
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1],))
            return (normalized * (1 + scale)).astype(dtype), None
        modulation = nn.Dense(
            x.shape[-1] * 3,
            kernel_init=nn.initializers.zeros,
            dtype=dtype,
        )(cond)
        if modulation.ndim == 2:
            modulation = modulation[:, None, :]
        elif modulation.ndim != 3:
            raise ValueError(f"AdaRMS condition must be [B,D] or [B,H,D], got {cond.shape}")
        if modulation.shape[1] not in (1, x.shape[1]):
            raise ValueError(
                "token-wise AdaRMS sequence length must be one or match hidden; "
                f"got {modulation.shape[1]} and {x.shape[1]}"
            )
        scale, shift, gate = jnp.split(modulation, 3, axis=-1)
        return (normalized * (1 + scale) + shift).astype(dtype), gate


class DualArmLatentFusion(nn.Module):
    """Prepare ordered arm tokens and retain the legacy fused diagnostic."""

    latent_dim: int
    arm_mlp_hidden_dim: int
    fusion_hidden_dim: int

    @nn.compact
    def __call__(self, arm_latents: jax.Array) -> jax.Array:
        if arm_latents.ndim != 3 or arm_latents.shape[1] != 2:
            raise ValueError(f"arm_latents must have shape [B,2,D], got {arm_latents.shape}")
        # No normalization here: preserve each arm latent's learned magnitude.
        arm_hidden = nn.swish(nn.Dense(self.arm_mlp_hidden_dim, name="shared_arm_in")(arm_latents))
        arm_hidden = nn.Dense(self.latent_dim, name="shared_arm_out")(arm_hidden)
        arm_type = self.param(
            "arm_type_embedding",
            nn.initializers.normal(stddev=0.02),
            (2, self.latent_dim),
        )
        arm_hidden = arm_hidden + arm_type[None, :, :]
        # Cross-attention uses one shared 512-D affine transform while
        # LayerNorm computes statistics independently for each arm token.
        nn.LayerNorm(name="arm_token_norm")(arm_hidden)
        concatenated = arm_hidden.reshape(arm_hidden.shape[0], 2 * arm_hidden.shape[-1])
        # First requested LayerNorm: immediately before the fusion MLP.
        concatenated = nn.LayerNorm(name="arm_fusion_norm")(concatenated)
        fused = nn.swish(nn.Dense(self.fusion_hidden_dim, name="arm_fusion_in")(concatenated))
        return nn.Dense(self.latent_dim, name="arm_fusion_out")(fused)


def _linear_from_params(x: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
    """Apply an NNX Linear parameter pair inside the scanned Linen block."""

    value = jnp.einsum("...d,df->...f", x, params["kernel"])
    bias = params.get("bias")
    return value if bias is None else value + bias


def _l2_normalize(x: jax.Array, eps: float = 1.0e-8) -> jax.Array:
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


def _final_query_norm(query_hidden: jax.Array, scale: jax.Array) -> jax.Array:
    """Match Gemma's learned final RMSNorm for an intermediate query state."""

    dtype = query_hidden.dtype
    variance = jnp.mean(jnp.square(query_hidden.astype(jnp.float32)), axis=-1, keepdims=True)
    normalized = jnp.asarray(query_hidden * jax.lax.rsqrt(variance + 1.0e-6))
    return (normalized * (1.0 + scale)).astype(dtype)


def _tangent_basis(
    direction: jax.Array,
    basis_seed: jax.Array,
) -> jax.Array:
    """Exact pure-JAX counterpart of ``AtomicQueries._tangent_basis``."""

    seeds = jnp.broadcast_to(
        basis_seed[None],
        (direction.shape[0], basis_seed.shape[0], direction.shape[-1]),
    )
    projected = (
        seeds - jnp.einsum("bkd,bd->bk", seeds, direction)[..., None] * direction[:, None, :]
    )
    basis, _ = jnp.linalg.qr(jnp.swapaxes(projected, -1, -2).astype(jnp.float32), mode="reduced")
    return jnp.swapaxes(basis, -1, -2).astype(direction.dtype)


def compose_intermediate_arm_latents(
    query_hidden: jax.Array,
    query_params: dict[str, object],
    max_shift_scale: float,
) -> jax.Array:
    """Compose Q1--Q4 at one Gemma depth into ordered right/left latents.

    The function consumes the *same* NNX query-head parameters used by the
    final supervised ``z_M``.  It therefore introduces no second semantic
    head and lets Flow gradients reach every intermediate Q representation.
    """

    if query_hidden.ndim != 3 or query_hidden.shape[1] != 4:
        raise ValueError(f"layerwise query hidden must be [B,4,D], got {query_hidden.shape}")

    def arm_latent(direction_hidden: jax.Array, detail_hidden: jax.Array) -> jax.Array:
        raw_direction = _linear_from_params(direction_hidden, query_params["q1_in"])
        direction = _l2_normalize(_linear_from_params(raw_direction, query_params["direction"]))
        detail_feature = _linear_from_params(detail_hidden, query_params["q3_in"])
        detail = jnp.tanh(_linear_from_params(detail_feature, query_params["q3_detail"]))
        tangent_shift = jnp.einsum(
            "bkd,bk->bd",
            _tangent_basis(direction, query_params["basis_seed"]),
            detail,
        )
        return direction + max_shift_scale * tangent_shift

    right = arm_latent(query_hidden[:, 0], query_hidden[:, 1])
    left = arm_latent(query_hidden[:, 2], query_hidden[:, 3])
    return jnp.stack([right, left], axis=1)


def fuse_intermediate_arm_latents(
    arm_latents: jax.Array,
    fusion_params: dict[str, object],
) -> jax.Array:
    """Pure-JAX application of the checkpointed shared dual-arm fusion."""

    arm_hidden = _shared_arm_hidden(arm_latents, fusion_params)
    concatenated = arm_hidden.reshape(arm_hidden.shape[0], -1)
    # Preserve the legacy fused branch exactly: it still uses one LayerNorm
    # over the concatenated pair for checkpointed diagnostics/auxiliary heads.
    norm_params = fusion_params["arm_fusion_norm"]
    mean = jnp.mean(concatenated, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(concatenated - mean), axis=-1, keepdims=True)
    concatenated = (concatenated - mean) * jax.lax.rsqrt(variance + 1.0e-6)
    concatenated = concatenated * norm_params["scale"] + norm_params["bias"]
    fused = nn.swish(_linear_from_params(concatenated, fusion_params["arm_fusion_in"]))
    return _linear_from_params(fused, fusion_params["arm_fusion_out"])


def _shared_arm_hidden(
    arm_latents: jax.Array,
    fusion_params: dict[str, object],
) -> jax.Array:
    """Apply the checkpointed common adapter and right/left embeddings."""

    if arm_latents.ndim != 3 or arm_latents.shape[1] != 2:
        raise ValueError(f"arm_latents must have shape [B,2,D], got {arm_latents.shape}")
    arm_hidden = nn.swish(_linear_from_params(arm_latents, fusion_params["shared_arm_in"]))
    arm_hidden = _linear_from_params(arm_hidden, fusion_params["shared_arm_out"])
    return arm_hidden + fusion_params["arm_type_embedding"][None]


def prepare_intermediate_arm_tokens(
    arm_latents: jax.Array,
    fusion_params: dict[str, object],
) -> jax.Array:
    """Return independently normalized right/left 512-D attention tokens.

    Both arms retain the same shared adapter and their original identity
    embeddings. Normalization statistics are computed independently so one
    arm's magnitude cannot rescale the other. One dedicated 512-D affine pair
    is shared across both tokens; the legacy 1024-D fusion norm is not reused.
    """

    arm_hidden = _shared_arm_hidden(arm_latents, fusion_params)
    norm_params = fusion_params["arm_token_norm"]
    mean = jnp.mean(arm_hidden, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(arm_hidden - mean), axis=-1, keepdims=True)
    normalized = (arm_hidden - mean) * jax.lax.rsqrt(variance + 1.0e-6)
    return normalized * norm_params["scale"][None, None] + norm_params["bias"][None, None]


class LatentCrossAttention(nn.Module):
    """AFRO-Depth-style Action-to-z_M cross-attention over two arm tokens.

    The source axis is deliberately fixed to ``[right, left]``. Gemma's
    prefix/action mask does not enter this module, so Q/subtask cache masking
    remains solely owned by the released self-attention path. ``update_mask``
    is an output-only RTC gate and cannot expose or hide either z_M token.
    """

    width: int
    latent_dim: int
    num_heads: int
    head_dim: int

    @nn.compact
    def __call__(
        self,
        action_hidden: jax.Array,
        arm_latents: jax.Array,
        update_mask: jax.Array | None = None,
    ) -> jax.Array:
        if action_hidden.ndim != 3 or action_hidden.shape[-1] != self.width:
            raise ValueError(
                f"action hidden must have shape [B,T,{self.width}], got {action_hidden.shape}"
            )
        expected_latents = (action_hidden.shape[0], 2, self.latent_dim)
        if arm_latents.shape != expected_latents:
            raise ValueError(
                f"arm latents must have shape {expected_latents}, got {arm_latents.shape}"
            )
        if update_mask is not None and update_mask.shape != action_hidden.shape[:2]:
            raise ValueError(
                f"latent update mask must be {action_hidden.shape[:2]}, got {update_mask.shape}"
            )

        dtype = action_hidden.dtype
        inner_dim = self.num_heads * self.head_dim
        # These are already the two tokens produced by the existing shared
        # arm MLP + right/left embedding and independent per-arm normalization.
        # Do not introduce a second identity embedding or source normalization.
        source = arm_latents.astype(dtype)
        query = nn.Dense(inner_dim, dtype=dtype, name="zm_q_proj")(action_hidden)
        key = nn.Dense(inner_dim, dtype=dtype, name="zm_k_proj")(source)
        value = nn.Dense(inner_dim, dtype=dtype, name="zm_v_proj")(source)
        query = einops.rearrange(
            query, "b t (n h) -> b t n h", n=self.num_heads, h=self.head_dim
        )
        key = einops.rearrange(
            key, "b s (n h) -> b s n h", n=self.num_heads, h=self.head_dim
        )
        value = einops.rearrange(
            value, "b s (n h) -> b s n h", n=self.num_heads, h=self.head_dim
        )
        logits = jnp.einsum(
            "btnh,bsnh->bnts",
            query * (self.head_dim**-0.5),
            key,
            preferred_element_type=jnp.float32,
        )
        # Both ordered arm tokens are always valid. There is intentionally no
        # inherited prefix/query/action mask on this independent source axis.
        probabilities = jax.nn.softmax(logits, axis=-1).astype(dtype)
        attended = jnp.einsum("bnts,bsnh->btnh", probabilities, value)
        attended = einops.rearrange(attended, "b t n h -> b t (n h)")
        residual = nn.Dense(
            self.width,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            dtype=dtype,
            name="zm_residual_proj",
        )(attended).astype(dtype)
        if update_mask is not None:
            residual = residual * update_mask[..., None].astype(dtype)
        return residual


class AtomicGemmaBlock(nn.Module):
    """OpenPI Gemma block plus an Action-Expert-only latent side path."""

    configs: tuple[_gemma.Config, ...]
    latent_dim: int
    adapter_condition_hidden_dim: int
    adapter_bottleneck_dim: int
    atomic_cross_attention_num_heads: int
    atomic_cross_attention_head_dim: int
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    @nn.compact
    def __call__(
        self,
        xs,
        kv_cache,
        layer_index,
        positions,
        attn_mask,
        adarms_cond,
        latent_condition,
        layerwise_arm_latent_condition,
        latent_update_mask,
        query_start,
        query_params,
        fusion_params,
        query_final_norm_scale,
        max_shift_scale,
        deterministic: bool = True,
    ):
        xs = _sharding.activation_sharding_constraint(xs)
        drop = (
            nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda value, _: value
        )
        attn = _gemma.Attention(configs=self.configs, name="attn")

        pre_attn, gates = [], []
        for index, value in enumerate(xs):
            if value is not None:
                value, gate = TokenwiseRMSNorm(name=_gemma._name("pre_attention_norm", index))(  # noqa: SLF001
                    value, adarms_cond[index]
                )
            pre_attn.append(value)
            gates.append(gate if value is not None else None)
        pre_attn = _sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        post_attn = jax.tree.map(lambda value: drop(value, deterministic), post_attn)
        post_attn = _sharding.activation_sharding_constraint(post_attn)
        xs = [
            _gemma._gated_residual(value, update, gate)  # noqa: SLF001
            for value, update, gate in zip(xs, post_attn, gates, strict=True)
        ]
        xs = _sharding.activation_sharding_constraint(xs)

        out, gates = [], []
        for index, (value, config) in enumerate(zip(xs, self.configs, strict=True)):
            if value is not None:
                value, gate = TokenwiseRMSNorm(name=_gemma._name("pre_ffw_norm", index))(  # noqa: SLF001
                    value, adarms_cond[index]
                )
                value = _lora.FeedForward(
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_gemma._name("mlp", index),  # noqa: SLF001
                    lora_config=config.lora_configs.get("ffn"),
                )(value)
            out.append(value)
            gates.append(gate if value is not None else None)
        out = _sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda value: drop(value, deterministic), out)
        xs = [
            _gemma._gated_residual(value, update, gate)  # noqa: SLF001
            for value, update, gate in zip(xs, out, gates, strict=True)
        ]
        # Prefer a latent composed from this layer's own Q1--Q4 outputs.  The
        # same query-head and arm-fusion parameters also produce the final,
        # supervised z_M after the last layer. Prefix-only inference returns
        # both fused diagnostics and ordered arm latents alongside its KV
        # cache; suffix-only denoising consumes the arm pairs without rerunning
        # the VLM.
        current_arm_latents = latent_condition
        if layerwise_arm_latent_condition is not None:
            current_arm_latents = layerwise_arm_latent_condition[layer_index]
        current_latent = None
        if (
            xs[0] is not None
            and query_start is not None
            and query_params is not None
            and fusion_params is not None
            and query_final_norm_scale is not None
        ):
            query_hidden = jax.lax.dynamic_slice_in_dim(xs[0], query_start, 4, axis=1)
            query_hidden = _final_query_norm(query_hidden, query_final_norm_scale)
            current_arm_latents = compose_intermediate_arm_latents(
                query_hidden,
                query_params,
                max_shift_scale,
            )
        current_zm_tokens = current_arm_latents
        if current_arm_latents is not None and fusion_params is not None:
            current_zm_tokens = prepare_intermediate_arm_tokens(
                current_arm_latents, fusion_params
            )
            current_latent = fuse_intermediate_arm_latents(current_arm_latents, fusion_params)

        # Run the released attention and FFN (including both original
        # time-AdaRMS conditions) first. The independent two-token z_M
        # cross-attention then adds a zero-init side residual. It cannot alter
        # the original attention mask or FFN input.
        if xs[1] is not None and current_zm_tokens is not None:
            delta = LatentCrossAttention(
                width=self.configs[1].width,
                latent_dim=self.latent_dim,
                num_heads=self.atomic_cross_attention_num_heads,
                head_dim=self.atomic_cross_attention_head_dim,
                name="atomic_cross_attention",
            )(xs[1], current_zm_tokens, latent_update_mask)
            xs[1] = xs[1] + delta
        if current_latent is None:
            batch_size = next(value for value in xs if value is not None).shape[0]
            current_latent = jnp.zeros((batch_size, self.latent_dim), dtype=jnp.float32)
        if current_arm_latents is None:
            current_arm_latents = jnp.zeros(
                (current_latent.shape[0], 2, self.latent_dim),
                dtype=current_latent.dtype,
            )
        return _sharding.activation_sharding_constraint(xs), (
            kv_cache,
            current_latent,
            current_arm_latents,
            xs[1],
        )


class AtomicGemmaModule(nn.Module):
    """Checkpoint-compatible Gemma module with an optional z_M condition."""

    configs: Sequence[_gemma.Config]
    embed_dtype: str
    latent_dim: int
    adapter_condition_hidden_dim: int
    adapter_bottleneck_dim: int
    arm_mlp_hidden_dim: int
    arm_fusion_hidden_dim: int
    atomic_cross_attention_num_heads: int
    atomic_cross_attention_head_dim: int
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    adarms: bool = False

    def setup(self):
        assert all(config.depth == self.configs[0].depth for config in self.configs)
        self.embedder = _gemma.Embedder(
            vocab_size=_gemma.PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,
            name="embedder",
        )
        self.arm_fusion = DualArmLatentFusion(
            latent_dim=self.latent_dim,
            arm_mlp_hidden_dim=self.arm_mlp_hidden_dim,
            fusion_hidden_dim=self.arm_fusion_hidden_dim,
            name="arm_fusion",
        )
        block_cls = nn.remat(
            AtomicGemmaBlock,
            prevent_cse=False,
            static_argnums=(14,),
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(
                0,
                0,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
            ),
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            latent_dim=self.latent_dim,
            adapter_condition_hidden_dim=self.adapter_condition_hidden_dim,
            adapter_bottleneck_dim=self.adapter_bottleneck_dim,
            atomic_cross_attention_num_heads=self.atomic_cross_attention_num_heads,
            atomic_cross_attention_head_dim=self.atomic_cross_attention_head_dim,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )
        self.final_norms = [
            TokenwiseRMSNorm(name=_gemma._name("final_norm", index))
            for index in range(len(self.configs))
        ]

    def embed(self, tokens):
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    def decode(self, pre_logits):
        """Tied PaliGemma vocabulary projection used by π0.5 FAST CE.

        The native atomic wrapper originally needed only ``embed`` because
        continuous flow predicts actions through the separate Action Expert.
        FAST restores the released autoregressive action-token objective, so
        it must use the same shared input-embedding vocabulary projection as
        OpenPI's :class:`gemma.Module`.
        """

        return self.embedder.decode(pre_logits.astype(self.embed_dtype))

    def __call__(
        self,
        embedded,
        positions,
        mask,
        adarms_cond=None,
        *,
        kv_cache=None,
        latent_condition=None,
        layerwise_arm_latent_condition=None,
        latent_update_mask=None,
        query_start=None,
        query_params=None,
        fusion_params=None,
        query_final_norm_scale=None,
        max_shift_scale: float = 0.0,
        return_layerwise_latents: bool = False,
        return_layerwise_arm_latents: bool = False,
        return_layerwise_action_hidden: bool = False,
        deterministic: bool = True,
    ):
        embedded = jax.tree.map(lambda value: value.astype(self.embed_dtype), embedded)
        has_zm_condition = (
            latent_condition is not None or layerwise_arm_latent_condition is not None
        )
        if has_zm_condition and fusion_params is None and not self.is_initializing():
            raise ValueError(
                "z_M cross-attention requires the checkpointed shared-arm/fusion "
                "parameters; refusing to bypass the right/left embedding path"
            )
        if latent_condition is not None:
            if latent_condition.ndim != 3 or latent_condition.shape[1:] != (
                2,
                self.latent_dim,
            ):
                raise ValueError(
                    "latent_condition must be ordered [right,left] z_M with "
                    f"shape [B,2,{self.latent_dim}], got {latent_condition.shape}"
                )
            # Keep the original shared-arm/fusion parameter tree initialized
            # and checkpoint-compatible. The fused result is no longer used
            # for Action-Expert injection; scanned blocks reuse these exact
            # parameters to prepare the two cross-attention source tokens.
            if self.is_initializing():
                self.arm_fusion(latent_condition)
        if layerwise_arm_latent_condition is not None:
            if (
                layerwise_arm_latent_condition.ndim != 4
                or layerwise_arm_latent_condition.shape[0] != self.configs[0].depth
                or layerwise_arm_latent_condition.shape[2:] != (2, self.latent_dim)
            ):
                raise ValueError(
                    "layerwise_arm_latent_condition must have shape "
                    f"[L,B,2,{self.latent_dim}], got "
                    f"{layerwise_arm_latent_condition.shape}"
                )
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)
        embedded, scan_outputs = self.layers(
            embedded,
            kv_cache,
            jnp.arange(self.configs[0].depth, dtype=jnp.int32),
            positions,
            mask,
            adarms_cond,
            latent_condition,
            layerwise_arm_latent_condition,
            latent_update_mask,
            query_start,
            query_params,
            fusion_params,
            query_final_norm_scale,
            max_shift_scale,
            deterministic,
        )
        (
            kv_cache,
            layerwise_latents,
            layerwise_arm_latents,
            layerwise_action_hidden,
        ) = scan_outputs
        normalized = [
            norm(value, condition)[0] if value is not None else value
            for norm, value, condition in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ]
        if return_layerwise_action_hidden:
            return normalized, (
                kv_cache,
                layerwise_latents,
                layerwise_arm_latents,
                layerwise_action_hidden,
            )
        if return_layerwise_arm_latents:
            return normalized, (
                kv_cache,
                layerwise_latents,
                layerwise_arm_latents,
            )
        if return_layerwise_latents:
            return normalized, (kv_cache, layerwise_latents)
        return normalized, kv_cache

    def init(self, use_adarms: Sequence[bool]):
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, config.width)) for config in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[
                jnp.zeros((1, config.width)) if use else None
                for use, config in zip(use_adarms, self.configs, strict=True)
            ],
            latent_condition=jnp.zeros((1, 2, self.latent_dim)),
            latent_update_mask=jnp.ones((1, 1), dtype=jnp.bool_),
        )


class CoefficientDiTModule(nn.Module):
    """A standalone DiT trunk for Q1's four-token DCT coefficient prior.

    Unlike :class:`AtomicGemmaModule`, this module intentionally has no token
    embedder, visual KV cache, or second expert.  Its only condition is the
    supplied Q1 latent plus diffusion time through AdaRMS.  Reusing Gemma's
    transformer block gives the coefficient prior a substantial joint model
    without allowing raw text/image/state context to bypass ``z_T``.
    """

    config: _gemma.Config
    embed_dtype: str

    def setup(self):
        block_cls = nn.remat(
            _gemma.Block,
            prevent_cse=False,
            static_argnums=(5,),
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(
                0,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
            ),
            length=self.config.depth,
        )(
            configs=(self.config,),
        )
        self.final_norm = _gemma.RMSNorm(name="final_norm")

    def __call__(
        self,
        action_tokens: jax.Array,
        positions: jax.Array,
        condition: jax.Array,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        action_tokens = action_tokens.astype(self.embed_dtype)
        batch_size, sequence_length = action_tokens.shape[:2]
        # The four frequency coefficients are predicted jointly and are
        # bidirectionally visible, unlike the causal Q tail.
        attention_mask = jnp.ones(
            (batch_size, 1, sequence_length, sequence_length), dtype=jnp.bool_
        )
        (output,), _ = self.layers(
            [action_tokens], None, positions, attention_mask, [condition], deterministic
        )
        return self.final_norm(output, condition)[0]

    def init(self):
        self(
            jnp.zeros((1, 1, self.config.width), dtype=self.embed_dtype),
            jnp.zeros((1, 1), dtype=jnp.int32),
            jnp.zeros((1, self.config.width), dtype=self.embed_dtype),
        )
