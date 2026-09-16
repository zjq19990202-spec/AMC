import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
from typing import Callable

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def _concat_observations(
    observation: _model.Observation,
    cfg_observation: _model.Observation,
) -> _model.Observation:
    if observation.images.keys() != cfg_observation.images.keys():
        raise ValueError("CFG observation images must use the same keys as the base observation.")
    if observation.image_masks.keys() != cfg_observation.image_masks.keys():
        raise ValueError("CFG observation image masks must use the same keys as the base observation.")
    if observation.tokenized_prompt is None or cfg_observation.tokenized_prompt is None:
        raise ValueError("CFG requires tokenized prompts for both conditional and unconditional observations.")
    if observation.tokenized_prompt_mask is None or cfg_observation.tokenized_prompt_mask is None:
        raise ValueError("CFG requires tokenized prompt masks for both conditional and unconditional observations.")

    return _model.Observation(
        images={
            key: jnp.concatenate([observation.images[key], cfg_observation.images[key]], axis=0)
            for key in observation.images
        },
        image_masks={
            key: jnp.concatenate([observation.image_masks[key], cfg_observation.image_masks[key]], axis=0)
            for key in observation.image_masks
        },
        state=jnp.concatenate([observation.state, cfg_observation.state], axis=0),
        tokenized_prompt=jnp.concatenate([observation.tokenized_prompt, cfg_observation.tokenized_prompt], axis=0),
        tokenized_prompt_mask=jnp.concatenate(
            [observation.tokenized_prompt_mask, cfg_observation.tokenized_prompt_mask], axis=0
        ),
        token_ar_mask=None,
        token_loss_mask=None,
    )


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        cfg_observation: _model.Observation | None = None,
        cfg_beta: float = 0.0,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        if cfg_observation is not None:
            cfg_observation = _model.preprocess_observation(None, cfg_observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        cfg_enabled = cfg_observation is not None
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_observation = _concat_observations(observation, cfg_observation) if cfg_enabled else observation
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(prefix_observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            x_t_input = jnp.concatenate([x_t, x_t], axis=0) if cfg_enabled else x_t
            time_input = (
                jnp.concatenate([jnp.broadcast_to(time, batch_size), jnp.broadcast_to(time, batch_size)], axis=0)
                if cfg_enabled
                else jnp.broadcast_to(time, batch_size)
            )
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                prefix_observation, x_t_input, time_input
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            expected_batch_size = suffix_tokens.shape[0]
            assert full_attn_mask.shape == (
                expected_batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            if cfg_enabled:
                v_uncond, v_cond = jnp.split(v_t, 2, axis=0)
                v_t = v_uncond + cfg_beta * (v_cond - v_uncond)

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def sample_actions_rtc(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        prefix_actions: jax.Array,
        inference_delay: int,
        execution_horizon: int,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        use_correction: bool = False,
        use_subtraction: bool = True,
        use_mask: bool = True,
        rtc_max_guidance_weight: float | None = None,
        stop_time: float = 0.0,
        noise: jax.Array | None = None,
        state_delta: jax.Array | None = None,
        cfg_observation: _model.Observation | None = None,
        cfg_beta: float = 0.0,
    ) -> _model.Actions:

        # Align RTC behavior with the LeRobot implementation:
        # - use execution_horizon directly as the prefix-attention end index
        # - use exp prefix schedule by default
        # - use a larger default guidance cap
        prefix_attention_end = execution_horizon
        prefix_attention_schedule = "linear"
        max_guidance_weight = 10.0 if rtc_max_guidance_weight is None else rtc_max_guidance_weight

        observation = _model.preprocess_observation(None, observation, train=False)
        if cfg_observation is not None:
            cfg_observation = _model.preprocess_observation(None, cfg_observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        cfg_enabled = cfg_observation is not None
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_observation = _concat_observations(observation, cfg_observation) if cfg_enabled else observation
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(prefix_observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        # Shift prefix target for joint dims to align delta actions across inferences.
        # prefix_actions are deltas relative to the old state; subtract state_delta to make them
        # relative to the current state, so the RTC error compares apples-to-apples.
        # Dims 0-6: left arm, 8-14: right arm are delta; 7, 15: grippers are absolute.
        if state_delta is not None:
            action_dim = prefix_actions.shape[-1]
            state_delta_aligned = state_delta[:, :action_dim]
            delta_indices = jnp.array(list(range(7)) + list(range(8, 15)))
            delta_mask = jnp.zeros(action_dim, dtype=prefix_actions.dtype).at[delta_indices].set(1.0)
            prefix_actions = prefix_actions - state_delta_aligned[:, None, :] * delta_mask[None, None, :]

        def get_prefix_weights(start: int, end: int, total: int, schedule: str) -> jax.Array:
            """With start=2, end=6, total=10, the output will be:
            1  1  4/5 3/5 2/5 1/5 0  0  0  0
                ^              ^
                start           end
            `start` (inclusive) is where the chunk starts being allowed to change. `end` (exclusive) is where the chunk stops
            paying attention to the prefix. if start == 0, then the entire chunk is allowed to change. if end == total, then the
            entire prefix is attended to.

            `end` takes precedence over `start` in the sense that, if `end < start`, then `start` is pushed down to `end`. Thus,
            if `end` is 0, then the entire prefix will always be ignored.
            """
            start = jnp.minimum(start, end)
            if schedule == "ones":
                w = jnp.ones(total)
            elif schedule == "zeros":
                w = (jnp.arange(total) < start).astype(jnp.float32)
            elif schedule == "linear" or schedule == "exp":
                w = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
                if schedule == "exp":
                    w = w * jnp.expm1(w) / (jnp.e - 1)
            else:
                raise ValueError(f"Invalid schedule: {schedule}")
            return jnp.where(jnp.arange(total) >= end, 0, w)

        def pinv_corrected_velocity(
            v_t_fn: Callable, # ([ah ad], float) -> [ah ad]
            x_t: jax.Array, # [b ah ad]
            t: float,
            prefix_actions: jax.Array, # [b ah ad]
            inference_delay: int,
            prefix_attention_end: int,
            max_guidance_weight: float,
        ) -> jax.Array: # [b ah ad]
            @jax.vmap
            def _pinv_corrected_velocity(
                x_t: jax.Array, # [ah ad]
                y: jax.Array, # [ah ad]
            ) -> jax.Array: # [ah ad]
                def denoiser(z_t: jax.Array) -> tuple[jax.Array, jax.Array]:
                    z_v_t = v_t_fn(z_t, t)
                    return z_t - z_v_t * t, z_v_t

                x_0, v_t = denoiser(x_t)
                weights = get_prefix_weights(
                    inference_delay,
                    prefix_attention_end,
                    prefix_actions.shape[1],
                    prefix_attention_schedule,
                )
                mask_flag = jnp.asarray(use_mask, dtype=weights.dtype)
                effective_weights = mask_flag * weights + (1.0 - mask_flag) * jnp.ones_like(weights)
                error = (y - x_0) * effective_weights[:, None]

                def weighted_error(z_t: jax.Array) -> jax.Array:
                    z_x0, _ = denoiser(z_t)
                    return (y - z_x0) * effective_weights[:, None]

                pinv_correction = jax.lax.cond(
                    use_correction,
                    # Apply the Jacobian of the weighted residual itself:
                    # d[(y - (x - t * v(x, t))) * w] / dx  acting on `error`.
                    lambda _: jax.jvp(weighted_error, (x_t,), (error,))[1],
                    lambda _: error,
                    operand=None,
                )
                # Guidance weight adapted for pi0 time convention (time: 1->0).
                # Convert to kinetix convention with tau = 1 - time (tau: 0->1).
                tau = 1.0 - t
                sq_one_minus_tau = (1.0 - tau) ** 2  # = time^2
                inv_r2 = (sq_one_minus_tau + tau**2) / sq_one_minus_tau
                c = jnp.nan_to_num((1.0 - tau) / tau, posinf=max_guidance_weight)  # = time / (1 - time)
                guidance_weight = jnp.minimum(c * inv_r2, max_guidance_weight)
                subtraction_flag = jnp.asarray(use_subtraction, dtype=v_t.dtype)
                sign = 1.0 - 2.0 * subtraction_flag  # subtraction=True -> -1, False -> +1
                return v_t + sign * guidance_weight * pinv_correction

            return _pinv_corrected_velocity(x_t, prefix_actions)

        def v_t_step(
                x_t: jax.Array, # [ah ad]
                time: jax.Array, # []
                ):
            # TODO: find better way to support jax.vmap
            x_t = x_t[None, ...]
            time = time[None, ...]
            if cfg_enabled:
                x_t = jnp.concatenate([x_t, x_t], axis=0)
                time = jnp.concatenate([time, time], axis=0)

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                prefix_observation, x_t, time
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                suffix_tokens.shape[0],
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            if cfg_enabled:
                v_uncond, v_cond = jnp.split(v_t, 2, axis=0)
                v_t = v_uncond + cfg_beta * (v_cond - v_uncond)

            return v_t[0, ...] # TODO: remove this since it's not super vectorized

        def rtc_step(carry):
            x_t, time = carry
            guided_vt = pinv_corrected_velocity(
                v_t_step,
                x_t,
                time,
                prefix_actions,
                inference_delay,
                prefix_attention_end,
                max_guidance_weight,
            )
            return x_t + dt * guided_vt, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= stop_time - dt / 2

        x_0, _ = jax.lax.while_loop(cond, rtc_step, (noise, 1.0))
        return x_0

    def sample_actions_rtc_with_cutoffs(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        prefix_actions: jax.Array,
        inference_delay: int,
        execution_horizon: int,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        use_correction: bool = True,
        use_subtraction: bool = True,
        use_mask: bool = True,
        rtc_max_guidance_weight: float | None = None,
        state_delta: jax.Array | None = None,
        cfg_observation: _model.Observation | None = None,
        cfg_beta: float = 0.0,
    ) -> dict[str, _model.Actions]:
        batch_size = observation.state.shape[0]
        shared_noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        outputs: dict[str, _model.Actions] = {
            "t0.0": self.sample_actions_rtc(
                rng,
                observation,
                prefix_actions,
                inference_delay,
                execution_horizon,
                num_steps=num_steps,
                use_correction=use_correction,
                use_subtraction=use_subtraction,
                use_mask=use_mask,
                rtc_max_guidance_weight=rtc_max_guidance_weight,
                stop_time=0.0,
                noise=shared_noise,
                state_delta=state_delta,
                cfg_observation=cfg_observation,
                cfg_beta=cfg_beta,
            )
        }
        # outputs["t0.7"] = self.sample_actions_rtc(
        #     rng, observation, prefix_actions, inference_delay, execution_horizon,
        #     num_steps=num_steps, use_correction=use_correction, use_subtraction=use_subtraction,
        #     use_mask=use_mask, rtc_max_guidance_weight=rtc_max_guidance_weight, stop_time=0.7, noise=shared_noise,
        # )
        # outputs["t0.3"] = self.sample_actions_rtc(
        #     rng, observation, prefix_actions, inference_delay, execution_horizon,
        #     num_steps=num_steps, use_correction=use_correction, use_subtraction=use_subtraction,
        #     use_mask=use_mask, rtc_max_guidance_weight=rtc_max_guidance_weight, stop_time=0.3, noise=shared_noise,
        # )
        return outputs
