#!/usr/bin/env python3
"""Run the multiframe fruit evaluator with final-zM FiLM disabled."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from openpi.models import model as _model

import evaluate_fruit_target_switch_multiframe as evaluator


@nnx.jit
def _sample_variants_empty_zm(model, observation, tokens, masks, noise):
    observation = _model.preprocess_observation(None, observation, train=False)
    batch = tokens.shape[0]
    observation = jax.tree.map(lambda value: jnp.repeat(value, batch, axis=0), observation)
    observation = model._with_prompt(observation, tokens, masks)  # noqa: SLF001
    query_hidden, prefix_mask, kv_cache, _ = model._prefix_forward(observation)  # noqa: SLF001
    active_state = model._controlled_state(observation.state)  # noqa: SLF001
    _, direction, z_model, _, _ = model._latent(query_hidden, active_state)  # noqa: SLF001
    actions = model._mask_action_condition(jnp.repeat(noise, batch, axis=0))  # noqa: SLF001

    def step(index, current):
        time = jnp.asarray(1.0 - index / 10.0, current.dtype)
        velocity = model._suffix_velocity(  # noqa: SLF001
            prefix_mask,
            kv_cache,
            current,
            jnp.broadcast_to(time, (batch,)),
            None,
        )
        return model._mask_action_condition(current - 0.1 * velocity)  # noqa: SLF001

    return jax.lax.fori_loop(0, 10, step, actions)[..., :16], direction, z_model


if __name__ == "__main__":
    evaluator._sample_variants_layerwise = _sample_variants_empty_zm
    evaluator.main()
