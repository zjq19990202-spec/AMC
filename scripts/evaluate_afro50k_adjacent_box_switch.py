#!/usr/bin/env python3
"""Evaluate AFRO destination-box steering while preserving the fruit name."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import evaluate_fruit_target_switch_multiframe as fruit_eval
from evaluate_global_episode_chunks import _sample_global


def _destination_template_prompts(original_prompt: str, targets: list[str]):
    matches = [target for target in targets if target.lower() in original_prompt.lower()]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one destination target in {original_prompt!r}; got {matches}"
        )
    native_target = matches[0]
    prompts = {
        target: original_prompt.replace(native_target, target) for target in targets
    }
    prompts[native_target] = original_prompt
    return prompts, native_target


def _sample_final_zm(model, observation, tokens, masks, noise):
    batch = int(tokens.shape[0])
    repeated_observation = jax.tree.map(
        lambda value: jnp.repeat(value, batch, axis=0), observation
    )
    repeated_noise = jnp.repeat(noise, batch, axis=0)
    actions = _sample_global(
        model, repeated_observation, repeated_noise, tokens, masks, None
    )
    unused = jnp.zeros((batch, 1), dtype=actions.dtype)
    return actions, unused, unused


if __name__ == "__main__":
    fruit_eval._native_template_prompts = _destination_template_prompts
    fruit_eval._sample_variants_layerwise = _sample_final_zm
    fruit_eval.main()
