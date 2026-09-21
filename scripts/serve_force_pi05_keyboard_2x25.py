#!/usr/bin/env python3
"""Serve the B2 force-conditioned Atomic PI0.5 policy with a strict 5x10 loop.

This module keeps its historical filename only so old staged source copies do
not break imports.  The production entry point is
``serve_force_pi05_keyboard_5x10.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import socket
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from serve_layerwise_pi05_keyboard import (
    ACTION_DIM,
    ACTION_HORIZON,
    CAMERAS,
    KeyboardPromptController,
    PromptBank,
    load_prompt_mapping,
    parse_prompt_assignment,
)

LOGGER = logging.getLogger(__name__)
STATE_DIM = 16
FORCE_SAMPLES = 120
FAST_SAMPLES = 40
UPDATE_STEP = 10
UPDATE_OFFSETS = (0, 10, 20, 30, 40)


class ForcePi05Policy5x10:
    """Cache slow VLM context/noise and update the fixed 0:50 axis every 10 steps."""

    def __init__(
        self,
        model: Any,
        *,
        input_transforms: Sequence[Any],
        output_transforms: Sequence[Any],
        normalize_type: type,
        force_norm: Any,
        prompt_bank: PromptBank,
        one_shot: bool,
        rng_seed: int,
        num_steps: int,
        metadata: dict[str, Any],
    ) -> None:
        import jax
        import jax.numpy as jnp
        from openpi.models import model as openpi_model
        from openpi.shared import nnx_utils

        self._jax, self._jnp = jax, jnp
        self._observation_type = openpi_model.Observation
        self._input_transforms = tuple(input_transforms)
        self._output_transforms = tuple(output_transforms)
        self._normalize_type = normalize_type
        self._force_norm = force_norm
        self._prompt_bank = prompt_bank
        self._one_shot = one_shot
        self._rng = jax.random.key(rng_seed)
        self._num_steps = num_steps
        self._model = model

        # Freeze the inference model state instead of threading the complete
        # NNX graph through every JIT call. OpenPI's module_jit exists
        # specifically to avoid the very large input/output buffers produced
        # by applying nnx.jit directly to module methods.
        self._prepare = nnx_utils.module_jit(
            self._model.prepare_force_policy_context
        )
        self._sample = nnx_utils.module_jit(
            self._model.sample_actions_force_update,
            static_argnames=("num_steps",),
        )
        self._cycles: dict[int, dict[str, Any]] = {}
        self.metadata = metadata

    def warmup(self) -> None:
        """Compile both graph phases before accepting a WebSocket connection."""

        common = {
            "state": np.zeros((STATE_DIM,), dtype=np.float32),
            # Match the production observation pytree exactly. JAX keys and
            # image-token length are static; warming only cam_high causes the
            # first real three-camera request to compile slow and fast again.
            "images": {
                name: np.zeros((3, 224, 224), dtype=np.uint8)
                for name in CAMERAS
            },
            "force_state_history_120hz": np.zeros((FORCE_SAMPLES, STATE_DIM), dtype=np.float32),
            "left_force_history_120hz": np.zeros((FORCE_SAMPLES, 6), dtype=np.float32),
            "right_force_history_120hz": np.zeros((FORCE_SAMPLES, 6), dtype=np.float32),
            "force_history_mask": np.ones((FORCE_SAMPLES,), dtype=np.bool_),
            "_force_cycle_id": 0,
            "_cr1_consume_prompt": False,
        }
        LOGGER.info("Compiling force slow_prepare warmup before opening the socket")
        self.infer({**common, "_force_request_kind": "slow_prepare", "_force_update_offset": 0})
        LOGGER.info("Compiling force fast_update warmup before opening the socket")
        self.infer({
            **common,
            "_force_request_kind": "fast_update",
            "_force_update_offset": UPDATE_STEP,
            "_force_executed_actions": np.zeros((UPDATE_STEP, ACTION_DIM), dtype=np.float32),
        })
        self._cycles.clear()
        LOGGER.info("Force 5x10 warmup complete")

    def _transform(self, raw: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
        values = self._jax.tree.map(lambda value: value, raw)
        pre_norm_state = None
        for transform in self._input_transforms:
            if isinstance(transform, self._normalize_type) and "state" in values:
                pre_norm_state = np.asarray(values["state"], dtype=np.float32).copy()
            values = transform(values)
        if pre_norm_state is None:
            raise RuntimeError("normalization transform did not observe state")
        return values, pre_norm_state

    def _decode(self, normalized_actions: np.ndarray, normalized_state: np.ndarray, pre_norm_state: np.ndarray) -> np.ndarray:
        values: dict[str, Any] = {"state": normalized_state.copy(), "actions": normalized_actions.copy()}
        for transform in self._output_transforms:
            values = transform(values)
            if transform.__class__.__name__ == "Unnormalize":
                values["state"] = pre_norm_state.copy()
        actions = np.asarray(values["actions"], dtype=np.float32)
        if actions.shape != (ACTION_HORIZON, ACTION_DIM) or not np.all(np.isfinite(actions)):
            raise ValueError(f"decoded actions must be finite (50,16), got {actions.shape}")
        return actions

    @staticmethod
    def _array(obs: dict[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray:
        value = np.asarray(obs.get(key), dtype=np.float32)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"{key} must be finite with shape {shape}, got {value.shape}")
        return value

    def _force_inputs(self, obs: dict[str, Any], *, offset: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from atomic_latent_vla.pi05.force_training_data import adapt_force_state_to_pi

        state = self._array(obs, "force_state_history_120hz", (FORCE_SAMPLES, STATE_DIM))
        right = self._array(obs, "right_force_history_120hz", (FORCE_SAMPLES, 6))
        left = self._array(obs, "left_force_history_120hz", (FORCE_SAMPLES, 6))
        mask = np.asarray(obs.get("force_history_mask"), dtype=np.bool_)
        if mask.shape != (FORCE_SAMPLES,):
            raise ValueError(f"force_history_mask must have shape (120,), got {mask.shape}")
        if not np.all(mask):
            raise ValueError(f"force request needs 120 valid samples, got {int(mask.sum())}")
        force = self._force_norm.normalize_force(np.stack([right, left], axis=0))
        state = self._force_norm.normalize_state(adapt_force_state_to_pi(state))
        current_mask = np.zeros((2, FORCE_SAMPLES), dtype=np.bool_)
        acquired_samples = min(offset * 4, FORCE_SAMPLES)
        if acquired_samples:
            current_mask[:, -acquired_samples:] = True
        return force[None], state[None], current_mask[None]

    def _validate_observation(self, obs: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        state = self._array(obs, "state", (STATE_DIM,))
        source = obs.get("images")
        if not isinstance(source, dict) or "cam_high" not in source:
            raise ValueError("images must contain cam_high")
        images: dict[str, np.ndarray] = {}
        for name, raw in source.items():
            if name not in CAMERAS:
                raise ValueError(f"unexpected camera {name}")
            image = np.asarray(raw)
            if image.shape != (3, 224, 224):
                raise ValueError(f"{name} must have shape (3,224,224), got {image.shape}")
            images[name] = image.astype(np.uint8, copy=False)
        return state, images

    def infer(
        self,
        obs: dict[str, Any],
        inference_delay: int = 0,
        execution_horizon: int = ACTION_HORIZON,
        **_: Any,
    ) -> dict[str, Any]:
        del inference_delay, execution_horizon
        kind = str(obs.get("_force_request_kind", ""))
        cycle_id = int(obs.get("_force_cycle_id", -1))
        offset = int(obs.get("_force_update_offset", -1))
        if kind not in {"slow_prepare", "fast_update"} or cycle_id < 0:
            raise ValueError("missing valid 5x10 force request metadata")
        if offset not in UPDATE_OFFSETS or kind != ("slow_prepare" if offset == 0 else "fast_update"):
            raise ValueError(f"invalid force phase {kind=} {offset=}")

        if kind == "slow_prepare":
            consume = bool(obs.get("_cr1_consume_prompt", True))
            if self._one_shot and consume and not self._prompt_bank.has_pending_prompt():
                return {"waiting_for_prompt": True, "prompt_grant_consumed": False}
            state, images = self._validate_observation(obs)
            try:
                key, prompt, revision, transition = self._prompt_bank.acquire(
                    one_shot=self._one_shot and consume,
                    consume_transition=consume,
                    wait_for_prompt=False,
                )
            except LookupError:
                return {"waiting_for_prompt": True, "prompt_grant_consumed": False}
            raw = {"state": state, "images": images, "prompt": prompt}
            values, pre_norm_state = self._transform(raw)
            inputs = self._jax.tree.map(lambda x: self._jnp.asarray(x)[None], values)
            force, force_state, current_mask = self._force_inputs(obs, offset=0)
            context = self._prepare(
                self._observation_type.from_dict(inputs),
                slow_force_history=self._jnp.asarray(force),
                slow_state_history=self._jnp.asarray(force_state),
                slow_history_mask=self._jnp.ones((1, 2, FORCE_SAMPLES), dtype=self._jnp.bool_),
            )
            self._rng, noise_rng, sample_rng = self._jax.random.split(self._rng, 3)
            noise = self._jax.random.normal(noise_rng, (1, ACTION_HORIZON, 32))
            normalized, _modulation = self._sample(
                sample_rng,
                context,
                current_force_history=self._jnp.asarray(force),
                current_state_history=self._jnp.asarray(force_state),
                current_history_mask=self._jnp.asarray(current_mask),
                update_offset=self._jnp.asarray([0], dtype=self._jnp.int32),
                num_steps=self._num_steps,
                noise=noise,
                executed_actions=None,
            )
            cache = {
                "context": context, "noise": noise, "raw": raw,
                "pre_norm_state": pre_norm_state,
                "normalized_state": np.asarray(inputs["state"][0]),
                "key": key, "prompt": prompt, "revision": revision,
            }
            self._cycles = {cycle_id: cache}
            actions = self._decode(np.asarray(normalized[0]), cache["normalized_state"], pre_norm_state)
            return {"actions": actions, "force_cycle_id": cycle_id, "force_update_offset": 0,
                    "active_prompt_key": key, "active_prompt": prompt, "prompt_revision": revision,
                    "prompt_transition_composed": transition, "prompt_grant_consumed": self._one_shot and consume}

        cache = self._cycles.get(cycle_id)
        if cache is None:
            raise ValueError(f"fast update has no cached slow cycle {cycle_id}")
        force, force_state, current_mask = self._force_inputs(obs, offset=offset)
        executed = self._array(obs, "_force_executed_actions", (offset, ACTION_DIM))
        padded = np.repeat(np.asarray(cache["raw"]["state"])[None], ACTION_HORIZON, axis=0)
        padded[:offset] = executed
        executed_values, _ = self._transform({**cache["raw"], "actions": padded})
        executed_norm = np.asarray(executed_values["actions"], dtype=np.float32)
        self._rng, sample_rng = self._jax.random.split(self._rng)
        normalized, _modulation = self._sample(
            sample_rng,
            cache["context"],
            current_force_history=self._jnp.asarray(force),
            current_state_history=self._jnp.asarray(force_state),
            current_history_mask=self._jnp.asarray(current_mask),
            update_offset=self._jnp.asarray([offset], dtype=self._jnp.int32),
            num_steps=self._num_steps,
            noise=cache["noise"],
            executed_actions=self._jnp.asarray(executed_norm[None]),
        )
        actions = self._decode(np.asarray(normalized[0]), cache["normalized_state"], cache["pre_norm_state"])
        # Preserve the actually executed prefix bit-for-bit in the native CR1 frame.
        actions[:offset] = executed
        return {"actions": actions, "force_cycle_id": cycle_id, "force_update_offset": offset,
                "active_prompt_key": cache["key"], "active_prompt": cache["prompt"],
                "prompt_revision": cache["revision"], "prompt_grant_consumed": False}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--force-norm", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--prompt", action="append", type=parse_prompt_assignment, default=[])
    parser.add_argument("--initial-key")
    parser.add_argument("--prompt-mode", choices=("one-shot", "continuous"), default="one-shot")
    parser.add_argument("--key-active-window-ms", type=float, default=250.0)
    parser.add_argument("--no-keyboard", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=12000)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model-recipe",
        choices=("legacy", "spherical-b2-final"),
        default="legacy",
        help="Construct the checkpoint-compatible force graph.",
    )
    return parser.parse_args(argv)


def build_policy(args: argparse.Namespace, prompt_bank: PromptBank) -> ForcePi05Policy5x10:
    import jax
    from flax import nnx
    from openpi import transforms
    from openpi.models import pi0_config
    from openpi.training import config as training_config
    from atomic_latent_vla.pi05 import AtomicPi05Config
    from atomic_latent_vla.pi05.force_training_data import ForceNormalization
    from atomic_latent_vla.pi05.training_data import _NormalizeWithoutQuantileClipping
    from atomic_latent_vla.pi05.weights import AtomicPi05CheckpointLoader

    config_kwargs: dict[str, Any] = dict(
        max_token_len=200, coefficient_target_kind="joint_delta", coefficient_target_dim=14,
        fast_action_ce_loss_weight=0.0, enable_force_stage=True,
        force_fast_history_samples=40, force_update_action_steps=10,
        force_future_decoder_stride=4, force_future_decoder_kind="phase_mlp",
        force_encoder_depth=2, force_position_base=10_000.0,
        force_history_train_lengths=(120,), force_future_loss_weight=0.0,
        force_flow_loss_weight=1.0, force_delta_regularization_weight=1.0e-4,
        force_stop_gradient_backbone=True,
    )
    if args.model_recipe == "spherical-b2-final":
        config_kwargs.update(
            max_token_len=192,
            force_encoder_width=512,
            force_encoder_depth=2,
            force_encoder_num_heads=8,
            force_encoder_mlp_dim=1024,
            force_latent_dim=512,
            force_context_from_prefix=False,
            force_future_condition_on_zm=True,
            force_improvement_loss_weight=1.0,
            force_improvement_margin=0.001,
            enable_layerwise_atomic_flow=False,
            force_full_token_adapter=True,
            force_full_token_adapter_heads=2,
            spherical_visual_latent=True,
            visual_max_update_angle_deg=45.0,
            spherical_force_update=True,
            force_max_update_angle_deg=45.0,
            force_rotation_loss_weight=0.005,
            force_rotation_free_angle_deg=20.0,
        )
    config = AtomicPi05Config(**config_kwargs)
    # Build the reference tree on CPU.  The checkpoint loader restores NumPy
    # arrays and needs this tree only for shapes/dtypes and missing force
    # leaves; creating it on GPU would retain a second multi-GiB parameter
    # copy when the merged inference state is placed below.
    with jax.default_device(jax.devices("cpu")[0]):
        initialized = config.create(jax.random.key(0))
    _, initialized_state = nnx.split(initialized)
    params = AtomicPi05CheckpointLoader(str(args.checkpoint / "params")).load(initialized_state.to_pure_dict())
    model = config.load(params, remove_extra_params=False)
    # AtomicPi05CheckpointLoader intentionally restores NumPy arrays. Pin the
    # final immutable inference state on the selected accelerator once so the
    # five requests in a force cycle share the same device buffers.
    model_graphdef, model_state = nnx.split(model)
    model_state = jax.device_put(model_state)
    model = nnx.merge(model_graphdef, model_state)
    model.eval()

    bridge = pi0_config.Pi0Config(pi05=True, max_token_len=config.max_token_len)
    factory = training_config.LeRobotMarvinDataConfig(
        repo_id=str(args.dataset_root), prompt_from_task=True, adapt_to_pi=True,
        assets=training_config.AssetsConfig(assets_dir=str(args.norm_assets_dir), asset_id=args.norm_asset_id),
    )
    data = factory.create(args.norm_assets_dir, bridge)
    initial_prompt = prompt_bank.snapshot()[1]
    input_transforms = [transforms.InjectDefaultPrompt(initial_prompt), *data.data_transforms.inputs,
                        _NormalizeWithoutQuantileClipping(data.norm_stats, use_quantiles=data.use_quantile_norm),
                        *data.model_transforms.inputs]
    output_transforms = [*data.model_transforms.outputs,
                         transforms.Unnormalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
                         *data.data_transforms.outputs]
    metadata = {
        "protocol": "cr1-force-atomic-pi05-5x10-v1", "force_fast_loop": "5x10",
        "force_history_samples": FORCE_SAMPLES, "force_fast_history_samples": FAST_SAMPLES,
        "force_rate_hz": 120, "force_update_offsets": list(UPDATE_OFFSETS),
        "force_strict_wait_offsets": list(UPDATE_OFFSETS[1:]), "action_horizon": 50, "action_dim": 16,
        "camera_names": list(CAMERAS), "required_camera_names": ["cam_high"],
        "state_dim": 16, "ready_gate_query_enabled": False,
        "nonblocking_prompt_wait": args.prompt_mode == "one-shot", "rtc_enabled": True,
        "fixed_action_axis": "0:50", "shared_noise_within_cycle": True,
        "norm_asset_id": args.norm_asset_id, "force_norm": str(args.force_norm),
        "model_recipe": args.model_recipe,
    }
    return ForcePi05Policy5x10(model, input_transforms=input_transforms,
        output_transforms=output_transforms, normalize_type=_NormalizeWithoutQuantileClipping,
        force_norm=ForceNormalization.load(args.force_norm), prompt_bank=prompt_bank,
        one_shot=args.prompt_mode == "one-shot", rng_seed=args.seed,
        num_steps=args.num_steps, metadata=metadata)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    prompts = load_prompt_mapping(args.prompt_file, args.prompt)
    initial = args.initial_key or next(iter(prompts))
    bank = PromptBank(prompts, initial, active_window_s=args.key_active_window_ms / 1000.0)
    print(json.dumps(bank.prompts, ensure_ascii=False, indent=2))
    policy = build_policy(args, bank)
    policy.warmup()
    from openpi.serving import websocket_policy_server
    LOGGER.info("Serving strict force 5x10 on %s:%d host=%s", args.host, args.port, socket.gethostname())
    keyboard = KeyboardPromptController(bank)
    try:
        if not args.no_keyboard:
            keyboard.start()
        websocket_policy_server.WebsocketPolicyServer(policy, args.host, args.port, metadata=policy.metadata).serve_forever()
    finally:
        keyboard.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
