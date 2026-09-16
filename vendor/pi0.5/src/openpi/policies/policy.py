from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

from openpi import transforms

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        default_prompt: str | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        rtc_max_guidance_weight: float | None = None,
        norm_stats: dict[str, Any] | None = None,
        use_quantile_norm: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transforms = tuple(transforms)
        self._output_transforms = tuple(output_transforms)
        self._input_transform = _transforms.compose(self._input_transforms)
        self._output_transform = _transforms.compose(self._output_transforms)
        sample_kwargs = sample_kwargs or {}
        self._use_correction = sample_kwargs.pop("use_correction", True)
        self._cfg_beta = float(sample_kwargs.pop("cfg_beta", 0.0))
        self._cfg_positive_suffix = sample_kwargs.pop("cfg_positive_suffix", "\nAdvantage: positive")
        self._cfg_negative_suffix = sample_kwargs.pop("cfg_negative_suffix", "\nAdvantage: negative")
        self._sample_kwargs = sample_kwargs
        self._metadata = metadata or {}
        self._default_prompt = default_prompt
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._rtc_max_guidance_weight = rtc_max_guidance_weight
        self._norm_stats = norm_stats or {}
        self._state_norm_stats = self._norm_stats.get("state")
        self._action_norm_stats = self._norm_stats.get("actions")
        self._use_quantile_norm = use_quantile_norm
        self.infer_count = 0
        self._init_sample_actions = nnx_utils.module_jit(model.sample_actions)
        self.prefix_actions = None
        self.prefix_state = None  # normalized state at time of prefix inference

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions_rtc)
            self._sample_actions_with_cutoffs = nnx_utils.module_jit(model.sample_actions_rtc_with_cutoffs)
            self._rng = jax.random.key(0) if rng is None else rng

    @override
    def infer(
        self,
        obs: dict,
        inference_delay: int = 5,
        execution_horizon: int = 25,
        clear_prefix: bool = False,
        noise_plot: bool = False,
    ) -> dict:  # type: ignore[misc]
        if self._is_pytorch_model:
            inputs = jax.tree.map(lambda x: x, obs)
            inputs, pre_norm_state = self._transform_input_preserving_state(inputs)
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)

            observation = _model.Observation.from_dict(inputs)
            start_time = time.monotonic()
            outputs = {
                "state": inputs["state"],
                "actions": self._sample_actions(self._pytorch_device, observation, **self._sample_kwargs),
            }
            model_time = time.monotonic() - start_time
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
            transformed = self._transform_output_with_exact_state(outputs, pre_norm_state)
            transformed["policy_timing"] = {
                "infer_ms": model_time * 1000,
            }
            return transformed

        # Make a copy since transformations may modify the inputs in place.
        raw_inputs = jax.tree.map(lambda x: x, obs)
        model_inputs = jax.tree.map(lambda x: x, raw_inputs)
        # Make a batch and convert to jax.Array.
        cfg_observation = None
        if self._cfg_beta > 0.0:
            uncond_inputs, cfg_inputs = self._build_cfg_inputs(raw_inputs)
            if uncond_inputs is not None and cfg_inputs is not None:
                model_inputs = uncond_inputs
                cfg_inputs = self._input_transform(cfg_inputs)
                cfg_inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], cfg_inputs)
                cfg_observation = _model.Observation.from_dict(cfg_inputs)
        inputs, pre_norm_state = self._transform_input_preserving_state(model_inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)
        prev_prefix_actions = self.prefix_actions
        sync_no_rtc = inference_delay == 0
        use_init_sampling = self.infer_count < 1 or prev_prefix_actions is None or sync_no_rtc
        start_time = time.monotonic()
        if use_init_sampling:
            outputs = {
                "state": inputs["state"],
                "actions": self._init_sample_actions(
                    sample_rng,
                    _model.Observation.from_dict(inputs),
                    cfg_observation=cfg_observation,
                    cfg_beta=self._cfg_beta,
                    **self._sample_kwargs,
                ),
            }
            self.infer_count += 1
        else:
            # state_delta: how much the robot moved since the prefix was inferred, expressed in
            # the action-normalized space so it can be compared directly with prefix actions.
            state_delta = None
            if self.prefix_state is not None:
                current_state_abs = self._unnormalize_state(inputs["state"])
                prefix_state_abs = self._unnormalize_state(self.prefix_state)
                if current_state_abs is not None and prefix_state_abs is not None:
                    state_delta = self._normalize_action_delta(current_state_abs - prefix_state_abs)
            outputs = {
                "state": inputs["state"],
                "actions": self._sample_actions(
                    sample_rng,
                    _model.Observation.from_dict(inputs),
                    prev_prefix_actions,
                    inference_delay,
                    execution_horizon,
                    rtc_max_guidance_weight=self._rtc_max_guidance_weight,
                    use_correction=self._use_correction,
                    state_delta=state_delta,
                    cfg_observation=cfg_observation,
                    cfg_beta=self._cfg_beta,
                    **self._sample_kwargs,
                ),
            }
            if noise_plot:
                outputs["noise_cutoff_actions"] = self._sample_actions_with_cutoffs(
                    sample_rng,
                    _model.Observation.from_dict(inputs),
                    prev_prefix_actions,
                    inference_delay,
                    execution_horizon,
                    rtc_max_guidance_weight=self._rtc_max_guidance_weight,
                    use_correction=self._use_correction,
                    state_delta=state_delta,
                    cfg_observation=cfg_observation,
                    cfg_beta=self._cfg_beta,
                    **self._sample_kwargs,
                )
        if clear_prefix:
            # Warmup with clear: discard result so next call starts with no prefix (init sampling).
            self.prefix_actions = None
            self.prefix_state = None
        else:
            self.prefix_actions = outputs["actions"]
            action_dim = self.prefix_actions.shape[-1]
            # Roll prefix by execution_horizon so the carried prefix lines up with the
            # next chunk handoff used by the client runtime.
            self.prefix_actions = jnp.concatenate([
                self.prefix_actions[:, execution_horizon:],
                jnp.zeros((1, execution_horizon, action_dim)),
            ], axis=1)
            self.prefix_state = inputs["state"]
        model_time = time.monotonic() - start_time

        noise_cutoff_outputs = outputs.pop("noise_cutoff_actions", None)
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        transformed = self._transform_output_with_exact_state(outputs, pre_norm_state)
        if noise_cutoff_outputs is not None:
            noise_cutoff_outputs = {
                key: np.asarray(value[0, ...])
                for key, value in noise_cutoff_outputs.items()
            }
            transformed["noise_cutoff_actions"] = {
                key: self._transform_output_with_exact_state(
                    {"state": outputs["state"], "actions": value}, pre_norm_state
                )["actions"]
                for key, value in noise_cutoff_outputs.items()
            }
        transformed["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return transformed

    def _transform_input_preserving_state(self, data: dict) -> tuple[dict, np.ndarray | None]:
        """Apply input transforms and retain state immediately before normalization.

        Quantile normalization clips state to ``[-1, 1]``.  Inverting that
        clipped value during action decoding can therefore change the current
        robot configuration before a predicted delta is added to it.
        """
        pre_norm_state = None
        for transform in self._input_transforms:
            if isinstance(transform, _transforms.Normalize) and "state" in data:
                pre_norm_state = np.asarray(data["state"]).copy()
            data = transform(data)
        return data, pre_norm_state

    def _transform_output_with_exact_state(
        self,
        data: dict,
        pre_norm_state: np.ndarray | None,
    ) -> dict:
        """Decode deltas while adding them to the exact pre-normalization state."""
        for transform in self._output_transforms:
            data = transform(data)
            if pre_norm_state is not None and isinstance(transform, _transforms.Unnormalize):
                data["state"] = np.asarray(pre_norm_state).copy()
        return data

    def _build_cfg_inputs(self, obs: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        prompt = obs.get("prompt", self._default_prompt)
        if prompt is None:
            return None, None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        uncond_inputs = jax.tree.map(lambda x: x, obs)
        cfg_inputs = jax.tree.map(lambda x: x, obs)
        uncond_inputs["prompt"] = f"{prompt}{self._cfg_negative_suffix}"
        cfg_inputs["prompt"] = f"{prompt}{self._cfg_positive_suffix}"
        return uncond_inputs, cfg_inputs

    def _unnormalize_state(self, state: jax.Array | np.ndarray) -> np.ndarray | None:
        if self._state_norm_stats is None:
            return None

        state_np = np.asarray(state)
        if self._use_quantile_norm:
            q01 = np.asarray(self._state_norm_stats.q01)
            q99 = np.asarray(self._state_norm_stats.q99)
            if q01.size == 0 or q99.size == 0:
                return None
            q01 = transforms.pad_to_dim(q01, state_np.shape[-1], axis=-1, value=0.0)
            q99 = transforms.pad_to_dim(q99, state_np.shape[-1], axis=-1, value=1.0)
            return (state_np + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

        mean = transforms.pad_to_dim(np.asarray(self._state_norm_stats.mean), state_np.shape[-1], axis=-1, value=0.0)
        std = transforms.pad_to_dim(np.asarray(self._state_norm_stats.std), state_np.shape[-1], axis=-1, value=1.0)
        return state_np * (std + 1e-6) + mean

    def _normalize_action_delta(self, delta: np.ndarray) -> np.ndarray | None:
        if self._action_norm_stats is None:
            return None

        delta_np = np.asarray(delta)
        if self._use_quantile_norm:
            q01 = np.asarray(self._action_norm_stats.q01)
            q99 = np.asarray(self._action_norm_stats.q99)
            if q01.size == 0 or q99.size == 0:
                return None
            scale = transforms.pad_to_dim(q99 - q01, delta_np.shape[-1], axis=-1, value=1.0)
            return 2.0 * delta_np / (scale + 1e-6)

        std = transforms.pad_to_dim(np.asarray(self._action_norm_stats.std), delta_np.shape[-1], axis=-1, value=1.0)
        return delta_np / (std + 1e-6)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict, *args, **kwargs) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs, *args, **kwargs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
