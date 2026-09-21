"""Checkpoint loader for adding AtomicPi05 modules to a π0.5 checkpoint."""

from __future__ import annotations

import dataclasses
import logging
import pickle
import re

import flax.traverse_util
import numpy as np

from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import download
from openpi.training import weight_loaders


logger = logging.getLogger(__name__)


def _merge_atomic_params(loaded: at.Params, reference: at.Params, *, missing_regex: str) -> at.Params:
    """Merge Atomic params while preserving NNX optional ``None`` leaves.

    Some Flax NNX releases serialize an optional GRU dense bias as an array,
    while another release represents the same disabled bias as ``None``.  The
    ordinary OpenPI merge assumes every intersecting leaf exposes ``dtype``;
    it therefore also crashes when both checkpoint and graph contain ``None``.
    Preserve matching ``None`` leaves and let only force-conditioner one-sided
    array/None mismatches retain the current graph default.
    """

    flat_loaded = flax.traverse_util.flatten_dict(loaded, sep="/")
    flat_reference = flax.traverse_util.flatten_dict(reference, sep="/")
    result = {}
    for path, value in flat_loaded.items():
        if path not in flat_reference:
            continue
        target = flat_reference[path]
        if value is None and target is None:
            result[path] = None
            continue
        if value is None or target is None:
            if "force_conditioner/" not in path:
                raise ValueError(
                    "unexpected array/None checkpoint mismatch outside force_conditioner: "
                    f"{path}"
                )
            logger.warning(
                "using current graph default for compatible optional force leaf %s "
                "(checkpoint=%s graph=%s)",
                path,
                "None" if value is None else "array",
                "None" if target is None else "array",
            )
            result[path] = target
            continue
        # Orbax restores a typed JAX PRNG key to its uint32 key data when the
        # requested restore type is NumPy.  NumPy cannot cast back to
        # ``key<fry>``. RNG state is not a learned tensor, so retain the fresh
        # graph key rather than corrupting or rejecting the learned weights.
        if str(target.dtype).startswith("key<"):
            logger.warning(
                "using fresh current graph PRNG key for %s (checkpoint dtype=%s graph dtype=%s)",
                path,
                value.dtype,
                target.dtype,
            )
            result[path] = target
            continue
        result[path] = value.astype(target.dtype) if value.dtype != target.dtype else value

    pattern = re.compile(missing_regex)
    for path in {key for key in flat_reference if pattern.fullmatch(key)}:
        if path not in result:
            result[path] = flat_reference[path]
    return flax.traverse_util.unflatten_dict(result, sep="/")


def _drop_compatible_none_mismatches(loaded: at.Params, reference: at.Params) -> at.Params:
    """Backward-compatible test helper returning only compatible checkpoint leaves."""

    flat_loaded = flax.traverse_util.flatten_dict(loaded, sep="/")
    flat_reference = flax.traverse_util.flatten_dict(reference, sep="/")
    for path in list(flat_loaded):
        if path not in flat_reference:
            continue
        value = flat_loaded[path]
        target = flat_reference[path]
        if value is None and target is None:
            continue
        if value is None or target is None:
            if "force_conditioner/" not in path:
                raise ValueError(
                    "unexpected array/None checkpoint mismatch outside force_conditioner: "
                    f"{path}"
                )
            del flat_loaded[path]
    return flax.traverse_util.unflatten_dict(flat_loaded, sep="/")


@dataclasses.dataclass(frozen=True)
class AtomicPi05CheckpointLoader(weight_loaders.WeightLoader):
    """Load matching π0.5 tensors and retain initialized atomic-only tensors.

    OpenPI's ordinary ``CheckpointWeightLoader`` only allows absent LoRA
    tensors.  This new policy has deliberately absent modules in a released
    π0.5 checkpoint: Q1--Q4, codebook, latent-control projection and deep
    Action-Expert adapters.  They remain at their prescribed initialization,
    while every matching PaliGemma/Action Expert tensor is restored.
    """

    params_path: str
    trainable_params_path: str | None = None

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        merged = _merge_atomic_params(
            loaded,
            params,
            missing_regex=(
                r".*(planner|queries|codebook|atomic_adapter|"
                r"force_cross_|"
                r"arm_fusion|coefficient_|"
                r"force_conditioner).*"
            ),
        )
        if self.trainable_params_path is not None:
            checkpoint_path = download.maybe_download(self.trainable_params_path)
            with checkpoint_path.open("rb") as handle:
                trainable_params = pickle.load(handle)
            merged = weight_loaders._overwrite_params(merged, trainable_params)  # noqa: SLF001
        return merged


@dataclasses.dataclass(frozen=True)
class AtomicPi05ForceOverlayCheckpointLoader(weight_loaders.WeightLoader):
    """Restore a clean base tree, then overlay only ``force_conditioner`` tensors."""

    base_params_path: str
    force_params_path: str

    def load(self, params: at.Params) -> at.Params:
        base = AtomicPi05CheckpointLoader(self.base_params_path).load(params)
        overlay = _model.restore_params(
            download.maybe_download(self.force_params_path), restore_type=np.ndarray
        )
        flat_base = flax.traverse_util.flatten_dict(base, sep="/")
        flat_overlay = flax.traverse_util.flatten_dict(overlay, sep="/")
        replaced = 0
        for path, value in flat_overlay.items():
            if "force_conditioner/" not in path or path not in flat_base:
                continue
            target = flat_base[path]
            if value is None or target is None:
                if value is None and target is None:
                    flat_base[path] = None
                continue
            if value.shape != target.shape:
                raise ValueError(f"force overlay shape mismatch at {path}: {value.shape} != {target.shape}")
            flat_base[path] = value.astype(target.dtype) if value.dtype != target.dtype else value
            replaced += 1
        if not replaced:
            raise ValueError("force overlay checkpoint contained no matching force_conditioner tensors")
        logger.info("overlaid %d force_conditioner leaves from %s", replaced, self.force_params_path)
        return flax.traverse_util.unflatten_dict(flat_base, sep="/")
