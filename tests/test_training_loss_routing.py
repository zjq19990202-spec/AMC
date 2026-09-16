from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import jax.numpy as jnp


def _training_entrypoint():
    path = Path(__file__).parents[1] / "scripts" / "train_atomic_pi05.py"
    spec = importlib.util.spec_from_file_location("atomic_train_entrypoint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_zt_dual_and_zm_drop_composition_masks_are_mutually_exclusive() -> None:
    train = _training_entrypoint()
    weights = jnp.zeros((2, 2, 13), dtype=jnp.float32)
    weights = weights.at[0, 0, :2].set(jnp.asarray([0.7, 0.3]))  # strict Dual
    weights = weights.at[0, 1, :5].set(jnp.asarray([0.4, 0.2, 0.15, 0.15, 0.1]))  # Drop
    weights = weights.at[1, 0, 2].set(1.0)  # strict Single; never composition
    available = jnp.asarray([[True, True], [True, False]])
    strict = jnp.asarray([[True, False], [True, False]])
    dual = jnp.asarray([[True, False], [False, False]])
    confidence = jnp.ones((2, 2), dtype=jnp.float32)

    zt = train._zt_dual_composition_targets(
        weights, confidence, available, strict, dual
    )
    zm = train._zm_drop_composition_targets(
        weights, confidence, available, strict
    )

    assert zt.supervision_mask.tolist() == [[True, False], [False, False]]
    assert zm.supervision_mask.tolist() == [[False, True], [False, False]]
    assert not bool(jnp.any(zt.supervision_mask & zm.supervision_mask))
    assert jnp.allclose(zt.weights[0, 0, :2], jnp.asarray([0.7, 0.3]))
    assert jnp.allclose(zm.weights[0, 1].sum(), 1.0)
