import numpy as np
import pytest
import flax.traverse_util
import jax

from atomic_latent_vla.pi05.weights import (
    _drop_compatible_none_mismatches,
    _merge_atomic_params,
)


def test_force_optional_bias_array_can_restore_into_none_graph_leaf():
    loaded = {
        "force_conditioner": {"future_gru_cells": {"layer_0": {"dense_h": {"bias": np.ones(3)}}}},
        "ordinary": {"kernel": np.ones(2)},
    }
    reference = {
        "force_conditioner": {"future_gru_cells": {"layer_0": {"dense_h": {"bias": None}}}},
        "ordinary": {"kernel": np.zeros(2)},
    }

    filtered = _drop_compatible_none_mismatches(loaded, reference)

    flat_filtered = flax.traverse_util.flatten_dict(filtered, sep="/")
    assert "force_conditioner/future_gru_cells/layer_0/dense_h/bias" not in flat_filtered
    assert np.array_equal(filtered["ordinary"]["kernel"], loaded["ordinary"]["kernel"])


def test_none_mismatch_outside_force_conditioner_is_rejected():
    with pytest.raises(ValueError, match="outside force_conditioner"):
        _drop_compatible_none_mismatches(
            {"ordinary": {"bias": np.ones(1)}},
            {"ordinary": {"bias": None}},
        )


def test_atomic_merge_preserves_matching_none_and_force_graph_default():
    loaded = {
        "ordinary": {"disabled_bias": None, "kernel": np.ones(2, dtype=np.float32)},
        "force_conditioner": {"optional_bias": np.ones(2, dtype=np.float32)},
    }
    reference = {
        "ordinary": {"disabled_bias": None, "kernel": np.zeros(2, dtype=np.float16)},
        "force_conditioner": {"optional_bias": None, "new_leaf": np.zeros(1)},
    }

    merged = _merge_atomic_params(loaded, reference, missing_regex=r".*force_conditioner.*")

    assert merged["ordinary"]["disabled_bias"] is None
    assert merged["ordinary"]["kernel"].dtype == np.float16
    assert merged["force_conditioner"]["optional_bias"] is None
    assert np.array_equal(merged["force_conditioner"]["new_leaf"], np.zeros(1))


def test_atomic_merge_keeps_fresh_typed_prng_key():
    reference_key = jax.random.key(7)
    merged = _merge_atomic_params(
        {"rng_state": np.asarray([0, 1], dtype=np.uint32)},
        {"rng_state": reference_key},
        missing_regex=r"a^",
    )

    assert str(merged["rng_state"].dtype).startswith("key<")
    assert np.array_equal(jax.random.key_data(merged["rng_state"]), jax.random.key_data(reference_key))
