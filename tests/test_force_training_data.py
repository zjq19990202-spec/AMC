"""Normalization and window-contract tests for force-stage training data."""

from types import SimpleNamespace

import numpy as np
from torch.utils.data import ConcatDataset

from atomic_latent_vla.pi05.force_training_data import (
    ForceNormalization,
    _ForceAnchorView,
    _ForceProcessedDataset,
    _episode_split_indices,
    adapt_force_state_to_pi,
)


def _norm() -> ForceNormalization:
    return ForceNormalization(
        force_q01=np.zeros(6, dtype=np.float32),
        force_q99=np.full(6, 2.0, dtype=np.float32),
        state_q01=np.zeros(16, dtype=np.float32),
        state_q99=np.full(16, 2.0, dtype=np.float32),
    )


def test_force_normalization_does_not_clip_contact_outliers():
    norm = _norm()
    values = np.asarray([[0.0, 1.0, 2.0, 3.0, -1.0, 1.0]], dtype=np.float32)
    result = norm.normalize_force(values)
    assert np.allclose(result[0, :3], [-1.0, 0.0, 1.0], atol=2.0e-6)
    assert result[0, 3] > 1.0
    assert result[0, 4] < -1.0


def test_constant_force_state_coordinate_is_zero():
    norm = ForceNormalization(
        force_q01=np.zeros(6, dtype=np.float32),
        force_q99=np.ones(6, dtype=np.float32),
        state_q01=np.zeros(16, dtype=np.float32),
        state_q99=np.asarray([1.0] * 7 + [0.0] + [1.0] * 8, dtype=np.float32),
    )
    result = norm.normalize_state(np.ones((3, 16), dtype=np.float32))
    assert np.array_equal(result[:, 7], np.zeros(3, dtype=np.float32))


def test_force_state_marvin_conversion_operates_on_last_axis():
    state = np.ones((2, 4, 16), dtype=np.float32)
    converted = adapt_force_state_to_pi(state)
    assert np.array_equal(converted[..., 1], -np.ones((2, 4), dtype=np.float32))
    assert np.array_equal(converted[..., 8], np.ones((2, 4), dtype=np.float32))
    assert np.allclose(converted[..., 7], converted[..., 15])
    assert np.all((converted[..., 7] > 0.0) & (converted[..., 7] < 1.0))


def test_force_window_contract_retains_both_sensors_with_shared_parameters():
    rows = 100
    state = np.zeros((rows, 4, 16), dtype=np.float32)
    left = np.zeros((rows, 4, 6), dtype=np.float32)
    right = np.full((rows, 4, 6), 2.0, dtype=np.float32)
    view = object.__new__(_ForceAnchorView)
    view.anchors = np.asarray([29], dtype=np.int64)
    view.history_rows = 30
    view.future_rows = 50
    view.update_action_steps = 10
    view.update_offsets = (0, 10, 20, 30, 40)
    view.seed = 0
    view.base = SimpleNamespace(
        _extra_numeric={
            "observation.force.left_120hz": left,
            "observation.force.right_120hz": right,
            "observation.state_120hz": state,
        }
    )
    result = view.force_metadata(0, _norm())
    assert result["slow_force_history"].shape == (2, 120, 6)
    assert result["slow_state_history"].shape == (120, 16)
    assert result["slow_history_mask"].shape == (2, 120)
    assert result["future_force"].shape == (2, 200, 6)
    assert result["update_offset"] in (0, 10, 20, 30, 40)
    expected_acquired = min(int(result["update_offset"]) * 4, 120)
    assert np.all(result["current_history_mask"].sum(axis=1) == expected_acquired)
    # Atomic latent order is [right,left]. Both use the exact same pooled norm
    # and are carried on a structural axis, never concatenated as a 12-D input.
    assert np.allclose(result["slow_force_history"][0], 1.0)
    assert np.allclose(result["slow_force_history"][1], -1.0)


def test_force_window_at_offset_zero_has_empty_fast_history():
    rows = 100
    view = object.__new__(_ForceAnchorView)
    view.anchors = np.asarray([29], dtype=np.int64)
    view.history_rows = 30
    view.future_rows = 50
    view.update_action_steps = 10
    view.update_offsets = (0, 10, 20, 30, 40)
    # This cancels the anchor hash and deterministically selects offset zero.
    view.seed = 29 * 0x85EBCA6B
    view.base = SimpleNamespace(
        _extra_numeric={
            "observation.force.left_120hz": np.zeros((rows, 4, 6), dtype=np.float32),
            "observation.force.right_120hz": np.zeros((rows, 4, 6), dtype=np.float32),
            "observation.state_120hz": np.zeros((rows, 4, 16), dtype=np.float32),
        }
    )

    result = view.force_metadata(0, _norm())
    assert int(result["update_offset"]) == 0
    assert not np.any(result["current_history_mask"])


def test_force_window_can_skip_unused_future_force_target():
    # Only the slow/current anchor rows exist. If the disabled path attempted
    # the normal 50-row future slice, _flatten_rows would raise immediately.
    rows = 30
    view = object.__new__(_ForceAnchorView)
    view.anchors = np.asarray([29], dtype=np.int64)
    view.history_rows = 30
    view.future_rows = 50
    view.load_future_force_targets = False
    view.update_action_steps = 10
    view.update_offsets = (0,)
    view.seed = 0
    view.base = SimpleNamespace(
        _extra_numeric={
            "observation.force.left_120hz": np.zeros((rows, 4, 6), dtype=np.float32),
            "observation.force.right_120hz": np.zeros((rows, 4, 6), dtype=np.float32),
            "observation.state_120hz": np.zeros((rows, 4, 16), dtype=np.float32),
        }
    )

    result = view.force_metadata(0, _norm())
    assert result["future_force"].shape == (2, 1, 6)
    assert result["future_force_mask"].shape == (2, 1)
    assert np.array_equal(result["future_force"], np.zeros((2, 1, 6), dtype=np.float32))
    assert not np.any(result["future_force_mask"])


def test_force_window_uses_explicit_training_time_rtc_offsets():
    rows = 120
    view = object.__new__(_ForceAnchorView)
    view.anchors = np.asarray([29], dtype=np.int64)
    view.history_rows = 30
    view.future_rows = 50
    view.update_action_steps = 10
    view.update_offsets = (0, 10, 20, 30, 40)
    view.seed = 0
    view.base = SimpleNamespace(
        _extra_numeric={
            "observation.force.left_120hz": np.zeros((rows, 4, 6), dtype=np.float32),
            "observation.force.right_120hz": np.zeros((rows, 4, 6), dtype=np.float32),
            "observation.state_120hz": np.zeros((rows, 4, 16), dtype=np.float32),
        }
    )
    result = view.force_metadata(0, _norm())
    assert int(result["update_offset"]) in (0, 10, 20, 30, 40)
    expected_acquired = int(result["update_offset"]) * 4
    assert np.all(result["current_history_mask"].sum(axis=1) == expected_acquired)


def test_force_holdout_split_keeps_whole_episodes_together():
    class Raw:
        def __init__(self, episodes):
            self.anchor_episodes = np.asarray(episodes, dtype=np.int64)

        def __len__(self):
            return len(self.anchor_episodes)

    def processed(episodes):
        value = object.__new__(_ForceProcessedDataset)
        value._raw = Raw(episodes)
        return value

    dataset = ConcatDataset(
        [processed([0, 0, 1, 1, 2]), processed([3, 4, 4, 5])]
    )
    train, validation = _episode_split_indices(
        dataset,
        validation_modulus=3,
        validation_remainder=0,
    )
    assert validation == [0, 1, 5]
    assert train == [2, 3, 4, 6, 7, 8]
