import numpy as np

from atomic_latent_vla.data.atomic_ratios import (
    aggregate_horizon_atomic_weights,
    local_atomic_weight_rows,
)


def test_local_ratio_blocks_expand_to_arbitrary_30hz_frames() -> None:
    first = [0.0] * 12
    second = [0.0] * 12
    first[0], first[2] = 0.8, 0.2
    second[0], second[2] = 0.25, 0.75
    segment = {
        "start_s": 10.0,
        "atomic_ratio_blocks": [
            {
                "start_offset_s": 0.0,
                "end_offset_s": 1 / 3,
                "weights": first,
                "valid": True,
            },
            {
                "start_offset_s": 1 / 3,
                "end_offset_s": 2 / 3,
                "weights": second,
                "valid": True,
            },
        ],
    }
    fallback = np.zeros(12, dtype=np.float32)
    fallback[[0, 2]] = 0.5
    rows = local_atomic_weight_rows(
        np.asarray([10.0, 10.3, 10.34, 10.6]), segment, fallback
    )
    np.testing.assert_allclose(rows[:2, [0, 2]], [[0.8, 0.2], [0.8, 0.2]])
    np.testing.assert_allclose(rows[2:, [0, 2]], [[0.25, 0.75], [0.25, 0.75]])


def test_horizon_aggregation_uses_all_varying_frame_ratios() -> None:
    atoms = np.zeros((50, 12), dtype=np.float32)
    atoms[:40, [0, 2]] = [0.75, 0.25]
    atoms[40:, [0, 2]] = [0.25, 0.75]
    supervised, weights = aggregate_horizon_atomic_weights(
        atoms,
        np.ones(50, dtype=np.int8),
        np.full(50, 7, dtype=np.int64),
    )
    assert supervised is True
    np.testing.assert_allclose(weights[[0, 2]], [0.65, 0.35], atol=1e-6)
    assert np.count_nonzero(weights) == 2


def test_horizon_aggregation_rejects_cross_segment_and_opposites() -> None:
    atoms = np.zeros((50, 12), dtype=np.float32)
    atoms[:, [0, 1]] = [0.6, 0.4]
    row_type = np.ones(50, dtype=np.int8)
    segments = np.zeros(50, dtype=np.int64)
    supervised, weights = aggregate_horizon_atomic_weights(atoms, row_type, segments)
    assert supervised is False
    assert not weights.any()

    atoms[:, 1] = 0.0
    segments[25:] = 1
    supervised, weights = aggregate_horizon_atomic_weights(atoms, row_type, segments)
    assert supervised is False
    assert not weights.any()
