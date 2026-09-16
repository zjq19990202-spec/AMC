from __future__ import annotations

import numpy as np

from scripts.train_plain_pi05_la4_baseline import _select_training_prompt


def _batch() -> dict[str, np.ndarray]:
    return {
        "atomic_supervision_mask": np.asarray(
            [[False, False], [True, False], [False, True], [True, True]],
            dtype=np.bool_,
        ),
        "atomic_prompt_tokens": np.asarray(
            [[10, 10], [11, 11], [12, 12], [13, 13]], dtype=np.int32
        ),
        "atomic_prompt_mask": np.ones((4, 2), dtype=np.bool_),
        "subtask_prompt_tokens": np.asarray(
            [[20, 20], [21, 21], [22, 22], [23, 23]], dtype=np.int32
        ),
        "subtask_prompt_mask": np.asarray(
            [[True, False], [True, False], [True, False], [True, False]],
            dtype=np.bool_,
        ),
    }


def test_hybrid_prompt_uses_atomic_if_either_arm_is_labeled() -> None:
    tokens, mask, use_atomic = _select_training_prompt(
        _batch(), "atomic_if_valid_else_subtask"
    )

    np.testing.assert_array_equal(use_atomic, [False, True, True, True])
    np.testing.assert_array_equal(tokens[:, 0], [20, 11, 12, 13])
    np.testing.assert_array_equal(mask[0], [True, False])
    np.testing.assert_array_equal(mask[1:], np.ones((3, 2), dtype=np.bool_))


def test_subtask_only_route_never_uses_atomic_prompt() -> None:
    tokens, mask, use_atomic = _select_training_prompt(_batch(), "subtask_only")

    np.testing.assert_array_equal(use_atomic, np.zeros(4, dtype=np.bool_))
    np.testing.assert_array_equal(tokens[:, 0], [20, 21, 22, 23])
    np.testing.assert_array_equal(
        mask, np.asarray([[True, False]] * 4, dtype=np.bool_)
    )
