from __future__ import annotations

import json

import numpy as np
from openpi.shared.normalize import NormStats

import atomic_latent_vla.pi05.training_data as training_data
from atomic_latent_vla.pi05.training_data import (
    _AtomicTextDataset,
    _NormalizeWithoutQuantileClipping,
    _TargetAnnotationSidecars,
    _ZTTeacherSidecar,
    _adapt_marvin_values_to_pi,
    _topk_gate_composition,
    atomic_text_collate,
    compose_arm_atomic_prompts,
    compose_horizon_subtasks,
)


def test_text_only_zt_tokenizes_full_pi05_padded_state(monkeypatch) -> None:
    captured_states: list[np.ndarray] = []

    class CapturingTokenizer:
        def tokenize(self, _prompt, state):
            captured_states.append(np.asarray(state).copy())
            return np.zeros(8, dtype=np.int32), np.ones(8, dtype=np.bool_)

    monkeypatch.setattr(
        training_data,
        "_paligemma_tokenizer",
        lambda _max_token_len: CapturingTokenizer(),
    )
    raw_state = np.arange(16, dtype=np.float32) / 10.0
    raw_state[[7, 15]] = 0.5
    raw_actions = np.broadcast_to(raw_state, (50, 16)).copy()

    class Raw:
        def metadata(self, _index):
            return {
                "raw_state": raw_state,
                "raw_actions": raw_actions,
                "atomic_prompt": "Right arm: move forward.",
                "subtask_prompt": "Move to the object.",
            }

        def __len__(self):
            return 1

    dataset = object.__new__(_AtomicTextDataset)
    dataset._raw = Raw()
    dataset._normalize = lambda values: values
    dataset._delta_actions = lambda values: values
    dataset._max_token_len = 200
    dataset._include_fast = False

    row = dataset[0]

    assert row["state"].shape == (32,)
    assert len(captured_states) == 2
    for tokenized_state in captured_states:
        assert tokenized_state.shape == (32,)
        np.testing.assert_array_equal(tokenized_state, row["state"])
        np.testing.assert_array_equal(tokenized_state[16:], np.zeros(16, dtype=np.float32))


def test_text_only_zt_uses_admin123_marvin_pi_internal_values() -> None:
    state = np.arange(16, dtype=np.float32) / 10.0
    state[[7, 15]] = np.asarray([1.0, 0.0], dtype=np.float32)
    actions = np.broadcast_to(state, (3, 16)).copy()
    actions[:, [7, 15]] = np.asarray([1.0, 0.0], dtype=np.float32)

    converted_state, converted_actions = _adapt_marvin_values_to_pi(state, actions)

    np.testing.assert_array_equal(
        np.signbit(converted_state[[1, 2, 8, 9]]), np.ones(4, dtype=np.bool_)
    )
    np.testing.assert_allclose(converted_state[7], 0.23624367, atol=1e-6)
    np.testing.assert_allclose(converted_state[15], 1.5705242, atol=1e-6)
    np.testing.assert_allclose(converted_actions[:, 7], 1.0, atol=2e-5)
    np.testing.assert_allclose(converted_actions[:, 15], -1.00001e-5, atol=2e-7)


def test_atomic_text_collate_keeps_zt_composition_targets() -> None:
    row = {
        "state": np.zeros(32, dtype=np.float32),
        "actions": np.zeros((50, 32), dtype=np.float32),
        "atomic_prompt_tokens": np.zeros(4, dtype=np.int32),
        "atomic_prompt_mask": np.ones(4, dtype=np.bool_),
        "atomic_weights": np.zeros((2, 13), dtype=np.float32),
        "atomic_supervision_mask": np.ones(2, dtype=np.bool_),
        "atomic_composition_weights": np.full((2, 13), 1 / 13, dtype=np.float32),
        "atomic_composition_confidence": np.asarray([0.7, 0.8], dtype=np.float32),
        "atomic_composition_mask": np.asarray([True, False]),
        "tcp_twist_delta": np.zeros((50, 12), dtype=np.float32),
        "joint_delta": np.zeros((50, 14), dtype=np.float32),
    }

    batch = atomic_text_collate([row, row])

    assert batch["atomic_composition_weights"].shape == (2, 2, 13)
    assert batch["atomic_composition_confidence"].shape == (2, 2)
    assert batch["atomic_composition_mask"].shape == (2, 2)


def test_pi05_quantile_outliers_are_not_clipped_for_state_or_actions() -> None:
    stats = {
        "state": NormStats(
            mean=np.zeros(1),
            std=np.ones(1),
            q01=np.asarray([-1.0]),
            q99=np.asarray([1.0]),
        ),
        "actions": NormStats(
            mean=np.zeros(1),
            std=np.ones(1),
            q01=np.asarray([-1.0]),
            q99=np.asarray([1.0]),
        ),
    }
    transformed = _NormalizeWithoutQuantileClipping(stats, use_quantiles=True)(
        {
            "state": np.asarray([2.0], dtype=np.float32),
            "actions": np.asarray([[2.0]], dtype=np.float32),
        }
    )

    np.testing.assert_allclose(transformed["state"], np.asarray([2.0]), atol=2e-6)
    np.testing.assert_allclose(transformed["actions"], np.asarray([[2.0]]), atol=2e-6)


def test_frozen_zt_teacher_sidecar_is_row_indexed_atomic_prompt_only(tmp_path) -> None:
    latent_dim = 4
    directions = np.zeros((5, 2, latent_dim), dtype=np.float16)
    directions[3, 0, 0] = 1
    directions[3, 1, 1] = 1
    valid = np.asarray([False, False, False, True, False])
    codebook = np.zeros((2, 13, latent_dim), dtype=np.float32)
    codebook[..., 0] = 1.0
    np.save(tmp_path / "directions.npy", directions)
    np.save(tmp_path / "valid.npy", valid)
    np.save(tmp_path / "codebook.npy", codebook)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"version": 1, "prompt": "atomic", "latent_dim": latent_dim}),
        encoding="utf-8",
    )

    sidecar = _ZTTeacherSidecar(tmp_path, expected_rows=5)
    value, available = sidecar.lookup(3)
    assert available
    assert value.dtype == np.float32
    np.testing.assert_array_equal(value, directions[3].astype(np.float32))


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_compose_horizon_subtasks_preserves_order_and_deduplicates() -> None:
    assert compose_horizon_subtasks([" grasp handle. "]) == "grasp handle"
    assert (
        compose_horizon_subtasks(["grasp handle", "grasp handle", "pull drawer"])
        == "grasp handle; then pull drawer"
    )


def test_compose_arm_atomic_prompts_always_names_labeled_arms() -> None:
    assert (
        compose_arm_atomic_prompts((("Right arm", "Move forward."),))
        == "Right arm: Move forward."
    )
    assert (
        compose_arm_atomic_prompts((("Left arm", "Move upward."),))
        == "Left arm: Move upward."
    )
    assert compose_arm_atomic_prompts(
        (
            ("Right arm", "Move forward."),
            ("Left arm", "Move upward."),
        )
    ) == "Right arm: Move forward. Left arm: Move upward."


def test_target_sidecars_cover_every_30hz_frame_in_each_3hz_block_and_restore_stay(
    tmp_path,
) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    _write_jsonl(
        meta / "global_episode_prompts.jsonl",
        [{"episode_index": 0, "global_prompt": "organize the drawer"}],
    )
    _write_jsonl(
        meta / "episode_subtasks.jsonl",
        [
            {
                "episode_index": 0,
                "semantic_segments": [
                    {
                        "start_frame_30hz": 0,
                        "end_frame_30hz_exclusive": 30,
                        "current_subtask": "grasp handle",
                    },
                    {
                        "start_frame_30hz": 30,
                        "end_frame_30hz_exclusive": 90,
                        "current_subtask": "pull drawer",
                    },
                ],
            }
        ],
    )
    right_fk = meta / "fk_horizon_3hz" / "right_recomputed_0p20m"
    left_fk = meta / "fk_horizon_3hz" / "left"
    right_fk.mkdir(parents=True)
    left_fk.mkdir(parents=True)
    (right_fk / "episode_000000.json").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "segments": [
                    {
                        "segment_id": 1,
                        "start_s": 1 / 3,
                        "gate_mode": "drop",
                        "gate_reason": "three-way compound horizon",
                        "atomic_probabilities": [0.5, 0.0, 0.3, 0.0, 0.2] + [0.0] * 8,
                        "atomic_composition_supervision_mask": True,
                        "atomic_composition_target": [
                            0.30 / 0.87,
                            0.20 / 0.87,
                            0.15 / 0.87,
                            0.12 / 0.87,
                            0.10 / 0.87,
                        ]
                        + [0.0] * 8,
                        "atomic_composition_confidence": 0.42,
                        "gate_probabilities": [
                            0.30,
                            0.20,
                            0.15,
                            0.12,
                            0.10,
                            0.08,
                            0.05,
                        ]
                        + [0.0] * 6,
                    },
                    {
                        "segment_id": 2,
                        "start_s": 2 / 3,
                        "gate_mode": "dual",
                        "gate_reason": "compatible dual",
                        "gate_probabilities": [0.42, 0.31, 0.13, 0.08, 0.04, 0.02]
                        + [0.0] * 7,
                    },
                    {"segment_id": 4, "start_s": 4 / 3, "gate_mode": "idle"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (left_fk / "episode_000000.json").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "segments": [
                    {"segment_id": 4, "start_s": 4 / 3, "gate_mode": "idle"}
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        meta / "atomic_horizon_prompts_3hz.jsonl",
        [
            {
                "episode_index": 0,
                "block_start_id": 2,
                "block_end_id": 3,
                "arm": "right",
                "fk_atomic_labels": ["move_x_pos", "rotate_z_neg"],
                "prompt": "move forward while rotating",
            },
            {
                "episode_index": 0,
                "block_start_id": 2,
                "block_end_id": 2,
                "arm": "left",
                "fk_atomic_labels": ["move_y_pos"],
                "prompt": "left arm prompt must not supervise right action",
            },
        ],
    )

    sidecars = _TargetAnnotationSidecars(tmp_path)
    assert sidecars.global_prompt(0, "fallback") == "organize the drawer"
    assert sidecars.subtask_prompt(0, 20, 70) == "grasp handle; then pull drawer"

    anchor_2 = sidecars.atomic_horizon(0, 20)
    assert anchor_2 is not None
    assert anchor_2.prompt == "move forward while rotating"
    np.testing.assert_array_equal(np.flatnonzero(anchor_2.weights), [0, 11])
    np.testing.assert_allclose(anchor_2.weights[[0, 11]], [0.5, 0.5])
    # Every start frame in block 2 (frames 20..29) shares its reviewed label.
    for frame in range(20, 30):
        covered = sidecars.atomic_horizon(0, frame)
        assert covered is not None
        np.testing.assert_array_equal(np.flatnonzero(covered.weights), [0, 11])
    assert sidecars.atomic_horizon(0, 30) is not None
    # Block 4 is restored from strict tcp200 FK idle rather than Qwen motion.
    stay = sidecars.atomic_horizon(0, 40)
    assert stay is not None
    assert stay.prompt == "Stay stationary."
    np.testing.assert_array_equal(np.flatnonzero(stay.weights), [12])
    assert sidecars.atomic_horizon(0, 50) is None
    composition = sidecars.atomic_composition(0, 10)
    assert composition is not None
    np.testing.assert_array_equal(np.flatnonzero(composition.weights), [0, 1, 2, 3, 4])
    np.testing.assert_allclose(
        composition.weights[:5],
        np.asarray([0.30, 0.20, 0.15, 0.12, 0.10]) / 0.87,
    )
    assert composition.confidence == 0.42
    dual_composition = sidecars.atomic_composition(0, 20)
    assert dual_composition is not None
    np.testing.assert_array_equal(
        np.flatnonzero(dual_composition.weights), [0, 1]
    )
    np.testing.assert_allclose(
        dual_composition.weights[:2],
        np.asarray([0.42, 0.31]) / 0.73,
    )
    left = sidecars.atomic_horizon(0, 20, arm="left")
    assert left is not None
    assert left.prompt == "left arm prompt must not supervise right action"
    np.testing.assert_array_equal(np.flatnonzero(left.weights), [2])


def test_topk_gate_composition_uses_full_distribution_for_confidence() -> None:
    gate = np.asarray([0.24, 0.20, 0.16, 0.13, 0.10, 0.09, 0.08] + [0.0] * 6)
    composition, confidence = _topk_gate_composition(gate)
    np.testing.assert_array_equal(np.flatnonzero(composition), [0, 1, 2, 3, 4])
    np.testing.assert_allclose(composition[:5], gate[:5] / gate[:5].sum())
    full_entropy = -np.sum(gate[gate > 0] * np.log(gate[gate > 0]))
    np.testing.assert_allclose(confidence, 1.0 - full_entropy / np.log(13))
