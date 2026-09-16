import pytest
from pydantic import ValidationError

from atomic_latent_vla.annotation.schema import CandidateSegment, FinalSegment
from atomic_latent_vla.atomic import (
    ATOMIC_BASE_INSTRUCTIONS,
    ATOMIC_SKILL_TO_ID,
    AtomicSkill,
)


def test_thirteen_atomic_labels_keep_motion_ids_stable_and_append_stay() -> None:
    assert len(ATOMIC_SKILL_TO_ID) == 13
    assert ATOMIC_SKILL_TO_ID[AtomicSkill.MOVE_X_POS] == 0
    assert ATOMIC_SKILL_TO_ID[AtomicSkill.ROTATE_Z_NEG] == 11
    assert ATOMIC_SKILL_TO_ID[AtomicSkill.STAY] == 12


def test_each_atom_has_one_base_instruction() -> None:
    assert set(ATOMIC_BASE_INSTRUCTIONS) == set(AtomicSkill)
    assert len(set(ATOMIC_BASE_INSTRUCTIONS.values())) == 13


def test_candidate_rejects_vlm_atomic_probabilities() -> None:
    with pytest.raises(ValidationError):
        CandidateSegment(
            segment_id=0,
            start_s=0,
            end_s=1,
            low_level_instruction="Maintain contact",
            visual_evidence="Contact is occluded",
            strong_interaction=True,
            atomic_probabilities=[1.0],
        )


def test_unsupervised_final_segment_has_no_targets() -> None:
    segment = FinalSegment(
        segment_id=0,
        start_s=0,
        end_s=1,
        training_eligible=False,
        atomic_supervision_mask=False,
        atomic_probabilities=[1 / 12] * 12,
        gate_mode="interaction",
        gate_reason="strong interaction disables atomic supervision",
        strong_interaction=True,
        low_level_instruction="Regulate contact",
        visual_evidence="The contact point is occluded",
        label_source="none",
    )
    assert segment.atomic_targets == []
    assert segment.atomic_supervision_mask is False
    assert len(segment.atomic_probabilities) == 13
    assert segment.atomic_probabilities[-1] == 0.0


def test_final_segment_normalizes_gate_probabilities_independently() -> None:
    segment = FinalSegment(
        segment_id=0,
        start_s=0,
        end_s=1,
        training_eligible=False,
        atomic_supervision_mask=False,
        atomic_probabilities=[1.0] + [0.0] * 11,
        gate_probabilities=[3.0, 2.0, 1.0] + [0.0] * 9,
        gate_mode="drop",
        gate_reason="complex gate distribution",
        strong_interaction=False,
        low_level_instruction="Complex motion",
        visual_evidence="Synthetic gate distribution",
        label_source="none",
    )
    assert segment.gate_probabilities is not None
    assert segment.gate_probabilities[:3] == pytest.approx([0.5, 1 / 3, 1 / 6])
    assert segment.atomic_probabilities[0] == pytest.approx(1.0)


def test_final_segment_rejects_opposite_dual_targets() -> None:
    with pytest.raises(ValidationError, match="opposite directions"):
        FinalSegment(
            segment_id=0,
            start_s=0,
            end_s=2,
            training_eligible=True,
            atomic_supervision_mask=True,
            atomic_probabilities=[0.5, 0.5] + [0.0] * 10,
            gate_mode="dual",
            gate_reason="invalid opposite pair",
            atomic_targets=[
                {"label": 0, "name": "move_x_pos", "confidence": 0.5},
                {"label": 1, "name": "move_x_neg", "confidence": 0.5},
            ],
            strong_interaction=False,
            base_instructions=[
                "Move the TCP forward along base-frame +x.",
                "Move the TCP backward along base-frame -x.",
            ],
            low_level_instruction="Invalid contradictory motion",
            visual_evidence="Synthetic validation case",
            label_source="fk",
        )


def test_final_segment_accepts_local_ratios_only_on_its_fixed_atoms() -> None:
    segment = FinalSegment(
        segment_id=0,
        start_s=1.0,
        end_s=3.0,
        training_eligible=True,
        atomic_supervision_mask=True,
        atomic_probabilities=[0.6, 0.0, 0.4] + [0.0] * 9,
        gate_mode="dual",
        gate_reason="compatible dual",
        atomic_targets=[
            {"label": 0, "name": "move_x_pos", "confidence": 0.6},
            {"label": 2, "name": "move_y_pos", "confidence": 0.4},
        ],
        atomic_ratio_blocks=[
            {
                "start_offset_s": 0.0,
                "end_offset_s": 1 / 3,
                "weights": [0.8, 0.0, 0.2] + [0.0] * 9,
                "valid": True,
            }
        ],
        strong_interaction=False,
        base_instructions=[
            "Move the TCP forward along base-frame +x.",
            "Move the TCP left along base-frame +y.",
        ],
        low_level_instruction="Move forward and left.",
        visual_evidence="Synthetic dual movement.",
        label_source="fk",
    )
    np_weights = segment.atomic_ratio_blocks[0].weights
    assert np_weights[0] == pytest.approx(0.8)
    assert np_weights[2] == pytest.approx(0.2)

    payload = segment.model_dump()
    payload["atomic_ratio_blocks"][0]["weights"][4] = 0.1
    with pytest.raises(ValidationError, match="fixed segment atoms"):
        FinalSegment.model_validate(payload)
