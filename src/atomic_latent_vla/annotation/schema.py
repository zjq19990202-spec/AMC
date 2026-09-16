from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atomic_latent_vla.atomic import (
    MOTION_ATOMIC_COUNT,
    NUM_ATOMIC_SKILLS,
    OPPOSITE_ATOMIC_ID,
    STAY_ATOMIC_ID,
    AtomicSkill,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CandidateSegment(StrictModel):
    segment_id: int = Field(ge=0)
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    low_level_instruction: str = Field(min_length=3)
    visual_evidence: str = Field(min_length=3)
    strong_interaction: bool = False

    @model_validator(mode="after")
    def validate_interval(self) -> "CandidateSegment":
        if self.end_s <= self.start_s:
            raise ValueError("segment end_s must be greater than start_s")
        return self


class CandidateAnnotation(StrictModel):
    global_description: str = Field(min_length=10)
    segments: list[CandidateSegment] = Field(min_length=1)


class AtomicTargetOutput(StrictModel):
    label: int = Field(ge=0, le=NUM_ATOMIC_SKILLS - 1)
    name: AtomicSkill
    confidence: float = Field(gt=0, le=1)


class AtomicRatioBlockOutput(StrictModel):
    """Local one/dual-atom ratio on one 1/3-second FK interval."""

    start_offset_s: float = Field(ge=0)
    end_offset_s: float = Field(gt=0)
    weights: list[float] = Field(
        min_length=MOTION_ATOMIC_COUNT, max_length=NUM_ATOMIC_SKILLS
    )
    valid: bool

    @model_validator(mode="after")
    def validate_ratio(self) -> "AtomicRatioBlockOutput":
        if self.end_offset_s <= self.start_offset_s:
            raise ValueError("atomic ratio block end must exceed start")
        weights = [float(value) for value in self.weights]
        if len(weights) == MOTION_ATOMIC_COUNT:
            weights.append(0.0)
        if any(value < 0 for value in weights):
            raise ValueError("atomic ratio block weights must be non-negative")
        total = sum(weights)
        if self.valid and total <= 0:
            raise ValueError("valid atomic ratio block requires positive weights")
        if not self.valid and total != 0:
            raise ValueError("invalid atomic ratio block must have zero weights")
        if self.valid:
            self.weights = [value / total for value in weights]
        else:
            self.weights = weights
        return self


class FinalSegment(StrictModel):
    segment_id: int = Field(ge=0)
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    training_eligible: bool
    atomic_supervision_mask: bool
    atomic_probabilities: list[float] = Field(
        min_length=MOTION_ATOMIC_COUNT, max_length=NUM_ATOMIC_SKILLS
    )
    gate_probabilities: list[float] | None = Field(
        default=None, min_length=MOTION_ATOMIC_COUNT, max_length=NUM_ATOMIC_SKILLS
    )
    gate_mode: Literal["single", "dual", "drop", "interaction"]
    gate_reason: str = Field(min_length=3)
    atomic_targets: list[AtomicTargetOutput] = Field(default_factory=list, max_length=2)
    atomic_ratio_blocks: list[AtomicRatioBlockOutput] = Field(default_factory=list)
    strong_interaction: bool
    base_instructions: list[str] = Field(default_factory=list, max_length=2)
    low_level_instruction: str
    visual_evidence: str
    label_source: str = Field(pattern="^(fk|none)$")

    @model_validator(mode="after")
    def validate_training_fields(self) -> "FinalSegment":
        if self.end_s <= self.start_s:
            raise ValueError("segment end_s must be greater than start_s")
        if len(self.base_instructions) != len(self.atomic_targets):
            raise ValueError("base_instructions must match atomic_targets")
        if len({target.label for target in self.atomic_targets}) != len(
            self.atomic_targets
        ):
            raise ValueError("atomic_targets must not contain duplicate labels")
        if (
            len(self.atomic_targets) == 2
            and (
                OPPOSITE_ATOMIC_ID.get(self.atomic_targets[0].label)
                == self.atomic_targets[1].label
                or STAY_ATOMIC_ID
                in (self.atomic_targets[0].label, self.atomic_targets[1].label)
            )
        ):
            raise ValueError(
                "dual atomic_targets must not contain opposite directions or stay"
            )
        for target in self.atomic_targets:
            if target.name.value != list(AtomicSkill)[target.label].value:
                raise ValueError("atomic target label/name mismatch")
        if self.atomic_targets and not self.training_eligible:
            raise ValueError("ineligible segments cannot carry atomic targets")
        if self.atomic_supervision_mask != bool(self.atomic_targets):
            raise ValueError(
                "atomic_supervision_mask must match whether atomic_targets is non-empty"
            )
        if self.training_eligible != self.atomic_supervision_mask:
            raise ValueError(
                "stage-1 training_eligible must match atomic_supervision_mask"
            )
        if self.atomic_ratio_blocks and not self.training_eligible:
            raise ValueError("ineligible segments cannot carry atomic ratio blocks")
        target_labels = {target.label for target in self.atomic_targets}
        previous_end = 0.0
        duration = self.end_s - self.start_s
        for block in self.atomic_ratio_blocks:
            if block.start_offset_s + 1e-6 < previous_end:
                raise ValueError("atomic ratio blocks must be ordered and non-overlapping")
            if block.end_offset_s > duration + 1e-6:
                raise ValueError("atomic ratio block exceeds its segment")
            nonzero = {index for index, value in enumerate(block.weights) if value > 0}
            if not nonzero.issubset(target_labels):
                raise ValueError("atomic ratio block may only use the fixed segment atoms")
            previous_end = block.end_offset_s
        probabilities = [float(value) for value in self.atomic_probabilities]
        if len(probabilities) == MOTION_ATOMIC_COUNT:
            probabilities.append(0.0)
            self.atomic_probabilities = probabilities
        if any(value < 0 for value in probabilities) or sum(probabilities) <= 0:
            raise ValueError(
                "atomic_probabilities must be non-negative with positive sum"
            )
        self.atomic_probabilities = [
            value / sum(probabilities) for value in probabilities
        ]
        if self.gate_probabilities is not None:
            gate_probabilities = [float(value) for value in self.gate_probabilities]
            if len(gate_probabilities) == MOTION_ATOMIC_COUNT:
                gate_probabilities.append(0.0)
            gate_total = sum(gate_probabilities)
            if any(value < 0 for value in gate_probabilities) or gate_total <= 0:
                raise ValueError(
                    "gate_probabilities must be non-negative with positive sum"
                )
            self.gate_probabilities = [
                value / gate_total for value in gate_probabilities
            ]
        expected_targets = {"single": 1, "dual": 2}.get(self.gate_mode, 0)
        if self.training_eligible and len(self.atomic_targets) != expected_targets:
            raise ValueError("eligible segment targets must match gate_mode")
        return self


class FinalAnnotation(StrictModel):
    schema_version: str = "2.3"
    episode_id: str
    task: str
    global_description: str
    duration_s: float = Field(gt=0)
    sampled_fps: float = Field(gt=0)
    sampled_timestamps_s: list[float]
    provider: Literal["qwen", "codex"]
    model: str
    reviewed: bool
    axis_convention: str
    source_videos: list[str]
    augmentation: Literal["none", "mirror_left_to_right"] = "none"
    tcp_trace: str | None = None
    joint_trace: str | None = None
    motion_source_type: str | None = Field(
        default=None, pattern="^(tcp_pose|joint_fk)$"
    )
    quantity_value: float | None = None
    quantity_unit: str | None = None
    quantity_scale: float | None = Field(default=None, gt=0)
    segments: list[FinalSegment] = Field(min_length=1)
    usage: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_quantity_metadata(self) -> "FinalAnnotation":
        fields = (self.quantity_value, self.quantity_unit, self.quantity_scale)
        if any(value is not None for value in fields) and not all(
            value is not None for value in fields
        ):
            raise ValueError(
                "quantity_value, quantity_unit, and quantity_scale must be supplied together"
            )
        return self
