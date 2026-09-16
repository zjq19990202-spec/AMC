from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from atomic_latent_vla.atomic import (
    ATOMIC_BASE_INSTRUCTIONS,
    ATOMIC_ID_TO_SKILL,
)
from atomic_latent_vla.data.gating import AtomicGateConfig, AtomicGateDecision

from .client import AnnotationClient
from .motion import BaseMotionTrace, JointMotionTrace, MotionTrace
from .prompts import (
    build_messages,
    proposal_user_text,
    review_user_text,
    system_prompt,
)
from .schema import (
    AtomicTargetOutput,
    CandidateAnnotation,
    CandidateSegment,
    FinalAnnotation,
    FinalSegment,
)
from .timeline import FixedAtomicSegment, build_fk_atomic_timeline
from .video import SampledVideo, sample_synchronized_videos


DEFAULT_AXIS_CONVENTION = (
    "+x points forward from the robot base, +y points to the robot's left, and +z points up. "
    "Positive rotations follow the right-hand rule about the base-frame +x, +y, and +z axes."
)


def _instruction_has_signed_axis(instruction: str, atom_name: str) -> bool:
    """Require an auditable axis marker for every retained FK atom."""
    _, axis, sign = atom_name.split("_")
    text = instruction.casefold()
    symbol = "+" if sign == "pos" else "-"
    if re.search(rf"(?<!\w){re.escape(symbol)}\s*{axis}\b", text):
        return True
    sign_word = "positive" if sign == "pos" else "negative"
    return bool(
        re.search(
            rf"\b{sign_word}(?:ly)?(?:\s+[a-z-]+){{0,6}}\s+{axis}\b",
            text,
        )
    )


_MOVE_DIRECTION_PATTERN = re.compile(
    r"\b(towards?|into|onto|through|approach|retract|withdraw|insert)\b|"
    r"\b(away from|out of|closer to|to the (?:left|right) of)\b"
)
_TARGET_LOCATION_PATTERN = re.compile(
    r"\b(ahead|in front|behind|left|right|above|below|over|under|inside|outside|"
    r"forward-left|forward-right|back-left|back-right|upper-left|upper-right|"
    r"lower-left|lower-right|centered|offset)\b|"
    r"\b(?:to|on) the (?:left|right|front|back|upper|lower) (?:of|side)\b"
)
_NATURAL_MOVE_PATTERNS = {
    "move_x_pos": re.compile(r"\b(forward|forwards)\b"),
    "move_x_neg": re.compile(r"\b(backward|backwards)\b"),
    "move_y_pos": re.compile(r"\b(left|leftward|leftwards)\b"),
    "move_y_neg": re.compile(r"\b(right|rightward|rightwards)\b"),
    "move_z_pos": re.compile(r"\b(up|upward|upwards|raise|lift)\b"),
    "move_z_neg": re.compile(r"\b(down|downward|downwards|lower|descend)\b"),
}
_NATURAL_MOVE_WORDS = {
    "move_x_pos": "forward",
    "move_x_neg": "backward",
    "move_y_pos": "left",
    "move_y_neg": "right",
    "move_z_pos": "up",
    "move_z_neg": "down",
}


def _normalize_translation_axis_redundancy(instruction: str) -> str:
    """Keep natural move words while removing forbidden redundant +/- axis prose."""
    return re.sub(
        r"\s+(?:along|in)\s+(?:the\s+)?base-frame\s*[+-]\s*[xyz](?:\s+direction)?",
        "",
        instruction,
        flags=re.IGNORECASE,
    )


def _normalize_rotation_axis_sign(instruction: str) -> str:
    """`negative about base-frame +x` is valid but normalize it for the checker."""
    return re.sub(r"(base-frame\s+)[+]\s*([xyz])\b", r"\1\2", instruction, flags=re.I)
_HUMAN_CONTROL_PATTERN = re.compile(
    r"\b(human|operator|person|controller|teleoperat(?:e|ed|es|ing|ion)|human assistance)\b"
)
_MULTIPLE_TOOL_PATTERN = re.compile(
    r"\b(second|another|additional)\s+(?:[a-z-]+\s+){0,2}"
    r"(tool|object|screwdriver|driver)\b"
)
_TASK_HUMAN_PATTERN = re.compile(
    r"\b(human|operator|person|controller|teleoperat(?:e|ed|es|ing|ion)|assist)\b"
)
_TASK_MULTIPLE_PATTERN = re.compile(
    r"\b(second|another|additional|two|multiple|both)\b"
)
_SIGNED_AXIS_PATTERN = re.compile(
    r"(?<!\w)[+-]\s*[xyz]\b|\b(?:positive|negative)(?:ly)?(?:\s+[a-z-]+){0,4}\s+[xyz]\b",
    re.IGNORECASE,
)
_ROTATION_AXIS_PHRASE_PATTERN = re.compile(
    r"\b(?:rotate|rotates|rotating|rotation)\b[^.;,]*?"
    r"\b(?:positive|negative)(?:ly)?\b[^.;,]*?"
    r"\b(?:about|around)\b[^.;,]*?\b[+-]?\s*[xyz]\b",
    re.IGNORECASE,
)


def _remove_matching_sentences(text: str, pattern: re.Pattern[str], fallback: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    retained = [sentence for sentence in sentences if not pattern.search(sentence.casefold())]
    cleaned = " ".join(retained).strip()
    return cleaned or fallback


def _instruction_has_natural_direction(instruction: str, atom_name: str) -> bool:
    pattern = _NATURAL_MOVE_PATTERNS.get(atom_name)
    return bool(pattern and pattern.search(instruction.casefold()))


def _instruction_has_direction_grounding(
    instruction: str, atom_names: list[str]
) -> bool:
    """Accept natural translation wording and explicit signed rotation axes."""
    text = instruction.casefold()
    translation_text = _ROTATION_AXIS_PHRASE_PATTERN.sub("", text)
    rotation_atoms = [
        atom_name for atom_name in atom_names if atom_name.startswith("rotate_")
    ]
    if any(
        not _instruction_has_signed_axis(instruction, atom_name)
        for atom_name in rotation_atoms
    ):
        return False
    rotation_signatures = {
        tuple(atom_name.split("_")[1:]) for atom_name in rotation_atoms
    }
    for atom_name in atom_names:
        if not atom_name.startswith("move_"):
            continue
        # Translation prompts must use natural base-frame words or a visible
        # target-relative relation.  Signed axes are reserved for rotation so
        # the model cannot produce contradictions such as "backward along +x".
        _, axis, _ = atom_name.split("_")
        if _SIGNED_AXIS_PATTERN.search(translation_text):
            return False
    missing_move_atoms = [
        atom_name
        for atom_name in atom_names
        if atom_name.startswith("move_")
        if not _instruction_has_natural_direction(instruction, atom_name)
    ]
    if missing_move_atoms:
        if not _MOVE_DIRECTION_PATTERN.search(text):
            return False
        if not _TARGET_LOCATION_PATTERN.search(text):
            return False
    return True


@dataclass(frozen=True)
class PipelineConfig:
    sample_fps: float = 3.0
    max_frames: int = 400
    tile_width: int = 448
    jpeg_quality: int = 82
    max_pixels: int = 655360
    min_segment_duration_s: float = 2.0
    single_p1: float = 0.65
    single_margin: float = 0.25
    dual_sum: float = 0.75
    dual_p2: float = 0.20
    dual_p3_max: float = 0.12
    fk_translation_scale_m: float = 0.02
    fk_rotation_scale_rad: float = 0.075
    fk_activity_threshold: float = 0.15
    fk_top2_temperature: float = 0.10
    write_video_file_input: bool = False
    use_vlm_interaction_gate: bool = False
    review: bool = False
    axis_convention: str = DEFAULT_AXIS_CONVENTION

    def __post_init__(self) -> None:
        if self.sample_fps <= 0 or self.max_frames <= 0:
            raise ValueError("sample_fps and max_frames must be positive")
        if self.min_segment_duration_s <= 0:
            raise ValueError("min_segment_duration_s must be positive")
        if self.fk_translation_scale_m <= 0 or self.fk_rotation_scale_rad <= 0:
            raise ValueError("FK translation/rotation scales must be positive")
        if self.fk_activity_threshold < 0:
            raise ValueError("fk_activity_threshold must be non-negative")
        if self.fk_top2_temperature <= 0:
            raise ValueError("fk_top2_temperature must be positive")

    def gate_config(self) -> AtomicGateConfig:
        return AtomicGateConfig(
            single_p1=self.single_p1,
            single_margin=self.single_margin,
            dual_sum=self.dual_sum,
            dual_p2=self.dual_p2,
            dual_p3_max=self.dual_p3_max,
        )


class AtomicSegmentationPipeline:
    def __init__(
        self, client: AnnotationClient, config: PipelineConfig | None = None
    ) -> None:
        self.client = client
        self.config = config or PipelineConfig()

    def _lock_candidate_to_fk_timeline(
        self,
        annotation: CandidateAnnotation,
        fixed_timeline: list[FixedAtomicSegment],
    ) -> CandidateAnnotation:
        if len(annotation.segments) != len(fixed_timeline):
            raise ValueError(
                f"expected exactly {len(fixed_timeline)} fixed segments, "
                f"received {len(annotation.segments)}"
            )
        normalized: list[CandidateSegment] = []
        for segment, fixed in zip(annotation.segments, fixed_timeline, strict=True):
            if segment.segment_id != fixed.segment_id:
                raise ValueError(
                    f"segment id changed: expected {fixed.segment_id}, got {segment.segment_id}"
                )
            if not np.isclose(
                segment.start_s, fixed.start_s, atol=1e-3
            ) or not np.isclose(segment.end_s, fixed.end_s, atol=1e-3):
                raise ValueError(
                    f"segment {fixed.segment_id} changed fixed bounds "
                    f"[{fixed.start_s}, {fixed.end_s}] to [{segment.start_s}, {segment.end_s}]"
                )
            if fixed.decision.mode in {"single", "dual"}:
                atom_names = [
                    ATOMIC_ID_TO_SKILL[label].value for label in fixed.decision.labels
                ]
                segment = segment.model_copy(
                    update={
                        "low_level_instruction": _normalize_rotation_axis_sign(
                            segment.low_level_instruction
                        )
                    }
                )
                if any(name.startswith("move_") for name in atom_names):
                    instruction = _normalize_translation_axis_redundancy(
                        segment.low_level_instruction
                    )
                    segment = segment.model_copy(
                        update={"low_level_instruction": instruction}
                    )
                if not _instruction_has_direction_grounding(
                    segment.low_level_instruction, atom_names
                ):
                    expected_directions = []
                    for atom_name in atom_names:
                        if atom_name.startswith("move_"):
                            expected_directions.append(
                                f"{atom_name} -> '{_NATURAL_MOVE_WORDS[atom_name]}'"
                            )
                        else:
                            _, axis, sign = atom_name.split("_")
                            sign_word = "positive" if sign == "pos" else "negative"
                            expected_directions.append(
                                f"{atom_name} -> '{sign_word} about base-frame {axis}'"
                            )
                    raise ValueError(
                        f"segment {fixed.segment_id} instruction lacks direction grounding for "
                        f"{atom_names}; translations need natural base directions or a located "
                        "target-relative motion without ±xyz; rotations always "
                        f"need their signed base-frame axes. Use: {expected_directions}. "
                        f"Received: {segment.low_level_instruction!r}"
                    )
                # Keep the requested 12--24-word style, but leave a small
                # validation margin for a valid dual-axis sentence from the
                # model instead of rejecting an otherwise usable annotation.
                if len(segment.low_level_instruction.split()) > 36:
                    raise ValueError(
                        f"segment {fixed.segment_id} low_level_instruction exceeds the 32-word "
                        f"hard limit ({len(segment.low_level_instruction.split())} words); "
                        f"rewrite it as one concise action clause. Received: "
                        f"{segment.low_level_instruction!r}"
                    )
            normalized.append(
                segment.model_copy(
                    update={"start_s": fixed.start_s, "end_s": fixed.end_s}
                )
            )
        return CandidateAnnotation(
            global_description=annotation.global_description,
            segments=normalized,
        )

    def _resolve_targets(
        self,
        segment: CandidateSegment,
        fixed: FixedAtomicSegment,
    ) -> tuple[list[AtomicTargetOutput], str, AtomicGateDecision]:
        if self.config.use_vlm_interaction_gate and segment.strong_interaction:
            return (
                [],
                "none",
                AtomicGateDecision(
                    "interaction",
                    (),
                    (),
                    "strong interaction disables atomic supervision",
                ),
            )
        probabilities = np.asarray(fixed.atomic_probabilities, dtype=np.float64)
        decision = fixed.decision
        if decision.mode == "drop":
            return [], "none", decision
        normalized = probabilities / probabilities.sum()
        targets = [
            AtomicTargetOutput(
                label=label,
                name=ATOMIC_ID_TO_SKILL[label],
                confidence=float(np.clip(normalized[label], 1e-6, 1.0)),
            )
            for label in decision.labels
        ]
        return targets, "fk", decision

    def _request_candidate(
        self,
        messages: list[dict],
        fixed_timeline: list[FixedAtomicSegment],
        *,
        task: str = "",
        schema_attempts: int = 3,
    ) -> tuple[CandidateAnnotation, list[dict]]:
        current_messages = list(messages)
        usages: list[dict] = []
        last_error: Exception | None = None
        # A full recording can contain 50--64 immutable FK intervals.  The
        # former fixed 4096-token cap truncates the JSON array mid-object,
        # even when the model follows the requested schema.  Budget for the
        # two required text fields per interval; all current CR1 recordings
        # remain below the conservative 16k ceiling.
        max_tokens = min(16_384, max(4_096, 512 + 128 * len(fixed_timeline)))
        for attempt in range(schema_attempts):
            # Preserve the default invocation for short timelines; this also
            # keeps third-party AnnotationClient implementations that only
            # expose the historical one-argument method compatible.
            if max_tokens == 4_096:
                payload, usage = self.client.complete_json(current_messages)
            else:
                payload, usage = self.client.complete_json(
                    current_messages, max_tokens=max_tokens
                )
            usages.append(usage)
            try:
                candidate = CandidateAnnotation.model_validate(payload)
                task_text = task.casefold()
                forbidden_patterns: list[tuple[re.Pattern[str], str]] = []
                if not _TASK_HUMAN_PATTERN.search(task_text):
                    forbidden_patterns.append(
                        (_HUMAN_CONTROL_PATTERN, "unsupported human/control claim")
                    )
                if not _TASK_MULTIPLE_PATTERN.search(task_text):
                    forbidden_patterns.append(
                        (_MULTIPLE_TOOL_PATTERN, "unsupported second/another tool claim")
                    )
                global_description = candidate.global_description
                segments: list[CandidateSegment] = []
                for fixed, segment in zip(
                    fixed_timeline, candidate.segments, strict=True
                ):
                    instruction = segment.low_level_instruction
                    evidence = segment.visual_evidence
                    for pattern, reason in forbidden_patterns:
                        if pattern.search(instruction.casefold()):
                            if fixed.decision.mode in {"single", "dual"}:
                                raise ValueError(
                                    f"segment {segment.segment_id} instruction contains {reason}; "
                                    "remove it while preserving the fixed robot motion"
                                )
                            instruction = _remove_matching_sentences(
                                instruction,
                                pattern,
                                "No stable right-arm atomic motion dominates this interval.",
                            )
                        evidence = _remove_matching_sentences(
                            evidence,
                            pattern,
                            "The supplied views show the right-arm state for this interval.",
                        )
                        global_description = _remove_matching_sentences(
                            global_description,
                            pattern,
                            f"The robot performs the episode task: {task}.",
                        )
                    segments.append(
                        segment.model_copy(
                            update={
                                "low_level_instruction": instruction,
                                "visual_evidence": evidence,
                            }
                        )
                    )
                candidate = candidate.model_copy(
                    update={
                        "global_description": global_description,
                        "segments": segments,
                    }
                )
                return (
                    self._lock_candidate_to_fk_timeline(candidate, fixed_timeline),
                    usages,
                )
            except Exception as error:
                last_error = error
                if attempt + 1 >= schema_attempts:
                    break
                current_messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(payload, ensure_ascii=False),
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "The JSON failed schema/timeline validation. Correct it and "
                                        "return the complete JSON object only. Validation error: "
                                        + str(error)[:1500]
                                    ),
                                }
                            ],
                        },
                    ]
                )
        raise RuntimeError(
            "annotation model returned invalid JSON after "
            f"{schema_attempts} attempts: {last_error}"
        ) from last_error

    def _finalize(
        self,
        *,
        episode_id: str,
        task: str,
        videos: list[str | Path],
        sampled: SampledVideo,
        annotation: CandidateAnnotation,
        fixed_timeline: list[FixedAtomicSegment],
        trace: BaseMotionTrace | None,
        reviewed: bool,
        usage: dict,
        quantity_value: float | None,
        quantity_unit: str | None,
        quantity_scale: float | None,
        augmentation: str = "none",
    ) -> FinalAnnotation:
        final_segments: list[FinalSegment] = []
        if len(annotation.segments) != len(fixed_timeline):
            raise ValueError(
                "semantic annotation and fixed FK timeline length mismatch"
            )
        for segment, fixed in zip(annotation.segments, fixed_timeline, strict=True):
            targets, source, decision = self._resolve_targets(segment, fixed)
            duration_eligible = (
                segment.end_s - segment.start_s
                >= self.config.min_segment_duration_s - 1e-6
            )
            # Stage 1 follows FK single/dual/drop. The optional VLM interaction gate can
            # exclude contact-rich intervals, but it is disabled by default.
            training_eligible = duration_eligible and bool(targets)
            if not training_eligible:
                targets = []
                source = "none"
            final_segments.append(
                FinalSegment(
                    segment_id=segment.segment_id,
                    start_s=segment.start_s,
                    end_s=segment.end_s,
                    training_eligible=training_eligible,
                    atomic_supervision_mask=bool(targets),
                    atomic_probabilities=list(fixed.atomic_probabilities),
                    gate_probabilities=(
                        list(fixed.gate_probabilities)
                        if fixed.gate_probabilities
                        else None
                    ),
                    gate_mode=decision.mode,
                    gate_reason=(
                        decision.reason
                        if duration_eligible
                        else (
                            f"{decision.reason}; duration "
                            f"{segment.end_s - segment.start_s:.3f}s is shorter than "
                            f"{self.config.min_segment_duration_s:.3f}s"
                        )
                    ),
                    atomic_targets=targets,
                    atomic_ratio_blocks=(
                        [
                            {
                                "start_offset_s": block.start_offset_s,
                                "end_offset_s": block.end_offset_s,
                                "weights": list(block.weights),
                                "valid": block.valid,
                            }
                            for block in fixed.atomic_ratio_blocks
                        ]
                        if training_eligible
                        else []
                    ),
                    strong_interaction=(
                        self.config.use_vlm_interaction_gate
                        and segment.strong_interaction
                    ),
                    base_instructions=[
                        ATOMIC_BASE_INSTRUCTIONS[target.name] for target in targets
                    ],
                    low_level_instruction=segment.low_level_instruction,
                    visual_evidence=segment.visual_evidence,
                    label_source=source,
                )
            )
        return FinalAnnotation(
            episode_id=episode_id,
            task=task,
            global_description=annotation.global_description,
            duration_s=sampled.duration_s,
            sampled_fps=sampled.sampled_fps,
            sampled_timestamps_s=sampled.timestamps_s,
            provider=self.client.provider,
            model=self.client.model,
            reviewed=reviewed,
            axis_convention=self.config.axis_convention,
            source_videos=[str(Path(video).expanduser().resolve()) for video in videos],
            augmentation=augmentation,
            tcp_trace=(
                str(trace.source) if trace and trace.source_type == "tcp_pose" else None
            ),
            joint_trace=(
                str(trace.source) if trace and trace.source_type == "joint_fk" else None
            ),
            motion_source_type=trace.source_type if trace else None,
            quantity_value=quantity_value,
            quantity_unit=quantity_unit,
            quantity_scale=quantity_scale,
            segments=final_segments,
            usage=usage,
        )

    def run(
        self,
        *,
        videos: list[str | Path],
        task: str,
        episode_id: str = "episode_0",
        view_names: list[str] | None = None,
        extra_context: str = "",
        tcp_trace: str | Path | None = None,
        joint_trace: str | Path | None = None,
        motion_trace: BaseMotionTrace | None = None,
        arm: str = "left",
        urdf_path: str | Path | None = None,
        tcp_frame: str | None = None,
        mount_xyz: tuple[float, float, float] | None = None,
        quantity_value: float | None = None,
        quantity_unit: str | None = None,
        quantity_scale: float | None = None,
        horizontal_flip: list[bool] | tuple[bool, ...] | None = None,
        augmentation: str = "none",
        fixed_timeline_override: list[FixedAtomicSegment] | None = None,
    ) -> FinalAnnotation:
        trace_inputs = sum(
            value is not None for value in (tcp_trace, joint_trace, motion_trace)
        )
        if trace_inputs > 1:
            raise ValueError(
                "tcp_trace, joint_trace, and motion_trace are mutually exclusive"
            )
        sampled = sample_synchronized_videos(
            videos,
            view_names=view_names,
            sample_fps=self.config.sample_fps,
            max_frames=self.config.max_frames,
            tile_width=self.config.tile_width,
            jpeg_quality=self.config.jpeg_quality,
            horizontal_flip=horizontal_flip,
            write_video_file=self.config.write_video_file_input,
        )
        try:
            if motion_trace is not None:
                trace = motion_trace
            elif joint_trace:
                if urdf_path is None:
                    raise ValueError("urdf_path is required with joint_trace")
                trace: BaseMotionTrace | None = JointMotionTrace.load_hdf5(
                    joint_trace,
                    arm=arm,
                    urdf_path=urdf_path,
                    tcp_frame=tcp_frame,
                    mount_xyz=mount_xyz,
                )
            else:
                trace = MotionTrace.load(tcp_trace) if tcp_trace else None
            if trace is None:
                raise ValueError(
                    "right-arm FK/TCP motion is required: VLM semantics cannot define atomic labels"
                )
            fixed_timeline = (
                fixed_timeline_override
                if fixed_timeline_override is not None
                else build_fk_atomic_timeline(
                    trace,
                    sampled.timestamps_s,
                    sampled.duration_s,
                    translation_scale_m_s=self.config.fk_translation_scale_m,
                    rotation_scale_rad_s=self.config.fk_rotation_scale_rad,
                    activity_threshold=self.config.fk_activity_threshold,
                    minimum_regime_s=self.config.min_segment_duration_s,
                    gate_config=self.config.gate_config(),
                    top2_temperature=self.config.fk_top2_temperature,
                )
            )
            video_item = sampled.api_item(self.config.max_pixels)
            system = system_prompt(
                self.config.axis_convention,
                min_segment_duration_s=self.config.min_segment_duration_s,
                enable_interaction_gate=self.config.use_vlm_interaction_gate,
            )
            proposal_messages = build_messages(
                system,
                video_item,
                proposal_user_text(
                    task=task,
                    duration_s=sampled.duration_s,
                    sampled_fps=sampled.sampled_fps,
                    view_names=sampled.view_names,
                    min_segment_duration_s=self.config.min_segment_duration_s,
                    fixed_timeline=fixed_timeline,
                    extra_context=extra_context,
                    enable_interaction_gate=self.config.use_vlm_interaction_gate,
                ),
            )
            proposal, proposal_usage = self._request_candidate(
                proposal_messages, fixed_timeline, task=task
            )

            final_candidate = proposal
            review_usage: list[dict] = []
            if self.config.review:
                review_messages = build_messages(
                    system,
                    video_item,
                    review_user_text(
                        task=task,
                        duration_s=sampled.duration_s,
                        sampled_fps=sampled.sampled_fps,
                        proposal=proposal,
                        min_segment_duration_s=self.config.min_segment_duration_s,
                        fixed_timeline=fixed_timeline,
                        extra_context=extra_context,
                        enable_interaction_gate=self.config.use_vlm_interaction_gate,
                    ),
                )
                final_candidate, review_usage = self._request_candidate(
                    review_messages, fixed_timeline, task=task
                )

            return self._finalize(
                episode_id=episode_id,
                task=task,
                videos=videos,
                sampled=sampled,
                annotation=final_candidate,
                fixed_timeline=fixed_timeline,
                trace=trace,
                reviewed=self.config.review,
                usage={
                    "proposal": proposal_usage,
                    "review": review_usage,
                },
                quantity_value=quantity_value,
                quantity_unit=quantity_unit,
                quantity_scale=quantity_scale,
                augmentation=augmentation,
            )
        finally:
            sampled.cleanup()
