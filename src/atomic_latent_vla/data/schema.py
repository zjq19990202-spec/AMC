from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atomic_latent_vla.atomic import ATOMIC_NAMES


NAME_TO_ID = {name: index for index, name in enumerate(ATOMIC_NAMES)}
NAME_TO_ID.update(
    {
        "roll_pos": 6,
        "roll_neg": 7,
        "pitch_pos": 8,
        "pitch_neg": 9,
        "yaw_pos": 10,
        "yaw_neg": 11,
    }
)


@dataclass(frozen=True)
class AtomicTarget:
    label: int
    confidence: float


@dataclass(frozen=True)
class SegmentAnnotation:
    segment_id: int
    start_s: float
    end_s: float
    instruction: str
    targets: tuple[AtomicTarget, ...]
    training_eligible: bool
    raw: dict[str, Any]

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def atomic_mode(self) -> str:
        return {0: "unlabeled", 1: "single", 2: "dual"}.get(len(self.targets), "drop")


@dataclass(frozen=True)
class EpisodeAnnotation:
    episode_id: str
    task: str
    global_description: str
    segments: tuple[SegmentAnnotation, ...]
    raw: dict[str, Any]


def _label_id(value: Any) -> int:
    if isinstance(value, str):
        if value not in NAME_TO_ID:
            raise ValueError(f"unknown atomic name: {value}")
        return NAME_TO_ID[value]
    label = int(value)
    if not 0 <= label < len(ATOMIC_NAMES):
        raise ValueError(f"atomic label is outside [0, 11]: {label}")
    return label


def _targets_from_segment(segment: dict[str, Any]) -> tuple[AtomicTarget, ...]:
    raw_targets = segment.get("atomic_targets")
    targets: list[AtomicTarget] = []
    if raw_targets is not None:
        if len(raw_targets) > 2:
            raise ValueError("atomic_targets supports at most two labels")
        for item in raw_targets:
            label_value = item.get("label", item.get("name"))
            confidence = float(item["confidence"])
            if confidence <= 0:
                raise ValueError("atomic target confidence must be positive")
            targets.append(AtomicTarget(_label_id(label_value), confidence))
    elif segment.get("atomic_supervision_mask", segment.get("atomic_label", -1) != -1):
        label_value = segment.get("atomic_label", segment.get("primary_atom"))
        confidence = float(
            segment.get("atomic_confidence", segment.get("direction_confidence", 1.0))
        )
        targets.append(AtomicTarget(_label_id(label_value), confidence))
    if len({target.label for target in targets}) != len(targets):
        raise ValueError("atomic_targets contains duplicate labels")
    if targets and sum(target.confidence for target in targets) <= 0:
        raise ValueError("atomic target confidences must sum to a positive value")
    return tuple(targets)


def load_annotation(path: str | Path) -> EpisodeAnnotation:
    with Path(path).open(encoding="utf-8") as handle:
        raw: dict[str, Any] = json.load(handle)
    segments = []
    for index, item in enumerate(raw.get("segments", [])):
        targets = _targets_from_segment(item)
        eligible = bool(item.get("training_eligible", True))
        if float(item["end_s"]) <= float(item["start_s"]):
            raise ValueError("segment end_s must be greater than start_s")
        segments.append(
            SegmentAnnotation(
                segment_id=int(item.get("segment_id", index)),
                start_s=float(item["start_s"]),
                end_s=float(item["end_s"]),
                instruction=str(
                    item.get("low_level_instruction")
                    or item.get("instruction")
                    or item.get("base_instruction")
                    or raw.get("task", "")
                ),
                targets=targets,
                training_eligible=eligible,
                raw=item,
            )
        )
    if not segments:
        raise ValueError("annotation contains no segments")
    return EpisodeAnnotation(
        episode_id=str(raw.get("episode_id", Path(path).stem)),
        task=str(raw.get("task", "")),
        global_description=str(raw.get("global_description", "")),
        segments=tuple(segments),
        raw=raw,
    )
