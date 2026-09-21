from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from atomic_latent_vla.atomic import (
    MOTION_ATOMIC_COUNT,
    NUM_ATOMIC_SKILLS,
    OPPOSITE_ATOMIC_ID,
    STAY_ATOMIC_ID,
)


def _upgrade_weights(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Read legacy twelve-motion rows as thirteen-way rows with stay=0."""

    weights = np.asarray(values, dtype=np.float32)
    if weights.shape == (MOTION_ATOMIC_COUNT,):
        weights = np.pad(weights, (0, NUM_ATOMIC_SKILLS - MOTION_ATOMIC_COUNT))
    return weights


def local_atomic_weight_rows(
    timestamps_s: np.ndarray,
    segment: dict[str, Any],
    fallback_weights: Sequence[float],
) -> np.ndarray:
    """Expand relative atomic-ratio blocks onto arbitrary source-frame times.

    Existing schema-2.2 annotations have no local blocks and retain their
    segment-level target. Schema-2.3 blocks override that fallback only on the
    intervals they explicitly cover; invalid/low-activity blocks become zero.
    """

    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    fallback = _upgrade_weights(fallback_weights)
    if fallback.shape != (NUM_ATOMIC_SKILLS,):
        raise ValueError(
            f"fallback_weights must have shape ({NUM_ATOMIC_SKILLS},), got {fallback.shape}"
        )
    rows = np.broadcast_to(fallback, (len(timestamps), NUM_ATOMIC_SKILLS)).copy()
    blocks = segment.get("atomic_ratio_blocks") or []
    if not blocks:
        return rows
    segment_start = float(segment["start_s"])
    offsets = timestamps - segment_start
    for block in blocks:
        start = float(block["start_offset_s"])
        end = float(block["end_offset_s"])
        hit = (offsets >= start - 1e-6) & (offsets < end - 1e-6)
        weights = _upgrade_weights(block["weights"])
        if weights.shape != (NUM_ATOMIC_SKILLS,):
            raise ValueError(
                f"atomic ratio block weights must have shape ({NUM_ATOMIC_SKILLS},), "
                f"got {weights.shape}"
            )
        rows[hit] = weights if bool(block.get("valid", True)) else 0.0
    return rows


def aggregate_horizon_atomic_weights(
    row_atoms: np.ndarray,
    row_type: np.ndarray,
    row_segment: np.ndarray,
) -> tuple[bool, np.ndarray]:
    """Aggregate frame-aligned local ratios for one fixed-horizon target."""

    atoms = np.asarray(row_atoms, dtype=np.float32)
    if atoms.ndim == 2 and atoms.shape[1] == MOTION_ATOMIC_COUNT:
        atoms = np.pad(
            atoms,
            ((0, 0), (0, NUM_ATOMIC_SKILLS - MOTION_ATOMIC_COUNT)),
        )
    kinds = np.asarray(row_type)
    segments = np.asarray(row_segment)
    if atoms.ndim != 2 or atoms.shape[1] != NUM_ATOMIC_SKILLS:
        raise ValueError(
            f"row_atoms must have shape [H,{NUM_ATOMIC_SKILLS}], got {atoms.shape}"
        )
    if kinds.shape != atoms.shape[:1] or segments.shape != atoms.shape[:1]:
        raise ValueError("row_type and row_segment must have shape [H]")
    weights = atoms.sum(axis=0, dtype=np.float64)
    total = float(weights.sum())
    positive = np.flatnonzero(weights > 0.0)
    same_valid_segment = bool(
        len(segments) > 0
        and np.all(kinds == 1)
        and int(segments[0]) >= 0
        and np.all(segments == segments[0])
    )
    labels_valid = bool(
        1 <= len(positive) <= 2
        and not (
            len(positive) == 2
            and (
                OPPOSITE_ATOMIC_ID.get(int(positive[0])) == int(positive[1])
                or STAY_ATOMIC_ID in positive
            )
        )
    )
    supervised = same_valid_segment and labels_valid and total > 0.0
    if not supervised:
        return False, np.zeros(NUM_ATOMIC_SKILLS, dtype=np.float32)
    return True, (weights / total).astype(np.float32)
