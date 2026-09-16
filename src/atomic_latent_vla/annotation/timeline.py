from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from atomic_latent_vla.atomic import (
    ATOMIC_BASE_INSTRUCTIONS,
    ATOMIC_ID_TO_SKILL,
    MOTION_ATOMIC_COUNT,
    NUM_ATOMIC_SKILLS,
    OPPOSITE_ATOMIC_ID,
    STAY_ATOMIC_ID,
)
from atomic_latent_vla.data.gating import (
    AtomicGateConfig,
    AtomicGateDecision,
    classify_atomic_targets,
)

from .motion import BaseMotionTrace


_ANNOTATION_DIRECTION_HINTS = {
    0: "move forward",
    1: "move backward",
    2: "move left",
    3: "move right",
    4: "move up",
    5: "move down",
}


_DUAL_RATIO_FLOOR = 1e-4


def _with_stay_channel(
    motion_probabilities: np.ndarray, stay_probability: float = 0.0
) -> np.ndarray:
    """Append the semantic stay channel to twelve-way FK motion evidence."""

    motion = np.asarray(motion_probabilities, dtype=np.float64)
    if motion.shape != (MOTION_ATOMIC_COUNT,):
        raise ValueError(
            f"motion probabilities must have shape ({MOTION_ATOMIC_COUNT},)"
        )
    result = np.zeros(NUM_ATOMIC_SKILLS, dtype=np.float64)
    result[:MOTION_ATOMIC_COUNT] = motion
    result[STAY_ATOMIC_ID] = float(stay_probability)
    total = float(result.sum())
    return result / total if total > 0.0 else result


@dataclass(frozen=True)
class AtomicRatioBlock:
    """One FK ratio target attached to a 1/3-second horizon-label block."""

    start_offset_s: float
    end_offset_s: float
    weights: tuple[float, ...]
    valid: bool


@dataclass(frozen=True)
class FixedAtomicSegment:
    """An FK-authoritative right-arm interval that a VLM may annotate but not alter."""

    segment_id: int
    start_s: float
    end_s: float
    atomic_probabilities: tuple[float, ...]
    activity_score: float
    decision: AtomicGateDecision
    # Fraction of constituent 3 Hz blocks whose strongest FK score exceeds
    # the activity threshold.  This is kept separate from the strict gate so
    # downstream composition targets can preserve stationary time mass
    # without changing historical Single/Dual/Drop decisions.
    active_fraction: float = 1.0
    # Full distribution used by the final single/dual/drop gate.  Keep this
    # separate from ``atomic_probabilities``, whose Top-2-mass weighting is
    # intended for reporting rather than KL supervision.
    gate_probabilities: tuple[float, ...] = ()
    atomic_ratio_blocks: tuple[AtomicRatioBlock, ...] = ()

    def prompt_payload(self) -> dict[str, object]:
        labels = [ATOMIC_ID_TO_SKILL[label].value for label in self.decision.labels]
        direction_hints = [
            _ANNOTATION_DIRECTION_HINTS.get(
                label, ATOMIC_BASE_INSTRUCTIONS[ATOMIC_ID_TO_SKILL[label]]
            )
            for label in self.decision.labels
        ]
        return {
            "segment_id": self.segment_id,
            "start_s": round(self.start_s, 6),
            "end_s": round(self.end_s, 6),
            "right_fk_mode": self.decision.mode,
            "right_fk_atoms": labels,
            # Audit context only; this never changes the fixed boundary/labels.
            "right_fk_gate_reason": self.decision.reason,
            "direction_hints": direction_hints,
            "right_fk_activity": round(self.activity_score, 4),
        }


@dataclass(frozen=True)
class _WindowState:
    labels: tuple[int, ...] | None
    probabilities: np.ndarray
    gate_probabilities: np.ndarray
    activity_score: float
    active_fraction: float
    decision: AtomicGateDecision
    stable: bool


def _top2_evidence(
    scores: np.ndarray,
    active: np.ndarray,
    *,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return conditional and mass-preserving top-2 evidence for each interval."""
    if temperature <= 0:
        raise ValueError("top-2 temperature must be positive")
    gate = np.zeros_like(scores, dtype=np.float64)
    reporting = np.zeros_like(scores, dtype=np.float64)
    for row_index in np.flatnonzero(active):
        row = np.asarray(scores[row_index], dtype=np.float64)
        total = float(row.sum())
        if total <= 0:
            continue
        normalized = row / total
        positive = np.flatnonzero(normalized > 0.0)
        if len(positive) > 2:
            order = np.argsort(normalized[positive])[-2:]
            indices = positive[order]
        else:
            indices = positive
        top2_mass = float(normalized[indices].sum())
        logits = normalized[indices] / temperature
        logits -= float(logits.max())
        weights = np.exp(logits)
        weights /= float(weights.sum())
        gate[row_index, indices] = weights
        reporting[row_index, indices] = top2_mass * weights
    return gate, reporting


def _classify_window(
    scores: np.ndarray,
    *,
    activity_threshold: float,
    gate_config: AtomicGateConfig,
    top2_temperature: float = 0.10,
) -> _WindowState:
    strongest = scores.max(axis=1)
    active = strongest >= activity_threshold
    active_fraction = float(active.mean())
    activity_score = float(strongest.mean())
    if not active.any():
        probabilities = _with_stay_channel(
            np.zeros(MOTION_ATOMIC_COUNT, dtype=np.float64), 1.0
        )
        decision = AtomicGateDecision(
            "single",
            (STAY_ATOMIC_ID,),
            (1.0,),
            f"right-arm FK stationary over window; active_fraction={active_fraction:.3f}",
        )
        return _WindowState(
            (STAY_ATOMIC_ID,),
            probabilities,
            probabilities,
            activity_score,
            active_fraction,
            decision,
            True,
        )

    gate_evidence, reporting_evidence = _top2_evidence(
        scores,
        active,
        temperature=top2_temperature,
    )
    probabilities = gate_evidence[active].sum(axis=0)
    probabilities /= float(probabilities.sum())
    probabilities = _with_stay_channel(probabilities)
    reporting_probabilities = reporting_evidence[active].sum(axis=0)
    reporting_total = float(reporting_probabilities.sum())
    if reporting_total > 0:
        reporting_probabilities /= reporting_total
    reporting_probabilities = _with_stay_channel(reporting_probabilities)
    decision = classify_atomic_targets(probabilities, gate_config)
    # One block's actual atom is its Top-1.  Once accumulated Top-2 evidence
    # has selected the horizon's final single/dual atoms, reject the horizon
    # if any constituent block actually moves opposite to one of those atoms.
    top1_labels = {int(np.argmax(scores[index])) for index in np.flatnonzero(active)}
    conflicting_top1 = tuple(
        label
        for label in sorted(top1_labels)
        if any(label == OPPOSITE_ATOMIC_ID[target] for target in decision.labels)
    )
    if decision.mode in {"single", "dual"} and conflicting_top1:
        conflict_decision = AtomicGateDecision(
            "drop",
            (),
            (),
            (
                "a block Top-1 atom opposes the final 50-frame horizon result; "
                f"final_labels={decision.labels}, conflicting_top1={conflicting_top1}"
            ),
        )
        return _WindowState(
            (),
            reporting_probabilities,
            probabilities,
            activity_score,
            active_fraction,
            conflict_decision,
            True,
        )
    if active_fraction < 0.5:
        # Mostly stationary is compatible with the preceding state, but is not a trainable atom.
        idle_decision = AtomicGateDecision(
            "drop",
            (),
            (),
            (
                "right-arm FK mostly stationary over window; "
                f"active_fraction={active_fraction:.3f} < 0.500"
            ),
        )
        return _WindowState(
            (),
            reporting_probabilities,
            probabilities,
            activity_score,
            active_fraction,
            idle_decision,
            True,
        )
    if decision.mode == "drop":
        return _WindowState(
            None,
            reporting_probabilities,
            probabilities,
            activity_score,
            active_fraction,
            decision,
            False,
        )
    return _WindowState(
        tuple(sorted(decision.labels)),
        reporting_probabilities,
        probabilities,
        activity_score,
        active_fraction,
        decision,
        True,
    )


def _step_atomic_label(score: np.ndarray, *, activity_threshold: float) -> int | None:
    """Return one step's Top-1 atom; Top-2 remains window-probability evidence."""
    if float(score.max()) < activity_threshold:
        return None
    row = np.asarray(score, dtype=np.float64)
    total = float(row.sum())
    if total <= 0:
        return None
    return int(np.argmax(row))


def _scaled_score_probabilities(
    scores: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    """Normalize accumulated scale-adjusted FK evidence over active intervals."""
    evidence = np.asarray(scores, dtype=np.float64)[active].sum(axis=0)
    total = float(evidence.sum())
    if total <= 0.0:
        return _with_stay_channel(
            np.full(MOTION_ATOMIC_COUNT, 1.0 / MOTION_ATOMIC_COUNT, dtype=np.float64)
        )
    return _with_stay_channel(evidence / total)


def _weighted_segment_probabilities(
    scores: np.ndarray,
    *,
    activity_threshold: float,
    top2_temperature: float,
) -> np.ndarray:
    """Compute reporting probabilities from continuous FK evidence, not gate votes."""
    strongest = scores.max(axis=1)
    active = strongest >= activity_threshold
    if not active.any():
        return _with_stay_channel(
            np.zeros(MOTION_ATOMIC_COUNT, dtype=np.float64), 1.0
        )
    _, reporting = _top2_evidence(
        scores,
        active,
        temperature=top2_temperature,
    )
    evidence = reporting[active].sum(axis=0)
    total = float(evidence.sum())
    if total > 0:
        return _with_stay_channel(evidence / total)
    return _with_stay_channel(
        np.full(MOTION_ATOMIC_COUNT, 1.0 / MOTION_ATOMIC_COUNT)
    )


def _local_ratio_blocks(
    scores: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    segment_start_s: float,
    labels: tuple[int, ...],
    top2_temperature: float,
) -> tuple[AtomicRatioBlock, ...]:
    """Normalize each interval using only the fixed segment atoms.

    The future action-horizon gate remains the sole owner of atom identity.
    These blocks only refine the relative strength of that already accepted
    one/two-atom set for a later KL target.
    """

    if not labels:
        return ()
    scores = np.asarray(scores, dtype=np.float64)
    blocks: list[AtomicRatioBlock] = []
    for index, (start_s, end_s) in enumerate(zip(starts, ends, strict=True)):
        weights = np.zeros(NUM_ATOMIC_SKILLS, dtype=np.float64)
        # Do not run another 12-way Top-2 here. The horizon gate has already
        # fixed identity; a third atom must not affect this local A/B ratio.
        selected = scores[index, list(labels)]
        selected_total = float(selected.sum())
        valid = selected_total > 0.0
        if valid:
            logits = selected / selected_total / top2_temperature
            logits -= float(logits.max())
            local_ratio = np.exp(logits)
            local_ratio /= float(local_ratio.sum())
            if len(labels) == 2:
                # Atom identity belongs to the two-second gate. Preserve both
                # members even when one is momentarily outside a block's Top-2,
                # otherwise AtomicTargets would silently turn this dual row
                # into a single and skip Q1's ratio KL entirely.
                local_ratio = np.maximum(local_ratio, _DUAL_RATIO_FLOOR)
                local_ratio /= float(local_ratio.sum())
            weights[list(labels)] = local_ratio
        blocks.append(
            AtomicRatioBlock(
                start_offset_s=round(float(start_s - segment_start_s), 6),
                end_offset_s=round(float(end_s - segment_start_s), 6),
                weights=tuple(float(value) for value in weights),
                valid=valid,
            )
        )
    return tuple(blocks)


def _discarded_window(
    scores: np.ndarray,
    *,
    activity_threshold: float,
    reason: str,
    top2_temperature: float = 0.10,
) -> _WindowState:
    """Describe rejected coverage without promoting it to atomic supervision."""
    if len(scores) == 0:
        raise ValueError("discarded FK window cannot be empty")
    strongest = scores.max(axis=1)
    active = strongest >= activity_threshold
    active_fraction = float(active.mean())
    activity_score = float(strongest.mean())
    if active.any():
        probabilities = _weighted_segment_probabilities(
            scores,
            activity_threshold=activity_threshold,
            top2_temperature=top2_temperature,
        )
        gate_evidence, _ = _top2_evidence(
            scores,
            active,
            temperature=top2_temperature,
        )
        gate_probabilities = gate_evidence[active].sum(axis=0)
        gate_probabilities /= float(gate_probabilities.sum())
        gate_probabilities = _with_stay_channel(gate_probabilities)
    else:
        probabilities = _with_stay_channel(
            np.full(MOTION_ATOMIC_COUNT, 1.0 / MOTION_ATOMIC_COUNT)
        )
        gate_probabilities = probabilities
    decision = AtomicGateDecision("drop", (), (), reason)
    return _WindowState(
        (),
        probabilities,
        gate_probabilities,
        activity_score,
        active_fraction,
        decision,
        True,
    )


def build_fk_atomic_timeline(
    trace: BaseMotionTrace,
    timestamps_s: list[float],
    duration_s: float,
    *,
    translation_scale_m_s: float,
    rotation_scale_rad_s: float,
    activity_threshold: float,
    minimum_regime_s: float,
    gate_config: AtomicGateConfig,
    top2_temperature: float = 0.10,
) -> list[FixedAtomicSegment]:
    """Assign an independent future-horizon atom to every sampled FK block.

    At 3 Hz, ``minimum_regime_s=5/3`` is exactly five 1/3-second blocks and
    matches the 50-frame action horizon at 30 FPS.  Block ``i`` is classified
    only from the accumulated Top-2 evidence in ``i..i+4``.  The decision is
    attached to block ``i`` itself; decisions are never extended, recovered,
    merged, or propagated between blocks.  A tail block without a complete
    future horizon is deterministically marked ``drop``.

    ``minimum_regime_s`` retains its historical name for call-site/schema
    compatibility, but now denotes the action-horizon evidence duration rather
    than a minimum output-segment duration.
    """
    times = np.asarray(timestamps_s, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("FK timeline requires at least two strictly increasing timestamps")
    if duration_s <= times[0]:
        raise ValueError("duration_s must exceed the first sampled timestamp")
    ends = np.concatenate([times[1:], [float(duration_s)]])
    valid = ends > times
    starts = times[valid]
    ends = ends[valid]
    if len(starts) == 0:
        raise ValueError("FK timeline has no positive-duration intervals")

    raw_scores = []
    for start_s, end_s in zip(starts, ends, strict=True):
        dt = float(end_s - start_s)
        raw_scores.append(
            trace.atomic_scores_interval(
                float(start_s),
                float(end_s),
                translation_scale_m=translation_scale_m_s * dt,
                rotation_scale_rad=rotation_scale_rad_s * dt,
            )
        )
    raw = np.asarray(raw_scores, dtype=np.float64)
    median_dt = float(np.median(ends - starts))
    timeline: list[FixedAtomicSegment] = []
    horizon_samples = max(
        1, int(np.ceil(minimum_regime_s / median_dt - 1e-9))
    )
    for block_index in range(len(raw)):
        horizon_end = block_index + horizon_samples
        available_end = min(horizon_end, len(raw))
        available_duration_s = float(
            ends[available_end - 1] - starts[block_index]
        )
        has_complete_horizon = (
            horizon_end <= len(raw)
            and available_duration_s >= minimum_regime_s - 1e-6
        )
        if has_complete_horizon:
            state = _classify_window(
                raw[block_index:horizon_end],
                activity_threshold=activity_threshold,
                gate_config=gate_config,
                top2_temperature=top2_temperature,
            )
        else:
            state = _discarded_window(
                raw[block_index:],
                activity_threshold=activity_threshold,
                reason=(
                    "discarded FK block; insufficient future coverage for "
                    f"the {minimum_regime_s:g}s action horizon "
                    f"({len(raw) - block_index}/{horizon_samples} blocks, "
                    f"{available_duration_s:.6f}s available)"
                ),
                top2_temperature=top2_temperature,
            )

        block_start_s = float(starts[block_index])
        block_end_s = float(ends[block_index])
        ratio_blocks: tuple[AtomicRatioBlock, ...] = ()
        if state.decision.mode in {"single", "dual"}:
            weights = np.zeros(NUM_ATOMIC_SKILLS, dtype=np.float64)
            weights[list(state.decision.labels)] = state.decision.weights
            total = float(weights.sum())
            if total > 0.0:
                weights /= total
                ratio_blocks = (
                    AtomicRatioBlock(
                        start_offset_s=0.0,
                        end_offset_s=round(block_end_s - block_start_s, 6),
                        weights=tuple(float(value) for value in weights),
                        valid=True,
                    ),
                )

        timeline.append(
            FixedAtomicSegment(
                segment_id=block_index,
                start_s=round(block_start_s, 6),
                end_s=round(block_end_s, 6),
                atomic_probabilities=tuple(float(value) for value in state.probabilities),
                activity_score=state.activity_score,
                decision=state.decision,
                active_fraction=state.active_fraction,
                gate_probabilities=tuple(
                    float(value) for value in state.gate_probabilities
                ),
                atomic_ratio_blocks=ratio_blocks,
            )
        )
    return timeline
