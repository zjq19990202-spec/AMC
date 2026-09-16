from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from atomic_latent_vla.atomic import (
    MOTION_ATOMIC_COUNT,
    NUM_ATOMIC_SKILLS,
    OPPOSITE_ATOMIC_ID,
    STAY_ATOMIC_ID,
)


@dataclass(frozen=True)
class AtomicGateConfig:
    single_p1: float = 0.65
    # A dominant calibrated atom is sufficient for a single-label segment.
    # Keep the field for backwards-compatible serialized configs, but disable
    # the extra margin criterion in the current gate.
    single_margin: float = 0.0
    dual_sum: float = 0.75
    dual_p2: float = 0.20
    dual_p3_max: float = 0.12


@dataclass(frozen=True)
class AtomicGateDecision:
    mode: str
    labels: tuple[int, ...]
    weights: tuple[float, ...]
    reason: str


def topk_gate_composition(
    gate_probabilities: np.ndarray,
    *,
    top_k: int = 5,
) -> tuple[np.ndarray, float]:
    """Build a normalized Top-k gate composition and report its confidence."""

    gate = np.asarray(gate_probabilities, dtype=np.float32)
    if gate.shape == (MOTION_ATOMIC_COUNT,):
        gate = np.pad(gate, (0, NUM_ATOMIC_SKILLS - MOTION_ATOMIC_COUNT))
    if gate.shape != (NUM_ATOMIC_SKILLS,):
        raise ValueError(
            f"gate_probabilities must have shape ({NUM_ATOMIC_SKILLS},), got {gate.shape}"
        )
    if not np.all(np.isfinite(gate)) or np.any(gate < 0.0):
        raise ValueError("gate_probabilities must be finite and non-negative")
    total = float(gate.sum())
    if total <= 0.0:
        raise ValueError("gate_probabilities must have positive mass")
    if not 0 < top_k <= NUM_ATOMIC_SKILLS:
        raise ValueError("top_k must be in [1, NUM_ATOMIC_SKILLS]")

    full = gate / total
    entropy = -float(
        np.sum(
            np.where(
                full > 0.0,
                full * np.log(np.maximum(full, 1e-12)),
                0.0,
            )
        )
    )
    confidence = float(
        np.clip(1.0 - entropy / np.log(NUM_ATOMIC_SKILLS), 0.0, 1.0)
    )
    selected = np.argsort(-full, kind="stable")[:top_k]
    composition = np.zeros_like(full)
    composition[selected] = full[selected]
    composition /= float(composition.sum())
    return composition.astype(np.float32, copy=False), confidence


def classify_atomic_targets(
    probabilities: np.ndarray,
    config: AtomicGateConfig = AtomicGateConfig(),
) -> AtomicGateDecision:
    """Classify calibrated 13-way scores with an exclusive stay atom."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape == (MOTION_ATOMIC_COUNT,):
        probabilities = np.pad(
            probabilities, (0, NUM_ATOMIC_SKILLS - MOTION_ATOMIC_COUNT)
        )
    if probabilities.shape != (NUM_ATOMIC_SKILLS,):
        raise ValueError(
            f"probabilities must have shape [{NUM_ATOMIC_SKILLS}]"
        )
    if np.any(probabilities < 0) or probabilities.sum() <= 0:
        raise ValueError("probabilities must be non-negative and have positive sum")
    probabilities = probabilities / probabilities.sum()
    order = np.argsort(probabilities)[::-1]
    first, second, third = map(int, order[:3])
    p1, p2, p3 = probabilities[[first, second, third]]

    if p1 >= config.single_p1:
        return AtomicGateDecision(
            "single",
            (first,),
            (1.0,),
            (
                f"label={first}, p1={p1:.4f} >= {config.single_p1:.4f}"
            ),
        )

    opposite_pair = OPPOSITE_ATOMIC_ID.get(first) == second
    contains_stay = STAY_ATOMIC_ID in (first, second)
    dual_ok = (
        p1 + p2 >= config.dual_sum
        and p2 >= config.dual_p2
        and p3 <= config.dual_p3_max
        and not opposite_pair
        and not contains_stay
    )
    if dual_ok:
        total = p1 + p2
        return AtomicGateDecision(
            "dual",
            (first, second),
            (float(p1 / total), float(p2 / total)),
            (
                f"labels=({first},{second}), p1+p2={p1 + p2:.4f} >= "
                f"{config.dual_sum:.4f}; p2={p2:.4f} >= {config.dual_p2:.4f}; "
                f"p3={p3:.4f} <= {config.dual_p3_max:.4f}"
            ),
        )
    return AtomicGateDecision(
        "drop",
        (),
        (),
        (
            f"top_labels=({first},{second},{third}), p1={p1:.4f}, p2={p2:.4f}, "
            f"p3={p3:.4f}; failed single(p1>={config.single_p1:.4f}) and "
            f"dual(p1+p2>={config.dual_sum:.4f}, "
            f"p2>={config.dual_p2:.4f}, p3<={config.dual_p3_max:.4f}, "
            f"opposite_pair={opposite_pair}, contains_stay={contains_stay})"
        ),
    )
