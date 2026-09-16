from __future__ import annotations

import numpy as np


def infer_timestamp_divisor(values: np.ndarray) -> float:
    """Infer seconds/ms/us/ns from a monotonic timestamp sequence."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("timestamps must contain finite values")
    positive = np.diff(values)
    positive = positive[positive > 0]
    median = float(np.median(positive)) if positive.size else 0.0
    # A 1 Hz trace (delta=1) is common and must remain seconds. The recorded
    # CR1 clocks use roughly 8/33 ms, 8e3/33e3 us, or 8e6/33e6 ns deltas.
    if median >= 5e6:
        return 1e9
    if median >= 5e3:
        return 1e6
    if median >= 5.0:
        return 1e3
    return 1.0


def timestamps_in_seconds(
    values: np.ndarray,
    *,
    origin: float | None = None,
    require_strict: bool = False,
) -> np.ndarray:
    """Convert one clock to seconds while optionally preserving a shared absolute origin."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("timestamps must be a non-empty finite 1D array")
    if require_strict and values.size < 2:
        raise ValueError("timestamps require at least two values")
    if require_strict and np.any(np.diff(values) <= 0):
        raise ValueError("timestamps must be unique and increasing")
    zero = float(values[0]) if origin is None else float(origin)
    return (values - zero) / infer_timestamp_divisor(values)
