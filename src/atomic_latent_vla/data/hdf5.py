from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from atomic_latent_vla.timebase import timestamps_in_seconds


def require_h5py():
    try:
        import h5py
    except (ImportError, ValueError) as exc:
        raise RuntimeError(
            "h5py is unavailable or binary-incompatible with NumPy; reinstall h5py in this environment"
        ) from exc
    return h5py


def first_existing(group: Any, paths: tuple[str, ...]) -> str | None:
    return next((path for path in paths if path in group), None)


def read_policy_timestamps(hdf5_path: str | Path) -> np.ndarray:
    h5py = require_h5py()
    with h5py.File(hdf5_path, "r") as handle:
        path = first_existing(
            handle,
            ("/observations/state_timestamps", "/observations/timestamps", "/timestamps"),
        )
        if path is None:
            actions = handle["/actions"]
            return np.arange(len(actions), dtype=np.float64) / 30.0
        camera_path = first_existing(handle, ("/observations/timestamps", "/timestamps"))
        origin = (
            float(np.asarray(handle[camera_path])[0])
            if camera_path is not None and camera_path != path
            else None
        )
        return timestamps_in_seconds(np.asarray(handle[path]), origin=origin)
