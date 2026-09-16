#!/usr/bin/env python3
"""Fast OpenPI-style statistics for precomputed TCP relative-delta chunks.

This mirrors ``checkrecord-dataset-ops-bundle/scripts/
compute_openpi_official_equiv_fast.py``: it uses every valid 50-step chunk,
clips a chunk at an episode boundary, and writes mean/std/q01/q99.  The
important difference is that the values come from the precomputed FK sidecar
selected by the shared TCP contract rather than from joint actions.

Each sidecar row stores ``[current TCP pose, action TCP pose]`` as
``[xyz, R(3x3), xyz, R(3x3)]``.  For a chunk starting at t, every target is
the base-frame relative pose from current TCP(t) to action TCP(t+h):
``[dx, dy, dz, rotvec_x, rotvec_y, rotvec_z]``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from atomic_latent_vla.tcp import BIMANUAL_TCP_POSE_SIDECAR


SIDECAR = BIMANUAL_TCP_POSE_SIDECAR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--action-horizon", default=50, type=int)
    parser.add_argument("--batch-size", default=4096, type=int)
    return parser.parse_args()


def episode_ranges(root: Path) -> list[tuple[int, int]]:
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet data under {root / 'data'}")
    episode = np.concatenate(
        [
            np.asarray(pq.read_table(path, columns=["episode_index"])["episode_index"], dtype=np.int64)
            for path in files
        ]
    )
    boundaries = np.flatnonzero(np.diff(episode) != 0) + 1
    starts = np.concatenate([np.asarray([0]), boundaries])
    ends = np.concatenate([boundaries, np.asarray([len(episode)])])
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _arm_delta(
    poses: np.ndarray,
    starts: np.ndarray,
    future_indices: np.ndarray,
    *,
    state_offset: int,
    action_offset: int,
) -> np.ndarray:
    """Vectorized base-frame [B,H,6] deltas for one arm."""

    horizon = future_indices.shape[1]
    future = poses[future_indices, action_offset : action_offset + 12].reshape(
        len(starts), horizon, 4, 3
    )
    current = poses[starts, state_offset : state_offset + 12].reshape(len(starts), 4, 3)
    translation = future[..., 0, :] - current[:, None, 0, :]
    r_future = future[..., 1:, :]
    r_current_t = np.swapaxes(current[:, 1:, :], -1, -2)
    relative = r_future @ r_current_t[:, None, :, :]
    rotation = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(
        len(starts), horizon, 3
    )
    return np.concatenate([translation, rotation], axis=-1).astype(np.float32, copy=False)


def tcp_delta_chunks(poses: np.ndarray, start: np.ndarray, horizon: int) -> np.ndarray:
    """Vectorized coordinated [B,H,12] target in [right, left] order."""

    future = start[:, None] + np.arange(horizon, dtype=np.int64)[None, :]
    right = _arm_delta(poses, start, future, state_offset=24, action_offset=36)
    left = _arm_delta(poses, start, future, state_offset=0, action_offset=12)
    return np.concatenate([right, left], axis=-1)


def main() -> None:
    args = parse_args()
    if args.action_horizon <= 0 or args.batch_size <= 0:
        raise ValueError("action-horizon and batch-size must be positive")

    # Same sufficient statistics and approximate streaming q01/q99 mechanism
    # used by OpenPI's official-equivalence fast script.
    from openpi.shared.normalize import RunningStats

    stats = RunningStats()
    frames = chunks = 0
    roots: list[str] = []
    for root in args.dataset_root:
        root = root.expanduser().resolve()
        sidecar_path = root / "meta" / SIDECAR
        if not sidecar_path.is_file():
            raise FileNotFoundError(f"missing FK TCP sidecar: {sidecar_path}")
        poses = np.load(sidecar_path, mmap_mode="r")
        if poses.ndim != 2 or poses.shape[1] != 48:
            raise ValueError(f"{sidecar_path}: expected [N,48], got {poses.shape}")
        ranges = episode_ranges(root)
        if ranges and ranges[-1][1] != len(poses):
            raise ValueError(f"{root}: parquet/sidecar frame count disagree ({ranges[-1][1]} vs {len(poses)})")
        roots.append(str(root))
        for episode_start, episode_end in ranges:
            frames += episode_end - episode_start
            for batch_start in range(episode_start, episode_end, args.batch_size):
                starts = np.arange(batch_start, min(batch_start + args.batch_size, episode_end), dtype=np.int64)
                # Exact π0.5/OpenPI end-of-episode behavior: repeat the last
                # valid action observation instead of crossing an episode.
                ends = np.minimum(starts[:, None] + np.arange(args.action_horizon), episode_end - 1)
                # ``tcp_delta_chunks`` receives consecutive rows; its custom
                # index path accepts the clipped future index matrix below.
                right = _arm_delta(
                    poses, starts, ends, state_offset=24, action_offset=36
                )
                left = _arm_delta(
                    poses, starts, ends, state_offset=0, action_offset=12
                )
                stats.update(np.concatenate([right, left], axis=-1))
                chunks += len(starts)
    result = stats.get_statistics()
    payload = {
        "norm_stats": {
            "tcp_twist_delta": {
                "mean": result.mean.tolist(), "std": result.std.tolist(),
                "q01": result.q01.tolist(), "q99": result.q99.tolist(),
            }
        },
        "_meta": {
            "dataset_roots": roots, "frames": frames, "chunks": chunks,
            "action_horizon": args.action_horizon,
            "sidecar": SIDECAR,
            "normalization": "2 * relative_delta / (q99 - q01)",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.output}")
    print(json.dumps(payload["_meta"], indent=2))
    print(json.dumps(payload["norm_stats"]["tcp_twist_delta"], indent=2))


if __name__ == "__main__":
    main()
