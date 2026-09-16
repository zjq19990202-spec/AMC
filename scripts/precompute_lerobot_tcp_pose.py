#!/usr/bin/env python3
"""Precompute the exact CR1 TCP poses used by FK atomic segmentation.

For every LeRobot row this stores two absolute base-frame poses:

* columns 0:12  -- FK(observation.state right arm)
* columns 12:24 -- FK(action right arm)

Each pose is ``[px, py, pz, flatten(R_3x3)]``.  Training can therefore build
any 50-step relative TCP target without running Pinocchio in DataLoader workers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from atomic_latent_vla.annotation.motion import PinocchioCR1FK, default_mount_xyz
from atomic_latent_vla.tcp import (
    BIMANUAL_TCP_POSE_METADATA,
    BIMANUAL_TCP_POSE_SIDECAR,
    BIMANUAL_TCP_LOCAL_Z_OFFSET_M,
)


SIDECAR_NAME = BIMANUAL_TCP_POSE_SIDECAR
METADATA_NAME = BIMANUAL_TCP_POSE_METADATA


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", action="append", type=Path, required=True)
    parser.add_argument(
        "--urdf",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "assets/cr1_right_arm_tcp_red/urdf/CR1ARMR_TCP_RED.urdf"
        ),
    )
    parser.add_argument("--batch-rows", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _arm(values: np.ndarray, arm: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"state/action rows must be rank two, got {values.shape}")
    if values.shape[1] >= 15:
        return values[:, :7] if arm == "left" else values[:, 8:15]
    if values.shape[1] in (7, 8):
        if arm == "left":
            raise ValueError("a single-arm row cannot provide a left-arm FK target")
        return values[:, :7]
    raise ValueError(f"cannot select the seven right-arm joints from {values.shape}")


def _as_matrix(batch: object, name: str) -> np.ndarray:
    column = batch.column(name)
    values = np.asarray(column.to_pylist(), dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a dense list column, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return values


def _tcp_pose_row(fk: PinocchioCR1FK, q: np.ndarray) -> np.ndarray:
    wrist = fk.pose(q)
    # This is exactly LastLinkOffsetFK(..., TCP_LOCAL_Z_OFFSET_M) used by the
    # dscrew/cabinet atomic segmentation pipeline.
    tcp_position = wrist.translation + wrist.rotation @ np.asarray(
        [0.0, 0.0, BIMANUAL_TCP_LOCAL_Z_OFFSET_M]
    )
    return np.concatenate([tcp_position, wrist.rotation.reshape(-1)])


def precompute(root: Path, urdf: Path, *, batch_rows: int, overwrite: bool) -> Path:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise RuntimeError("precomputation requires pyarrow") from error

    root = root.expanduser().resolve()
    paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet files under {root / 'data'}")
    output = root / "meta" / SIDECAR_NAME
    metadata_path = root / "meta" / METADATA_NAME
    if output.exists() and not overwrite:
        existing_poses = np.load(output, mmap_mode="r")
        if existing_poses.ndim == 2 and existing_poses.shape[1] == 48 and existing_poses.dtype == np.float32:
            print(f"SKIP {root}: existing {existing_poses.shape} {output}", flush=True)
            return output
        raise ValueError(f"existing sidecar is invalid; rerun with --overwrite: {output}")

    total_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    poses = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(total_rows, 48)
    )
    fk_by_arm = {
        arm: PinocchioCR1FK(
            urdf_path=urdf,
            tcp_frame="right_wrist_x_link",
            mount_xyz=default_mount_xyz(arm),
        )
        for arm in ("left", "right")
    }
    cursor = 0
    started = time.monotonic()
    try:
        for path in paths:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(
                batch_size=batch_rows, columns=["observation.state", "action"]
            ):
                all_states = _as_matrix(batch, "observation.state")
                all_actions = _as_matrix(batch, "action")
                if all_states.shape != all_actions.shape:
                    raise ValueError(
                        f"state/action shape mismatch in {path}: {all_states.shape}, {all_actions.shape}"
                    )
                stop = cursor + len(all_states)
                # Layout is left(state, action), right(state, action). The
                # loader returns model order [right, left].
                for arm_index, arm in enumerate(("left", "right")):
                    states = _arm(all_states, arm)
                    actions = _arm(all_actions, arm)
                    offset = arm_index * 24
                    fk = fk_by_arm[arm]
                    for local_index, (state_q, action_q) in enumerate(
                        zip(states, actions, strict=True)
                    ):
                        poses[cursor + local_index, offset : offset + 12] = _tcp_pose_row(fk, state_q)
                        poses[cursor + local_index, offset + 12 : offset + 24] = _tcp_pose_row(fk, action_q)
                cursor = stop
                if cursor % 50_000 < len(states) or cursor == total_rows:
                    elapsed = max(time.monotonic() - started, 1e-6)
                    print(
                        f"{root.name}: {cursor}/{total_rows} ({cursor / elapsed:.0f} rows/s)",
                        flush=True,
                    )
        if cursor != total_rows:
            raise RuntimeError(f"wrote {cursor} rows but parquet metadata reports {total_rows}")
        poses.flush()
        del poses
        poses = None
        os.replace(temporary, output)
    except BaseException:
        try:
            if poses is not None:
                del poses
        finally:
            temporary.unlink(missing_ok=True)
        raise

    metadata = {
        "schema": "atomic_latent_vla.tcp_pose.v1",
        "rows": total_rows,
        "dtype": "float32",
        "layout": {
            "left_state_pose": "columns[0:12]",
            "left_action_pose": "columns[12:24]",
            "right_state_pose": "columns[24:36]",
            "right_action_pose": "columns[36:48]",
            "pose": "[position_base(3), rotation_base_row_major(9)]",
        },
        "arms": ["left", "right"],
        "mount_xyz_m": {arm: list(default_mount_xyz(arm)) for arm in ("left", "right")},
        "source_frame": "right_wrist_x_link",
        "tcp_local_offset_m": [0.0, 0.0, BIMANUAL_TCP_LOCAL_Z_OFFSET_M],
        "urdf": str(urdf.expanduser().resolve()),
        "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"DONE {root}: {output}", flush=True)
    return output


def main() -> None:
    args = _parser().parse_args()
    if args.batch_rows <= 0:
        raise ValueError("--batch-rows must be positive")
    urdf = args.urdf.expanduser().resolve()
    if not urdf.is_file():
        raise FileNotFoundError(urdf)
    for root in args.dataset_root:
        precompute(root, urdf, batch_rows=args.batch_rows, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
