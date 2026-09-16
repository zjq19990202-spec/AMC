#!/usr/bin/env python3
"""Move Vase first-subtask ends to sustained left-gripper state closure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--radius-frames", type=int, default=90)
    parser.add_argument("--closed-threshold", type=float, default=0.60)
    parser.add_argument("--sustain-frames", type=int, default=30)
    parser.add_argument("--backup-suffix", default=".before_first_grasp_state_closed_20260825")
    parser.add_argument("--audit-jsonl", type=Path)
    args = parser.parse_args()

    sidecar = args.dataset_root / "meta" / "episode_subtasks.jsonl"
    episodes = load_jsonl(sidecar)
    paths = sorted((args.dataset_root / "data").glob("**/*.parquet"))
    table = pq.read_table(paths, columns=["episode_index", "frame_index", "observation.state"])
    episode_ids = np.asarray(table["episode_index"], dtype=np.int64)
    frame_ids = np.asarray(table["frame_index"], dtype=np.int64)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)

    audit = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        mask = episode_ids == episode_index
        order = np.argsort(frame_ids[mask])
        frames = frame_ids[mask][order]
        left_gripper = states[mask][order, 7]
        segments = episode["semantic_segments"]
        old = int(segments[0]["end_frame_30hz_exclusive"])
        start_row = int(np.searchsorted(frames, old))
        candidate = None
        closed_fraction = None
        stop = min(len(frames) - args.sustain_frames, start_row + args.radius_frames + 1)
        for row in range(start_row, stop):
            window = left_gripper[row : row + args.sustain_frames]
            fraction = float(np.mean(window <= args.closed_threshold))
            if left_gripper[row] <= args.closed_threshold and fraction >= 0.90:
                candidate = int(frames[row])
                closed_fraction = fraction
                break
        if candidate is None:
            raise ValueError(f"episode {episode_index}: no sustained closed state near frame {old}")
        if candidate >= int(segments[1]["end_frame_30hz_exclusive"]):
            raise ValueError(f"episode {episode_index}: corrected first boundary crosses segment 2")
        segments[0]["end_frame_30hz_exclusive"] = candidate
        segments[1]["start_frame_30hz"] = candidate
        audit.append(
            {
                "episode_index": episode_index,
                "split_from_episode_index": episode.get("split_from_episode_index"),
                "old_frame": old,
                "new_frame": candidate,
                "shift_frames": candidate - old,
                "left_state_at_boundary": float(left_gripper[np.searchsorted(frames, candidate)]),
                "closed_fraction_next_1s": closed_fraction,
            }
        )

    audit_path = args.audit_jsonl or sidecar.with_name("vase_first_grasp_state_closed_audit.jsonl")
    write_jsonl(audit_path, audit)
    backup = Path(str(sidecar) + args.backup_suffix)
    before = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    after = None
    if args.apply:
        if backup.exists():
            raise FileExistsError(backup)
        shutil.copy2(sidecar, backup)
        write_jsonl(sidecar, episodes)
        after = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    shifts = np.asarray([row["shift_frames"] for row in audit])
    print(
        json.dumps(
            {
                "apply": args.apply,
                "episodes": len(audit),
                "shift_min": int(shifts.min()),
                "shift_median": float(np.median(shifts)),
                "shift_max": int(shifts.max()),
                "before_sha256": before,
                "after_sha256": after,
                "backup": str(backup) if args.apply else None,
                "audit_jsonl": str(audit_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
