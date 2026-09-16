#!/usr/bin/env python3
"""Realign Vase wipe->rotate boundaries to the physical arm handoff.

The old Vase annotations were initialized from an even temporal split.  For a
wipe->rotate transition, the observable handoff is:

1. the left arm finishes the wipe and becomes quiet;
2. the right gripper starts opening/releasing;
3. the right arm takes over the vase motion.

This script only moves a boundary forward when that sequence is supported by
the recorded 30 Hz joint/gripper state.  Ambiguous boundaries are left intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pyarrow.parquet as pq


WIPE = "Wipe the writing in front of the vase with the blackboard eraser using an up-and-down motion"
ROTATE = "Rotate the vase counterclockwise, then press it down onto the table"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_states(dataset_root: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    parquet_paths = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"no parquet files below {dataset_root / 'data'}")
    table = pq.read_table(
        parquet_paths,
        columns=["episode_index", "frame_index", "observation.state"],
    )
    episodes = np.asarray(table["episode_index"], dtype=np.int64)
    frames = np.asarray(table["frame_index"], dtype=np.int64)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for episode in np.unique(episodes):
        mask = episodes == episode
        order = np.argsort(frames[mask])
        result[int(episode)] = (frames[mask][order], states[mask][order])
    return result


def _path_sum(states: np.ndarray, start: int, stop: int, joint_slice: slice) -> float:
    start = max(0, start)
    stop = min(len(states), stop)
    if stop - start < 2:
        return 0.0
    delta = np.diff(states[start:stop, joint_slice], axis=0)
    return float(np.linalg.norm(delta, axis=1).sum())


def _candidate_boundary(
    frames: np.ndarray,
    states: np.ndarray,
    old_frame: int,
    segment_end: int,
    max_shift_frames: int,
) -> tuple[int | None, dict[str, Any]]:
    frame_to_row = {int(frame): row for row, frame in enumerate(frames)}
    if old_frame not in frame_to_row:
        return None, {"reason": "old_frame_missing"}

    old_row = frame_to_row[old_frame]
    last_frame = min(segment_end - 31, old_frame + max_shift_frames)
    candidates: list[tuple[float, int, dict[str, float]]] = []

    for frame in range(old_frame, last_frame + 1):
        row = frame_to_row.get(frame)
        row5 = frame_to_row.get(frame + 5)
        row10 = frame_to_row.get(frame + 10)
        row15 = frame_to_row.get(frame + 15)
        row30 = frame_to_row.get(frame + 30)
        if row is None or row5 is None or row10 is None or row15 is None or row30 is None:
            continue

        # The left arm must already be quiet around the proposed handoff.
        left_prev = _path_sum(states, row - 15, row + 1, slice(0, 7))
        left_next = _path_sum(states, row, row30 + 1, slice(0, 7))
        right_next = _path_sum(states, row, row30 + 1, slice(8, 15))
        grip5 = float(states[row5, 15] - states[row, 15])
        grip10 = float(states[row10, 15] - states[row, 15])
        grip15 = float(states[row15, 15] - states[row, 15])
        grip30 = float(states[row30, 15] - states[row, 15])

        # Use a near-term rise to mark the actual release onset.  A 30-frame
        # look-ahead alone would place the boundary before the gripper moves.
        gripper_release = (grip5 >= 0.01 or grip10 >= 0.02) and grip30 >= 0.03
        left_quiet = left_next <= 0.08 and left_prev <= 0.12
        right_takeover = right_next >= 0.04
        if not (gripper_release and left_quiet and right_takeover):
            continue

        # Prefer the earliest reliable onset.  The score only breaks ties among
        # nearby frames and rewards a strong release/takeover signal.
        score = frame - old_frame - 10.0 * max(grip15, grip30) - right_next
        candidates.append(
            (
                score,
                frame,
                {
                    "left_prev_0p5s": left_prev,
                    "left_next_1s": left_next,
                    "right_next_1s": right_next,
                    "right_gripper_delta_0p167s": grip5,
                    "right_gripper_delta_0p333s": grip10,
                    "right_gripper_delta_0p5s": grip15,
                    "right_gripper_delta_1s": grip30,
                },
            )
        )

    if not candidates:
        return None, {"reason": "no_reliable_handoff"}

    # Select the first valid onset, allowing the score to choose only within a
    # three-frame neighborhood so a later stronger event cannot skip a handoff.
    earliest = min(item[1] for item in candidates)
    local = [item for item in candidates if item[1] <= earliest + 3]
    _, frame, evidence = min(local, key=lambda item: item[0])
    evidence["reason"] = "left_quiet_then_right_gripper_release"
    return frame, evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-suffix", default=".before_vase_handoff_realign_20260824")
    parser.add_argument("--audit-jsonl", type=Path)
    parser.add_argument(
        "--propagate-sidecar",
        type=Path,
        help="optional merged/source sidecar addressed by split_from_episode_index",
    )
    parser.add_argument("--max-shift-frames", type=int, default=300)
    parser.add_argument("--minimum-shift-frames", type=int, default=6)
    args = parser.parse_args()

    sidecar = args.dataset_root / "meta" / "episode_subtasks.jsonl"
    before_sha = _sha256(sidecar)
    with sidecar.open() as stream:
        episodes = [json.loads(line) for line in stream if line.strip()]
    states_by_episode = _load_states(args.dataset_root)

    audit: list[dict[str, Any]] = []
    changed = 0
    examined = 0
    shifts: list[int] = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        if episode_index not in states_by_episode:
            raise KeyError(f"episode {episode_index} missing from parquet")
        frames, states = states_by_episode[episode_index]
        segments = episode["semantic_segments"]
        for index in range(1, len(segments)):
            previous = segments[index - 1]
            current = segments[index]
            if previous["current_subtask"] != WIPE or current["current_subtask"] != ROTATE:
                continue
            examined += 1
            old = int(current["start_frame_30hz"])
            candidate, evidence = _candidate_boundary(
                frames,
                states,
                old,
                int(current["end_frame_30hz_exclusive"]),
                args.max_shift_frames,
            )
            record: dict[str, Any] = {
                "episode_index": episode_index,
                "split_from_episode_index": episode.get("split_from_episode_index"),
                "segment_index": index,
                "old_frame": old,
                "candidate_frame": candidate,
                **evidence,
            }
            if candidate is not None and candidate - old >= args.minimum_shift_frames:
                previous["end_frame_30hz_exclusive"] = candidate
                current["start_frame_30hz"] = candidate
                changed += 1
                shifts.append(candidate - old)
                record["applied"] = True
            else:
                record["applied"] = False
            audit.append(record)

    audit_path = args.audit_jsonl or sidecar.with_name("vase_wipe_rotate_handoff_realign_audit.jsonl")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w") as stream:
        for record in audit:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    after_sha = None
    backup = Path(str(sidecar) + args.backup_suffix)
    propagated_before_sha = None
    propagated_after_sha = None
    propagated_backup = None
    propagated_episodes: list[dict[str, Any]] | None = None
    if args.propagate_sidecar is not None:
        propagated_before_sha = _sha256(args.propagate_sidecar)
        with args.propagate_sidecar.open() as stream:
            propagated_episodes = [json.loads(line) for line in stream if line.strip()]
        propagated_by_index = {int(item["episode_index"]): item for item in propagated_episodes}
        for record in audit:
            if not record["applied"]:
                continue
            source_index = record["split_from_episode_index"]
            if source_index is None or int(source_index) not in propagated_by_index:
                raise KeyError(f"source episode missing for audit record: {record}")
            source = propagated_by_index[int(source_index)]
            index = int(record["segment_index"])
            previous = source["semantic_segments"][index - 1]
            current = source["semantic_segments"][index]
            if previous["current_subtask"] != WIPE or current["current_subtask"] != ROTATE:
                raise ValueError(f"source segment text mismatch: {record}")
            if int(current["start_frame_30hz"]) != int(record["old_frame"]):
                raise ValueError(f"source boundary mismatch: {record}")
            candidate = int(record["candidate_frame"])
            previous["end_frame_30hz_exclusive"] = candidate
            current["start_frame_30hz"] = candidate
        propagated_backup = Path(str(args.propagate_sidecar) + args.backup_suffix)

    if args.apply:
        if backup.exists():
            raise FileExistsError(f"refusing to overwrite backup: {backup}")
        if propagated_backup is not None and propagated_backup.exists():
            raise FileExistsError(f"refusing to overwrite backup: {propagated_backup}")
        shutil.copy2(sidecar, backup)
        if args.propagate_sidecar is not None and propagated_backup is not None:
            shutil.copy2(args.propagate_sidecar, propagated_backup)
        temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
        with temporary.open("w") as stream:
            for episode in episodes:
                stream.write(json.dumps(episode, ensure_ascii=False) + "\n")
        temporary.replace(sidecar)
        after_sha = _sha256(sidecar)
        if args.propagate_sidecar is not None and propagated_episodes is not None:
            propagated_temporary = args.propagate_sidecar.with_suffix(
                args.propagate_sidecar.suffix + ".tmp"
            )
            with propagated_temporary.open("w") as stream:
                for episode in propagated_episodes:
                    stream.write(json.dumps(episode, ensure_ascii=False) + "\n")
            propagated_temporary.replace(args.propagate_sidecar)
            propagated_after_sha = _sha256(args.propagate_sidecar)

    summary = {
        "dataset_root": str(args.dataset_root),
        "sidecar": str(sidecar),
        "before_sha256": before_sha,
        "after_sha256": after_sha,
        "apply": args.apply,
        "examined_boundaries": examined,
        "changed_boundaries": changed,
        "shift_frames_min": min(shifts) if shifts else None,
        "shift_frames_median": float(np.median(shifts)) if shifts else None,
        "shift_frames_max": max(shifts) if shifts else None,
        "audit_jsonl": str(audit_path),
        "backup": str(backup) if args.apply else None,
        "propagate_sidecar": str(args.propagate_sidecar) if args.propagate_sidecar else None,
        "propagated_before_sha256": propagated_before_sha,
        "propagated_after_sha256": propagated_after_sha,
        "propagated_backup": str(propagated_backup) if args.apply and propagated_backup else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
