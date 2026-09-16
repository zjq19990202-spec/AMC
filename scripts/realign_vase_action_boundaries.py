#!/usr/bin/env python3
"""Realign Vase semantic boundaries from commanded 30 Hz actions.

Gripper convention in this corpus is 1=open and 0=closed.  The alignment
rules are intentionally tied to commanded action rather than lagging state:

* the first grasp segment ends once the left gripper is stably closed;
* wipe -> rotate starts at the corresponding right-gripper release onset;
* rotate -> wipe starts after the right gripper is closed, when left-arm
  commanded motion resumes;
* adjacent identical rotate segments are merged.
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


FIRST = "Move to the blackboard eraser, tip it bottom-side up, and grasp it"
WIPE = "Wipe the writing in front of the vase with the blackboard eraser using an up-and-down motion"
ROTATE = "Rotate the vase counterclockwise, then press it down onto the table"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_actions(dataset_root: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    paths = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet below {dataset_root / 'data'}")
    table = pq.read_table(paths, columns=["episode_index", "frame_index", "action"])
    episodes = np.asarray(table["episode_index"], dtype=np.int64)
    frames = np.asarray(table["frame_index"], dtype=np.int64)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    output = {}
    for episode in np.unique(episodes):
        mask = episodes == episode
        order = np.argsort(frames[mask])
        output[int(episode)] = frames[mask][order], actions[mask][order]
    return output


def stable_crossings(values: np.ndarray, rising: bool) -> list[int]:
    """Return sustained relative open/close onsets without assuming 0/1 endpoints."""
    events = []
    for row in range(6, len(values) - 16):
        baseline = float(values[row])
        previous = values[row - 6 : row + 1]
        following = values[row : row + 16]
        if rising:
            future_change = float(np.max(following) - baseline)
            previous_change = float(np.max(previous) - np.min(previous))
            starts_moving = float(values[row + 2] - values[row]) >= 0.004
        else:
            future_change = float(baseline - np.min(following))
            previous_change = float(np.max(previous) - np.min(previous))
            starts_moving = float(values[row] - values[row + 2]) >= 0.004
        # Select the onset of a meaningful command ramp.  A quiet preceding
        # window keeps later points on the same ramp from becoming events.
        if future_change >= 0.04 and previous_change <= 0.035 and starts_moving:
            events.append(row)
    return events


def monotonic_match(
    targets: list[int], events: list[int], max_shift: int = 360
) -> list[int | None]:
    """Greedily match distinct ordered events without making huge jumps."""
    matched: list[int | None] = []
    previous = -1
    for target in targets:
        candidates = [
            event
            for event in events
            if event > previous and abs(event - target) <= max_shift
        ]
        if not candidates:
            matched.append(None)
            continue
        selected = min(candidates, key=lambda event: abs(event - target))
        matched.append(selected)
        previous = selected
    return matched


def left_motion_resume(actions: np.ndarray, close_row: int, stop_row: int) -> tuple[int, bool]:
    """Find left-arm action motion onset at/after a right-gripper closure."""
    velocity = np.r_[0.0, np.linalg.norm(np.diff(actions[:, :7], axis=0), axis=1)]
    stop_row = min(stop_row, len(actions) - 10)
    # Prefer a new bout after a quiet command interval.
    for row in range(close_row, stop_row):
        previous = float(velocity[max(0, row - 8) : row].sum())
        following = float(velocity[row : row + 8].sum())
        if previous <= 0.008 and following >= 0.015:
            return row, True
    # If motion was already ramping as closure completed, the handoff is the
    # closure itself; flag it as weaker evidence for the audit report.
    if float(velocity[close_row : min(stop_row, close_row + 30)].sum()) >= 0.02:
        return close_row, False
    return close_row, False


def merge_adjacent_rotate(segments: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    output = []
    merged = 0
    for segment in segments:
        if output and output[-1]["current_subtask"] == ROTATE == segment["current_subtask"]:
            output[-1]["end_frame_30hz_exclusive"] = segment["end_frame_30hz_exclusive"]
            merged += 1
        else:
            output.append(segment)
    return output, merged


def align_episode(
    episode: dict[str, Any],
    frames: np.ndarray,
    actions: np.ndarray,
    search_radius_frames: int,
) -> list[dict[str, Any]]:
    audit = []
    segments, merged = merge_adjacent_rotate(episode["semantic_segments"])
    episode["semantic_segments"] = segments
    if merged:
        audit.append({"kind": "merge_adjacent_rotate", "count": merged})

    # First grasp: include the close command, then begin the next segment.
    if len(segments) > 1 and segments[0]["current_subtask"] == FIRST:
        old = int(segments[1]["start_frame_30hz"])
        close_rows = stable_crossings(actions[:, 7], rising=False)
        candidates = [
            row
            for row in close_rows
            if abs(int(frames[row]) - old) <= search_radius_frames
        ]
        if candidates:
            row = min(candidates, key=lambda item: abs(int(frames[item]) - old))
            new = int(frames[row])
            segments[0]["end_frame_30hz_exclusive"] = new
            segments[1]["start_frame_30hz"] = new
            audit.append({"kind": "first_left_gripper_closed", "old_frame": old, "new_frame": new})
        else:
            audit.append({"kind": "first_left_gripper_closed", "old_frame": old, "new_frame": None})

    wr_indices = [
        i for i in range(1, len(segments))
        if segments[i - 1]["current_subtask"] == WIPE and segments[i]["current_subtask"] == ROTATE
    ]
    if not wr_indices:
        return audit
    frame_to_row = {int(frame): row for row, frame in enumerate(frames)}
    release_rows = stable_crossings(actions[:, 15], rising=True)
    close_rows = stable_crossings(actions[:, 15], rising=False)
    for index in wr_indices:
        old = int(segments[index]["start_frame_30hz"])
        pair_start = max(
            int(segments[index - 1]["start_frame_30hz"]) + 5,
            old - search_radius_frames,
        )
        pair_end = min(
            int(segments[index]["end_frame_30hz_exclusive"]) - 5,
            old + search_radius_frames,
        )
        candidates = [
            int(frames[row])
            for row in release_rows
            if pair_start <= int(frames[row]) <= pair_end
        ]
        if not candidates:
            audit.append({
                "kind": "wipe_to_rotate_unmatched",
                "segment_index": index,
                "old_frame": old,
            })
            continue
        release_frame = min(candidates, key=lambda event: abs(event - old))
        segments[index - 1]["end_frame_30hz_exclusive"] = release_frame
        segments[index]["start_frame_30hz"] = release_frame
        audit.append({"kind": "wipe_to_rotate_right_release", "segment_index": index, "old_frame": old, "new_frame": release_frame})

        # Only a following rotate->wipe boundary needs the close+left-resume rule.
        if index + 1 >= len(segments) or segments[index + 1]["current_subtask"] != WIPE:
            continue
        rotate_to_wipe_old = int(segments[index + 1]["start_frame_30hz"])
        next_release = min(
            int(segments[index + 1]["end_frame_30hz_exclusive"]) - 5,
            rotate_to_wipe_old + search_radius_frames,
        )
        release_row = frame_to_row[release_frame]
        candidates = [
            row
            for row in close_rows
            if release_row < row
            and rotate_to_wipe_old - search_radius_frames <= int(frames[row]) < next_release
        ]
        if not candidates:
            audit.append({"kind": "rotate_to_wipe_no_right_close", "segment_index": index + 1})
            continue
        close_row = candidates[0]
        stop_row = frame_to_row.get(next_release, len(actions) - 1)
        resume_row, strong = left_motion_resume(actions, close_row, stop_row)
        new = int(frames[resume_row])
        old = int(segments[index + 1]["start_frame_30hz"])
        segments[index]["end_frame_30hz_exclusive"] = new
        segments[index + 1]["start_frame_30hz"] = new
        audit.append({
            "kind": "rotate_to_wipe_right_closed_left_resume",
            "segment_index": index + 1,
            "old_frame": old,
            "right_closed_frame": int(frames[close_row]),
            "new_frame": new,
            "strong_left_resume": strong,
        })
    return audit


def validate(episodes: list[dict[str, Any]]) -> None:
    for episode in episodes:
        segments = episode["semantic_segments"]
        for previous, current in zip(segments, segments[1:], strict=False):
            if int(previous["end_frame_30hz_exclusive"]) != int(current["start_frame_30hz"]):
                raise ValueError(f"episode {episode['episode_index']} has a gap/overlap")
            if int(previous["start_frame_30hz"]) >= int(previous["end_frame_30hz_exclusive"]):
                raise ValueError(f"episode {episode['episode_index']} has an empty segment")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-suffix", default=".before_action_boundary_realign_20260824")
    parser.add_argument("--audit-jsonl", type=Path)
    parser.add_argument("--search-radius-frames", type=int, default=90)
    args = parser.parse_args()
    sidecar = args.sidecar or args.dataset_root / "meta" / "episode_subtasks.jsonl"
    episodes = load_jsonl(sidecar)
    actions = load_actions(args.dataset_root)
    audit = []
    for episode in episodes:
        index = int(episode["episode_index"])
        rows = align_episode(
            episode, *actions[index], search_radius_frames=args.search_radius_frames
        )
        audit.extend({"episode_index": index, "split_from_episode_index": episode.get("split_from_episode_index"), **row} for row in rows)
    validate(episodes)
    audit_path = args.audit_jsonl or sidecar.with_name("vase_action_boundary_realign_audit.jsonl")
    write_jsonl(audit_path, audit)
    before = sha256(sidecar)
    backup = Path(str(sidecar) + args.backup_suffix)
    after = None
    if args.apply:
        if backup.exists():
            raise FileExistsError(f"refusing to overwrite backup: {backup}")
        shutil.copy2(sidecar, backup)
        write_jsonl(sidecar, episodes)
        after = sha256(sidecar)
    kinds = {}
    for row in audit:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    print(json.dumps({"apply": args.apply, "episodes": len(episodes), "events": kinds, "before_sha256": before, "after_sha256": after, "backup": str(backup) if args.apply else None, "audit_jsonl": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
