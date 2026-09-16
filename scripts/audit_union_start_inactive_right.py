#!/usr/bin/env python3
"""Audit unexpected right-arm motion at episode starts in Atomic datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads


def _expand_stay_blocks(path: Path) -> dict[int, set[int]]:
    blocks: dict[int, set[int]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("arm") != "right" or row.get("fk_atomic_labels") != ["stay"]:
                continue
            episode = int(row["episode_index"])
            start = int(row["block_start_id"])
            end = int(row["block_end_id"])
            blocks.setdefault(episode, set()).update(range(start, end + 1))
    return blocks


def _subtasks(path: Path) -> dict[int, list[dict]]:
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            result[int(row["episode_index"])] = row.get("semantic_segments", [])
    return result


def _subtask_at(segments: list[dict], frame: int) -> str:
    for segment in segments:
        if int(segment["start_frame_30hz"]) <= frame < int(segment["end_frame_30hz_exclusive"]):
            return str(segment.get("current_subtask", ""))
    return ""


def _runs(frames: np.ndarray, expansion: int = 0) -> list[list[int]]:
    intervals: list[list[int]] = []
    for value in frames.tolist():
        start, end = int(value), int(value) + expansion
        if intervals and start <= intervals[-1][1] + 1:
            intervals[-1][1] = max(intervals[-1][1], end)
        else:
            intervals.append([start, end])
    return intervals


def audit(root: Path, start_seconds: float, threshold: float, horizon: int) -> dict:
    fps = int(json.loads((root / "meta/info.json").read_text())["fps"])
    stay = _expand_stay_blocks(root / "meta/atomic_horizon_prompts_3hz.jsonl")
    subtasks = _subtasks(root / "meta/episode_subtasks.jsonl")
    table = pads.dataset(str(root / "data"), format="parquet").to_table(
        columns=["episode_index", "frame_index", "action"]
    )
    episode = np.asarray(table["episode_index"])
    frame = np.asarray(table["frame_index"])
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    changes = np.flatnonzero(np.diff(episode)) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, len(episode)]
    records = []
    for lo, hi in zip(starts, ends, strict=True):
        ep = int(episode[lo])
        ep_frame = frame[lo:hi]
        ep_action = action[lo:hi]
        candidates = []
        max_start = min(int(start_seconds * fps), len(ep_frame) - horizon + 1)
        for local in range(max(0, max_start)):
            f = int(ep_frame[local])
            if f // 10 not in stay.get(ep, set()):
                continue
            if int(ep_frame[local + horizon - 1]) != f + horizon - 1:
                continue
            delta = float(
                np.linalg.norm(
                    ep_action[local + horizon - 1, 8:15] - ep_action[local, 8:15]
                )
            )
            if delta > threshold:
                candidates.append((f, delta))
        if not candidates:
            continue
        bad_frames = np.asarray([item[0] for item in candidates], dtype=np.int64)
        start_runs = _runs(bad_frames)
        coverage_runs = _runs(bad_frames, horizon - 1)
        records.append(
            {
                "episode_index": ep,
                "bad_horizon_starts": len(candidates),
                "max_right_delta_rad": max(item[1] for item in candidates),
                "subtasks": sorted({_subtask_at(subtasks.get(ep, []), int(f)) for f in bad_frames}),
                "bad_start_intervals_frames": start_runs,
                "covered_intervals_frames": coverage_runs,
                "bad_start_seconds": sum(b - a + 1 for a, b in start_runs) / fps,
                "covered_seconds": sum(b - a + 1 for a, b in coverage_runs) / fps,
            }
        )
    return {
        "dataset_root": str(root),
        "episodes_total": int(len(starts)),
        "episodes_flagged": len(records),
        "bad_horizon_starts": sum(r["bad_horizon_starts"] for r in records),
        "bad_start_seconds": sum(r["bad_start_seconds"] for r in records),
        "covered_seconds": sum(r["covered_seconds"] for r in records),
        "threshold_rad": threshold,
        "horizon_steps": horizon,
        "start_window_seconds": start_seconds,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seconds", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--horizon", type=int, default=50)
    args = parser.parse_args()
    reports = [
        audit(root, args.start_seconds, args.threshold, args.horizon)
        for root in args.dataset_root
    ]
    payload = {
        "contract": {
            "right_arm_slice": "action[8:15]",
            "right_inactive_source": "right atomic label exactly [stay]",
            "fps": 30,
        },
        "datasets": reports,
        "summary": {
            "episodes_total": sum(r["episodes_total"] for r in reports),
            "episodes_flagged": sum(r["episodes_flagged"] for r in reports),
            "bad_horizon_starts": sum(r["bad_horizon_starts"] for r in reports),
            "bad_start_seconds": sum(r["bad_start_seconds"] for r in reports),
            "covered_seconds": sum(r["covered_seconds"] for r in reports),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload["summary"], indent=2))
    for report in reports:
        print(Path(report["dataset_root"]).name, report["episodes_flagged"], report["covered_seconds"])


if __name__ == "__main__":
    main()
