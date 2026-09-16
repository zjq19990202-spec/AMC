#!/usr/bin/env python3
"""Find arm-neutral fruit SUBtask horizons with left-active/right-still GT motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds


def _numeric(column: pa.ChunkedArray, dtype: np.dtype) -> np.ndarray:
    array = column.combine_chunks()
    if pa.types.is_fixed_size_list(array.type):
        return np.asarray(array.values.to_numpy(), dtype=dtype).reshape(
            len(array), array.type.list_size
        )
    return np.asarray(array.to_numpy(), dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--minimum-left-path-deg", type=float, default=8.0)
    parser.add_argument("--maximum-right-path-deg", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = ds.dataset(str(args.dataset_root / "data"), format="parquet")
    table = dataset.to_table(
        columns=["episode_index", "frame_index", "observation.state", "action"],
        filter=ds.field("source_dataset") == args.source_dataset,
    )
    episodes = _numeric(table["episode_index"], np.dtype(np.int64))
    frames = _numeric(table["frame_index"], np.dtype(np.int64))
    states = _numeric(table["observation.state"], np.dtype(np.float32))
    actions = _numeric(table["action"], np.dtype(np.float32))

    bounds = {}
    changes = np.flatnonzero(np.diff(episodes)) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, len(episodes)]
    for start, end in zip(starts, ends, strict=True):
        bounds[int(episodes[start])] = (int(start), int(end))

    segments = []
    with (args.dataset_root / "meta" / "episode_subtasks.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            episode = int(row["episode_index"])
            if row.get("source_dataset") != args.source_dataset or episode not in bounds:
                continue
            for segment in row["semantic_segments"]:
                prompt = segment["current_subtask"]
                if not re.search(
                    r"(move|bring) the gripper toward .* and grasp it", prompt, re.I
                ):
                    continue
                segments.append(
                    (
                        episode,
                        int(segment["start_frame_30hz"]),
                        int(segment["end_frame_30hz_exclusive"]),
                        prompt,
                    )
                )

    candidates = []
    for episode, segment_start, segment_end, prompt in segments:
        lower, upper = bounds[episode]
        episode_frames = frames[lower:upper]
        episode_states = states[lower:upper]
        episode_actions = actions[lower:upper]
        frame_to_local = {int(value): index for index, value in enumerate(episode_frames)}
        final_frame = min(segment_end - args.horizon, int(episode_frames[-1]) - args.horizon)
        for frame in range(segment_start, final_frame + 1, args.frame_stride):
            index = frame_to_local.get(frame)
            if index is None or index + args.horizon > len(episode_actions):
                continue
            horizon = episode_actions[index : index + args.horizon]
            left = np.vstack((episode_states[index, :7], horizon[:, :7]))
            right = np.vstack((episode_states[index, 8:15], horizon[:, 8:15]))
            left_path = float(np.linalg.norm(np.diff(left, axis=0), axis=1).sum())
            right_path = float(np.linalg.norm(np.diff(right, axis=0), axis=1).sum())
            candidates.append(
                {
                    "episode": episode,
                    "frame": frame,
                    "prompt": prompt,
                    "left_path_deg": float(np.degrees(left_path)),
                    "right_path_deg": float(np.degrees(right_path)),
                    "left_endpoint_deg": float(np.degrees(np.linalg.norm(left[-1] - left[0]))),
                    "right_endpoint_deg": float(np.degrees(np.linalg.norm(right[-1] - right[0]))),
                    "left_right_path_ratio": left_path / (right_path + 1e-8),
                }
            )

    accepted = [
        row
        for row in candidates
        if row["left_path_deg"] >= args.minimum_left_path_deg
        and row["right_path_deg"] <= args.maximum_right_path_deg
    ]
    accepted.sort(key=lambda row: row["left_right_path_ratio"], reverse=True)
    selected = []
    seen = set()
    for row in accepted:
        target = row["prompt"].lower().split("toward ", 1)[1].split(" and grasp", 1)[0]
        key = (row["episode"], target)
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) >= args.limit:
            break

    report = {
        "dataset_root": str(args.dataset_root),
        "source_dataset": args.source_dataset,
        "horizon": args.horizon,
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
