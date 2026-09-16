#!/usr/bin/env python3
"""Build model-independent, publication-facing GT horizon selections.

Rows are selected from native semantic segments and recorded actions only.  No
checkpoint is loaded, which prevents candidate-model performance from leaking
into the selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds


JOINT_SLICES = {"left": slice(0, 7), "right": slice(8, 15)}


def _numeric(column: pa.ChunkedArray, dtype: np.dtype) -> np.ndarray:
    array = column.combine_chunks()
    if pa.types.is_fixed_size_list(array.type):
        return np.asarray(array.values.to_numpy(), dtype=dtype).reshape(
            len(array), array.type.list_size
        )
    return np.asarray(array.to_numpy(), dtype=dtype)


def _parse_sources(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        name, count = value.rsplit("=", 1)
        if not name or name in result or int(count) <= 0:
            raise ValueError(f"invalid source specification: {value!r}")
        result[name] = int(count)
    return result


def _parse_target_progress(values: list[str]) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for value in values:
        target, interval = value.rsplit("=", 1)
        lower_text, upper_text = interval.split(":", 1)
        lower, upper = float(lower_text), float(upper_text)
        if not target or target in result or not 0.0 <= lower <= upper <= 1.0:
            raise ValueError(f"invalid fruit target progress specification: {value!r}")
        result[target] = (lower, upper)
    return result


def _stable_rng(seed: int, value: str) -> np.random.Generator:
    digest = hashlib.sha256(value.encode()).digest()
    folded = seed ^ int.from_bytes(digest[:8], "little")
    return np.random.default_rng(folded)


def _target_in_prompt(prompt: str, targets: list[str]) -> str | None:
    matches = [target for target in targets if target.lower() in prompt.lower()]
    if not matches:
        return None
    return max(matches, key=len)


def _episode_candidates(
    metadata: dict[str, Any],
    frames: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    *,
    horizon: int,
    stride: int,
    fruit_targets: list[str],
) -> list[dict[str, Any]]:
    frame_to_local = {int(frame): index for index, frame in enumerate(frames)}
    candidates: list[dict[str, Any]] = []
    episode_end = max(
        int(segment["end_frame_30hz_exclusive"])
        for segment in metadata["semantic_segments"]
    )
    for segment in metadata["semantic_segments"]:
        start = int(segment["start_frame_30hz"])
        end = int(segment["end_frame_30hz_exclusive"])
        prompt = str(segment["current_subtask"]).strip()
        target = _target_in_prompt(prompt, fruit_targets) if fruit_targets else None
        for frame in frames:
            frame = int(frame)
            if frame < start or frame + horizon > end or frame % stride:
                continue
            local = frame_to_local[frame]
            if local + horizon > len(actions):
                continue
            expected = np.arange(frame, frame + horizon, dtype=np.int64)
            if not np.array_equal(frames[local : local + horizon], expected):
                continue
            horizon_actions = actions[local : local + horizon]
            paths: dict[str, float] = {}
            endpoints: dict[str, float] = {}
            for arm, joint_slice in JOINT_SLICES.items():
                trajectory = np.vstack((states[local, joint_slice], horizon_actions[:, joint_slice]))
                paths[arm] = float(
                    np.degrees(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum())
                )
                endpoints[arm] = float(
                    np.degrees(np.linalg.norm(trajectory[-1] - trajectory[0]))
                )
            active_arm = max(paths, key=paths.__getitem__)
            passive_arm = "right" if active_arm == "left" else "left"
            candidates.append(
                {
                    "episode": int(metadata["episode_index"]),
                    "frame": frame,
                    "prompt": prompt,
                    "prompt_provenance": "native episode_subtasks.jsonl semantic segment",
                    "source_dataset": metadata["source_dataset"],
                    "source_episode": metadata.get("source_episode"),
                    "episode_task": metadata.get("episode_task"),
                    "segment_id": int(segment["subtask_id"]),
                    "segment_start": start,
                    "segment_end_exclusive": end,
                    "episode_progress": start / episode_end,
                    "target": target,
                    "active_arm": active_arm,
                    "left_gt_path_deg": paths["left"],
                    "right_gt_path_deg": paths["right"],
                    "left_gt_endpoint_deg": endpoints["left"],
                    "right_gt_endpoint_deg": endpoints["right"],
                    "dominant_gt_path_deg": paths[active_arm],
                    "arm_path_ratio": paths[active_arm] / (paths[passive_arm] + 1e-8),
                    "selection_basis": "complete in-segment horizon; ranked only by recorded GT joint path",
                }
            )
    return candidates


def _best_per_segment(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["episode"], row["segment_id"])].append(row)
    return [
        max(values, key=lambda item: item["dominant_gt_path_deg"])
        for values in grouped.values()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default="target2058")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="SOURCE=EPISODE_COUNT; repeated for task-stratified selection",
    )
    parser.add_argument("--frames-per-episode", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--minimum-dominant-path-deg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--fruit-target",
        action="append",
        default=[],
        help="When supplied, select one diverse native-prompt row per target.",
    )
    parser.add_argument("--fruit-max-per-episode", type=int, default=2)
    parser.add_argument("--fruit-min-left", type=int, default=2)
    parser.add_argument(
        "--fruit-max-segment-id",
        type=int,
        default=None,
        help=(
            "Optional inclusive semantic-segment ceiling for fruit selection. "
            "Use 0 to evaluate the first pick while the source scene is still full."
        ),
    )
    parser.add_argument(
        "--fruit-min-episode-progress",
        type=float,
        default=None,
        help="Optional minimum segment-start / episode-end ratio for fruit rows.",
    )
    parser.add_argument(
        "--fruit-max-episode-progress",
        type=float,
        default=None,
        help="Optional maximum segment-start / episode-end ratio for fruit rows.",
    )
    parser.add_argument(
        "--fruit-target-progress",
        action="append",
        default=[],
        metavar="TARGET=MIN:MAX",
        help=(
            "Optional per-target episode-progress interval. Repeat to build a "
            "checkpoint-independent early/middle-stage mixture."
        ),
    )
    parser.add_argument(
        "--fruit-prompt-regex",
        default=None,
        help="Optional native-prompt regex, e.g. '(?i)(grasp|take hold)'.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target_progress = _parse_target_progress(args.fruit_target_progress)
    unknown_progress_targets = set(target_progress) - set(args.fruit_target)
    if unknown_progress_targets:
        raise ValueError(
            "fruit target progress specified for unrequested targets: "
            f"{sorted(unknown_progress_targets)}"
        )

    sources = _parse_sources(args.source)
    metadata_rows = [
        json.loads(line)
        for line in (args.dataset_root / "meta" / "episode_subtasks.jsonl").open()
    ]
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metadata_rows:
        if row.get("source_dataset") in sources:
            by_source[row["source_dataset"]].append(row)
    missing = set(sources) - set(by_source)
    if missing:
        raise ValueError(f"unknown or empty source datasets: {sorted(missing)}")

    if args.fruit_target:
        candidate_metadata = [row for source in sources for row in by_source[source]]
    else:
        candidate_metadata = []
        for source, requested in sources.items():
            eligible = [
                row
                for row in by_source[source]
                if sum(
                    int(segment["end_frame_30hz_exclusive"])
                    - int(segment["start_frame_30hz"])
                    >= args.horizon
                    for segment in row["semantic_segments"]
                )
                >= args.frames_per_episode
            ]
            order = _stable_rng(args.seed, source).permutation(len(eligible))
            candidate_metadata.extend(
                eligible[index] for index in order[: max(requested * 4, requested)]
            )

    candidate_ids = sorted({int(row["episode_index"]) for row in candidate_metadata})
    table = ds.dataset(str(args.dataset_root / "data"), format="parquet").to_table(
        columns=["episode_index", "frame_index", "observation.state", "action"],
        filter=ds.field("episode_index").isin(candidate_ids),
    )
    episodes = _numeric(table["episode_index"], np.dtype(np.int64))
    frames = _numeric(table["frame_index"], np.dtype(np.int64))
    states = _numeric(table["observation.state"], np.dtype(np.float32))
    actions = _numeric(table["action"], np.dtype(np.float32))
    arrays: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for episode in candidate_ids:
        mask = episodes == episode
        order = np.argsort(frames[mask])
        arrays[episode] = (frames[mask][order], states[mask][order], actions[mask][order])

    all_candidates = []
    for metadata in candidate_metadata:
        episode = int(metadata["episode_index"])
        all_candidates.extend(
            _episode_candidates(
                metadata,
                *arrays[episode],
                horizon=args.horizon,
                stride=args.frame_stride,
                fruit_targets=args.fruit_target,
            )
        )
    all_candidates = [
        row
        for row in all_candidates
        if row["dominant_gt_path_deg"] >= args.minimum_dominant_path_deg
    ]
    if args.fruit_target and args.fruit_max_segment_id is not None:
        all_candidates = [
            row for row in all_candidates if row["segment_id"] <= args.fruit_max_segment_id
        ]
    if args.fruit_target and args.fruit_min_episode_progress is not None:
        all_candidates = [
            row
            for row in all_candidates
            if row["episode_progress"] >= args.fruit_min_episode_progress
        ]
    if args.fruit_target and args.fruit_max_episode_progress is not None:
        all_candidates = [
            row
            for row in all_candidates
            if row["episode_progress"] <= args.fruit_max_episode_progress
        ]
    segment_best = _best_per_segment(all_candidates)
    if args.fruit_target and args.fruit_prompt_regex:
        pattern = re.compile(args.fruit_prompt_regex)
        segment_best = [row for row in segment_best if pattern.search(row["prompt"])]

    selected: list[dict[str, Any]] = []
    if args.fruit_target:
        episode_counts: Counter[int] = Counter()
        for target in args.fruit_target:
            lower, upper = target_progress.get(target, (0.0, 1.0))
            choices = sorted(
                (
                    row
                    for row in segment_best
                    if row["target"] == target
                    and lower <= row["episode_progress"] <= upper
                ),
                key=lambda row: (
                    episode_counts[row["episode"]] >= args.fruit_max_per_episode,
                    -row["dominant_gt_path_deg"],
                    row["episode"],
                ),
            )
            choice = next(
                (
                    row
                    for row in choices
                    if episode_counts[row["episode"]] < args.fruit_max_per_episode
                ),
                None,
            )
            if choice is None:
                raise ValueError(f"no eligible fruit row for target {target!r}")
            selected.append(choice)
            episode_counts[choice["episode"]] += 1

        while sum(row["active_arm"] == "left" for row in selected) < args.fruit_min_left:
            replacement = None
            for index, current in enumerate(selected):
                lower, upper = target_progress.get(current["target"], (0.0, 1.0))
                alternatives = sorted(
                    (
                        row
                        for row in segment_best
                        if row["target"] == current["target"]
                        and lower <= row["episode_progress"] <= upper
                        and row["active_arm"] == "left"
                        and row["episode"] != current["episode"]
                        and episode_counts[row["episode"]] < args.fruit_max_per_episode
                    ),
                    key=lambda row: -row["dominant_gt_path_deg"],
                )
                if alternatives:
                    replacement = (index, alternatives[0])
                    break
            if replacement is None:
                raise ValueError("cannot satisfy requested minimum number of left-active fruit rows")
            index, alternative = replacement
            episode_counts[selected[index]["episode"]] -= 1
            selected[index] = alternative
            episode_counts[alternative["episode"]] += 1
    else:
        by_source_episode: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in segment_best:
            by_source_episode[row["source_dataset"]][row["episode"]].append(row)
        metadata_priority = {
            source: [
                int(row["episode_index"])
                for row in candidate_metadata
                if row["source_dataset"] == source
            ]
            for source in sources
        }
        for source, episode_count in sources.items():
            used = 0
            for episode in metadata_priority[source]:
                rows = sorted(
                    by_source_episode[source].get(episode, []),
                    key=lambda row: -row["dominant_gt_path_deg"],
                )
                if len(rows) < args.frames_per_episode:
                    continue
                selected.extend(rows[: args.frames_per_episode])
                used += 1
                if used == episode_count:
                    break
            if used != episode_count:
                raise ValueError(f"only selected {used}/{episode_count} episodes for {source}")

    selected.sort(key=lambda row: (row["source_dataset"], row["episode"], row["frame"]))
    manifest = {
        "selection_contract": {
            "method": "checkpoint-independent native-segment GT selection",
            "dataset_root": str(args.dataset_root),
            "horizon": args.horizon,
            "frame_stride": args.frame_stride,
            "seed": args.seed,
            "sources": sources,
            "frames_per_episode": args.frames_per_episode,
            "minimum_dominant_path_deg": args.minimum_dominant_path_deg,
            "fruit_targets": args.fruit_target,
            "fruit_max_per_episode": args.fruit_max_per_episode,
            "fruit_min_left": args.fruit_min_left,
            "fruit_prompt_regex": args.fruit_prompt_regex,
            "fruit_max_segment_id": args.fruit_max_segment_id,
            "fruit_min_episode_progress": args.fruit_min_episode_progress,
            "fruit_max_episode_progress": args.fruit_max_episode_progress,
            "fruit_target_progress": target_progress,
            "candidate_count": len(all_candidates),
            "segment_best_count": len(segment_best),
        },
        "datasets": {args.dataset_name: selected},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected": len(selected),
                "episodes": len({row["episode"] for row in selected}),
                "sources": Counter(row["source_dataset"] for row in selected),
                "active_arms": Counter(row["active_arm"] for row in selected),
                "targets": Counter(row["target"] for row in selected if row["target"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
