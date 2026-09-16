#!/usr/bin/env python3
"""Build a model-independent, fruit/arm-balanced 100-frame evaluation manifest."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re

import numpy as np
import pyarrow.parquet as pq


FRUITS = (
    "carrot",
    "orange",
    "red chili pepper",
    "green radish",
    "green bitter melon",
    "banana",
    "yellow pear",
)
QUOTAS = {
    "carrot": {"left": 8, "right": 7},
    "orange": {"left": 7, "right": 8},
    "red chili pepper": {"left": 7, "right": 7},
    "green radish": {"left": 7, "right": 7},
    "green bitter melon": {"left": 7, "right": 7},
    "banana": {"left": 7, "right": 7},
    "yellow pear": {"left": 7, "right": 7},
}


def fruit_in(text: str) -> str | None:
    text = text.lower()
    if "orange pumpkin" in text:
        return None
    for fruit in sorted(FRUITS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(fruit)}\b", text):
            return fruit
    if "yellow banana" in text:
        return "banana"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--horizon", type=int, default=50)
    args = ap.parse_args()

    table = pq.read_table(
        args.dataset_root / "data/chunk-000/file-000.parquet",
        columns=["episode_index", "frame_index", "action"],
    )
    episodes = np.asarray(table["episode_index"])
    frames = np.asarray(table["frame_index"])
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    metadata = [json.loads(line) for line in (args.dataset_root / "meta/episode_subtasks.jsonl").open()]

    candidates: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for episode in metadata:
        ep = int(episode["episode_index"])
        ep_mask = episodes == ep
        for segment in episode["semantic_segments"]:
            prompt = str(segment["current_subtask"])
            fruit = fruit_in(prompt)
            start = int(segment["start_frame_30hz"])
            end = int(segment["end_frame_30hz_exclusive"])
            if fruit is None or not any(token in prompt.lower() for token in ("grasp", "take hold", "pick up")):
                continue
            if end - start < args.horizon:
                continue
            chunk = actions[ep_mask & (frames >= start) & (frames < start + args.horizon)]
            if len(chunk) != args.horizon:
                continue
            left_path = float(np.linalg.norm(np.diff(chunk[:, :7], axis=0), axis=1).sum())
            right_path = float(np.linalg.norm(np.diff(chunk[:, 8:15], axis=0), axis=1).sum())
            arm = "left" if left_path > right_path else "right"
            dominant, passive = (left_path, right_path) if arm == "left" else (right_path, left_path)
            if dominant < 0.04 or dominant < 1.25 * max(passive, 1e-6):
                continue
            candidates[(fruit, arm)].append({
                "episode": ep,
                "frame": start,
                "active_arm": arm,
                "placed_fruits": int(segment.get("subtask_id", 0)) // 2,
                "prompt": prompt,
                "native_target": fruit,
                "source_dataset": episode.get("source_dataset"),
                "segment_end": end,
                "gt_active_path_rad": dominant,
                "gt_passive_path_rad": passive,
            })

    rng = random.Random(args.seed)
    keys = [(fruit, arm) for fruit in FRUITS for arm in ("left", "right")]
    best = None
    for _ in range(20_000):
        rng.shuffle(keys)
        used_episodes: set[int] = set()
        selected: list[dict] = []
        success = True
        for key in keys:
            pool = candidates[key].copy()
            rng.shuffle(pool)
            pool.sort(key=lambda row: (row["episode"] in used_episodes, -row["gt_active_path_rad"]))
            chosen = [row for row in pool if row["episode"] not in used_episodes][: QUOTAS[key[0]][key[1]]]
            if len(chosen) != QUOTAS[key[0]][key[1]]:
                success = False
                break
            selected.extend(chosen)
            used_episodes.update(row["episode"] for row in chosen)
        if success and len(selected) == 100:
            best = selected
            break
    if best is None:
        availability = {f"{fruit}:{arm}": len(candidates[(fruit, arm)]) for fruit, arm in keys}
        raise RuntimeError(f"could not satisfy unique-episode quotas; availability={availability}")

    best.sort(key=lambda row: (row["episode"], row["frame"]))
    report = {
        "selection_contract": {
            "purpose": "model-independent 100-frame fruit steering evaluation",
            "seed": args.seed,
            "horizon": args.horizon,
            "unique_episode_per_frame": True,
            "fruit_arm_quotas": QUOTAS,
            "selection_inputs": "metadata prompts, segment bounds, and recorded GT motion only",
            "reviewed": False,
        },
        "datasets": {"target2058": best},
        "summary": {
            "frames": len(best),
            "episodes": len({row["episode"] for row in best}),
            "fruits": dict(Counter(row["native_target"] for row in best)),
            "arms": dict(Counter(row["active_arm"] for row in best)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
