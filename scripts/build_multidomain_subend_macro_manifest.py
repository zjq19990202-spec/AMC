#!/usr/bin/env python3
"""Build a model-independent multi-domain SUB-end steering manifest."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pyarrow.dataset as ds


def _parse_dataset(value: str) -> tuple[str, Path]:
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError(f"invalid dataset specification: {value!r}")
    return name, Path(path)


def _parse_quota(value: str) -> tuple[str, int]:
    name, count = value.split("=", 1)
    return name, int(count)


def _sample_frames(start: int, end: int, horizon: int, count: int) -> list[int]:
    latest = end - horizon
    if latest < start:
        return []
    return list(dict.fromkeys(np.rint(np.linspace(start, latest, count)).astype(int).tolist()))


def _motion(actions: np.ndarray, tcp_rows: np.ndarray) -> dict[str, Any]:
    joint_paths = {
        "left": float(np.linalg.norm(np.diff(actions[:, :7], axis=0), axis=1).sum()),
        "right": float(np.linalg.norm(np.diff(actions[:, 8:15], axis=0), axis=1).sum()),
    }
    tcp_paths = {
        "left": float(np.linalg.norm(np.diff(tcp_rows[:, 0:3], axis=0), axis=1).sum()),
        "right": float(np.linalg.norm(np.diff(tcp_rows[:, 24:27], axis=0), axis=1).sum()),
    }
    joint_arm = max(joint_paths, key=joint_paths.__getitem__)
    tcp_arm = max(tcp_paths, key=tcp_paths.__getitem__)
    arm = tcp_arm
    passive = "right" if arm == "left" else "left"
    return {
        "arm": arm,
        "joint_agrees_tcp": joint_arm == tcp_arm,
        "active_joint_path_rad": joint_paths[arm],
        "passive_joint_path_rad": joint_paths[passive],
        "active_tcp_path_m": tcp_paths[arm],
        "passive_tcp_path_m": tcp_paths[passive],
    }


def _is_spatial_target_prompt(prompt: str) -> bool:
    lowered = prompt.lower().strip()
    excluded = (
        "return to home",
        "retract to idle",
        "retract both arms",
        "then retract",
        "and retract the arm",
    )
    return not lowered.startswith("retract") and not any(token in lowered for token in excluded)


def _eligible_dataset(
    name: str,
    root: Path,
    *,
    horizon: int,
    frames_per_episode: int,
    minimum_joint_path_rad: float,
    minimum_joint_ratio: float,
    minimum_tcp_path_m: float,
    minimum_tcp_ratio: float,
    minimum_target_separation_m: float,
) -> list[dict[str, Any]]:
    metadata = [
        json.loads(line)
        for line in (root / "meta" / "episode_subtasks.jsonl").open()
    ]
    table = ds.dataset(str(root / "data"), format="parquet").to_table(
        columns=["episode_index", "frame_index", "action", "index"]
    )
    episode_values = np.asarray(table["episode_index"], dtype=np.int64)
    frame_values = np.asarray(table["frame_index"], dtype=np.int64)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    global_indices = np.asarray(table["index"], dtype=np.int64)
    tcp = np.load(root / "meta" / "tcp_pose_bimanual_base_tcp200.npy", mmap_mode="r")
    mask_path = root / "meta" / "start_inactive_arm_block_mask.json"
    mask_payload = json.loads(mask_path.read_text(encoding="utf-8")) if mask_path.exists() else {"records": []}
    mask_ends = {
        int(record["episode_index"]): int(record["mask_end_frame_exclusive"])
        for record in mask_payload.get("records", [])
    }

    episode_arrays = {}
    for payload in metadata:
        episode = int(payload["episode_index"])
        positions = np.flatnonzero(episode_values == episode)
        order = np.argsort(frame_values[positions])
        positions = positions[order]
        episode_arrays[episode] = {
            "frames": frame_values[positions],
            "actions": actions[positions],
            "indices": global_indices[positions],
        }

    candidates = []
    for payload in metadata:
        episode = int(payload["episode_index"])
        arrays = episode_arrays[episode]
        records = []
        for segment in payload.get("semantic_segments", []):
            start = int(segment["start_frame_30hz"])
            end = int(segment["end_frame_30hz_exclusive"])
            prompt = " ".join(str(segment.get("current_subtask", "")).split())
            mask = (arrays["frames"] >= start) & (arrays["frames"] < end)
            action_chunk = arrays["actions"][mask]
            index_chunk = arrays["indices"][mask]
            if not prompt or not _is_spatial_target_prompt(prompt) or len(action_chunk) < 2:
                continue
            motion = _motion(action_chunk, np.asarray(tcp[index_chunk], dtype=float))
            arm = str(motion["arm"])
            if not bool(motion["joint_agrees_tcp"]):
                continue
            if float(motion["active_joint_path_rad"]) < minimum_joint_path_rad:
                continue
            if float(motion["active_joint_path_rad"]) < minimum_joint_ratio * max(
                float(motion["passive_joint_path_rad"]), 1.0e-8
            ):
                continue
            if float(motion["active_tcp_path_m"]) < minimum_tcp_path_m:
                continue
            if float(motion["active_tcp_path_m"]) < minimum_tcp_ratio * max(
                float(motion["passive_tcp_path_m"]), 1.0e-8
            ):
                continue
            final_frame = end - 1
            final_position = np.flatnonzero(arrays["frames"] == final_frame)
            if len(final_position) != 1:
                continue
            tcp_row = int(arrays["indices"][int(final_position[0])])
            offset = 0 if arm == "left" else 24
            records.append(
                {
                    "segment_id": int(segment.get("subtask_id", -1)),
                    "start": start,
                    "end": end,
                    "prompt": prompt,
                    "arm": arm,
                    "target_xyz_m": np.asarray(tcp[tcp_row, offset : offset + 3], dtype=float).tolist(),
                    **motion,
                }
            )
        episode_candidates = []
        for native in records:
            valid_start = max(int(native["start"]), mask_ends.get(episode, int(native["start"])))
            frames = _sample_frames(
                valid_start, int(native["end"]), horizon, frames_per_episode
            )
            if len(frames) != frames_per_episode:
                continue
            alternatives = []
            seen_prompts = {str(native["prompt"]).lower()}
            for record in sorted(records, key=lambda row: int(row["start"])):
                prompt_key = str(record["prompt"]).lower()
                if record is native or record["arm"] != native["arm"]:
                    continue
                if int(record["end"]) < int(native["start"]):
                    continue
                if prompt_key in seen_prompts:
                    continue
                seen_prompts.add(prompt_key)
                alternatives.append(record)
            if len(alternatives) < 2:
                continue
            native_xyz = np.asarray(native["target_xyz_m"], dtype=float)
            candidate_pairs = []
            for first, second in combinations(alternatives, 2):
                points = [
                    native_xyz,
                    np.asarray(first["target_xyz_m"], dtype=float),
                    np.asarray(second["target_xyz_m"], dtype=float),
                ]
                distances = [
                    float(np.linalg.norm(points[i] - points[j]))
                    for i in range(3)
                    for j in range(i + 1, 3)
                ]
                candidate_pairs.append((min(distances), float(np.mean(distances)), first, second))
            best_pair = max(candidate_pairs, key=lambda item: (item[0], item[1]))
            if best_pair[0] < minimum_target_separation_m:
                continue
            episode_candidates.append(
                {
                    "dataset": name,
                    "episode": episode,
                    "source_dataset": payload.get("source_dataset"),
                    "source_episode": payload.get("source_episode"),
                    "episode_task": payload.get("episode_task"),
                    "native": native,
                    "alternatives": [best_pair[2], best_pair[3]],
                    "minimum_target_separation_m": best_pair[0],
                    "frames": frames,
                    "valid_frame_start": valid_start,
                }
            )
        if episode_candidates:
            candidates.append({"episode": episode, "segments": episode_candidates})
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, help="NAME=ROOT")
    parser.add_argument("--episodes-per-dataset", type=int, default=20)
    parser.add_argument("--quota", action="append", default=[], help="Optional NAME=COUNT override")
    parser.add_argument("--frames-per-episode", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--minimum-joint-path-rad", type=float, default=0.04)
    parser.add_argument("--minimum-joint-ratio", type=float, default=1.25)
    parser.add_argument("--minimum-tcp-path-m", type=float, default=0.02)
    parser.add_argument("--minimum-tcp-ratio", type=float, default=1.20)
    parser.add_argument("--minimum-target-separation-m", type=float, default=0.03)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    datasets = dict(_parse_dataset(value) for value in args.dataset)
    quota_overrides = dict(_parse_quota(value) for value in args.quota)
    unknown_quotas = set(quota_overrides) - set(datasets)
    if unknown_quotas:
        raise ValueError(f"quota names not present in --dataset: {sorted(unknown_quotas)}")
    quotas = {
        name: quota_overrides.get(name, args.episodes_per_dataset) for name in datasets
    }
    if len(datasets) != len(args.dataset):
        raise ValueError("dataset names must be unique")
    rng = random.Random(args.seed)
    output_rows: dict[str, list[dict[str, Any]]] = {}
    eligible_counts = {}
    for name, root in datasets.items():
        pool = _eligible_dataset(
            name,
            root,
            horizon=args.horizon,
            frames_per_episode=args.frames_per_episode,
            minimum_joint_path_rad=args.minimum_joint_path_rad,
            minimum_joint_ratio=args.minimum_joint_ratio,
            minimum_tcp_path_m=args.minimum_tcp_path_m,
            minimum_tcp_ratio=args.minimum_tcp_ratio,
            minimum_target_separation_m=args.minimum_target_separation_m,
        )
        eligible_counts[name] = len(pool)
        requested = quotas[name]
        if len(pool) < requested:
            raise RuntimeError(
                f"{name}: only {len(pool)} eligible episodes, need {requested}"
            )
        rng.shuffle(pool)
        selected = pool[:requested]
        rows = []
        for episode_payload in selected:
            candidate = rng.choice(episode_payload["segments"])
            targets = [candidate["native"], *candidate["alternatives"]]
            prompt_variants = {
                "native": str(targets[0]["prompt"]),
                "cf1": str(targets[1]["prompt"]),
                "cf2": str(targets[2]["prompt"]),
            }
            target_positions = {
                "native": targets[0]["target_xyz_m"],
                "cf1": targets[1]["target_xyz_m"],
                "cf2": targets[2]["target_xyz_m"],
            }
            target_segments = {
                "native": int(targets[0]["segment_id"]),
                "cf1": int(targets[1]["segment_id"]),
                "cf2": int(targets[2]["segment_id"]),
            }
            stages = ("early", "middle", "late")
            for frame_index, frame in enumerate(candidate["frames"]):
                native = candidate["native"]
                rows.append(
                    {
                        "dataset": name,
                        "episode": int(candidate["episode"]),
                        "frame": int(frame),
                        "frame_stage": stages[frame_index],
                        "active_arm": str(native["arm"]),
                        "source_dataset": candidate["source_dataset"],
                        "source_episode": candidate["source_episode"],
                        "episode_task": candidate["episode_task"],
                        "native_target": "native",
                        "native_segment_id": int(native["segment_id"]),
                        "native_segment_start": int(native["start"]),
                        "native_segment_end_exclusive": int(native["end"]),
                        "valid_frame_start": int(candidate["valid_frame_start"]),
                        "prompt_variants": prompt_variants,
                        "target_positions_m": target_positions,
                        "target_segment_ids": target_segments,
                        "active_arm_evidence": "dominant 14-D joint path and FK-TCP translation path agree",
                        "selection_active_joint_path_rad": float(native["active_joint_path_rad"]),
                        "selection_passive_joint_path_rad": float(native["passive_joint_path_rad"]),
                        "selection_active_tcp_path_m": float(native["active_tcp_path_m"]),
                        "selection_passive_tcp_path_m": float(native["passive_tcp_path_m"]),
                        "selection_minimum_target_separation_m": float(candidate["minimum_target_separation_m"]),
                    }
                )
        rows.sort(key=lambda row: (row["episode"], row["frame"]))
        output_rows[name] = rows

    report = {
        "selection_contract": {
            "purpose": "random non-fruit multi-domain SUB-end steering macro evaluation",
            "seed": args.seed,
            "episodes_per_dataset": args.episodes_per_dataset,
            "episode_quotas": quotas,
            "frames_per_episode": args.frames_per_episode,
            "prompt_variants_per_frame": 3,
            "horizon": args.horizon,
            "selection_inputs": "semantic SUB boundaries, recorded actions, and recorded FK-TCP only; no model outputs",
            "target_definition": "recorded active-arm TCP at the final frame of each candidate SUB",
            "frame_definition": "uniform early/middle/late positions in the native SUB after inactive-arm masking and before a complete horizon",
            "counterfactual_definition": "two distinct same-arm SUB prompts and their recorded terminal TCP targets from the same episode",
            "active_arm_definition": "joint-path and FK-TCP-path dominant arms must agree; gripper channels excluded",
            "thresholds": {
                "minimum_joint_path_rad": args.minimum_joint_path_rad,
                "minimum_joint_path_ratio": args.minimum_joint_ratio,
                "minimum_tcp_path_m": args.minimum_tcp_path_m,
                "minimum_tcp_path_ratio": args.minimum_tcp_ratio,
                "minimum_target_separation_m": args.minimum_target_separation_m,
            },
            "dataset_roots": {name: str(path.resolve()) for name, path in datasets.items()},
            "eligible_episodes": eligible_counts,
        },
        "datasets": output_rows,
        "summary": {
            "domains": len(output_rows),
            "episodes": sum(len({row["episode"] for row in rows}) for rows in output_rows.values()),
            "frames": sum(len(rows) for rows in output_rows.values()),
            "episodes_by_domain": {
                name: len({row["episode"] for row in rows}) for name, rows in output_rows.items()
            },
            "frames_by_domain": {name: len(rows) for name, rows in output_rows.items()},
            "arms": dict(
                Counter(
                    row["active_arm"]
                    for rows in output_rows.values()
                    for row in rows[:: args.frames_per_episode]
                )
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("eligible", json.dumps(eligible_counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
