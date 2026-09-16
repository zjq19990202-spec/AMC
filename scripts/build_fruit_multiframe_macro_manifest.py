#!/usr/bin/env python3
"""Build a random, model-independent multi-frame fruit-steering manifest.

Each selected episode contributes one native grasp subtask and several frames
within that subtask.  Fruit locations are estimated from the recorded active-
arm TCP at the final frame of that grasp subtask.  Selection never reads model
predictions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
from typing import Any

import numpy as np
import pyarrow.dataset as ds


FRUITS = (
    "carrot",
    "orange",
    "green bitter melon",
    "green radish",
    "yellow pear",
    "banana",
    "red chili pepper",
)


def fruit_in(text: str) -> str | None:
    lowered = text.lower()
    if "orange pumpkin" in lowered:
        return None
    for fruit in sorted(FRUITS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(fruit)}\b", lowered):
            return fruit
    if "yellow banana" in lowered:
        return "banana"
    return None


def _motion_arm(actions: np.ndarray, tcp_rows: np.ndarray) -> dict[str, Any]:
    if len(actions) < 2:
        return {
            "arm": "right",
            "joint_agrees_tcp": False,
            "active_joint_path_rad": 0.0,
            "passive_joint_path_rad": 0.0,
            "active_tcp_path_m": 0.0,
            "passive_tcp_path_m": 0.0,
        }
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


def _proportional_quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    available_total = sum(counts.values())
    if total > available_total:
        raise ValueError(f"requested {total} episodes but only {available_total} are eligible")
    exact = {name: total * count / available_total for name, count in counts.items()}
    quotas = {name: min(counts[name], int(value)) for name, value in exact.items()}
    # Preserve every represented source when possible, then distribute the
    # remaining slots by largest fractional remainder.
    if total >= len(counts):
        for name, count in counts.items():
            if count and quotas[name] == 0:
                quotas[name] = 1
    while sum(quotas.values()) > total:
        removable = [name for name in quotas if quotas[name] > 1]
        name = min(removable, key=lambda key: exact[key] - quotas[key])
        quotas[name] -= 1
    while sum(quotas.values()) < total:
        expandable = [name for name in quotas if quotas[name] < counts[name]]
        name = max(expandable, key=lambda key: (exact[key] - quotas[key], counts[key]))
        quotas[name] += 1
    return quotas


def _sample_frames(start: int, end: int, horizon: int, count: int) -> list[int]:
    latest = end - horizon
    if latest < start:
        return []
    candidates = np.rint(np.linspace(start, latest, count)).astype(int).tolist()
    return list(dict.fromkeys(candidates))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--frames-per-episode", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--minimum-active-path-rad", type=float, default=0.04)
    parser.add_argument("--minimum-arm-ratio", type=float, default=1.25)
    parser.add_argument("--minimum-active-tcp-path-m", type=float, default=0.02)
    parser.add_argument("--minimum-tcp-arm-ratio", type=float, default=1.20)
    args = parser.parse_args()

    metadata = [
        json.loads(line)
        for line in (args.dataset_root / "meta" / "episode_subtasks.jsonl").open()
    ]
    mask_path = args.dataset_root / "meta" / "start_inactive_arm_block_mask.json"
    mask_payload = json.loads(mask_path.read_text(encoding="utf-8")) if mask_path.exists() else {"records": []}
    mask_end_by_episode = {
        int(record["episode_index"]): int(record["mask_end_frame_exclusive"])
        for record in mask_payload.get("records", [])
    }
    table = ds.dataset(str(args.dataset_root / "data"), format="parquet").to_table(
        columns=["episode_index", "frame_index", "action", "index"]
    )
    episode_values = np.asarray(table["episode_index"], dtype=np.int64)
    frame_values = np.asarray(table["frame_index"], dtype=np.int64)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    global_indices = np.asarray(table["index"], dtype=np.int64)
    tcp = np.load(
        args.dataset_root / "meta" / "tcp_pose_bimanual_base_tcp200.npy",
        mmap_mode="r",
    )

    episode_arrays: dict[int, dict[str, np.ndarray]] = {}
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

    def interval(episode: int, start: int, end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arrays = episode_arrays[episode]
        mask = (arrays["frames"] >= start) & (arrays["frames"] < end)
        return arrays["frames"][mask], arrays["actions"][mask], arrays["indices"][mask]

    def tcp_at(episode: int, frame: int, arm: str) -> np.ndarray:
        arrays = episode_arrays[episode]
        found = np.flatnonzero(arrays["frames"] == frame)
        if len(found) != 1:
            raise KeyError((episode, frame))
        row = int(arrays["indices"][int(found[0])])
        offset = 0 if arm == "left" else 24
        return np.asarray(tcp[row, offset : offset + 3], dtype=float)

    episode_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in metadata:
        episode = int(payload["episode_index"])
        segments = payload.get("semantic_segments", [])
        contacts: list[dict[str, Any]] = []
        for segment in segments:
            prompt = str(segment.get("current_subtask", "")).strip()
            lowered = prompt.lower()
            fruit = fruit_in(prompt)
            if fruit is None or not any(
                token in lowered for token in ("grasp", "take hold", "pick up")
            ):
                continue
            start = int(segment["start_frame_30hz"])
            end = int(segment["end_frame_30hz_exclusive"])
            _, action_chunk, index_chunk = interval(episode, start, end)
            motion = _motion_arm(action_chunk, np.asarray(tcp[index_chunk], dtype=float))
            arm = str(motion["arm"])
            if end <= start:
                continue
            contacts.append({
                "fruit": fruit,
                "segment_id": int(segment.get("subtask_id", -1)),
                "segment_start": start,
                "segment_end_exclusive": end,
                "prompt": prompt,
                "frame": end - 1,
                "arm": arm,
                "xyz_m": tcp_at(episode, end - 1, arm).tolist(),
                **motion,
            })

        candidates = []
        for contact in contacts:
            prompt = str(contact["prompt"])
            native = str(contact["fruit"])
            start = int(contact["segment_start"])
            end = int(contact["segment_end_exclusive"])
            valid_start = max(start, mask_end_by_episode.get(episode, start))
            sampled_frames = _sample_frames(
                valid_start, end, args.horizon, args.frames_per_episode
            )
            if len(sampled_frames) != args.frames_per_episode:
                continue
            arm = str(contact["arm"])
            active_path = float(contact["active_joint_path_rad"])
            passive_path = float(contact["passive_joint_path_rad"])
            active_tcp_path = float(contact["active_tcp_path_m"])
            passive_tcp_path = float(contact["passive_tcp_path_m"])
            if not bool(contact["joint_agrees_tcp"]):
                continue
            if active_path < args.minimum_active_path_rad:
                continue
            if active_path < args.minimum_arm_ratio * max(passive_path, 1.0e-8):
                continue
            if active_tcp_path < args.minimum_active_tcp_path_m:
                continue
            if active_tcp_path < args.minimum_tcp_arm_ratio * max(passive_tcp_path, 1.0e-8):
                continue
            future_targets: dict[str, dict[str, Any]] = {}
            for candidate_contact in sorted(contacts, key=lambda item: int(item["frame"])):
                candidate_fruit = str(candidate_contact["fruit"])
                if candidate_contact["arm"] != arm or int(candidate_contact["frame"]) < start:
                    continue
                future_targets.setdefault(candidate_fruit, candidate_contact)
            # The current SUB endpoint is the authoritative native object
            # location even if the same fruit appears again later.
            future_targets[native] = contact
            candidates.append(
                {
                    "episode": episode,
                    "source_dataset": str(payload.get("source_dataset") or "unknown"),
                    "source_episode": payload.get("source_episode"),
                    "episode_task": payload.get("episode_task"),
                    "segment_id": int(contact["segment_id"]),
                    "segment_start": start,
                    "valid_frame_start": valid_start,
                    "segment_end_exclusive": end,
                    "prompt": prompt,
                    "native_target": native,
                    "active_arm": arm,
                    "active_path_rad": active_path,
                    "passive_path_rad": passive_path,
                    "active_tcp_path_m": active_tcp_path,
                    "passive_tcp_path_m": passive_tcp_path,
                    "active_arm_evidence": "dominant 14-D joint path and FK-TCP translation path agree",
                    "frames": sampled_frames,
                    "future_targets": future_targets,
                }
            )
        if candidates:
            source = str(payload.get("source_dataset") or "unknown")
            episode_candidates[source].append({"episode": episode, "segments": candidates})

    source_counts = {source: len(rows) for source, rows in episode_candidates.items()}
    quotas = _proportional_quotas(source_counts, args.episodes)
    rng = random.Random(args.seed)
    selected_segments: list[dict[str, Any]] = []
    for source in sorted(episode_candidates):
        episode_pool = episode_candidates[source].copy()
        rng.shuffle(episode_pool)
        for episode_payload in episode_pool[: quotas[source]]:
            selected_segments.append(rng.choice(episode_payload["segments"]))

    rows = []
    stage_names = ("early", "middle", "late") if args.frames_per_episode == 3 else None
    for selected in selected_segments:
        targets = {
            name: value["xyz_m"] for name, value in selected["future_targets"].items()
        }
        contact_frames = {
            name: int(value["frame"]) for name, value in selected["future_targets"].items()
        }
        for index, frame in enumerate(selected["frames"]):
            rows.append(
                {
                    "episode": selected["episode"],
                    "frame": int(frame),
                    "frame_stage": stage_names[index] if stage_names else f"stage_{index}",
                    "active_arm": selected["active_arm"],
                    "source_dataset": selected["source_dataset"],
                    "source_episode": selected["source_episode"],
                    "episode_task": selected["episode_task"],
                    "segment_id": selected["segment_id"],
                    "segment_start": selected["segment_start"],
                    "valid_frame_start": selected["valid_frame_start"],
                    "segment_end_exclusive": selected["segment_end_exclusive"],
                    "prompt": selected["prompt"],
                    "native_target": selected["native_target"],
                    "target_positions_m": targets,
                    "target_contact_frames": contact_frames,
                    "selection_active_path_rad": selected["active_path_rad"],
                    "selection_passive_path_rad": selected["passive_path_rad"],
                    "selection_active_tcp_path_m": selected["active_tcp_path_m"],
                    "selection_passive_tcp_path_m": selected["passive_tcp_path_m"],
                    "active_arm_evidence": selected["active_arm_evidence"],
                }
            )
    rows.sort(key=lambda row: (row["episode"], row["frame"]))
    report = {
        "selection_contract": {
            "purpose": "random multi-frame fruit target steering macro evaluation",
            "seed": args.seed,
            "episodes": args.episodes,
            "frames_per_episode": args.frames_per_episode,
            "horizon": args.horizon,
            "selection_inputs": "semantic SUB boundaries, recorded actions, and recorded TCP only; no model outputs",
            "target_definition": "recorded active-arm TCP at the final frame of the fruit grasp SUB (object-contact endpoint)",
            "frame_definition": "uniform early/middle/late positions in [max(SUB start, inactive-arm mask end), SUB end - horizon]",
            "inactive_arm_start_mask": str(mask_path) if mask_path.exists() else None,
            "source_sampling": "proportional stratified random sampling over eligible source datasets",
            "active_arm_definition": "joint-path and FK-TCP-path dominant arms must agree; gripper channels excluded",
            "active_arm_thresholds": {
                "minimum_joint_path_rad": args.minimum_active_path_rad,
                "minimum_joint_path_ratio": args.minimum_arm_ratio,
                "minimum_tcp_path_m": args.minimum_active_tcp_path_m,
                "minimum_tcp_path_ratio": args.minimum_tcp_arm_ratio,
            },
            "source_eligible_episodes": source_counts,
            "source_episode_quotas": quotas,
        },
        "target_names": list(FRUITS),
        "datasets": {"fruit_macro": rows},
        "summary": {
            "episodes": len({row["episode"] for row in rows}),
            "frames": len(rows),
            "sources": dict(Counter(row["source_dataset"] for row in rows[:: args.frames_per_episode])),
            "native_targets": dict(Counter(row["native_target"] for row in rows[:: args.frames_per_episode])),
            "arms": dict(Counter(row["active_arm"] for row in rows[:: args.frames_per_episode])),
            "scored_prompt_targets": sum(len(row["target_positions_m"]) for row in rows),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
