#!/usr/bin/env python3
"""Compare plain PI0.5, AFRO, and AFRO with prompt-mismatched final zM.

The reviewed observations are fixed before inference.  Physical fruit targets
are estimated from the first TCP pose of the later lift/raise segment; fruits
already placed before the reviewed frame use a secondary box-center estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


FRUITS = (
    "carrot",
    "orange",
    "green bitter melon",
    "green radish",
    "yellow pear",
    "banana",
    "red chili pepper",
)
ROUTES = ("plain_pi05", "afro", "afro_wrong_zm")


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


def safe_cosine(vector: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.linalg.norm(vector) * np.linalg.norm(target))
    if denominator < 1.0e-9:
        return float("nan")
    return float(np.dot(vector, target) / denominator)


def mean(values: list[float]) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--plain-json", type=Path, required=True)
    parser.add_argument("--afro-json", type=Path, required=True)
    parser.add_argument("--wrong-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(
        args.dataset_root / "data/chunk-000/file-000.parquet",
        columns=["episode_index", "frame_index", "action", "index"],
    )
    episodes = np.asarray(table["episode_index"])
    frames = np.asarray(table["frame_index"])
    actions = np.asarray(table["action"].to_pylist())
    indices = np.asarray(table["index"])
    tcp = np.load(
        args.dataset_root / "meta/tcp_pose_bimanual_base_tcp200.npy", mmap_mode="r"
    )
    with (args.dataset_root / "meta/episode_subtasks.jsonl").open() as stream:
        segments = {
            int(payload["episode_index"]): payload["semantic_segments"]
            for payload in map(json.loads, stream)
        }
    selection = json.loads(args.selection.read_text(encoding="utf-8"))["datasets"][
        "target2058"
    ]
    plain = json.loads(args.plain_json.read_text(encoding="utf-8"))
    afro = json.loads(args.afro_json.read_text(encoding="utf-8"))
    wrong = json.loads(args.wrong_json.read_text(encoding="utf-8"))
    plain_rows = {(int(row["episode"]), int(row["frame"])): row for row in plain["rows"]}
    afro_rows = {(int(row["episode"]), int(row["frame"])): row for row in afro["results"]}
    wrong_rows = {
        (int(row["episode"]), int(row["frame"])): row for row in wrong["results"]
    }
    endpoint_route_change_mm = []
    for key, correct_row in afro_rows.items():
        wrong_row = wrong_rows[key]
        for fruit in FRUITS:
            endpoint_route_change_mm.append(
                float(
                    np.linalg.norm(
                        np.asarray(wrong_row["endpoint_displacement_mm"][fruit])
                        - np.asarray(correct_row["endpoint_displacement_mm"][fruit])
                    )
                )
            )

    def row_at(episode: int, frame: int) -> int:
        found = np.flatnonzero((episodes == episode) & (frames == frame))
        if not len(found):
            raise KeyError((episode, frame))
        return int(found[0])

    def arm_for_interval(episode: int, start: int, end: int) -> str:
        chunk = actions[(episodes == episode) & (frames >= start) & (frames < end)]
        left_motion = np.linalg.norm(chunk[-1, :7] - chunk[0, :7])
        right_motion = np.linalg.norm(chunk[-1, 8:15] - chunk[0, 8:15])
        return "left" if left_motion > right_motion else "right"

    def scene_targets(episode: int, source_frame: int) -> dict[str, dict]:
        episode_segments = segments[episode]
        placements: list[dict] = []
        contacts: dict[str, dict] = {}
        for index, segment in enumerate(episode_segments):
            text = segment["current_subtask"].lower()
            fruit = fruit_in(text)
            if not (text.startswith("lift ") or text.startswith("raise ")):
                continue
            start = int(segment["start_frame_30hz"])
            end = (
                int(episode_segments[index + 1]["start_frame_30hz"])
                if index + 1 < len(episode_segments)
                else int(frames[episodes == episode].max()) + 1
            )
            arm = arm_for_interval(episode, start, end)
            offset = 0 if arm == "left" else 24
            contact_row = row_at(episode, start)
            if fruit is not None:
                contacts[fruit] = {
                    "xyz": np.asarray(
                        tcp[int(indices[contact_row]), offset : offset + 3], dtype=float
                    ),
                    "frame": start,
                    "arm": arm,
                }
            box_match = re.search(r"\b(left|center|right) box\b", text)
            if box_match is None:
                continue
            interval_rows = np.flatnonzero(
                (episodes == episode) & (frames >= start) & (frames < end)
            )
            grip_column = 7 if arm == "left" else 15
            grip = actions[interval_rows, grip_column]
            release_local = (
                int(np.argmax(np.diff(grip)) + 1) if len(grip) > 1 else len(grip) - 1
            )
            release_row = int(interval_rows[release_local])
            placements.append(
                {
                    "fruit": fruit,
                    "box": box_match.group(1),
                    "start": start,
                    "endpoint": np.asarray(
                        tcp[int(indices[release_row]), offset : offset + 3], dtype=float
                    ),
                }
            )
        box_points: dict[str, np.ndarray] = {}
        for box in ("left", "center", "right"):
            points = [item["endpoint"] for item in placements if item["box"] == box]
            if points:
                box_points[box] = np.mean(points, axis=0)
        targets: dict[str, dict] = {}
        for fruit in FRUITS:
            earlier = [
                item
                for item in placements
                if item["fruit"] == fruit and item["start"] < source_frame
            ]
            if earlier and earlier[-1]["box"] in box_points:
                targets[fruit] = {
                    "xyz": box_points[earlier[-1]["box"]],
                    "kind": "placed_box_center",
                    "box": earlier[-1]["box"],
                }
            elif fruit in contacts and contacts[fruit]["frame"] >= source_frame:
                targets[fruit] = {
                    "xyz": contacts[fruit]["xyz"],
                    "kind": "future_grasp_contact",
                    "contact_frame": contacts[fruit]["frame"],
                }
        return targets

    rows: list[dict] = []
    for specification in selection:
        episode = int(specification["episode"])
        frame = int(specification["frame"])
        arm = specification["active_arm"]
        source = row_at(episode, frame)
        offset = 0 if arm == "left" else 24
        start = np.asarray(tcp[int(indices[source]), offset : offset + 3], dtype=float)
        plain_row = plain_rows[(episode, frame)]
        afro_row = afro_rows[(episode, frame)]
        wrong_row = wrong_rows[(episode, frame)]
        endpoints = {
            "plain_pi05": dict(
                zip(FRUITS, np.asarray(plain_row[f"{arm}_endpoint_xyz_m"]), strict=True)
            ),
            "afro": {
                fruit: start + np.asarray(delta, dtype=float) / 1000.0
                for fruit, delta in afro_row["endpoint_displacement_mm"].items()
            },
            "afro_wrong_zm": {
                fruit: start + np.asarray(delta, dtype=float) / 1000.0
                for fruit, delta in wrong_row["endpoint_displacement_mm"].items()
            },
        }
        for fruit, target_info in scene_targets(episode, frame).items():
            target = np.asarray(target_info["xyz"], dtype=float)
            initial_distance = float(np.linalg.norm(start - target) * 1000.0)
            row = {
                "episode": episode,
                "frame": frame,
                "active_arm": arm,
                "placed_fruits": int(specification["placed_fruits"]),
                "native_target": specification["native_target"],
                "fruit": fruit,
                "is_native_target": fruit == specification["native_target"],
                "target_kind": target_info["kind"],
                "initial_distance_mm": initial_distance,
            }
            for route in ROUTES:
                endpoint = np.asarray(endpoints[route][fruit], dtype=float)
                distance = float(np.linalg.norm(endpoint - target) * 1000.0)
                all_prompt_distances = {
                    prompt: float(np.linalg.norm(np.asarray(value) - target) * 1000.0)
                    for prompt, value in endpoints[route].items()
                }
                row[f"{route}_distance_mm"] = distance
                row[f"{route}_reduction_mm"] = initial_distance - distance
                row[f"{route}_direction_cosine"] = safe_cosine(
                    endpoint - start, target - start
                )
                row[f"{route}_moves_closer"] = distance < initial_distance
                row[f"{route}_prompt_top1"] = min(
                    all_prompt_distances, key=all_prompt_distances.get
                ) == fruit
            row["afro_gain_vs_plain_mm"] = (
                row["plain_pi05_distance_mm"] - row["afro_distance_mm"]
            )
            row["wrong_zm_penalty_mm"] = (
                row["afro_wrong_zm_distance_mm"] - row["afro_distance_mm"]
            )
            rows.append(row)

    def aggregate(selected: list[dict]) -> dict:
        result: dict = {"n": len(selected)}
        for route in ROUTES:
            result[route] = {
                "mean_endpoint_distance_mm": mean(
                    [row[f"{route}_distance_mm"] for row in selected]
                ),
                "mean_distance_reduction_mm": mean(
                    [row[f"{route}_reduction_mm"] for row in selected]
                ),
                "mean_direction_cosine": mean(
                    [row[f"{route}_direction_cosine"] for row in selected]
                ),
                "moves_closer_rate": mean(
                    [float(row[f"{route}_moves_closer"]) for row in selected]
                ),
                "prompt_top1_rate": mean(
                    [float(row[f"{route}_prompt_top1"]) for row in selected]
                ),
            }
        afro_gain = np.asarray([row["afro_gain_vs_plain_mm"] for row in selected])
        wrong_penalty = np.asarray([row["wrong_zm_penalty_mm"] for row in selected])
        result["paired"] = {
            "afro_gain_vs_plain_mean_mm": float(np.mean(afro_gain)),
            "afro_better_than_plain_rate": float(np.mean(afro_gain > 0)),
            "wrong_zm_penalty_mean_mm": float(np.mean(wrong_penalty)),
            "correct_zm_better_than_wrong_rate": float(np.mean(wrong_penalty > 0)),
        }
        return result

    future = [row for row in rows if row["target_kind"] == "future_grasp_contact"]
    native = [row for row in future if row["is_native_target"]]
    placed = [row for row in rows if row["target_kind"] == "placed_box_center"]
    report = {
        "metric_contract": {
            "primary_subset": "future_grasp_contact",
            "target_position": "TCP xyz at the start of the later lift/raise segment",
            "secondary_subset": "placed_box_center",
            "prompt_top1": "the endpoint under the matching fruit prompt is closest to that physical fruit among all seven prompts",
            "wrong_zm": "AFRO Context KV remains prompt-correct; only final two-arm zM is cyclically shifted across prompt variants",
        },
        "summary": {
            "future_grasp_contact": aggregate(future),
            "native_future_grasp_contact": aggregate(native),
            "placed_box_center_secondary": aggregate(placed),
            "all_known_targets": aggregate(rows),
            "correct_vs_wrong_zm_endpoint_change": {
                "n_prompt_cases": len(endpoint_route_change_mm),
                "mean_tcp_endpoint_separation_mm": float(
                    np.mean(endpoint_route_change_mm)
                ),
                "median_tcp_endpoint_separation_mm": float(
                    np.median(endpoint_route_change_mm)
                ),
                "p90_tcp_endpoint_separation_mm": float(
                    np.percentile(endpoint_route_change_mm, 90)
                ),
                **{
                    f"fraction_over_{threshold}mm": float(
                        np.mean(np.asarray(endpoint_route_change_mm) > threshold)
                    )
                    for threshold in (2, 5, 10, 20)
                },
            },
        },
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "per_target.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = report["summary"]["future_grasp_contact"]
    colors = ("#64748b", "#2563eb", "#dc2626")
    labels = ("PI0.5", "AFRO", "AFRO + wrong zM")
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.25), constrained_layout=True)
    x = np.arange(3)
    axes[0].bar(
        x,
        [summary[route]["mean_endpoint_distance_mm"] for route in ROUTES],
        color=colors,
    )
    axes[0].set_ylabel("endpoint distance to fruit (mm) ↓")
    axes[1].bar(
        x,
        [summary[route]["moves_closer_rate"] * 100 for route in ROUTES],
        color=colors,
    )
    axes[1].set_ylabel("moves closer (%) ↑")
    axes[1].set_ylim(0, 100)
    axes[2].bar(
        x,
        [summary[route]["prompt_top1_rate"] * 100 for route in ROUTES],
        color=colors,
    )
    axes[2].set_ylabel("matching-prompt top-1 (%) ↑")
    axes[2].set_ylim(0, 100)
    for axis in axes:
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(f"Fruit steering on frozen reviewed scenes (n={len(future)} target cases)")
    figure.savefig(args.output_dir / "three_route_fruit_steering.png", dpi=220)
    plt.close(figure)

    native_summary = report["summary"]["native_future_grasp_contact"]
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.25), constrained_layout=True)
    axes[0].bar(
        x,
        [native_summary[route]["mean_endpoint_distance_mm"] for route in ROUTES],
        color=colors,
    )
    axes[0].set_ylabel("endpoint distance to native fruit (mm) ↓")
    axes[1].bar(
        x,
        [native_summary[route]["moves_closer_rate"] * 100 for route in ROUTES],
        color=colors,
    )
    axes[1].set_ylabel("moves closer (%) ↑")
    axes[1].set_ylim(0, 100)
    axes[2].bar(
        x,
        [native_summary[route]["prompt_top1_rate"] * 100 for route in ROUTES],
        color=colors,
    )
    axes[2].set_ylabel("matching-prompt top-1 (%) ↑")
    axes[2].set_ylim(0, 100)
    for axis in axes:
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(f"Native fruit prompts on frozen reviewed scenes (n={len(native)})")
    figure.savefig(args.output_dir / "native_three_route_fruit_steering.png", dpi=220)
    plt.close(figure)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
