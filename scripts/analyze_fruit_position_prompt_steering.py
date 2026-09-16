#!/usr/bin/env python3
"""Score fruit-prompt trajectories against episode-derived fruit contact positions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-9 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--plain-json", type=Path, required=True)
    parser.add_argument("--afro-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plain = json.loads(args.plain_json.read_text())
    afro = json.loads(args.afro_json.read_text())
    fruits = ["carrot", "red bell pepper", "green radish", "green bitter melon", "orange"]

    table = pq.read_table(
        args.dataset_root / "data/chunk-000/file-000.parquet",
        columns=["episode_index", "frame_index", "action", "index"],
    )
    episodes = np.asarray(table["episode_index"])
    frames = np.asarray(table["frame_index"])
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    indices = np.asarray(table["index"])
    tcp = np.load(args.dataset_root / "meta/tcp_pose_bimanual_base_tcp200.npy", mmap_mode="r")

    subtasks = {}
    for line in (args.dataset_root / "meta/episode_subtasks.jsonl").open():
        item = json.loads(line)
        subtasks[int(item["episode_index"])] = item["semantic_segments"]

    def fruit_contacts(episode: int) -> dict[str, dict]:
        contacts = {}
        segments = subtasks[episode]
        for seg_index, segment in enumerate(segments):
            text = segment["current_subtask"].lower()
            if "box" not in text or not (text.startswith("lift ") or text.startswith("raise ")):
                continue
            matched = next((fruit for fruit in fruits if re.search(rf"\b{re.escape(fruit)}\b", text)), None)
            if matched is None:
                continue
            boundary = int(segment["start_frame_30hz"])
            grasp_start = int(segments[max(0, seg_index - 1)]["start_frame_30hz"])
            mask = (episodes == episode) & (frames >= grasp_start) & (frames < boundary)
            chunk = actions[mask]
            if len(chunk) < 2:
                continue
            left_motion = float(np.linalg.norm(chunk[-1, :7] - chunk[0, :7]))
            right_motion = float(np.linalg.norm(chunk[-1, 8:15] - chunk[0, 8:15]))
            arm = "left" if left_motion > right_motion else "right"
            row = np.flatnonzero((episodes == episode) & (frames == boundary))[0]
            data_index = int(indices[row])
            offset = 0 if arm == "left" else 24
            contacts[matched] = {
                "frame": boundary,
                "arm": arm,
                "xyz_m": np.asarray(tcp[data_index, offset : offset + 3], dtype=float),
            }
        return contacts

    plain_rows = {(int(r["episode"]), int(r["frame"])): r for r in plain["rows"]}
    afro_rows = {(int(r["episode"]), int(r["frame"])): r for r in afro["results"]}
    scored = []
    summaries = {}

    for model_name in ("Plain PI0.5 25K", "AFRO 50K"):
        model_rows = plain_rows if model_name.startswith("Plain") else afro_rows
        model_scores = []
        for key, row in model_rows.items():
            episode, frame = key
            arm = row["active_arm"]
            source_row = np.flatnonzero((episodes == episode) & (frames == frame))[0]
            source_index = int(indices[source_row])
            offset = 0 if arm == "left" else 24
            start = np.asarray(tcp[source_index, offset : offset + 3], dtype=float)
            contacts = fruit_contacts(episode)
            available = {
                fruit: value for fruit, value in contacts.items()
                if value["frame"] >= frame and value["arm"] == arm
            }
            if model_name.startswith("Plain"):
                prompt_names = fruits
                endpoints = np.asarray(row[f"{arm}_endpoint_xyz_m"], dtype=float)
            else:
                prompt_names = list(row["endpoint_displacement_mm"])
                endpoints = np.asarray([
                    start + np.asarray(row["endpoint_displacement_mm"][fruit]) / 1000
                    for fruit in prompt_names
                ])
            predictions = dict(zip(prompt_names, endpoints, strict=True))
            for fruit, endpoint in predictions.items():
                if fruit not in available:
                    continue
                target = available[fruit]["xyz_m"]
                delta = endpoint - start
                target_delta = target - start
                direction_cosine = cosine(delta, target_delta)
                progress_mm = float(np.dot(delta, target_delta / np.linalg.norm(target_delta)) * 1000)
                ranked = sorted(
                    available,
                    key=lambda candidate: cosine(delta, available[candidate]["xyz_m"] - start),
                    reverse=True,
                )
                other_endpoints = [value for name, value in predictions.items() if name != fruit]
                own_distance = float(np.linalg.norm(endpoint - target) * 1000)
                other_distance = float(np.mean([np.linalg.norm(value - target) * 1000 for value in other_endpoints]))
                record = {
                    "model": model_name,
                    "episode": episode,
                    "frame": frame,
                    "active_arm": arm,
                    "prompt_fruit": fruit,
                    "available_fruits": list(available),
                    "fruit_contact_frame": available[fruit]["frame"],
                    "fruit_xyz_m": target.tolist(),
                    "predicted_endpoint_xyz_m": endpoint.tolist(),
                    "direction_cosine": direction_cosine,
                    "projected_progress_mm": progress_mm,
                    "endpoint_distance_to_fruit_mm": own_distance,
                    "distance_advantage_vs_other_prompts_mm": other_distance - own_distance,
                    "direction_top1": ranked[0] == fruit,
                    "direction_rank": ranked.index(fruit) + 1,
                }
                scored.append(record)
                model_scores.append(record)
        summaries[model_name] = {
            "eligible_prompt_scene_pairs": len(model_scores),
            "direction_top1": int(sum(x["direction_top1"] for x in model_scores)),
            "direction_top1_rate": float(np.mean([x["direction_top1"] for x in model_scores])),
            "mean_direction_cosine": float(np.mean([x["direction_cosine"] for x in model_scores])),
            "mean_projected_progress_mm": float(np.mean([x["projected_progress_mm"] for x in model_scores])),
            "mean_endpoint_distance_to_fruit_mm": float(np.mean([x["endpoint_distance_to_fruit_mm"] for x in model_scores])),
            "mean_distance_advantage_vs_other_prompts_mm": float(np.mean([x["distance_advantage_vs_other_prompts_mm"] for x in model_scores])),
        }

    report = {"position_definition": "active-arm TCP at grasp-to-lift subtask boundary", "summaries": summaries, "rows": scored}
    (args.output_dir / "fruit_position_prompt_steering.json").write_text(json.dumps(report, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for model_index, model_name in enumerate(summaries):
        rows = [x for x in scored if x["model"] == model_name]
        labels = [f"ep{x['episode']} f{x['frame']}\n{x['prompt_fruit']}" for x in rows]
        colors = ["#16a34a" if x["direction_top1"] else "#dc2626" for x in rows]
        axes[model_index].bar(np.arange(len(rows)), [x["direction_cosine"] for x in rows], color=colors)
        axes[model_index].set_xticks(np.arange(len(rows)), labels, rotation=70, ha="right", fontsize=7)
        axes[model_index].set_ylim(-1, 1)
        axes[model_index].axhline(0, color="black", lw=0.7)
        axes[model_index].set_title(f"{model_name}\nfruit-direction cosine; green = target is Top-1")
        axes[model_index].set_ylabel("cos(predicted displacement, fruit direction)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fruit_position_direction_score.png", dpi=180)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
