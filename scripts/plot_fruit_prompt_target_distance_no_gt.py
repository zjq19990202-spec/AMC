#!/usr/bin/env python3
"""Plot same-observation fruit-prompt predictions and distance to named fruit, without GT."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from evaluate_global_episode_chunks import _endpoint_tcp_from_actions


FRUITS = ["carrot", "red bell pepper", "green radish", "green bitter melon", "orange"]
COLORS = dict(zip(FRUITS, ["#f97316", "#dc2626", "#22c55e", "#15803d", "#f59e0b"], strict=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--plain-json", type=Path, required=True)
    ap.add_argument("--afro-json", type=Path, required=True)
    ap.add_argument("--afro-npz", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    plain = json.loads(args.plain_json.read_text())
    afro = json.loads(args.afro_json.read_text())
    archive = np.load(args.afro_npz)
    table = pq.read_table(
        args.dataset_root / "data/chunk-000/file-000.parquet",
        columns=["episode_index", "frame_index", "action", "index"],
    )
    eps = np.asarray(table["episode_index"])
    frames = np.asarray(table["frame_index"])
    actions = np.asarray(table["action"].to_pylist(), dtype=float)
    indices = np.asarray(table["index"])
    tcp = np.load(args.dataset_root / "meta/tcp_pose_bimanual_base_tcp200.npy", mmap_mode="r")
    segments = {}
    for line in (args.dataset_root / "meta/episode_subtasks.jsonl").open():
        item = json.loads(line)
        segments[int(item["episode_index"])] = item["semantic_segments"]

    def contacts(ep: int):
        out = {}
        for i, seg in enumerate(segments[ep]):
            text = seg["current_subtask"].lower()
            if "box" not in text or not (text.startswith("lift ") or text.startswith("raise ")):
                continue
            fruit = next((x for x in FRUITS if re.search(rf"\b{re.escape(x)}\b", text)), None)
            if fruit is None:
                continue
            boundary = int(seg["start_frame_30hz"])
            begin = int(segments[ep][max(0, i - 1)]["start_frame_30hz"])
            mask = (eps == ep) & (frames >= begin) & (frames < boundary)
            chunk = actions[mask]
            left = np.linalg.norm(chunk[-1, :7] - chunk[0, :7])
            right = np.linalg.norm(chunk[-1, 8:15] - chunk[0, 8:15])
            arm = "left" if left > right else "right"
            row = np.flatnonzero((eps == ep) & (frames == boundary))[0]
            offset = 0 if arm == "left" else 24
            out[fruit] = {"frame": boundary, "xyz": np.asarray(tcp[int(indices[row]), offset:offset + 3])}
        return out

    plain_rows = {(r["episode"], r["frame"]): r for r in plain["rows"]}
    afro_rows = {(r["episode"], r["frame"]): r for r in afro["results"]}
    requested = [(0, 143, "carrot"), (76, 796, "red bell pepper"), (114, 1580, "green radish"), (152, 1960, "green bitter melon"), (0, 610, "orange")]
    fig = plt.figure(figsize=(18, 5.2 * len(requested)), constrained_layout=True)
    report = []
    for row_i, (ep, frame, native) in enumerate(requested):
        pr = plain_rows[(ep, frame)]
        ar = afro_rows[(ep, frame)]
        arm = pr["active_arm"]
        available = {k: v for k, v in contacts(ep).items() if v["frame"] >= frame}
        plain_tracks = dict(zip(FRUITS, np.asarray(pr[f"{arm}_tcp_trajectories_m"]), strict=True))
        prefix = f"episode_{ep:06d}_frame_{frame:06d}"
        names = list(archive[f"{prefix}_prompt_names"])
        pred_actions = archive[f"{prefix}_prediction_actions"]
        afro_tracks = {name: _endpoint_tcp_from_actions(None, 0, pred_actions[i], arm) for i, name in enumerate(names)}

        ax = fig.add_subplot(len(requested), 2, row_i * 2 + 1, projection="3d")
        ax2 = fig.add_subplot(len(requested), 2, row_i * 2 + 2)
        for model, tracks, ls in (("PI0.5", plain_tracks, "--"), ("AFRO", afro_tracks, "-")):
            for fruit in FRUITS:
                if fruit not in available:
                    continue
                track = np.asarray(tracks[fruit])
                target = np.asarray(available[fruit]["xyz"])
                width = 3.0 if fruit == native else 1.25
                alpha = 1.0 if fruit == native else 0.72
                ax.plot(*track.T, color=COLORS[fruit], ls=ls, lw=width, alpha=alpha, label=f"{model}: {fruit}")
                if model == "AFRO":
                    ax.scatter(*target, color=COLORS[fruit], marker="*", s=120)
                distance = np.linalg.norm(track - target, axis=1) * 1000
                decay = float(distance[0] - distance.min())
                ax2.plot(np.arange(1, len(distance) + 1), distance, color=COLORS[fruit], ls=ls, lw=width, alpha=alpha, label=f"{model}: {fruit} (drop {decay:.1f} mm)")
                report.append({"model": model, "episode": ep, "frame": frame, "native_prompt": native, "prompt_fruit": fruit, "initial_distance_mm": float(distance[0]), "minimum_distance_mm": float(distance.min()), "distance_decay_mm": decay, "minimum_step": int(distance.argmin() + 1)})
        ax.set_title(f"episode {ep} frame {frame} · active {arm} · native: {native}\nPredicted TCP only; stars are named-fruit positions")
        ax.set_xlabel("base x (m)"); ax.set_ylabel("base y (m)"); ax.set_zlabel("base z (m)")
        ax2.set_title("Distance from each predicted trajectory to its named fruit")
        ax2.set_xlabel("prediction step"); ax2.set_ylabel("distance (mm)"); ax2.grid(alpha=.25)
        ax2.legend(fontsize=7, ncol=2, frameon=False)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
