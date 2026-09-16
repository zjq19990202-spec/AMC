#!/usr/bin/env python3
"""Plot recorded realtime policy requests: measured state and predicted chunks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


JOINT_NAMES = [
    "left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_j6", "left_j7", "left_gripper",
    "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6", "right_j7", "right_gripper",
]


def load_requests(path: Path):
    grouped: dict[int, list[dict[str, str]]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            grouped.setdefault(int(row["request_id"]), []).append(row)
    requests = []
    for request_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda x: int(x["action_step"]))
        requests.append({
            "request_id": request_id,
            "timestamp": rows[0]["timestamp"],
            "prompt_key": rows[0]["prompt_key"],
            "prompt_revision": int(rows[0]["prompt_revision"]),
            "prompt": rows[0]["prompt"],
            "execution_horizon": int(rows[0]["execution_horizon"]),
            "state": np.array([float(rows[0][f"state_{i}"]) for i in range(16)]),
            "actions": np.array([[float(row[f"action_{i}"]) for i in range(16)] for row in rows]),
        })
    return requests


def norm_stats(values: np.ndarray):
    if len(values) == 0:
        return {"mean": None, "median": None, "p95": None, "max": None}
    norms = np.linalg.norm(values, axis=1)
    return {k: float(v) for k, v in {
        "mean": norms.mean(), "median": np.median(norms), "p95": np.percentile(norms, 95), "max": norms.max()
    }.items()}


def analyze(requests):
    start_errors, realized_end_errors, chunk_jumps, state_jumps = [], [], [], []
    per_boundary = []
    for request in requests:
        start_errors.append(request["actions"][0] - request["state"])
    for previous, current in zip(requests[:-1], requests[1:]):
        previous_executed_end = previous["actions"][min(previous["execution_horizon"], len(previous["actions"])) - 1]
        realized = current["state"] - previous_executed_end
        chunk_jump = current["actions"][0] - previous_executed_end
        state_jump = current["state"] - previous["state"]
        realized_end_errors.append(realized)
        chunk_jumps.append(chunk_jump)
        state_jumps.append(state_jump)
        per_boundary.append({
            "previous_request": previous["request_id"],
            "next_request": current["request_id"],
            "revision_changed": previous["prompt_revision"] != current["prompt_revision"],
            "prompt_key_changed": previous["prompt_key"] != current["prompt_key"],
            "prompt_text_changed": previous["prompt"] != current["prompt"],
            "likely_manual_reset": bool(np.linalg.norm(state_jump) > 0.5),
            "state_jump_l2": float(np.linalg.norm(state_jump)),
            "previous_end_to_next_state_l2": float(np.linalg.norm(realized)),
            "previous_end_to_next_action0_l2": float(np.linalg.norm(chunk_jump)),
            "next_action0_to_state_l2": float(np.linalg.norm(current["actions"][0] - current["state"])),
        })
    arrays = {
        "action0_minus_state": np.asarray(start_errors),
        "next_state_minus_previous_action_end": np.asarray(realized_end_errors),
        "next_action0_minus_previous_action_end": np.asarray(chunk_jumps),
        "next_state_minus_previous_state": np.asarray(state_jumps),
    }
    return {name: norm_stats(value) for name, value in arrays.items()}, per_boundary, arrays


def plot_trajectory(requests, out: Path):
    fig, axes = plt.subplots(4, 4, figsize=(19, 13), sharex=True)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(requests), 1)))
    for dim, ax in enumerate(axes.flat):
        cursor = 0
        for color, request in zip(colors, requests):
            executed = min(request["execution_horizon"], len(request["actions"]))
            actions = request["actions"][:executed, dim]
            x = cursor + np.arange(len(actions))
            ax.plot(x, actions, color=color, lw=1.25, alpha=0.9)
            ax.scatter([cursor], [request["state"][dim]], color="black", s=14, zorder=4)
            ax.plot([cursor, cursor], [request["state"][dim], actions[0]], color="red", lw=0.8, alpha=0.8)
            ax.axvline(cursor, color="0.75", lw=0.45)
            cursor += request["execution_horizon"]
        ax.set_title(JOINT_NAMES[dim], fontsize=9)
        ax.grid(alpha=0.18)
    fig.suptitle("Recorded 40K inference: predicted action chunks and measured request states\nblack dot=measured state; colored line=predicted action; red segment=action[0]-state", fontsize=13)
    fig.supxlabel("concatenated executed-action index (vertical lines are request boundaries)")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_boundaries(requests, arrays, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(18, 9), sharex=False)
    items = [
        ("action0_minus_state", "Action[0] - measured state at same request"),
        ("next_state_minus_previous_action_end", "Next measured state - previous action[-1]"),
        ("next_action0_minus_previous_action_end", "Next action[0] - previous action[-1]"),
        ("next_state_minus_previous_state", "Measured state change between requests"),
    ]
    for ax, (key, title) in zip(axes.flat, items):
        value = arrays[key]
        if value.size:
            ax.plot(value[:, :7], alpha=0.75, lw=1)
            ax.plot(value[:, 8:15], alpha=0.75, lw=1, ls="--")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(title)
        ax.set_xlabel("request boundary")
        ax.set_ylabel("rad (arm joints)")
        ax.grid(alpha=0.2)
    fig.suptitle("Chunk-boundary discontinuities (solid=left arm, dashed=right arm)")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requests = load_requests(args.csv)
    summary, boundaries, arrays = analyze(requests)
    plot_trajectory(requests, args.output_dir / "state_and_action_chunks.png")
    plot_boundaries(requests, arrays, args.output_dir / "chunk_boundary_discontinuities.png")
    report = {
        "source_csv": str(args.csv.resolve()),
        "num_requests": len(requests),
        "num_action_rows": sum(len(r["actions"]) for r in requests),
        "execution_horizons": sorted(set(r["execution_horizon"] for r in requests)),
        "summary_l2_16d": summary,
        "boundaries": boundaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report | {"boundaries": f"{len(boundaries)} rows"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
