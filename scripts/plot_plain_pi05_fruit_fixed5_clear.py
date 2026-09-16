#!/usr/bin/env python3
"""Render clear orthographic fixed-5 fruit steering plots from evaluator JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COLORS = ("#2563eb", "#f97316", "#16a34a", "#dc2626", "#7c3aed")


def _short_name(prompt: str) -> str:
    for destination in ("right box", "center box", "left box"):
        if destination in prompt.lower():
            return destination
    for target in ("green bitter melon", "red bell pepper", "green radish", "orange", "carrot", "potato"):
        if target in prompt.lower():
            return target
    return prompt


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _equal_2d(ax, tracks: list[np.ndarray], dims: tuple[int, int]) -> None:
    values = np.concatenate([track[:, dims] for track in tracks])
    lo, hi = values.min(axis=0), values.max(axis=0)
    center = (lo + hi) / 2
    radius = max(float(np.max(hi - lo)) / 2 * 1.10, 4.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal", adjustable="box")


def _draw_projection(ax, gt, predictions, dims, labels, native_index, title) -> None:
    tracks = [gt, *predictions]
    ax.plot(gt[:, dims[0]], gt[:, dims[1]], color="black", linewidth=3.2, linestyle="--", zorder=8)
    ax.scatter(gt[-1, dims[0]], gt[-1, dims[1]], color="black", marker="X", s=70, zorder=9)
    for index, track in enumerate(predictions):
        width = 3.0 if index == native_index else 1.5
        alpha = 1.0 if index == native_index else 0.72
        ax.plot(track[:, dims[0]], track[:, dims[1]], color=COLORS[index], linewidth=width, alpha=alpha)
        ax.scatter(track[-1, dims[0]], track[-1, dims[1]], color=COLORS[index], s=36, alpha=alpha)
        ax.annotate(str(index + 1), track[-1, dims], fontsize=8, color=COLORS[index], fontweight="bold")
    _equal_2d(ax, tracks, dims)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(labels[dims[0]])
    ax.set_ylabel(labels[dims[1]])
    ax.grid(alpha=0.20)


def _draw_error(ax, gt, predictions, native_index) -> None:
    steps = np.arange(1, len(gt) + 1)
    for index, track in enumerate(predictions):
        error = np.linalg.norm(track - gt, axis=1)
        ax.plot(
            steps,
            error,
            color=COLORS[index],
            linewidth=3.0 if index == native_index else 1.5,
            alpha=1.0 if index == native_index else 0.72,
        )
    ax.set_title("TCP error to GT", fontsize=10, fontweight="bold")
    ax.set_xlabel("horizon step")
    ax.set_ylabel("error (mm)")
    ax.grid(alpha=0.25)


def _prepare(row: dict) -> tuple[np.ndarray, list[np.ndarray], int]:
    arm = row["active_arm"]
    gt = np.asarray(row[f"gt_{arm}_tcp_trajectory_m"], dtype=np.float64) * 1000.0
    predictions = [
        np.asarray(track, dtype=np.float64) * 1000.0
        for track in row[f"{arm}_tcp_trajectories_m"]
    ]
    native_index = next(i for i, metric in enumerate(row["prompt_metrics"]) if metric["is_native"])
    return gt, predictions, native_index


def main() -> None:
    args = _args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = payload["rows"]
    short_names = tuple(_short_name(prompt) for prompt in payload["prompts"])
    labels = ("x (mm)", "y (mm)", "z (mm)")

    fig, axes = plt.subplots(
        len(rows), 4, figsize=(19, 4.0 * len(rows)), constrained_layout=True, squeeze=False
    )
    for row_index, row in enumerate(rows):
        gt, predictions, native_index = _prepare(row)
        _draw_projection(axes[row_index, 0], gt, predictions, (0, 1), labels, native_index, "top view: XY")
        _draw_projection(axes[row_index, 1], gt, predictions, (0, 2), labels, native_index, "side view: XZ")
        _draw_projection(axes[row_index, 2], gt, predictions, (1, 2), labels, native_index, "front view: YZ")
        _draw_error(axes[row_index, 3], gt, predictions, native_index)
        axes[row_index, 0].set_ylabel(f"ep{row['episode']} f{row['frame']} {row['active_arm']}\n{labels[1]}")
        axes[row_index, 0].text(
            0.02,
            0.98,
            f"native = {short_names[native_index]}",
            transform=axes[row_index, 0].transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
    handles = [Line2D([0], [0], color="black", lw=3, ls="--", label="GT")] + [
        Line2D([0], [0], color=color, lw=2.5, label=f"{index + 1} {name}")
        for index, (name, color) in enumerate(zip(short_names, COLORS))
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, frameon=False)
    fig.suptitle("Plain PI0.5 25K — fixed-5 fruit SUBtask steering (same image/state/noise)", fontsize=15, fontweight="bold")
    fig.savefig(args.output_dir / "plain25k_fruit_fixed5_clear_views.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    row = rows[-1]
    gt, predictions, native_index = _prepare(row)
    native_name = short_names[native_index]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    _draw_projection(axes[0, 0], gt, predictions, (0, 1), labels, native_index, f"{native_name}: top view XY")
    _draw_projection(axes[0, 1], gt, predictions, (0, 2), labels, native_index, f"{native_name}: side view XZ")
    _draw_projection(axes[1, 0], gt, predictions, (1, 2), labels, native_index, f"{native_name}: front view YZ")
    _draw_error(axes[1, 1], gt, predictions, native_index)
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(
        f"Plain PI0.5 25K — episode {row['episode']} frame {row['frame']}\n"
        f"Native {native_name} prompt is thick line; dots are step-50 endpoints",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(args.output_dir / "plain25k_focus_frame_clear_zoom.png", dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
