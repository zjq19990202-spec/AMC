#!/usr/bin/env python3
"""Plot a complete episode assembled from RTC-updated 50-step windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OFFSETS = (0, 10, 20, 30, 40)


def _assemble(data: np.lib.npyio.NpzFile, prefix: str) -> np.ndarray:
    blocks = []
    for window in range(data["gt_decoded"].shape[0]):
        pieces = [data[f"{prefix}offset{offset}"][window, offset : offset + 10] for offset in OFFSETS]
        blocks.append(np.concatenate(pieces, axis=0))
    return np.concatenate(blocks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="RTC complete-episode action trajectory")
    args = parser.parse_args()

    data = np.load(args.input)
    gt = np.concatenate(list(data["gt_decoded"]), axis=0)
    series = {
        "GT": gt,
        "Correct force": _assemble(data, "prediction_"),
        "Wrong force": _assemble(data, "wrong_force_prediction_"),
        "Force disabled": _assemble(data, "zero_force_prediction_"),
    }
    colors = {"GT": "#111827", "Correct force": "#16a34a", "Wrong force": "#ea580c", "Force disabled": "#2563eb"}
    styles = {"GT": "-", "Correct force": "-", "Wrong force": "--", "Force disabled": ":"}
    labels = [f"R-J{i + 1}" for i in range(7)] + ["R-grip"] + [f"L-J{i + 1}" for i in range(7)] + ["L-grip"]

    figure, axes = plt.subplots(4, 4, figsize=(22, 13), sharex=True, constrained_layout=True)
    x = np.arange(gt.shape[0])
    for dim, axis in enumerate(axes.flat):
        for name, values in series.items():
            axis.plot(x, values[:, dim], color=colors[name], ls=styles[name], lw=1.0 if name != "GT" else 1.35, alpha=0.9, label=name)
        for boundary in range(50, gt.shape[0], 50):
            axis.axvline(boundary, color="#9ca3af", lw=0.35, alpha=0.35)
        axis.set_title(labels[dim])
        axis.grid(alpha=0.16)
        if dim // 4 == 3:
            axis.set_xlabel("Episode action step (30 Hz)")
        axis.set_ylabel("joint rad" if dim not in (7, 15) else "gripper")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.978),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(args.title, y=0.998, fontsize=16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
