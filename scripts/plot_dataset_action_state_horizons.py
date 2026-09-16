#!/usr/bin/env python3
"""Plot recorded dataset action and measured state over consecutive horizons."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NAMES = [
    "left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_j6", "left_j7", "left_gripper",
    "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6", "right_j7", "right_gripper",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset-label", default="Dataset")
    args = parser.parse_args()
    data = np.load(args.input)
    state, action, frame = data["state"], data["action"], data["frame"]
    episode = int(data["episode"])
    fig, axes = plt.subplots(4, 4, figsize=(19, 13), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(frame, state[:, i], color="black", lw=1.5, label="measured state")
        ax.plot(frame, action[:, i], color="#e45756", lw=1.0, alpha=0.9, label="recorded action")
        for boundary in frame[0] + np.arange(0, len(frame), 50):
            ax.axvline(boundary, color="#4c78a8", lw=0.7, ls="--", alpha=0.7)
        ax.set_title(NAMES[i], fontsize=9)
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"{args.dataset_label} episode {episode}: recorded action vs measured state (three consecutive 50-step horizons)")
    fig.supxlabel("dataset frame; blue dashed lines = 50-step boundaries")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
