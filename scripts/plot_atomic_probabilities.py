#!/usr/bin/env python3
"""Plot per-segment probabilities over the 12 atomic skills."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ATOMIC_NAMES = (
    "move_x_pos",
    "move_x_neg",
    "move_y_pos",
    "move_y_neg",
    "move_z_pos",
    "move_z_neg",
    "rotate_x_pos",
    "rotate_x_neg",
    "rotate_y_pos",
    "rotate_y_neg",
    "rotate_z_pos",
    "rotate_z_neg",
)
SHORT_NAMES = (
    "move x+",
    "move x-",
    "move y+",
    "move y-",
    "move z+",
    "move z-",
    "rot x+",
    "rot x-",
    "rot y+",
    "rot y-",
    "rot z+",
    "rot z-",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotation", type=Path, help="Final episode annotation JSON")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: next to the annotation)",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix (default: annotation filename stem)",
    )
    return parser.parse_args()


def load_segments(path: Path) -> tuple[list[dict], np.ndarray]:
    document = json.loads(path.read_text(encoding="utf-8"))
    segments = document.get("segments", [])
    if not segments:
        raise ValueError(f"No segments found in {path}")

    probabilities = np.asarray(
        [segment["atomic_probabilities"] for segment in segments], dtype=np.float64
    )
    if probabilities.shape != (len(segments), len(ATOMIC_NAMES)):
        raise ValueError(
            "Expected one 12-value atomic_probabilities vector per segment, "
            f"got {probabilities.shape}"
        )
    return segments, probabilities * 100.0


def selected_indices(segment: dict) -> set[int]:
    return {
        int(target["label"])
        for target in segment.get("atomic_targets", [])
        if 0 <= int(target["label"]) < len(ATOMIC_NAMES)
    }


def plot_bars(segments: list[dict], values: np.ndarray, output: Path) -> None:
    columns = 2
    rows = int(np.ceil(len(segments) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(16, 4.2 * rows), squeeze=False)
    x = np.arange(len(ATOMIC_NAMES))

    for index, axis in enumerate(axes.flat):
        if index >= len(segments):
            axis.axis("off")
            continue

        segment = segments[index]
        selected = selected_indices(segment)
        gate_mode = str(segment.get("gate_mode", "unknown"))
        colors = []
        for atom_index in x:
            if atom_index in selected:
                colors.append("#16a34a")
            elif gate_mode == "drop" and values[index, atom_index] > 0:
                colors.append("#f59e0b")
            else:
                colors.append("#64748b")

        bars = axis.bar(x, values[index], color=colors, width=0.78)
        axis.set_ylim(0, 100)
        axis.set_ylabel("Probability (%)")
        axis.set_xticks(x, SHORT_NAMES, rotation=42, ha="right", fontsize=9)
        axis.grid(axis="y", alpha=0.22)
        axis.set_title(
            f"Segment {segment.get('segment_id', index)} | "
            f"{segment['start_s']:.1f}-{segment['end_s']:.1f} s | {gate_mode.upper()}",
            fontsize=11,
        )
        for bar, value in zip(bars, values[index], strict=True):
            if value > 0:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 1.5,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    figure.suptitle(
        "Atomic-skill probabilities by segment\n"
        "green = retained target, orange = non-retained distribution in dropped segment",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_heatmap(segments: list[dict], values: np.ndarray, output: Path) -> None:
    height = max(4.5, 0.72 * len(segments) + 2.2)
    figure, axis = plt.subplots(figsize=(16, height))
    image = axis.imshow(values, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")

    axis.set_xticks(np.arange(len(ATOMIC_NAMES)), SHORT_NAMES, rotation=42, ha="right")
    row_labels = [
        f"S{s.get('segment_id', i)}  {s['start_s']:.1f}-{s['end_s']:.1f}s  "
        f"[{str(s.get('gate_mode', 'unknown')).upper()}]"
        for i, s in enumerate(segments)
    ]
    axis.set_yticks(np.arange(len(segments)), row_labels)
    axis.set_xlabel("12 atomic skills")
    axis.set_ylabel("Segment")
    axis.set_title("Atomic-skill probability heatmap (%)")

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            text_color = "white" if value >= 45 else "#172554"
            axis.text(
                column,
                row,
                f"{value:.1f}" if value > 0 else "0",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    colorbar = figure.colorbar(image, ax=axis, pad=0.015)
    colorbar.set_label("Probability (%)")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.annotation.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or args.annotation.stem

    segments, probabilities = load_segments(args.annotation)
    bars_path = output_dir / f"{prefix}_atomic_bars.png"
    heatmap_path = output_dir / f"{prefix}_atomic_heatmap.png"
    plot_bars(segments, probabilities, bars_path)
    plot_heatmap(segments, probabilities, heatmap_path)
    print(bars_path)
    print(heatmap_path)


if __name__ == "__main__":
    main()
