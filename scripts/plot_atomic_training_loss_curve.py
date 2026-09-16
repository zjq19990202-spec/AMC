#!/usr/bin/env python3
"""Plot Atomic PI0.5 training-loss history from the persistent text log."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=np.float64)
    if len(values) >= window:
        result[window - 1 :] = np.convolve(values, np.ones(window) / window, mode="valid")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--rolling-reports", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    rows = []
    for line in args.log.open(encoding="utf-8", errors="ignore"):
        if " INFO step=" not in line:
            continue
        values = {key: value for key, value in re.findall(r"(\w+)=([^ ]+)", line)}
        try:
            values["step"] = int(values["step"])
            values["timestamp"] = line[:23]
        except (KeyError, ValueError):
            continue
        rows.append(values)
    if not rows:
        raise RuntimeError("no training rows found")

    steps = np.asarray([row["step"] for row in rows], dtype=np.int64)
    metrics = (
        "loss",
        "flow_loss",
        "flow_active_loss",
        "flow_left_loss",
        "flow_right_loss",
        "full_weighted_projection_loss",
        "full_weighted_atomic_total_loss",
        "weighted_counterfactual_margin_loss",
        "counterfactual_violation_fraction",
        "full_text_teacher_cosine_similarity",
    )
    arrays = {
        metric: np.asarray([float(row.get(metric, "nan")) for row in rows], dtype=np.float64)
        for metric in metrics
    }
    checkpoints = sorted(
        int(path.name)
        for path in args.checkpoint_dir.iterdir()
        if path.is_dir() and path.name.isdigit() and (path / "params" / "_METADATA").is_file()
    )

    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    axis = axes[0, 0]
    for metric, color, label in (("loss", "#d95f02", "total"), ("flow_loss", "#1b9e77", "flow")):
        axis.plot(steps, arrays[metric], color=color, alpha=0.14, linewidth=0.7)
        axis.plot(steps, _rolling(arrays[metric], args.rolling_reports), color=color, linewidth=2.0, label=f"{label} (200-step mean)")
    axis.set_title("Overall loss")
    axis.set_ylabel("loss")
    axis.legend(frameon=False)

    axis = axes[0, 1]
    for metric, color, label in (
        ("flow_left_loss", "#377eb8", "left flow"),
        ("flow_right_loss", "#e41a1c", "right flow"),
        ("flow_active_loss", "#984ea3", "active flow diagnostic"),
    ):
        axis.plot(steps, _rolling(arrays[metric], args.rolling_reports), color=color, linewidth=1.8, label=label)
    axis.set_title("Per-arm and active-flow diagnostics")
    axis.set_ylabel("loss")
    axis.legend(frameon=False)

    axis = axes[1, 0]
    for metric, color, label in (
        ("full_weighted_projection_loss", "#7570b3", "weighted projection"),
        ("full_weighted_atomic_total_loss", "#66a61e", "weighted Q teacher"),
        ("weighted_counterfactual_margin_loss", "#e7298a", "weighted counterfactual"),
    ):
        values = _rolling(arrays[metric], args.rolling_reports)
        axis.plot(steps, np.maximum(values, 1e-8), color=color, linewidth=1.8, label=label)
    axis.set_yscale("log")
    axis.set_title("Weighted auxiliary contributions")
    axis.set_ylabel("weighted loss (log scale)")
    axis.legend(frameon=False)

    axis = axes[1, 1]
    axis.plot(steps, _rolling(arrays["counterfactual_violation_fraction"], args.rolling_reports), color="#e7298a", linewidth=1.8, label="counterfactual violation")
    axis.plot(steps, _rolling(arrays["full_text_teacher_cosine_similarity"], args.rolling_reports), color="#1b9e77", linewidth=1.8, label="teacher cosine similarity")
    axis.set_ylim(0, 1.03)
    axis.set_title("Counterfactual and teacher diagnostics")
    axis.legend(frameon=False)

    for axis in axes.flat:
        axis.grid(alpha=0.22)
        axis.set_xlabel("training step")
        for checkpoint in checkpoints:
            if steps.min() <= checkpoint <= steps.max():
                axis.axvline(checkpoint, color="black", linestyle="--", alpha=0.28, linewidth=0.9)
    figure.suptitle(
        f"Atomic AFRO ZM training — steps {steps[0]:,}–{steps[-1]:,}; raw points every 10 steps, rolling={args.rolling_reports * 10} steps",
        fontsize=14,
    )
    figure.savefig(args.output_dir / "training_loss_curve.png", dpi=190, bbox_inches="tight")
    plt.close(figure)

    with (args.output_dir / "training_loss_curve.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["step", "timestamp", *metrics, *(f"{metric}_rolling" for metric in metrics)]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            output = {"step": steps[index], "timestamp": row["timestamp"]}
            for metric in metrics:
                output[metric] = arrays[metric][index]
                output[f"{metric}_rolling"] = _rolling(arrays[metric], args.rolling_reports)[index]
            writer.writerow(output)
    print(f"rows={len(rows)} range={steps[0]}..{steps[-1]} checkpoints={checkpoints}")


if __name__ == "__main__":
    main()
