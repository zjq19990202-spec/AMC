#!/usr/bin/env python3
"""Summarize paired multi-episode fruit target-switch reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parse(value: str) -> tuple[str, Path]:
    label, path = value.split("=", 1)
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, help="LABEL=report.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports = [
        (label, path, json.loads(path.read_text(encoding="utf-8")))
        for label, path in map(_parse, args.report)
    ]
    reference = reports[0][2]
    for label, _, report in reports[1:]:
        for key in ("selection_manifest_sha256", "evaluation_step", "prompt_style"):
            if report.get(key) != reference.get(key):
                raise ValueError(f"{label} differs on {key}")
        if report["evaluation_contract"] != reference["evaluation_contract"]:
            raise ValueError(f"{label} has a different evaluation contract")

    detail_rows = []
    for label, path, report in reports:
        for row in report["results"]:
            native = row["native_target"]
            detail_rows.append(
                {
                    "checkpoint": label,
                    "report": str(path),
                    "episode": row["episode"],
                    "frame": row["frame"],
                    "arm": row["active_arm"],
                    "native_target": native,
                    "native_rank": row["native_target_gt_error_rank"],
                    "native_endpoint_error_mm": row["gt_endpoint_error_mm"][native],
                    "native_trajectory_rmse_mm": row["gt_trajectory_rmse_mm"][native],
                    "native_margin_mm": row["native_target_margin_vs_best_wrong_mm"],
                    "mean_pairwise_cosine": row["mean_pairwise_displacement_cosine"],
                    "max_endpoint_separation_mm": row["max_endpoint_separation_mm"],
                    "native_prompt": row["original_subtask_prompt"],
                }
            )

    fields = list(detail_rows[0])
    with (args.output_dir / "fruit_sample_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_rows = []
    for label, _, _ in reports:
        rows = [row for row in detail_rows if row["checkpoint"] == label]
        for arm in ("all", "left", "right"):
            selected = rows if arm == "all" else [row for row in rows if row["arm"] == arm]
            summary_rows.append(
                {
                    "checkpoint": label,
                    "arm": arm,
                    "samples": len(selected),
                    "native_top1_rate": float(np.mean([row["native_rank"] == 1 for row in selected])),
                    "native_endpoint_error_mm": float(np.mean([row["native_endpoint_error_mm"] for row in selected])),
                    "native_trajectory_rmse_mm": float(np.mean([row["native_trajectory_rmse_mm"] for row in selected])),
                    "native_margin_mm": float(np.mean([row["native_margin_mm"] for row in selected])),
                    "mean_pairwise_cosine": float(np.mean([row["mean_pairwise_cosine"] for row in selected])),
                    "max_endpoint_separation_mm": float(np.mean([row["max_endpoint_separation_mm"] for row in selected])),
                }
            )
    with (args.output_dir / "fruit_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    labels = [item[0] for item in reports]
    targets = [
        row["native_target"]
        for row in detail_rows
        if row["checkpoint"] == labels[0]
    ]
    x = np.arange(len(targets))
    width = 0.8 / len(labels)
    figure, axes = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)
    for index, label in enumerate(labels):
        rows = [row for row in detail_rows if row["checkpoint"] == label]
        rows.sort(key=lambda row: (row["episode"], row["frame"]))
        offset = (index - (len(labels) - 1) / 2) * width
        axes[0].bar(x + offset, [row["native_endpoint_error_mm"] for row in rows], width, label=label)
        axes[1].bar(x + offset, [row["native_margin_mm"] for row in rows], width, label=label)
    for axis, title, ylabel in (
        (axes[0], "Native-target endpoint error", "mm (lower is better)"),
        (axes[1], "Native-target margin over best wrong target", "mm (higher is better)"),
    ):
        axis.set_xticks(x, targets, rotation=20, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    plot = args.output_dir / "fruit_checkpoint_comparison.png"
    figure.savefig(plot, dpi=190, bbox_inches="tight")
    plt.close(figure)

    markdown = [
        "# Multi-episode fruit target comparison",
        "",
        "The native sidecar prompt is used unchanged. Counterfactual prompts preserve its template and replace only the fruit target name.",
        "",
        "| Checkpoint | Arm | N | Native top-1 | Endpoint error (mm) | Trajectory RMSE (mm) | Margin vs best wrong (mm) | Pairwise cosine |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        markdown.append(
            f"| {row['checkpoint']} | {row['arm']} | {row['samples']} | "
            f"{100 * row['native_top1_rate']:.1f}% | {row['native_endpoint_error_mm']:.2f} | "
            f"{row['native_trajectory_rmse_mm']:.2f} | {row['native_margin_mm']:.2f} | "
            f"{row['mean_pairwise_cosine']:.3f} |"
        )
    (args.output_dir / "RESULTS.md").write_text("\n".join(markdown) + "\n")
    (args.output_dir / "comparison_contract.json").write_text(
        json.dumps(
            {
                "selection_manifest_sha256": reference["selection_manifest_sha256"],
                "evaluation_contract": reference["evaluation_contract"],
                "reports": {label: str(path) for label, path, _ in reports},
                "plot": str(plot),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
