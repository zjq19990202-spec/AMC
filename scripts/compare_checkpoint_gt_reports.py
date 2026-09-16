#!/usr/bin/env python3
"""Verify and summarize paired multi-checkpoint GT evaluation reports."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    "joint_14d_rmse_deg",
    "joint_14d_endpoint_rmse_deg",
    "left_joint_rmse_deg",
    "right_joint_rmse_deg",
    "left_gripper_rmse_native",
    "right_gripper_rmse_native",
)


def _parse_report(value: str) -> tuple[str, Path]:
    label, path = value.split("=", 1)
    if not label:
        raise ValueError(value)
    return label, Path(path)


def _rms(values: list[float]) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _aggregate(rows: list[dict]) -> dict[str, float]:
    result = {
        metric: _rms([float(row[metric]) for row in rows]) for metric in METRICS
    }
    result["normalized_16d_mse"] = float(
        np.mean([float(row["normalized_16d_mse"]) for row in rows])
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, help="LABEL=report.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for value in args.report:
        label, path = _parse_report(value)
        reports.append((label, path, json.loads(path.read_text(encoding="utf-8"))))
    if len(reports) < 2:
        raise ValueError("at least two reports are required")

    contract_keys = (
        "norm_stats_sha256",
        "input_selection_manifest_sha256",
        "noise_seed",
        "noise_key_mode",
        "flow_ode_steps",
        "max_token_len",
        "action_horizon",
        "action_dim",
    )
    reference = reports[0][2]
    for label, _, report in reports[1:]:
        mismatches = {
            key: (reference.get(key), report.get(key))
            for key in contract_keys
            if reference.get(key) != report.get(key)
        }
        if mismatches:
            raise ValueError(f"incompatible report {label}: {mismatches}")

    sample_rows = []
    keys_by_checkpoint = {}
    for label, path, report in reports:
        checkpoint_keys = set()
        for dataset_name, dataset in report["datasets"].items():
            for sample in dataset["samples"]:
                key = (dataset_name, int(sample["episode"]), int(sample["frame"]))
                checkpoint_keys.add(key)
                row = {
                    "checkpoint": label,
                    "checkpoint_path": report["checkpoint"],
                    "report": str(path),
                    "dataset": dataset_name,
                    "source_dataset": sample.get("source_dataset") or dataset_name,
                    "episode": int(sample["episode"]),
                    "frame": int(sample["frame"]),
                    "active_arm": sample.get("active_arm"),
                    "target": sample.get("target"),
                    "prompt": sample["prompt"],
                    "normalized_16d_mse": float(sample["normalized_16d_mse"]),
                    **{
                        metric: float(sample["metrics"][metric])
                        for metric in METRICS
                    },
                }
                sample_rows.append(row)
        keys_by_checkpoint[label] = checkpoint_keys
    if any(keys != next(iter(keys_by_checkpoint.values())) for keys in keys_by_checkpoint.values()):
        raise ValueError(f"checkpoint sample keys differ: {keys_by_checkpoint}")

    sample_fields = list(sample_rows[0])
    with (args.output_dir / "sample_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_fields)
        writer.writeheader()
        writer.writerows(sample_rows)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    overall_grouped: dict[str, list[dict]] = defaultdict(list)
    for row in sample_rows:
        grouped[(row["checkpoint"], row["source_dataset"])].append(row)
        overall_grouped[row["checkpoint"]].append(row)

    task_rows = []
    for (checkpoint, source), rows in sorted(grouped.items()):
        task_rows.append(
            {
                "checkpoint": checkpoint,
                "source_dataset": source,
                "samples": len(rows),
                "episodes": len({row["episode"] for row in rows}),
                **_aggregate(rows),
            }
        )
    overall_rows = []
    for checkpoint, rows in overall_grouped.items():
        per_task = [row for row in task_rows if row["checkpoint"] == checkpoint]
        aggregate = _aggregate(rows)
        aggregate["macro_task_joint_14d_rmse_deg"] = float(
            np.mean([row["joint_14d_rmse_deg"] for row in per_task])
        )
        overall_rows.append(
            {
                "checkpoint": checkpoint,
                "samples": len(rows),
                "episodes": len({row["episode"] for row in rows}),
                "task_families": len(per_task),
                **aggregate,
            }
        )

    for filename, rows in (
        ("task_summary.csv", task_rows),
        ("overall_summary.csv", overall_rows),
    ):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    labels = [item[0] for item in reports]
    sources = sorted({row["source_dataset"] for row in task_rows})
    x = np.arange(len(sources))
    width = 0.8 / len(labels)
    figure, axes = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)
    for checkpoint_index, label in enumerate(labels):
        rows = {
            row["source_dataset"]: row
            for row in task_rows
            if row["checkpoint"] == label
        }
        offset = (checkpoint_index - (len(labels) - 1) / 2) * width
        axes[0].bar(
            x + offset,
            [rows[source]["joint_14d_rmse_deg"] for source in sources],
            width,
            label=label,
        )
        axes[1].bar(
            x + offset,
            [rows[source]["joint_14d_endpoint_rmse_deg"] for source in sources],
            width,
            label=label,
        )
    for axis, ylabel, title in (
        (axes[0], "degrees", "50-step joint RMSE"),
        (axes[1], "degrees", "Endpoint joint RMSE"),
    ):
        axis.set_xticks(x, sources, rotation=20, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    plot = args.output_dir / "checkpoint_by_task_comparison.png"
    figure.savefig(plot, dpi=190, bbox_inches="tight")
    plt.close(figure)

    markdown = [
        "# Paired checkpoint GT comparison",
        "",
        "All checkpoints use the identical selection manifest, native prompts, normalization, flow noise, and sampler settings.",
        "",
        "| Checkpoint | Samples | Episodes | Joint RMSE (deg) | Endpoint RMSE (deg) | Left joint | Right joint | Left gripper | Right gripper |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall_rows:
        markdown.append(
            f"| {row['checkpoint']} | {row['samples']} | {row['episodes']} | "
            f"{row['joint_14d_rmse_deg']:.3f} | {row['joint_14d_endpoint_rmse_deg']:.3f} | "
            f"{row['left_joint_rmse_deg']:.3f} | {row['right_joint_rmse_deg']:.3f} | "
            f"{row['left_gripper_rmse_native']:.4f} | {row['right_gripper_rmse_native']:.4f} |"
        )
    markdown.extend(
        [
            "",
            "## Per-task joint RMSE",
            "",
            "| Task/source | " + " | ".join(labels) + " |",
            "|---|" + "---:|" * len(labels),
        ]
    )
    lookup = {(row["checkpoint"], row["source_dataset"]): row for row in task_rows}
    for source in sources:
        markdown.append(
            f"| {source} | "
            + " | ".join(
                f"{lookup[(label, source)]['joint_14d_rmse_deg']:.3f}"
                for label in labels
            )
            + " |"
        )
    (args.output_dir / "RESULTS.md").write_text("\n".join(markdown) + "\n")

    contract = {
        "verified_equal_fields": {key: reference.get(key) for key in contract_keys},
        "reports": {label: str(path) for label, path, _ in reports},
        "sample_key_count": len(next(iter(keys_by_checkpoint.values()))),
        "plot": str(plot),
    }
    (args.output_dir / "comparison_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"overall": overall_rows, "plot": str(plot)}, indent=2))


if __name__ == "__main__":
    main()
