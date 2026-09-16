#!/usr/bin/env python3
"""Merge four atomic-sweep shards and score TCP steering against no-atom baselines."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


TWIST_NAMES = ("x_mm", "y_mm", "z_mm", "rx_deg", "ry_deg", "rz_deg")


def _component(atom: str) -> tuple[int, float]:
    family, axis, sign_name = atom.split("_")
    index = "xyz".index(axis) + (3 if family == "rotate" else 0)
    return index, 1.0 if sign_name == "pos" else -1.0


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_bool(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def _plot_summary(rows: list[dict], output: Path, checkpoint_name: str) -> None:
    import matplotlib.pyplot as plt

    table = {
        (row["arm"], row["scheme"], int(row["step"])): row
        for row in rows
        if row["other_status"] == "all" and row["arm"] in ("right", "left")
    }
    categories = (("right", "single"), ("right", "dual"), ("left", "single"), ("left", "dual"))
    labels = ("right\nsingle", "right\ndual", "left\nsingle", "left\ndual")
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, constrained_layout=True)
    x = np.arange(len(categories))
    width = 0.25
    for axis, step in zip(axes, (25, 50), strict=True):
        selected = [table[(arm, scheme, step)] for arm, scheme in categories]
        axis.bar(
            x - width,
            [row["empty_direction_match_rate"] for row in selected],
            width,
            label="empty prompt direction bias",
            color="#9CA3AF",
        )
        axis.bar(
            x,
            [row["prompt_absolute_success_rate"] for row in selected],
            width,
            label="atomic prompt absolute match",
            color="#2563EB",
        )
        axis.bar(
            x + width,
            [row["steer_vs_empty_success_rate"] for row in selected],
            width,
            label="atomic displacement vs empty",
            color="#F59E0B",
        )
        axis.set_xticks(x, labels)
        axis.set_ylim(0.0, 0.7)
        axis.set_title(f"TCP endpoint at horizon step {step}")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("strict success rate")
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    figure.suptitle(
        f"500-anchor canonical atomic sweep · {checkpoint_name} · TCP 0.20 m",
        fontweight="bold",
    )
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--translation-threshold-mm", type=float, default=5.0)
    parser.add_argument("--rotation-threshold-deg", type=float, default=1.0)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = sorted(args.input_dir.glob("shard_*.json"))
    if len(shard_paths) != 4:
        raise RuntimeError(f"expected four shards, found {len(shard_paths)}")
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in shard_paths]
    outputs = [row for shard in shards for row in shard["outputs"]]
    by_anchor: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in outputs:
        by_anchor[row["anchor_key"]][row["variant"]] = row
    if len(by_anchor) != 500:
        raise RuntimeError(f"expected 500 anchors, found {len(by_anchor)}")
    variant_counts = {len(rows) for rows in by_anchor.values()}
    if variant_counts != {76}:
        raise RuntimeError(f"expected 76 variants per anchor, found {variant_counts}")

    endpoint_rows = []
    empty_rows = []
    metric_rows = []
    for anchor_key, variants in sorted(by_anchor.items()):
        empty = variants["empty"]
        subtask = variants["subtask_no_atom"]
        for variant, row in variants.items():
            flat = {
                "anchor_key": anchor_key,
                "cluster_key": row["cluster_key"],
                "episode": row["episode"],
                "frame": row["frame"],
                "arm": row["arm"],
                "native_kind": row["kind"],
                "other_status": row["other_status"],
                "variant": variant,
                "requested_atoms": "+".join(row["requested_atoms"]),
                "prompt": row["prompt"],
            }
            for step in (25, 50):
                twist = row["trajectory_twist"][str(step)]
                for name, value in zip(TWIST_NAMES, twist, strict=True):
                    flat[f"s{step}_{name}"] = value
            endpoint_rows.append(flat)
        empty_flat = {
            "anchor_key": anchor_key,
            "cluster_key": empty["cluster_key"],
            "episode": empty["episode"],
            "frame": empty["frame"],
            "arm": empty["arm"],
            "native_kind": empty["kind"],
            "other_status": empty["other_status"],
        }
        for baseline_name, baseline in (("empty", empty), ("subtask", subtask)):
            for step in (25, 50):
                for name, value in zip(
                    TWIST_NAMES, baseline["trajectory_twist"][str(step)], strict=True
                ):
                    empty_flat[f"{baseline_name}_s{step}_{name}"] = value
        empty_rows.append(empty_flat)

        for variant, row in variants.items():
            if not variant.startswith("canonical_"):
                continue
            atoms = row["requested_atoms"]
            scheme = "single" if len(atoms) == 1 else "dual"
            for step in (25, 50):
                predicted = np.asarray(row["trajectory_twist"][str(step)], dtype=float)
                empty_twist = np.asarray(empty["trajectory_twist"][str(step)], dtype=float)
                subtask_twist = np.asarray(subtask["trajectory_twist"][str(step)], dtype=float)
                absolute, empty_match, subtask_match = [], [], []
                versus_empty, versus_subtask = [], []
                absolute_margins, empty_delta_margins = [], []
                for atom in atoms:
                    component, sign = _component(atom)
                    threshold = (
                        args.translation_threshold_mm
                        if component < 3
                        else args.rotation_threshold_deg
                    )
                    absolute_value = sign * predicted[component]
                    empty_value = sign * empty_twist[component]
                    subtask_value = sign * subtask_twist[component]
                    empty_delta = sign * (predicted[component] - empty_twist[component])
                    subtask_delta = sign * (predicted[component] - subtask_twist[component])
                    absolute.append(absolute_value > threshold)
                    empty_match.append(empty_value > threshold)
                    subtask_match.append(subtask_value > threshold)
                    versus_empty.append(empty_delta > threshold)
                    versus_subtask.append(subtask_delta > threshold)
                    absolute_margins.append(float(absolute_value - threshold))
                    empty_delta_margins.append(float(empty_delta - threshold))
                metric_rows.append(
                    {
                        "anchor_key": anchor_key,
                        "cluster_key": row["cluster_key"],
                        "episode": row["episode"],
                        "frame": row["frame"],
                        "arm": row["arm"],
                        "native_kind": row["kind"],
                        "other_status": row["other_status"],
                        "scheme": scheme,
                        "variant": variant,
                        "requested_atoms": "+".join(atoms),
                        "step": step,
                        "prompt_absolute_success": bool(all(absolute)),
                        "empty_direction_match": bool(all(empty_match)),
                        "subtask_direction_match": bool(all(subtask_match)),
                        "steer_vs_empty_success": bool(all(versus_empty)),
                        "steer_vs_subtask_success": bool(all(versus_subtask)),
                        "corrected_empty_bias": bool(all(absolute) and not all(empty_match)),
                        "minimum_absolute_margin": min(absolute_margins),
                        "minimum_steer_vs_empty_margin": min(empty_delta_margins),
                    }
                )

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in metric_rows:
        for arm in (row["arm"], "both"):
            for status in (row["other_status"], "all"):
                grouped[(arm, row["scheme"], status, row["step"])].append(row)
    summary_rows = []
    for (arm, scheme, status, step), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "arm": arm,
                "scheme": scheme,
                "other_status": status,
                "step": step,
                "trials": len(rows),
                "anchors": len({row["anchor_key"] for row in rows}),
                "prompt_absolute_success_rate": _mean_bool(rows, "prompt_absolute_success"),
                "empty_direction_match_rate": _mean_bool(rows, "empty_direction_match"),
                "absolute_net_gain_over_empty": _mean_bool(rows, "prompt_absolute_success")
                - _mean_bool(rows, "empty_direction_match"),
                "steer_vs_empty_success_rate": _mean_bool(rows, "steer_vs_empty_success"),
                "steer_vs_subtask_success_rate": _mean_bool(rows, "steer_vs_subtask_success"),
                "corrected_empty_bias_rate": _mean_bool(rows, "corrected_empty_bias"),
                "median_minimum_steer_vs_empty_margin": float(
                    np.median([row["minimum_steer_vs_empty_margin"] for row in rows])
                ),
            }
        )

    variant_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in metric_rows:
        for arm in (row["arm"], "both"):
            variant_groups[(arm, row["scheme"], row["variant"], row["step"])].append(row)
    variant_rows = []
    for (arm, scheme, variant, step), rows in sorted(variant_groups.items()):
        variant_rows.append(
            {
                "arm": arm,
                "scheme": scheme,
                "variant": variant,
                "step": step,
                "trials": len(rows),
                "prompt_absolute_success_rate": _mean_bool(rows, "prompt_absolute_success"),
                "empty_direction_match_rate": _mean_bool(rows, "empty_direction_match"),
                "absolute_net_gain_over_empty": _mean_bool(rows, "prompt_absolute_success")
                - _mean_bool(rows, "empty_direction_match"),
                "steer_vs_empty_success_rate": _mean_bool(rows, "steer_vs_empty_success"),
                "corrected_empty_bias_rate": _mean_bool(rows, "corrected_empty_bias"),
            }
        )

    _write_csv(output_dir / "all_500x76_endpoints.csv", endpoint_rows)
    _write_csv(output_dir / "empty_and_subtask_endpoints_500.csv", empty_rows)
    _write_csv(output_dir / "canonical_trial_metrics.csv", metric_rows)
    _write_csv(output_dir / "canonical_summary.csv", summary_rows)
    _write_csv(output_dir / "canonical_variant_summary.csv", variant_rows)
    checkpoint_name = shards[0].get("checkpoint_name", "checkpoint")
    _plot_summary(
        summary_rows,
        output_dir / "canonical_vs_no_atom.png",
        checkpoint_name,
    )
    report = {
        "checkpoint": shards[0]["checkpoint"],
        "checkpoint_name": checkpoint_name,
        "anchor_count": len(by_anchor),
        "variants_per_anchor": 76,
        "trajectory_count": len(outputs),
        "stored_endpoint_count": len(outputs) * 2,
        "canonical_trial_count": len(metric_rows),
        "thresholds": {
            "translation_mm": args.translation_threshold_mm,
            "rotation_deg": args.rotation_threshold_deg,
        },
        "summary": summary_rows,
    }
    (output_dir / "merged_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
