#!/usr/bin/env python3
"""Compare paired Fruit steering outputs at several physical thresholds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

import evaluate_isolated_arm_atomic_steering as steering


def _load_outputs(path: Path) -> tuple[str, list[dict], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if len(payload["outputs"]) != 1:
        raise ValueError(f"expected one checkpoint in {path}")
    name, outputs = next(iter(payload["outputs"].items()))
    return name, outputs, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layerwise", type=Path, required=True)
    parser.add_argument("--final-zm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        _load_outputs(args.layerwise),
        _load_outputs(args.final_zm),
    ]
    thresholds = [(5.0, 1.0), (10.0, 2.0)]
    rows: list[dict] = []
    for translation_mm, rotation_deg in thresholds:
        by_name: dict[str, dict[tuple[str, str, int], dict]] = {}
        for name, outputs, _payload in candidates:
            records = steering._score(  # noqa: SLF001
                outputs,
                translation_mm=translation_mm,
                rotation_deg=rotation_deg,
            )
            summaries = steering._summarize(records)  # noqa: SLF001
            selected = {
                (row["arm"], row["kind"], row["step"]): row
                for row in summaries
                if row["other_status"] == "all"
            }
            by_name[name] = selected
            for key, summary in selected.items():
                arm, kind, step = key
                rows.append(
                    {
                        "checkpoint": name,
                        "translation_threshold_mm": translation_mm,
                        "rotation_threshold_deg": rotation_deg,
                        "arm": arm,
                        "kind": kind,
                        "step": step,
                        "paired_trials": summary["paired_trials"],
                        "pair_steer_success_rate": summary["pair_steer_success_rate"],
                        "atomic_absolute_success_rate": summary[
                            "original_absolute_success_rate"
                        ],
                        "reverse_absolute_success_rate": summary[
                            "reverse_absolute_success_rate"
                        ],
                        "atomic_vs_empty_success_rate": summary[
                            "original_vs_empty_success_rate"
                        ],
                        "reverse_vs_empty_success_rate": summary[
                            "reverse_vs_empty_success_rate"
                        ],
                        "median_minimum_pair_margin": summary[
                            "median_minimum_pair_margin"
                        ],
                    }
                )

    csv_path = args.output_dir / "fruit_steering_route_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    compact = [
        row
        for row in rows
        if row["arm"] == "both" and row["step"] == 50
    ]
    (args.output_dir / "fruit_steering_route_comparison.json").write_text(
        json.dumps(
            {
                "layerwise_source": str(args.layerwise),
                "final_zm_source": str(args.final_zm),
                "contract": (
                    f"same {candidates[0][2]['selection']['anchor_count']} anchors, "
                    "images/state/noise/prompts/FK/norm; native atomic versus exact reverse "
                    "and empty; TCP offset 0.20 m"
                ),
                "rows": rows,
                "step50_both_arms": compact,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    metrics = [
        ("pair_steer_success_rate", "Atomic ↔ exact reverse"),
        ("atomic_vs_empty_success_rate", "Atomic vs empty"),
    ]
    colors = {candidates[0][0]: "#2563eb", candidates[1][0]: "#ef4444"}
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        labels = []
        x = []
        values = []
        bar_colors = []
        cursor = 0
        for threshold in thresholds:
            for kind in ("single", "dual", "all"):
                labels.append(f"{kind}\n{threshold[0]:g}mm/{threshold[1]:g}°")
                for name, _outputs, _payload in candidates:
                    row = next(
                        item
                        for item in compact
                        if item["checkpoint"] == name
                        and item["kind"] == kind
                        and item["translation_threshold_mm"] == threshold[0]
                    )
                    x.append(cursor + (-0.18 if name == candidates[0][0] else 0.18))
                    values.append(100.0 * row[metric])
                    bar_colors.append(colors[name])
                cursor += 1
        axis.bar(x, values, width=0.34, color=bar_colors)
        axis.set_xticks(range(len(labels)), labels, fontsize=8)
        axis.set_ylim(0, 105)
        axis.set_ylabel("success rate (%)")
        axis.set_title(title + " at horizon step 50")
        axis.grid(axis="y", alpha=0.2)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[name], label=name)
        for name, _outputs, _payload in candidates
    ]
    figure.legend(handles=handles, loc="upper center", ncol=2)
    figure.savefig(args.output_dir / "fruit_steering_route_comparison.png", dpi=180)


if __name__ == "__main__":
    main()
