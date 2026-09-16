#!/usr/bin/env python3
"""Merge sharded full atomic-prompt live steering results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _truth(value: str) -> bool:
    return value.lower() == "true"


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict]) -> dict:
    return {
        "count": len(rows),
        "paired_order_success": sum(_truth(row["paired_order_success"]) for row in rows) / len(rows),
        "paired_threshold_success": sum(_truth(row["paired_threshold_success"]) for row in rows) / len(rows),
        "absolute_both_sign_success": sum(_truth(row["absolute_both_sign_success"]) for row in rows) / len(rows),
        "mean_paired_separation": sum(float(row["paired_separation"]) for row in rows) / len(rows),
        "mean_target_joint_rmse_rad": sum(float(row["target_joint_rmse_rad"]) for row in rows) / len(rows),
        "mean_other_joint_rmse_rad": sum(float(row["other_joint_rmse_rad"]) for row in rows) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    rows = []
    for shard_index in range(args.num_shards):
        with (args.input_root / f"shard_{shard_index}" / "rows.csv").open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    rows.sort(key=lambda row: (int(row["request_id"]), int(row["noise_index"]), row["arm"], row["positive_atom"]))
    _write_csv(args.output_dir / "rows.csv", rows)

    axis_groups = defaultdict(list)
    subtask_groups = defaultdict(list)
    arm_groups = defaultdict(list)
    for row in rows:
        axis_groups[(row["arm"], row["positive_atom"], row["unit"])].append(row)
        subtask_groups[row["native_subtask"]].append(row)
        arm_groups[row["arm"]].append(row)
    per_axis = [
        {"arm": arm, "positive_atom": atom, "unit": unit, **_summarize(group)}
        for (arm, atom, unit), group in sorted(axis_groups.items())
    ]
    per_subtask = [
        {"subtask": subtask, **_summarize(group)}
        for subtask, group in sorted(subtask_groups.items())
    ]
    per_arm = [
        {"arm": arm, **_summarize(group)}
        for arm, group in sorted(arm_groups.items())
    ]
    _write_csv(args.output_dir / "per_axis_summary.csv", per_axis)
    _write_csv(args.output_dir / "per_subtask_summary.csv", per_subtask)
    _write_csv(args.output_dir / "per_arm_summary.csv", per_arm)
    summary = {
        "anchors": len({int(row["request_id"]) for row in rows}),
        "noise_repeats": len({int(row["noise_index"]) for row in rows}),
        "rows": len(rows),
        **_summarize(rows),
        "per_arm": per_arm,
        "per_axis": per_axis,
        "per_subtask": per_subtask,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if not isinstance(value, list)}, indent=2))


if __name__ == "__main__":
    main()
