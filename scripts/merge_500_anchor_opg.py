#!/usr/bin/env python3
"""Merge OPG shards and summarize target-direction steering at TCP step 50."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _component(atom: str) -> tuple[int, float, float]:
    family, axis, sign_name = atom.split("_")
    index = "xyz".index(axis) + (3 if family == "rotate" else 0)
    sign = 1.0 if sign_name == "pos" else -1.0
    threshold = 1.0 if family == "rotate" else 5.0
    return index, sign, threshold


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.input_dir.glob("shard_*.json"))
    ]
    if len(payloads) != 4:
        raise RuntimeError(f"expected four shards, found {len(payloads)}")
    outputs = [row for payload in payloads for row in payload["outputs"]]
    baselines: dict[tuple[str, str], dict[str, float]] = {}
    trials: list[dict] = []
    for row in outputs:
        twist = np.asarray(row["trajectory_twist"]["50"], dtype=float)
        signed = [
            sign * twist[index]
            for atom in row["target_atoms"]
            for index, sign, _ in [_component(atom)]
        ]
        thresholds = [_component(atom)[2] for atom in row["target_atoms"]]
        record = {
            "anchor_key": row["anchor_key"],
            "arm": row["arm"],
            "kind": row["kind"],
            "variant": row["variant"],
            "gamma": float(row["gamma"]),
            "target_atoms": "+".join(row["target_atoms"]),
            "direction_success": bool(all(value > 0.0 for value in signed)),
            "strict_success": bool(
                all(value > threshold for value, threshold in zip(signed, thresholds, strict=True))
            ),
            "minimum_signed_component": float(min(signed)),
            "mean_signed_component": float(np.mean(signed)),
        }
        trials.append(record)
        if float(row["gamma"]) == 0.0:
            baselines[(row["anchor_key"], row["variant"])] = {
                "minimum": record["minimum_signed_component"],
                "mean": record["mean_signed_component"],
            }
    for record in trials:
        baseline = baselines[(record["anchor_key"], record["variant"])]
        record["minimum_gain_vs_gamma0"] = record["minimum_signed_component"] - baseline["minimum"]
        record["mean_gain_vs_gamma0"] = record["mean_signed_component"] - baseline["mean"]
        record["all_components_improved_vs_gamma0"] = record["minimum_gain_vs_gamma0"] > 0.0

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in trials:
        grouped[(row["arm"], row["kind"], row["variant"], row["gamma"])].append(row)
    summary: list[dict] = []
    for (arm, kind, variant, gamma), rows in sorted(grouped.items()):
        summary.append(
            {
                "arm": arm,
                "kind": kind,
                "variant": variant,
                "gamma": gamma,
                "n": len(rows),
                "direction_success_rate": float(np.mean([row["direction_success"] for row in rows])),
                "strict_success_rate": float(np.mean([row["strict_success"] for row in rows])),
                "mean_minimum_signed_component": float(np.mean([row["minimum_signed_component"] for row in rows])),
                "mean_gain_vs_gamma0": float(np.mean([row["mean_gain_vs_gamma0"] for row in rows])),
                "all_components_improved_rate": float(
                    np.mean([row["all_components_improved_vs_gamma0"] for row in rows])
                ),
            }
        )
    _write_csv(args.input_dir / "opg_trials.csv", trials)
    _write_csv(args.input_dir / "opg_summary.csv", summary)
    (args.input_dir / "opg_merged.json").write_text(
        json.dumps(
            {
                "rows": len(trials),
                "anchors": len({row["anchor_key"] for row in trials}),
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(trials), "summary": str(args.input_dir / "opg_summary.csv")}, indent=2))


if __name__ == "__main__":
    main()
