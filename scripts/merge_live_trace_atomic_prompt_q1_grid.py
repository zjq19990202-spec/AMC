#!/usr/bin/env python3
"""Merge sharded live atomic-prompt Q1 grid results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def _truth(value: str) -> bool:
    return value.lower() == "true"


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict]) -> dict:
    return {
        "count": len(rows),
        "requested_top1_rate": sum(_truth(row["requested_is_top1"]) for row in rows) / len(rows),
        "requested_top5_rate": sum(_truth(row["requested_is_top5"]) for row in rows) / len(rows),
        "mean_requested_code_cos": sum(float(row["requested_code_cos"]) for row in rows) / len(rows),
        "target_arm_stay_rate": sum(row["target_top1"] == "stay" for row in rows) / len(rows),
        "other_arm_stay_rate": sum(_truth(row["other_top1_is_stay"]) for row in rows) / len(rows),
        "dominant_target_top1": Counter(row["target_top1"] for row in rows).most_common(1)[0][0],
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
    rows.sort(key=lambda row: (int(row["request_id"]), row["requested_arm"], row["requested_atom"]))
    _write_csv(args.output_dir / "rows.csv", rows)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["requested_arm"], row["requested_atom"])].append(row)
    per_atom = [
        {"arm": arm, "atom": atom, **_summary(group)}
        for (arm, atom), group in sorted(groups.items())
    ]
    _write_csv(args.output_dir / "per_atom_summary.csv", per_atom)
    summary = {
        "anchors": len({int(row["request_id"]) for row in rows}),
        "rows": len(rows),
        **_summary(rows),
        "per_atom": per_atom,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "per_atom"}, indent=2))


if __name__ == "__main__":
    main()
