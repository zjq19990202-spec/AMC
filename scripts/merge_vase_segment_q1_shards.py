#!/usr/bin/env python3
"""Merge sharded outputs from evaluate_vase_segment_q1_timeline.py."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _truth(value: str | bool) -> bool:
    return value is True or str(value).lower() == "true"


def _mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows]
    values = [value for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def _summarize(rows: list[dict]) -> dict:
    supervised = [row for row in rows if row["target_kind"] != "none"]
    predictions = Counter(row["predicted_top1"] for row in rows)
    dominant = predictions.most_common(3)
    result = {
        "count": len(rows),
        "supervised_count": len(supervised),
        "top1_stay_fraction": float(np.mean([row["predicted_top1"] == "stay" for row in rows])),
        "target_support_hit_rate": (
            float(np.mean([_truth(row["top1_in_target_support"]) for row in supervised]))
            if supervised
            else float("nan")
        ),
        "mean_weighted_target_cos": _mean(supervised, "weighted_target_cos"),
        "mean_top1_cos": _mean(rows, "predicted_top1_cos"),
    }
    for index in range(3):
        name, count = dominant[index] if index < len(dominant) else ("", 0)
        result[f"prediction_{index + 1}"] = name
        result[f"prediction_{index + 1}_fraction"] = count / len(rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict] = []
    selections: list[dict] = []
    directions, zms, indices = [], [], []
    codebook = None
    summaries = []
    for shard_index in range(args.num_shards):
        shard = args.input_root / f"shard_{shard_index}"
        summary = json.loads((shard / "summary.json").read_text(encoding="utf-8"))
        selection = json.loads((shard / "selection.json").read_text(encoding="utf-8"))
        summaries.append(summary)
        rows.extend(_read_csv(shard / "per_arm_rows.csv"))
        selections.extend(selection["rows"])
        raw = np.load(shard / "raw_q1.npz")
        directions.append(raw["directions"])
        zms.append(raw["zm"])
        indices.append(raw["dataset_indices"])
        if codebook is None:
            codebook = raw["codebook"]
        elif not np.array_equal(codebook, raw["codebook"]):
            raise ValueError(f"codebook differs in shard {shard_index}")

    global_counts = {int(summary["global_selection_count"]) for summary in summaries}
    checkpoints = {summary["checkpoint"] for summary in summaries}
    if len(global_counts) != 1 or len(checkpoints) != 1:
        raise ValueError(f"inconsistent shards: global_counts={global_counts}, checkpoints={checkpoints}")
    expected = global_counts.pop()
    if len(selections) != expected or len(rows) != expected * 2:
        raise ValueError(f"incomplete merge: selections={len(selections)}/{expected}, arm_rows={len(rows)}/{expected * 2}")
    keys = [(int(row["episode"]), int(row["segment_index"]), int(row["frame"])) for row in selections]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate selected segment anchors across shards")

    selections.sort(key=lambda row: (int(row["episode"]), int(row["segment_index"]), float(row["phase"])))
    rows.sort(
        key=lambda row: (
            int(row["episode"]),
            int(row["segment_index"]),
            float(row["phase"]),
            row["arm"],
        )
    )
    _write_csv(args.output_dir / "per_arm_rows.csv", rows)
    (args.output_dir / "selection.json").write_text(
        json.dumps(
            {
                "selection_basis": "all Vase semantic segment occurrences at 10%, 50%, 90% phase",
                "selected": len(selections),
                "num_shards": args.num_shards,
                "rows": selections,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output_dir / "raw_q1.npz",
        directions=np.concatenate(directions),
        zm=np.concatenate(zms),
        codebook=codebook,
        dataset_indices=np.concatenate(indices),
    )

    phase_groups = defaultdict(list)
    subtask_groups = defaultdict(list)
    for row in rows:
        phase_groups[(row["subtask"], row["arm"], float(row["phase"]))].append(row)
        subtask_groups[(row["subtask"], row["arm"])].append(row)
    phase_rows = [
        {"subtask": key[0], "arm": key[1], "phase": key[2], **_summarize(group)}
        for key, group in sorted(phase_groups.items())
    ]
    subtask_rows = [
        {"subtask": key[0], "arm": key[1], **_summarize(group)}
        for key, group in sorted(subtask_groups.items())
    ]
    _write_csv(args.output_dir / "subtask_arm_phase_summary.csv", phase_rows)
    _write_csv(args.output_dir / "subtask_arm_summary.csv", subtask_rows)

    summary = {
        "checkpoint": next(iter(checkpoints)),
        "selection_count": len(selections),
        "segment_occurrences": len({(int(row["episode"]), int(row["segment_index"])) for row in selections}),
        "per_arm_rows": len(rows),
        **_summarize(rows),
        "subtask_arm_rows": subtask_rows,
        "subtask_arm_phase_rows": phase_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    subtasks = list(dict.fromkeys(row["subtask"] for row in phase_rows))
    figure, axes = plt.subplots(len(subtasks), 2, figsize=(13, 3.2 * len(subtasks)), constrained_layout=True)
    axes = np.asarray(axes).reshape(len(subtasks), 2)
    for row_index, subtask in enumerate(subtasks):
        for arm_index, arm in enumerate(("right", "left")):
            axis = axes[row_index, arm_index]
            group = [row for row in phase_rows if row["subtask"] == subtask and row["arm"] == arm]
            group.sort(key=lambda row: row["phase"])
            axis.plot([row["phase"] for row in group], [row["top1_stay_fraction"] for row in group], "o-", label="predicted stay")
            axis.plot([row["phase"] for row in group], [row["target_support_hit_rate"] for row in group], "s-", label="Top1 in target")
            axis.set_ylim(-0.03, 1.03)
            axis.set_xticks((0.1, 0.5, 0.9), ("10%", "50%", "90%"))
            axis.grid(alpha=0.25)
            axis.set_title(f"{arm}: {subtask}", fontsize=9)
            if row_index == 0 and arm_index == 0:
                axis.legend(frameon=False)
    figure.savefig(args.output_dir / "q1_segment_phase_summary.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    shutil.copy2(args.input_root / "shard_0" / "run_contract.md", args.output_dir / "run_contract_shard0.md")
    print(json.dumps({key: value for key, value in summary.items() if not key.endswith("rows")}, indent=2))


if __name__ == "__main__":
    main()
