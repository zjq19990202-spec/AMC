#!/usr/bin/env python3
"""Merge and score arbitrary-size cluster-seen atomic sweep shards."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


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


def _reverse_atom(atom: str) -> str:
    if atom.endswith("_pos"):
        return atom.removesuffix("_pos") + "_neg"
    if atom.endswith("_neg"):
        return atom.removesuffix("_neg") + "_pos"
    raise ValueError(f"cannot reverse motion atom {atom!r}")


def _score_variant(
    row: dict,
    empty: dict,
    subtask: dict,
    reverse: dict | None,
    *,
    step: int,
    translation_threshold_mm: float,
    rotation_threshold_deg: float,
) -> dict:
    predicted = np.asarray(row["trajectory_twist"][str(step)], dtype=float)
    empty_twist = np.asarray(empty["trajectory_twist"][str(step)], dtype=float)
    subtask_twist = np.asarray(subtask["trajectory_twist"][str(step)], dtype=float)
    atoms = row["requested_atoms"]
    absolute = []
    empty_match = []
    subtask_match = []
    versus_empty = []
    versus_subtask = []
    absolute_margins = []
    empty_delta_margins = []
    pair_margins = []
    for atom in atoms:
        component, sign = _component(atom)
        threshold = (
            translation_threshold_mm if component < 3 else rotation_threshold_deg
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
        if reverse is not None:
            reverse_twist = np.asarray(
                reverse["trajectory_twist"][str(step)], dtype=float
            )
            pair_margins.append(
                float(sign * (predicted[component] - reverse_twist[component]) - threshold)
            )
    return {
        "anchor_key": row["anchor_key"],
        "cluster_key": row["cluster_key"],
        "episode": row["episode"],
        "frame": row["frame"],
        "arm": row["arm"],
        "other_status": row["other_status"],
        "scheme": "single" if len(atoms) == 1 else "dual",
        "variant": row["variant"],
        "requested_atoms": "+".join(atoms),
        "repeat": int(row["repeat"]),
        "step": step,
        "prompt_absolute_success": bool(all(absolute)),
        "empty_direction_match": bool(all(empty_match)),
        "subtask_direction_match": bool(all(subtask_match)),
        "steer_vs_empty_success": bool(all(versus_empty)),
        "steer_vs_subtask_success": bool(all(versus_subtask)),
        "pair_steer_success": (
            bool(all(margin > 0.0 for margin in pair_margins))
            if reverse is not None
            else None
        ),
        "minimum_absolute_margin": min(absolute_margins),
        "minimum_steer_vs_empty_margin": min(empty_delta_margins),
    }


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
    if not shard_paths:
        raise RuntimeError(f"no shard JSON files found in {args.input_dir}")
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in shard_paths]
    expected_shards = int(shards[0]["num_shards"])
    shard_indices = {int(shard["shard_index"]) for shard in shards}
    if len(shards) != expected_shards or shard_indices != set(range(expected_shards)):
        raise RuntimeError(
            f"expected shards 0..{expected_shards - 1}, found {sorted(shard_indices)}"
        )
    if not all(
        bool(shard.get("cluster_seen_only") or shard.get("cluster_seen_all"))
        for shard in shards
    ):
        raise RuntimeError("all inputs must be --cluster-seen-only shards")

    outputs = [row for shard in shards for row in shard["outputs"]]
    grouped: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for row in outputs:
        key = (row["anchor_key"], int(row["repeat"]))
        variant = row["variant"]
        if variant in grouped[key]:
            raise RuntimeError(f"duplicate {key=} {variant=}")
        grouped[key][variant] = row

    metric_rows = []
    mode_counts = []
    for key, variants in sorted(grouped.items()):
        if "empty" not in variants or "subtask_no_atom" not in variants:
            raise RuntimeError(f"missing baseline for {key}")
        atomic_rows = [
            row
            for variant, row in variants.items()
            if variant.startswith("cluster_seen_")
        ]
        if not atomic_rows:
            raise RuntimeError(f"no cluster-seen atomic variants for {key}")
        mode_counts.append(len(atomic_rows))
        for row in atomic_rows:
            reverse_variant = "cluster_seen_" + "+".join(
                _reverse_atom(atom) for atom in row["requested_atoms"]
            )
            for step in (25, 50):
                metric_rows.append(
                    _score_variant(
                        row,
                        variants["empty"],
                        variants["subtask_no_atom"],
                        variants.get(reverse_variant),
                        step=step,
                        translation_threshold_mm=args.translation_threshold_mm,
                        rotation_threshold_deg=args.rotation_threshold_deg,
                    )
                )

    summary_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in metric_rows:
        for arm in (row["arm"], "both"):
            for scheme in (row["scheme"], "all"):
                summary_groups[(arm, scheme, row["step"])].append(row)
    summary_rows = []
    for (arm, scheme, step), rows in sorted(summary_groups.items()):
        paired_rows = [row for row in rows if row["pair_steer_success"] is not None]
        summary_rows.append(
            {
                "arm": arm,
                "scheme": scheme,
                "step": step,
                "trials": len(rows),
                "anchors": len({row["anchor_key"] for row in rows}),
                "clusters": len({row["cluster_key"] for row in rows}),
                "paired_trials": len(paired_rows),
                "pair_steer_success_rate": (
                    _mean_bool(paired_rows, "pair_steer_success")
                    if paired_rows
                    else None
                ),
                "prompt_absolute_success_rate": _mean_bool(
                    rows, "prompt_absolute_success"
                ),
                "empty_direction_match_rate": _mean_bool(
                    rows, "empty_direction_match"
                ),
                "steer_vs_empty_success_rate": _mean_bool(
                    rows, "steer_vs_empty_success"
                ),
                "steer_vs_subtask_success_rate": _mean_bool(
                    rows, "steer_vs_subtask_success"
                ),
                "median_minimum_steer_vs_empty_margin": float(
                    np.median(
                        [row["minimum_steer_vs_empty_margin"] for row in rows]
                    )
                ),
            }
        )

    variant_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in metric_rows:
        for arm in (row["arm"], "both"):
            variant_groups[
                (arm, row["scheme"], row["requested_atoms"], row["step"])
            ].append(row)
    variant_rows = []
    for (arm, scheme, atoms, step), rows in sorted(variant_groups.items()):
        paired_rows = [row for row in rows if row["pair_steer_success"] is not None]
        variant_rows.append(
            {
                "arm": arm,
                "scheme": scheme,
                "requested_atoms": atoms,
                "step": step,
                "trials": len(rows),
                "anchors": len({row["anchor_key"] for row in rows}),
                "clusters": len({row["cluster_key"] for row in rows}),
                "paired_trials": len(paired_rows),
                "pair_steer_success_rate": (
                    _mean_bool(paired_rows, "pair_steer_success")
                    if paired_rows
                    else None
                ),
                "prompt_absolute_success_rate": _mean_bool(
                    rows, "prompt_absolute_success"
                ),
                "steer_vs_empty_success_rate": _mean_bool(
                    rows, "steer_vs_empty_success"
                ),
                "steer_vs_subtask_success_rate": _mean_bool(
                    rows, "steer_vs_subtask_success"
                ),
            }
        )

    _write_csv(output_dir / "cluster_seen_trial_metrics.csv", metric_rows)
    _write_csv(output_dir / "cluster_seen_summary.csv", summary_rows)
    _write_csv(output_dir / "cluster_seen_variant_summary.csv", variant_rows)
    report = {
        "checkpoint": shards[0]["checkpoint"],
        "checkpoint_name": shards[0]["checkpoint_name"],
        "selection_manifest": shards[0]["selection_manifest"],
        "cluster_seen_only": True,
        "anchor_repeat_count": len(grouped),
        "anchor_count": len({key[0] for key in grouped}),
        "cluster_count": len(
            {row["cluster_key"] for row in outputs if "cluster_key" in row}
        ),
        "trajectory_count": len(outputs),
        "cluster_seen_atomic_trial_count": len(metric_rows) // 2,
        "mode_count_per_anchor": {
            "min": min(mode_counts),
            "max": max(mode_counts),
            "mean": float(np.mean(mode_counts)),
        },
        "thresholds": {
            "translation_mm": args.translation_threshold_mm,
            "rotation_deg": args.rotation_threshold_deg,
        },
        "evaluation_contract": {
            "max_token_len": shards[0].get("max_token_len"),
            "coefficient_target_kind": shards[0].get(
                "coefficient_target_kind"
            ),
            "coefficient_target_dim": shards[0].get("coefficient_target_dim"),
            "enable_layerwise_atomic_flow": shards[0].get(
                "enable_layerwise_atomic_flow"
            ),
            "norm_assets_dir": shards[0].get("norm_assets_dir"),
            "norm_asset_id": shards[0].get("norm_asset_id"),
            "atomic_composition_sidecar": shards[0].get(
                "atomic_composition_sidecar"
            ),
            "tcp_offset_m": shards[0].get("tcp_offset_m"),
            "noise_repeats": shards[0].get("noise_repeats"),
        },
        "summary": summary_rows,
    }
    (output_dir / "merged_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
