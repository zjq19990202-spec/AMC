#!/usr/bin/env python3
"""Merge and score stock-PI0.5 native atomic/reverse steering sweeps."""

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


def _all_margin(
    candidate: np.ndarray,
    reference: np.ndarray,
    atoms: list[str],
    *,
    translation_threshold_mm: float,
    rotation_threshold_deg: float,
) -> tuple[bool, float]:
    margins = []
    for atom in atoms:
        component, sign = _component(atom)
        threshold = (
            translation_threshold_mm if component < 3 else rotation_threshold_deg
        )
        margins.append(float(sign * (candidate[component] - reference[component]) - threshold))
    return bool(all(margin > 0.0 for margin in margins)), min(margins)


def _absolute(
    candidate: np.ndarray,
    atoms: list[str],
    *,
    translation_threshold_mm: float,
    rotation_threshold_deg: float,
) -> bool:
    return _all_margin(
        candidate,
        np.zeros(6),
        atoms,
        translation_threshold_mm=translation_threshold_mm,
        rotation_threshold_deg=rotation_threshold_deg,
    )[0]


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--translation-threshold-mm", type=float, default=5.0)
    parser.add_argument("--rotation-threshold-deg", type=float, default=1.0)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.input_dir.glob("shard_*.json"))
    if not paths:
        raise RuntimeError(f"no shard JSON files under {args.input_dir}")
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    expected = int(shards[0]["num_shards"])
    indices = {int(shard["shard_index"]) for shard in shards}
    if len(shards) != expected or indices != set(range(expected)):
        raise RuntimeError(f"expected shards 0..{expected - 1}, got {sorted(indices)}")

    grouped: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for shard in shards:
        for row in shard["outputs"]:
            key = (row["anchor_key"], int(row["repeat"]))
            if row["variant"] in grouped[key]:
                raise RuntimeError(f"duplicate {key} {row['variant']}")
            grouped[key][row["variant"]] = row

    trial_rows = []
    required = {"empty", "subtask_no_atom", "native_atomic", "native_reverse"}
    for (anchor_key, repeat), variants in sorted(grouped.items()):
        if set(variants) != required:
            raise RuntimeError(f"{anchor_key}/{repeat} has variants {sorted(variants)}")
        native = variants["native_atomic"]
        reverse = variants["native_reverse"]
        atoms = list(native["requested_atoms"])
        reverse_atoms = list(reverse["requested_atoms"])
        for step in (25, 50):
            values = {
                name: np.asarray(row["trajectory_twist"][str(step)], dtype=float)
                for name, row in variants.items()
            }
            native_vs_reverse, pair_margin = _all_margin(
                values["native_atomic"],
                values["native_reverse"],
                atoms,
                translation_threshold_mm=args.translation_threshold_mm,
                rotation_threshold_deg=args.rotation_threshold_deg,
            )
            native_vs_empty, native_empty_margin = _all_margin(
                values["native_atomic"],
                values["empty"],
                atoms,
                translation_threshold_mm=args.translation_threshold_mm,
                rotation_threshold_deg=args.rotation_threshold_deg,
            )
            reverse_vs_empty, reverse_empty_margin = _all_margin(
                values["native_reverse"],
                values["empty"],
                reverse_atoms,
                translation_threshold_mm=args.translation_threshold_mm,
                rotation_threshold_deg=args.rotation_threshold_deg,
            )
            native_vs_subtask, _ = _all_margin(
                values["native_atomic"],
                values["subtask_no_atom"],
                atoms,
                translation_threshold_mm=args.translation_threshold_mm,
                rotation_threshold_deg=args.rotation_threshold_deg,
            )
            trial_rows.append(
                {
                    "anchor_key": anchor_key,
                    "cluster_key": native["cluster_key"],
                    "episode": native["episode"],
                    "frame": native["frame"],
                    "arm": native["arm"],
                    "scheme": "single" if len(atoms) == 1 else "dual",
                    "other_status": native["other_status"],
                    "atoms": "+".join(atoms),
                    "repeat": repeat,
                    "step": step,
                    "native_absolute_success": _absolute(
                        values["native_atomic"],
                        atoms,
                        translation_threshold_mm=args.translation_threshold_mm,
                        rotation_threshold_deg=args.rotation_threshold_deg,
                    ),
                    "reverse_absolute_success": _absolute(
                        values["native_reverse"],
                        reverse_atoms,
                        translation_threshold_mm=args.translation_threshold_mm,
                        rotation_threshold_deg=args.rotation_threshold_deg,
                    ),
                    "bidirectional_absolute_success": _absolute(
                        values["native_atomic"],
                        atoms,
                        translation_threshold_mm=args.translation_threshold_mm,
                        rotation_threshold_deg=args.rotation_threshold_deg,
                    )
                    and _absolute(
                        values["native_reverse"],
                        reverse_atoms,
                        translation_threshold_mm=args.translation_threshold_mm,
                        rotation_threshold_deg=args.rotation_threshold_deg,
                    ),
                    "native_vs_reverse_success": native_vs_reverse,
                    "native_vs_empty_success": native_vs_empty,
                    "reverse_vs_empty_success": reverse_vs_empty,
                    "native_vs_subtask_success": native_vs_subtask,
                    "minimum_native_reverse_margin": pair_margin,
                    "minimum_native_empty_margin": native_empty_margin,
                    "minimum_reverse_empty_margin": reverse_empty_margin,
                    "native_reverse_endpoint_separation": float(
                        np.linalg.norm(values["native_atomic"] - values["native_reverse"])
                    ),
                }
            )

    buckets: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in trial_rows:
        for arm in (row["arm"], "both"):
            for scheme in (row["scheme"], "all"):
                buckets[(arm, scheme, row["step"])].append(row)
    summary_rows = []
    for (arm, scheme, step), rows in sorted(buckets.items()):
        summary_rows.append(
            {
                "arm": arm,
                "scheme": scheme,
                "step": step,
                "trials": len(rows),
                "anchors": len({row["anchor_key"] for row in rows}),
                "native_absolute_success_rate": _mean(rows, "native_absolute_success"),
                "reverse_absolute_success_rate": _mean(rows, "reverse_absolute_success"),
                "bidirectional_absolute_success_rate": _mean(
                    rows, "bidirectional_absolute_success"
                ),
                "native_vs_reverse_success_rate": _mean(
                    rows, "native_vs_reverse_success"
                ),
                "native_vs_empty_success_rate": _mean(rows, "native_vs_empty_success"),
                "reverse_vs_empty_success_rate": _mean(rows, "reverse_vs_empty_success"),
                "native_vs_subtask_success_rate": _mean(
                    rows, "native_vs_subtask_success"
                ),
                "median_minimum_native_reverse_margin": float(
                    np.median([row["minimum_native_reverse_margin"] for row in rows])
                ),
                "median_native_reverse_endpoint_separation": float(
                    np.median([row["native_reverse_endpoint_separation"] for row in rows])
                ),
            }
        )

    _write_csv(output_dir / "native_atomic_trial_metrics.csv", trial_rows)
    _write_csv(output_dir / "native_atomic_summary.csv", summary_rows)
    report = {
        "checkpoint": shards[0]["checkpoint"],
        "checkpoint_name": shards[0]["checkpoint_name"],
        "selection_manifest": shards[0]["selection_manifest"],
        "anchor_count": len({key[0] for key in grouped}),
        "anchor_repeat_count": len(grouped),
        "trajectory_count": sum(len(shard["outputs"]) for shard in shards),
        "thresholds": {
            "translation_mm": args.translation_threshold_mm,
            "rotation_deg": args.rotation_threshold_deg,
        },
        "evaluation_contract": {
            "same_image_state_noise_within_pair": True,
            "tcp_offset_m": shards[0]["tcp_offset_m"],
            "norm_assets_dir": shards[0]["norm_assets_dir"],
            "norm_asset_id": shards[0]["norm_asset_id"],
            "max_token_len": shards[0]["max_token_len"],
            "noise_repeats": shards[0]["noise_repeats"],
        },
        "summary": summary_rows,
    }
    (output_dir / "merged_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
