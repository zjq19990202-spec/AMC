#!/usr/bin/env python3
"""Merge per-domain spherical B2 zF attribution summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    offsets = [str(value) for value in reports[0]["offsets"]]
    branches = ("full", "zero_zf_content", "wrong_zf", "no_fast")
    fields = (
        "normalized_action_rmse_to_gt",
        "gt_rmse_change_pct_vs_full",
        "prediction_change_rmse_vs_full",
        "joint_rmse_rad_to_gt",
        "left_tcp_rmse_mm_to_gt",
        "right_tcp_rmse_mm_to_gt",
        "mean_force_rotation_deg",
    )
    macro = {}
    for offset in offsets:
        macro[offset] = {}
        for branch in branches:
            macro[offset][branch] = {
                field: float(
                    np.mean(
                        [report["by_offset"][offset][branch][field] for report in reports]
                    )
                )
                for field in fields
            }
    result = {
        "domains": [report["dataset_root"] for report in reports],
        "sample_counts": [report["selection_count"] for report in reports],
        "task_macro_by_offset": macro,
    }
    (args.output_dir / "macro_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Spherical B2 zF attribution",
        "",
        f"Domains: {len(reports)}; samples: {sum(result['sample_counts'])} total.",
        "",
        "| Offset | Branch | GT RMSE | ΔGT vs full | Prediction ΔRMSE | Force angle |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for offset in offsets:
        for branch in branches:
            row = macro[offset][branch]
            lines.append(
                f"| {offset} | {branch} | "
                f"{row['normalized_action_rmse_to_gt']:.6f} | "
                f"{row['gt_rmse_change_pct_vs_full']:+.2f}% | "
                f"{row['prediction_change_rmse_vs_full']:.6f} | "
                f"{row['mean_force_rotation_deg']:.2f}° |"
            )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
