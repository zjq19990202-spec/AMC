#!/usr/bin/env python3
"""Merge paired base25k/B2 RTC-offset GT reports across force domains."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PLOT_METRICS = (
    ("normalized_action_rmse", "Normalized 16D action RMSE"),
    ("joint_rmse_rad", "Decoded 14-joint RMSE (rad)"),
    ("left_tcp_translation_rmse_mm", "Left TCP translation RMSE (mm)"),
    ("right_tcp_translation_rmse_mm", "Right TCP translation RMSE (mm)"),
)
CONTRACT_FIELDS = (
    "dataset_root",
    "episode",
    "window_count",
    "offsets",
    "suffix_lengths",
    "selection_sha256",
    "norm_stats_sha256",
    "force_norm_sha256",
    "seed",
    "sampler_steps",
)


def _parse_domain(value: str) -> tuple[str, Path, Path]:
    name, paths = value.split("=", 1)
    base, b2 = paths.split(",", 1)
    return name, Path(base), Path(b2)


def _relative(base: float, b2: float) -> float:
    return 100.0 * (b2 / base - 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        action="append",
        required=True,
        help="NAME=BASE_SUMMARY,B2_SUMMARY",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    domains = {}
    for value in args.domain:
        name, base_path, b2_path = _parse_domain(value)
        base = json.loads(base_path.read_text(encoding="utf-8"))
        b2 = json.loads(b2_path.read_text(encoding="utf-8"))
        mismatches = {
            field: (base.get(field), b2.get(field))
            for field in CONTRACT_FIELDS
            if base.get(field) != b2.get(field)
        }
        if mismatches:
            raise ValueError(f"{name} contract mismatch: {mismatches}")
        if base["model_kind"] != "base25k" or b2["model_kind"] != "b2":
            raise ValueError(f"{name} model kinds are not base25k/B2")
        domains[name] = {
            "base_path": str(base_path),
            "b2_path": str(b2_path),
            "contract": {field: base[field] for field in CONTRACT_FIELDS},
            "base": base["offset_metrics"],
            "b2": b2["offset_metrics"],
        }

    offsets = next(iter(domains.values()))["contract"]["offsets"]
    if any(domain["contract"]["offsets"] != offsets for domain in domains.values()):
        raise ValueError("domains use different RTC offsets")
    metric_names = tuple(next(iter(domains.values()))["base"][str(offsets[0])])
    metric_names = tuple(metric for metric in metric_names if metric != "delta_z_norm")

    rows = []
    for domain_name, domain in domains.items():
        for offset in offsets:
            key = str(offset)
            row = {
                "domain": domain_name,
                "episode": domain["contract"]["episode"],
                "windows": domain["contract"]["window_count"],
                "offset": offset,
                "suffix_steps": domain["contract"]["suffix_lengths"][key],
            }
            for metric in metric_names:
                base_value = float(domain["base"][key][metric])
                b2_value = float(domain["b2"][key][metric])
                row[f"base25k_{metric}"] = base_value
                row[f"b2_{metric}"] = b2_value
                row[f"change_pct_{metric}"] = _relative(base_value, b2_value)
            row["b2_delta_z_norm"] = float(domain["b2"][key]["delta_z_norm"])
            rows.append(row)

    overall = {}
    for offset in offsets:
        key = str(offset)
        weights = {
            name: domain["contract"]["window_count"]
            * domain["contract"]["suffix_lengths"][key]
            for name, domain in domains.items()
        }
        overall[key] = {}
        for metric in metric_names:
            for method, source_key in (("base25k", "base"), ("b2", "b2")):
                overall[key][f"{method}_{metric}"] = math.sqrt(
                    sum(
                        weights[name] * float(domain[source_key][key][metric]) ** 2
                        for name, domain in domains.items()
                    )
                    / sum(weights.values())
                )
            overall[key][f"change_pct_{metric}"] = _relative(
                overall[key][f"base25k_{metric}"],
                overall[key][f"b2_{metric}"],
            )

    with (args.output_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    x = np.asarray(offsets)
    for axis, (metric, title) in zip(axes.flat, PLOT_METRICS, strict=True):
        axis.plot(
            x,
            [overall[str(offset)][f"base25k_{metric}"] for offset in offsets],
            "o--",
            label="25K original chunk",
        )
        axis.plot(
            x,
            [overall[str(offset)][f"b2_{metric}"] for offset in offsets],
            "o-",
            label="B2 force RTC",
        )
        axis.set_xticks(x)
        axis.set_xlabel("RTC offset")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    plot_path = args.output_dir / "base25k_vs_b2_all_offsets.png"
    figure.savefig(plot_path, dpi=190)
    plt.close(figure)

    result = {
        "comparison": "25K original fixed chunk suffix versus B2 force RTC replanned suffix",
        "domains": domains,
        "overall": overall,
        "rows": rows,
        "plot": str(plot_path),
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "run_contract.md").write_text(
        "# Base25K versus B2 RTC-offset comparison\n\n"
        "- Same native prompts, normalization, selection and flow noise within each domain.\n"
        "- Base25K: one original offset-0 chunk; score its unexecuted suffix.\n"
        "- B2: clean GT committed prefix, causal fast force/state, replan and score suffix only.\n"
        "- Domains: " + ", ".join(domains) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"overall": overall, "plot": str(plot_path)}, indent=2))


if __name__ == "__main__":
    main()
