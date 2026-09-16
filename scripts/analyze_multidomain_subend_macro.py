#!/usr/bin/env python3
"""Aggregate paired multi-domain SUB-end steering endpoints."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    "distance_mm",
    "reduction_mm",
    "fractional_progress",
    "direction_cosine",
    "moves_closer",
    "target_prompt_top1",
    "target_prompt_advantage_mm",
    "best_wrong_margin_mm",
    "directional_advantage",
)


def _parse_model(value: str) -> tuple[str, Path]:
    name, path = value.split("=", 1)
    return name, Path(path)


def _mean(values) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > 1.0e-9 else float("nan")


def _aggregate(rows: list[dict[str, Any]], models: list[str]) -> dict[str, Any]:
    output: dict[str, Any] = {"n": len(rows)}
    for model in models:
        output[model] = {
            "endpoint_distance_mm": _mean(row[f"{model}_distance_mm"] for row in rows),
            "distance_reduction_mm": _mean(row[f"{model}_reduction_mm"] for row in rows),
            "fractional_progress": _mean(row[f"{model}_fractional_progress"] for row in rows),
            "direction_cosine": _mean(row[f"{model}_direction_cosine"] for row in rows),
            "moves_closer_rate": _mean(row[f"{model}_moves_closer"] for row in rows),
            "target_prompt_top1_rate": _mean(row[f"{model}_target_prompt_top1"] for row in rows),
            "target_prompt_advantage_mm": _mean(row[f"{model}_target_prompt_advantage_mm"] for row in rows),
            "best_wrong_margin_mm": _mean(row[f"{model}_best_wrong_margin_mm"] for row in rows),
            "directional_advantage": _mean(row[f"{model}_directional_advantage"] for row in rows),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help="NAME=JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    model_paths = dict(_parse_model(value) for value in args.model)
    models = list(model_paths)
    selection_payload = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    selections = {
        (dataset, int(row["episode"]), int(row["frame"])): row
        for dataset, values in selection_payload["datasets"].items()
        for row in values
    }
    expected = len(selections)
    model_rows = {}
    hashes = set()
    for model, path in model_paths.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("rows_complete", -1)) != expected:
            raise ValueError(f"{model} incomplete: {payload.get('rows_complete')}/{expected}")
        hashes.add(payload.get("selection_manifest_sha256"))
        model_rows[model] = {
            (str(row["dataset"]), int(row["episode"]), int(row["frame"])): row
            for row in payload["rows"]
        }
    if len(hashes) != 1:
        raise ValueError("model outputs use different selection manifests")

    rows = []
    frame_separations = []
    for key, spec in selections.items():
        start = np.asarray(model_rows[models[0]][key]["start_xyz_m"], dtype=float)
        for model in models[1:]:
            if not np.allclose(start, model_rows[model][key]["start_xyz_m"], atol=1.0e-6):
                raise ValueError(f"start TCP mismatch at {key}")
        separation_row = {
            "dataset": key[0],
            "episode": key[1],
            "frame": key[2],
            "frame_stage": spec["frame_stage"],
        }
        for model in models:
            endpoints = [
                np.asarray(value, dtype=float)
                for value in model_rows[model][key]["endpoint_xyz_m"].values()
            ]
            pairwise = [
                float(np.linalg.norm(endpoints[i] - endpoints[j]) * 1000.0)
                for i in range(len(endpoints))
                for j in range(i + 1, len(endpoints))
            ]
            separation_row[f"{model}_pair_separation_mm"] = _mean(pairwise)
        frame_separations.append(separation_row)

        for target_name, target_xyz in spec["target_positions_m"].items():
            target = np.asarray(target_xyz, dtype=float)
            initial_mm = float(np.linalg.norm(target - start) * 1000.0)
            row: dict[str, Any] = {
                "dataset": key[0],
                "episode": key[1],
                "frame": key[2],
                "frame_stage": spec["frame_stage"],
                "active_arm": spec["active_arm"],
                "target": target_name,
                "is_native": target_name == spec["native_target"],
                "initial_distance_mm": initial_mm,
            }
            for model in models:
                endpoints = {
                    name: np.asarray(value, dtype=float)
                    for name, value in model_rows[model][key]["endpoint_xyz_m"].items()
                }
                distances = {
                    name: float(np.linalg.norm(target - value) * 1000.0)
                    for name, value in endpoints.items()
                }
                cosines = {
                    name: _cosine(value - start, target - start)
                    for name, value in endpoints.items()
                }
                distance = distances[target_name]
                wrong_distances = [value for name, value in distances.items() if name != target_name]
                wrong_cosines = [value for name, value in cosines.items() if name != target_name]
                reduction = initial_mm - distance
                row[f"{model}_distance_mm"] = distance
                row[f"{model}_reduction_mm"] = reduction
                row[f"{model}_fractional_progress"] = reduction / max(initial_mm, 1.0)
                row[f"{model}_direction_cosine"] = cosines[target_name]
                row[f"{model}_moves_closer"] = distance < initial_mm
                row[f"{model}_target_prompt_top1"] = min(distances, key=distances.get) == target_name
                row[f"{model}_target_prompt_advantage_mm"] = _mean(wrong_distances) - distance
                row[f"{model}_best_wrong_margin_mm"] = min(wrong_distances) - distance
                row[f"{model}_directional_advantage"] = cosines[target_name] - _mean(wrong_cosines)
            rows.append(row)

    by_domain = {
        domain: _aggregate([row for row in rows if row["dataset"] == domain], models)
        for domain in sorted(selection_payload["datasets"])
    }
    domain_macro = {
        model: {
            metric: _mean(by_domain[domain][model][metric] for domain in by_domain)
            for metric in by_domain[next(iter(by_domain))][model]
        }
        for model in models
    }
    episode_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        episode_groups[(row["dataset"], int(row["episode"]))].append(row)
    episode_rows = []
    winner_counts = {model: 0 for model in models}
    for key, values in episode_groups.items():
        item: dict[str, Any] = {"dataset": key[0], "episode": key[1]}
        for model in models:
            for metric in METRICS:
                item[f"{model}_{metric}"] = _mean(row[f"{model}_{metric}"] for row in values)
        winner = min(models, key=lambda model: item[f"{model}_distance_mm"])
        winner_counts[winner] += 1
        item["distance_winner"] = winner
        episode_rows.append(item)

    rng = np.random.default_rng(20260902)
    indices = rng.integers(0, len(episode_rows), size=(20_000, len(episode_rows)))
    bootstrap = {}
    for first, second in combinations(models, 2):
        comparison = {}
        for metric in (
            "distance_mm",
            "fractional_progress",
            "target_prompt_advantage_mm",
            "best_wrong_margin_mm",
            "target_prompt_top1",
        ):
            differences = np.asarray(
                [row[f"{first}_{metric}"] - row[f"{second}_{metric}"] for row in episode_rows]
            )
            samples = differences[indices].mean(axis=1)
            comparison[metric] = {
                "mean_difference": float(differences.mean()),
                "bootstrap_95_ci": np.quantile(samples, (0.025, 0.975)).tolist(),
            }
        bootstrap[f"{first}_minus_{second}"] = comparison
    separation_summary = {
        model: _mean(row[f"{model}_pair_separation_mm"] for row in frame_separations)
        for model in models
    }
    separation_by_domain = {
        domain: {
            model: _mean(
                row[f"{model}_pair_separation_mm"]
                for row in frame_separations
                if row["dataset"] == domain
            )
            for model in models
        }
        for domain in by_domain
    }
    separation_domain_macro = {
        model: _mean(separation_by_domain[domain][model] for domain in by_domain)
        for model in models
    }
    report = {
        "contract": selection_payload["selection_contract"],
        "models": {name: str(path.resolve()) for name, path in model_paths.items()},
        "summary": {
            "all": _aggregate(rows, models),
            "native": _aggregate([row for row in rows if row["is_native"]], models),
            "counterfactual": _aggregate([row for row in rows if not row["is_native"]], models),
            "by_domain": by_domain,
            "domain_macro": domain_macro,
            "prompt_pair_separation_mm": separation_summary,
            "prompt_pair_separation_by_domain_mm": separation_by_domain,
            "prompt_pair_separation_domain_macro_mm": separation_domain_macro,
            "episode_distance_winner_counts": winner_counts,
            "pairwise_episode_bootstrap": bootstrap,
            "episodes": len(episode_groups),
            "frames": len(selections),
            "target_frame_pairs": len(rows),
        },
        "episode_means": episode_rows,
        "frame_separations": frame_separations,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "macro_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "target_frame_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Non-fruit multi-domain SUB-end steering",
        "",
        f"- Domains: {len(by_domain)}; episodes: {len(episode_groups)}; frames: {len(selections)}.",
        "- Domain macro gives Cabinet, Drawer, Screw, Plug, and Vase equal weight.",
        "",
        "| Model | endpoint mm ↓ | distance reduction ↑ | prompt advantage ↑ | Top-1 ↑ | separation ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        value = domain_macro[model]
        lines.append(
            f"| {model} | {value['endpoint_distance_mm']:.1f} | "
            f"{value['distance_reduction_mm']:.1f} mm | "
            f"{value['target_prompt_advantage_mm']:.1f} mm | "
            f"{100 * value['target_prompt_top1_rate']:.1f}% | "
            f"{separation_domain_macro[model]:.1f} mm |"
        )
    lines.extend(["", "Episode endpoint winners: " + ", ".join(f"{name}={count}" for name, count in winner_counts.items()), ""])
    (args.output_dir / "macro_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
