#!/usr/bin/env python3
"""Aggregate paired multi-frame fruit steering for three or more policies."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import numpy as np


def _parse_model(value: str) -> tuple[str, Path]:
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError(f"invalid --model specification: {value!r}")
    return name, Path(path)


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > 1.0e-9 else float("nan")


def _mean(values) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


def _aggregate(rows: list[dict[str, Any]], model_names: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"n_target_frame_pairs": len(rows)}
    for model in model_names:
        result[model] = {
            "mean_endpoint_distance_mm": _mean(row[f"{model}_distance_mm"] for row in rows),
            "median_endpoint_distance_mm": float(np.median([row[f"{model}_distance_mm"] for row in rows])) if rows else float("nan"),
            "mean_reduction_mm": _mean(row[f"{model}_reduction_mm"] for row in rows),
            "mean_fractional_progress": _mean(row[f"{model}_fractional_progress"] for row in rows),
            "mean_direction_cosine": _mean(row[f"{model}_direction_cosine"] for row in rows),
            "moves_closer_rate": _mean(row[f"{model}_moves_closer"] for row in rows),
            "target_prompt_top1_rate": _mean(row[f"{model}_target_prompt_top1"] for row in rows),
            "mean_target_prompt_advantage_mm": _mean(
                row[f"{model}_target_prompt_advantage_mm"] for row in rows
            ),
            "mean_best_wrong_margin_mm": _mean(
                row[f"{model}_best_wrong_margin_mm"] for row in rows
            ),
            "mean_directional_advantage": _mean(
                row[f"{model}_directional_advantage"] for row in rows
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--selection-dataset-name", default="fruit_macro")
    parser.add_argument("--model", action="append", required=True, help="NAME=SUMMARY_JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    models = dict(_parse_model(value) for value in args.model)
    if len(models) < 2:
        raise ValueError("at least two --model inputs are required")
    model_names = list(models)
    selection_payload = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    selection = selection_payload["datasets"][args.selection_dataset_name]
    selections = {(int(row["episode"]), int(row["frame"])): row for row in selection}
    model_rows: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    manifest_hashes = set()
    for name, path in models.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("rows_complete", -1)) != len(selection):
            raise ValueError(f"{name} is incomplete: {payload.get('rows_complete')}/{len(selection)}")
        manifest_hashes.add(payload.get("selection_manifest_sha256"))
        model_rows[name] = {
            (int(row["episode"]), int(row["frame"])): row for row in payload["rows"]
        }
    if len(manifest_hashes) != 1:
        raise ValueError("model summaries were produced from different selection manifests")

    rows: list[dict[str, Any]] = []
    for key, spec in selections.items():
        native = str(spec["native_target"])
        targets = spec["target_positions_m"]
        reference_row = model_rows[model_names[0]][key]
        start = np.asarray(reference_row["start_xyz_m"], dtype=float)
        for other_name in model_names[1:]:
            other_start = np.asarray(model_rows[other_name][key]["start_xyz_m"], dtype=float)
            if not np.allclose(start, other_start, atol=1.0e-6):
                raise ValueError(f"start TCP mismatch for {key}: {model_names[0]} vs {other_name}")
        for target_name, target_xyz in targets.items():
            target = np.asarray(target_xyz, dtype=float)
            initial_mm = float(np.linalg.norm(target - start) * 1000.0)
            row: dict[str, Any] = {
                "episode": key[0],
                "frame": key[1],
                "frame_stage": spec.get("frame_stage"),
                "source_dataset": spec.get("source_dataset"),
                "active_arm": spec.get("active_arm"),
                "native_target": native,
                "target": target_name,
                "is_native_target": target_name == native,
                "initial_distance_mm": initial_mm,
            }
            for model in model_names:
                endpoints = {
                    name: np.asarray(value, dtype=float)
                    for name, value in model_rows[model][key]["endpoint_xyz_m"].items()
                }
                endpoint = endpoints[target_name]
                distance_mm = float(np.linalg.norm(target - endpoint) * 1000.0)
                reduction_mm = initial_mm - distance_mm
                distances = {
                    prompt: float(np.linalg.norm(target - value) * 1000.0)
                    for prompt, value in endpoints.items()
                }
                direction_cosines = {
                    prompt: _cosine(value - start, target - start)
                    for prompt, value in endpoints.items()
                }
                wrong_distances = [
                    value for prompt, value in distances.items() if prompt != target_name
                ]
                wrong_cosines = [
                    value
                    for prompt, value in direction_cosines.items()
                    if prompt != target_name
                ]
                row[f"{model}_distance_mm"] = distance_mm
                row[f"{model}_reduction_mm"] = reduction_mm
                row[f"{model}_fractional_progress"] = reduction_mm / max(initial_mm, 1.0)
                row[f"{model}_direction_cosine"] = direction_cosines[target_name]
                row[f"{model}_moves_closer"] = distance_mm < initial_mm
                row[f"{model}_target_prompt_top1"] = min(distances, key=distances.get) == target_name
                row[f"{model}_target_prompt_advantage_mm"] = _mean(wrong_distances) - distance_mm
                row[f"{model}_best_wrong_margin_mm"] = min(wrong_distances) - distance_mm
                row[f"{model}_directional_advantage"] = (
                    direction_cosines[target_name] - _mean(wrong_cosines)
                )
            rows.append(row)

    native_rows = [row for row in rows if row["is_native_target"]]
    counterfactual_rows = [row for row in rows if not row["is_native_target"]]
    by_stage = {
        stage: _aggregate([row for row in rows if row["frame_stage"] == stage], model_names)
        for stage in sorted({str(row["frame_stage"]) for row in rows})
    }
    by_source = {
        source: _aggregate([row for row in rows if row["source_dataset"] == source], model_names)
        for source in sorted({str(row["source_dataset"]) for row in rows})
    }

    episode_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        episode_groups[int(row["episode"])].append(row)
    episode_means = []
    winner_counts = {name: 0 for name in model_names}
    for episode, values in episode_groups.items():
        item: dict[str, Any] = {"episode": episode}
        for model in model_names:
            for metric in (
                "distance_mm",
                "reduction_mm",
                "fractional_progress",
                "direction_cosine",
                "moves_closer",
                "target_prompt_top1",
                "target_prompt_advantage_mm",
                "best_wrong_margin_mm",
                "directional_advantage",
            ):
                item[f"{model}_{metric}"] = _mean(
                    row[f"{model}_{metric}"] for row in values
                )
        winner = min(model_names, key=lambda name: item[f"{name}_distance_mm"])
        item["distance_winner"] = winner
        winner_counts[winner] += 1
        episode_means.append(item)

    episode_macro: dict[str, Any] = {}
    for model in model_names:
        episode_macro[model] = {
            "mean_endpoint_distance_mm": _mean(
                row[f"{model}_distance_mm"] for row in episode_means
            ),
            "mean_reduction_mm": _mean(
                row[f"{model}_reduction_mm"] for row in episode_means
            ),
            "mean_fractional_progress": _mean(
                row[f"{model}_fractional_progress"] for row in episode_means
            ),
            "mean_direction_cosine": _mean(
                row[f"{model}_direction_cosine"] for row in episode_means
            ),
            "moves_closer_rate": _mean(
                row[f"{model}_moves_closer"] for row in episode_means
            ),
            "target_prompt_top1_rate": _mean(
                row[f"{model}_target_prompt_top1"] for row in episode_means
            ),
            "mean_target_prompt_advantage_mm": _mean(
                row[f"{model}_target_prompt_advantage_mm"] for row in episode_means
            ),
            "mean_best_wrong_margin_mm": _mean(
                row[f"{model}_best_wrong_margin_mm"] for row in episode_means
            ),
            "mean_directional_advantage": _mean(
                row[f"{model}_directional_advantage"] for row in episode_means
            ),
        }

    source_macro: dict[str, Any] = {}
    for model in model_names:
        source_macro[model] = {
            metric: _mean(by_source[source][model][metric] for source in by_source)
            for metric in (
                "mean_endpoint_distance_mm",
                "mean_reduction_mm",
                "mean_fractional_progress",
                "mean_direction_cosine",
                "moves_closer_rate",
                "target_prompt_top1_rate",
                "mean_target_prompt_advantage_mm",
                "mean_best_wrong_margin_mm",
                "mean_directional_advantage",
            )
        }

    frame_separations: list[dict[str, Any]] = []
    for key, spec in selections.items():
        frame_row: dict[str, Any] = {
            "episode": key[0],
            "frame": key[1],
            "frame_stage": spec.get("frame_stage"),
            "source_dataset": spec.get("source_dataset"),
        }
        for model in model_names:
            endpoints = [
                np.asarray(value, dtype=float)
                for value in model_rows[model][key]["endpoint_xyz_m"].values()
            ]
            pairwise = [
                float(np.linalg.norm(endpoints[i] - endpoints[j]) * 1000.0)
                for i in range(len(endpoints))
                for j in range(i + 1, len(endpoints))
            ]
            frame_row[f"{model}_mean_pairwise_endpoint_separation_mm"] = _mean(pairwise)
            frame_row[f"{model}_max_pairwise_endpoint_separation_mm"] = max(pairwise)
        frame_separations.append(frame_row)
    separation_summary = {
        model: {
            "mean_pairwise_endpoint_separation_mm": _mean(
                row[f"{model}_mean_pairwise_endpoint_separation_mm"]
                for row in frame_separations
            ),
            "mean_max_endpoint_separation_mm": _mean(
                row[f"{model}_max_pairwise_endpoint_separation_mm"]
                for row in frame_separations
            ),
        }
        for model in model_names
    }
    bootstrap_rng = np.random.default_rng(20260902)
    bootstrap_indices = bootstrap_rng.integers(
        0, len(episode_means), size=(20_000, len(episode_means))
    )
    pairwise_episode_bootstrap: dict[str, Any] = {}
    for first, second in combinations(model_names, 2):
        comparison: dict[str, Any] = {}
        for metric in (
            "distance_mm",
            "fractional_progress",
            "target_prompt_advantage_mm",
            "best_wrong_margin_mm",
            "target_prompt_top1",
        ):
            differences = np.asarray(
                [
                    row[f"{first}_{metric}"] - row[f"{second}_{metric}"]
                    for row in episode_means
                ],
                dtype=float,
            )
            bootstrap_means = differences[bootstrap_indices].mean(axis=1)
            comparison[metric] = {
                "mean_difference": float(differences.mean()),
                "bootstrap_95_ci": np.quantile(
                    bootstrap_means, (0.025, 0.975)
                ).tolist(),
            }
        pairwise_episode_bootstrap[f"{first}_minus_{second}"] = comparison
    report = {
        "contract": selection_payload["selection_contract"],
        "models": {name: str(path.resolve()) for name, path in models.items()},
        "summary": {
            "all_scored_targets": _aggregate(rows, model_names),
            "native_targets": _aggregate(native_rows, model_names),
            "counterfactual_future_targets": _aggregate(counterfactual_rows, model_names),
            "episode_macro": episode_macro,
            "source_macro": source_macro,
            "by_stage": by_stage,
            "by_source": by_source,
            "episode_distance_winner_counts": winner_counts,
            "prompt_endpoint_separation": separation_summary,
            "pairwise_episode_bootstrap": pairwise_episode_bootstrap,
            "episodes": len(episode_groups),
            "frames": len(selection),
        },
        "episode_means": episode_means,
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
        "# Multi-frame fruit steering macro evaluation",
        "",
        f"- Episodes: {len(episode_groups)}",
        f"- Frames: {len(selection)}",
        f"- Scored future fruit target/frame pairs: {len(rows)}",
        "- Target: recorded active-arm TCP at the final frame of the fruit grasp SUB.",
        "- Macro: target/frame scores are averaged inside each episode first; every episode then contributes equally.",
        "",
        "| Model | endpoint mm ↓ | progress ↑ | correct-vs-wrong advantage ↑ | best-wrong margin ↑ | Top-1 ↑ | pair separation ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in model_names:
        values = episode_macro[model]
        lines.append(
            f"| {model} | {values['mean_endpoint_distance_mm']:.1f} | "
            f"{100 * values['mean_fractional_progress']:.1f}% | "
            f"{values['mean_target_prompt_advantage_mm']:.1f} mm | "
            f"{values['mean_best_wrong_margin_mm']:.1f} mm | "
            f"{100 * values['target_prompt_top1_rate']:.1f}% | "
            f"{separation_summary[model]['mean_pairwise_endpoint_separation_mm']:.1f} mm |"
        )
    lines.extend(["", "Episode-level endpoint winner counts: " + ", ".join(f"{name}={count}" for name, count in winner_counts.items()), ""])
    (args.output_dir / "macro_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
