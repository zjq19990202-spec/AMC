#!/usr/bin/env python3
"""Measure TCP direction bias when the atomic instruction is empty."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import build_atomic_dataset

import evaluate_many_cluster_single_dual_steering as base


COMPONENTS = (
    "move_x",
    "move_y",
    "move_z",
    "rotate_x",
    "rotate_y",
    "rotate_z",
)


def _summarize(outputs: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in outputs:
        for step in (25, 50):
            twist = np.asarray(row["trajectory_twist"][str(step)], dtype=np.float64)
            for index, component in enumerate(COMPONENTS):
                grouped[(row["arm"], step, component)].append(float(twist[index]))
                grouped[("both", step, component)].append(float(twist[index]))
    summary = []
    for (arm, step, component), values in sorted(grouped.items()):
        array = np.asarray(values)
        positive = float(np.mean(array > 0.0))
        negative = float(np.mean(array < 0.0))
        summary.append(
            {
                "arm": arm,
                "step": step,
                "component": component,
                "samples": len(array),
                "positive_rate": positive,
                "negative_rate": negative,
                "zero_rate": float(np.mean(array == 0.0)),
                "majority_direction_rate": max(positive, negative),
                "signed_mean": float(np.mean(array)),
                "signed_median": float(np.median(array)),
                "mean_absolute": float(np.mean(np.abs(array))),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cluster-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--noise-repeats", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = dict(item.split("=", 1) for item in args.checkpoint)
    report = json.loads(args.cluster_report.read_text(encoding="utf-8"))
    clusters, _ = base._select_tests(  # noqa: SLF001
        report,
        minimum_mode_count=3,
        minimum_mode_episodes=2,
        minimum_modes_per_cluster=4,
        minimum_cluster_samples=100,
        minimum_cluster_episodes=10,
        clusters_per_arm=0,
        max_single_per_cluster=4,
        max_dual_per_cluster=4,
    )
    config = AtomicPi05Config(max_token_len=250, fast_action_ce_loss_weight=0.0)
    dataset = build_atomic_dataset(
        (args.dataset_root,),
        norm_assets_dir="/mnt/cunchu/zjq/target",
        norm_asset_id="openpi_norm_compact_accepted_v3",
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        include_fast=False,
    )
    rows = []
    noises = []
    first_rng = np.random.default_rng(args.seed)
    first_noises = {
        cluster["cluster_key"]: first_rng.standard_normal(
            (config.action_horizon, config.action_dim), dtype=np.float32
        )
        for cluster in clusters
    }
    extra_rng = np.random.default_rng(args.seed + 1)
    for cluster in clusters:
        for repeat in range(args.noise_repeats):
            rows.append(
                {
                    "cluster_key": cluster["cluster_key"],
                    "arm": cluster["arm"],
                    "dataset_index": cluster["dataset_index"],
                    "repeat": repeat,
                    "prompt": "",
                }
            )
            noises.append(
                first_noises[cluster["cluster_key"]]
                if repeat == 0
                else extra_rng.standard_normal(
                    (config.action_horizon, config.action_dim), dtype=np.float32
                )
            )
    noises_array = np.stack(noises)
    model_outputs, model_summaries = {}, {}
    for name, checkpoint in checkpoints.items():
        outputs = base._evaluate_checkpoint(  # noqa: SLF001
            name,
            Path(checkpoint),
            config,
            dataset,
            args.dataset_root,
            rows,
            noises_array,
            batch_size=args.batch_size,
            stored_steps=(25, 50),
        )
        summary = _summarize(outputs)
        model_outputs[name] = outputs
        model_summaries[name] = summary
        with (args.output_dir / f"{name}_no_atom_direction_bias.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    payload = {
        "checkpoints": checkpoints,
        "prompt": "",
        "clusters": len(clusters),
        "noise_repeats": args.noise_repeats,
        "samples_per_model": len(rows),
        "summary": model_summaries,
        "outputs": model_outputs,
    }
    (args.output_dir / "no_atom_direction_bias.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"samples": len(rows), "summary": model_summaries}, indent=2))


if __name__ == "__main__":
    main()
