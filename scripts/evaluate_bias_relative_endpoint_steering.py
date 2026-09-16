#!/usr/bin/env python3
"""Evaluate prompt-induced endpoint steering relative to an empty-prompt baseline."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import build_atomic_dataset

import evaluate_many_cluster_single_dual_steering as base


ATOMS = tuple(base.PHRASES)
MODES = tuple((atom,) for atom in ATOMS) + tuple(
    pair for pair in itertools.combinations(ATOMS, 2) if base._valid_mode(pair)  # noqa: SLF001
)
THRESHOLDS = ((0.0, 0.0), (2.5, 0.5), (5.0, 1.0), (10.0, 2.0))


def _relative_twist(prompt: np.ndarray, empty: np.ndarray) -> np.ndarray:
    translation_mm = prompt[:3] - empty[:3]
    prompt_rotation = Rotation.from_rotvec(np.deg2rad(prompt[3:])).as_matrix()
    empty_rotation = Rotation.from_rotvec(np.deg2rad(empty[3:])).as_matrix()
    rotation_deg = np.rad2deg(
        Rotation.from_matrix(prompt_rotation @ empty_rotation.T).as_rotvec()
    )
    return np.concatenate([translation_mm, rotation_deg])


def _score(row: dict, empty: np.ndarray, step: int, move_mm: float, rotate_deg: float) -> dict:
    prompt = np.asarray(row["trajectory_twist"][str(step)], dtype=np.float64)
    delta = _relative_twist(prompt, empty)
    margins = []
    for atom in row["atoms"]:
        index, sign = base._component(atom)  # noqa: SLF001
        threshold = move_mm if index < 3 else rotate_deg
        margins.append(float(sign * delta[index] - threshold))
    return {
        "cluster_key": row["cluster_key"],
        "arm": row["arm"],
        "kind": row["kind"],
        "atoms": "+".join(row["atoms"]),
        "step": step,
        "translation_threshold_mm": move_mm,
        "rotation_threshold_deg": rotate_deg,
        "success": bool(all(margin > 0.0 for margin in margins)),
        "minimum_margin": min(margins),
        "delta_move_x_mm": float(delta[0]),
        "delta_move_y_mm": float(delta[1]),
        "delta_move_z_mm": float(delta[2]),
        "delta_rotate_x_deg": float(delta[3]),
        "delta_rotate_y_deg": float(delta[4]),
        "delta_rotate_z_deg": float(delta[5]),
    }


def _summarize(records: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in records:
        threshold = (row["translation_threshold_mm"], row["rotation_threshold_deg"])
        for arm in (row["arm"], "both"):
            for kind in (row["kind"], "all"):
                grouped[(arm, kind, row["step"], *threshold)].append(row)
    output = []
    for key, rows in sorted(grouped.items()):
        arm, kind, step, move_mm, rotate_deg = key
        cluster_rates = []
        for cluster in sorted({row["cluster_key"] for row in rows}):
            selected = [row["success"] for row in rows if row["cluster_key"] == cluster]
            cluster_rates.append(float(np.mean(selected)))
        output.append(
            {
                "arm": arm,
                "kind": kind,
                "step": step,
                "translation_threshold_mm": move_mm,
                "rotation_threshold_deg": rotate_deg,
                "clusters": len(cluster_rates),
                "prompt_tests": len(rows),
                "success_rate": float(np.mean([row["success"] for row in rows])),
                "macro_cluster_success_rate": float(np.mean(cluster_rates)),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cluster-report", type=Path, required=True)
    parser.add_argument("--empty-baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    rng = np.random.default_rng(args.seed)
    cluster_noises = {
        cluster["cluster_key"]: rng.standard_normal(
            (config.action_horizon, config.action_dim), dtype=np.float32
        )
        for cluster in clusters
    }
    rows, noises = [], []
    for cluster in clusters:
        for atoms in MODES:
            rows.append(
                {
                    "cluster_key": cluster["cluster_key"],
                    "arm": cluster["arm"],
                    "kind": "single" if len(atoms) == 1 else "dual",
                    "atoms": list(atoms),
                    "dataset_index": cluster["dataset_index"],
                    "prompt": base._prompt(cluster["arm"], atoms),  # noqa: SLF001
                }
            )
            noises.append(cluster_noises[cluster["cluster_key"]])
    baseline_payload = json.loads(args.empty_baseline.read_text(encoding="utf-8"))
    model_records, model_summaries = {}, {}
    for name, checkpoint in checkpoints.items():
        baseline = {
            row["cluster_key"]: row["trajectory_twist"]
            for row in baseline_payload["outputs"][name]
            if row["repeat"] == 0
        }
        outputs = base._evaluate_checkpoint(  # noqa: SLF001
            name,
            Path(checkpoint),
            config,
            dataset,
            args.dataset_root,
            rows,
            np.stack(noises),
            batch_size=args.batch_size,
            stored_steps=(25, 50),
        )
        records = []
        for row in outputs:
            for step in (25, 50):
                empty = np.asarray(baseline[row["cluster_key"]][str(step)], dtype=np.float64)
                for move_mm, rotate_deg in THRESHOLDS:
                    records.append(_score(row, empty, step, move_mm, rotate_deg))
        summary = _summarize(records)
        model_records[name] = records
        model_summaries[name] = summary
        with (args.output_dir / f"{name}_bias_relative_endpoint_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    payload = {
        "checkpoints": checkpoints,
        "metric": "prompt endpoint minus empty-prompt endpoint; every target component must exceed its physical threshold",
        "thresholds": [
            {"translation_mm": move_mm, "rotation_deg": rotate_deg}
            for move_mm, rotate_deg in THRESHOLDS
        ],
        "clusters": len(clusters),
        "modes": len(MODES),
        "prompt_tests_per_model": len(rows),
        "summary": model_summaries,
        "records": model_records,
    }
    (args.output_dir / "bias_relative_endpoint_steering.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"summary": model_summaries}, indent=2))


if __name__ == "__main__":
    main()
