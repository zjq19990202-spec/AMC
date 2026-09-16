#!/usr/bin/env python3
"""Sweep every single/dual atom at broad state clusters and score target angles."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import jax
import numpy as np

from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import build_atomic_dataset

import evaluate_many_cluster_single_dual_steering as base


ATOMS = tuple(base.PHRASES)
SINGLE_MODES = tuple((atom,) for atom in ATOMS)
DUAL_MODES = tuple(
    pair for pair in itertools.combinations(ATOMS, 2) if base._valid_mode(pair)  # noqa: SLF001
)


def _target_groups(atoms: tuple[str, ...]) -> list[tuple[str, np.ndarray]]:
    translation = np.zeros(3, dtype=np.float64)
    rotation = np.zeros(3, dtype=np.float64)
    for atom in atoms:
        coordinate, sign = base._component(atom)  # noqa: SLF001
        if coordinate < 3:
            translation[coordinate] += sign
        else:
            rotation[coordinate - 3] += sign
    groups = []
    if np.linalg.norm(translation) > 0:
        groups.append(("translation", translation / np.linalg.norm(translation)))
    if np.linalg.norm(rotation) > 0:
        groups.append(("rotation", rotation / np.linalg.norm(rotation)))
    return groups


def _angle_degrees(vector: np.ndarray, target: np.ndarray) -> float:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return 180.0
    cosine = float(np.clip(np.dot(vector, target) / norm, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _score_row(row: dict, step: int) -> dict:
    twist = np.asarray(row["trajectory_twist"][str(step)], dtype=np.float64)
    atoms = tuple(row["atoms"])
    components = [base._component(atom) for atom in atoms]  # noqa: SLF001
    component_sign_success = all(sign * twist[index] > 0 for index, sign in components)
    angles, magnitudes = {}, {}
    magnitude_success = True
    for family, target in _target_groups(atoms):
        vector = twist[:3] if family == "translation" else twist[3:]
        angles[family] = _angle_degrees(vector, target)
        magnitudes[family] = float(np.linalg.norm(vector))
        threshold = 5.0 if family == "translation" else 1.0
        magnitude_success = magnitude_success and magnitudes[family] >= threshold
    max_angle = max(angles.values())
    return {
        "cluster_key": row["cluster_key"],
        "arm": row["arm"],
        "kind": row["kind"],
        "atoms": "+".join(atoms),
        "step": step,
        "component_sign_success": bool(component_sign_success),
        "magnitude_success": bool(magnitude_success),
        "max_group_angle_deg": float(max_angle),
        "angle_45_success": bool(component_sign_success and magnitude_success and max_angle <= 45.0),
        "angle_60_success": bool(component_sign_success and magnitude_success and max_angle <= 60.0),
        "translation_angle_deg": angles.get("translation"),
        "rotation_angle_deg": angles.get("rotation"),
        "translation_magnitude_mm": magnitudes.get("translation"),
        "rotation_magnitude_deg": magnitudes.get("rotation"),
    }


def _summarize(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in records:
        for arm in (row["arm"], "both"):
            for kind in (row["kind"], "all"):
                groups[(arm, kind, row["step"])].append(row)
    output = []
    for (arm, kind, step), rows in sorted(groups.items()):
        cluster_rates_45, cluster_rates_60 = [], []
        for cluster in sorted({row["cluster_key"] for row in rows}):
            cluster_rows = [row for row in rows if row["cluster_key"] == cluster]
            cluster_rates_45.append(np.mean([row["angle_45_success"] for row in cluster_rows]))
            cluster_rates_60.append(np.mean([row["angle_60_success"] for row in cluster_rows]))
        output.append(
            {
                "arm": arm,
                "kind": kind,
                "step": step,
                "clusters": len(cluster_rates_45),
                "prompt_tests": len(rows),
                "component_sign_rate": float(np.mean([row["component_sign_success"] for row in rows])),
                "angle_45_micro_rate": float(np.mean([row["angle_45_success"] for row in rows])),
                "angle_45_macro_cluster_rate": float(np.mean(cluster_rates_45)),
                "angle_60_micro_rate": float(np.mean([row["angle_60_success"] for row in rows])),
                "angle_60_macro_cluster_rate": float(np.mean(cluster_rates_60)),
                "median_max_group_angle_deg": float(np.median([row["max_group_angle_deg"] for row in rows])),
            }
        )
    return output


def _write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cluster-report", type=Path, required=True)
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
    modes = SINGLE_MODES + DUAL_MODES
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
    rows, noises = [], []
    for cluster in clusters:
        cluster_noise = rng.standard_normal(
            (config.action_horizon, config.action_dim), dtype=np.float32
        )
        for atoms in modes:
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
            noises.append(cluster_noise)
    noises_array = np.stack(noises)
    model_records, model_summaries = {}, {}
    for name, checkpoint in checkpoints.items():
        compact_outputs = base._evaluate_checkpoint(  # noqa: SLF001
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
        records = [
            _score_row(row, step)
            for row in compact_outputs
            for step in (25, 50)
        ]
        summary = _summarize(records)
        model_records[name] = records
        model_summaries[name] = summary
        _write_csv(summary, args.output_dir / f"{name}_arbitrary_atom_angle_summary.csv")
    payload = {
        "checkpoints": checkpoints,
        "clusters": clusters,
        "single_modes": len(SINGLE_MODES),
        "dual_modes": len(DUAL_MODES),
        "prompt_tests_per_model": len(rows),
        "metric": {
            "translation_magnitude_min_mm": 5.0,
            "rotation_magnitude_min_deg": 1.0,
            "angle_thresholds_deg": [45.0, 60.0],
            "dual_same_family": "equal-weight target direction within translation or rotation 3-space",
            "dual_mixed_family": "translation and rotation angles scored separately; both must pass",
        },
        "summary": model_summaries,
        "records": model_records,
    }
    (args.output_dir / "broad_arbitrary_atom_angle_steering.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"clusters": len(clusters), "modes": len(modes), "summary": model_summaries}, indent=2))


if __name__ == "__main__":
    main()
