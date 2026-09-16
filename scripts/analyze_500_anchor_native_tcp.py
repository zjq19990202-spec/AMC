#!/usr/bin/env python3
"""Compare native-atomic free-rollout TCP endpoints with ground truth."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import build_atomic_dataset

import evaluate_many_cluster_single_dual_steering as steering
import evaluate_zm_fk_trajectory_ablation as fk_eval


def _endpoint_error(predicted: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    translation_mm = float(np.linalg.norm(predicted[:3] - target[:3]))
    predicted_rotation = Rotation.from_rotvec(np.deg2rad(predicted[3:]))
    target_rotation = Rotation.from_rotvec(np.deg2rad(target[3:]))
    rotation_deg = float(
        np.rad2deg((predicted_rotation * target_rotation.inv()).magnitude())
    )
    return translation_mm, rotation_deg


def _direction_success(twist: np.ndarray, atoms: list[str], move_mm: float, rotate_deg: float) -> bool:
    margins = []
    for atom in atoms:
        component, sign = steering._component(atom)  # noqa: SLF001
        threshold = move_mm if component < 3 else rotate_deg
        margins.append(sign * twist[component] > threshold)
    return bool(all(margins))


def _summary(rows: list[dict]) -> dict:
    translation = np.asarray([row["translation_error_mm"] for row in rows])
    rotation = np.asarray([row["rotation_error_deg"] for row in rows])
    return {
        "count": len(rows),
        "translation_error_mm_mean": float(translation.mean()),
        "translation_error_mm_median": float(np.median(translation)),
        "translation_error_mm_p90": float(np.quantile(translation, 0.9)),
        "rotation_error_deg_mean": float(rotation.mean()),
        "rotation_error_deg_median": float(np.median(rotation)),
        "rotation_error_deg_p90": float(np.quantile(rotation, 0.9)),
        "ground_truth_direction_success_rate": float(
            np.mean([row["ground_truth_direction_success"] for row in rows])
        ),
        "predicted_direction_success_rate": float(
            np.mean([row["predicted_direction_success"] for row in rows])
        ),
    }


def _plot(summary: dict[str, dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    categories = [
        (arm, kind, step)
        for step in (25, 50)
        for arm in ("right", "left")
        for kind in ("single", "dual")
    ]
    labels = [f"s{step}\n{arm}\n{kind}" for arm, kind, step in categories]
    translation = [summary[f"{arm}/{kind}/s{step}"]["translation_error_mm_mean"] for arm, kind, step in categories]
    rotation = [summary[f"{arm}/{kind}/s{step}"]["rotation_error_deg_mean"] for arm, kind, step in categories]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].bar(np.arange(len(labels)), translation, color="#2563EB")
    axes[1].bar(np.arange(len(labels)), rotation, color="#F59E0B")
    for axis, title, ylabel in zip(
        axes,
        ("Native-prompt TCP translation error", "Native-prompt TCP rotation error"),
        ("mean endpoint error (mm)", "mean endpoint error (deg)"),
        strict=True,
    ):
        axis.set_xticks(np.arange(len(labels)), labels)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Old bimanual 80K · 500 clustered anchors · TCP 0.20 m", fontweight="bold")
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--translation-threshold-mm", type=float, default=5.0)
    parser.add_argument("--rotation-threshold-deg", type=float, default=1.0)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    shards = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.input_dir.glob("shard_*.json"))
    ]
    if len(shards) != 4:
        raise RuntimeError(f"expected four shards, got {len(shards)}")
    native = {
        row["anchor_key"]: row
        for shard in shards
        for row in shard["outputs"]
        if row["variant"] == "native_atomic"
    }
    if len(native) != 500:
        raise RuntimeError(f"expected 500 native predictions, got {len(native)}")

    config = AtomicPi05Config(max_token_len=250, fast_action_ce_loss_weight=0.0)
    dataset = build_atomic_dataset(
        (args.dataset_root,),
        norm_assets_dir="/mnt/cunchu/zjq/target",
        norm_asset_id="openpi_norm_compact_accepted_v3",
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        include_fast=False,
    )
    fk_eval.TCP_LOCAL_Z_OFFSET_M = 0.20
    fk = fk_eval.SimpleCR1FK(Path("assets/cr1_right_arm_tcp_red/urdf/CR1ARMR_TCP_RED.urdf"))
    rows = []
    for anchor_key, prediction in sorted(native.items()):
        metadata = dataset._raw.metadata(int(prediction["dataset_index"]))  # noqa: SLF001
        ground_truth = steering._trajectory(  # noqa: SLF001
            fk,
            np.asarray(metadata["raw_state"]),
            np.asarray(metadata["raw_actions"]),
            prediction["arm"],
        )
        atoms = list(prediction["requested_atoms"])
        for step in (25, 50):
            predicted = np.asarray(prediction["trajectory_twist"][str(step)])
            target = ground_truth[step]
            translation_error, rotation_error = _endpoint_error(predicted, target)
            rows.append(
                {
                    "anchor_key": anchor_key,
                    "cluster_key": prediction["cluster_key"],
                    "episode": prediction["episode"],
                    "frame": prediction["frame"],
                    "arm": prediction["arm"],
                    "kind": prediction["kind"],
                    "other_status": prediction["other_status"],
                    "atoms": "+".join(atoms),
                    "step": step,
                    "translation_error_mm": translation_error,
                    "rotation_error_deg": rotation_error,
                    "ground_truth_direction_success": _direction_success(
                        target, atoms, args.translation_threshold_mm, args.rotation_threshold_deg
                    ),
                    "predicted_direction_success": _direction_success(
                        predicted, atoms, args.translation_threshold_mm, args.rotation_threshold_deg
                    ),
                }
            )

    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        for arm in (row["arm"], "both"):
            for kind in (row["kind"], "all"):
                grouped[(arm, kind, row["step"])].append(row)
    summary = {
        f"{arm}/{kind}/s{step}": _summary(group)
        for (arm, kind, step), group in sorted(grouped.items())
    }
    report = {
        "checkpoint": shards[0]["checkpoint"],
        "tcp_offset_m": 0.20,
        "thresholds": {
            "translation_mm": args.translation_threshold_mm,
            "rotation_deg": args.rotation_threshold_deg,
        },
        "summary": summary,
        "rows": rows,
    }
    (output_dir / "native_atomic_tcp_vs_gt.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "native_atomic_tcp_vs_gt.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(summary, output_dir / "native_atomic_tcp_vs_gt.png")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
