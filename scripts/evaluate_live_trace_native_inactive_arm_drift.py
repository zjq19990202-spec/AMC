#!/usr/bin/env python3
"""Measure inactive-arm drift for native Subtask inference on live observations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as openpi_model

from atomic_latent_vla.pi05.config import AtomicPi05Config

import evaluate_global_episode_chunks as global_eval
import evaluate_live_trace_afro_prompt_causality as live
import evaluate_live_trace_q1_many as many
import evaluate_many_cluster_single_dual_steering as steer
import evaluate_zm_fk_trajectory_ablation as fk_eval


JOINTS = {"left": slice(0, 7), "right": slice(8, 15)}


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace-csv", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--anchors", type=int, default=4)
    parser.add_argument("--noise-repeats", type=int, default=4)
    parser.add_argument("--max-token-len", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    candidates = [
        anchor for anchor in many._load(args.trace_csv, args.images_dir)  # noqa: SLF001
        if anchor["prompt"] == args.prompt
    ]
    anchors = sorted(candidates, key=lambda anchor: anchor["request_id"])[: args.anchors]
    if len(anchors) != args.anchors:
        raise RuntimeError(f"found only {len(anchors)} matching anchors")
    config = AtomicPi05Config(
        max_token_len=args.max_token_len,
        coefficient_target_kind="joint_delta",
        coefficient_target_dim=14,
        enable_layerwise_atomic_flow=False,
        zm_teacher_action_conditioning=True,
        fast_action_ce_loss_weight=0.0,
        subtask_ce_loss_weight=0.0,
    )
    input_transforms, output_transforms = live._build_transforms(args, config)  # noqa: SLF001
    params = openpi_model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    fk_eval.TCP_LOCAL_Z_OFFSET_M = 0.20
    fk = fk_eval.SimpleCR1FK(Path("assets/cr1_right_arm_tcp_red/urdf/CR1ARMR_TCP_RED.urdf"))

    rows = []
    raw_actions = []
    for anchor_index, anchor in enumerate(anchors):
        encoded = live._encode(many._raw(anchor), anchor["prompt"], input_transforms)  # noqa: SLF001
        stacked = live._stack([encoded])  # noqa: SLF001
        normalized_state = np.asarray(stacked["state"])
        observation = openpi_model.Observation.from_dict(jax.tree.map(jnp.asarray, stacked))
        anchor_actions = []
        for noise_index in range(args.noise_repeats):
            key = jax.random.fold_in(jax.random.key(args.seed), anchor_index * 100 + noise_index)
            noise = jax.random.normal(key, (1, config.action_horizon, config.action_dim))
            sampled = np.asarray(
                jax.device_get(global_eval._sample_global(model, observation, noise, None, None, None))  # noqa: SLF001
            )[0, ..., :16]
            decoded = live._decode(normalized_state[0], anchor["state"], sampled, output_transforms)  # noqa: SLF001
            anchor_actions.append(decoded)
            for arm in ("right", "left"):
                joint_slice = JOINTS[arm]
                initial = anchor["state"][joint_slice]
                deltas = decoded[:, joint_slice] - initial
                trajectory = steer._trajectory(fk, anchor["state"], decoded, arm)  # noqa: SLF001
                endpoint = trajectory[-1]
                rows.append(
                    {
                        "request_id": anchor["request_id"],
                        "noise_index": noise_index,
                        "arm": arm,
                        "first_joint_rmse_rad": float(np.sqrt(np.mean(np.square(deltas[0])))),
                        "endpoint_joint_rmse_rad": float(np.sqrt(np.mean(np.square(deltas[-1])))),
                        "max_horizon_joint_abs_rad": float(np.max(np.abs(deltas))),
                        "endpoint_translation_mm": float(np.linalg.norm(endpoint[:3])),
                        "endpoint_rotation_deg": float(np.linalg.norm(endpoint[3:])),
                        "max_translation_mm": float(np.max(np.linalg.norm(trajectory[:, :3], axis=1))),
                        "max_rotation_deg": float(np.max(np.linalg.norm(trajectory[:, 3:], axis=1))),
                        "last_wrist_endpoint_delta_rad": float(deltas[-1, -1]),
                        "last_wrist_max_abs_delta_rad": float(np.max(np.abs(deltas[:, -1]))),
                    }
                )
        raw_actions.append(np.stack(anchor_actions))
        print(f"anchor {anchor_index + 1}/{len(anchors)} request={anchor['request_id']}", flush=True)

    _write_csv(args.output_dir / "rows.csv", rows)
    np.savez_compressed(
        args.output_dir / "raw_actions.npz",
        actions=np.stack(raw_actions),
        request_ids=np.asarray([anchor["request_id"] for anchor in anchors]),
        raw_states=np.stack([anchor["state"] for anchor in anchors]),
    )
    per_arm = []
    for arm in ("right", "left"):
        group = [row for row in rows if row["arm"] == arm]
        per_arm.append(
            {
                "arm": arm,
                "count": len(group),
                **{
                    key: float(np.mean([row[key] for row in group]))
                    for key in (
                        "first_joint_rmse_rad",
                        "endpoint_joint_rmse_rad",
                        "max_horizon_joint_abs_rad",
                        "endpoint_translation_mm",
                        "endpoint_rotation_deg",
                        "max_translation_mm",
                        "max_rotation_deg",
                        "last_wrist_endpoint_delta_rad",
                        "last_wrist_max_abs_delta_rad",
                    )
                },
            }
        )
    _write_csv(args.output_dir / "per_arm_summary.csv", per_arm)
    summary = {
        "checkpoint": str(args.checkpoint),
        "prompt": args.prompt,
        "request_ids": [anchor["request_id"] for anchor in anchors],
        "noise_repeats": args.noise_repeats,
        "per_arm": per_arm,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
