#!/usr/bin/env python3
"""Full-prompt Atomic↔Reverse steering on saved live observations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as openpi_model

from atomic_latent_vla.atomic import ATOMIC_NAMES
from atomic_latent_vla.pi05.config import AtomicPi05Config

import evaluate_global_episode_chunks as global_eval
import evaluate_live_trace_afro_prompt_causality as live
import evaluate_live_trace_q1_many as many
import evaluate_many_cluster_single_dual_steering as steer
import evaluate_zm_fk_trajectory_ablation as fk_eval


ARMS = ("right", "left")
ARM_JOINTS = {"right": slice(8, 15), "left": slice(0, 7)}
PAIRS = (
    ("move_x_pos", "move_x_neg"),
    ("move_y_pos", "move_y_neg"),
    ("move_z_pos", "move_z_neg"),
    ("rotate_x_pos", "rotate_x_neg"),
    ("rotate_y_pos", "rotate_y_neg"),
    ("rotate_z_pos", "rotate_z_neg"),
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _select(anchors: list[dict], per_prompt: int) -> list[dict]:
    groups = defaultdict(list)
    for anchor in anchors:
        groups[anchor["prompt"]].append(anchor)
    selected = []
    for rows in groups.values():
        count = min(per_prompt, len(rows))
        indices = np.linspace(0, len(rows) - 1, num=count, dtype=int)
        selected.extend(rows[int(index)] for index in sorted(set(indices.tolist())))
    return sorted(selected, key=lambda anchor: anchor["request_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace-csv", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--max-token-len", type=int, default=192)
    parser.add_argument("--anchors-per-prompt", type=int, default=4)
    parser.add_argument("--noise-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard specification")

    all_anchors = _select(many._load(args.trace_csv, args.images_dir), args.anchors_per_prompt)  # noqa: SLF001
    for global_index, anchor in enumerate(all_anchors):
        anchor["global_anchor_index"] = global_index
    anchors = [anchor for index, anchor in enumerate(all_anchors) if index % args.num_shards == args.shard_index]
    if not anchors:
        raise RuntimeError("empty anchor shard")
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
    atom_specs = [(arm, atom) for arm in ARMS for atom in ATOMIC_NAMES[:12]]
    rows = []
    raw_actions = []
    for anchor_index, anchor in enumerate(anchors):
        raw = many._raw(anchor)  # noqa: SLF001
        prompts = [steer._prompt(arm, (atom,)) for arm, atom in atom_specs]  # noqa: SLF001
        encoded = [live._encode(raw, prompt, input_transforms) for prompt in prompts]  # noqa: SLF001
        stacked = live._stack(encoded)  # noqa: SLF001
        normalized_state = np.asarray(stacked["state"])
        observation = openpi_model.Observation.from_dict(jax.tree.map(jnp.asarray, stacked))
        index_by_spec = {spec: index for index, spec in enumerate(atom_specs)}
        anchor_actions = []

        for noise_index in range(args.noise_repeats):
            global_anchor_index = anchor["global_anchor_index"]
            key = jax.random.fold_in(jax.random.key(args.seed), global_anchor_index * 100 + noise_index)
            one_noise = jax.random.normal(key, (1, config.action_horizon, config.action_dim))
            noise = jnp.repeat(one_noise, len(atom_specs), axis=0)
            sampled = np.asarray(
                jax.device_get(global_eval._sample_global(model, observation, noise, None, None, None))  # noqa: SLF001
            )[..., :16]
            decoded = np.stack(
                [
                    live._decode(normalized_state[index], anchor["state"], sampled[index], output_transforms)  # noqa: SLF001
                    for index in range(len(atom_specs))
                ]
            )
            anchor_actions.append(decoded)
            for arm in ARMS:
                other = "left" if arm == "right" else "right"
                for positive, negative in PAIRS:
                    pos_actions = decoded[index_by_spec[(arm, positive)]]
                    neg_actions = decoded[index_by_spec[(arm, negative)]]
                    pos_trajectory = steer._trajectory(fk, anchor["state"], pos_actions, arm)  # noqa: SLF001
                    neg_trajectory = steer._trajectory(fk, anchor["state"], neg_actions, arm)  # noqa: SLF001
                    component, _ = steer._component(positive)  # noqa: SLF001
                    separation = float(pos_trajectory[-1, component] - neg_trajectory[-1, component])
                    positive_endpoint = float(pos_trajectory[-1, component])
                    negative_endpoint = float(neg_trajectory[-1, component])
                    target_rmse = float(
                        np.sqrt(np.mean(np.square(pos_actions[:, ARM_JOINTS[arm]] - neg_actions[:, ARM_JOINTS[arm]])))
                    )
                    other_rmse = float(
                        np.sqrt(np.mean(np.square(pos_actions[:, ARM_JOINTS[other]] - neg_actions[:, ARM_JOINTS[other]])))
                    )
                    threshold = 10.0 if component < 3 else 2.0
                    rows.append(
                        {
                            "request_id": anchor["request_id"],
                            "native_subtask": anchor["prompt"],
                            "noise_index": noise_index,
                            "arm": arm,
                            "positive_atom": positive,
                            "negative_atom": negative,
                            "unit": "mm" if component < 3 else "deg",
                            "positive_endpoint": positive_endpoint,
                            "negative_endpoint": negative_endpoint,
                            "paired_separation": separation,
                            "paired_order_success": separation > 0,
                            "paired_threshold_success": separation > threshold,
                            "absolute_both_sign_success": positive_endpoint > 0 and negative_endpoint < 0,
                            "target_joint_rmse_rad": target_rmse,
                            "other_joint_rmse_rad": other_rmse,
                        }
                    )
        raw_actions.append(np.stack(anchor_actions))
        print(f"anchor {anchor_index + 1}/{len(anchors)} request={anchor['request_id']}", flush=True)

    _write_csv(args.output_dir / "rows.csv", rows)
    np.savez_compressed(
        args.output_dir / "raw_actions.npz",
        actions=np.stack(raw_actions),
        request_ids=np.asarray([anchor["request_id"] for anchor in anchors]),
        atoms=np.asarray(atom_specs),
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "global_anchors": len(all_anchors),
        "shard_anchors": len(anchors),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "noise_repeats": args.noise_repeats,
        "rows": len(rows),
        "paired_order_success": float(np.mean([row["paired_order_success"] for row in rows])),
        "paired_threshold_success": float(np.mean([row["paired_threshold_success"] for row in rows])),
        "absolute_both_sign_success": float(np.mean([row["absolute_both_sign_success"] for row in rows])),
        "mean_target_joint_rmse_rad": float(np.mean([row["target_joint_rmse_rad"] for row in rows])),
        "mean_other_joint_rmse_rad": float(np.mean([row["other_joint_rmse_rad"] for row in rows])),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
