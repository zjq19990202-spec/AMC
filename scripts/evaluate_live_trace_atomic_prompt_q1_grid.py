#!/usr/bin/env python3
"""Evaluate all single-atom prompts on many saved live observations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from openpi.models import model as openpi_model

from atomic_latent_vla.atomic import ATOMIC_NAMES
from atomic_latent_vla.pi05.config import AtomicPi05Config

import evaluate_live_trace_afro_prompt_causality as live
import evaluate_live_trace_q1_many as many
import evaluate_many_cluster_single_dual_steering as steer


ARMS = ("right", "left")
ATOMS = tuple(ATOMIC_NAMES[:12])


@nnx.jit
def _encode(model, observation):
    observation = openpi_model.preprocess_observation(None, observation, train=False)
    query_hidden, _, _, _ = model._prefix_forward(observation)  # noqa: SLF001
    state = model._controlled_state(observation.state)  # noqa: SLF001
    _, right, z_model, left, _ = model._latent(query_hidden, state)  # noqa: SLF001
    codes = model.codebook.value
    codes = codes / jnp.maximum(jnp.linalg.norm(codes, axis=-1, keepdims=True), 1.0e-8)
    return jnp.stack([right, left], axis=1), z_model, codes


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
    parser.add_argument("--max-token-len", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard specification")

    all_anchors = many._load(args.trace_csv, args.images_dir)  # noqa: SLF001
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
    input_transforms, _ = live._build_transforms(args, config)  # noqa: SLF001
    params = openpi_model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()

    specs = []
    for anchor in anchors:
        raw = many._raw(anchor)  # noqa: SLF001
        for arm_index, arm in enumerate(ARMS):
            for atom in ATOMS:
                specs.append(
                    {
                        "anchor": anchor,
                        "raw": raw,
                        "arm": arm,
                        "arm_index": arm_index,
                        "atom": atom,
                        "atom_index": ATOMIC_NAMES.index(atom),
                        "prompt": steer._prompt(arm, (atom,)),  # noqa: SLF001
                    }
                )

    rows = []
    directions_values, zm_values = [], []
    codebook = None
    for start in range(0, len(specs), args.batch_size):
        chunk = specs[start : start + args.batch_size]
        encoded = [live._encode(spec["raw"], spec["prompt"], input_transforms) for spec in chunk]  # noqa: SLF001
        observation = openpi_model.Observation.from_dict(jax.tree.map(jnp.asarray, live._stack(encoded)))  # noqa: SLF001
        directions, z_model, codes = jax.device_get(_encode(model, observation))
        directions = np.asarray(directions, dtype=np.float32)
        z_model = np.asarray(z_model, dtype=np.float32)
        codebook = np.asarray(codes, dtype=np.float32)
        directions_values.append(directions)
        zm_values.append(z_model)
        for local, spec in enumerate(chunk):
            target_similarities = codebook[spec["arm_index"]] @ directions[local, spec["arm_index"]]
            other_index = 1 - spec["arm_index"]
            other_similarities = codebook[other_index] @ directions[local, other_index]
            target_order = np.argsort(-target_similarities)
            other_order = np.argsort(-other_similarities)
            rows.append(
                {
                    "request_id": spec["anchor"]["request_id"],
                    "native_prompt": spec["anchor"]["prompt"],
                    "atomic_prompt": spec["prompt"],
                    "requested_arm": spec["arm"],
                    "requested_atom": spec["atom"],
                    "requested_code_cos": float(target_similarities[spec["atom_index"]]),
                    "requested_is_top1": bool(int(target_order[0]) == spec["atom_index"]),
                    "requested_is_top5": bool(spec["atom_index"] in target_order[:5]),
                    "target_top1": ATOMIC_NAMES[int(target_order[0])],
                    "target_top1_cos": float(target_similarities[target_order[0]]),
                    "target_stay_cos": float(target_similarities[ATOMIC_NAMES.index("stay")]),
                    "other_top1": ATOMIC_NAMES[int(other_order[0])],
                    "other_top1_cos": float(other_similarities[other_order[0]]),
                    "other_top1_is_stay": bool(int(other_order[0]) == ATOMIC_NAMES.index("stay")),
                }
            )
        print(f"encoded {min(start + args.batch_size, len(specs))}/{len(specs)}", flush=True)

    _write_csv(args.output_dir / "rows.csv", rows)
    np.savez_compressed(
        args.output_dir / "raw_q1.npz",
        directions=np.concatenate(directions_values),
        zm=np.concatenate(zm_values),
        codebook=codebook,
    )
    groups = defaultdict(list)
    for row in rows:
        groups[(row["requested_arm"], row["requested_atom"])].append(row)
    group_rows = []
    for (arm, atom), group in sorted(groups.items()):
        group_rows.append(
            {
                "arm": arm,
                "atom": atom,
                "count": len(group),
                "requested_top1_rate": float(np.mean([row["requested_is_top1"] for row in group])),
                "requested_top5_rate": float(np.mean([row["requested_is_top5"] for row in group])),
                "mean_requested_code_cos": float(np.mean([row["requested_code_cos"] for row in group])),
                "other_arm_stay_rate": float(np.mean([row["other_top1_is_stay"] for row in group])),
                "dominant_target_top1": Counter(row["target_top1"] for row in group).most_common(1)[0][0],
            }
        )
    _write_csv(args.output_dir / "per_atom_summary.csv", group_rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "global_anchors": len(all_anchors),
        "shard_anchors": len(anchors),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "rows": len(rows),
        "requested_top1_rate": float(np.mean([row["requested_is_top1"] for row in rows])),
        "requested_top5_rate": float(np.mean([row["requested_is_top5"] for row in rows])),
        "mean_requested_code_cos": float(np.mean([row["requested_code_cos"] for row in rows])),
        "other_arm_stay_rate": float(np.mean([row["other_top1_is_stay"] for row in rows])),
        "per_atom": group_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "per_atom"}, indent=2))


if __name__ == "__main__":
    main()
