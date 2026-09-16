#!/usr/bin/env python3
"""Evaluate native Subtask-conditioned Q1 on many live trace observations."""

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
from PIL import Image

from openpi.models import model as openpi_model

from atomic_latent_vla.atomic import ATOMIC_NAMES
from atomic_latent_vla.pi05.config import AtomicPi05Config

import evaluate_live_trace_afro_prompt_causality as live


CAMERAS = ("cam_left_wrist", "cam_right_wrist", "cam_high")
ARMS = ("right", "left")


@nnx.jit
def _encode(model, observation):
    observation = openpi_model.preprocess_observation(None, observation, train=False)
    query_hidden, _, _, _ = model._prefix_forward(observation)  # noqa: SLF001
    state = model._controlled_state(observation.state)  # noqa: SLF001
    _, right, z_model, left, _ = model._latent(query_hidden, state)  # noqa: SLF001
    codes = model.codebook.value
    codes = codes / jnp.maximum(jnp.linalg.norm(codes, axis=-1, keepdims=True), 1.0e-8)
    return jnp.stack([right, left], axis=1), z_model, codes


def _load(trace_csv: Path, images_dir: Path) -> list[dict]:
    anchors = []
    with trace_csv.open(newline="", encoding="utf-8", errors="ignore") as stream:
        for row in csv.DictReader(stream):
            if int(row["action_step"]) != 0:
                continue
            image_paths = {camera: images_dir / Path(row[f"{camera}_image"]).name for camera in CAMERAS}
            if not all(path.is_file() for path in image_paths.values()):
                continue
            anchors.append(
                {
                    "request_id": int(row["request_id"]),
                    "timestamp": row["timestamp"],
                    "prompt": " ".join(row["prompt"].split()),
                    "state": np.asarray([float(row[f"state_{index}"]) for index in range(16)], dtype=np.float32),
                    "images": image_paths,
                }
            )
    return anchors


def _raw(anchor: dict) -> dict:
    images = {}
    for camera, path in anchor["images"].items():
        value = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        if value.shape != (224, 224, 3):
            raise ValueError(f"{path} has shape {value.shape}, expected (224,224,3)")
        images[camera] = np.moveaxis(value, -1, 0)
    return {"state": anchor["state"], "images": images}


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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    anchors = _load(args.trace_csv, args.images_dir)
    if not anchors:
        raise RuntimeError("no complete trace anchors")
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
    output_rows = []
    raw_directions, raw_zm = [], []
    codes_np = None

    for start in range(0, len(anchors), args.batch_size):
        chunk = anchors[start : start + args.batch_size]
        encoded = [live._encode(_raw(anchor), anchor["prompt"], input_transforms) for anchor in chunk]  # noqa: SLF001
        observation = openpi_model.Observation.from_dict(jax.tree.map(jnp.asarray, live._stack(encoded)))  # noqa: SLF001
        directions, z_model, codes = jax.device_get(_encode(model, observation))
        directions = np.asarray(directions, dtype=np.float32)
        z_model = np.asarray(z_model, dtype=np.float32)
        codes_np = np.asarray(codes, dtype=np.float32)
        raw_directions.append(directions)
        raw_zm.append(z_model)
        for local, anchor in enumerate(chunk):
            for arm_index, arm in enumerate(ARMS):
                similarities = codes_np[arm_index] @ directions[local, arm_index]
                order = np.argsort(-similarities)
                output_rows.append(
                    {
                        "request_id": anchor["request_id"],
                        "timestamp": anchor["timestamp"],
                        "prompt": anchor["prompt"],
                        "arm": arm,
                        "top1": ATOMIC_NAMES[int(order[0])],
                        "top1_cos": float(similarities[order[0]]),
                        "top2": ATOMIC_NAMES[int(order[1])],
                        "top2_cos": float(similarities[order[1]]),
                        "top3": ATOMIC_NAMES[int(order[2])],
                        "top3_cos": float(similarities[order[2]]),
                        "stay_cos": float(similarities[ATOMIC_NAMES.index("stay")]),
                        "top1_margin": float(similarities[order[0]] - similarities[order[1]]),
                    }
                )
        print(f"encoded {min(start + args.batch_size, len(anchors))}/{len(anchors)}", flush=True)

    _write_csv(args.output_dir / "per_arm_q1.csv", output_rows)
    np.savez_compressed(
        args.output_dir / "raw_q1.npz",
        directions=np.concatenate(raw_directions),
        zm=np.concatenate(raw_zm),
        codebook=codes_np,
        request_ids=np.asarray([anchor["request_id"] for anchor in anchors]),
    )
    groups = defaultdict(list)
    for row in output_rows:
        groups[(row["prompt"], row["arm"])].append(row)
    group_rows = []
    for (prompt, arm), rows in sorted(groups.items()):
        counts = Counter(row["top1"] for row in rows)
        common = counts.most_common(3)
        group_rows.append(
            {
                "prompt": prompt,
                "arm": arm,
                "count": len(rows),
                "top1_stay_fraction": float(np.mean([row["top1"] == "stay" for row in rows])),
                "mean_top1_cos": float(np.mean([row["top1_cos"] for row in rows])),
                "mean_top1_margin": float(np.mean([row["top1_margin"] for row in rows])),
                "prediction_1": common[0][0],
                "prediction_1_fraction": common[0][1] / len(rows),
                "prediction_2": common[1][0] if len(common) > 1 else "",
                "prediction_2_fraction": common[1][1] / len(rows) if len(common) > 1 else 0.0,
                "prediction_3": common[2][0] if len(common) > 2 else "",
                "prediction_3_fraction": common[2][1] / len(rows) if len(common) > 2 else 0.0,
            }
        )
    _write_csv(args.output_dir / "prompt_arm_summary.csv", group_rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "trace_csv": str(args.trace_csv),
        "anchors": len(anchors),
        "per_arm_rows": len(output_rows),
        "top1_stay_fraction": float(np.mean([row["top1"] == "stay" for row in output_rows])),
        "mean_top1_cos": float(np.mean([row["top1_cos"] for row in output_rows])),
        "mean_top1_margin": float(np.mean([row["top1_margin"] for row in output_rows])),
        "prompt_arm_rows": group_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "prompt_arm_rows"}, indent=2))


if __name__ == "__main__":
    main()
