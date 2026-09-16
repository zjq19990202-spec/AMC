#!/usr/bin/env python3
"""Evaluate visual Subtask Q1 across every Vase semantic-segment occurrence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from openpi.models import model as openpi_model

from atomic_latent_vla.atomic import ATOMIC_NAMES
from atomic_latent_vla.pi05.config import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import (
    atomic_collate,
    batch_to_observation,
    build_atomic_dataset,
)


ARMS = ("right", "left")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@nnx.jit
def _encode(model, observation):
    observation = openpi_model.preprocess_observation(None, observation, train=False)
    query_hidden, _, _, _ = model._prefix_forward(observation)  # noqa: SLF001
    state = model._controlled_state(observation.state)  # noqa: SLF001
    _, right, z_model, left, _ = model._latent(query_hidden, state)  # noqa: SLF001
    codes = model.codebook.value
    codes = codes / jnp.maximum(jnp.linalg.norm(codes, axis=-1, keepdims=True), 1.0e-8)
    return jnp.stack([right, left], axis=1), z_model, codes


def _segment_rows(sidecar: Path, phases: tuple[float, ...]) -> list[dict]:
    rows = []
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        episode = json.loads(line)
        episode_index = int(episode["episode_index"])
        for segment_index, segment in enumerate(episode["semantic_segments"]):
            start = int(segment["start_frame_30hz"])
            end = int(segment["end_frame_30hz_exclusive"])
            if end <= start:
                continue
            for phase in phases:
                raw_frame = start + phase * (end - start - 1)
                frame = int(round(raw_frame / 10.0) * 10)
                frame = min(max(frame, start), end - 1)
                rows.append(
                    {
                        "episode": episode_index,
                        "segment_index": segment_index,
                        "segment_occurrence": segment_index + 1,
                        "segment_start": start,
                        "segment_end_exclusive": end,
                        "phase": phase,
                        "frame": frame,
                        "subtask": str(segment["current_subtask"]),
                    }
                )
    unique = {}
    for row in rows:
        unique[(row["episode"], row["frame"], row["segment_index"])] = row
    return list(unique.values())


def _dataset_index_map(dataset) -> dict[tuple[int, int], int]:
    raw = dataset._raw  # noqa: SLF001
    base = raw.base
    view = (
        np.arange(len(base), dtype=np.int64)
        if raw._sample_indices is None  # noqa: SLF001
        else np.asarray(raw._sample_indices, dtype=np.int64)  # noqa: SLF001
    )
    visible = getattr(base, "_visible_indices", None)
    data = view if visible is None else np.asarray(visible, dtype=np.int64)[view]
    episodes = np.asarray(base._episode_index[data], dtype=np.int64)  # noqa: SLF001
    frames = np.asarray(base._frame_index[data], dtype=np.int64)  # noqa: SLF001
    return {(int(ep), int(frame)): index for index, (ep, frame) in enumerate(zip(episodes, frames))}


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--atomic-composition-sidecar", required=True)
    parser.add_argument("--max-token-len", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    phases = (0.1, 0.5, 0.9)
    requested = _segment_rows(args.dataset_root / "meta/episode_subtasks.jsonl", phases)

    config = AtomicPi05Config(
        max_token_len=args.max_token_len,
        coefficient_target_kind="joint_delta",
        coefficient_target_dim=14,
        enable_layerwise_atomic_flow=False,
        zm_teacher_action_conditioning=True,
        fast_action_ce_loss_weight=0.0,
        subtask_ce_loss_weight=0.0,
    )
    dataset = build_atomic_dataset(
        (args.dataset_root,),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        atomic_composition_sidecar=args.atomic_composition_sidecar,
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        include_fast=False,
        pad_subtask_horizon=True,
    )
    index_map = _dataset_index_map(dataset)
    selected = []
    for row in requested:
        index = index_map.get((row["episode"], row["frame"]))
        if index is None:
            continue
        metadata = dataset._raw.metadata(index)  # noqa: SLF001
        selected.append({**row, "dataset_index": index, "dataset_subtask": metadata["subtask_prompt"]})
    if not selected:
        raise RuntimeError("no requested segment anchors survived the dataset view")
    if len(selected) > 10000:
        raise ValueError(f"selection exceeds latent-eval cap: {len(selected)}")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards > 0 and 0 <= shard_index < num_shards")
    global_selected_count = len(selected)
    selected = [
        row for index, row in enumerate(selected)
        if index % args.num_shards == args.shard_index
    ]
    if not selected:
        raise RuntimeError("this shard has no selected rows")

    (args.output_dir / "selection.json").write_text(
        json.dumps(
            {
                "selection_basis": "all Vase semantic segment occurrences at 10%, 50%, 90% phase; frame rounded to 10-frame atomic grid",
                "phases": phases,
                "requested": len(requested),
                "global_selected": global_selected_count,
                "selected": len(selected),
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "rows": selected,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    params = openpi_model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    output_rows = []
    vectors, zm_values = [], []
    codes_np = None

    for start in range(0, len(selected), args.batch_size):
        chunk = selected[start : start + args.batch_size]
        samples = [dataset[row["dataset_index"]] for row in chunk]
        batch = atomic_collate(samples)
        observation_np, _ = batch_to_observation(batch)
        observation = jax.tree.map(jnp.asarray, observation_np)
        observation = model._with_prompt(  # noqa: SLF001
            observation,
            jnp.asarray(batch["subtask_prompt_tokens"]),
            jnp.asarray(batch["subtask_prompt_mask"]),
        )
        directions, z_model, codes = jax.device_get(_encode(model, observation))
        directions = np.asarray(directions, dtype=np.float32)
        z_model = np.asarray(z_model, dtype=np.float32)
        codes_np = np.asarray(codes, dtype=np.float32)
        vectors.append(directions)
        zm_values.append(z_model)

        strict_weights = np.asarray(batch["atomic_weights"], dtype=np.float32)
        strict_mask = np.asarray(batch["atomic_supervision_mask"], dtype=bool)
        composition_weights = np.asarray(batch["atomic_composition_weights"], dtype=np.float32)
        composition_mask = np.asarray(batch["atomic_composition_mask"], dtype=bool)
        for local, spec in enumerate(chunk):
            for arm_index, arm in enumerate(ARMS):
                similarities = codes_np[arm_index] @ directions[local, arm_index]
                order = np.argsort(-similarities)
                if strict_mask[local, arm_index]:
                    target_kind = "strict"
                    weights = strict_weights[local, arm_index]
                elif composition_mask[local, arm_index]:
                    target_kind = "composition"
                    weights = composition_weights[local, arm_index]
                else:
                    target_kind = "none"
                    weights = np.zeros(13, dtype=np.float32)
                support = np.flatnonzero(weights > 1.0e-6)
                total = float(weights.sum())
                normalized = weights / total if total > 0 else weights
                output_rows.append(
                    {
                        **spec,
                        "arm": arm,
                        "target_kind": target_kind,
                        "target_support": "+".join(ATOMIC_NAMES[int(i)] for i in support),
                        "target_support_count": int(len(support)),
                        "predicted_top1": ATOMIC_NAMES[int(order[0])],
                        "predicted_top1_cos": float(similarities[order[0]]),
                        "predicted_top2": ATOMIC_NAMES[int(order[1])],
                        "predicted_top2_cos": float(similarities[order[1]]),
                        "top1_in_target_support": bool(int(order[0]) in support),
                        "weighted_target_cos": float(normalized @ similarities) if total > 0 else float("nan"),
                        "stay_cos": float(similarities[ATOMIC_NAMES.index("stay")]),
                    }
                )
        print(f"encoded {min(start + args.batch_size, len(selected))}/{len(selected)}", flush=True)

    _write_csv(args.output_dir / "per_arm_rows.csv", output_rows)
    np.savez_compressed(
        args.output_dir / "raw_q1.npz",
        directions=np.concatenate(vectors),
        zm=np.concatenate(zm_values),
        codebook=codes_np,
        dataset_indices=np.asarray([row["dataset_index"] for row in selected]),
    )

    groups: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    for row in output_rows:
        groups[(row["subtask"], row["arm"], float(row["phase"]))].append(row)
    group_rows = []
    for (subtask, arm, phase), rows in sorted(groups.items()):
        supervised = [row for row in rows if row["target_kind"] != "none"]
        counts = {name: sum(row["predicted_top1"] == name for row in rows) for name in ATOMIC_NAMES}
        order = sorted(counts, key=counts.get, reverse=True)
        group_rows.append(
            {
                "subtask": subtask,
                "arm": arm,
                "phase": phase,
                "count": len(rows),
                "top1_stay_fraction": float(np.mean([row["predicted_top1"] == "stay" for row in rows])),
                "target_support_hit_rate": float(np.mean([row["top1_in_target_support"] for row in supervised])) if supervised else float("nan"),
                "mean_weighted_target_cos": float(np.mean([row["weighted_target_cos"] for row in supervised])) if supervised else float("nan"),
                "dominant_prediction": order[0],
                "dominant_prediction_fraction": counts[order[0]] / len(rows),
                "second_prediction": order[1],
                "second_prediction_fraction": counts[order[1]] / len(rows),
            }
        )
    _write_csv(args.output_dir / "subtask_arm_phase_summary.csv", group_rows)

    supervised = [row for row in output_rows if row["target_kind"] != "none"]
    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "selection_count": len(selected),
        "global_selection_count": global_selected_count,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "segment_occurrences": len({(row["episode"], row["segment_index"]) for row in selected}),
        "per_arm_rows": len(output_rows),
        "supervised_arm_rows": len(supervised),
        "top1_stay_fraction": float(np.mean([row["predicted_top1"] == "stay" for row in output_rows])),
        "target_support_hit_rate": float(np.mean([row["top1_in_target_support"] for row in supervised])),
        "mean_weighted_target_cos": float(np.mean([row["weighted_target_cos"] for row in supervised])),
        "mean_top1_cos": float(np.mean([row["predicted_top1_cos"] for row in output_rows])),
        "group_rows": group_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    subtasks = list(dict.fromkeys(row["subtask"] for row in group_rows))
    figure, axes = plt.subplots(len(subtasks), 2, figsize=(12, 3.2 * len(subtasks)), constrained_layout=True)
    axes = np.asarray(axes).reshape(len(subtasks), 2)
    for row_index, subtask in enumerate(subtasks):
        for arm_index, arm in enumerate(ARMS):
            axis = axes[row_index, arm_index]
            rows = [row for row in group_rows if row["subtask"] == subtask and row["arm"] == arm]
            rows.sort(key=lambda row: row["phase"])
            x = [row["phase"] for row in rows]
            axis.plot(x, [row["top1_stay_fraction"] for row in rows], "o-", label="predicted stay")
            axis.plot(x, [row["target_support_hit_rate"] for row in rows], "s-", label="Top1 in target support")
            axis.set_ylim(-0.03, 1.03)
            axis.set_xticks(phases, ["10%", "50%", "90%"])
            axis.grid(alpha=0.25)
            axis.set_title(f"{arm}: {subtask}", fontsize=9)
            if row_index == 0 and arm_index == 0:
                axis.legend(frameon=False)
    figure.savefig(args.output_dir / "q1_segment_phase_summary.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    selection_path = args.output_dir / "selection.json"
    contract = [
        "# Vase segment Q1 evaluation contract",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- dataset: `{args.dataset_root}`",
        f"- norm: `{args.norm_assets_dir / args.norm_asset_id}`",
        f"- tokenizer max length: `{args.max_token_len}`",
        "- prompt: exact dataset-sidecar current Subtask; no atomic prompt mixing",
        "- observation: training-faithful three images + normalized state + Subtask",
        "- selection: every semantic-segment occurrence at 10%, 50%, 90%, rounded to 10-frame grid",
        f"- shard: `{args.shard_index}/{args.num_shards}`",
        f"- selection SHA256: `{_sha256(selection_path)}`",
        f"- norm SHA256: `{_sha256(args.norm_assets_dir / args.norm_asset_id / 'norm_stats.json')}`",
        f"- checkpoint metadata SHA256: `{_sha256(args.checkpoint / 'params' / '_METADATA')}`",
        f"- script SHA256: `{_sha256(Path(__file__))}`",
    ]
    (args.output_dir / "run_contract.md").write_text("\n".join(contract) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "group_rows"}, indent=2))


if __name__ == "__main__":
    main()
