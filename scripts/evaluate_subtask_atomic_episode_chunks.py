#!/usr/bin/env python3
"""Evaluate complete episodes with subtask prompts and paired atomic overrides."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import sys
import textwrap
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from openpi.models import model as _model
from openpi.shared import nnx_utils

from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import (
    ATOMIC_NAMES,
    atomic_collate,
    batch_to_observation,
    build_atomic_dataset,
)
from evaluate_zm_fk_trajectory_ablation import _output_transform


JOINT_INDICES = tuple(range(7)) + tuple(range(8, 15))
ARM_JOINT_INDICES = {
    "right": tuple(range(8, 15)),
    "left": tuple(range(7)),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _select_episode_chunks(dataset, episodes: list[int], horizon: int):
    """Resolve all non-overlapping, non-padded horizon rows in one dataset scan."""
    raw = dataset._raw  # noqa: SLF001
    requested = set(episodes)
    selected: dict[int, list[tuple[int, int, int, np.ndarray, dict[str, Any]]]] = {
        episode: [] for episode in episodes
    }
    base = raw.base
    if hasattr(base, "_visible_indices"):
        visible = np.asarray(base._visible_indices, dtype=np.int64)  # noqa: SLF001
    else:
        visible = np.arange(len(base), dtype=np.int64)
    episode_values = np.asarray(base._episode_index[visible], dtype=np.int64)  # noqa: SLF001
    frame_values = np.asarray(base._frame_index[visible], dtype=np.int64)  # noqa: SLF001
    candidates = np.flatnonzero(
        np.isin(episode_values, np.asarray(episodes, dtype=np.int64))
        & (frame_values % horizon == 0)
    )
    for dataset_index in candidates:
        data_index = int(visible[dataset_index])
        episode = int(episode_values[dataset_index])
        frame = int(frame_values[dataset_index])
        if episode not in requested:
            continue
        query_indices, _ = raw.base._get_query_indices(data_index, episode)  # noqa: SLF001
        action_indices = np.asarray(query_indices["action"], dtype=np.int64)
        if action_indices.shape != (horizon,) or len(np.unique(action_indices)) != horizon:
            continue
        metadata = raw.metadata(int(dataset_index))
        if np.asarray(metadata["raw_actions"]).shape[0] != horizon:
            continue
        selected[episode].append(
            (frame, int(dataset_index), data_index, action_indices, metadata)
        )
    for episode in episodes:
        selected[episode].sort(key=lambda row: row[0])
        if not selected[episode]:
            raise RuntimeError(f"episode {episode} has no complete 50-frame chunks")
        observed = [row[0] for row in selected[episode]]
        expected = list(range(0, observed[-1] + horizon, horizon))
        if observed != expected:
            raise ValueError(
                f"episode {episode} complete 50-frame prefix has an internal gap: "
                f"expected {expected}, got {observed}"
            )
    return selected


def _slice_observation(observation: _model.Observation, start: int, end: int):
    return jax.tree.map(lambda value: value[start:end], observation)


def _sample_batched(
    sample_actions,
    observation: _model.Observation,
    noise: jax.Array,
    *,
    seed: int,
    num_steps: int,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    size = int(noise.shape[0])
    for start in range(0, size, batch_size):
        end = min(start + batch_size, size)
        outputs.append(
            np.asarray(
                jax.device_get(
                    sample_actions(
                        jax.random.fold_in(jax.random.key(seed), start),
                        _slice_observation(observation, start, end),
                        num_steps=num_steps,
                        noise=noise[start:end],
                    )
                )
            )[..., :16]
        )
    return np.concatenate(outputs, axis=0)


def _joint_metrics(gt: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - gt
    joint_error = error[..., JOINT_INDICES]
    endpoint_error = joint_error[..., -1, :] if joint_error.ndim == 3 else joint_error[-1]
    return {
        "joint_14d_rmse_deg": float(
            np.degrees(np.sqrt(np.mean(np.square(joint_error))))
        ),
        "joint_14d_endpoint_rmse_deg": float(
            np.degrees(np.sqrt(np.mean(np.square(endpoint_error))))
        ),
        "left_joint_rmse_deg": float(
            np.degrees(np.sqrt(np.mean(np.square(error[..., :7]))))
        ),
        "right_joint_rmse_deg": float(
            np.degrees(np.sqrt(np.mean(np.square(error[..., 8:15]))))
        ),
        "left_gripper_rmse_native": float(
            np.sqrt(np.mean(np.square(error[..., 7])))
        ),
        "right_gripper_rmse_native": float(
            np.sqrt(np.mean(np.square(error[..., 15])))
        ),
    }


def _supervised_arm_rmse(
    gt: np.ndarray,
    prediction: np.ndarray,
    supervision_mask: np.ndarray,
) -> float:
    errors = []
    for arm_index, arm in enumerate(("right", "left")):
        if bool(supervision_mask[arm_index]):
            errors.append(prediction[..., ARM_JOINT_INDICES[arm]] - gt[..., ARM_JOINT_INDICES[arm]])
    if not errors:
        raise ValueError("supervised-arm metric requires at least one strict arm")
    return float(np.degrees(np.sqrt(np.mean(np.square(np.concatenate(errors, axis=-1))))))


def _labels(metadata: dict[str, Any]) -> dict[str, list[str]]:
    weights = np.asarray(metadata["atomic_weights"])
    result = {}
    for arm_index, arm in enumerate(("right", "left")):
        result[arm] = [
            ATOMIC_NAMES[index]
            for index in np.flatnonzero(weights[arm_index] > 0.0)
        ]
    return result


def _plot_episode(
    *,
    episode: int,
    global_prompt: str,
    chunks: list[dict[str, Any]],
    gt: np.ndarray,
    subtask: np.ndarray,
    atomic: np.ndarray,
    boundary_states: np.ndarray,
    output: Path,
) -> None:
    figure, axes = plt.subplots(
        4, 4, figsize=(21, 14), sharex=True, constrained_layout=True
    )
    steps = np.arange(gt.shape[0])
    for dimension, axis in enumerate(axes.flat):
        arm = "L" if dimension < 8 else "R"
        local = dimension if dimension < 8 else dimension - 8
        gripper = local == 7
        gt_values = gt[:, dimension] if gripper else np.degrees(gt[:, dimension])
        sub_values = (
            subtask[:, dimension] if gripper else np.degrees(subtask[:, dimension])
        )
        atomic_values = (
            atomic[:, dimension] if gripper else np.degrees(atomic[:, dimension])
        )
        axis.plot(steps, gt_values, color="#111827", linewidth=2.0, label="GT")
        axis.plot(
            steps,
            sub_values,
            color="#2563eb",
            linewidth=1.35,
            label="Subtask infer",
        )
        axis.plot(
            steps,
            atomic_values,
            color="#f97316",
            linewidth=1.35,
            label="Atomic infer (strict chunks only)",
        )
        for chunk_index, chunk in enumerate(chunks):
            boundary = int(chunk["start_frame"])
            axis.axvline(boundary, color="#9ca3af", linewidth=0.45, alpha=0.45)
            state_value = boundary_states[chunk_index, dimension]
            if not gripper:
                state_value = np.degrees(state_value)
            axis.scatter(
                [boundary], [state_value], color="#16a34a", s=18, zorder=5,
                label="Measured state at chunk start" if chunk_index == 0 else None,
            )
            if chunk["atomic_supervised"]:
                axis.axvspan(
                    boundary,
                    int(chunk["end_frame_exclusive"]) - 1,
                    color="#fed7aa",
                    alpha=0.065,
                )
        axis.set_title(f"{arm} gripper" if gripper else f"{arm} J{local + 1}")
        axis.set_ylabel("native" if gripper else "deg")
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        if dimension >= 12:
            axis.set_xlabel("episode frame")
        if dimension == 0:
            axis.legend(frameon=False, fontsize=8)
    atomic_count = sum(bool(row["atomic_supervised"]) for row in chunks)
    figure.suptitle(
        f"Training episode {episode} · {len(chunks)} × 50-frame chunks · "
        f"{atomic_count} strict-atomic paired chunks\n"
        + textwrap.fill(global_prompt, width=120),
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_atomic_comparison(
    episode: int,
    chunks: list[dict[str, Any]],
    output: Path,
) -> None:
    paired = [row for row in chunks if row["atomic_supervised"]]
    labels = [str(row["start_frame"]) for row in paired]
    subtask = [row["subtask_supervised_arm_rmse_deg"] for row in paired]
    atomic = [row["atomic_supervised_arm_rmse_deg"] for row in paired]
    x = np.arange(len(paired))
    figure, axis = plt.subplots(figsize=(max(11, len(paired) * 0.52), 6), constrained_layout=True)
    axis.bar(x - 0.19, subtask, width=0.38, color="#2563eb", label="Subtask")
    axis.bar(x + 0.19, atomic, width=0.38, color="#f97316", label="Atomic")
    axis.set_xticks(x, labels, rotation=55, ha="right")
    axis.set_xlabel("50-frame chunk start")
    axis.set_ylabel("strict supervised-arm joint RMSE (deg)")
    axis.set_title(f"Episode {episode} · same-state/same-noise prompt comparison")
    axis.grid(axis="y", alpha=0.28)
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-label", default="weighted-cos latest")
    parser.add_argument("--source-revision", default="unknown")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, action="append", required=True)
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument(
        "--atomic-composition-sidecar", default="fk_horizon_3hz_gate_top5_stay_v2"
    )
    parser.add_argument(
        "--pad-subtask-horizon",
        action="store_true",
        help=(
            "Hold the final in-segment action to 50 steps and use only the "
            "active subtask, matching padded-subtask training runs."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    norm_stats = args.norm_assets_dir / args.norm_asset_id / "norm_stats.json"
    if not norm_stats.is_file():
        raise FileNotFoundError(norm_stats)
    if not (args.checkpoint / "params").is_dir():
        raise FileNotFoundError(args.checkpoint / "params")

    config = AtomicPi05Config(
        max_token_len=200,
        fast_action_ce_loss_weight=0.0,
        coefficient_target_kind="joint_delta",
        coefficient_target_dim=14,
        enable_layerwise_atomic_flow=True,
    )
    dataset = build_atomic_dataset(
        (args.dataset_root,),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        include_fast=False,
        atomic_composition_sidecar=args.atomic_composition_sidecar,
        pad_subtask_horizon=args.pad_subtask_horizon,
    )
    selected_by_episode = _select_episode_chunks(
        dataset, args.episode, config.action_horizon
    )
    flat_selected = [row for episode in args.episode for row in selected_by_episode[episode]]
    rows = [dataset[row[1]] for row in flat_selected]
    batch = atomic_collate(rows)
    observation_np, _ = batch_to_observation(batch)
    observation = jax.tree.map(jnp.asarray, observation_np)

    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    subtask_observation = model._with_prompt(  # noqa: SLF001
        observation,
        jnp.asarray(batch["subtask_prompt_tokens"]),
        jnp.asarray(batch["subtask_prompt_mask"]),
    )
    atomic_observation = model._with_prompt(  # noqa: SLF001
        observation,
        jnp.asarray(batch["atomic_prompt_tokens"]),
        jnp.asarray(batch["atomic_prompt_mask"]),
    )
    noise = jax.random.normal(
        jax.random.key(args.seed),
        (len(rows), config.action_horizon, config.action_dim),
    )
    noise_np = np.asarray(jax.device_get(noise))
    sample_actions = nnx_utils.module_jit(model.sample_actions)
    subtask_normalized = _sample_batched(
        sample_actions,
        subtask_observation,
        noise,
        seed=args.seed,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
    )
    atomic_normalized = _sample_batched(
        sample_actions,
        atomic_observation,
        noise,
        seed=args.seed,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
    )

    decode = _output_transform(
        args.dataset_root,
        config,
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
    )
    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_label": args.checkpoint_label,
        "source_revision": args.source_revision,
        "dataset_root": str(args.dataset_root),
        "norm_asset_id": args.norm_asset_id,
        "norm_stats_sha256": _sha256(norm_stats),
        "atomic_composition_sidecar": args.atomic_composition_sidecar,
        "max_token_len": config.max_token_len,
        "action_horizon": config.action_horizon,
        "action_dim": config.active_action_dim,
        "flow_ode_steps": args.num_steps,
        "seed": args.seed,
        "noise_sha256": _array_sha256(noise_np),
        "pairing": "same images, normalized state, and explicit initial flow noise; only prompt tokens differ",
        "episodes": {},
    }
    selection: dict[str, Any] = {"episodes": {}}
    trajectories: dict[str, np.ndarray] = {}
    csv_rows = []
    cursor = 0
    for episode in args.episode:
        selected = selected_by_episode[episode]
        decoded_subtask = []
        decoded_atomic = []
        gt_chunks = []
        boundary_states = []
        chunk_rows = []
        for local_index, (frame, dataset_index, data_index, action_indices, metadata) in enumerate(selected):
            index = cursor + local_index
            sub_prediction = np.asarray(
                decode(
                    np.asarray(batch["state"][index]),
                    np.asarray(metadata["raw_state"]),
                    subtask_normalized[index],
                )["actions"]
            )[..., :16]
            atomic_prediction = np.asarray(
                decode(
                    np.asarray(batch["state"][index]),
                    np.asarray(metadata["raw_state"]),
                    atomic_normalized[index],
                )["actions"]
            )[..., :16]
            gt = np.asarray(metadata["raw_actions"])[..., :16]
            supervision_mask = np.asarray(
                metadata["atomic_supervision_mask"], dtype=np.bool_
            )
            atomic_supervised = bool(np.any(supervision_mask))
            labels = _labels(metadata)
            row: dict[str, Any] = {
                "chunk": local_index,
                "dataset_index": dataset_index,
                "data_index": data_index,
                "start_frame": frame,
                "end_frame_exclusive": frame + config.action_horizon,
                "subtask_prompt": str(metadata["subtask_prompt"]),
                "atomic_supervised": atomic_supervised,
                "atomic_prompt": str(metadata["atomic_prompt"]) if atomic_supervised else None,
                "right_atomic_labels": labels["right"],
                "left_atomic_labels": labels["left"],
                "subtask_metrics": _joint_metrics(gt, sub_prediction),
                "atomic_metrics": _joint_metrics(gt, atomic_prediction) if atomic_supervised else None,
                "subtask_supervised_arm_rmse_deg": (
                    _supervised_arm_rmse(gt, sub_prediction, supervision_mask)
                    if atomic_supervised
                    else None
                ),
                "atomic_supervised_arm_rmse_deg": (
                    _supervised_arm_rmse(gt, atomic_prediction, supervision_mask)
                    if atomic_supervised
                    else None
                ),
            }
            chunk_rows.append(row)
            csv_rows.append({
                "episode": episode,
                "chunk": local_index,
                "start_frame": frame,
                "end_frame_exclusive": frame + config.action_horizon,
                "subtask_prompt": row["subtask_prompt"],
                "atomic_supervised": atomic_supervised,
                "atomic_prompt": row["atomic_prompt"] or "",
                "right_atomic_labels": ",".join(labels["right"]),
                "left_atomic_labels": ",".join(labels["left"]),
                "subtask_joint_14d_rmse_deg": row["subtask_metrics"]["joint_14d_rmse_deg"],
                "atomic_joint_14d_rmse_deg": (
                    "" if row["atomic_metrics"] is None else row["atomic_metrics"]["joint_14d_rmse_deg"]
                ),
                "subtask_supervised_arm_rmse_deg": row["subtask_supervised_arm_rmse_deg"] or "",
                "atomic_supervised_arm_rmse_deg": row["atomic_supervised_arm_rmse_deg"] or "",
            })
            gt_chunks.append(gt)
            boundary_states.append(np.asarray(metadata["raw_state"])[..., :16])
            decoded_subtask.append(sub_prediction)
            decoded_atomic.append(atomic_prediction)

        gt_array = np.stack(gt_chunks)
        subtask_array = np.stack(decoded_subtask)
        atomic_array = np.stack(decoded_atomic)
        boundary_state_array = np.stack(boundary_states)
        atomic_mask = np.asarray(
            [row["atomic_supervised"] for row in chunk_rows], dtype=np.bool_
        )
        paired_subtask = np.asarray(
            [row["subtask_supervised_arm_rmse_deg"] for row in chunk_rows if row["atomic_supervised"]]
        )
        paired_atomic = np.asarray(
            [row["atomic_supervised_arm_rmse_deg"] for row in chunk_rows if row["atomic_supervised"]]
        )
        global_prompt = str(selected[0][4]["global_prompt"])
        joint_plot = args.output_dir / f"episode_{episode:06d}_16d_subtask_atomic.png"
        atomic_plot = args.output_dir / f"episode_{episode:06d}_atomic_chunk_comparison.png"
        atomic_episode = np.full_like(atomic_array.reshape(-1, 16), np.nan)
        for chunk_index in np.flatnonzero(atomic_mask):
            start = int(chunk_index) * config.action_horizon
            atomic_episode[start : start + config.action_horizon] = atomic_array[chunk_index]
        _plot_episode(
            episode=episode,
            global_prompt=global_prompt,
            chunks=chunk_rows,
            gt=gt_array.reshape(-1, 16),
            subtask=subtask_array.reshape(-1, 16),
            atomic=atomic_episode,
            boundary_states=boundary_state_array,
            output=joint_plot,
        )

        arm_dims = np.asarray(JOINT_INDICES, dtype=np.int64)
        def _boundary_stats(values: np.ndarray) -> dict[str, float | None]:
            if values.size == 0:
                return {"mean_rad": None, "median_rad": None, "p95_rad": None, "max_rad": None}
            norms = np.linalg.norm(values[..., arm_dims], axis=-1)
            return {
                "mean_rad": float(np.mean(norms)),
                "median_rad": float(np.median(norms)),
                "p95_rad": float(np.percentile(norms, 95)),
                "max_rad": float(np.max(norms)),
            }
        subtask_start_state = subtask_array[:, 0, :] - boundary_state_array
        gt_start_state = gt_array[:, 0, :] - boundary_state_array
        subtask_chunk_seam = subtask_array[1:, 0, :] - subtask_array[:-1, -1, :]
        gt_chunk_seam = gt_array[1:, 0, :] - gt_array[:-1, -1, :]
        _plot_atomic_comparison(episode, chunk_rows, atomic_plot)
        report["episodes"][str(episode)] = {
            "global_prompt": global_prompt,
            "chunk_count": len(chunk_rows),
            "covered_frames": len(chunk_rows) * config.action_horizon,
            "atomic_supervised_chunk_count": int(atomic_mask.sum()),
            "subtask_metrics": _joint_metrics(gt_array, subtask_array),
            "atomic_chunks_subtask_supervised_arm_rmse_deg": float(paired_subtask.mean()),
            "atomic_chunks_atomic_supervised_arm_rmse_deg": float(paired_atomic.mean()),
            "atomic_prompt_win_fraction": float(np.mean(paired_atomic < paired_subtask)),
            "boundary_metrics": {
                "subtask_action0_minus_state": _boundary_stats(subtask_start_state),
                "gt_action0_minus_state": _boundary_stats(gt_start_state),
                "subtask_new_chunk_minus_previous_end": _boundary_stats(subtask_chunk_seam),
                "gt_new_chunk_minus_previous_end": _boundary_stats(gt_chunk_seam),
            },
            "joint_plot": str(joint_plot),
            "atomic_comparison_plot": str(atomic_plot),
            "chunks": chunk_rows,
        }
        selection["episodes"][str(episode)] = [
            {
                key: row[key]
                for key in (
                    "chunk",
                    "dataset_index",
                    "data_index",
                    "start_frame",
                    "end_frame_exclusive",
                    "subtask_prompt",
                    "atomic_supervised",
                    "atomic_prompt",
                    "right_atomic_labels",
                    "left_atomic_labels",
                )
            }
            for row in chunk_rows
        ]
        trajectories[f"episode_{episode}_gt"] = gt_array
        trajectories[f"episode_{episode}_subtask"] = subtask_array
        trajectories[f"episode_{episode}_atomic"] = atomic_array
        trajectories[f"episode_{episode}_atomic_mask"] = atomic_mask
        trajectories[f"episode_{episode}_boundary_states"] = boundary_state_array
        cursor += len(selected)

    selection_path = args.output_dir / "selection.json"
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report["selection"] = str(selection_path)
    report["selection_sha256"] = _sha256(selection_path)
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "chunks.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    np.savez_compressed(args.output_dir / "trajectories.npz", **trajectories)
    command = " ".join(shlex.quote(value) for value in sys.argv)
    (args.output_dir / "run_contract.md").write_text(
        "\n".join(
            [
                "# Complete-episode subtask/atomic paired evaluation",
                "",
                f"- Checkpoint: `{args.checkpoint}`",
                f"- Source revision: `{args.source_revision}`",
                f"- Dataset: `{args.dataset_root}`",
                f"- Episodes: `{', '.join(map(str, args.episode))}`",
                f"- Norm: `{args.norm_assets_dir / args.norm_asset_id}`",
                f"- norm_stats SHA-256: `{report['norm_stats_sha256']}`",
                f"- Atomic sidecar: `{args.atomic_composition_sidecar}`",
                "- Prompt/token length: training sidecar subtask or strict atomic, 200 tokens",
                "- Windowing: exact non-overlapping 50-frame chunks with no episode-edge padding",
                f"- Subtask-boundary padding: `{args.pad_subtask_horizon}`",
                f"- Flow ODE steps: {args.num_steps}",
                f"- Seed: {args.seed}",
                f"- Explicit flow-noise SHA-256: `{report['noise_sha256']}`",
                "- Pairing: same images, normalized state, and flow noise; only prompt tokens differ",
                "- Output reconstruction: v3 unnormalize delta, restore exact raw state, then add delta",
                "- TCP/FK: not used; metrics and plots are decoded 16-D joint/gripper targets",
                f"- Selection: `{selection_path}`",
                f"- Selection SHA-256: `{report['selection_sha256']}`",
                f"- Command: `{command}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "checkpoint": report["checkpoint"],
        "episodes": {
            episode: {
                key: value
                for key, value in payload.items()
                if key not in {"chunks", "global_prompt"}
            }
            for episode, payload in report["episodes"].items()
        },
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
