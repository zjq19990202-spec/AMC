#!/usr/bin/env python3
"""Paired native-SUBtask steering evaluation for a stock PI0.5 checkpoint.

For every selected fruit observation, all prompt variants share the exact image,
normalized state, flow noise, sampler, normalization, and output decoder.  Only
the native SUBtask text is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config

from atomic_latent_vla.pi05.training_data import (
    _paligemma_tokenizer,
    atomic_collate,
    batch_to_observation,
    build_atomic_dataset,
)
from evaluate_fruit_target_switch import _find_row
from evaluate_global_episode_chunks import _endpoint_tcp_from_actions, _output_transform


@nnx.jit
def _sample(model, observation, noise):
    return model.sample_actions(jax.random.key(0), observation, num_steps=10, noise=noise)


def _repeat_observation(observation: _model.Observation, count: int, tokens, masks):
    repeated = jax.tree.map(lambda value: jnp.repeat(value, count, axis=0), observation)
    return repeated.replace(tokenized_prompt=tokens, tokenized_prompt_mask=masks)


def _image_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        if image.min() >= -1.01 and image.max() <= 1.01:
            image = (image + 1.0) * 127.5
        elif image.max() <= 1.01:
            image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--episode", type=int)
    parser.add_argument("--frames", type=int, nargs="+")
    parser.add_argument("--active-arm", choices=("left", "right"), default="right")
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--selection-dataset-name", default="fruit")
    parser.add_argument("--selection-start", type=int, default=0)
    parser.add_argument("--selection-stop", type=int)
    parser.add_argument("--prompt", action="append")
    parser.add_argument(
        "--destination-target",
        action="append",
        help="Replace only the native destination box in each row; repeat for paired targets.",
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--prompt-batch-size",
        type=int,
        default=0,
        help="Prompt microbatch size; 0 evaluates all matched prompts together.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.prompt) == bool(args.destination_target):
        parser.error("provide either --prompt or --destination-target")
    if args.selection_manifest is None and (args.episode is None or not args.frames):
        parser.error("provide --selection-manifest or both --episode and --frames")
    if args.selection_manifest is not None and (args.episode is not None or args.frames):
        parser.error("--selection-manifest is mutually exclusive with --episode/--frames")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.selection_manifest is not None:
        selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        frame_specs = selection["datasets"][args.selection_dataset_name][
            args.selection_start : args.selection_stop
        ]
    else:
        frame_specs = [
            {"episode": args.episode, "frame": frame, "active_arm": args.active_arm}
            for frame in args.frames
        ]

    config = Pi0Config(pi05=True, action_dim=32, action_horizon=50, max_token_len=200)
    dataset = build_atomic_dataset(
        (args.dataset_root,),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        action_horizon=50,
        max_token_len=200,
        include_fast=False,
        pad_subtask_horizon=True,
    )
    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    decoder = _output_transform(
        args.dataset_root,
        config,
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
    )
    tokenizer = _paligemma_tokenizer(200)
    raw = dataset._raw  # noqa: SLF001
    rows = []
    source_images = []

    for spec in frame_specs:
        episode = int(spec["episode"])
        frame = int(spec["frame"])
        active_arm = str(spec.get("active_arm", "right"))
        dataset_index, data_index, _, metadata = _find_row(dataset, episode, frame)
        row = dataset[dataset_index]
        batch = atomic_collate([row])
        obs_np, gt_normalized_batch = batch_to_observation(batch)
        observation = jax.tree.map(jnp.asarray, obs_np)
        frame_prompts = args.prompt
        if args.destination_target:
            native = metadata["subtask_prompt"]
            matches = [target for target in args.destination_target if target in native]
            if len(matches) != 1:
                raise ValueError(f"expected one native destination in {native!r}; got {matches}")
            frame_prompts = [native.replace(matches[0], target) for target in args.destination_target]
        token_rows, mask_rows = zip(
            *(tokenizer.tokenize(prompt, np.asarray(row["state"])) for prompt in frame_prompts),
            strict=True,
        )
        one_noise = jax.random.normal(
            jax.random.fold_in(jax.random.key(args.seed), episode * 10000 + frame),
            (1, config.action_horizon, config.action_dim),
        )
        prompt_batch_size = args.prompt_batch_size or len(frame_prompts)
        if prompt_batch_size < 1:
            raise ValueError("--prompt-batch-size must be positive or zero")
        normalized_chunks = []
        for start in range(0, len(frame_prompts), prompt_batch_size):
            stop = min(start + prompt_batch_size, len(frame_prompts))
            count = stop - start
            paired_observation = _repeat_observation(
                observation,
                count,
                jnp.asarray(np.stack(token_rows[start:stop])),
                jnp.asarray(np.stack(mask_rows[start:stop])),
            )
            paired_noise = jnp.repeat(one_noise, count, axis=0)
            normalized_chunks.append(
                np.asarray(jax.device_get(_sample(model, paired_observation, paired_noise)))
            )
        normalized = np.concatenate(normalized_chunks, axis=0)
        print(
            f"evaluated episode={episode} frame={frame} "
            f"({len(rows) + 1}/{len(frame_specs)})",
            flush=True,
        )

        decoded = []
        endpoints = {"left": [], "right": []}
        displacements = {"left": [], "right": []}
        tcp_trajectories = {"left": [], "right": []}
        for prompt_index in range(len(frame_prompts)):
            actions = np.asarray(
                decoder(
                    np.asarray(batch["state"][0]),
                    np.asarray(metadata["raw_state"]),
                    normalized[prompt_index, :, :16],
                )["actions"]
            )
            decoded.append(actions)
            for arm, pose_offset in (("left", 0), ("right", 24)):
                xyz = _endpoint_tcp_from_actions(raw, data_index, actions, arm)
                current = np.asarray(raw.tcp_pose[data_index, pose_offset : pose_offset + 3])
                tcp_trajectories[arm].append(xyz)
                endpoints[arm].append(xyz[-1])
                displacements[arm].append(xyz[-1] - current)

        decoded_np = np.stack(decoded)
        gt_normalized = np.asarray(gt_normalized_batch[0])
        gt_decoded = np.asarray(
            decoder(
                np.asarray(batch["state"][0]),
                np.asarray(metadata["raw_state"]),
                gt_normalized[:, :16],
            )["actions"]
        )
        gt_tcp_trajectories = {
            arm: _endpoint_tcp_from_actions(raw, data_index, gt_decoded, arm)
            for arm in ("left", "right")
        }
        gt_endpoints = {arm: trajectory[-1] for arm, trajectory in gt_tcp_trajectories.items()}
        prompt_metrics = []
        for prompt_index, prompt in enumerate(frame_prompts):
            metric = {
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "is_native": prompt == metadata["subtask_prompt"],
                    "normalized_action_rmse_to_gt": float(
                        np.sqrt(np.mean((normalized[prompt_index, :, :16] - gt_normalized[:, :16]) ** 2))
                    ),
                    "decoded_joint_rmse_to_gt_rad": float(
                        np.sqrt(np.mean((decoded_np[prompt_index, :, :15] - gt_decoded[:, :15]) ** 2))
                    ),
                    "left_endpoint_error_to_gt_m": float(
                        np.linalg.norm(np.asarray(endpoints["left"])[prompt_index] - gt_endpoints["left"])
                    ),
                    "right_endpoint_error_to_gt_m": float(
                        np.linalg.norm(np.asarray(endpoints["right"])[prompt_index] - gt_endpoints["right"])
                    ),
                }
            metric["active_endpoint_error_to_gt_m"] = metric[f"{active_arm}_endpoint_error_to_gt_m"]
            prompt_metrics.append(metric)
        pair_metrics = []
        for i, j in combinations(range(len(frame_prompts)), 2):
            pair_metrics.append(
                {
                    "prompt_i": i,
                    "prompt_j": j,
                    "normalized_action_rmse": float(np.sqrt(np.mean((normalized[i] - normalized[j]) ** 2))),
                    "decoded_joint_rmse_rad": float(np.sqrt(np.mean((decoded_np[i, :, :15] - decoded_np[j, :, :15]) ** 2))),
                    "left_endpoint_separation_m": float(np.linalg.norm(np.asarray(endpoints["left"])[i] - np.asarray(endpoints["left"])[j])),
                    "right_endpoint_separation_m": float(np.linalg.norm(np.asarray(endpoints["right"])[i] - np.asarray(endpoints["right"])[j])),
                }
            )
        rows.append(
            {
                "episode": episode,
                "frame": frame,
                "active_arm": active_arm,
                "native_subtask": metadata["subtask_prompt"],
                "prompts": frame_prompts,
                "normalized_actions": normalized[:, :, :16].tolist(),
                "decoded_actions": decoded_np.tolist(),
                "left_endpoint_xyz_m": np.asarray(endpoints["left"]).tolist(),
                "right_endpoint_xyz_m": np.asarray(endpoints["right"]).tolist(),
                "left_displacement_m": np.asarray(displacements["left"]).tolist(),
                "right_displacement_m": np.asarray(displacements["right"]).tolist(),
                "left_tcp_trajectories_m": np.asarray(tcp_trajectories["left"]).tolist(),
                "right_tcp_trajectories_m": np.asarray(tcp_trajectories["right"]).tolist(),
                "gt_left_tcp_trajectory_m": np.asarray(gt_tcp_trajectories["left"]).tolist(),
                "gt_right_tcp_trajectory_m": np.asarray(gt_tcp_trajectories["right"]).tolist(),
                "raw_state": np.asarray(metadata["raw_state"]).tolist(),
                "gt_decoded_actions": gt_decoded.tolist(),
                "gt_left_endpoint_xyz_m": np.asarray(gt_endpoints["left"]).tolist(),
                "gt_right_endpoint_xyz_m": np.asarray(gt_endpoints["right"]).tolist(),
                "prompt_metrics": prompt_metrics,
                "pair_metrics": pair_metrics,
            }
        )
        source_images.append({name: _image_uint8(value[0]) for name, value in obs_np.images.items()})

    pair_rows = [metric for row in rows for metric in row["pair_metrics"]]
    native_metrics = [metric for row in rows for metric in row["prompt_metrics"] if metric["is_native"]]
    non_native_metrics = [metric for row in rows for metric in row["prompt_metrics"] if not metric["is_native"]]
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "selection_manifest": str(args.selection_manifest.resolve()) if args.selection_manifest else None,
        "selection_manifest_sha256": _sha256(args.selection_manifest) if args.selection_manifest else None,
        "frames": [{"episode": int(x["episode"]), "frame": int(x["frame"]), "active_arm": x.get("active_arm", "right")} for x in frame_specs],
        "prompts": args.prompt,
        "destination_targets": args.destination_target,
        "seed": args.seed,
        "prompt_batch_size": args.prompt_batch_size or len(args.prompt or args.destination_target),
        "pairwise_mean_normalized_action_rmse": float(np.mean([x["normalized_action_rmse"] for x in pair_rows])),
        "pairwise_mean_decoded_joint_rmse_rad": float(np.mean([x["decoded_joint_rmse_rad"] for x in pair_rows])),
        "pairwise_mean_left_endpoint_separation_m": float(np.mean([x["left_endpoint_separation_m"] for x in pair_rows])),
        "pairwise_mean_right_endpoint_separation_m": float(np.mean([x["right_endpoint_separation_m"] for x in pair_rows])),
        "native_mean_decoded_joint_rmse_to_gt_rad": float(np.mean([x["decoded_joint_rmse_to_gt_rad"] for x in native_metrics])),
        "non_native_mean_decoded_joint_rmse_to_gt_rad": float(np.mean([x["decoded_joint_rmse_to_gt_rad"] for x in non_native_metrics])),
        "native_mean_right_endpoint_error_to_gt_m": float(np.mean([x["right_endpoint_error_to_gt_m"] for x in native_metrics])),
        "non_native_mean_right_endpoint_error_to_gt_m": float(np.mean([x["right_endpoint_error_to_gt_m"] for x in non_native_metrics])),
        "native_best_right_endpoint_frames": int(sum(
            min(row["prompt_metrics"], key=lambda x: x["right_endpoint_error_to_gt_m"])["is_native"]
            for row in rows
        )),
        "native_mean_active_endpoint_error_to_gt_m": float(np.mean([x["active_endpoint_error_to_gt_m"] for x in native_metrics])),
        "non_native_mean_active_endpoint_error_to_gt_m": float(np.mean([x["active_endpoint_error_to_gt_m"] for x in non_native_metrics])),
        "native_best_active_endpoint_frames": int(sum(
            min(row["prompt_metrics"], key=lambda x: x["active_endpoint_error_to_gt_m"])["is_native"]
            for row in rows
        )),
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(len(frame_specs), 2, figsize=(14, 3.5 * len(frame_specs)), squeeze=False)
    palette = plt.get_cmap("tab10")
    for row_index, row in enumerate(rows):
        for col, arm in enumerate(("left", "right")):
            ax = axes[row_index, col]
            values = np.asarray(row[f"{arm}_displacement_m"]) * 1000.0
            for prompt_index, prompt in enumerate(row["prompts"]):
                ax.scatter(values[prompt_index, 0], values[prompt_index, 1], color=palette(prompt_index % 10), s=55, label=prompt if row_index == 0 and col == 0 else None)
                ax.annotate(str(prompt_index + 1), values[prompt_index, :2], fontsize=8)
            ax.axhline(0, color="0.8", linewidth=0.8)
            ax.axvline(0, color="0.8", linewidth=0.8)
            ax.set_title(f"episode {row['episode']} frame {row['frame']} | {arm} TCP")
            ax.set_xlabel("endpoint dx (mm)")
            ax.set_ylabel("endpoint dy (mm)")
            ax.set_aspect("equal", adjustable="datalim")
    fig.legend(loc="lower center", ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(args.output_dir / "paired_subtask_tcp_xy.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(16, 4.2 * len(rows)))
    for row_index, row in enumerate(rows):
        ax = fig.add_subplot(len(rows), 2, 2 * row_index + 1, projection="3d")
        arm = row["active_arm"]
        predictions = np.asarray(row[f"{arm}_tcp_trajectories_m"]) * 1000.0
        gt_track = np.asarray(row[f"gt_{arm}_tcp_trajectory_m"]) * 1000.0
        all_tracks = [gt_track]
        ax.plot(*gt_track.T, color="black", linewidth=3.0, linestyle="--", label="GT")
        for prompt_index, prompt in enumerate(row["prompts"]):
            track = predictions[prompt_index]
            all_tracks.append(track)
            ax.plot(*track.T, color=palette(prompt_index % 10), linewidth=1.8, label=f"{prompt_index + 1}: {prompt}")
            ax.scatter(*track[-1], color=palette(prompt_index % 10), s=24)
        points = np.concatenate(all_tracks)
        lo, hi = points.min(axis=0), points.max(axis=0)
        center = (lo + hi) / 2
        radius = max(float(np.max(hi - lo)) / 2, 8.0)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(f"ep {row['episode']} f{row['frame']} | {arm} | native: {row['native_subtask']}")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_zlabel("z (mm)")

        ax2 = fig.add_subplot(len(rows), 2, 2 * row_index + 2)
        steps = np.arange(1, gt_track.shape[0] + 1)
        for prompt_index, prompt in enumerate(row["prompts"]):
            error = np.linalg.norm(predictions[prompt_index] - gt_track, axis=1)
            ax2.plot(steps, error, color=palette(prompt_index % 10), linewidth=1.8, label=f"{prompt_index + 1}")
        ax2.set_title("TCP distance to GT over the 50-step horizon")
        ax2.set_xlabel("horizon step")
        ax2.set_ylabel("error (mm)")
        ax2.grid(alpha=0.25)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(args.output_dir / "paired_subtask_active_tcp_trajectories_3d.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(source_images), 3, figsize=(12, 3.2 * len(source_images)), squeeze=False)
    for row_index, images in enumerate(source_images):
        for col, (name, image) in enumerate(images.items()):
            if col >= 3:
                break
            axes[row_index, col].imshow(image)
            axes[row_index, col].set_title(f"ep {rows[row_index]['episode']} frame {rows[row_index]['frame']} | {name}")
            axes[row_index, col].axis("off")
    fig.tight_layout()
    fig.savefig(args.output_dir / "source_observations.png", dpi=160)
    plt.close(fig)

    norm_path = args.norm_assets_dir / args.norm_asset_id / "norm_stats.json"
    (args.output_dir / "run_contract.md").write_text(
        "# Evaluation contract\n\n"
        f"- checkpoint: `{args.checkpoint.resolve()}`\n"
        f"- dataset: `{args.dataset_root.resolve()}`\n"
        f"- norm: `{norm_path.resolve()}`; sha256 `{_sha256(norm_path)}`\n"
        "- model: stock PI0.5, action_dim=32, horizon=50, max_token_len=200\n"
        "- prompt: native fruit SUBtask variants from the training sidecar; no global prompt\n"
        "- pairing: identical image, normalized state, flow noise, sampler, and decoder within each frame\n"
        "- decoding: normalized joint_delta is inverse-transformed before adding to raw state; TCP offset=0.20 m\n"
        f"- seed: {args.seed}; sampler steps: 10; prompt microbatch: "
        f"{args.prompt_batch_size or len(args.prompt or args.destination_target)}\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
