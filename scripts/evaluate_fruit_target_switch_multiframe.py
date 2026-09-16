#!/usr/bin/env python3
"""Evaluate target-name steering at several fixed frames with one model load."""

from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from openpi.models import model as _model

from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import (
    _paligemma_tokenizer,
    atomic_collate,
    batch_to_observation,
    build_atomic_dataset,
)
from evaluate_fruit_target_switch import (
    COLORS,
    PROMPTS,
    _cosine,
    _find_row,
    _sample_variants_layerwise,
    _sample_variants_no_zm,
)
from evaluate_global_episode_chunks import (
    _endpoint_tcp_from_actions,
    _output_transform,
    _tcp200_trajectory,
)


FRUIT_TARGETS = (
    "green bitter melon",
    "orange pumpkin",
    "yellow banana",
    "yellow pear",
    "green radish",
    "red bell pepper",
    "red chili pepper",
    "red apple",
    "bitter melon",
    "mangosteen",
    "pumpkin",
    "banana",
    "potato",
    "tomato",
    "radish",
    "apple",
    "pear",
    "carrot",
    "orange",
)
FRUIT_COLORS = {
    **COLORS,
    "green bitter melon": "#15803d",
    "yellow pear": "#eab308",
    "orange": "#f97316",
    "red apple": "#dc2626",
    "red bell pepper": "#be123c",
    "red chili pepper": "#7f1d1d",
}


def _native_template_prompts(
    original_prompt: str,
    counterfactual_targets: list[str],
) -> tuple[dict[str, str], str]:
    matches = [name for name in FRUIT_TARGETS if name.lower() in original_prompt.lower()]
    if not matches:
        raise ValueError(
            f"expected a known fruit target in native prompt, got {matches}: "
            f"{original_prompt!r}"
        )
    # Prefer the most specific target (e.g. red apple over apple, orange pumpkin
    # over orange) so native-template replacement never corrupts the prompt.
    native_target = max(matches, key=len)
    prompts = {
        target: original_prompt.replace(native_target, target)
        for target in counterfactual_targets
    }
    # This assignment is intentionally last: the native candidate is exactly
    # the sidecar text, without a canonical rewrite.
    prompts[native_target] = original_prompt
    return prompts, native_target


def _equal_3d_axes(ax, trajectories: list[np.ndarray]) -> None:
    points = np.concatenate(trajectories, axis=0)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = (lo + hi) / 2
    radius = max(float(np.max(hi - lo)) / 2, 0.025)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int)
    parser.add_argument("--frames", type=int, nargs="+")
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="Optional multi-episode manifest; rows may specify active_arm.",
    )
    parser.add_argument("--selection-dataset-name", default="target2058")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--evaluation-step",
        type=int,
        default=50,
        help="Plot and score only the first N predicted action steps.",
    )
    parser.add_argument(
        "--prompt-set",
        choices=("fruit", "marker", "atomic_y", "cabinet_subtask"),
        default="fruit",
        help="Counterfactual target names and instructions.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=("canonical", "native_template"),
        default="canonical",
        help=(
            "For fruit prompts, native_template preserves the recorded subtask wording "
            "and changes only its fruit target name."
        ),
    )
    parser.add_argument(
        "--fruit-target",
        action="append",
        default=[],
        help=(
            "Optional counterfactual fruit target; repeat to override the default "
            "fruit set. With native_template, only these target names are replaced."
        ),
    )
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--max-token-len", type=int, default=250)
    parser.add_argument(
        "--coefficient-target-kind",
        choices=("tcp_twist", "joint_delta"),
        default="tcp_twist",
    )
    parser.add_argument("--coefficient-target-dim", type=int, default=12)
    parser.add_argument(
        "--enable-layerwise-atomic-flow",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--disable-zm-adapter",
        action="store_true",
        help=(
            "Keep prompt-specific Context KV but pass both final and per-layer "
            "zM conditions as None to the Action Expert."
        ),
    )
    parser.add_argument(
        "--pad-subtask-horizon",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--norm-assets-dir",
        type=Path,
        default=Path("/mnt/cunchu/zjq/target"),
    )
    parser.add_argument(
        "--norm-asset-id",
        default="openpi_norm_compact_accepted_v3",
    )
    parser.add_argument("--atomic-composition-sidecar", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.selection_manifest is None and (args.episode is None or not args.frames):
        parser.error("provide --selection-manifest or both --episode and --frames")
    if args.selection_manifest is not None and (args.episode is not None or args.frames):
        parser.error("--selection-manifest is mutually exclusive with --episode/--frames")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = AtomicPi05Config(
        max_token_len=args.max_token_len,
        fast_action_ce_loss_weight=0.0,
        coefficient_target_kind=args.coefficient_target_kind,
        coefficient_target_dim=args.coefficient_target_dim,
        enable_layerwise_atomic_flow=args.enable_layerwise_atomic_flow,
    )
    dataset = build_atomic_dataset(
        (args.dataset_root,),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        action_horizon=50,
        max_token_len=args.max_token_len,
        include_fast=False,
        atomic_composition_sidecar=args.atomic_composition_sidecar,
        pad_subtask_horizon=args.pad_subtask_horizon,
    )
    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    tokenizer = _paligemma_tokenizer(args.max_token_len)
    decoder = _output_transform(
        args.dataset_root,
        config,
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
    )
    raw = dataset._raw  # noqa: SLF001
    prompts = PROMPTS
    colors = COLORS
    if args.fruit_target:
        if args.prompt_set != "fruit":
            parser.error("--fruit-target requires --prompt-set fruit")
        if len(set(args.fruit_target)) != len(args.fruit_target):
            parser.error("--fruit-target values must be unique")
        prompts = {
            target: f"Move the gripper toward the {target} and grasp it"
            for target in args.fruit_target
        }
        palette = plt.get_cmap("tab20")
        colors = {
            "recorded GT": FRUIT_COLORS["recorded GT"],
            **{
                target: FRUIT_COLORS.get(target, palette(index % 20))
                for index, target in enumerate(args.fruit_target)
            },
        }
    if args.prompt_set == "marker":
        prompts = {
            "black marker": "Move the right hand toward the black marker and grasp it",
            "blue marker": "Move the right hand toward the blue marker and grasp it",
            "white marker": "Move the right hand toward the white marker and grasp it",
        }
        colors = {
            "recorded GT": "#111827",
            "black marker": "#6b7280",
            "blue marker": "#2563eb",
            "white marker": "#f59e0b",
        }
    elif args.prompt_set == "atomic_y":
        prompts = {
            "move_y_pos": "Move leftward along base-frame +y.",
            "move_y_neg": "Move rightward along base-frame -y.",
        }
        colors = {
            "recorded GT": "#111827",
            "move_y_pos": "#dc2626",
            "move_y_neg": "#2563eb",
        }
    elif args.prompt_set == "cabinet_subtask":
        prompts = {
            "circuit breaker": (
                "Approach and brace the circuit-breaker switch, then lift and hold it raised."
            ),
            "ON button": "Approach and press the ON button.",
        }
        colors = {
            "recorded GT": "#111827",
            "circuit breaker": "#2563eb",
            "ON button": "#dc2626",
        }
    prompt_names = list(prompts)

    if args.selection_manifest is not None:
        selection_payload = json.loads(
            args.selection_manifest.read_text(encoding="utf-8")
        )
        frame_specs = selection_payload["datasets"][args.selection_dataset_name]
    else:
        frame_specs = [
            {"episode": args.episode, "frame": frame, "active_arm": args.arm}
            for frame in args.frames
        ]
    if not frame_specs:
        raise ValueError("no fruit frames selected")

    cols = 3
    rows = int(np.ceil(len(frame_specs) / cols))
    fig = plt.figure(figsize=(6.7 * cols, 6.2 * rows), constrained_layout=True)
    fig_y, axes_y = plt.subplots(
        rows,
        cols,
        figsize=(6.4 * cols, 4.6 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    report: dict[str, object] = {
        "episode": args.episode,
        "episodes": sorted({int(item["episode"]) for item in frame_specs}),
        "frames": [int(item["frame"]) for item in frame_specs],
        "checkpoint": str(args.checkpoint),
        "paired_control": "within each frame: identical images, state and flow noise; target prompt only changes",
        "evaluation_step": args.evaluation_step,
        "prompt_style": args.prompt_style,
        "selection_manifest": (
            str(args.selection_manifest) if args.selection_manifest is not None else None
        ),
        "selection_manifest_sha256": (
            hashlib.sha256(args.selection_manifest.read_bytes()).hexdigest()
            if args.selection_manifest is not None
            else None
        ),
        "evaluation_contract": {
            "max_token_len": args.max_token_len,
            "coefficient_target_kind": args.coefficient_target_kind,
            "coefficient_target_dim": args.coefficient_target_dim,
            "enable_layerwise_atomic_flow": args.enable_layerwise_atomic_flow,
            "disable_zm_adapter": args.disable_zm_adapter,
            "pad_subtask_horizon": args.pad_subtask_horizon,
            "norm_assets_dir": str(args.norm_assets_dir),
            "norm_asset_id": args.norm_asset_id,
            "atomic_composition_sidecar": args.atomic_composition_sidecar,
            "tcp_offset_m": 0.20,
        },
        "results": [],
    }
    trajectory_archive: dict[str, np.ndarray] = {
        "episodes": np.asarray([item["episode"] for item in frame_specs], dtype=np.int64),
        "frames": np.asarray([item["frame"] for item in frame_specs], dtype=np.int64),
        "arms": np.asarray(
            [item.get("active_arm", args.arm) for item in frame_specs], dtype=np.str_
        ),
    }

    for plot_index, specification in enumerate(frame_specs):
        episode = int(specification["episode"])
        frame = int(specification["frame"])
        frame_arm = str(specification.get("active_arm", args.arm))
        if frame_arm not in ("left", "right"):
            raise ValueError(f"invalid active_arm {frame_arm!r} for episode {episode}, frame {frame}")
        dataset_index, data_index, action_indices, metadata = _find_row(dataset, episode, frame)
        row = dataset[dataset_index]
        batch = atomic_collate([row])
        obs_np, _ = batch_to_observation(batch)
        observation = jax.tree.map(jnp.asarray, obs_np)
        original_prompt = metadata["subtask_prompt"]
        frame_prompts = prompts
        frame_colors = colors
        native_target_from_template = None
        if args.prompt_style == "native_template":
            if args.prompt_set != "fruit":
                raise ValueError("native_template prompt style requires --prompt-set fruit")
            frame_prompts, native_target_from_template = _native_template_prompts(
                original_prompt,
                list(prompts),
            )
            palette = plt.get_cmap("tab20")
            frame_colors = {
                "recorded GT": FRUIT_COLORS["recorded GT"],
                **{
                    name: FRUIT_COLORS.get(name, palette(index % 20))
                    for index, name in enumerate(frame_prompts)
                },
            }
        prompt_names = list(frame_prompts)
        token_rows, mask_rows = zip(
            *[
                tokenizer.tokenize(frame_prompts[name], np.asarray(row["state"]))
                for name in prompt_names
            ],
            strict=True,
        )
        noise_key = jax.random.fold_in(jax.random.key(args.seed), episode)
        noise = jax.random.normal(
            jax.random.fold_in(noise_key, frame),
            (1, 50, config.action_dim),
        )
        sample_variants = (
            _sample_variants_no_zm
            if args.disable_zm_adapter
            else _sample_variants_layerwise
        )
        normalized, _, _ = jax.device_get(
            sample_variants(
                model,
                observation,
                jnp.asarray(np.stack(token_rows)),
                jnp.asarray(np.stack(mask_rows)),
                noise,
            )
        )
        # Sidecar layout is [left_state, left_action, right_state, right_action],
        # with 12 pose values per block.  The plotted shared start must come
        # from the state block, not the action block.
        pose_offset = 0 if frame_arm == "left" else 24
        current_xyz = np.asarray(raw.tcp_pose[data_index, pose_offset : pose_offset + 3])
        gt_xyz = _tcp200_trajectory(raw, action_indices, frame_arm)
        horizon = min(args.evaluation_step, len(gt_xyz))
        if horizon <= 0:
            raise ValueError("evaluation-step must be positive")
        trajectories = {
            "recorded GT": np.concatenate([current_xyz[None], gt_xyz[:horizon]])
        }
        endpoints: dict[str, list[float]] = {}
        gt_endpoint_errors: dict[str, float] = {}
        gt_trajectory_rmse: dict[str, float] = {}
        decoded_actions: list[np.ndarray] = []
        for index, name in enumerate(prompt_names):
            actions = np.asarray(
                decoder(
                    np.asarray(batch["state"][0]),
                    metadata["raw_state"],
                    normalized[index],
                )["actions"]
            )
            decoded_actions.append(actions)
            xyz = _endpoint_tcp_from_actions(raw, data_index, actions, frame_arm)
            trajectories[name] = np.concatenate([current_xyz[None], xyz[:horizon]])
            endpoints[name] = ((xyz[horizon - 1] - current_xyz) * 1000).tolist()
            gt_endpoint_errors[name] = float(
                np.linalg.norm(xyz[horizon - 1] - gt_xyz[horizon - 1]) * 1000
            )
            gt_trajectory_rmse[name] = float(
                np.sqrt(np.mean(np.square((xyz[:horizon] - gt_xyz[:horizon]) * 1000)))
            )

        archive_prefix = f"episode_{episode:06d}_frame_{frame:06d}"
        trajectory_archive[f"{archive_prefix}_raw_state"] = np.asarray(
            metadata["raw_state"], dtype=np.float32
        )[:16]
        trajectory_archive[f"{archive_prefix}_gt_actions"] = np.asarray(
            metadata["raw_actions"], dtype=np.float32
        )[:horizon, :16]
        trajectory_archive[f"{archive_prefix}_prompt_names"] = np.asarray(
            prompt_names, dtype=np.str_
        )
        trajectory_archive[f"{archive_prefix}_original_subtask_prompt"] = np.asarray(
            original_prompt, dtype=np.str_
        )
        trajectory_archive[f"{archive_prefix}_prompt_texts"] = np.asarray(
            [frame_prompts[name] for name in prompt_names], dtype=np.str_
        )
        trajectory_archive[f"{archive_prefix}_prediction_actions"] = np.asarray(
            decoded_actions, dtype=np.float32
        )[:, :horizon, :16]

        pairwise_cosines = []
        endpoint_separations = []
        for i, first in enumerate(prompt_names):
            for second in prompt_names[i + 1 :]:
                first_delta = trajectories[first][-1] - current_xyz
                second_delta = trajectories[second][-1] - current_xyz
                pairwise_cosines.append(_cosine(first_delta, second_delta))
                endpoint_separations.append(
                    float(np.linalg.norm(trajectories[first][-1] - trajectories[second][-1]) * 1000)
                )

        ax = fig.add_subplot(rows, cols, plot_index + 1, projection="3d")
        for name, trajectory in trajectories.items():
            is_gt = name == "recorded GT"
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                color=frame_colors[name],
                lw=2.2 if is_gt else 1.8,
                ls="--" if is_gt else "-",
                label=name,
            )
            ax.scatter(*trajectory[-1], color=frame_colors[name], s=28)
        ax.scatter(*current_xyz, color="#7c3aed", marker="*", s=90)
        _equal_3d_axes(ax, list(trajectories.values()))
        ax.set_xlabel("base x (m)")
        ax.set_ylabel("base y (m)")
        ax.set_zlabel("base z (m)")
        ax.set_title(
            f"episode {episode} · frame {frame} · {frame_arm} arm\n"
            + textwrap.fill(original_prompt, width=54)
            + f"\nstep {horizon}: mean cos={np.mean(pairwise_cosines):.3f}, max gap={max(endpoint_separations):.1f} mm",
            fontsize=9,
        )
        if plot_index == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left")

        ax_y = axes_y.flat[plot_index]
        for name, trajectory in trajectories.items():
            is_gt = name == "recorded GT"
            y_mm = (trajectory[:, 1] - current_xyz[1]) * 1000
            ax_y.plot(
                np.arange(len(y_mm)),
                y_mm,
                color=frame_colors[name],
                lw=2.2 if is_gt else 1.8,
                ls="--" if is_gt else "-",
                label=name,
            )
            ax_y.scatter(len(y_mm) - 1, y_mm[-1], color=frame_colors[name], s=24)
        ax_y.axhline(0, color="#9ca3af", lw=0.7)
        ax_y.set_title(
            f"episode {episode} · frame {frame} · {frame_arm} arm\n"
            + textwrap.fill(original_prompt, width=52),
            fontsize=9,
        )
        ax_y.set_xlabel("predicted action step")
        ax_y.set_ylabel("base ΔY (mm): +left / −right")
        ax_y.grid(alpha=0.25)
        if plot_index == 0:
            ax_y.legend(frameon=False, fontsize=8)

        native_targets = [
            name for name in prompt_names if name.lower() in original_prompt.lower()
        ]
        native_target = (
            native_target_from_template
            if native_target_from_template is not None
            else (native_targets[0] if len(native_targets) == 1 else None)
        )
        native_rank = None
        native_margin_mm = None
        if native_target is not None:
            ranked = sorted(prompt_names, key=gt_endpoint_errors.__getitem__)
            native_rank = ranked.index(native_target) + 1
            native_margin_mm = min(
                gt_endpoint_errors[name]
                for name in prompt_names
                if name != native_target
            ) - gt_endpoint_errors[native_target]

        report["results"].append(
            {
                "episode": episode,
                "frame": frame,
                "time_s": frame / 30,
                "active_arm": frame_arm,
                "source_dataset": specification.get("source_dataset"),
                "target_from_selection": specification.get("target"),
                "original_subtask_prompt": original_prompt,
                "prompts_used": frame_prompts,
                "native_target": native_target,
                "native_target_gt_error_rank": native_rank,
                "native_target_margin_vs_best_wrong_mm": native_margin_mm,
                "endpoint_displacement_mm": endpoints,
                "gt_endpoint_error_mm": gt_endpoint_errors,
                "gt_trajectory_rmse_mm": gt_trajectory_rmse,
                "mean_pairwise_displacement_cosine": float(np.mean(pairwise_cosines)),
                "min_pairwise_displacement_cosine": float(np.min(pairwise_cosines)),
                "max_endpoint_separation_mm": float(np.max(endpoint_separations)),
            }
        )

    eligible = [row for row in report["results"] if row["native_target"] is not None]
    if eligible:
        report["native_target_summary"] = {
            "eligible_frames": len(eligible),
            "top1_frames": sum(
                row["native_target_gt_error_rank"] == 1 for row in eligible
            ),
            "top1_rate": float(
                np.mean([row["native_target_gt_error_rank"] == 1 for row in eligible])
            ),
            "mean_native_margin_vs_best_wrong_mm": float(
                np.mean(
                    [row["native_target_margin_vs_best_wrong_mm"] for row in eligible]
                )
            ),
        }

    fig.suptitle(
        f"Target steering across {len(frame_specs)} reviewed frames · mixed episodes/arms\n"
        f"{args.prompt_set} prompts · style={args.prompt_style} · fixed scene/state per frame",
        fontsize=16,
        fontweight="bold",
    )
    figure = args.output_dir / "fruit_target_switch_multiframe_3d.png"
    fig.savefig(figure, dpi=180, bbox_inches="tight")
    plt.close(fig)
    for empty_index in range(len(frame_specs), rows * cols):
        axes_y.flat[empty_index].axis("off")
    expected_y = (
        "Expected: move_y_pos gives +ΔY/left; move_y_neg gives −ΔY/right"
        if args.prompt_set == "atomic_y"
        else (
            "Expected marker ordering: white (+Y/left) > blue > black (−Y/right)"
            if args.prompt_set == "marker"
            else "Native target is evaluated by GT endpoint and trajectory error"
        )
    )
    fig_y.suptitle(
        f"Target steering along base Y · reviewed multi-episode selection\n{expected_y}",
        fontsize=15,
        fontweight="bold",
    )
    y_figure = args.output_dir / "target_switch_multiframe_delta_y.png"
    fig_y.savefig(y_figure, dpi=180, bbox_inches="tight")
    plt.close(fig_y)
    report["figure"] = str(figure)
    report["y_figure"] = str(y_figure)
    trajectory_npz = args.output_dir / "fruit_target_switch_joint_trajectories.npz"
    np.savez_compressed(trajectory_npz, **trajectory_archive)
    report["trajectory_npz"] = str(trajectory_npz)
    (args.output_dir / "fruit_target_switch_multiframe_3d.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
