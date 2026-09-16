#!/usr/bin/env python3
"""Test whether the force policy distinguishes a closed versus open drawer image.

For each manually reviewed close/open frame pair, evaluate both directions:
the force/state history at the closed frame with closed/open images, and the
force/state history at the open frame with closed/open images. Prompt, target,
normalization, image masks, and diffusion noise are paired exactly; only the
three processed camera tensors change.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from atomic_latent_vla.pi05.force_training_data import (
    batch_to_force_inputs,
    build_force_dataset,
    force_collate,
)
from atomic_latent_vla.pi05.weights import AtomicPi05CheckpointLoader

from evaluate_force_b2_episode50_ablation import (
    _metrics,
    _output_transform,
    _prepare_context,
    _sample_offset0,
    _tcp,
)
from evaluate_force_image_misalignment import _config, _cosine, _decode_image, _rmse


BRANCHES = ("closed_image", "open_image")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_by_episode_frame(dataset) -> dict[tuple[int, int], int]:
    raw = dataset._raw  # noqa: SLF001
    result: dict[tuple[int, int], int] = {}
    for dataset_index, anchor in enumerate(raw.anchors):
        episode = int(raw.anchor_episodes[dataset_index])
        frame = int(raw.base._frame_index[int(anchor)])  # noqa: SLF001
        result[(episode, frame)] = int(dataset_index)
    return result


def _selection(dataset, manifest: dict) -> list[dict]:
    index = _index_by_episode_frame(dataset)
    result: list[dict] = []
    for pair_index, pair in enumerate(manifest["pairs"]):
        episode = int(pair["episode"])
        closed_frame = int(pair["closed_frame"])
        open_frame = int(pair["open_frame"])
        closed_index = index[(episode, closed_frame)]
        open_index = index[(episode, open_frame)]
        for state_condition, target_index, target_frame in (
            ("closed", closed_index, closed_frame),
            ("open", open_index, open_frame),
        ):
            result.append(
                {
                    "pair_index": pair_index,
                    "episode": episode,
                    "state_condition": state_condition,
                    "target_frame": target_frame,
                    "target_index": target_index,
                    "closed_image_index": closed_index,
                    "closed_image_frame": closed_frame,
                    "open_image_index": open_index,
                    "open_image_frame": open_frame,
                }
            )
    return result


def _aggregate(rows: list[dict], branch: str) -> dict:
    prediction = np.concatenate([row[f"{branch}_normalized"] for row in rows])
    target = np.concatenate([row["gt_normalized"] for row in rows])
    decoded = np.concatenate([row[f"{branch}_decoded"] for row in rows])
    gt_decoded = np.concatenate([row["gt_decoded"] for row in rows])
    result = _metrics(prediction, target, decoded, gt_decoded)
    result["normalized_action_cosine_to_gt"] = _cosine(prediction, target)
    return result


def _plot_contact_sheet(examples: list[dict], output: Path) -> None:
    camera_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    fig, axes = plt.subplots(len(examples) * 3, 2, figsize=(9, 3.2 * len(examples) * 3))
    axes = np.asarray(axes).reshape(len(examples) * 3, 2)
    row_axis = 0
    for example in examples:
        for camera in camera_keys:
            axes[row_axis, 0].imshow(_decode_image(example["closed_image"][camera]))
            axes[row_axis, 1].imshow(_decode_image(example["open_image"][camera]))
            axes[row_axis, 0].set_title(
                f"E{example['episode']} closed F{example['closed_frame']} · {camera}", fontsize=8
            )
            axes[row_axis, 1].set_title(
                f"E{example['episode']} open F{example['open_frame']} · {camera}", fontsize=8
            )
            axes[row_axis, 0].axis("off")
            axes[row_axis, 1].axis("off")
            row_axis += 1
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_summary(rows: list[dict], output: Path) -> None:
    labels = [f"E{row['episode']} F{row['target_frame']}\nstate={row['state_condition']}" for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True)
    axes[0].bar(
        x - width / 2,
        [row["closed_image_metrics"]["normalized_action_rmse"] for row in rows],
        width,
        label="closed image",
        color="#d97706",
    )
    axes[0].bar(
        x + width / 2,
        [row["open_image_metrics"]["normalized_action_rmse"] for row in rows],
        width,
        label="open image",
        color="#2563eb",
    )
    axes[0].set_ylabel("normalized 16D RMSE to target-frame GT")
    axes[1].bar(x, [row["closed_vs_open_prediction_rmse"] for row in rows], color="#7c3aed")
    axes[1].set_ylabel("prediction RMSE: closed image vs open image")
    axes[1].set_xticks(x, labels, fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False) if axis is axes[0] else None
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_horizon(row: dict, output: Path) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(18, 12), sharex=True, constrained_layout=True)
    labels = [*[f"L q{i + 1}" for i in range(7)], "L grip", *[f"R q{i + 1}" for i in range(7)], "R grip"]
    for dim, axis in enumerate(axes.flat):
        axis.plot(row["gt_decoded"][:, dim], color="#111827", lw=2.0, label="GT")
        axis.plot(row["closed_image_decoded"][:, dim], color="#d97706", lw=1.3, ls="--", label="closed image")
        axis.plot(row["open_image_decoded"][:, dim], color="#2563eb", lw=1.3, label="open image")
        axis.set_title(labels[dim])
        axis.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"E{row['episode']} target F{row['target_frame']} state={row['state_condition']} — {row['prompt']}"
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--force-norm", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = _config()
    dataset = build_force_dataset(
        (args.dataset_root,),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        force_norm_path=args.force_norm,
        max_token_len=config.max_token_len,
        force_update_action_steps=10,
        force_update_offsets=(0, 10, 20, 30, 40),
        load_future_force_targets=False,
        seed=0,
    )
    pair_manifest = json.loads(args.pair_manifest.read_text(encoding="utf-8"))
    selection = _selection(dataset, pair_manifest)
    raw = dataset._raw  # noqa: SLF001
    for row in selection:
        anchor = int(raw.anchors[row["target_index"]])
        task_index = int(raw.base._task_index[anchor])  # noqa: SLF001
        row["prompt"] = str(raw.base.tasks[task_index])
        row["anchor_data_index"] = anchor
    selection_path = args.output_dir / "selection.json"
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")

    decode = _output_transform(
        args.dataset_root,
        config,
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
    )
    initialized = config.create(jax.random.key(0))
    _, initialized_state = nnx.split(initialized)
    params = AtomicPi05CheckpointLoader(str(args.checkpoint / "params")).load(
        initialized_state.to_pure_dict()
    )
    del initialized, initialized_state
    model = config.load(params, remove_extra_params=False)
    del params
    model.eval()

    rows: list[dict] = []
    contact_examples: list[dict] = []
    for start in range(0, len(selection), args.batch_size):
        selected = selection[start : start + args.batch_size]
        real_count = len(selected)
        if real_count < args.batch_size:
            selected += [selected[-1]] * (args.batch_size - real_count)
        target_batch = force_collate([dataset[row["target_index"]] for row in selected])
        closed_batch = force_collate([dataset[row["closed_image_index"]] for row in selected])
        open_batch = force_collate([dataset[row["open_image_index"]] for row in selected])
        target_obs_np, gt_normalized, force = batch_to_force_inputs(target_batch)
        closed_obs_np, _, _ = batch_to_force_inputs(closed_batch)
        open_obs_np, _, _ = batch_to_force_inputs(open_batch)
        target_obs = jax.tree.map(jnp.asarray, target_obs_np)
        observations = {
            "closed_image": target_obs.replace(
                images=jax.tree.map(jnp.asarray, closed_obs_np.images)
            ),
            "open_image": target_obs.replace(
                images=jax.tree.map(jnp.asarray, open_obs_np.images)
            ),
        }
        noise = jax.random.normal(
            jax.random.fold_in(jax.random.key(args.seed), start),
            (args.batch_size, config.action_horizon, config.action_dim),
        )
        current_mask = jnp.zeros_like(jnp.asarray(force["current_history_mask"]))
        predictions: dict[str, np.ndarray] = {}
        for branch in BRANCHES:
            context = _prepare_context(
                model,
                observations[branch],
                jnp.asarray(force["slow_force_history"]),
                jnp.asarray(force["slow_state_history"]),
                jnp.asarray(force["slow_history_mask"]),
            )
            prediction, _ = _sample_offset0(
                model,
                context,
                jnp.asarray(force["current_force_history"]),
                jnp.asarray(force["current_state_history"]),
                current_mask,
                noise,
                jnp.asarray(True),
            )
            predictions[branch] = np.asarray(jax.device_get(prediction))[:real_count]

        closed_images = {key: np.asarray(value)[:real_count] for key, value in closed_obs_np.images.items()}
        open_images = {key: np.asarray(value)[:real_count] for key, value in open_obs_np.images.items()}
        for local, selected_row in enumerate(selected[:real_count]):
            anchor = int(selected_row["anchor_data_index"])
            raw_state = np.asarray(raw.base._states[anchor])  # noqa: SLF001
            state_norm = np.asarray(target_batch["state"][local])
            gt_norm = np.asarray(gt_normalized[local])[..., :16]
            gt_decoded = np.asarray(decode(state_norm, raw_state, gt_norm)["actions"])
            row = dict(selected_row)
            row["gt_normalized"] = gt_norm
            row["gt_decoded"] = gt_decoded
            for branch in BRANCHES:
                pred = predictions[branch][local]
                decoded = np.asarray(decode(state_norm, raw_state, pred)["actions"])
                row[f"{branch}_normalized"] = pred
                row[f"{branch}_decoded"] = decoded
                row[f"{branch}_metrics"] = _metrics(pred, gt_norm, decoded, gt_decoded)
                row[f"{branch}_cosine_to_gt"] = _cosine(pred, gt_norm)
            row["closed_vs_open_prediction_rmse"] = _rmse(
                row["closed_image_normalized"], row["open_image_normalized"]
            )
            row["closed_vs_open_prediction_cosine"] = _cosine(
                row["closed_image_normalized"], row["open_image_normalized"]
            )
            for arm in ("left", "right"):
                closed_tcp = _tcp(row["closed_image_decoded"], arm)
                open_tcp = _tcp(row["open_image_decoded"], arm)
                row[f"{arm}_final_tcp_closed_vs_open_mm"] = float(
                    np.linalg.norm(closed_tcp[-1] - open_tcp[-1]) * 1000.0
                )
            native = f"{row['state_condition']}_image"
            counterfactual = "open_image" if native == "closed_image" else "closed_image"
            row["native_branch"] = native
            row["counterfactual_branch"] = counterfactual
            row["native_rmse"] = row[f"{native}_metrics"]["normalized_action_rmse"]
            row["counterfactual_rmse"] = row[f"{counterfactual}_metrics"]["normalized_action_rmse"]
            rows.append(row)
            if row["state_condition"] == "closed":
                contact_examples.append(
                    {
                        "episode": row["episode"],
                        "closed_frame": row["closed_image_frame"],
                        "open_frame": row["open_image_frame"],
                        "closed_image": {key: value[local] for key, value in closed_images.items()},
                        "open_image": {key: value[local] for key, value in open_images.items()},
                    }
                )

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "window_count": len(rows),
        "paired_contract": "same target prompt/state/force-state histories/GT/noise/masks; only all three image tensors are closed/open",
        "closed_image": _aggregate(rows, "closed_image"),
        "open_image": _aggregate(rows, "open_image"),
        "mean_closed_vs_open_prediction_rmse": float(
            np.mean([row["closed_vs_open_prediction_rmse"] for row in rows])
        ),
        "median_closed_vs_open_prediction_rmse": float(
            np.median([row["closed_vs_open_prediction_rmse"] for row in rows])
        ),
        "mean_closed_vs_open_prediction_cosine": float(
            np.mean([row["closed_vs_open_prediction_cosine"] for row in rows])
        ),
        "native_image_lower_gt_rmse_count": int(
            sum(row["native_rmse"] < row["counterfactual_rmse"] for row in rows)
        ),
        "mean_final_tcp_difference_mm": {
            arm: float(np.mean([row[f"{arm}_final_tcp_closed_vs_open_mm"] for row in rows]))
            for arm in ("left", "right")
        },
    }
    serializable = [
        {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in row.items()}
        for row in rows
    ]
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary | {"windows": serializable}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (args.output_dir / "windows.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "episode", "target_frame", "state_condition", "prompt",
            "closed_image_frame", "open_image_frame", "native_branch",
            "closed_image_rmse", "open_image_rmse", "native_rmse", "counterfactual_rmse",
            "closed_vs_open_prediction_rmse", "closed_vs_open_prediction_cosine",
            "left_final_tcp_closed_vs_open_mm", "right_final_tcp_closed_vs_open_mm",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "episode": row["episode"],
                    "target_frame": row["target_frame"],
                    "state_condition": row["state_condition"],
                    "prompt": row["prompt"],
                    "closed_image_frame": row["closed_image_frame"],
                    "open_image_frame": row["open_image_frame"],
                    "native_branch": row["native_branch"],
                    "closed_image_rmse": row["closed_image_metrics"]["normalized_action_rmse"],
                    "open_image_rmse": row["open_image_metrics"]["normalized_action_rmse"],
                    "native_rmse": row["native_rmse"],
                    "counterfactual_rmse": row["counterfactual_rmse"],
                    "closed_vs_open_prediction_rmse": row["closed_vs_open_prediction_rmse"],
                    "closed_vs_open_prediction_cosine": row["closed_vs_open_prediction_cosine"],
                    "left_final_tcp_closed_vs_open_mm": row["left_final_tcp_closed_vs_open_mm"],
                    "right_final_tcp_closed_vs_open_mm": row["right_final_tcp_closed_vs_open_mm"],
                }
            )
    _plot_contact_sheet(contact_examples, args.output_dir / "drawer_closed_open_contact_sheet.png")
    _plot_summary(rows, args.output_dir / "drawer_visual_effect_summary.png")
    _plot_horizon(next(row for row in rows if row["state_condition"] == "open"), args.output_dir / "representative_open_state_16d.png")

    base_norm = args.norm_assets_dir / args.norm_asset_id / "norm_stats.json"
    contract = (
        "# Evaluation contract\n\n"
        f"- checkpoint: `{args.checkpoint.resolve()}`\n"
        f"- checkpoint manifest sha256: `{_sha256(args.checkpoint / 'params' / 'manifest.ocdbt')}`\n"
        f"- dataset: `{args.dataset_root.resolve()}`\n"
        f"- base norm: `{base_norm.resolve()}` sha256 `{_sha256(base_norm)}`\n"
        f"- force norm: `{args.force_norm.resolve()}` sha256 `{_sha256(args.force_norm)}`\n"
        f"- prompt: exact task text from dataset metadata; max token length 200\n"
        f"- manually reviewed pair manifest: `{args.pair_manifest.resolve()}` sha256 `{_sha256(args.pair_manifest)}`\n"
        f"- derived selection: `{selection_path.resolve()}` sha256 `{_sha256(selection_path)}`\n"
        f"- target anchors: {len(rows)}; seed: {args.seed}; horizon: 50; offset: 0\n"
        f"- paired change: closed versus open images for all three cameras only\n"
        f"- TCP offset: 0.20 m\n"
    )
    (args.output_dir / "run_contract.md").write_text(contract, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
