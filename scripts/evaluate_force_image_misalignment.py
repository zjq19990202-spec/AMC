#!/usr/bin/env python3
"""Paired image-misalignment audit for the force-conditioned PI0.5 policy.

For every selected training anchor, prompt, current state, force/state history,
GT action target, sampler noise, and image masks stay fixed.  Only the three
camera tensors are replaced by either a distant frame from the same episode or
by a frame from another episode.  This isolates visual dependence from a
possible force/state-history shortcut.
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

from atomic_latent_vla.pi05 import AtomicPi05Config
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
)


BRANCHES = ("correct", "same_episode_wrong", "cross_episode_wrong")


def _config() -> AtomicPi05Config:
    """Exact architecture used by the deployed 5x10 nocommit/nofuture B2."""

    return AtomicPi05Config(
        max_token_len=200,
        coefficient_target_kind="joint_delta",
        coefficient_target_dim=14,
        fast_action_ce_loss_weight=0.0,
        enable_force_stage=True,
        force_fast_history_samples=40,
        force_update_action_steps=10,
        force_future_decoder_stride=4,
        force_future_decoder_kind="phase_mlp",
        force_encoder_depth=2,
        force_position_base=10_000.0,
        force_history_train_lengths=(120,),
        force_future_loss_weight=0.0,
        force_flow_loss_weight=1.0,
        force_delta_regularization_weight=1.0e-4,
        force_stop_gradient_backbone=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 1.0e-12 else float("nan")


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(left) - np.asarray(right)))))


def _sign_agreement(prediction: np.ndarray, target: np.ndarray, threshold: float = 0.05) -> float:
    prediction = np.asarray(prediction)[..., :16]
    target = np.asarray(target)[..., :16]
    active = np.abs(target) >= threshold
    if not np.any(active):
        return float("nan")
    return float(np.mean(np.sign(prediction[active]) == np.sign(target[active])))


def _candidate_indices(dataset) -> dict[int, list[int]]:
    """Use genuine 50-frame anchors from the B2 training split only."""

    raw = dataset._raw  # noqa: SLF001
    result: dict[int, list[int]] = {}
    for dataset_index, anchor in enumerate(raw.anchors):
        episode = int(raw.anchor_episodes[dataset_index])
        frame = int(raw.base._frame_index[int(anchor)])  # noqa: SLF001
        if episode % 10 == 0 or frame % 50:
            continue
        result.setdefault(episode, []).append(dataset_index)
    return result


def _evenly_spaced(values: list[int], count: int) -> list[int]:
    if len(values) < count:
        raise ValueError(f"need {count} values, got {len(values)}")
    positions = np.linspace(0, len(values) - 1, count + 2, dtype=int)[1:-1]
    return [values[int(position)] for position in positions]


def _selection(dataset, episodes: int, horizons_per_episode: int) -> list[dict]:
    candidates = _candidate_indices(dataset)
    eligible = sorted(episode for episode, rows in candidates.items() if len(rows) >= horizons_per_episode + 2)
    chosen_episodes = _evenly_spaced(eligible, episodes)
    raw = dataset._raw  # noqa: SLF001
    rows: list[dict] = []
    for episode in chosen_episodes:
        ordered = sorted(
            candidates[episode],
            key=lambda index: int(raw.base._frame_index[int(raw.anchors[index])]),  # noqa: SLF001
        )
        targets = _evenly_spaced(ordered, horizons_per_episode)
        for target in targets:
            target_anchor = int(raw.anchors[target])
            target_frame = int(raw.base._frame_index[target_anchor])  # noqa: SLF001
            same_donor = max(
                ordered,
                key=lambda index: abs(
                    int(raw.base._frame_index[int(raw.anchors[index])]) - target_frame  # noqa: SLF001
                ),
            )
            rows.append(
                {
                    "target_index": int(target),
                    "episode": int(episode),
                    "frame": target_frame,
                    "same_episode_donor_index": int(same_donor),
                    "same_episode_donor_frame": int(
                        raw.base._frame_index[int(raw.anchors[same_donor])]  # noqa: SLF001
                    ),
                }
            )
    # Rotate by one episode block so every cross donor comes from another episode.
    block = horizons_per_episode
    for index, row in enumerate(rows):
        donor = rows[(index + block) % len(rows)]
        if donor["episode"] == row["episode"]:
            raise RuntimeError("cross-episode donor construction failed")
        row["cross_episode_donor_index"] = int(donor["target_index"])
        row["cross_episode_donor_episode"] = int(donor["episode"])
        row["cross_episode_donor_frame"] = int(donor["frame"])
    return rows


def _replace_images(target, donor):
    """Replace exactly the image tensors; retain target masks/state/text."""

    return target.replace(images=donor.images)


def _decode_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"expected one image, got {image.shape}")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.moveaxis(image, 0, -1)
    return np.clip((image + 1.0) * 127.5, 0, 255).astype(np.uint8)


def _plot_contact_sheet(examples: list[dict], output: Path) -> None:
    camera_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    fig, axes = plt.subplots(len(examples) * len(camera_keys), 3, figsize=(12, 3.2 * len(examples) * len(camera_keys)))
    axes = np.asarray(axes).reshape(len(examples) * len(camera_keys), 3)
    titles = ("correct image", "same-episode wrong image", "cross-episode wrong image")
    row_index = 0
    for example in examples:
        for camera in camera_keys:
            for column, branch in enumerate(BRANCHES):
                axes[row_index, column].imshow(_decode_image(example[branch][camera]))
                axes[row_index, column].axis("off")
                axes[row_index, column].set_title(
                    f"{titles[column]}\nE{example['episode']} F{example['frame']} {camera}", fontsize=8
                )
            row_index += 1
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _aggregate(rows: list[dict], branch: str) -> dict:
    predictions = np.concatenate([row[f"{branch}_normalized"] for row in rows])
    targets = np.concatenate([row["gt_normalized"] for row in rows])
    decoded = np.concatenate([row[f"{branch}_decoded"] for row in rows])
    gt_decoded = np.concatenate([row["gt_decoded"] for row in rows])
    result = _metrics(predictions, targets, decoded, gt_decoded)
    result["normalized_action_cosine_to_gt"] = _cosine(predictions, targets)
    result["active_sign_agreement"] = _sign_agreement(predictions, targets)
    return result


def _plot_summary(rows: list[dict], output: Path) -> None:
    labels = [f"E{row['episode']}:F{row['frame']}" for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(18, 9), constrained_layout=True)
    for branch, color in zip(BRANCHES, ("#111827", "#d97706", "#dc2626"), strict=True):
        axes[0].plot(x, [row[f"{branch}_metrics"]["normalized_action_rmse"] for row in rows], "o-", ms=3, label=branch, color=color)
        axes[1].plot(x, [row[f"{branch}_cosine_to_gt"] for row in rows], "o-", ms=3, label=branch, color=color)
    axes[0].set_ylabel("normalized 16D RMSE to GT")
    axes[1].set_ylabel("normalized action cosine to GT")
    axes[1].set_xticks(x, labels, rotation=70, ha="right", fontsize=7)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_horizon(row: dict, output: Path) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(18, 12), sharex=True, constrained_layout=True)
    labels = [*[f"L q{i + 1}" for i in range(7)], "L grip", *[f"R q{i + 1}" for i in range(7)], "R grip"]
    for dim, axis in enumerate(axes.flat):
        axis.plot(row["gt_decoded"][:, dim], color="#111827", lw=2.0, label="GT")
        axis.plot(row["correct_decoded"][:, dim], color="#2563eb", lw=1.4, label="correct image")
        axis.plot(row["same_episode_wrong_decoded"][:, dim], color="#d97706", lw=1.2, ls="--", label="same-episode wrong")
        axis.plot(row["cross_episode_wrong_decoded"][:, dim], color="#dc2626", lw=1.2, ls=":", label="cross-episode wrong")
        axis.set_title(labels[dim])
        axis.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Episode {row['episode']} frame {row['frame']} — {row['prompt']}")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--force-norm", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--horizons-per-episode", type=int, default=4)
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
    selection = _selection(dataset, args.episodes, args.horizons_per_episode)
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
    # Build the reference directly on the selected accelerator, then release it
    # before reconstructing the checkpoint model.  An explicit whole-tree
    # host->device put transiently duplicates the 12 GB parameter tree in host
    # RAM on this JAX/NNX version.
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
        selected_rows = selection[start : start + args.batch_size]
        real_count = len(selected_rows)
        if real_count < args.batch_size:
            selected_rows += [selected_rows[-1]] * (args.batch_size - real_count)
        target_samples = [dataset[row["target_index"]] for row in selected_rows]
        same_samples = [dataset[row["same_episode_donor_index"]] for row in selected_rows]
        cross_samples = [dataset[row["cross_episode_donor_index"]] for row in selected_rows]
        target_batch = force_collate(target_samples)
        same_batch = force_collate(same_samples)
        cross_batch = force_collate(cross_samples)
        target_obs_np, gt_normalized, force = batch_to_force_inputs(target_batch)
        same_obs_np, _, _ = batch_to_force_inputs(same_batch)
        cross_obs_np, _, _ = batch_to_force_inputs(cross_batch)
        observations = {
            "correct": jax.tree.map(jnp.asarray, target_obs_np),
            "same_episode_wrong": _replace_images(
                jax.tree.map(jnp.asarray, target_obs_np), jax.tree.map(jnp.asarray, same_obs_np)
            ),
            "cross_episode_wrong": _replace_images(
                jax.tree.map(jnp.asarray, target_obs_np), jax.tree.map(jnp.asarray, cross_obs_np)
            ),
        }
        noise = jax.random.normal(
            jax.random.fold_in(jax.random.key(args.seed), start),
            (args.batch_size, config.action_horizon, config.action_dim),
        )
        current_mask = jnp.zeros_like(jnp.asarray(force["current_history_mask"]))
        predictions: dict[str, np.ndarray] = {}
        delta_z: dict[str, np.ndarray] = {}
        for branch in BRANCHES:
            context = _prepare_context(
                model,
                observations[branch],
                jnp.asarray(force["slow_force_history"]),
                jnp.asarray(force["slow_state_history"]),
                jnp.asarray(force["slow_history_mask"]),
            )
            prediction, branch_delta = _sample_offset0(
                model,
                context,
                jnp.asarray(force["current_force_history"]),
                jnp.asarray(force["current_state_history"]),
                current_mask,
                noise,
                jnp.asarray(True),
            )
            predictions[branch] = np.asarray(jax.device_get(prediction))[:real_count]
            delta_z[branch] = np.asarray(jax.device_get(branch_delta))[:real_count]

        target_images = {key: np.asarray(value)[:real_count] for key, value in target_obs_np.images.items()}
        same_images = {key: np.asarray(value)[:real_count] for key, value in same_obs_np.images.items()}
        cross_images = {key: np.asarray(value)[:real_count] for key, value in cross_obs_np.images.items()}
        for local, selection_row in enumerate(selected_rows[:real_count]):
            anchor = int(selection_row["anchor_data_index"])
            raw_state = np.asarray(raw.base._states[anchor])  # noqa: SLF001
            state_norm = np.asarray(target_batch["state"][local])
            gt_norm = np.asarray(gt_normalized[local])[..., :16]
            gt_decoded = np.asarray(decode(state_norm, raw_state, gt_norm)["actions"])
            row = dict(selection_row)
            row["gt_normalized"] = gt_norm
            row["gt_decoded"] = gt_decoded
            for branch in BRANCHES:
                pred = predictions[branch][local]
                decoded = np.asarray(decode(state_norm, raw_state, pred)["actions"])
                row[f"{branch}_normalized"] = pred
                row[f"{branch}_decoded"] = decoded
                row[f"{branch}_metrics"] = _metrics(pred, gt_norm, decoded, gt_decoded)
                row[f"{branch}_cosine_to_gt"] = _cosine(pred, gt_norm)
                row[f"{branch}_active_sign_agreement"] = _sign_agreement(pred, gt_norm)
                row[f"{branch}_delta_z_norm"] = float(np.mean(np.linalg.norm(delta_z[branch][local], axis=-1)))
            for branch in BRANCHES[1:]:
                row[f"{branch}_prediction_rmse_vs_correct"] = _rmse(
                    row[f"{branch}_normalized"], row["correct_normalized"]
                )
                row[f"{branch}_prediction_cosine_vs_correct"] = _cosine(
                    row[f"{branch}_normalized"], row["correct_normalized"]
                )
            rows.append(row)
            if len(contact_examples) < 2:
                contact_examples.append(
                    {
                        "episode": row["episode"],
                        "frame": row["frame"],
                        "correct": {key: value[local] for key, value in target_images.items()},
                        "same_episode_wrong": {key: value[local] for key, value in same_images.items()},
                        "cross_episode_wrong": {key: value[local] for key, value in cross_images.items()},
                    }
                )

    aggregates = {branch: _aggregate(rows, branch) for branch in BRANCHES}
    correct_rmse = aggregates["correct"]["normalized_action_rmse"]
    for branch in BRANCHES[1:]:
        aggregates[branch]["gt_rmse_change_vs_correct_pct"] = 100.0 * (
            aggregates[branch]["normalized_action_rmse"] / correct_rmse - 1.0
        )
        aggregates[branch]["mean_prediction_rmse_vs_correct"] = float(
            np.mean([row[f"{branch}_prediction_rmse_vs_correct"] for row in rows])
        )
        aggregates[branch]["mean_prediction_cosine_vs_correct"] = float(
            np.mean([row[f"{branch}_prediction_cosine_vs_correct"] for row in rows])
        )
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "split": "B2 training episodes only (episode % 10 != 0)",
        "window_count": len(rows),
        "covered_gt_steps": len(rows) * 50,
        "selection": "episode coverage plus four evenly spaced 50-step anchors; prediction-independent",
        "paired_contract": "same prompt/state/force-state histories/GT/noise/masks; only all three image tensors change",
        "aggregates": aggregates,
    }
    serializable_rows = [
        {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in row.items()}
        for row in rows
    ]
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary | {"windows": serializable_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (args.output_dir / "windows.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["episode", "frame", "prompt"]
        for branch in BRANCHES:
            fields += [
                f"{branch}_normalized_action_rmse",
                f"{branch}_cosine_to_gt",
                f"{branch}_active_sign_agreement",
            ]
            if branch != "correct":
                fields += [
                    f"{branch}_prediction_rmse_vs_correct",
                    f"{branch}_prediction_cosine_vs_correct",
                ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in ("episode", "frame", "prompt")}
            for branch in BRANCHES:
                flat[f"{branch}_normalized_action_rmse"] = row[f"{branch}_metrics"]["normalized_action_rmse"]
                flat[f"{branch}_cosine_to_gt"] = row[f"{branch}_cosine_to_gt"]
                flat[f"{branch}_active_sign_agreement"] = row[f"{branch}_active_sign_agreement"]
                if branch != "correct":
                    flat[f"{branch}_prediction_rmse_vs_correct"] = row[f"{branch}_prediction_rmse_vs_correct"]
                    flat[f"{branch}_prediction_cosine_vs_correct"] = row[f"{branch}_prediction_cosine_vs_correct"]
            writer.writerow(flat)
    _plot_summary(rows, args.output_dir / "image_misalignment_summary.png")
    _plot_horizon(rows[len(rows) // 2], args.output_dir / "representative_16d_horizon.png")
    _plot_contact_sheet(contact_examples, args.output_dir / "image_pair_contact_sheet.png")

    base_norm = args.norm_assets_dir / args.norm_asset_id / "norm_stats.json"
    contract = (
        "# Evaluation contract\n\n"
        f"- repository commit: `52796fb3bd863419c69f815503a406aaaeb574f7` (dirty worktree preserved)\n"
        f"- checkpoint: `{args.checkpoint.resolve()}`\n"
        f"- checkpoint manifest sha256: `{_sha256(args.checkpoint / 'params' / 'manifest.ocdbt')}`\n"
        f"- dataset: `{args.dataset_root.resolve()}`\n"
        f"- base norm: `{base_norm.resolve()}` sha256 `{_sha256(base_norm)}`\n"
        f"- force norm: `{args.force_norm.resolve()}` sha256 `{_sha256(args.force_norm)}`\n"
        f"- prompt: exact task/subtask from `meta/tasks.parquet`; max token length 200\n"
        f"- selection manifest: `{selection_path.resolve()}` sha256 `{_sha256(selection_path)}`\n"
        f"- selection: {len(rows)} training-split 50-step horizons across {args.episodes} episodes\n"
        f"- seed: {args.seed}; offset: 0; sampler: 10 Euler steps; same-noise paired\n"
        f"- paired change: replace all three processed image tensors only\n"
        f"- TCP offset: 0.20 m\n"
    )
    (args.output_dir / "run_contract.md").write_text(contract, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
