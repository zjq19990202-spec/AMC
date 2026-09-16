#!/usr/bin/env python3
"""Plot deterministic 50-step joint horizons on arbitrary CR1 LeRobot datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import nnx_utils
from openpi.training import config as _training_config
from openpi.training.data_loader import TransformedDataset
from openpi.training.lerobot_v3_dataset import LeRobotV3Dataset

from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import (
    _LeRobotV21CompatDataset,
    _NormalizeWithoutQuantileClipping,
    _paligemma_tokenizer,
)


JOINT_INDICES = tuple(range(7)) + tuple(range(8, 15))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_datasets(values: list[str]) -> dict[str, Path]:
    datasets: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"dataset must be NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        if not name or name in datasets:
            raise ValueError(f"invalid or duplicate dataset name {name!r}")
        root = Path(path)
        if not (root / "meta" / "info.json").is_file():
            raise FileNotFoundError(root / "meta" / "info.json")
        datasets[name] = root
    return datasets


def _stack(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, dict):
        return {key: _stack([value[key] for value in values]) for key in first}
    array = np.stack([np.asarray(value) for value in values])
    if np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32, copy=False)
    return array


def _build_dataset(
    root: Path,
    norm_assets_dir: Path,
    norm_asset_id: str,
    config: AtomicPi05Config,
):
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = int(info["fps"])
    dataset_type = (
        _LeRobotV21CompatDataset
        if str(info.get("codebase_version", "")).startswith("v2")
        else LeRobotV3Dataset
    )
    base = dataset_type(
        root,
        delta_timestamps={
            "action": [step / fps for step in range(config.action_horizon)]
        },
    )
    bridge = pi0_config.Pi0Config(pi05=True, max_token_len=config.max_token_len)
    factory = _training_config.LeRobotMarvinDataConfig(
        repo_id=str(root),
        prompt_from_task=True,
        adapt_to_pi=True,
        assets=_training_config.AssetsConfig(
            assets_dir=str(norm_assets_dir), asset_id=norm_asset_id
        ),
        repack_transforms=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.base_0_rgb",
                            "cam_left_wrist": "observation.images.left_wrist_0_rgb",
                            "cam_right_wrist": "observation.images.right_wrist_0_rgb",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        ),
    )
    data_config = factory.create(norm_assets_dir, bridge)
    transforms = (
        [
            _transforms.PromptFromLeRobotTask(base.tasks),
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _NormalizeWithoutQuantileClipping(
                data_config.norm_stats,
                use_quantiles=data_config.use_quantile_norm,
            ),
            *data_config.model_transforms.inputs,
        ]
    )
    return base, TransformedDataset(base, transforms), data_config


def _select_complete_horizons(
    base: LeRobotV3Dataset,
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for dataset_index, data_index in enumerate(base._visible_indices):  # noqa: SLF001
        data_index = int(data_index)
        episode = int(base._episode_index[data_index])  # noqa: SLF001
        query_indices, _ = base._get_query_indices(data_index, episode)  # noqa: SLF001
        action_indices = np.asarray(query_indices["action"], dtype=np.int64)
        if (
            action_indices.shape != (50,)
            or len(np.unique(action_indices)) != 50
        ):
            continue
        frame = int(base._frame_index[data_index])  # noqa: SLF001
        by_episode[episode].append(
            {
                "dataset_index": dataset_index,
                "data_index": data_index,
                "episode": episode,
                "frame": frame,
                "action_indices": action_indices,
            }
        )
    if len(by_episode) < count:
        raise ValueError(
            f"requested {count} distinct episodes but only {len(by_episode)} have complete horizons"
        )
    rng = np.random.default_rng(seed)
    chosen_episodes = rng.choice(np.asarray(sorted(by_episode)), size=count, replace=False)
    selected = []
    for episode in chosen_episodes:
        candidates = by_episode[int(episode)]
        selected.append(candidates[int(rng.integers(len(candidates)))])
    return selected


def _select_manifest_horizons(
    base: LeRobotV3Dataset,
    requested: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve explicit episode/frame rows and validate their 50-step targets."""
    wanted = {(int(row["episode"]), int(row["frame"])): row for row in requested}
    if len(wanted) != len(requested):
        raise ValueError("selection manifest contains duplicate episode/frame rows")
    selected: dict[tuple[int, int], dict[str, Any]] = {}
    for dataset_index, data_index in enumerate(base._visible_indices):  # noqa: SLF001
        data_index = int(data_index)
        episode = int(base._episode_index[data_index])  # noqa: SLF001
        frame = int(base._frame_index[data_index])  # noqa: SLF001
        key = (episode, frame)
        if key not in wanted:
            continue
        query_indices, _ = base._get_query_indices(data_index, episode)  # noqa: SLF001
        action_indices = np.asarray(query_indices["action"], dtype=np.int64)
        if action_indices.shape != (50,) or len(np.unique(action_indices)) != 50:
            raise ValueError(f"episode {episode}, frame {frame} lacks a complete horizon")
        selected[key] = {
            "dataset_index": dataset_index,
            "data_index": data_index,
            "episode": episode,
            "frame": frame,
            "action_indices": action_indices,
            **{
                key: value
                for key, value in wanted[key].items()
                if key not in {"dataset_index", "data_index", "action_indices"}
            },
        }
    missing = [key for key in wanted if key not in selected]
    if missing:
        raise ValueError(f"selection rows not found in dataset: {missing}")
    return [selected[(int(row["episode"]), int(row["frame"]))] for row in requested]


def _override_prompt(row: dict[str, Any], prompt: str, max_token_len: int) -> dict[str, Any]:
    result = dict(row)
    tokens, mask = _paligemma_tokenizer(max_token_len).tokenize(
        prompt, np.asarray(row["state"], dtype=np.float32)
    )
    result["tokenized_prompt"] = tokens
    result["tokenized_prompt_mask"] = mask
    return result


def _decoder(data_config):
    """Apply the exact output-transform order used by OpenPI Policy.infer."""
    output_transforms = (
        *data_config.model_transforms.outputs,
        _transforms.Unnormalize(
            data_config.norm_stats, use_quantiles=data_config.use_quantile_norm
        ),
        *data_config.data_transforms.outputs,
        *data_config.repack_transforms.outputs,
    )

    def decode(normalized_state, raw_state, normalized_actions):
        values = {
            "state": np.asarray(normalized_state),
            "actions": np.asarray(normalized_actions),
        }
        for transform in output_transforms:
            values = transform(values)
        return np.asarray(values["actions"])[..., :16]

    return decode


def _metrics(gt: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = prediction - gt
    joint_error = error[..., JOINT_INDICES]
    endpoint_error = joint_error[-1] if joint_error.ndim == 2 else joint_error[:, -1]
    per_dimension = []
    per_dimension_endpoint = []
    per_dimension_max_abs = []
    for dimension in range(16):
        values = error[..., dimension]
        if dimension not in (7, 15):
            values = np.degrees(values)
        per_dimension.append(float(np.sqrt(np.mean(np.square(values)))))
        per_dimension_endpoint.append(float(np.mean(np.asarray(values)[..., -1])))
        per_dimension_max_abs.append(float(np.max(np.abs(values))))
    return {
        "joint_14d_rmse_deg": float(np.degrees(np.sqrt(np.mean(np.square(joint_error))))),
        "joint_14d_endpoint_rmse_deg": float(
            np.degrees(np.sqrt(np.mean(np.square(endpoint_error))))
        ),
        "left_joint_rmse_deg": float(
            np.degrees(np.sqrt(np.mean(np.square(error[..., :7]))))
        ),
        "right_joint_rmse_deg": float(
            np.degrees(np.sqrt(np.mean(np.square(error[..., 8:15]))))
        ),
        "left_gripper_rmse_native": float(np.sqrt(np.mean(np.square(error[..., 7])))),
        "right_gripper_rmse_native": float(np.sqrt(np.mean(np.square(error[..., 15])))),
        "per_dimension_rmse": per_dimension,
        "per_dimension_endpoint_signed": per_dimension_endpoint,
        "per_dimension_max_abs": per_dimension_max_abs,
    }


def _plot(
    *,
    dataset_name: str,
    checkpoint_label: str,
    episode: int,
    frame: int,
    prompt: str,
    gt: np.ndarray,
    prediction: np.ndarray,
    output: Path,
) -> None:
    figure, axes = plt.subplots(
        4, 4, figsize=(19, 13), sharex=True, constrained_layout=True
    )
    steps = np.arange(50)
    for dimension, axis in enumerate(axes.flat):
        arm = "L" if dimension < 8 else "R"
        local = dimension if dimension < 8 else dimension - 8
        gripper = local == 7
        gt_values = gt[:, dimension] if gripper else np.degrees(gt[:, dimension])
        pred_values = (
            prediction[:, dimension]
            if gripper
            else np.degrees(prediction[:, dimension])
        )
        axis.plot(steps, gt_values, color="#111827", linewidth=2.2, label="GT")
        axis.plot(
            steps,
            pred_values,
            color="#2563eb",
            linewidth=1.8,
            label=f"{checkpoint_label} · pi0.5 infer",
        )
        axis.set_title(f"{arm} gripper" if gripper else f"{arm} J{local + 1}")
        axis.set_ylabel("native" if gripper else "deg")
        axis.grid(color="#d1d5db", linewidth=0.7, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        if dimension >= 12:
            axis.set_xlabel("horizon step")
        if dimension == 0:
            axis.legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"{dataset_name} · episode {episode}, frame {frame} · 50-step joint horizon\n"
        + textwrap.fill(prompt, width=110),
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_right_arm(
    *,
    dataset_name: str,
    checkpoint_label: str,
    episode: int,
    frame: int,
    prompt: str,
    gt: np.ndarray,
    prediction: np.ndarray,
    output: Path,
) -> None:
    figure, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True, constrained_layout=True)
    steps = np.arange(50)
    for local, axis in enumerate(axes.flat):
        dimension = local + 8
        gripper = local == 7
        gt_values = gt[:, dimension] if gripper else np.degrees(gt[:, dimension])
        pred_values = prediction[:, dimension] if gripper else np.degrees(prediction[:, dimension])
        axis.plot(steps, gt_values, color="#111827", linewidth=2.3, label="recorded GT")
        axis.plot(
            steps,
            pred_values,
            color="#dc2626",
            linewidth=1.9,
            label=f"{checkpoint_label} · native subtask prediction",
        )
        axis.set_title("Right gripper" if gripper else f"Right J{local + 1}")
        axis.set_ylabel("native units" if gripper else "degrees")
        axis.set_xlabel("horizon step")
        axis.grid(color="#d1d5db", linewidth=0.7, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        if local == 0:
            axis.legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"{dataset_name} · episode {episode}, frame {frame} · right-arm 50-step joint horizon\n"
        + textwrap.fill(prompt, width=90),
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_summary(report: dict[str, Any], output: Path) -> None:
    names = list(report["datasets"])
    joint = [report["datasets"][name]["aggregate"]["joint_14d_rmse_deg"] for name in names]
    endpoint = [
        report["datasets"][name]["aggregate"]["joint_14d_endpoint_rmse_deg"]
        for name in names
    ]
    x = np.arange(len(names))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    axis.bar(x - width / 2, joint, width, label="50-step 14D RMSE", color="#2563eb")
    axis.bar(x + width / 2, endpoint, width, label="endpoint 14D RMSE", color="#f97316")
    axis.set_xticks(x, names)
    axis.set_ylabel("degrees")
    axis.set_title(f"{report['checkpoint_label']} · GT joint-horizon error")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-label", default="weighted-cos latest")
    parser.add_argument("--source-revision", default="unknown")
    parser.add_argument("--dataset", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--samples-per-dataset", type=int, default=4)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="Optional explicit episode/frame selection with per-row prompt overrides",
    )
    parser.add_argument("--selection-seed", type=int, default=20260819)
    parser.add_argument("--noise-seed", type=int, default=20260820)
    parser.add_argument(
        "--noise-key-mode",
        choices=("batch", "frame_fold_in"),
        default="batch",
        help="frame_fold_in matches steering evaluators that key noise by episode-local frame.",
    )
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    datasets = _parse_datasets(args.dataset)
    selection_manifest = None
    if args.selection_manifest is not None:
        selection_manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=False)
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
    prepared: dict[str, Any] = {}
    all_rows = []
    all_metadata = []
    for dataset_offset, (name, root) in enumerate(datasets.items()):
        base, transformed, data_config = _build_dataset(
            root, args.norm_assets_dir, args.norm_asset_id, config
        )
        if selection_manifest is None:
            selected = _select_complete_horizons(
                base,
                count=args.samples_per_dataset,
                seed=args.selection_seed + dataset_offset,
            )
        else:
            selected = _select_manifest_horizons(
                base, selection_manifest["datasets"][name]
            )
        rows = [transformed[item["dataset_index"]] for item in selected]
        rows = [
            _override_prompt(row, str(item["prompt"]), config.max_token_len)
            if str(item.get("prompt", "")).strip()
            else row
            for row, item in zip(rows, selected, strict=True)
        ]
        prepared[name] = (root, base, data_config, selected, rows)
        all_rows.extend(rows)
        all_metadata.extend((name, item) for item in selected)

    batch = _stack(all_rows)
    normalized_gt = np.asarray(batch.pop("actions"))[..., :16]
    observation = _model.Observation.from_dict(batch)
    observation = jax.tree.map(jnp.asarray, observation)

    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    if args.noise_key_mode == "frame_fold_in":
        noise = jnp.stack(
            [
                jax.random.normal(
                    jax.random.fold_in(jax.random.key(args.noise_seed), int(item["frame"])),
                    (config.action_horizon, config.action_dim),
                )
                for _, item in all_metadata
            ]
        )
    else:
        noise = jax.random.normal(
            jax.random.key(args.noise_seed),
            (len(all_rows), config.action_horizon, config.action_dim),
        )
    sample_actions = nnx_utils.module_jit(model.sample_actions)
    normalized_prediction = np.asarray(
        jax.device_get(
            sample_actions(
                jax.random.key(args.noise_seed),
                observation,
                num_steps=args.num_steps,
                noise=noise,
            )
        )
    )[..., :16]

    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_label": args.checkpoint_label,
        "source_checkout": str(Path(__file__).resolve().parents[1]),
        "source_revision": args.source_revision,
        "norm_assets_dir": str(args.norm_assets_dir),
        "norm_asset_id": args.norm_asset_id,
        "norm_stats_sha256": _sha256(norm_stats),
        "max_token_len": config.max_token_len,
        "action_horizon": config.action_horizon,
        "action_dim": config.active_action_dim,
        "selection_seed": args.selection_seed,
        "noise_seed": args.noise_seed,
        "noise_key_mode": args.noise_key_mode,
        "flow_ode_steps": args.num_steps,
        "prompt_source": (
            "explicit per-horizon prompt manifest"
            if selection_manifest is not None
            else "native LeRobot task prompt"
        ),
        "input_selection_manifest": (
            str(args.selection_manifest) if args.selection_manifest is not None else None
        ),
        "input_selection_manifest_sha256": (
            _sha256(args.selection_manifest) if args.selection_manifest is not None else None
        ),
        "inference_path": "AtomicPi05.sample_actions plus exact OpenPI Policy.infer output transforms",
        "datasets": {},
    }
    trajectories: dict[str, Any] = {}
    cursor = 0
    for name, (root, base, data_config, selected, rows) in prepared.items():
        decode = _decoder(data_config)
        samples = []
        gt_rows, prediction_rows = [], []
        for local_index, item in enumerate(selected):
            index = cursor + local_index
            raw_state = np.asarray(base._states[item["data_index"]])  # noqa: SLF001
            raw_actions = np.asarray(base._actions[item["action_indices"]])  # noqa: SLF001
            prediction = decode(rows[local_index]["state"], raw_state, normalized_prediction[index])
            gt = raw_actions[..., :16]
            task_index = int(base._task_index[item["data_index"]])  # noqa: SLF001
            prompt = str(item.get("prompt") or base.tasks[task_index])
            metrics = _metrics(gt, prediction)
            source_dataset = str(item.get("source_dataset") or name)
            display_name = f"{name} · {source_dataset}"
            plot = args.output_dir / (
                f"{name}_episode_{item['episode']:06d}_frame_{item['frame']:06d}_16d.png"
            )
            _plot(
                dataset_name=display_name,
                checkpoint_label=args.checkpoint_label,
                episode=item["episode"],
                frame=item["frame"],
                prompt=prompt,
                gt=gt,
                prediction=prediction,
                output=plot,
            )
            right_plot = args.output_dir / (
                f"{name}_episode_{item['episode']:06d}_frame_{item['frame']:06d}_right8.png"
            )
            _plot_right_arm(
                dataset_name=display_name,
                checkpoint_label=args.checkpoint_label,
                episode=item["episode"],
                frame=item["frame"],
                prompt=prompt,
                gt=gt,
                prediction=prediction,
                output=right_plot,
            )
            normalized_state = np.asarray(rows[local_index]["state"])[..., :16]
            samples.append(
                {
                    "dataset_index": item["dataset_index"],
                    "episode": item["episode"],
                    "frame": item["frame"],
                    "prompt": prompt,
                    "prompt_provenance": item.get("prompt_provenance"),
                    "source_dataset": item.get("source_dataset"),
                    "source_episode": item.get("source_episode"),
                    "episode_task": item.get("episode_task"),
                    "segment_id": item.get("segment_id"),
                    "segment_start": item.get("segment_start"),
                    "segment_end_exclusive": item.get("segment_end_exclusive"),
                    "target": item.get("target"),
                    "active_arm": item.get("active_arm"),
                    "selection_basis": item.get("selection_basis"),
                    "left_gt_path_deg": item.get("left_gt_path_deg"),
                    "right_gt_path_deg": item.get("right_gt_path_deg"),
                    "normalized_state_outlier_fraction": float(
                        np.mean(np.abs(normalized_state) > 1.0)
                    ),
                    "normalized_16d_mse": float(
                        np.mean(np.square(normalized_prediction[index] - normalized_gt[index]))
                    ),
                    "metrics": metrics,
                    "plot": str(plot),
                    "right_arm_plot": str(right_plot),
                }
            )
            gt_rows.append(gt)
            prediction_rows.append(prediction)
        gt_all = np.stack(gt_rows)
        prediction_all = np.stack(prediction_rows)
        report["datasets"][name] = {
            "root": str(root),
            "sample_count": len(samples),
            "aggregate": _metrics(gt_all, prediction_all),
            "normalized_16d_mse": float(
                np.mean(np.square(normalized_prediction[cursor : cursor + len(rows)] - normalized_gt[cursor : cursor + len(rows)]))
            ),
            "samples": samples,
        }
        trajectories[f"{name}_gt"] = np.stack(gt_rows)
        trajectories[f"{name}_prediction"] = np.stack(prediction_rows)
        cursor += len(rows)

    summary_plot = args.output_dir / "dataset_joint_rmse_summary.png"
    _plot_summary(report, summary_plot)
    report["summary_plot"] = str(summary_plot)
    selection = {
        "selection_seed": args.selection_seed,
        "samples_per_dataset": args.samples_per_dataset,
        "datasets": {
            name: [
                {
                    "dataset_index": sample["dataset_index"],
                    "episode": sample["episode"],
                    "frame": sample["frame"],
                    "prompt": sample["prompt"],
                    "prompt_provenance": sample["prompt_provenance"],
                    "source_dataset": sample["source_dataset"],
                    "source_episode": sample["source_episode"],
                    "episode_task": sample["episode_task"],
                    "segment_id": sample["segment_id"],
                    "segment_start": sample["segment_start"],
                    "segment_end_exclusive": sample["segment_end_exclusive"],
                    "target": sample["target"],
                    "active_arm": sample["active_arm"],
                    "selection_basis": sample["selection_basis"],
                    "left_gt_path_deg": sample["left_gt_path_deg"],
                    "right_gt_path_deg": sample["right_gt_path_deg"],
                }
                for sample in value["samples"]
            ]
            for name, value in report["datasets"].items()
        },
    }
    selection_path = args.output_dir / "selection.json"
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report["selection"] = str(selection_path)
    report["selection_sha256"] = _sha256(selection_path)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(args.output_dir / "trajectories.npz", **trajectories)
    command = " ".join(__import__("shlex").quote(value) for value in __import__("sys").argv)
    sampling_contract = (
        "Explicit sidecar-selected horizons: "
        + ", ".join(
            f"{name}={value['sample_count']}" for name, value in report["datasets"].items()
        )
        if selection_manifest is not None
        else f"Samples per dataset: {args.samples_per_dataset}, distinct episodes"
    )
    (args.output_dir / "run_contract.md").write_text(
        "\n".join(
            [
                "# Weighted-cos multidataset joint-horizon evaluation",
                "",
                f"- Checkpoint: `{args.checkpoint}`",
                f"- Source revision: `{args.source_revision}`",
                f"- Datasets: `{', '.join(f'{name}={root}' for name, root in datasets.items())}`",
                f"- Norm: `{args.norm_assets_dir / args.norm_asset_id}`",
                f"- norm_stats SHA-256: `{report['norm_stats_sha256']}`",
                "- Prompt length: 200 tokens",
                "- Coefficient target: normalized 14-D joint delta",
                "- Action horizon/dimension: 50 x 16",
                f"- {sampling_contract}",
                f"- Selection seed: {args.selection_seed}",
                f"- Flow noise seed: {args.noise_seed}",
                f"- Flow noise key mode: {args.noise_key_mode}",
                f"- Flow ODE steps: {args.num_steps}",
                f"- Selection: `{selection_path}`",
                f"- Selection SHA-256: `{report['selection_sha256']}`",
                f"- Prompt source: {report['prompt_source']}",
                *(
                    [
                        f"- Input selection manifest: `{args.selection_manifest}`",
                        f"- Input selection manifest SHA-256: `{report['input_selection_manifest_sha256']}`",
                    ]
                    if args.selection_manifest is not None
                    else []
                ),
                "- Inference: AtomicPi05.sample_actions plus original OpenPI Policy output-transform chain",
                "- Output: model outputs -> v3 Unnormalize -> exact raw state restoration -> AbsoluteActions -> Cr1Outputs",
                "- TCP/FK: not used; this evaluation compares decoded 16-D joint/gripper horizons directly",
                f"- Command: `{command}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
