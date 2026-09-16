#!/usr/bin/env python3
"""Training-faithful Fruit zM adapter ablation on fixed subtask horizons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
from openpi.models import model as _model

from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import (
    _paligemma_tokenizer,
    atomic_collate,
    batch_to_observation,
    build_atomic_dataset,
)
from evaluate_force_b2_episode50_ablation import _metrics, _output_transform


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _stack(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, dict):
        return {key: _stack([value[key] for value in values]) for key in first}
    return np.stack([np.asarray(value) for value in values])


def _resolve_selection(dataset, requested: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw = dataset._raw  # noqa: SLF001
    wanted = {(int(row["episode"]), int(row["frame"])): row for row in requested}
    resolved: dict[tuple[int, int], dict[str, Any]] = {}
    visible = getattr(raw.base, "_visible_indices", np.arange(len(raw.base)))
    for dataset_index, data_index in enumerate(visible):
        data_index = int(data_index)
        key = (
            int(raw.base._episode_index[data_index]),  # noqa: SLF001
            int(raw.base._frame_index[data_index]),  # noqa: SLF001
        )
        if key not in wanted:
            continue
        metadata = raw.metadata(dataset_index)
        resolved[key] = {
            **wanted[key],
            "dataset_index": dataset_index,
            "data_index": data_index,
            "subtask_prompt": str(
                wanted[key].get("prompt") or metadata["subtask_prompt"]
            ),
            "subtask_boundary_padded": bool(metadata["subtask_boundary_padded"]),
        }
    missing = [key for key in wanted if key not in resolved]
    if missing:
        raise ValueError(f"selection rows absent from dataset: {missing}")
    return [resolved[(int(row["episode"]), int(row["frame"]))] for row in requested]


@nnx.jit
def _sample_pair(model, observation, noise):
    """Same prefix/noise; normal layerwise zM versus no latent FiLM residual."""

    observation = _model.preprocess_observation(None, observation, train=False)
    query_hidden, prefix_mask, kv_cache, _, layerwise_latents = model._prefix_forward(  # noqa: SLF001
        observation,
        return_layerwise_latents=True,
    )
    active_state = model._controlled_state(observation.state)  # noqa: SLF001
    _, _, z_model, _, _ = model._latent(query_hidden, active_state)  # noqa: SLF001
    initial = model._mask_action_condition(noise)  # noqa: SLF001

    def integrate(use_zm: bool):
        def step(index, actions):
            time = jnp.asarray(1.0 - index / 10.0, dtype=actions.dtype)
            velocity = model._suffix_velocity(  # noqa: SLF001
                prefix_mask,
                kv_cache,
                actions,
                jnp.broadcast_to(time, (actions.shape[0],)),
                z_model if use_zm else None,
                layerwise_latents=layerwise_latents if use_zm else None,
            )
            return model._mask_action_condition(actions - 0.1 * velocity)  # noqa: SLF001

        return jax.lax.fori_loop(0, 10, step, initial)[
            ..., : model.config.active_action_dim
        ]

    return integrate(True), integrate(False), z_model, layerwise_latents


@nnx.jit
def _sample_pair_final_zm(model, observation, noise):
    """Legacy route: one final zM is reused by every Action-Expert block."""

    observation = _model.preprocess_observation(None, observation, train=False)
    query_hidden, prefix_mask, kv_cache, _ = model._prefix_forward(observation)  # noqa: SLF001
    active_state = model._controlled_state(observation.state)  # noqa: SLF001
    _, _, z_model, _, _ = model._latent(query_hidden, active_state)  # noqa: SLF001
    initial = model._mask_action_condition(noise)  # noqa: SLF001

    def integrate(use_zm: bool):
        def step(index, actions):
            time = jnp.asarray(1.0 - index / 10.0, dtype=actions.dtype)
            velocity = model._suffix_velocity(  # noqa: SLF001
                prefix_mask,
                kv_cache,
                actions,
                jnp.broadcast_to(time, (actions.shape[0],)),
                z_model if use_zm else None,
                layerwise_latents=None,
            )
            return model._mask_action_condition(actions - 0.1 * velocity)  # noqa: SLF001

        return jax.lax.fori_loop(0, 10, step, initial)[
            ..., : model.config.active_action_dim
        ]

    return integrate(True), integrate(False), z_model


def _image_uint8(value: np.ndarray) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        if float(np.nanmin(image)) < -0.01:
            image = (image + 1.0) * 127.5
        elif float(np.nanmax(image)) <= 1.01:
            image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _save_contact_sheet(observation, selection: list[dict[str, Any]], output: Path) -> None:
    cameras = list(observation.images)
    figure, axes = plt.subplots(
        len(selection), len(cameras), figsize=(4.2 * len(cameras), 3.1 * len(selection)),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(len(selection), len(cameras))
    for row_index, selected in enumerate(selection):
        for camera_index, camera in enumerate(cameras):
            axis = axes[row_index, camera_index]
            axis.imshow(_image_uint8(np.asarray(observation.images[camera])[row_index]))
            axis.axis("off")
            if row_index == 0:
                axis.set_title(camera)
            if camera_index == 0:
                axis.text(
                    0.01,
                    0.01,
                    f"ep{selected['episode']} f{selected['frame']}\n{selected['subtask_prompt']}",
                    transform=axis.transAxes,
                    fontsize=7,
                    color="white",
                    va="bottom",
                    bbox={"facecolor": "black", "alpha": 0.65, "pad": 2},
                )
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _plot_summary(cases: list[dict[str, Any]], output: Path, *, layerwise: bool) -> None:
    labels = [f"ep{row['episode']} f{row['frame']}" for row in cases]
    x = np.arange(len(cases))
    normal = [row["normal_to_gt"]["normalized_action_rmse"] for row in cases]
    no_zm = [row["no_zm_to_gt"]["normalized_action_rmse"] for row in cases]
    adapter = [row["no_zm_vs_normal"]["normalized_action_rmse"] for row in cases]
    width = 0.25
    figure, axis = plt.subplots(figsize=(13, 5.8), constrained_layout=True)
    route = "layerwise zM" if layerwise else "legacy final zM"
    axis.bar(x - width, normal, width, label=f"normal {route} → GT", color="#2563eb")
    axis.bar(x, no_zm, width, label="no zM adapter → GT", color="#f97316")
    axis.bar(x + width, adapter, width, label="no zM ↔ normal", color="#7c3aed")
    axis.set_xticks(x, labels, rotation=22, ha="right")
    axis.set_ylabel("normalized 16D action RMSE")
    axis.set_title(f"Fruit native-subtask horizons: {route} adapter ablation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--atomic-composition-sidecar", default="fk_horizon_3hz_gate_top5_stay_v2")
    parser.add_argument(
        "--layerwise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use per-layer zM conditions; --no-layerwise reproduces the legacy final-zM route.",
    )
    parser.add_argument(
        "--pad-subtask-horizon",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--coefficient-target-kind", default="joint_delta")
    parser.add_argument("--coefficient-target-dim", type=int, default=14)
    parser.add_argument(
        "--quantile-clip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Clip normalized state/action to [-1,1], matching the legacy yc PI0.5 fork.",
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--prompt-source-sidecar",
        type=Path,
        help="Authoritative historical sidecar used to create prompt overrides in selection.",
    )
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    norm_stats = args.norm_assets_dir / args.norm_asset_id / "norm_stats.json"
    sidecar = args.dataset_root / "meta" / "episode_subtasks.jsonl"
    for path in (norm_stats, sidecar, args.selection, args.checkpoint / "params" / "_METADATA"):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.prompt_source_sidecar is not None and not args.prompt_source_sidecar.is_file():
        raise FileNotFoundError(args.prompt_source_sidecar)

    config = AtomicPi05Config(
        max_token_len=200,
        fast_action_ce_loss_weight=0.0,
        coefficient_target_kind=args.coefficient_target_kind,
        coefficient_target_dim=args.coefficient_target_dim,
        enable_layerwise_atomic_flow=args.layerwise,
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
    requested = json.loads(args.selection.read_text(encoding="utf-8"))["rows"]
    selection = _resolve_selection(dataset, requested)
    rows = [dataset[int(row["dataset_index"])] for row in selection]
    if args.quantile_clip:
        tokenizer = _paligemma_tokenizer(config.max_token_len)
        clipped_rows = []
        for selected, row in zip(selection, rows, strict=True):
            clipped = dict(row)
            clipped["state"] = np.clip(
                np.asarray(row["state"]), -1.0, 1.0
            ).astype(np.float32)
            clipped["actions"] = np.clip(
                np.asarray(row["actions"]), -1.0, 1.0
            ).astype(np.float32)
            tokens, mask = tokenizer.tokenize(
                selected["subtask_prompt"], clipped["state"]
            )
            clipped["subtask_prompt_tokens"] = tokens
            clipped["subtask_prompt_mask"] = mask
            clipped_rows.append(clipped)
        rows = clipped_rows
    batch = atomic_collate(rows)
    observation_np, gt_normalized = batch_to_observation(batch)
    observation_np = _model.Observation(
        images=observation_np.images,
        image_masks=observation_np.image_masks,
        state=observation_np.state,
        tokenized_prompt=batch["subtask_prompt_tokens"],
        tokenized_prompt_mask=batch["subtask_prompt_mask"],
        token_ar_mask=observation_np.token_ar_mask,
        token_loss_mask=observation_np.token_loss_mask,
    )
    _save_contact_sheet(
        observation_np,
        selection,
        args.output_dir / "source_observations.png",
    )
    observation = jax.tree.map(jnp.asarray, observation_np)
    noise = jax.random.normal(
        jax.random.key(args.seed),
        (len(rows), config.action_horizon, config.action_dim),
    )

    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    if args.layerwise:
        normal, no_zm, z_model, layerwise = jax.device_get(
            _sample_pair(model, observation, noise)
        )
        layerwise = np.asarray(layerwise)
    else:
        normal, no_zm, z_model = jax.device_get(
            _sample_pair_final_zm(model, observation, noise)
        )
        layerwise = np.empty((0,), dtype=np.float32)
    normal = np.asarray(normal)[..., :16]
    no_zm = np.asarray(no_zm)[..., :16]
    gt_normalized = np.asarray(gt_normalized)[..., :16]
    decode = _output_transform(
        args.dataset_root,
        config,
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
    )

    cases: list[dict[str, Any]] = []
    normal_decoded, no_zm_decoded, gt_decoded = [], [], []
    for index, selected in enumerate(selection):
        metadata = dataset._raw.metadata(int(selected["dataset_index"]))  # noqa: SLF001
        state_norm = np.asarray(batch["state"][index])
        raw_state = np.asarray(metadata["raw_state"])
        decoded_gt = np.asarray(metadata["raw_actions"])[..., :16]
        decoded_normal = np.asarray(decode(state_norm, raw_state, normal[index])["actions"])[..., :16]
        decoded_no_zm = np.asarray(decode(state_norm, raw_state, no_zm[index])["actions"])[..., :16]
        normal_to_gt = _metrics(normal[index], gt_normalized[index], decoded_normal, decoded_gt)
        no_zm_to_gt = _metrics(no_zm[index], gt_normalized[index], decoded_no_zm, decoded_gt)
        no_zm_vs_normal = _metrics(no_zm[index], normal[index], decoded_no_zm, decoded_normal)
        cases.append(
            {
                **selected,
                "normal_to_gt": normal_to_gt,
                "no_zm_to_gt": no_zm_to_gt,
                "no_zm_vs_normal": no_zm_vs_normal,
                "gt_rmse_delta_no_zm_minus_normal": (
                    no_zm_to_gt["normalized_action_rmse"]
                    - normal_to_gt["normalized_action_rmse"]
                ),
            }
        )
        normal_decoded.append(decoded_normal)
        no_zm_decoded.append(decoded_no_zm)
        gt_decoded.append(decoded_gt)

    def aggregate(method: str) -> dict[str, Any]:
        pred_norm = normal if method == "normal" else no_zm
        pred_decoded = np.stack(normal_decoded if method == "normal" else no_zm_decoded)
        return _metrics(
            pred_norm.reshape(-1, 16),
            gt_normalized.reshape(-1, 16),
            pred_decoded.reshape(-1, 16),
            np.stack(gt_decoded).reshape(-1, 16),
        )

    normal_aggregate = aggregate("normal")
    no_zm_aggregate = aggregate("no_zm")
    adapter_aggregate = _metrics(
        no_zm.reshape(-1, 16),
        normal.reshape(-1, 16),
        np.stack(no_zm_decoded).reshape(-1, 16),
        np.stack(normal_decoded).reshape(-1, 16),
    )
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "selection_count": len(selection),
        "prompt_contract": (
            "explicit native subtask overrides from historical training sidecar; 144-token tokenizer"
            if args.prompt_source_sidecar is not None
            else "native active subtask only; same 144-token tokenizer as training"
        ),
        "horizon_contract": (
            "50 steps; active subtask repeated/held across boundary"
            if args.pad_subtask_horizon
            else "50 steps; legacy cross-boundary subtask composition"
        ),
        "latent_route": "layerwise_zm" if args.layerwise else "legacy_final_zm_reused_at_all_layers",
        "quantile_clip": args.quantile_clip,
        "ablation": "same images/state/subtask/noise/prefix/KV; pass z_model=None and layerwise_latents=None only in suffix",
        "normal_to_gt": normal_aggregate,
        "no_zm_to_gt": no_zm_aggregate,
        "no_zm_vs_normal": adapter_aggregate,
        "normal_better_fraction": float(
            np.mean([row["gt_rmse_delta_no_zm_minus_normal"] > 0 for row in cases])
        ),
        "mean_gt_rmse_delta_no_zm_minus_normal": float(
            np.mean([row["gt_rmse_delta_no_zm_minus_normal"] for row in cases])
        ),
        "mean_zm_norm": float(np.mean(np.linalg.norm(np.asarray(z_model), axis=-1))),
        "mean_layerwise_zm_norm": (
            float(np.mean(np.linalg.norm(layerwise, axis=-1)))
            if args.layerwise
            else None
        ),
        "norm_stats_sha256": _sha256(norm_stats),
        "subtask_sidecar_sha256": _sha256(sidecar),
        "prompt_source_sidecar": (
            str(args.prompt_source_sidecar.resolve())
            if args.prompt_source_sidecar is not None
            else None
        ),
        "prompt_source_sidecar_sha256": (
            _sha256(args.prompt_source_sidecar)
            if args.prompt_source_sidecar is not None
            else None
        ),
        "selection_sha256": _sha256(args.selection),
        "noise_sha256": _array_sha256(np.asarray(jax.device_get(noise))),
        "seed": args.seed,
        "cases": cases,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "selection.json").write_text(
        json.dumps({"rows": selection}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output_dir / "raw_predictions.npz",
        gt_normalized=gt_normalized,
        normal_normalized=normal,
        no_zm_normalized=no_zm,
        gt_decoded=np.stack(gt_decoded),
        normal_decoded=np.stack(normal_decoded),
        no_zm_decoded=np.stack(no_zm_decoded),
        z_model=np.asarray(z_model),
        layerwise_zm=layerwise,
        noise=np.asarray(jax.device_get(noise)),
    )
    with (args.output_dir / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "episode", "frame", "subtask_prompt", "subtask_boundary_padded",
            "normal_gt_normalized_rmse", "no_zm_gt_normalized_rmse",
            "no_zm_vs_normal_normalized_rmse", "gt_delta_no_zm_minus_normal",
            "no_zm_vs_normal_joint_rmse_rad", "no_zm_vs_normal_left_tcp_mm",
            "no_zm_vs_normal_right_tcp_mm",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in cases:
            writer.writerow(
                {
                    "episode": row["episode"],
                    "frame": row["frame"],
                    "subtask_prompt": row["subtask_prompt"],
                    "subtask_boundary_padded": row["subtask_boundary_padded"],
                    "normal_gt_normalized_rmse": row["normal_to_gt"]["normalized_action_rmse"],
                    "no_zm_gt_normalized_rmse": row["no_zm_to_gt"]["normalized_action_rmse"],
                    "no_zm_vs_normal_normalized_rmse": row["no_zm_vs_normal"]["normalized_action_rmse"],
                    "gt_delta_no_zm_minus_normal": row["gt_rmse_delta_no_zm_minus_normal"],
                    "no_zm_vs_normal_joint_rmse_rad": row["no_zm_vs_normal"]["joint_rmse_rad"],
                    "no_zm_vs_normal_left_tcp_mm": row["no_zm_vs_normal"]["left_tcp_translation_rmse_mm"],
                    "no_zm_vs_normal_right_tcp_mm": row["no_zm_vs_normal"]["right_tcp_translation_rmse_mm"],
                }
            )
    _plot_summary(
        cases,
        args.output_dir / "fruit_zm_adapter_ablation.png",
        layerwise=args.layerwise,
    )
    run_contract = f"""# Fruit zM adapter ablation contract

- repository: {Path(__file__).resolve().parents[1]}
- checkpoint: {args.checkpoint.resolve()}
- dataset: {args.dataset_root.resolve()}
- norm: {args.norm_asset_id} (`{_sha256(norm_stats)}`)
- subtask sidecar SHA256: `{_sha256(sidecar)}`
- authoritative prompt source: {args.prompt_source_sidecar or sidecar}
- authoritative prompt source SHA256: `{_sha256(args.prompt_source_sidecar or sidecar)}`
- prompt: native active subtask, max token length 200
- horizon: 50; `pad_subtask_horizon={args.pad_subtask_horizon}`
- input/target quantile clipping: {args.quantile_clip}
- output: q01/q99 inverse transform
- TCP local z offset: 0.20 m
- seed/noise: {args.seed} / `{_array_sha256(np.asarray(jax.device_get(noise)))}`
- latent route: {"layerwise zM" if args.layerwise else "legacy one final zM reused at every layer"}
- ablation: normal zM versus suffix with both `z_model=None` and `layerwise_latents=None`
- paired controls: identical cameras, normalized state, prompt tokens, prefix/KV cache, and initial flow noise
"""
    (args.output_dir / "run_contract.md").write_text(run_contract, encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
