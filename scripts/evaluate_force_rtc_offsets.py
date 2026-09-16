#!/usr/bin/env python3
"""Evaluate base PI0.5 and force B2 over RTC offsets 0/10/20/30/40.

The base checkpoint samples one ordinary 50-step chunk; its suffix is scored at
every offset.  The B2 checkpoint reuses one slow VLA context and one noise
tensor, clamps the clean GT committed prefix, exposes only force/state samples
available at the requested offset, and resamples the remaining suffix.  Metrics
never include the clamped prefix.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from openpi.models import model as _model

from atomic_latent_vla.pi05.force_training_data import (
    ForceNormalization,
    batch_to_force_inputs,
    build_force_dataset,
    force_collate,
)
from atomic_latent_vla.pi05.weights import AtomicPi05CheckpointLoader

from evaluate_force_b2_episode50_ablation import (
    _anchor_prompt,
    _candidate_indices,
    _choose_episode,
    _config as _force_config,
    _metrics,
)
from evaluate_zm_fk_trajectory_ablation import _output_transform


OFFSETS = (0, 10, 20, 30, 40)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rms(values: list[float]) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


@nnx.jit
def _sample_base(model, observation, noise):
    return model.sample_actions(jax.random.key(0), observation, num_steps=10, noise=noise)


@nnx.jit
def _prepare_force_context(model, observation, slow_force, slow_state, slow_mask):
    return model.prepare_force_policy_context(
        observation,
        slow_force_history=slow_force,
        slow_state_history=slow_state,
        slow_history_mask=slow_mask,
    )


@nnx.jit
def _sample_force_update(
    model,
    context,
    current_force,
    current_state,
    current_mask,
    update_offset,
    noise,
    executed_actions,
):
    return model.sample_actions_force_update(
        jax.random.key(0),
        context,
        current_force_history=current_force,
        current_state_history=current_state,
        current_history_mask=current_mask,
        update_offset=update_offset,
        num_steps=10,
        noise=noise,
        executed_actions=executed_actions,
    )


@nnx.jit
def _sample_force_update_scaled(
    model,
    context,
    current_force,
    current_state,
    current_mask,
    update_offset,
    noise,
    executed_actions,
    force_update_scale,
):
    return model.sample_actions_force_update(
        jax.random.key(0),
        context,
        current_force_history=current_force,
        current_state_history=current_state,
        current_history_mask=current_mask,
        update_offset=update_offset,
        num_steps=10,
        noise=noise,
        executed_actions=executed_actions,
        force_update_scale=force_update_scale,
    )


def _force_metadata_at_offset(raw, norm, indices: list[int], offset: int) -> dict:
    """Read the exact causal fast history for one fixed offset."""

    original = raw.update_offsets
    raw.update_offsets = (offset,)
    try:
        return force_collate([raw.force_metadata(index, norm) for index in indices])
    finally:
        raw.update_offsets = original


def _load_model(kind: str, checkpoint: Path, args: argparse.Namespace):
    config = dataclasses.replace(
        _force_config(args),
        enable_force_stage=kind == "b2",
        force_future_loss_weight=0.0,
    )
    if kind == "base25k":
        params = _model.restore_params(checkpoint / "params", dtype=jnp.bfloat16)
        model = config.load(params)
    else:
        initialized = config.create(jax.random.key(0))
        _, initialized_state = nnx.split(initialized)
        params = AtomicPi05CheckpointLoader(str(checkpoint / "params")).load(
            initialized_state.to_pure_dict()
        )
        del initialized, initialized_state
        model = config.load(params, remove_extra_params=False)
    model.eval()
    return config, model


def _plot_summary(metrics: dict[str, dict], output: Path, label: str) -> None:
    offsets = np.asarray(OFFSETS)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    specs = (
        ("normalized_action_rmse", "Normalized 16D action RMSE"),
        ("joint_rmse_rad", "Decoded 14-joint RMSE (rad)"),
        ("left_tcp_translation_rmse_mm", "Left TCP translation RMSE (mm)"),
        ("right_tcp_translation_rmse_mm", "Right TCP translation RMSE (mm)"),
    )
    for axis, (key, title) in zip(axes.flat, specs, strict=True):
        axis.plot(offsets, [metrics[str(offset)][key] for offset in OFFSETS], "o-")
        axis.set_xticks(offsets)
        axis.set_xlabel("RTC offset")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    figure.suptitle(label)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=("base25k", "b2"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--subtask-sidecar", type=Path)
    parser.add_argument("--pad-subtask-horizon", action="store_true")
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--force-norm", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder-width", type=int, default=512)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--encoder-mlp-dim", type=int, default=1024)
    parser.add_argument("--force-latent-dim", type=int, default=512)
    parser.add_argument("--full-token-force-adapter", action="store_true")
    parser.add_argument("--full-token-force-adapter-heads", type=int, default=2)
    parser.add_argument(
        "--skip-wrong-tokens",
        action="store_true",
        help="Skip the wrong-force control when only the correct RTC trajectories are needed.",
    )
    parser.add_argument(
        "--include-zero-force-update",
        action="store_true",
        help="Also sample the identical RTC suffix with z_exec fixed to the original zM.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = dataclasses.replace(
        _force_config(),
        enable_force_stage=args.model_kind == "b2",
        force_future_loss_weight=0.0,
    )
    dataset = build_force_dataset(
        (args.dataset_root,),
        subtask_sidecars=(args.subtask_sidecar,) if args.subtask_sidecar else None,
        pad_subtask_horizon=args.pad_subtask_horizon,
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        force_norm_path=args.force_norm,
        max_token_len=config.max_token_len,
        force_update_offsets=OFFSETS,
        load_future_force_targets=False,
        seed=0,
    )
    candidates = _candidate_indices(dataset)
    episode = _choose_episode(candidates, args.episode)
    raw = dataset._raw  # noqa: SLF001
    selected = sorted(
        candidates[episode],
        key=lambda index: int(raw.base._frame_index[int(raw.anchors[index])]),  # noqa: SLF001
    )
    if args.max_windows > 0:
        selected = selected[: args.max_windows]
    if not selected:
        raise RuntimeError("selection produced no windows")

    selection = []
    for index in selected:
        anchor = int(raw.anchors[index])
        task_index = int(raw.base._task_index[anchor])  # noqa: SLF001
        selection.append(
            {
                "dataset_index": int(index),
                "anchor_data_index": anchor,
                "episode": episode,
                "frame": int(raw.base._frame_index[anchor]),  # noqa: SLF001
                "prompt": _anchor_prompt(raw, anchor),
            }
        )
    selection_path = args.output_dir / "selection.json"
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    selection_sha = _sha256(selection_path)

    decode = _output_transform(
        args.dataset_root,
        config,
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
    )
    force_norm = ForceNormalization.load(args.force_norm)
    config, model = _load_model(args.model_kind, args.checkpoint, args)

    rows: list[dict] = []
    decoded_predictions: dict[int, list[np.ndarray]] = {offset: [] for offset in OFFSETS}
    decoded_zero_force_predictions: dict[int, list[np.ndarray]] = {
        offset: [] for offset in OFFSETS
    }
    decoded_wrong_force_predictions: dict[int, list[np.ndarray]] = {
        offset: [] for offset in OFFSETS
    }
    decoded_ground_truth: list[np.ndarray] = []
    raw_states: list[np.ndarray] = []
    decoded_frames: list[int] = []
    representative_index = len(selected) // 2
    representative: dict[str, np.ndarray] = {}
    for start in range(0, len(selected), args.batch_size):
        indices = selected[start : start + args.batch_size]
        real_count = len(indices)
        if real_count < args.batch_size:
            indices += [indices[-1]] * (args.batch_size - real_count)
        batch = force_collate([dataset[index] for index in indices])
        observation_np, gt_normalized_full, initial_force = batch_to_force_inputs(batch)
        observation = jax.tree.map(jnp.asarray, observation_np)
        gt_normalized = np.asarray(gt_normalized_full)[..., :16]
        noise = jax.random.normal(
            jax.random.fold_in(jax.random.key(args.seed), start),
            (args.batch_size, config.action_horizon, config.action_dim),
        )

        predictions: dict[int, np.ndarray] = {}
        zero_force_predictions: dict[int, np.ndarray] = {}
        wrong_token_predictions: dict[int, np.ndarray] = {}
        delta_norms: dict[int, np.ndarray] = {}
        if args.model_kind == "base25k":
            prediction = np.asarray(jax.device_get(_sample_base(model, observation, noise)))
            predictions = {offset: prediction for offset in OFFSETS}
            delta_norms = {
                offset: np.full((args.batch_size,), np.nan, dtype=np.float32)
                for offset in OFFSETS
            }
        else:
            context = _prepare_force_context(
                model,
                observation,
                jnp.asarray(initial_force["slow_force_history"]),
                jnp.asarray(initial_force["slow_state_history"]),
                jnp.asarray(initial_force["slow_history_mask"]),
            )
            # Keep observation/prompt/zM/KV/noise correct while sourcing all
            # slow and fast force/state tokens from far-away windows in the
            # same held-out episode. A half-selection cyclic shift avoids the
            # weak adjacent-window control used by the older evaluator.
            if not args.skip_wrong_tokens:
                wrong_indices = [
                    selected[(start + local + len(selected) // 2) % len(selected)]
                    for local in range(real_count)
                ]
                if real_count < args.batch_size:
                    wrong_indices += [wrong_indices[-1]] * (args.batch_size - real_count)
                wrong_batch = force_collate([dataset[index] for index in wrong_indices])
                _, _, wrong_initial_force = batch_to_force_inputs(wrong_batch)
                wrong_context = _prepare_force_context(
                    model,
                    observation,
                    jnp.asarray(wrong_initial_force["slow_force_history"]),
                    jnp.asarray(wrong_initial_force["slow_state_history"]),
                    jnp.asarray(wrong_initial_force["slow_history_mask"]),
                )
            for offset in OFFSETS:
                force = _force_metadata_at_offset(raw, force_norm, indices, offset)
                prediction, modulation = _sample_force_update(
                    model,
                    context,
                    jnp.asarray(force["current_force_history"]),
                    jnp.asarray(force["current_state_history"]),
                    jnp.asarray(force["current_history_mask"]),
                    jnp.asarray(force["update_offset"]),
                    noise,
                    jnp.asarray(gt_normalized),
                )
                predictions[offset] = np.asarray(jax.device_get(prediction))
                delta = np.asarray(jax.device_get(modulation.delta_z))
                delta_norms[offset] = np.mean(np.linalg.norm(delta, axis=-1), axis=-1)
                if args.include_zero_force_update:
                    zero_prediction, _ = _sample_force_update_scaled(
                        model,
                        context,
                        jnp.asarray(force["current_force_history"]),
                        jnp.asarray(force["current_state_history"]),
                        jnp.asarray(force["current_history_mask"]),
                        jnp.asarray(force["update_offset"]),
                        noise,
                        jnp.asarray(gt_normalized),
                        jnp.asarray(0.0, dtype=jnp.float32),
                    )
                    zero_force_predictions[offset] = np.asarray(
                        jax.device_get(zero_prediction)
                    )
                if not args.skip_wrong_tokens:
                    wrong_force = _force_metadata_at_offset(
                        raw, force_norm, wrong_indices, offset
                    )
                    wrong_prediction, _ = _sample_force_update(
                        model,
                        wrong_context,
                        jnp.asarray(wrong_force["current_force_history"]),
                        jnp.asarray(wrong_force["current_state_history"]),
                        jnp.asarray(wrong_force["current_history_mask"]),
                        jnp.asarray(force["update_offset"]),
                        noise,
                        jnp.asarray(gt_normalized),
                    )
                    wrong_token_predictions[offset] = np.asarray(
                        jax.device_get(wrong_prediction)
                    )

        for local, dataset_index in enumerate(indices[:real_count]):
            global_index = start + local
            anchor = int(raw.anchors[dataset_index])
            frame = int(raw.base._frame_index[anchor])  # noqa: SLF001
            task_index = int(raw.base._task_index[anchor])  # noqa: SLF001
            prompt = _anchor_prompt(raw, anchor)
            raw_state = np.asarray(raw.base._states[anchor])  # noqa: SLF001
            state_norm = np.asarray(batch["state"][local])
            gt_norm = gt_normalized[local]
            gt_decoded = np.asarray(decode(state_norm, raw_state, gt_norm)["actions"])
            decoded_ground_truth.append(gt_decoded)
            raw_states.append(raw_state)
            decoded_frames.append(frame)
            metrics_by_offset = {}
            for offset in OFFSETS:
                pred_norm = predictions[offset][local]
                pred_decoded = np.asarray(
                    decode(state_norm, raw_state, pred_norm)["actions"]
                )
                decoded_predictions[offset].append(pred_decoded)
                if args.include_zero_force_update and args.model_kind == "b2":
                    zero_pred_norm = zero_force_predictions[offset][local]
                    zero_pred_decoded = np.asarray(
                        decode(state_norm, raw_state, zero_pred_norm)["actions"]
                    )
                    decoded_zero_force_predictions[offset].append(zero_pred_decoded)
                metrics_by_offset[str(offset)] = _metrics(
                    pred_norm[offset:],
                    gt_norm[offset:],
                    pred_decoded[offset:],
                    gt_decoded[offset:],
                ) | {
                    "delta_z_norm": (
                        None
                        if args.model_kind == "base25k"
                        else float(delta_norms[offset][local])
                    )
                }
                if args.include_zero_force_update and args.model_kind == "b2":
                    zero_metrics = _metrics(
                        zero_pred_norm[offset:],
                        gt_norm[offset:],
                        zero_pred_decoded[offset:],
                        gt_decoded[offset:],
                    )
                    metrics_by_offset[str(offset)].update(
                        {f"no_force_{key}": value for key, value in zero_metrics.items()}
                    )
                if args.model_kind == "b2" and not args.skip_wrong_tokens:
                    wrong_norm = wrong_token_predictions[offset][local]
                    wrong_decoded = np.asarray(
                        decode(state_norm, raw_state, wrong_norm)["actions"]
                    )
                    decoded_wrong_force_predictions[offset].append(wrong_decoded)
                    wrong_metrics = _metrics(
                        wrong_norm[offset:],
                        gt_norm[offset:],
                        wrong_decoded[offset:],
                        gt_decoded[offset:],
                    )
                    metrics_by_offset[str(offset)].update(
                        {f"wrong_tokens_{key}": value for key, value in wrong_metrics.items()}
                    )
                if global_index == representative_index:
                    representative[f"prediction_offset{offset}"] = pred_decoded
            if global_index == representative_index:
                representative["gt_decoded"] = gt_decoded
                representative["frame"] = np.asarray(frame)
            rows.append(
                {
                    "episode": episode,
                    "frame": frame,
                    "dataset_index": int(dataset_index),
                    "prompt": prompt,
                    "metrics_by_offset": metrics_by_offset,
                }
            )

    metric_names = tuple(rows[0]["metrics_by_offset"]["0"])
    offset_metrics = {}
    for offset in OFFSETS:
        key = str(offset)
        offset_metrics[key] = {}
        for metric in metric_names:
            raw_values = [row["metrics_by_offset"][key][metric] for row in rows]
            if metric == "delta_z_norm":
                values = [float(value) for value in raw_values if value is not None]
                offset_metrics[key][metric] = float(np.mean(values)) if values else None
            else:
                offset_metrics[key][metric] = _rms([float(value) for value in raw_values])

    summary = {
        "model_kind": args.model_kind,
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "episode": episode,
        "window_count": len(rows),
        "offsets": list(OFFSETS),
        "suffix_lengths": {str(offset): 50 - offset for offset in OFFSETS},
        "selection_sha256": selection_sha,
        "prompt_contract": (
            "current subtask from aligned sidecar; padded at semantic boundary; no prompt mixing"
            if args.subtask_sidecar and args.pad_subtask_horizon
            else "current subtask from aligned sidecar; unmodified horizon; no prompt mixing"
            if args.subtask_sidecar
            else "PromptFromLeRobotTask using meta/tasks.parquet; no prompt mixing"
        ),
        "base_plan_contract": "one offset0 50-step prediction; score its remaining suffix",
        "b2_contract": "same noise across offsets; clean GT committed prefix; causal fast force/state; score suffix only",
        "wrong_token_contract": (
            "skipped"
            if args.skip_wrong_tokens
            else
            "same correct observation/prompt/zM/KV/noise/offset/GT; all 30 slow and every valid fast force+state content token come from a half-episode-shifted window"
            if args.model_kind == "b2"
            else "not applicable"
        ),
        "norm_stats_sha256": _sha256(
            args.norm_assets_dir / args.norm_asset_id / "norm_stats.json"
        ),
        "force_norm_sha256": _sha256(args.force_norm),
        "seed": args.seed,
        "sampler_steps": 10,
        "offset_metrics": offset_metrics,
        "windows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(args.output_dir / "representative_predictions.npz", **representative)
    np.savez_compressed(
        args.output_dir / "decoded_predictions_all_windows.npz",
        **{
            **{
                f"prediction_offset{offset}": np.stack(decoded_predictions[offset])
                for offset in OFFSETS
            },
            **(
                {
                    f"zero_force_prediction_offset{offset}": np.stack(
                        decoded_zero_force_predictions[offset]
                    )
                    for offset in OFFSETS
                }
                if args.include_zero_force_update and args.model_kind == "b2"
                else {}
            ),
            **(
                {
                    f"wrong_force_prediction_offset{offset}": np.stack(
                        decoded_wrong_force_predictions[offset]
                    )
                    for offset in OFFSETS
                }
                if args.model_kind == "b2" and not args.skip_wrong_tokens
                else {}
            ),
            "gt_decoded": np.stack(decoded_ground_truth),
            "raw_state": np.stack(raw_states),
            "frame": np.asarray(decoded_frames, dtype=np.int32),
        },
    )

    with (args.output_dir / "offset_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = ["offset", "suffix_steps", *metric_names]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for offset in OFFSETS:
            writer.writerow(
                {
                    "offset": offset,
                    "suffix_steps": 50 - offset,
                    **offset_metrics[str(offset)],
                }
            )
    _plot_summary(
        offset_metrics,
        args.output_dir / "offset_metrics.png",
        f"{args.model_kind} — episode {episode}",
    )
    (args.output_dir / "run_contract.md").write_text(
        "# RTC offset GT evaluation contract\n\n"
        f"- model: `{args.model_kind}`; checkpoint: `{args.checkpoint.resolve()}`\n"
        f"- dataset: `{args.dataset_root.resolve()}`; episode: {episode}\n"
        f"- selection: `{selection_path.resolve()}`; sha256: `{selection_sha}`\n"
        f"- offsets: {OFFSETS}; suffix-only metrics; TCP offset: 0.20 m\n"
        f"- prompt: {'current aligned subtask' if args.subtask_sidecar else 'native task text'}; "
        f"subtask padding: {args.pad_subtask_horizon}; seed: {args.seed}; sampler steps: 10\n"
        f"- exact command: `{' '.join(sys.argv)}`\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in summary.items() if key not in {"windows"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
