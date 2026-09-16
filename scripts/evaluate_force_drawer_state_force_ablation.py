#!/usr/bin/env python3
"""Disentangle state and wrench influence in the drawer closed/open conflict.

The target anchor is always the reviewed open-drawer frame and all branches use
the reviewed closed-drawer images.  Prompt, image masks, diffusion noise and
sampler settings stay paired.  Branches independently replace the main PI
state, the force-side 120 Hz state history, and the slow wrench history.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from atomic_latent_vla.pi05.force_training_data import (
    ForceNormalization,
    batch_to_force_inputs,
    build_force_dataset,
    force_collate,
)
from atomic_latent_vla.pi05.weights import AtomicPi05CheckpointLoader

from evaluate_force_b2_episode50_ablation import _prepare_context, _sample_offset0
from evaluate_force_drawer_visual_success import _selection
from evaluate_force_image_misalignment import _config, _cosine, _rmse


BRANCHES = (
    "open_state_open_force",
    "open_state_closed_force",
    "closed_state_open_force",
    "closed_state_closed_force",
    "main_state_closed_only",
    "force_state_closed_only",
    "raw_force_zero",
    "normalized_force_zero",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_metrics(prediction: np.ndarray, baseline: np.ndarray, target: np.ndarray) -> dict:
    return {
        "rmse_vs_baseline": _rmse(prediction, baseline),
        "cosine_vs_baseline": _cosine(prediction, baseline),
        "rmse_to_open_gt": _rmse(prediction, target),
        "cosine_to_open_gt": _cosine(prediction, target),
    }


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
    force_norm = ForceNormalization.load(args.force_norm)
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
    selection = [
        row for row in _selection(dataset, pair_manifest) if row["state_condition"] == "open"
    ]
    raw = dataset._raw  # noqa: SLF001
    for row in selection:
        anchor = int(raw.anchors[row["target_index"]])
        task_index = int(raw.base._task_index[anchor])  # noqa: SLF001
        row["prompt"] = str(raw.base.tasks[task_index])
        row["anchor_data_index"] = anchor
    selection_path = args.output_dir / "selection.json"
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")

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
    raw_zero_normalized = force_norm.normalize_force(np.zeros((1, 1, 6), dtype=np.float32))[0, 0]
    for start in range(0, len(selection), args.batch_size):
        selected = selection[start : start + args.batch_size]
        real_count = len(selected)
        if real_count < args.batch_size:
            selected += [selected[-1]] * (args.batch_size - real_count)

        open_batch = force_collate([dataset[row["target_index"]] for row in selected])
        closed_batch = force_collate([dataset[row["closed_image_index"]] for row in selected])
        open_obs_np, open_gt, open_force = batch_to_force_inputs(open_batch)
        closed_obs_np, _, closed_force = batch_to_force_inputs(closed_batch)
        open_obs = jax.tree.map(jnp.asarray, open_obs_np)
        closed_obs = jax.tree.map(jnp.asarray, closed_obs_np)
        # Every branch sees the closed images. PI0.5 pads the 16 physical
        # coordinates to 32-D and discretizes that full state into the prompt,
        # so a valid
        # main-state intervention must use the complete closed observation
        # (same global prompt here), not merely Observation.replace(state=...).
        obs_open_state = open_obs.replace(images=closed_obs.images)
        obs_closed_state = closed_obs

        open_slow_force = jnp.asarray(open_force["slow_force_history"])
        closed_slow_force = jnp.asarray(closed_force["slow_force_history"])
        open_slow_state = jnp.asarray(open_force["slow_state_history"])
        closed_slow_state = jnp.asarray(closed_force["slow_state_history"])
        slow_mask = jnp.asarray(open_force["slow_history_mask"])
        current_force = jnp.asarray(open_force["current_force_history"])
        current_state = jnp.asarray(open_force["current_state_history"])
        current_mask = jnp.zeros_like(jnp.asarray(open_force["current_history_mask"]))
        raw_force_zero = jnp.broadcast_to(jnp.asarray(raw_zero_normalized), open_slow_force.shape)
        normalized_force_zero = jnp.zeros_like(open_slow_force)
        noise = jax.random.normal(
            jax.random.fold_in(jax.random.key(args.seed), start),
            (args.batch_size, config.action_horizon, config.action_dim),
        )

        specs = {
            "open_state_open_force": (obs_open_state, open_slow_state, open_slow_force),
            "open_state_closed_force": (obs_open_state, open_slow_state, closed_slow_force),
            "closed_state_open_force": (obs_closed_state, closed_slow_state, open_slow_force),
            "closed_state_closed_force": (obs_closed_state, closed_slow_state, closed_slow_force),
            "main_state_closed_only": (obs_closed_state, open_slow_state, open_slow_force),
            "force_state_closed_only": (obs_open_state, closed_slow_state, open_slow_force),
            "raw_force_zero": (obs_open_state, open_slow_state, raw_force_zero),
            "normalized_force_zero": (obs_open_state, open_slow_state, normalized_force_zero),
        }
        predictions: dict[str, np.ndarray] = {}
        delta_z: dict[str, np.ndarray] = {}
        for name, (observation, slow_state, slow_force) in specs.items():
            context = _prepare_context(
                model, observation, slow_force, slow_state, slow_mask
            )
            prediction, modulation = _sample_offset0(
                model,
                context,
                current_force,
                current_state,
                current_mask,
                noise,
                jnp.asarray(True),
            )
            predictions[name] = np.asarray(jax.device_get(prediction))[:real_count]
            delta_z[name] = np.asarray(jax.device_get(modulation))[:real_count]

        for local, selected_row in enumerate(selected[:real_count]):
            baseline = predictions["open_state_open_force"][local]
            gt = np.asarray(open_gt[local])[..., :16]
            row = dict(selected_row)
            row["branches"] = {}
            for name in BRANCHES:
                metrics = _prediction_metrics(predictions[name][local], baseline, gt)
                metrics["delta_z_rmse_vs_baseline"] = _rmse(
                    delta_z[name][local], delta_z["open_state_open_force"][local]
                )
                row["branches"][name] = metrics
            rows.append(row)

    aggregate: dict[str, dict] = {}
    for name in BRANCHES:
        aggregate[name] = {
            key: float(np.mean([row["branches"][name][key] for row in rows]))
            for key in (
                "rmse_vs_baseline",
                "cosine_vs_baseline",
                "rmse_to_open_gt",
                "cosine_to_open_gt",
                "delta_z_rmse_vs_baseline",
            )
        }
    state_effect = aggregate["closed_state_open_force"]["rmse_vs_baseline"]
    force_effect = aggregate["open_state_closed_force"]["rmse_vs_baseline"]
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "window_count": len(rows),
        "contract": (
            "open target anchor and closed images for all branches; paired prompt/masks/noise/sampler; "
            "state and slow wrench histories independently replaced with reviewed closed-frame values"
        ),
        "offset": 0,
        "raw_force_zero_normalized_value": raw_zero_normalized.tolist(),
        "aggregate": aggregate,
        "matched_closed_state_effect_rmse": state_effect,
        "matched_closed_force_effect_rmse": force_effect,
        "state_to_force_effect_ratio": float(state_effect / max(force_effect, 1.0e-12)),
    }
    serializable_rows = []
    for row in rows:
        serializable_rows.append(row)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary | {"windows": serializable_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (args.output_dir / "windows.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["episode", "target_frame", "closed_image_frame", "prompt"]
        for name in BRANCHES:
            fields += [
                f"{name}_rmse_vs_baseline",
                f"{name}_rmse_to_open_gt",
                f"{name}_delta_z_rmse_vs_baseline",
            ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in ("episode", "target_frame", "closed_image_frame", "prompt")}
            for name in BRANCHES:
                flat[f"{name}_rmse_vs_baseline"] = row["branches"][name]["rmse_vs_baseline"]
                flat[f"{name}_rmse_to_open_gt"] = row["branches"][name]["rmse_to_open_gt"]
                flat[f"{name}_delta_z_rmse_vs_baseline"] = row["branches"][name]["delta_z_rmse_vs_baseline"]
            writer.writerow(flat)

    base_norm = args.norm_assets_dir / args.norm_asset_id / "norm_stats.json"
    contract = (
        "# Evaluation contract\n\n"
        f"- checkpoint: `{args.checkpoint.resolve()}`\n"
        f"- checkpoint manifest sha256: `{_sha256(args.checkpoint / 'params' / 'manifest.ocdbt')}`\n"
        f"- dataset: `{args.dataset_root.resolve()}`\n"
        f"- base norm: `{base_norm.resolve()}` sha256 `{_sha256(base_norm)}`\n"
        f"- force norm: `{args.force_norm.resolve()}` sha256 `{_sha256(args.force_norm)}`\n"
        f"- pair manifest: `{args.pair_manifest.resolve()}` sha256 `{_sha256(args.pair_manifest)}`\n"
        f"- selection: `{selection_path.resolve()}` sha256 `{_sha256(selection_path)}`\n"
        f"- target anchors: {len(rows)}; seed: {args.seed}; horizon: 50; offset: 0\n"
        "- fixed: closed images, open-anchor prompt/GT, image masks, noise and sampler\n"
        "- ablated independently: main state, force-side 120 Hz state, slow wrench history\n"
        "- raw physical force zero is normalized with the recorded force q01/q99; normalized-zero is reported separately\n"
    )
    (args.output_dir / "run_contract.md").write_text(contract, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
