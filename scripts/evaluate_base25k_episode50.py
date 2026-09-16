#!/usr/bin/env python3
"""Evaluate the exact force-free 25k parent on a fixed force-domain episode."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from openpi.models import model as _model

from atomic_latent_vla.pi05.force_training_data import (
    batch_to_force_inputs,
    build_force_dataset,
    force_collate,
)

from evaluate_force_b2_episode50_ablation import (
    _candidate_indices,
    _choose_episode,
    _config as _force_config,
    _metrics,
)
from evaluate_zm_fk_trajectory_ablation import _output_transform


@nnx.jit
def _sample(model, observation, noise):
    return model.sample_actions(jax.random.key(0), observation, num_steps=10, noise=noise)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--force-norm", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = dataclasses.replace(_force_config(), enable_force_stage=False)
    dataset = build_force_dataset(
        (args.dataset_root,),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        force_norm_path=args.force_norm,
        max_token_len=config.max_token_len,
        seed=0,
    )
    candidates = _candidate_indices(dataset)
    episode = _choose_episode(candidates, args.episode)
    selected = sorted(
        candidates[episode],
        key=lambda index: int(dataset._raw.base._frame_index[int(dataset._raw.anchors[index])]),  # noqa: SLF001
    )
    raw = dataset._raw  # noqa: SLF001
    decode = _output_transform(
        args.dataset_root,
        config,
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
    )
    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()

    rows: list[dict] = []
    for start in range(0, len(selected), args.batch_size):
        indices = selected[start : start + args.batch_size]
        real_count = len(indices)
        if real_count < args.batch_size:
            indices += [indices[-1]] * (args.batch_size - real_count)
        batch = force_collate([dataset[index] for index in indices])
        observation_np, gt_normalized, _ = batch_to_force_inputs(batch)
        observation = jax.tree.map(jnp.asarray, observation_np)
        noise = jax.random.normal(
            jax.random.fold_in(jax.random.key(args.seed), start),
            (args.batch_size, config.action_horizon, config.action_dim),
        )
        prediction = np.asarray(jax.device_get(_sample(model, observation, noise)))[:real_count]
        for local, dataset_index in enumerate(indices[:real_count]):
            anchor = int(raw.anchors[dataset_index])
            frame = int(raw.base._frame_index[anchor])  # noqa: SLF001
            task_index = int(raw.base._task_index[anchor])  # noqa: SLF001
            prompt = str(raw.base.tasks[task_index])
            raw_state = np.asarray(raw.base._states[anchor])  # noqa: SLF001
            gt_norm = np.asarray(gt_normalized[local])[..., :16]
            state_norm = np.asarray(batch["state"][local])
            gt_decoded = np.asarray(decode(state_norm, raw_state, gt_norm)["actions"])
            pred_decoded = np.asarray(decode(state_norm, raw_state, prediction[local])["actions"])
            rows.append(
                {
                    "episode": episode,
                    "frame": frame,
                    "dataset_index": int(dataset_index),
                    "prompt": prompt,
                    "gt_normalized": gt_norm.tolist(),
                    "prediction_normalized": prediction[local].tolist(),
                    "gt_decoded": gt_decoded.tolist(),
                    "prediction_decoded": pred_decoded.tolist(),
                    "metrics": _metrics(prediction[local], gt_norm, pred_decoded, gt_decoded),
                }
            )

    norm_pred = np.concatenate([np.asarray(row["prediction_normalized"]) for row in rows])
    norm_gt = np.concatenate([np.asarray(row["gt_normalized"]) for row in rows])
    pred = np.concatenate([np.asarray(row["prediction_decoded"]) for row in rows])
    gt = np.concatenate([np.asarray(row["gt_decoded"]) for row in rows])
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "episode": episode,
        "window_count": len(rows),
        "covered_gt_steps": len(rows) * 50,
        "prompt_contract": "PromptFromLeRobotTask using meta/tasks.parquet; no prompt mixing",
        "input_contract": "force-free Atomic PI0.5: images + state + task prompt",
        "noise_contract": f"seed={args.seed}; same batch-size/start fold-in as paired B2 evaluation",
        "metrics": _metrics(norm_pred, norm_gt, pred, gt),
        "windows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "run_contract.md").write_text(
        "# Evaluation contract\n\n"
        f"- checkpoint: `{args.checkpoint.resolve()}`\n"
        f"- dataset: `{args.dataset_root.resolve()}`\n"
        f"- base norm: `{(args.norm_assets_dir / args.norm_asset_id / 'norm_stats.json').resolve()}`\n"
        f"- episode: {episode}; every genuine frame%50==0 anchor\n"
        "- input: images + state + native task prompt; no force branch\n"
        f"- seed: {args.seed}; batch size: {args.batch_size}; sampler steps: 10\n"
        "- output: normalized 16D, decoded joints/grippers, CR1 FK with 0.20 m TCP\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "windows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
