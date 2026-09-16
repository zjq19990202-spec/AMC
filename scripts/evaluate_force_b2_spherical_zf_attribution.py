#!/usr/bin/env python3
"""Training-faithful zF attribution for the spherical full-token B2 policy.

The paired branches keep observation, prompt, zM, fast history, diffusion
noise, and committed GT prefix fixed.  Only the 30 slow-history memory tokens
that constitute zF for the full-token adapter are changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.force_training_data import (
    ForceNormalization,
    batch_to_force_inputs,
    build_force_dataset,
    force_collate,
)
from atomic_latent_vla.pi05.weights import AtomicPi05CheckpointLoader
from evaluate_force_b2_episode50_ablation import _candidate_indices, _metrics
from evaluate_force_rtc_offsets import (
    _force_metadata_at_offset,
    _prepare_force_context,
    _sample_force_update,
)
from evaluate_zm_fk_trajectory_ablation import _output_transform


OFFSETS = (0, 10, 20, 30, 40)
BRANCHES = ("full", "zero_zf_content", "wrong_zf", "no_fast")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config() -> AtomicPi05Config:
    """Recreate the exact queued spherical45/h20 full-token B2 graph."""

    return AtomicPi05Config(
        max_token_len=192,
        coefficient_target_kind="joint_delta",
        coefficient_target_dim=14,
        fast_action_ce_loss_weight=0.0,
        enable_force_stage=True,
        force_fast_history_samples=40,
        force_update_action_steps=10,
        force_future_decoder_stride=4,
        force_future_decoder_kind="phase_mlp",
        force_encoder_width=512,
        force_encoder_depth=2,
        force_encoder_num_heads=8,
        force_encoder_mlp_dim=1024,
        force_latent_dim=512,
        force_context_from_prefix=False,
        force_future_condition_on_zm=True,
        force_position_base=10_000.0,
        force_history_train_lengths=(120,),
        force_future_loss_weight=0.0,
        force_flow_loss_weight=1.0,
        force_delta_regularization_weight=1.0e-4,
        force_improvement_loss_weight=1.0,
        force_improvement_margin=0.001,
        enable_layerwise_atomic_flow=False,
        force_full_token_adapter=True,
        force_full_token_adapter_heads=2,
        spherical_visual_latent=True,
        visual_max_update_angle_deg=45.0,
        spherical_force_update=True,
        force_max_update_angle_deg=45.0,
        force_rotation_loss_weight=0.005,
        force_rotation_free_angle_deg=20.0,
        force_stop_gradient_backbone=True,
    )


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _rms(values: list[float]) -> float:
    values_array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(values_array))))


def _angle_deg(base: np.ndarray, updated: np.ndarray) -> np.ndarray:
    base = base.astype(np.float64)
    updated = updated.astype(np.float64)
    base /= np.maximum(np.linalg.norm(base, axis=-1, keepdims=True), 1.0e-12)
    updated /= np.maximum(np.linalg.norm(updated, axis=-1, keepdims=True), 1.0e-12)
    cosine = np.clip(np.sum(base * updated, axis=-1), -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine)).mean(axis=-1)


def _select(dataset, count: int, seed: int) -> list[int]:
    candidates_by_episode = _candidate_indices(dataset)
    candidates = [
        index
        for episode in sorted(candidates_by_episode)
        for index in candidates_by_episode[episode]
    ]
    if not candidates:
        raise RuntimeError("no held-out complete 50-step anchors")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(candidates))
    return [candidates[index] for index in permutation[: min(count, len(candidates))]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--force-norm", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_count <= 0 or args.batch_size <= 1:
        raise ValueError("sample-count must be positive and batch-size must exceed one")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = _config()
    dataset = build_force_dataset(
        (args.dataset_root,),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        force_norm_path=args.force_norm,
        max_token_len=config.max_token_len,
        force_update_action_steps=10,
        force_update_offsets=OFFSETS,
        load_future_force_targets=False,
        seed=0,
    )
    selected = _select(dataset, args.sample_count, args.seed)
    raw = dataset._raw  # noqa: SLF001
    selection = []
    for index in selected:
        anchor = int(raw.anchors[index])
        task_index = int(raw.base._task_index[anchor])  # noqa: SLF001
        selection.append(
            {
                "dataset_index": int(index),
                "episode": int(raw.anchor_episodes[index]),
                "frame": int(raw.base._frame_index[anchor]),  # noqa: SLF001
                "prompt": str(raw.base.tasks[task_index]),
            }
        )
    selection_path = args.output_dir / "selection.json"
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    decode = _output_transform(
        args.dataset_root,
        config,
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
    )
    force_norm = ForceNormalization.load(args.force_norm)
    initialized = config.create(jax.random.key(0))
    _, initialized_state = nnx.split(initialized)
    params = AtomicPi05CheckpointLoader(str(args.checkpoint / "params")).load(
        initialized_state.to_pure_dict()
    )
    del initialized, initialized_state
    model = config.load(params, remove_extra_params=False)
    del params
    model.eval()

    statistics = {
        branch: {
            offset: {
                "normalized_gt_rmse": [],
                "normalized_change_rmse": [],
                "joint_gt_rmse_rad": [],
                "left_tcp_gt_rmse_mm": [],
                "right_tcp_gt_rmse_mm": [],
                "force_angle_deg": [],
            }
            for offset in OFFSETS
        }
        for branch in BRANCHES
    }
    cases: list[dict] = []
    for start in range(0, len(selected), args.batch_size):
        indices = selected[start : start + args.batch_size]
        real_count = len(indices)
        if real_count < args.batch_size:
            indices += [indices[-1]] * (args.batch_size - real_count)
        batch = force_collate([dataset[index] for index in indices])
        observation_np, gt_full, initial_force = batch_to_force_inputs(batch)
        observation = jax.tree.map(jnp.asarray, observation_np)
        gt = np.asarray(gt_full)[..., : config.active_action_dim]
        full_context = _prepare_force_context(
            model,
            observation,
            jnp.asarray(initial_force["slow_force_history"]),
            jnp.asarray(initial_force["slow_state_history"]),
            jnp.asarray(initial_force["slow_history_mask"]),
        )
        zero_context = full_context.replace(
            force_latent=jnp.zeros_like(full_context.force_latent),
            slow_history_tokens=jnp.zeros_like(full_context.slow_history_tokens),
        )
        wrong_context = full_context.replace(
            force_latent=jnp.roll(full_context.force_latent, 1, axis=0),
            slow_history_tokens=jnp.roll(full_context.slow_history_tokens, 1, axis=0),
            slow_history_token_mask=jnp.roll(
                full_context.slow_history_token_mask, 1, axis=0
            ),
        )
        noise = jax.random.normal(
            jax.random.fold_in(jax.random.key(args.seed), start),
            (args.batch_size, config.action_horizon, config.action_dim),
        )
        raw_states = [
            np.asarray(raw.base._states[int(raw.anchors[index])])  # noqa: SLF001
            for index in indices
        ]
        state_norms = np.asarray(batch["state"])

        for offset in OFFSETS:
            force = _force_metadata_at_offset(raw, force_norm, indices, offset)
            current_mask = jnp.asarray(force["current_history_mask"])
            specifications = {
                "full": (full_context, current_mask),
                "zero_zf_content": (zero_context, current_mask),
                "wrong_zf": (wrong_context, current_mask),
                "no_fast": (full_context, jnp.zeros_like(current_mask)),
            }
            predictions: dict[str, np.ndarray] = {}
            angles: dict[str, np.ndarray] = {}
            for branch, (context, branch_mask) in specifications.items():
                prediction, modulation = _sample_force_update(
                    model,
                    context,
                    jnp.asarray(force["current_force_history"]),
                    jnp.asarray(force["current_state_history"]),
                    branch_mask,
                    jnp.asarray(force["update_offset"]),
                    noise,
                    jnp.asarray(gt),
                )
                predictions[branch] = np.asarray(jax.device_get(prediction))[:real_count]
                angles[branch] = _angle_deg(
                    np.asarray(jax.device_get(context.z_model))[:real_count],
                    np.asarray(jax.device_get(modulation.z_exec))[:real_count],
                )

            for local in range(real_count):
                gt_decoded = np.asarray(
                    decode(state_norms[local], raw_states[local], gt[local])["actions"]
                )
                full_prediction = predictions["full"][local]
                for branch in BRANCHES:
                    prediction = predictions[branch][local]
                    decoded = np.asarray(
                        decode(state_norms[local], raw_states[local], prediction)["actions"]
                    )
                    gt_metrics = _metrics(
                        prediction[offset:],
                        gt[local, offset:],
                        decoded[offset:],
                        gt_decoded[offset:],
                    )
                    change_rmse = float(
                        np.sqrt(
                            np.mean(
                                np.square(
                                    prediction[offset:] - full_prediction[offset:]
                                )
                            )
                        )
                    )
                    values = statistics[branch][offset]
                    values["normalized_gt_rmse"].append(
                        gt_metrics["normalized_action_rmse"]
                    )
                    values["normalized_change_rmse"].append(change_rmse)
                    values["joint_gt_rmse_rad"].append(gt_metrics["joint_rmse_rad"])
                    values["left_tcp_gt_rmse_mm"].append(
                        gt_metrics["left_tcp_translation_rmse_mm"]
                    )
                    values["right_tcp_gt_rmse_mm"].append(
                        gt_metrics["right_tcp_translation_rmse_mm"]
                    )
                    values["force_angle_deg"].append(float(angles[branch][local]))
                    cases.append(
                        selection[start + local]
                        | {
                            "offset": offset,
                            "branch": branch,
                            **{key: float(value[-1]) for key, value in values.items()},
                        }
                    )

    summary: dict = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "selection_count": len(selected),
        "offsets": list(OFFSETS),
        "branches": {
            "zero_zf_content": (
                "set the 30 encoded slow-history zF tokens to zero while retaining "
                "their valid mask and learned slow-token type embedding"
            ),
            "wrong_zf": (
                "cyclically permute zF tokens across batch rows while preserving each "
                "row's observation, zM, fast history, noise, and GT prefix"
            ),
            "no_fast": "mask only the 10 current fast-history tokens",
        },
        "by_offset": {},
    }
    for offset in OFFSETS:
        full_gt = _rms(statistics["full"][offset]["normalized_gt_rmse"])
        offset_summary = {}
        for branch in BRANCHES:
            values = statistics[branch][offset]
            branch_gt = _rms(values["normalized_gt_rmse"])
            offset_summary[branch] = {
                "normalized_action_rmse_to_gt": branch_gt,
                "gt_rmse_change_pct_vs_full": 100.0 * (branch_gt / full_gt - 1.0),
                "prediction_change_rmse_vs_full": _rms(
                    values["normalized_change_rmse"]
                ),
                "joint_rmse_rad_to_gt": _rms(values["joint_gt_rmse_rad"]),
                "left_tcp_rmse_mm_to_gt": _rms(values["left_tcp_gt_rmse_mm"]),
                "right_tcp_rmse_mm_to_gt": _rms(values["right_tcp_gt_rmse_mm"]),
                "mean_force_rotation_deg": _mean(values["force_angle_deg"]),
            }
        summary["by_offset"][str(offset)] = offset_summary

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary | {"cases": cases}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    base_norm = args.norm_assets_dir / args.norm_asset_id / "norm_stats.json"
    contract = (
        "# Spherical B2 zF attribution contract\n\n"
        f"- checkpoint: `{args.checkpoint.resolve()}`\n"
        f"- checkpoint manifest sha256: `{_sha256(args.checkpoint / 'params' / 'manifest.ocdbt')}`\n"
        f"- dataset: `{args.dataset_root.resolve()}`\n"
        f"- base norm: `{base_norm.resolve()}` sha256 `{_sha256(base_norm)}`\n"
        f"- force norm: `{args.force_norm.resolve()}` sha256 `{_sha256(args.force_norm)}`\n"
        f"- selection: `{selection_path.resolve()}` sha256 `{_sha256(selection_path)}`\n"
        f"- seed: {args.seed}; offsets: {OFFSETS}; sampler steps: 10; horizon: 50\n"
        "- prompt: exact task metadata; max_token_len=192; no global prompt\n"
        "- architecture: final zM reused for all 18 blocks; 2-head full-token adapter; "
        "30 SlowProj + 10 FastProj; spherical force cap=45deg, free-cone hinge=20deg\n"
        "- pairing: same observation/state/prompt/noise/fast history/GT prefix; only zF changes\n"
        "- zero_zf_content retains the valid-token mask and learned token-type embedding, "
        "so it measures encoded slow-history content rather than deleting the route\n"
        "- TCP offset: 0.20 m; normalized deltas are unnormalized before raw-state addition and FK\n"
    )
    (args.output_dir / "run_contract.md").write_text(contract, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
