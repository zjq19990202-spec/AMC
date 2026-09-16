#!/usr/bin/env python3
"""Measure the causal contribution of repeated final zM to Action-Expert hidden states.

The paired paths share observation, native subtask, prefix/KV cache, diffusion
noise, timestep, and current noisy action.  The intervention only removes the
    same final two-arm zM condition from every suffix Action-Expert block. Consequently
the hidden-state difference is a direct zM attribution rather than rollout
divergence accumulated from two independently sampled action trajectories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import einops
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from openpi.models import gemma as _gemma
from openpi.models import model as _model

from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.model import make_attn_mask, posemb_sincos
from atomic_latent_vla.pi05.training_data import (
    atomic_collate,
    batch_to_observation,
    build_atomic_dataset,
)


EPSILON = 1.0e-12
THRESHOLDS = (0.01, 0.05, 0.10)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_stratified_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        raise ValueError("dataset is empty")
    count = min(count, length)
    # Midpoint of equal-width strata avoids special-casing the first/last row
    # while deterministically covering the full dataset and episode range.
    indices = np.floor((np.arange(count, dtype=np.float64) + 0.5) * length / count)
    return np.clip(indices.astype(np.int64), 0, length - 1).tolist()


def _suffix_velocity_and_final_hidden(
    model,
    prefix_mask: jax.Array,
    kv_cache: object,
    noisy_actions: jax.Array,
    timestep: jax.Array,
    z_model: jax.Array | None,
    layerwise_arm_latents: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    """Exact ``_suffix_velocity`` path plus its final normalized hidden."""

    action_tokens = model.action_in_proj(noisy_actions)
    if timestep.ndim not in (1, 2):
        raise ValueError("timestep must have shape [B] or [B,H]")
    if timestep.ndim == 2 and timestep.shape != action_tokens.shape[:2]:
        raise ValueError(
            f"token timestep must have shape {action_tokens.shape[:2]}, got {timestep.shape}"
        )
    committed_mask = jnp.zeros(action_tokens.shape[:2], dtype=jnp.bool_)
    time = nnx.swish(
        model.time_mlp_out(
            nnx.swish(model.time_mlp_in(posemb_sincos(timestep, action_tokens.shape[-1])))
        )
    )
    suffix_mask = jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_)
    suffix_ar = jnp.asarray([True] + [False] * (model.config.action_horizon - 1))
    suffix_attention = make_attn_mask(suffix_mask, suffix_ar)
    prefix_attention = einops.repeat(prefix_mask, "b p -> b s p", s=action_tokens.shape[1])
    full_attention = jnp.concatenate([prefix_attention, suffix_attention], axis=-1)
    positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    _, fusion_params, _ = model._layerwise_atomic_inputs()  # noqa: SLF001
    (_, suffix_out), _ = model.PaliGemma.llm(
        [None, action_tokens],
        mask=full_attention,
        positions=positions,
        kv_cache=kv_cache,
        adarms_cond=[None, time],
        latent_condition=z_model,
        layerwise_arm_latent_condition=layerwise_arm_latents,
        latent_update_mask=~committed_mask,
        fusion_params=fusion_params,
    )
    final_hidden = suffix_out[:, -model.config.action_horizon :]
    return model.action_out_proj(final_hidden), final_hidden


@nnx.jit
def _trace_pair(model, observation, noise):
    observation = _model.preprocess_observation(None, observation, train=False)
    query_hidden, prefix_mask, kv_cache, _ = model._prefix_forward(observation)  # noqa: SLF001
    active_state = model._controlled_state(observation.state)  # noqa: SLF001
    _, _, z_model, _, _ = model._latent(query_hidden, active_state)  # noqa: SLF001
    # A deterministic within-batch cyclic permutation supplies a genuinely
    # model-produced but mismatched zM while keeping every other input fixed.
    wrong_z_model = jnp.roll(z_model, shift=1, axis=0)
    initial = model._mask_action_condition(noise)  # noqa: SLF001

    def scan_step(current, index):
        time = jnp.asarray(1.0 - index / 10.0, dtype=current.dtype)
        timestep = jnp.broadcast_to(time, (current.shape[0],))
        velocity_on, hidden_on = _suffix_velocity_and_final_hidden(
            model,
            prefix_mask,
            kv_cache,
            current,
            timestep,
            z_model,
            None,
        )
        velocity_off, hidden_off = _suffix_velocity_and_final_hidden(
            model,
            prefix_mask,
            kv_cache,
            current,
            timestep,
            None,
            None,
        )
        velocity_wrong, hidden_wrong = _suffix_velocity_and_final_hidden(
            model,
            prefix_mask,
            kv_cache,
            current,
            timestep,
            wrong_z_model,
            None,
        )

        hidden_on = hidden_on.astype(jnp.float32)
        velocity_on_f32 = velocity_on.astype(jnp.float32)

        def compare(hidden_other, velocity_other):
            hidden_other = hidden_other.astype(jnp.float32)
            delta = hidden_on - hidden_other
            on_ss = jnp.mean(jnp.square(hidden_on), axis=(1, 2))
            other_ss = jnp.mean(jnp.square(hidden_other), axis=(1, 2))
            delta_ss = jnp.mean(jnp.square(delta), axis=(1, 2))
            on_token_norm = jnp.sqrt(jnp.sum(jnp.square(hidden_on), axis=-1) + EPSILON)
            other_token_norm = jnp.sqrt(
                jnp.sum(jnp.square(hidden_other), axis=-1) + EPSILON
            )
            delta_token_norm = jnp.sqrt(jnp.sum(jnp.square(delta), axis=-1) + EPSILON)
            relative_token_on = delta_token_norm / on_token_norm
            relative_token_other = delta_token_norm / other_token_norm
            cosine = jnp.sum(hidden_on * hidden_other, axis=-1) / (
                on_token_norm * other_token_norm
            )
            threshold_fractions = jnp.stack(
                [jnp.mean(relative_token_on > threshold, axis=1) for threshold in THRESHOLDS],
                axis=-1,
            )
            velocity_delta = velocity_on_f32 - velocity_other.astype(jnp.float32)
            return (
                on_ss,
                other_ss,
                delta_ss,
                jnp.mean(relative_token_on, axis=1),
                jnp.mean(relative_token_other, axis=1),
                jnp.mean(cosine, axis=1),
                threshold_fractions,
                jnp.mean(jnp.square(velocity_on_f32), axis=(1, 2)),
                jnp.mean(jnp.square(velocity_delta), axis=(1, 2)),
            )

        remove_metrics = compare(hidden_off, velocity_off)
        wrong_metrics = compare(hidden_wrong, velocity_wrong)
        next_current = model._mask_action_condition(  # noqa: SLF001
            current - 0.1 * velocity_on
        )
        return next_current, (remove_metrics, wrong_metrics)

    _, (remove_metrics, wrong_metrics) = jax.lax.scan(scan_step, initial, jnp.arange(10))
    return z_model, remove_metrics, wrong_metrics


def _aggregate(metric_arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    on_ss = metric_arrays["hidden_on_ss"]
    off_ss = metric_arrays["hidden_off_ss"]
    delta_ss = metric_arrays["hidden_delta_ss"]
    velocity_on_ss = metric_arrays["velocity_on_ss"]
    velocity_delta_ss = metric_arrays["velocity_delta_ss"]
    rms_ratio_on = float(np.sqrt(np.sum(delta_ss) / np.maximum(np.sum(on_ss), EPSILON)))
    rms_ratio_off = float(np.sqrt(np.sum(delta_ss) / np.maximum(np.sum(off_ss), EPSILON)))
    velocity_ratio = float(
        np.sqrt(np.sum(velocity_delta_ss) / np.maximum(np.sum(velocity_on_ss), EPSILON))
    )
    threshold_values = metric_arrays["threshold_fractions"]
    threshold_fraction = np.mean(
        threshold_values,
        axis=tuple(range(threshold_values.ndim - 1)),
    )
    return {
        "hidden_delta_rms_over_with_zm": rms_ratio_on,
        "hidden_delta_rms_over_with_zm_percent": 100.0 * rms_ratio_on,
        "hidden_delta_energy_over_with_zm_percent": 100.0 * rms_ratio_on**2,
        "hidden_delta_rms_over_without_zm": rms_ratio_off,
        "hidden_delta_rms_over_without_zm_percent": 100.0 * rms_ratio_off,
        "mean_token_relative_l2_vs_with_zm": float(
            np.mean(metric_arrays["token_relative_on"])
        ),
        "mean_token_relative_l2_vs_without_zm": float(
            np.mean(metric_arrays["token_relative_off"])
        ),
        "mean_hidden_cosine": float(np.mean(metric_arrays["hidden_cosine"])),
        "token_fraction_relative_change_gt_1pct": float(threshold_fraction[0]),
        "token_fraction_relative_change_gt_5pct": float(threshold_fraction[1]),
        "token_fraction_relative_change_gt_10pct": float(threshold_fraction[2]),
        "velocity_delta_rms_over_with_zm": velocity_ratio,
        "velocity_delta_rms_over_with_zm_percent": 100.0 * velocity_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument(
        "--atomic-composition-sidecar",
        default="fk_horizon_3hz_gate_top5_stay_v2",
    )
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.sample_count <= 0 or args.batch_size <= 0:
        parser.error("--sample-count and --batch-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_metadata = args.checkpoint / "params" / "_METADATA"
    norm_stats = args.norm_assets_dir / args.norm_asset_id / "norm_stats.json"
    subtask_sidecar = args.dataset_root / "meta" / "episode_subtasks.jsonl"
    for path in (checkpoint_metadata, norm_stats, subtask_sidecar):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = AtomicPi05Config(
        max_token_len=192,
        fast_action_ce_loss_weight=0.0,
        coefficient_target_kind="joint_delta",
        coefficient_target_dim=14,
        # This is the exact AFRO50K final route: the same final two-arm zM is
        # reused by all 18 Action-Expert blocks. It is not the retired path
        # that feeds independently composed per-layer Q latents.
        enable_layerwise_atomic_flow=False,
        spherical_visual_latent=True,
        visual_max_update_angle_deg=45.0,
    )
    dataset = build_atomic_dataset(
        (args.dataset_root,),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        include_fast=False,
        atomic_composition_sidecar=args.atomic_composition_sidecar,
        pad_subtask_horizon=True,
    )
    selected_indices = _select_stratified_indices(len(dataset), args.sample_count)
    selection: list[dict[str, Any]] = []
    rows = []
    raw = dataset._raw  # noqa: SLF001
    visible = getattr(raw.base, "_visible_indices", np.arange(len(raw.base)))
    for dataset_index in selected_indices:
        row = dataset[dataset_index]
        metadata = dataset._raw.metadata(dataset_index)  # noqa: SLF001
        data_index = int(visible[dataset_index])
        selection.append(
            {
                "dataset_index": dataset_index,
                "data_index": data_index,
                "episode": int(raw.base._episode_index[data_index]),  # noqa: SLF001
                "frame": int(raw.base._frame_index[data_index]),  # noqa: SLF001
                "subtask_prompt": str(metadata["subtask_prompt"]),
                "subtask_boundary_padded": bool(metadata["subtask_boundary_padded"]),
            }
        )
        rows.append(row)
    (args.output_dir / "selection.json").write_text(
        json.dumps({"domain": args.domain, "rows": selection}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    rng = np.random.default_rng(args.seed)
    metric_names = (
        "hidden_on_ss",
        "hidden_off_ss",
        "hidden_delta_ss",
        "token_relative_on",
        "token_relative_off",
        "hidden_cosine",
        "threshold_fractions",
        "velocity_on_ss",
        "velocity_delta_ss",
    )
    collected = {
        condition: {name: [] for name in metric_names}
        for condition in ("remove_zm", "wrong_zm")
    }
    z_norms: list[np.ndarray] = []

    for start in range(0, len(rows), args.batch_size):
        chunk_rows = rows[start : start + args.batch_size]
        batch = atomic_collate(chunk_rows)
        observation_np, _ = batch_to_observation(batch)
        observation_np = _model.Observation(
            images=observation_np.images,
            image_masks=observation_np.image_masks,
            state=observation_np.state,
            tokenized_prompt=batch["subtask_prompt_tokens"],
            tokenized_prompt_mask=batch["subtask_prompt_mask"],
            token_ar_mask=observation_np.token_ar_mask,
            token_loss_mask=observation_np.token_loss_mask,
        )
        noise = rng.standard_normal(
            (len(chunk_rows), config.action_horizon, config.action_dim),
            dtype=np.float32,
        )
        z_model, remove_metrics, wrong_metrics = jax.device_get(
            _trace_pair(
                model,
                jax.tree.map(jnp.asarray, observation_np),
                jnp.asarray(noise),
            )
        )
        z_norms.append(np.linalg.norm(np.asarray(z_model), axis=-1))
        for condition, metrics in (
            ("remove_zm", remove_metrics),
            ("wrong_zm", wrong_metrics),
        ):
            for name, value in zip(metric_names, metrics, strict=True):
                # scan returns [diffusion_step, batch, ...]; concatenate on batch.
                collected[condition][name].append(np.asarray(value))

    metric_arrays = {
        condition: {
            name: np.concatenate(values, axis=1) for name, values in condition_values.items()
        }
        for condition, condition_values in collected.items()
    }
    z_norm = np.concatenate(z_norms, axis=0)
    aggregates = {condition: _aggregate(values) for condition, values in metric_arrays.items()}

    case_rows: list[dict[str, Any]] = []
    for sample_index, selected in enumerate(selection):
        case = dict(selected)
        for condition, condition_values in metric_arrays.items():
            sample_metrics = {
                name: value[:, sample_index] for name, value in condition_values.items()
            }
            case.update(
                {f"{condition}_{name}": value for name, value in _aggregate(sample_metrics).items()}
            )
        case_rows.append(case)
    with (args.output_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)
    np.savez_compressed(
        args.output_dir / "raw_metrics.npz",
        **{
            f"{condition}_{name}": value
            for condition, condition_values in metric_arrays.items()
            for name, value in condition_values.items()
        },
        z_model_norm=z_norm,
    )

    summary = {
        "domain": args.domain,
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "sample_count": len(selection),
        "diffusion_steps": 10,
        "action_horizon": config.action_horizon,
        "final_hidden_width": int(_gemma.get_config(config.action_expert_variant).width),
        "prompt_contract": "native active subtask from episode_subtasks.jsonl; 192 tokens",
        "normalization": args.norm_asset_id,
        "quantile_clip": False,
        "latent_route": "same final spherical two-arm zM reused in all 18 Action-Expert blocks",
        "interventions": (
            "same observation/state/subtask/noise/timestep/prefix/KV/current noisy action; "
            "compare correct zM against removal and a cyclically shuffled wrong zM"
        ),
        "hidden_location": "Action Expert final RMSNorm output, before action_out_proj",
        "aggregate": aggregates,
        "mean_final_zm_norm": float(np.mean(z_norm)),
        "checkpoint_metadata_sha256": _sha256(checkpoint_metadata),
        "norm_stats_sha256": _sha256(norm_stats),
        "subtask_sidecar_sha256": _sha256(subtask_sidecar),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "run_contract.md").write_text(
        "\n".join(
            [
                f"# zM final-hidden attribution: {args.domain}",
                "",
                f"- Checkpoint: `{summary['checkpoint']}`",
                f"- Dataset: `{summary['dataset_root']}`",
                f"- Selection: {len(selection)} deterministic equal-stratum frames",
                f"- Prompt: {summary['prompt_contract']}",
                f"- Norm: `{args.norm_asset_id}`; q01/q99 transform without clipping",
                "- Horizon: 50 actions; native subtask held/padded across boundaries",
                "- Pairing: identical observation, state, prompt, noise, timestep, prefix/KV, and noisy action",
                "- Interventions: remove zM, or cyclically shuffle zM within each batch; all other inputs fixed",
                "- Hidden: final Action Expert RMSNorm output before action_out_proj",
                f"- Seed: {args.seed}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
