#!/usr/bin/env python3
"""Measure force/state influence after a reviewed contact onset.

Selection is independent of model predictions.  Each row starts before a
large wrench transition and records the first action step at which contact is
observed.  At RTC offsets 10/20/30/40, all branches share observation, prompt,
noise, clean committed GT prefix, and state except for the named intervention.
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
from atomic_latent_vla.pi05.model import rtc_clamp_prefix, rtc_committed_mask
from atomic_latent_vla.pi05.weights import AtomicPi05CheckpointLoader

from evaluate_force_b2_episode50_ablation import _config, _metrics
from evaluate_force_image_misalignment import _cosine, _rmse
from evaluate_force_rtc_offsets import _force_metadata_at_offset
from evaluate_zm_fk_trajectory_ablation import _output_transform


OFFSETS = (10, 20, 30, 40)
BRANCHES = (
    "normal",
    "current_force_zero",
    "current_state_hold",
    "all_force_zero",
    "delta_z_zero",
    "no_zm_adapter",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@nnx.jit
def _prepare_context(model, observation, slow_force, slow_state, slow_mask):
    return model.prepare_force_policy_context(
        observation,
        slow_force_history=slow_force,
        slow_state_history=slow_state,
        slow_history_mask=slow_mask,
    )


@nnx.jit
def _sample_update(
    model,
    context,
    current_force,
    current_state,
    current_mask,
    update_offset,
    noise,
    executed_actions,
    use_delta_z,
):
    conditioner = model._require_force_conditioner()  # noqa: SLF001
    modulation = conditioner.modulate(
        context.z_model,
        context.force_latent,
        current_force,
        current_state,
        current_mask,
        update_offset,
    )
    injected_delta = jnp.where(
        use_delta_z, modulation.delta_z, jnp.zeros_like(modulation.delta_z)
    )
    layerwise_latents = model._force_layerwise_latents(  # noqa: SLF001
        context.layerwise_arm_latents, injected_delta
    )
    committed = rtc_committed_mask(update_offset, model.action_horizon)
    noise = model._mask_action_condition(noise)  # noqa: SLF001
    executed = jnp.pad(
        executed_actions,
        ((0, 0), (0, 0), (0, model.action_dim - model.config.active_action_dim)),
    )
    executed = model._mask_action_condition(executed)  # noqa: SLF001
    initial = rtc_clamp_prefix(noise, executed, committed)

    def step(index, actions):
        time = jnp.asarray(1.0 - index / 10.0, dtype=actions.dtype)
        token_time = jnp.where(
            committed, 0.0, jnp.broadcast_to(time, committed.shape)
        )
        velocity = model._suffix_velocity(  # noqa: SLF001
            context.prefix_mask,
            context.kv_cache,
            actions,
            token_time,
            context.z_model,
            committed,
            layerwise_latents=layerwise_latents,
        )
        updated = model._mask_action_condition(actions - 0.1 * velocity)  # noqa: SLF001
        return rtc_clamp_prefix(updated, executed, committed)

    result = jax.lax.fori_loop(0, 10, step, initial)
    result = rtc_clamp_prefix(result, executed, committed)
    return result[..., : model.config.active_action_dim], modulation.delta_z


@nnx.jit
def _sample_update_no_zm(
    model,
    context,
    update_offset,
    noise,
    executed_actions,
):
    """Sample with the same prefix/KV cache but no latent FiLM residual."""

    committed = rtc_committed_mask(update_offset, model.action_horizon)
    noise = model._mask_action_condition(noise)  # noqa: SLF001
    executed = jnp.pad(
        executed_actions,
        ((0, 0), (0, 0), (0, model.action_dim - model.config.active_action_dim)),
    )
    executed = model._mask_action_condition(executed)  # noqa: SLF001
    initial = rtc_clamp_prefix(noise, executed, committed)

    def step(index, actions):
        time = jnp.asarray(1.0 - index / 10.0, dtype=actions.dtype)
        token_time = jnp.where(
            committed, 0.0, jnp.broadcast_to(time, committed.shape)
        )
        # z_model=None and layerwise_latents=None are the exact suffix path
        # that skips LatentFiLMAdapter while retaining the same prefix cache,
        # Action Expert, time AdaRMS, noise, and committed prefix.
        velocity = model._suffix_velocity(  # noqa: SLF001
            context.prefix_mask,
            context.kv_cache,
            actions,
            token_time,
            None,
            committed,
            layerwise_latents=None,
        )
        updated = model._mask_action_condition(actions - 0.1 * velocity)  # noqa: SLF001
        return rtc_clamp_prefix(updated, executed, committed)

    result = jax.lax.fori_loop(0, 10, step, initial)
    return rtc_clamp_prefix(result, executed, committed)[
        ..., : model.config.active_action_dim
    ]


def _dataset_index(dataset) -> dict[tuple[int, int], int]:
    raw = dataset._raw  # noqa: SLF001
    result = {}
    for index, anchor in enumerate(raw.anchors):
        episode = int(raw.anchor_episodes[index])
        frame = int(raw.base._frame_index[int(anchor)])  # noqa: SLF001
        result[(episode, frame)] = int(index)
    return result


def _mean_metrics(rows: list[dict], branch: str) -> dict:
    fields = (
        "normalized_rmse_vs_normal",
        "normalized_cosine_vs_normal",
        "joint_rmse_rad_vs_normal",
        "left_tcp_translation_rmse_mm_vs_normal",
        "right_tcp_translation_rmse_mm_vs_normal",
        "normalized_rmse_to_gt",
        "gt_rmse_change_vs_normal",
        "delta_z_rmse_vs_normal",
    )
    return {
        field: float(np.mean([row["branches"][branch][field] for row in rows]))
        for field in fields
    } | {
        "worse_gt_fraction": float(
            np.mean([row["branches"][branch]["gt_rmse_change_vs_normal"] > 0 for row in rows])
        )
    }


def _mean_zm_only_effect(rows: list[dict]) -> dict:
    fields = (
        "normalized_rmse_vs_pure_zm",
        "joint_rmse_rad_vs_pure_zm",
        "left_tcp_translation_rmse_mm_vs_pure_zm",
        "right_tcp_translation_rmse_mm_vs_pure_zm",
    )
    return {
        field: float(np.mean([row["branches"]["no_zm_adapter"][field] for row in rows]))
        for field in fields
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--force-norm", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder-width", type=int, default=256)
    parser.add_argument("--encoder-heads", type=int, default=4)
    parser.add_argument("--encoder-mlp-dim", type=int, default=1024)
    parser.add_argument("--force-latent-dim", type=int, default=256)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = _config(args)
    force_norm = ForceNormalization.load(args.force_norm)
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
    manifest = json.loads(args.selection.read_text(encoding="utf-8"))
    lookup = _dataset_index(dataset)
    selection = []
    for row in manifest["rows"]:
        selected = dict(row)
        selected["dataset_index"] = lookup[(int(row["episode"]), int(row["anchor_frame"]))]
        selection.append(selected)
    resolved_selection = args.output_dir / "selection.json"
    resolved_selection.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

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

    raw = dataset._raw  # noqa: SLF001
    count = len(selection)
    raw_predictions = {
        branch: np.zeros((count, len(OFFSETS), config.action_horizon, config.active_action_dim), np.float32)
        for branch in BRANCHES
    }
    raw_gt = np.zeros((count, config.action_horizon, config.active_action_dim), np.float32)
    rows: list[dict] = []
    raw_zero_value = force_norm.normalize_force(
        np.zeros((1, 1, 6), dtype=np.float32)
    )[0, 0]

    for start in range(0, count, args.batch_size):
        selected_rows = selection[start : start + args.batch_size]
        real_count = len(selected_rows)
        if real_count < args.batch_size:
            selected_rows += [selected_rows[-1]] * (args.batch_size - real_count)
        indices = [int(row["dataset_index"]) for row in selected_rows]
        batch = force_collate([dataset[index] for index in indices])
        observation_np, gt_full, initial_force = batch_to_force_inputs(batch)
        observation = jax.tree.map(jnp.asarray, observation_np)
        gt = np.asarray(gt_full)[..., : config.active_action_dim]
        normal_context = _prepare_context(
            model,
            observation,
            jnp.asarray(initial_force["slow_force_history"]),
            jnp.asarray(initial_force["slow_state_history"]),
            jnp.asarray(initial_force["slow_history_mask"]),
        )
        slow_force_zero = jnp.broadcast_to(
            jnp.asarray(raw_zero_value),
            jnp.asarray(initial_force["slow_force_history"]).shape,
        )
        zero_force_context = _prepare_context(
            model,
            observation,
            slow_force_zero,
            jnp.asarray(initial_force["slow_state_history"]),
            jnp.asarray(initial_force["slow_history_mask"]),
        )
        noise = jax.random.normal(
            jax.random.fold_in(jax.random.key(args.seed), start),
            (args.batch_size, config.action_horizon, config.action_dim),
        )

        predictions_by_offset: dict[int, dict[str, np.ndarray]] = {}
        delta_by_offset: dict[int, dict[str, np.ndarray]] = {}
        for offset in OFFSETS:
            current = _force_metadata_at_offset(raw, force_norm, indices, offset)
            current_force = jnp.asarray(current["current_force_history"])
            current_state = jnp.asarray(current["current_state_history"])
            current_mask = jnp.asarray(current["current_history_mask"])
            force_zero = jnp.broadcast_to(jnp.asarray(raw_zero_value), current_force.shape)
            state_hold = jnp.broadcast_to(
                jnp.asarray(initial_force["slow_state_history"])[:, -1:, :],
                current_state.shape,
            )
            specifications = {
                "normal": (normal_context, current_force, current_state, True),
                "current_force_zero": (normal_context, force_zero, current_state, True),
                "current_state_hold": (normal_context, current_force, state_hold, True),
                "all_force_zero": (zero_force_context, force_zero, current_state, True),
                "delta_z_zero": (normal_context, current_force, current_state, False),
            }
            predictions_by_offset[offset] = {}
            delta_by_offset[offset] = {}
            for branch, (context, branch_force, branch_state, use_delta) in specifications.items():
                prediction, delta_z = _sample_update(
                    model,
                    context,
                    branch_force,
                    branch_state,
                    current_mask,
                    jnp.asarray(current["update_offset"]),
                    noise,
                    jnp.asarray(gt),
                    jnp.asarray(use_delta),
                )
                predictions_by_offset[offset][branch] = np.asarray(jax.device_get(prediction))
                delta_by_offset[offset][branch] = np.asarray(jax.device_get(delta_z))
            no_zm = _sample_update_no_zm(
                model,
                normal_context,
                jnp.asarray(current["update_offset"]),
                noise,
                jnp.asarray(gt),
            )
            predictions_by_offset[offset]["no_zm_adapter"] = np.asarray(
                jax.device_get(no_zm)
            )
            delta_by_offset[offset]["no_zm_adapter"] = np.zeros_like(
                delta_by_offset[offset]["normal"]
            )

        for local, selected in enumerate(selected_rows[:real_count]):
            global_index = start + local
            dataset_index = int(selected["dataset_index"])
            anchor = int(raw.anchors[dataset_index])
            task_index = int(raw.base._task_index[anchor])  # noqa: SLF001
            raw_state = np.asarray(raw.base._states[anchor])  # noqa: SLF001
            state_norm = np.asarray(batch["state"][local])
            gt_normalized = gt[local]
            gt_decoded = np.asarray(
                decode(state_norm, raw_state, gt_normalized)["actions"]
            )
            raw_gt[global_index] = gt_normalized
            for offset_index, offset in enumerate(OFFSETS):
                normal_normalized = predictions_by_offset[offset]["normal"][local]
                normal_decoded = np.asarray(
                    decode(state_norm, raw_state, normal_normalized)["actions"]
                )
                normal_gt = _metrics(
                    normal_normalized[offset:],
                    gt_normalized[offset:],
                    normal_decoded[offset:],
                    gt_decoded[offset:],
                )
                result = {
                    **selected,
                    "prompt": str(raw.base.tasks[task_index]),
                    "offset": offset,
                    "contact_observed": bool(offset >= int(selected["contact_onset_step"])),
                    "normal_gt_rmse": normal_gt["normalized_action_rmse"],
                    "branches": {},
                }
                normal_delta = delta_by_offset[offset]["normal"][local]
                pure_zm_normalized = predictions_by_offset[offset]["delta_z_zero"][local]
                pure_zm_decoded = np.asarray(
                    decode(state_norm, raw_state, pure_zm_normalized)["actions"]
                )
                for branch in BRANCHES:
                    prediction = predictions_by_offset[offset][branch][local]
                    raw_predictions[branch][global_index, offset_index] = prediction
                    decoded = np.asarray(decode(state_norm, raw_state, prediction)["actions"])
                    versus_normal = _metrics(
                        prediction[offset:],
                        normal_normalized[offset:],
                        decoded[offset:],
                        normal_decoded[offset:],
                    )
                    versus_gt = _metrics(
                        prediction[offset:],
                        gt_normalized[offset:],
                        decoded[offset:],
                        gt_decoded[offset:],
                    )
                    branch_delta = (
                        np.zeros_like(normal_delta)
                        if branch == "delta_z_zero"
                        else delta_by_offset[offset][branch][local]
                    )
                    result["branches"][branch] = {
                        "normalized_rmse_vs_normal": versus_normal["normalized_action_rmse"],
                        "normalized_cosine_vs_normal": _cosine(
                            prediction[offset:], normal_normalized[offset:]
                        ),
                        "joint_rmse_rad_vs_normal": versus_normal["joint_rmse_rad"],
                        "left_tcp_translation_rmse_mm_vs_normal": versus_normal[
                            "left_tcp_translation_rmse_mm"
                        ],
                        "right_tcp_translation_rmse_mm_vs_normal": versus_normal[
                            "right_tcp_translation_rmse_mm"
                        ],
                        "normalized_rmse_to_gt": versus_gt["normalized_action_rmse"],
                        "gt_rmse_change_vs_normal": (
                            versus_gt["normalized_action_rmse"]
                            - normal_gt["normalized_action_rmse"]
                        ),
                        "delta_z_rmse_vs_normal": _rmse(branch_delta, normal_delta),
                    }
                    if branch == "no_zm_adapter":
                        versus_pure_zm = _metrics(
                            prediction[offset:],
                            pure_zm_normalized[offset:],
                            decoded[offset:],
                            pure_zm_decoded[offset:],
                        )
                        result["branches"][branch] |= {
                            "normalized_rmse_vs_pure_zm": versus_pure_zm[
                                "normalized_action_rmse"
                            ],
                            "joint_rmse_rad_vs_pure_zm": versus_pure_zm[
                                "joint_rmse_rad"
                            ],
                            "left_tcp_translation_rmse_mm_vs_pure_zm": versus_pure_zm[
                                "left_tcp_translation_rmse_mm"
                            ],
                            "right_tcp_translation_rmse_mm_vs_pure_zm": versus_pure_zm[
                                "right_tcp_translation_rmse_mm"
                            ],
                        }
                rows.append(result)

    observed = [row for row in rows if row["contact_observed"]]
    unobserved = [row for row in rows if not row["contact_observed"]]
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "selection_count": count,
        "offsets": list(OFFSETS),
        "contact_observed_cases": len(observed),
        "precontact_cases": len(unobserved),
        "raw_force_zero_normalized_value": raw_zero_value.tolist(),
        "contact_observed": {
            branch: _mean_metrics(observed, branch) for branch in BRANCHES
        },
        "zm_only_effect_contact_observed": _mean_zm_only_effect(observed),
        "precontact": {
            branch: _mean_metrics(unobserved, branch) for branch in BRANCHES
        },
        "zm_only_effect_precontact": _mean_zm_only_effect(unobserved),
        "by_offset": {
            str(offset): {
                branch: _mean_metrics([row for row in rows if row["offset"] == offset], branch)
                for branch in BRANCHES
            }
            for offset in OFFSETS
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary | {"cases": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output_dir / "raw_predictions.npz",
        gt=raw_gt,
        offsets=np.asarray(OFFSETS),
        **raw_predictions,
    )
    with (args.output_dir / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "episode", "anchor_frame", "peak_frame", "peak_arm", "peak_score",
            "contact_onset_step", "offset", "contact_observed", "branch",
            "normalized_rmse_vs_normal", "joint_rmse_rad_vs_normal",
            "normalized_cosine_vs_normal",
            "left_tcp_translation_rmse_mm_vs_normal",
            "right_tcp_translation_rmse_mm_vs_normal", "normalized_rmse_to_gt",
            "gt_rmse_change_vs_normal", "delta_z_rmse_vs_normal",
            "normalized_rmse_vs_pure_zm", "joint_rmse_rad_vs_pure_zm",
            "left_tcp_translation_rmse_mm_vs_pure_zm",
            "right_tcp_translation_rmse_mm_vs_pure_zm",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for branch in BRANCHES:
                writer.writerow(
                    {key: row[key] for key in fields[:8]}
                    | {"branch": branch}
                    | row["branches"][branch]
                )

    base_norm = args.norm_assets_dir / args.norm_asset_id / "norm_stats.json"
    contract = (
        "# Force contact ablation contract\n\n"
        f"- checkpoint: `{args.checkpoint.resolve()}`\n"
        f"- checkpoint manifest sha256: `{_sha256(args.checkpoint / 'params' / 'manifest.ocdbt')}`\n"
        f"- dataset: `{args.dataset_root.resolve()}`\n"
        f"- base norm: `{base_norm.resolve()}` sha256 `{_sha256(base_norm)}`\n"
        f"- force norm: `{args.force_norm.resolve()}` sha256 `{_sha256(args.force_norm)}`\n"
        f"- input selection: `{args.selection.resolve()}` sha256 `{_sha256(args.selection)}`\n"
        f"- resolved selection: `{resolved_selection.resolve()}` sha256 `{_sha256(resolved_selection)}`\n"
        f"- seed: {args.seed}; offsets: {OFFSETS}; sampler steps: 10; horizon: 50\n"
        "- prompt: exact task metadata; same observation/noise/GT committed prefix within each branch\n"
        "- force zero: raw physical six-axis zero transformed by the recorded q01/q99\n"
        "- current_state_hold: replace acquired post-anchor force-side state with the anchor state\n"
        "- no_zm_adapter: same prefix/KV and Action Expert, but latent_condition and layerwise_latent_condition are both None\n"
        "- TCP offset: 0.20 m; normalized delta unnormalized before raw-state addition and FK\n"
    )
    (args.output_dir / "run_contract.md").write_text(contract, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
