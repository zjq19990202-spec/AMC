#!/usr/bin/env python3
"""Keep Fruit Context KV factual while swapping only the prompt used for zM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
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
from evaluate_fruit_target_switch import _find_row
from evaluate_fruit_target_switch import _sample_variants_no_zm
from evaluate_fruit_target_switch_multiframe import _native_template_prompts
from evaluate_global_episode_chunks import _output_transform


REVERSE = {
    "move_x_pos": "Move backward along base-frame -x.",
    "move_x_neg": "Move forward along base-frame +x.",
    "move_y_pos": "Move rightward along base-frame -y.",
    "move_y_neg": "Move leftward along base-frame +y.",
    "move_z_pos": "Move downward along base-frame -z.",
    "move_z_neg": "Move upward along base-frame +z.",
    "rotate_x_pos": "Rotate negatively about the base-frame x axis.",
    "rotate_x_neg": "Rotate positively about the base-frame x axis.",
    "rotate_y_pos": "Rotate negatively about the base-frame y axis.",
    "rotate_y_neg": "Rotate positively about the base-frame y axis.",
    "rotate_z_pos": "Rotate negatively about the base-frame z axis.",
    "rotate_z_neg": "Rotate positively about the base-frame z axis.",
}


@nnx.jit
def _sample(model, observation, noise, context_tokens, context_mask, z_tokens, z_mask):
    observation = _model.preprocess_observation(None, observation, train=False)
    context_obs = model._with_prompt(observation, context_tokens, context_mask)  # noqa: SLF001
    layerwise = bool(model.config.enable_layerwise_atomic_flow)
    if layerwise:
        _, prefix_mask, kv_cache, _, _ = model._prefix_forward(  # noqa: SLF001
            context_obs, return_layerwise_latents=True
        )
    else:
        _, prefix_mask, kv_cache, _ = model._prefix_forward(context_obs)  # noqa: SLF001
    active_state = model._controlled_state(observation.state)  # noqa: SLF001

    z_models = []
    directions = []
    layerwise_models = []
    for tokens, mask in zip(z_tokens, z_mask, strict=True):
        z_obs = model._with_prompt(observation, tokens, mask)  # noqa: SLF001
        if layerwise:
            query, _, _, _, per_layer = model._prefix_forward(  # noqa: SLF001
                z_obs, return_layerwise_latents=True
            )
            layerwise_models.append(per_layer)
        else:
            query, _, _, _ = model._prefix_forward(z_obs)  # noqa: SLF001
        _, right_direction, z_model, left_direction, _ = model._latent(  # noqa: SLF001
            query, active_state
        )
        z_models.append(z_model)
        directions.append(jnp.stack([right_direction, left_direction], axis=1))
    z_models = jnp.stack(z_models, axis=0)
    directions = jnp.stack(directions, axis=0)
    noise = model._mask_action_condition(noise)  # noqa: SLF001

    def generate(z_model, per_layer=None):
        def step(index, current):
            time = jnp.asarray(1.0 - index / 10.0, dtype=current.dtype)
            velocity = model._suffix_velocity(  # noqa: SLF001
                prefix_mask,
                kv_cache,
                current,
                jnp.broadcast_to(time, (current.shape[0],)),
                z_model,
                layerwise_latents=per_layer,
            )
            return model._mask_action_condition(current - 0.1 * velocity)  # noqa: SLF001

        return jax.lax.fori_loop(0, 10, step, noise)[..., : model.config.active_action_dim]

    if layerwise:
        actions = jax.vmap(generate)(z_models, jnp.stack(layerwise_models, axis=0))
    else:
        actions = jax.vmap(generate)(z_models)
    return actions, z_models, directions


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--atomic-sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--wrong-target",
        help="Optional fixed counterfactual fruit target for every selected frame.",
    )
    parser.add_argument(
        "--context-target",
        help="Optional target substituted into the factual Context/zM baseline prompt.",
    )
    parser.add_argument(
        "--enable-layerwise-atomic-flow",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    config = AtomicPi05Config(
        max_token_len=200,
        fast_action_ce_loss_weight=0.0,
        coefficient_target_kind="joint_delta",
        coefficient_target_dim=14,
        enable_layerwise_atomic_flow=args.enable_layerwise_atomic_flow,
    )
    dataset = build_atomic_dataset(
        (args.dataset_root,),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        action_horizon=50,
        max_token_len=200,
        include_fast=False,
        atomic_composition_sidecar="fk_horizon_3hz_gate_top5_stay_v2",
        pad_subtask_horizon=True,
    )
    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    tokenizer = _paligemma_tokenizer(144)
    decoder = _output_transform(
        args.dataset_root,
        config,
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
    )

    selection = json.loads(args.selection_manifest.read_text())["datasets"]["target2058"]
    atomic_rows: dict[int, list[dict]] = {}
    for line in args.atomic_sidecar.read_text().splitlines():
        row = json.loads(line)
        atomic_rows.setdefault(int(row["episode_index"]), []).append(row)
    target_names = [str(row["native_target"]) for row in selection]
    results = []
    archive = {}

    for index, spec in enumerate(selection):
        episode, frame = int(spec["episode"]), int(spec["frame"])
        dataset_index, _, _, metadata = _find_row(dataset, episode, frame)
        sample = dataset[dataset_index]
        batch = atomic_collate([sample])
        obs_np, _ = batch_to_observation(batch)
        observation = jax.tree.map(jnp.asarray, obs_np)
        token_state = np.asarray(batch["state"])[0]
        native_prompt = str(metadata["subtask_prompt"])
        if args.context_target:
            correct_prompt = _native_template_prompts(
                native_prompt, [args.context_target]
            )[0][args.context_target]
        else:
            correct_prompt = native_prompt
        wrong_target = args.wrong_target or target_names[(index + 1) % len(target_names)]
        wrong_prompt = _native_template_prompts(correct_prompt, [wrong_target])[0][wrong_target]

        block = frame // 10
        candidates = [
            row for row in atomic_rows.get(episode, [])
            if int(row["block_start_id"]) <= block <= int(row["block_end_id"])
            and row["arm"] == spec["active_arm"]
        ]
        reverse_prompt = None
        labels = []
        if candidates:
            chosen = min(candidates, key=lambda row: abs(int(row["block_start_id"]) - block))
            labels = list(chosen["fk_atomic_labels"])
            reverse_prompt = " ".join(REVERSE[label] for label in labels)

        routes = [("correct_z", correct_prompt), ("wrong_fruit_z", wrong_prompt)]
        if reverse_prompt:
            routes.append(("reverse_atomic_z", reverse_prompt))
        tokenized = [tokenizer.tokenize(prompt, token_state) for _, prompt in routes]
        context_tokens, context_mask = tokenizer.tokenize(correct_prompt, token_state)
        noise = jax.random.normal(
            jax.random.fold_in(jax.random.key(args.seed), episode * 10000 + frame),
            (1, 50, config.action_dim),
        )
        normalized, z_models, directions = jax.device_get(
            _sample(
                model,
                observation,
                noise,
                jnp.asarray(context_tokens)[None],
                jnp.asarray(context_mask)[None],
                jnp.asarray(np.stack([x[0] for x in tokenized]))[:, None, :],
                jnp.asarray(np.stack([x[1] for x in tokenized]))[:, None, :],
            )
        )
        normalized = np.asarray(normalized)[:, 0]
        z_models = np.asarray(z_models)[:, 0]
        directions = np.asarray(directions)[:, 0]
        no_zm_normalized, _, _ = jax.device_get(
            _sample_variants_no_zm(
                model,
                observation,
                jnp.asarray(context_tokens)[None],
                jnp.asarray(context_mask)[None],
                noise,
            )
        )
        no_zm_normalized = np.asarray(no_zm_normalized)[0]
        decoded = np.stack([
            np.asarray(decoder(np.asarray(batch["state"])[0], metadata["raw_state"], value)["actions"])
            for value in normalized
        ])
        no_zm_decoded = np.asarray(
            decoder(
                np.asarray(batch["state"])[0],
                metadata["raw_state"],
                no_zm_normalized,
            )["actions"]
        )
        base_norm, base_decoded, base_z = normalized[0], decoded[0], z_models[0]
        active_arm_index = 0 if spec["active_arm"] == "right" else 1
        base_active_z = base_z[active_arm_index]
        base_active_direction = directions[0, active_arm_index]
        route_metrics = {}
        for route_index, (name, prompt) in enumerate(routes):
            dz = z_models[route_index] - base_z
            active_z = z_models[route_index, active_arm_index]
            active_direction = directions[route_index, active_arm_index]
            route_metrics[name] = {
                "z_prompt": prompt,
                "z_cosine_to_correct": _cosine(z_models[route_index], base_z),
                "z_l2": float(np.linalg.norm(dz)),
                "z_relative_l2": float(np.linalg.norm(dz) / max(np.linalg.norm(base_z), 1e-12)),
                "active_arm_z_cosine_to_correct": _cosine(active_z, base_active_z),
                "active_arm_z_relative_l2": float(
                    np.linalg.norm(active_z - base_active_z)
                    / max(np.linalg.norm(base_active_z), 1e-12)
                ),
                "active_arm_direction_cosine_to_correct": _cosine(
                    active_direction, base_active_direction
                ),
                "active_arm_direction_relative_l2": float(
                    np.linalg.norm(active_direction - base_active_direction)
                    / max(np.linalg.norm(base_active_direction), 1e-12)
                ),
                "normalized_action_rmse_to_correct": float(np.sqrt(np.mean((normalized[route_index] - base_norm) ** 2))),
                "decoded_action_rmse_to_correct": float(np.sqrt(np.mean((decoded[route_index, :, :16] - base_decoded[:, :16]) ** 2))),
                "decoded_action_max_abs_to_correct": float(np.max(np.abs(decoded[route_index, :, :16] - base_decoded[:, :16]))),
            }
        route_metrics["context_no_zm"] = {
            "z_prompt": None,
            "normalized_action_rmse_to_correct": float(
                np.sqrt(np.mean((no_zm_normalized - base_norm) ** 2))
            ),
            "decoded_action_rmse_to_correct": float(
                np.sqrt(np.mean((no_zm_decoded[:, :16] - base_decoded[:, :16]) ** 2))
            ),
            "decoded_action_max_abs_to_correct": float(
                np.max(np.abs(no_zm_decoded[:, :16] - base_decoded[:, :16]))
            ),
        }
        key = f"episode_{episode:06d}_frame_{frame:06d}"
        archive[f"{key}_normalized"] = normalized.astype(np.float32)
        archive[f"{key}_decoded"] = decoded[:, :, :16].astype(np.float32)
        archive[f"{key}_z_models"] = z_models.astype(np.float32)
        archive[f"{key}_directions"] = directions.astype(np.float32)
        archive[f"{key}_route_names"] = np.asarray([x[0] for x in routes], dtype=np.str_)
        archive[f"{key}_no_zm_normalized"] = no_zm_normalized.astype(np.float32)
        archive[f"{key}_no_zm_decoded"] = no_zm_decoded[:, :16].astype(np.float32)
        results.append({
            "episode": episode,
            "frame": frame,
            "active_arm": spec["active_arm"],
            "context_prompt": correct_prompt,
            "wrong_target": wrong_target,
            "atomic_labels": labels,
            "routes": route_metrics,
        })

    summary = {}
    for route in ("wrong_fruit_z", "reverse_atomic_z"):
        rows = [row["routes"][route] for row in results if route in row["routes"]]
        summary[route] = {
            "frames": len(rows),
            **{
                key: float(np.mean([row[key] for row in rows]))
                for key in (
                    "z_cosine_to_correct",
                    "z_relative_l2",
                    "active_arm_z_cosine_to_correct",
                    "active_arm_z_relative_l2",
                    "active_arm_direction_cosine_to_correct",
                    "active_arm_direction_relative_l2",
                    "normalized_action_rmse_to_correct",
                    "decoded_action_rmse_to_correct",
                    "decoded_action_max_abs_to_correct",
                )
            },
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    np.savez_compressed(args.output / "raw_routes.npz", **archive)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
