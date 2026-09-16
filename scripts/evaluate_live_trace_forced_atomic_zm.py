#!/usr/bin/env python3
"""Causal intervention: replace one final AFRO z_M arm token by an atomic one.

The observation, native subtask Context KV, opposite-arm z_M token, and flow
noise remain fixed.  Only the selected arm's final 512-D z_M token is copied
from a canonical single-atom prompt at the same observation and state.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from openpi.models import model as openpi_model

from atomic_latent_vla.atomic import ATOMIC_NAMES
from atomic_latent_vla.pi05.config import AtomicPi05Config

import evaluate_live_trace_afro_prompt_causality as live
import evaluate_many_cluster_single_dual_steering as steer
import evaluate_zm_fk_trajectory_ablation as fk_eval


ATOMS = tuple(ATOMIC_NAMES[:12])
ARMS = ("right", "left")
ARM_INDEX = {"right": 0, "left": 1}
JOINT_SLICE = {"right": slice(8, 15), "left": slice(0, 7)}


@nnx.jit
def _sample_forced_atomic_zm(model, observation, noise, replace_arm):
    """Return native-KV actions, Q1 directions, and actual final z_M tokens."""

    observation = openpi_model.preprocess_observation(None, observation, train=False)
    query_hidden, prefix_mask, kv_cache, _ = model._prefix_forward(observation)  # noqa: SLF001
    active_state = model._controlled_state(observation.state)  # noqa: SLF001
    _, right_direction, z_model, left_direction, _ = model._latent(  # noqa: SLF001
        query_hidden, active_state
    )
    directions = jnp.stack([right_direction, left_direction], axis=1)

    batch_size = z_model.shape[0]
    native_mask = jnp.repeat(prefix_mask[0:1], batch_size, axis=0)
    native_kv = live._repeat_scanned_cache_batch(kv_cache, 0, batch_size)  # noqa: SLF001
    native_z = jnp.repeat(z_model[0:1], batch_size, axis=0)
    arm_selector = jnp.arange(2)[None, :] == replace_arm[:, None]
    hybrid_z = jnp.where(arm_selector[..., None], z_model, native_z)
    initial = model._mask_action_condition(noise)  # noqa: SLF001

    def step(index, actions):
        time = jnp.asarray(1.0 - index / 10.0, dtype=actions.dtype)
        velocity = model._suffix_velocity(  # noqa: SLF001
            native_mask,
            native_kv,
            actions,
            jnp.broadcast_to(time, (batch_size,)),
            hybrid_z,
        )
        return model._mask_action_condition(actions - 0.1 * velocity)  # noqa: SLF001

    actions = jax.lax.fori_loop(0, 10, step, initial)
    return actions[..., : model.config.active_action_dim], directions, z_model, hybrid_z


def _prompt_specs(native_prompt: str) -> list[dict]:
    specs = [{"arm": "native", "atom": "native", "prompt": native_prompt, "replace": -1}]
    for arm in ARMS:
        for atom in ATOMS:
            specs.append(
                {
                    "arm": arm,
                    "atom": atom,
                    "prompt": steer._prompt(arm, (atom,)),  # noqa: SLF001
                    "replace": ARM_INDEX[arm],
                }
            )
    return specs


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace-csv", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--max-token-len", type=int, default=192)
    parser.add_argument("--noise-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    prompts, anchors = live._load_trace(args.trace_csv, args.images_dir)  # noqa: SLF001
    del prompts
    live._save_contact_sheet(anchors, args.output_dir / "source_observations.png")  # noqa: SLF001

    config = AtomicPi05Config(
        max_token_len=args.max_token_len,
        coefficient_target_kind="joint_delta",
        coefficient_target_dim=14,
        enable_layerwise_atomic_flow=False,
        zm_teacher_action_conditioning=True,
        fast_action_ce_loss_weight=0.0,
        subtask_ce_loss_weight=0.0,
    )
    input_transforms, output_transforms = live._build_transforms(args, config)  # noqa: SLF001
    params = openpi_model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    codebook = np.asarray(jax.device_get(model.codebook.value), dtype=np.float64)
    codebook /= np.maximum(np.linalg.norm(codebook, axis=-1, keepdims=True), 1.0e-8)

    fk_eval.TCP_LOCAL_Z_OFFSET_M = 0.20
    fk = fk_eval.SimpleCR1FK(Path("assets/cr1_right_arm_tcp_red/urdf/CR1ARMR_TCP_RED.urdf"))
    rows: list[dict] = []
    native_geometry: list[dict] = []
    raw_actions = []
    raw_directions = []
    raw_zm = []

    for anchor_index, anchor in enumerate(anchors):
        specs = _prompt_specs(anchor["prompt"])
        raw = live._load_raw_observation(anchor)  # noqa: SLF001
        encoded = [live._encode(raw, spec["prompt"], input_transforms) for spec in specs]  # noqa: SLF001
        stacked = live._stack(encoded)  # noqa: SLF001
        normalized_state = np.asarray(stacked["state"])
        observation = openpi_model.Observation.from_dict(jax.tree.map(jnp.asarray, stacked))
        replace_arm = jnp.asarray([spec["replace"] for spec in specs], dtype=jnp.int32)
        anchor_actions = []

        for noise_index in range(args.noise_repeats):
            key = jax.random.fold_in(jax.random.key(args.seed), anchor_index * 100 + noise_index)
            one_noise = jax.random.normal(key, (1, config.action_horizon, config.action_dim))
            noise = jnp.repeat(one_noise, len(specs), axis=0)
            sampled, directions, z_model, _ = jax.device_get(
                _sample_forced_atomic_zm(model, observation, noise, replace_arm)
            )
            sampled = np.asarray(sampled)[..., :16]
            directions = np.asarray(directions, dtype=np.float64)
            z_model = np.asarray(z_model, dtype=np.float64)
            decoded = np.stack(
                [
                    live._decode(  # noqa: SLF001
                        normalized_state[index], anchor["state"], sampled[index], output_transforms
                    )
                    for index in range(len(specs))
                ]
            )
            anchor_actions.append(decoded)
            if noise_index == 0:
                raw_directions.append(directions)
                raw_zm.append(z_model)

            native_action = decoded[0]
            for spec_index, spec in enumerate(specs[1:], start=1):
                arm = spec["arm"]
                other = "left" if arm == "right" else "right"
                atom = spec["atom"]
                component, sign = steer._component(atom)  # noqa: SLF001
                native_tcp = steer._trajectory(fk, anchor["state"], native_action, arm)  # noqa: SLF001
                forced_tcp = steer._trajectory(fk, anchor["state"], decoded[spec_index], arm)  # noqa: SLF001
                delta = forced_tcp[-1] - native_tcp[-1]
                motion = np.asarray((*range(7), *range(8, 15)), dtype=np.int64)
                rows.append(
                    {
                        "request_id": int(anchor["request_id"]),
                        "native_prompt": anchor["prompt"],
                        "noise_index": noise_index,
                        "arm": arm,
                        "atom": atom,
                        "native_to_atomic_q1_cos": live._cosine(  # noqa: SLF001
                            directions[0, ARM_INDEX[arm]], directions[spec_index, ARM_INDEX[arm]]
                        ),
                        "native_to_atomic_zm_cos": live._cosine(  # noqa: SLF001
                            z_model[0, ARM_INDEX[arm]], z_model[spec_index, ARM_INDEX[arm]]
                        ),
                        "atomic_q1_requested_code_cos": float(
                            codebook[ARM_INDEX[arm], ATOMIC_NAMES.index(atom)]
                            @ directions[spec_index, ARM_INDEX[arm]]
                        ),
                        "forced_vs_native_joint14_rmse_rad": live._rmse(  # noqa: SLF001
                            decoded[spec_index][:, motion], native_action[:, motion]
                        ),
                        "forced_vs_native_target7_rmse_rad": live._rmse(  # noqa: SLF001
                            decoded[spec_index, :, JOINT_SLICE[arm]], native_action[:, JOINT_SLICE[arm]]
                        ),
                        "forced_vs_native_other7_rmse_rad": live._rmse(  # noqa: SLF001
                            decoded[spec_index, :, JOINT_SLICE[other]], native_action[:, JOINT_SLICE[other]]
                        ),
                        "requested_endpoint_causal": float(sign * delta[component]),
                        "endpoint_causal_component": float(delta[component]),
                        "requested_unit": "mm" if component < 3 else "deg",
                    }
                )
        raw_actions.append(np.stack(anchor_actions))

        directions = raw_directions[-1]
        for arm in ARMS:
            arm_index = ARM_INDEX[arm]
            similarities = codebook[arm_index] @ directions[0, arm_index]
            order = np.argsort(-similarities)
            native_geometry.append(
                {
                    "request_id": int(anchor["request_id"]),
                    "native_prompt": anchor["prompt"],
                    "arm": arm,
                    "top1_atom": ATOMIC_NAMES[int(order[0])],
                    "top1_cos": float(similarities[order[0]]),
                    "top2_atom": ATOMIC_NAMES[int(order[1])],
                    "top2_cos": float(similarities[order[1]]),
                    "top3_atom": ATOMIC_NAMES[int(order[2])],
                    "top3_cos": float(similarities[order[2]]),
                    "stay_cos": float(similarities[ATOMIC_NAMES.index("stay")]),
                    "top1_margin": float(similarities[order[0]] - similarities[order[1]]),
                }
            )
        print(f"anchor {anchor_index + 1}/{len(anchors)} request={anchor['request_id']}", flush=True)

    _write_csv(args.output_dir / "forced_atomic_zm_rows.csv", rows)
    _write_csv(args.output_dir / "native_zm_codebook.csv", native_geometry)
    np.savez_compressed(
        args.output_dir / "raw_intervention.npz",
        actions=np.stack(raw_actions),
        directions=np.stack(raw_directions),
        zm=np.stack(raw_zm),
        atoms=np.asarray(ATOMS),
        arms=np.asarray(ARMS),
        request_ids=np.asarray([anchor["request_id"] for anchor in anchors]),
    )

    per_atom = []
    for arm in ARMS:
        for atom in ATOMS:
            selected = [row for row in rows if row["arm"] == arm and row["atom"] == atom]
            per_atom.append(
                {
                    "arm": arm,
                    "atom": atom,
                    "joint14_rmse_rad": _mean(selected, "forced_vs_native_joint14_rmse_rad"),
                    "target7_rmse_rad": _mean(selected, "forced_vs_native_target7_rmse_rad"),
                    "other7_rmse_rad": _mean(selected, "forced_vs_native_other7_rmse_rad"),
                    "requested_endpoint_causal": _mean(selected, "requested_endpoint_causal"),
                    "requested_sign_rate": float(np.mean([row["requested_endpoint_causal"] > 0 for row in selected])),
                    "native_to_atomic_q1_cos": _mean(selected, "native_to_atomic_q1_cos"),
                    "native_to_atomic_zm_cos": _mean(selected, "native_to_atomic_zm_cos"),
                    "atomic_q1_requested_code_cos": _mean(selected, "atomic_q1_requested_code_cos"),
                    "unit": selected[0]["requested_unit"],
                }
            )
    _write_csv(args.output_dir / "per_atom_summary.csv", per_atom)

    summary = {
        "checkpoint": str(args.checkpoint),
        "anchors": len(anchors),
        "noise_repeats": args.noise_repeats,
        "contract": "native subtask KV/state/images/noise fixed; replace only selected arm final z_M; opposite arm z_M unchanged",
        "mean_native_codebook_top1_cos": _mean(native_geometry, "top1_cos"),
        "mean_native_codebook_top1_margin": _mean(native_geometry, "top1_margin"),
        "mean_forced_vs_native_joint14_rmse_rad": _mean(rows, "forced_vs_native_joint14_rmse_rad"),
        "mean_forced_vs_native_target7_rmse_rad": _mean(rows, "forced_vs_native_target7_rmse_rad"),
        "mean_forced_vs_native_other7_rmse_rad": _mean(rows, "forced_vs_native_other7_rmse_rad"),
        "mean_native_to_atomic_q1_cos": _mean(rows, "native_to_atomic_q1_cos"),
        "mean_native_to_atomic_zm_cos": _mean(rows, "native_to_atomic_zm_cos"),
        "mean_atomic_q1_requested_code_cos": _mean(rows, "atomic_q1_requested_code_cos"),
        "requested_sign_rate": float(np.mean([row["requested_endpoint_causal"] > 0 for row in rows])),
        "per_atom": per_atom,
        "native_geometry": native_geometry,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    matrix_rmse = np.asarray([row["joint14_rmse_rad"] for row in per_atom]).reshape(2, 12)
    matrix_causal = np.asarray([row["requested_endpoint_causal"] for row in per_atom]).reshape(2, 12)
    figure, axes = plt.subplots(2, 1, figsize=(15, 7.5), constrained_layout=True)
    for axis, matrix, title in (
        (axes[0], matrix_rmse, "Forced atomic zM vs native: 14-joint horizon RMSE (rad)"),
        (axes[1], matrix_causal, "Requested endpoint causal change (translation mm / rotation deg)"),
    ):
        image = axis.imshow(matrix, aspect="auto", cmap="viridis")
        axis.set_yticks(range(2), ARMS)
        axis.set_xticks(range(12), ATOMS, rotation=35, ha="right")
        axis.set_title(title)
        for y in range(2):
            for x in range(12):
                axis.text(x, y, f"{matrix[y, x]:.3f}", ha="center", va="center", fontsize=7, color="white")
        figure.colorbar(image, ax=axis, fraction=0.025, pad=0.015)
    figure.savefig(args.output_dir / "forced_atomic_zm_effect.png", dpi=190, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
