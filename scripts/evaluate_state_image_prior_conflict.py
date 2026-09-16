#!/usr/bin/env python3
"""Direct aligned-prompt/state versus image-prior conflict.

Use same-state-cluster pairs whose recorded atomic labels conflict in sign on
the same TCP twist component.  For each orientation A<-B, compare:

  factual: image A + state A + atomic prompt A
  mixed:   image B + state A + atomic prompt A

If the mixed endpoint follows atom A, the aligned prompt+state condition wins;
if it follows atom B, the image-associated prior wins.  Both pair orientations
are evaluated with matched flow noise.  Final statistics additionally require
the two factual baselines to reproduce their own requested signs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
import evaluate_many_cluster_single_dual_steering as base
import evaluate_zm_fk_trajectory_ablation as fk_eval
from evaluate_zm_fk_trajectory_ablation import _output_transform


def _component(atom: str) -> tuple[str, int, float, float]:
    family, axis, sign_name = atom.split("_")
    key = f"{family}_{axis}"
    index = "xyz".index(axis) + (3 if family == "rotate" else 0)
    sign = 1.0 if sign_name == "pos" else -1.0
    threshold = 1.0 if family == "rotate" else 5.0
    return key, index, sign, threshold


def _conflicts(a_atoms: list[str], b_atoms: list[str]) -> list[dict]:
    a = {_component(atom)[0]: (atom, *_component(atom)[1:]) for atom in a_atoms}
    b = {_component(atom)[0]: (atom, *_component(atom)[1:]) for atom in b_atoms}
    rows = []
    for key in sorted(a.keys() & b.keys()):
        a_atom, index, a_sign, threshold = a[key]
        b_atom, b_index, b_sign, b_threshold = b[key]
        if a_sign == b_sign:
            continue
        assert index == b_index and threshold == b_threshold
        rows.append(
            {
                "component": key,
                "index": index,
                "threshold": threshold,
                "state_atom": a_atom,
                "image_atom": b_atom,
                "state_sign": a_sign,
                "image_sign": b_sign,
            }
        )
    return rows


@nnx.jit
def _sample_pair(model, factual_observation, mixed_observation, noise, tokens, token_mask):
    factual_observation = _model.preprocess_observation(
        None, factual_observation, train=False
    )
    mixed_observation = _model.preprocess_observation(None, mixed_observation, train=False)

    def prepare(observation):
        observation = model._with_prompt(observation, tokens, token_mask)  # noqa: SLF001
        query, prefix_mask, kv_cache, _ = model._prefix_forward(observation)  # noqa: SLF001
        active_state = model._controlled_state(observation.state)  # noqa: SLF001
        _, _, z_model, _, _ = model._latent(query, active_state)  # noqa: SLF001
        return prefix_mask, kv_cache, z_model

    factual = prepare(factual_observation)
    mixed = prepare(mixed_observation)
    noise = model._mask_action_condition(noise)  # noqa: SLF001
    batch_size = noise.shape[0]

    def sample(prefix_mask, kv_cache, z_model):
        def step(index, current):
            time = jnp.asarray(1.0 - index / 10.0, dtype=current.dtype)
            velocity = model._suffix_velocity(  # noqa: SLF001
                prefix_mask,
                kv_cache,
                current,
                jnp.broadcast_to(time, (batch_size,)),
                z_model,
            )
            velocity = model._mask_action_condition(velocity)  # noqa: SLF001
            return model._mask_action_condition(current - 0.1 * velocity)  # noqa: SLF001

        result = jax.lax.fori_loop(0, 10, step, noise)
        return result[..., : model.config.active_action_dim]

    factual_mask, factual_kv, factual_z = factual
    mixed_mask, mixed_kv, mixed_z = mixed
    return jnp.stack(
        [
            sample(factual_mask, factual_kv, factual_z),
            sample(mixed_mask, mixed_kv, factual_z),
            sample(factual_mask, factual_kv, mixed_z),
            sample(mixed_mask, mixed_kv, mixed_z),
        ],
        axis=0,
    )


def _vote(value: float, state_sign: float, image_sign: float, threshold: float) -> str:
    if state_sign * value > threshold:
        return "prompt_state"
    if image_sign * value > threshold:
        return "image"
    return "neither"


def _route_vote(value: float, context_sign: float, zm_sign: float, threshold: float) -> str:
    if context_sign * value > threshold:
        return "context"
    if zm_sign * value > threshold:
        return "zm"
    return "neither"


def _summarize(rows: list[dict]) -> dict:
    # Pair orientations share a stable unordered pair key.  A component is a
    # reliable conflict only when both factual orientations reproduce their
    # respective recorded signs.
    factual_ok: dict[tuple[str, str], bool] = {}
    for row in rows:
        for vote in row["votes"]:
            factual_ok[(row["anchor"], vote["component"])] = vote["factual_matches_state"]

    all_votes = []
    reliable_votes = []
    for row in rows:
        for vote in row["votes"]:
            all_votes.append((row, vote))
            donor_ok = factual_ok.get((row["donor_anchor"], vote["component"]), False)
            if vote["factual_matches_state"] and donor_ok:
                reliable_votes.append((row, vote))

    def aggregate(selected):
        values = [vote["mixed_winner"] for _, vote in selected]
        if not values:
            return {"n": 0, "prompt_state": None, "image": None, "neither": None}
        return {
            "n": len(values),
            **{
                key: float(np.mean([value == key for value in values]))
                for key in ("prompt_state", "image", "neither")
            },
        }

    result = {"all": aggregate(all_votes), "reliable_baselines": aggregate(reliable_votes)}
    for arm in ("right", "left"):
        result[f"{arm}_reliable"] = aggregate(
            [(row, vote) for row, vote in reliable_votes if row["arm"] == arm]
        )
    for route in ("donor_context_factual_zm", "factual_context_donor_zm"):
        values = [vote[f"{route}_winner"] for _, vote in reliable_votes]
        result[route] = {
            "n": len(values),
            **{
                key: float(np.mean([value == key for value in values])) if values else None
                for key in ("context", "zm", "neither")
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for anchor in manifest["anchors"]:
        grouped[anchor["cluster_key"]].append(anchor)
    specs = []
    for cluster, anchors in sorted(grouped.items()):
        if len(anchors) != 2:
            continue
        first, second = sorted(anchors, key=lambda row: int(row["pair_index"]))
        for anchor, donor in ((first, second), (second, first)):
            conflicts = _conflicts(anchor["atoms"], donor["atoms"])
            if conflicts:
                specs.append(
                    {"cluster_key": cluster, "anchor": anchor, "donor": donor, "conflicts": conflicts}
                )

    config = AtomicPi05Config(max_token_len=250, fast_action_ce_loss_weight=0.0)
    dataset = build_atomic_dataset(
        (args.dataset_root,),
        norm_assets_dir="/mnt/cunchu/zjq/target",
        norm_asset_id="openpi_norm_compact_accepted_v3",
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        include_fast=False,
    )
    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    tokenizer = _paligemma_tokenizer(config.max_token_len)
    decode = _output_transform(args.dataset_root, config)
    fk_eval.TCP_LOCAL_Z_OFFSET_M = 0.20
    fk = fk_eval.SimpleCR1FK(Path("assets/cr1_right_arm_tcp_red/urdf/CR1ARMR_TCP_RED.urdf"))
    rng = np.random.default_rng(args.seed)
    noise = {
        spec["anchor"]["anchor_key"]: rng.standard_normal(
            (config.action_horizon, config.action_dim), dtype=np.float32
        )
        for spec in specs
    }
    sample_cache = {
        index: dataset[index]
        for spec in specs
        for index in (spec["anchor"]["dataset_index"], spec["donor"]["dataset_index"])
    }
    metadata_cache = {
        index: dataset._raw.metadata(index)  # noqa: SLF001
        for index in sample_cache
    }
    rows = []

    for start in range(0, len(specs), args.batch_size):
        chunk = specs[start : start + args.batch_size]
        real_count = len(chunk)
        if real_count < args.batch_size:
            chunk = chunk + [chunk[-1]] * (args.batch_size - real_count)
        factual_samples = [sample_cache[spec["anchor"]["dataset_index"]] for spec in chunk]
        donor_samples = [sample_cache[spec["donor"]["dataset_index"]] for spec in chunk]
        factual_batch = atomic_collate(factual_samples)
        donor_batch = atomic_collate(donor_samples)
        factual_np, _ = batch_to_observation(factual_batch)
        donor_np, _ = batch_to_observation(donor_batch)
        factual_observation = jax.tree.map(jnp.asarray, factual_np)
        mixed_observation = factual_observation.replace(
            images=jax.tree.map(jnp.asarray, donor_np.images),
            image_masks=jax.tree.map(jnp.asarray, donor_np.image_masks),
        )
        states = np.asarray(factual_batch["state"])
        aligned_prompt = [
            tokenizer.tokenize(spec["anchor"]["atomic_prompt"], state)
            for spec, state in zip(chunk, states, strict=True)
        ]
        predictions = np.asarray(
            jax.device_get(
                _sample_pair(
                    model,
                    factual_observation,
                    mixed_observation,
                    jnp.asarray(np.stack([noise[spec["anchor"]["anchor_key"]] for spec in chunk])),
                    jnp.asarray(np.stack([value[0] for value in aligned_prompt])),
                    jnp.asarray(np.stack([value[1] for value in aligned_prompt])),
                )
            )
        )
        for local_index, spec in enumerate(chunk[:real_count]):
            anchor = spec["anchor"]
            metadata = metadata_cache[anchor["dataset_index"]]
            endpoints = {}
            route_names = (
                "factual",
                "donor_context_factual_zm",
                "factual_context_donor_zm",
                "mixed",
            )
            for prediction_index, name in enumerate(route_names):
                decoded = decode(
                    np.asarray(factual_batch["state"][local_index]),
                    np.asarray(metadata["raw_state"]),
                    predictions[prediction_index, local_index],
                )["actions"]
                trajectory = base._trajectory(  # noqa: SLF001
                    fk, np.asarray(metadata["raw_state"]), decoded, anchor["arm"]
                )
                endpoints[name] = trajectory[50]
            votes = []
            for conflict in spec["conflicts"]:
                index = conflict["index"]
                threshold = conflict["threshold"]
                factual_value = float(endpoints["factual"][index])
                mixed_value = float(endpoints["mixed"][index])
                votes.append(
                    {
                        **conflict,
                        "factual_value": factual_value,
                        "mixed_value": mixed_value,
                        "factual_matches_state": bool(
                            conflict["state_sign"] * factual_value > threshold
                        ),
                        "mixed_winner": _vote(
                            mixed_value,
                            conflict["state_sign"],
                            conflict["image_sign"],
                            threshold,
                        ),
                        "donor_context_factual_zm_value": float(
                            endpoints["donor_context_factual_zm"][index]
                        ),
                        "donor_context_factual_zm_winner": _route_vote(
                            float(endpoints["donor_context_factual_zm"][index]),
                            conflict["image_sign"],
                            conflict["state_sign"],
                            threshold,
                        ),
                        "factual_context_donor_zm_value": float(
                            endpoints["factual_context_donor_zm"][index]
                        ),
                        "factual_context_donor_zm_winner": _route_vote(
                            float(endpoints["factual_context_donor_zm"][index]),
                            conflict["state_sign"],
                            conflict["image_sign"],
                            threshold,
                        ),
                    }
                )
            rows.append(
                {
                    "cluster_key": spec["cluster_key"],
                    "arm": anchor["arm"],
                    "anchor": anchor["anchor_key"],
                    "donor_anchor": spec["donor"]["anchor_key"],
                    "state_atoms": anchor["atoms"],
                    "image_atoms": spec["donor"]["atoms"],
                    "votes": votes,
                }
            )
        print(f"state-image conflict: {min(start + args.batch_size, len(specs))}/{len(specs)}", flush=True)

    payload = {
        "checkpoint": str(args.checkpoint),
        "pair_orientations": len(specs),
        "tcp_offset_m": 0.20,
        "contract": "atomic prompt follows evaluated state/atom A; same state/noise; cross-compose factual/donor-image Context KV and zM; donor B is opposite atom from same state cluster",
        "summary": _summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
