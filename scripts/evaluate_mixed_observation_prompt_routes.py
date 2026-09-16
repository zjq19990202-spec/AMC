#!/usr/bin/env python3
"""Prompt-route rescue under a same-cluster state/image atomic conflict.

The evaluated observation always uses state A and opposite-atom donor images
B.  Empty and aligned atomic-prompt-A prefix passes are cross-composed as:

  empty Context + empty zM
  prompt Context + empty zM
  empty Context + prompt zM
  prompt Context + prompt zM

On a direct sign-conflicting TCP component this reveals whether the state/image
prior initially follows A or B, and whether prompt A rescues the requested
direction through Context or through the atomic zM path.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model

from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import (
    _paligemma_tokenizer,
    atomic_collate,
    batch_to_observation,
    build_atomic_dataset,
)
import evaluate_many_cluster_single_dual_steering as base
from evaluate_prompt_route_attribution import _sample_prompt_routes
from evaluate_state_image_prior_conflict import _conflicts, _vote
import evaluate_zm_fk_trajectory_ablation as fk_eval
from evaluate_zm_fk_trajectory_ablation import _output_transform


ROUTES = ("empty", "prompt_context", "prompt_zm", "prompt_both")


def _reliable_keys(path: Path) -> set[tuple[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    factual = {}
    donor_by_anchor = {}
    for row in payload["rows"]:
        donor_by_anchor[row["anchor"]] = row["donor_anchor"]
        for vote in row["votes"]:
            factual[(row["anchor"], vote["component"])] = vote["factual_matches_state"]
    result = set()
    for (anchor, component), ok in factual.items():
        donor = donor_by_anchor[anchor]
        if ok and factual.get((donor, component), False):
            result.add((anchor, component))
    return result


def _aggregate(rows: list[dict], reliable: set[tuple[str, str]]) -> dict:
    def summarize(votes):
        result = {"n": len(votes)}
        for route in ROUTES:
            values = [vote[f"{route}_winner"] for _, vote in votes]
            result[route] = {
                key: float(np.mean([value == key for value in values])) if values else None
                for key in ("prompt_state", "image", "neither")
            }
        return result

    all_votes = [(row, vote) for row in rows for vote in row["votes"]]
    reliable_votes = [
        (row, vote)
        for row, vote in all_votes
        if (row["anchor"], vote["component"]) in reliable
    ]
    result = {"all": summarize(all_votes), "reliable": summarize(reliable_votes)}
    for arm in ("right", "left"):
        result[f"{arm}_reliable"] = summarize(
            [(row, vote) for row, vote in reliable_votes if row["arm"] == arm]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--reliability-result", type=Path, required=True)
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
        anchor_samples = [sample_cache[spec["anchor"]["dataset_index"]] for spec in chunk]
        donor_samples = [sample_cache[spec["donor"]["dataset_index"]] for spec in chunk]
        anchor_batch = atomic_collate(anchor_samples)
        donor_batch = atomic_collate(donor_samples)
        anchor_np, _ = batch_to_observation(anchor_batch)
        donor_np, _ = batch_to_observation(donor_batch)
        observation = jax.tree.map(jnp.asarray, anchor_np).replace(
            images=jax.tree.map(jnp.asarray, donor_np.images),
            image_masks=jax.tree.map(jnp.asarray, donor_np.image_masks),
        )
        states = np.asarray(anchor_batch["state"])
        empty = [tokenizer.tokenize("", state) for state in states]
        prompt = [
            tokenizer.tokenize(spec["anchor"]["atomic_prompt"], state)
            for spec, state in zip(chunk, states, strict=True)
        ]
        predictions = np.asarray(
            jax.device_get(
                _sample_prompt_routes(
                    model,
                    observation,
                    jnp.asarray(np.stack([noise[spec["anchor"]["anchor_key"]] for spec in chunk])),
                    jnp.asarray(np.stack([value[0] for value in empty])),
                    jnp.asarray(np.stack([value[1] for value in empty])),
                    jnp.asarray(np.stack([value[0] for value in prompt])),
                    jnp.asarray(np.stack([value[1] for value in prompt])),
                )
            )
        )
        for local_index, spec in enumerate(chunk[:real_count]):
            anchor = spec["anchor"]
            metadata = metadata_cache[anchor["dataset_index"]]
            endpoints = {}
            for route_index, route in enumerate(ROUTES):
                decoded = decode(
                    np.asarray(anchor_batch["state"][local_index]),
                    np.asarray(metadata["raw_state"]),
                    predictions[route_index, local_index],
                )["actions"]
                trajectory = base._trajectory(  # noqa: SLF001
                    fk, np.asarray(metadata["raw_state"]), decoded, anchor["arm"]
                )
                endpoints[route] = trajectory[50]
            votes = []
            for conflict in spec["conflicts"]:
                index = conflict["index"]
                votes.append(
                    {
                        **conflict,
                        **{
                            f"{route}_value": float(endpoints[route][index])
                            for route in ROUTES
                        },
                        **{
                            f"{route}_winner": _vote(
                                float(endpoints[route][index]),
                                conflict["state_sign"],
                                conflict["image_sign"],
                                conflict["threshold"],
                            )
                            for route in ROUTES
                        },
                    }
                )
            rows.append(
                {
                    "cluster_key": spec["cluster_key"],
                    "arm": anchor["arm"],
                    "anchor": anchor["anchor_key"],
                    "donor_anchor": spec["donor"]["anchor_key"],
                    "prompt_state_atoms": anchor["atoms"],
                    "image_atoms": spec["donor"]["atoms"],
                    "votes": votes,
                }
            )
        print(f"mixed prompt routes: {min(start + args.batch_size, len(specs))}/{len(specs)}", flush=True)

    reliable = _reliable_keys(args.reliability_result)
    payload = {
        "checkpoint": str(args.checkpoint),
        "pair_orientations": len(specs),
        "tcp_offset_m": 0.20,
        "contract": "state and atomic prompt follow A; all three images come from opposite atom B in same state cluster; prompt injected into Context only, zM only, both, or neither",
        "summary": _aggregate(rows, reliable),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
