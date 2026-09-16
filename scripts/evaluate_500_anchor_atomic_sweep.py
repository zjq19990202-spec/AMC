#!/usr/bin/env python3
"""Sweep canonical atomic prompts on the reviewed 500 isolated-arm anchors.

Each distributed shard owns a disjoint subset of anchors.  Within one anchor,
the observation, raw robot state, and flow noise stay fixed while the text is
changed among:

* an empty prompt;
* the native subtask prompt (no explicit atomic label);
* all twelve canonical single-atom prompts;
* all sixty valid two-atom prompts over distinct motion components;
* the reviewed native single/dual atomic prompt;
* its semantic sign reversal.

The output stores TCP twists at horizon steps 25 and 50.  Four independent
processes can therefore run on four GPUs without collective communication.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from atomic_latent_vla.atomic import ATOMIC_BASE_INSTRUCTIONS, AtomicSkill
from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import build_atomic_dataset

import evaluate_many_cluster_single_dual_steering as base


MOTION_ATOMS = tuple(skill.value for skill in AtomicSkill if skill != AtomicSkill.STAY)


def _component(atom: str) -> tuple[str, str]:
    family, axis, _ = atom.split("_")
    return family, axis


DUAL_ATOMS = tuple(
    pair
    for pair in itertools.combinations(MOTION_ATOMS, 2)
    if _component(pair[0]) != _component(pair[1])
)


def _motion_instruction(atoms: tuple[str, ...]) -> str:
    commands = [
        ATOMIC_BASE_INSTRUCTIONS[AtomicSkill(atom)].rstrip(".") for atom in atoms
    ]
    if len(commands) == 1:
        return commands[0] + "."
    second = commands[1][0].lower() + commands[1][1:]
    return f"{commands[0]} and simultaneously {second}."


def _canonical_prompt(anchor: dict, atoms: tuple[str, ...]) -> str:
    """Match the arm-format distribution used by the evaluated checkpoints."""

    arm = anchor["arm"]
    instruction = _motion_instruction(atoms)
    if anchor["other_status"] == "unlabeled":
        # Existing 60K/70K checkpoints saw sole-arm atomic text without an arm
        # prefix.  Keep that historical format for a checkpoint-faithful test.
        return instruction
    other = "left" if arm == "right" else "right"
    return (
        f"{arm.capitalize()} arm: {instruction} "
        f"{other.capitalize()} arm: Keep the {other} TCP stationary in its "
        "current base-frame pose."
    )


def _variants(
    anchor: dict,
    *,
    cluster_seen_only: bool = False,
) -> list[tuple[str, str, list[str]]]:
    rows = [
        ("empty", "", []),
        ("subtask_no_atom", anchor["subtask_prompt"], []),
    ]
    if cluster_seen_only:
        modes = [
            tuple(mode)
            for mode in anchor.get("cluster_isolated_modes", [])
            if len(mode) in (1, 2)
        ]
        if not modes:
            raise ValueError(
                "cluster-seen sweep requires cluster_isolated_modes in the selection manifest"
            )
        rows.extend(
            (
                "cluster_seen_" + "+".join(atoms),
                _canonical_prompt(anchor, atoms),
                list(atoms),
            )
            for atoms in modes
        )
        return rows
    rows.extend(
        (f"canonical_single_{atom}", _canonical_prompt(anchor, (atom,)), [atom])
        for atom in MOTION_ATOMS
    )
    rows.extend(
        (
            "canonical_dual_" + "+".join(atoms),
            _canonical_prompt(anchor, atoms),
            list(atoms),
        )
        for atoms in DUAL_ATOMS
    )
    rows.extend(
        (
            ("native_atomic", anchor["atomic_prompt"], list(anchor["atoms"])),
            ("native_reverse", anchor["reverse_prompt"], []),
        )
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="dual70")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--noise-repeats", type=int, default=1)
    parser.add_argument(
        "--cluster-seen-only",
        action="store_true",
        help="Evaluate only single/dual modes actually seen with the opposite arm quiet in this cluster.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-token-len", type=int, default=250)
    parser.add_argument(
        "--coefficient-target-kind",
        choices=("tcp_twist", "joint_delta"),
        default="tcp_twist",
    )
    parser.add_argument("--coefficient-target-dim", type=int, default=12)
    parser.add_argument(
        "--enable-layerwise-atomic-flow",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--norm-assets-dir",
        type=Path,
        default=Path("/mnt/cunchu/zjq/target"),
    )
    parser.add_argument(
        "--norm-asset-id",
        default="openpi_norm_compact_accepted_v3",
    )
    parser.add_argument(
        "--atomic-composition-sidecar",
        default=None,
        help="Optional target sidecar ID used by the evaluated training data path.",
    )
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")

    selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    all_anchors = selection["anchors"]
    anchors = [
        anchor
        for index, anchor in enumerate(all_anchors)
        if index % args.num_shards == args.shard_index
    ]
    if not anchors:
        raise RuntimeError("this shard has no anchors")

    config = AtomicPi05Config(
        max_token_len=args.max_token_len,
        fast_action_ce_loss_weight=0.0,
        coefficient_target_kind=args.coefficient_target_kind,
        coefficient_target_dim=args.coefficient_target_dim,
        enable_layerwise_atomic_flow=args.enable_layerwise_atomic_flow,
    )
    dataset = build_atomic_dataset(
        (args.dataset_root,),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        include_fast=False,
        atomic_composition_sidecar=args.atomic_composition_sidecar,
    )

    rng = np.random.default_rng(args.seed)
    # Generate noise in global-anchor order so sharding cannot alter the noise
    # assigned to a reviewed anchor.
    noise_by_anchor_repeat: dict[tuple[str, int], np.ndarray] = {}
    for anchor in all_anchors:
        for repeat in range(args.noise_repeats):
            noise_by_anchor_repeat[(anchor["anchor_key"], repeat)] = rng.standard_normal(
                (config.action_horizon, config.action_dim), dtype=np.float32
            )

    row_specs: list[dict] = []
    noises: list[np.ndarray] = []
    for anchor in anchors:
        for repeat in range(args.noise_repeats):
            noise = noise_by_anchor_repeat[(anchor["anchor_key"], repeat)]
            for variant, prompt, requested_atoms in _variants(
                anchor,
                cluster_seen_only=args.cluster_seen_only,
            ):
                row_specs.append(
                    {
                        **{key: value for key, value in anchor.items() if key != "state_7d"},
                        "repeat": repeat,
                        "variant": variant,
                        "requested_atoms": requested_atoms,
                        "prompt": prompt,
                    }
                )
                noises.append(noise)

    outputs = base._evaluate_checkpoint(  # noqa: SLF001
        args.checkpoint_name,
        args.checkpoint,
        config,
        dataset,
        args.dataset_root,
        row_specs,
        np.stack(noises),
        batch_size=args.batch_size,
        stored_steps=(25, 50),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
    )
    payload = {
        "checkpoint_name": args.checkpoint_name,
        "checkpoint": str(args.checkpoint),
        "selection_manifest": str(args.selection_manifest),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "anchor_count": len(anchors),
        "variant_count_per_anchor": len(
            _variants(anchors[0], cluster_seen_only=args.cluster_seen_only)
        ),
        "cluster_seen_only": args.cluster_seen_only,
        "canonical_single_count": len(MOTION_ATOMS),
        "canonical_dual_count": len(DUAL_ATOMS),
        "noise_repeats": args.noise_repeats,
        "max_token_len": args.max_token_len,
        "coefficient_target_kind": args.coefficient_target_kind,
        "coefficient_target_dim": args.coefficient_target_dim,
        "enable_layerwise_atomic_flow": args.enable_layerwise_atomic_flow,
        "norm_assets_dir": str(args.norm_assets_dir),
        "norm_asset_id": args.norm_asset_id,
        "atomic_composition_sidecar": args.atomic_composition_sidecar,
        "tcp_offset_m": 0.20,
        "stored_steps": [25, 50],
        "outputs": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "shard": args.shard_index,
                "anchors": len(anchors),
                "rows": len(outputs),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
