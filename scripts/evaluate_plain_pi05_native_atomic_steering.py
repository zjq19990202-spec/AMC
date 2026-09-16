#!/usr/bin/env python3
"""Paired native-atomic steering sweep for a stock PI0.5 checkpoint.

The evaluator is intentionally limited to four prompt routes per reviewed
anchor: empty, native SUBtask, the exact native atomic prompt, and its exact
semantic sign reversal. Images, normalized state, sampling noise, decoder,
and FK remain fixed within every anchor/repeat pair.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config

from atomic_latent_vla.pi05.training_data import (
    _paligemma_tokenizer,
    atomic_collate,
    batch_to_observation,
    build_atomic_dataset,
)
from evaluate_500_anchor_atomic_sweep import _canonical_prompt
import evaluate_many_cluster_single_dual_steering as steering
import evaluate_zm_fk_trajectory_ablation as fk_eval
from evaluate_zm_fk_trajectory_ablation import _output_transform


@nnx.jit
def _sample(model, observation, noise):
    return model.sample_actions(
        jax.random.key(0), observation, num_steps=10, noise=noise
    )


def _reverse_atom(atom: str) -> str:
    if atom.endswith("_pos"):
        return atom.removesuffix("_pos") + "_neg"
    if atom.endswith("_neg"):
        return atom.removesuffix("_neg") + "_pos"
    raise ValueError(f"cannot reverse motion atom {atom!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(bytes.fromhex(_sha256_file(item)))
    return digest.hexdigest()


def _evaluate(
    checkpoint: Path,
    config: Pi0Config,
    dataset,
    dataset_root: Path,
    row_specs: list[dict],
    noises: np.ndarray,
    *,
    batch_size: int,
    norm_assets_dir: Path,
    norm_asset_id: str,
) -> list[dict]:
    params = _model.restore_params(checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    decoder = _output_transform(
        dataset_root,
        config,
        norm_assets_dir=norm_assets_dir,
        norm_asset_id=norm_asset_id,
    )
    fk_eval.TCP_LOCAL_Z_OFFSET_M = 0.20
    fk = fk_eval.SimpleCR1FK(
        Path("assets/cr1_right_arm_tcp_red/urdf/CR1ARMR_TCP_RED.urdf")
    )
    tokenizer = _paligemma_tokenizer(config.max_token_len)
    sample_cache = {
        index: dataset[index] for index in {row["dataset_index"] for row in row_specs}
    }
    metadata_cache = {
        index: dataset._raw.metadata(index)  # noqa: SLF001
        for index in sample_cache
    }
    outputs = []
    for start in range(0, len(row_specs), batch_size):
        rows = row_specs[start : start + batch_size]
        real_count = len(rows)
        if real_count < batch_size:
            rows = rows + [rows[-1]] * (batch_size - real_count)
        samples = [sample_cache[row["dataset_index"]] for row in rows]
        batch = atomic_collate(samples)
        observation_np, _ = batch_to_observation(batch)
        tokenized = [
            tokenizer.tokenize(row["prompt"], np.asarray(sample["state"]))
            for row, sample in zip(rows, samples, strict=True)
        ]
        observation = jax.tree.map(jnp.asarray, observation_np).replace(
            tokenized_prompt=jnp.asarray(np.stack([item[0] for item in tokenized])),
            tokenized_prompt_mask=jnp.asarray(np.stack([item[1] for item in tokenized])),
        )
        chunk_noise = noises[start : start + real_count]
        if real_count < batch_size:
            chunk_noise = np.concatenate(
                [
                    chunk_noise,
                    np.repeat(chunk_noise[-1:], batch_size - real_count, axis=0),
                ],
                axis=0,
            )
        predictions = np.asarray(
            jax.device_get(_sample(model, observation, jnp.asarray(chunk_noise)))
        )
        for local_index, (row, prediction) in enumerate(
            zip(rows[:real_count], predictions[:real_count], strict=True)
        ):
            metadata = metadata_cache[row["dataset_index"]]
            actions = decoder(
                np.asarray(batch["state"][local_index]),
                np.asarray(metadata["raw_state"]),
                prediction[:, :16],
            )["actions"]
            trajectory = steering._trajectory(  # noqa: SLF001
                fk,
                np.asarray(metadata["raw_state"]),
                np.asarray(actions),
                row["arm"],
            )
            outputs.append(
                {
                    **{key: value for key, value in row.items() if key != "prompt"},
                    "prompt": row["prompt"],
                    "trajectory_twist": {
                        str(step): trajectory[step].tolist() for step in (25, 50)
                    },
                }
            )
        print(f"{min(start + batch_size, len(row_specs))}/{len(row_specs)}", flush=True)
    del model, params
    gc.collect()
    jax.clear_caches()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--noise-repeats", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--cluster-seen-all",
        action="store_true",
        help="Sweep every single/compatible-dual mode recorded for the anchor cluster.",
    )
    parser.add_argument(
        "--norm-assets-dir", type=Path, default=Path("/mnt/cunchu/zjq/target")
    )
    parser.add_argument(
        "--norm-asset-id", default="openpi_norm_union2375_allframes_v1"
    )
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0,num-shards)")

    selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    all_anchors = selection["anchors"]
    anchors = [
        anchor
        for index, anchor in enumerate(all_anchors)
        if index % args.num_shards == args.shard_index
    ]
    if not anchors:
        raise RuntimeError("this shard has no anchors")

    config = Pi0Config(pi05=True, action_dim=32, action_horizon=50, max_token_len=200)
    dataset = build_atomic_dataset(
        (args.dataset_root,),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        action_horizon=50,
        max_token_len=200,
        include_fast=False,
        pad_subtask_horizon=True,
    )

    rng = np.random.default_rng(args.seed)
    noise_by_key = {}
    for anchor in all_anchors:
        for repeat in range(args.noise_repeats):
            noise_by_key[(anchor["anchor_key"], repeat)] = rng.standard_normal(
                (config.action_horizon, config.action_dim), dtype=np.float32
            )

    row_specs = []
    noises = []
    for anchor in anchors:
        reverse_atoms = [_reverse_atom(atom) for atom in anchor["atoms"]]
        for repeat in range(args.noise_repeats):
            if args.cluster_seen_all:
                variants = [
                    ("empty", "", []),
                    ("subtask_no_atom", anchor["subtask_prompt"], []),
                ]
                variants.extend(
                    (
                        "cluster_seen_" + "+".join(atoms),
                        _canonical_prompt(anchor, atoms),
                        list(atoms),
                    )
                    for atoms in map(tuple, anchor["cluster_isolated_modes"])
                    if len(atoms) in (1, 2)
                )
            else:
                variants = (
                    ("empty", "", []),
                    ("subtask_no_atom", anchor["subtask_prompt"], []),
                    ("native_atomic", anchor["atomic_prompt"], list(anchor["atoms"])),
                    ("native_reverse", anchor["reverse_prompt"], reverse_atoms),
                )
            for variant, prompt, requested_atoms in variants:
                row_specs.append(
                    {
                        **anchor,
                        "repeat": repeat,
                        "variant": variant,
                        "requested_atoms": requested_atoms,
                        "prompt": prompt,
                    }
                )
                noises.append(noise_by_key[(anchor["anchor_key"], repeat)])

    outputs = _evaluate(
        args.checkpoint,
        config,
        dataset,
        args.dataset_root,
        row_specs,
        np.stack(noises),
        batch_size=args.batch_size,
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
    )
    payload = {
        "checkpoint_name": args.checkpoint_name,
        "checkpoint": str(args.checkpoint),
        "selection_manifest": str(args.selection_manifest),
        "selection_manifest_sha256": _sha256_file(args.selection_manifest),
        "selection_anchor_count": len(all_anchors),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "anchor_count": len(anchors),
        "variants_per_anchor": None if args.cluster_seen_all else 4,
        "cluster_seen_all": args.cluster_seen_all,
        "noise_repeats": args.noise_repeats,
        "norm_assets_dir": str(args.norm_assets_dir),
        "norm_asset_id": args.norm_asset_id,
        "norm_tree_sha256": _sha256_tree(
            args.norm_assets_dir / args.norm_asset_id
        ),
        "max_token_len": 200,
        "tcp_offset_m": 0.20,
        "stored_steps": [25, 50],
        "outputs": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output.parent / "run_contract.md").write_text(
        "# Evaluation contract\n\n"
        f"- checkpoint: `{args.checkpoint}`\n"
        f"- checkpoint name: `{args.checkpoint_name}`\n"
        f"- dataset root: `{args.dataset_root}`\n"
        f"- norm: `{args.norm_assets_dir / args.norm_asset_id}`\n"
        f"- norm tree sha256: `{payload['norm_tree_sha256']}`\n"
        f"- selection manifest: `{args.selection_manifest}`\n"
        f"- selection sha256: `{payload['selection_manifest_sha256']}`\n"
        "- prompt routes: exact native atomic, exact semantic sign reverse, native "
        "SUBtask, and empty prompt\n"
        "- paired controls: image, raw/normalized state, sampler, decoder, and flow "
        "noise fixed within anchor/repeat\n"
        "- action horizon: `50`; stored steps: `25, 50`\n"
        "- action decode: unnormalize joint delta, then add to raw state\n"
        "- TCP offset: `0.20 m`\n"
        f"- seed: `{args.seed}`; noise repeats: `{args.noise_repeats}`\n"
        f"- command: `{' '.join(sys.argv)}`\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"anchors": len(anchors), "trajectories": len(outputs), "output": str(args.output)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
