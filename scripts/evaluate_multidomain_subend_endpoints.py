#!/usr/bin/env python3
"""Evaluate paired per-row SUB prompts for stock or Atomic PI0.5 without plots."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config

from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import (
    _paligemma_tokenizer,
    atomic_collate,
    batch_to_observation,
    build_atomic_dataset,
)
from evaluate_global_episode_chunks import _endpoint_tcp_from_actions, _output_transform


def _parse_dataset(value: str) -> tuple[str, Path]:
    name, path = value.split("=", 1)
    return name, Path(path)


@nnx.jit
def _sample_model(model, observation, noise):
    return model.sample_actions(jax.random.key(0), observation, num_steps=10, noise=noise)


def _repeat_observation(observation: _model.Observation, count: int, tokens, masks):
    repeated = jax.tree.map(lambda value: jnp.repeat(value, count, axis=0), observation)
    return repeated.replace(tokenized_prompt=tokens, tokenized_prompt_mask=masks)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_row(dataset, episode: int, frame: int):
    raw = dataset._raw  # noqa: SLF001
    for dataset_index in range(len(raw)):
        data_index = raw._data_index(dataset_index)  # noqa: SLF001
        if (
            int(raw.base._episode_index[data_index]) == episode  # noqa: SLF001
            and int(raw.base._frame_index[data_index]) == frame  # noqa: SLF001
        ):
            metadata = raw.metadata(dataset_index)
            query, _ = raw.base._get_query_indices(data_index, episode)  # noqa: SLF001
            return dataset_index, data_index, np.asarray(query["action"], dtype=np.int64), metadata
    raise RuntimeError(f"missing episode={episode} frame={frame}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=("stock", "atomic"), required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", action="append", required=True, help="NAME=ROOT")
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--seed", type=int, default=20260834)
    parser.add_argument("--max-token-len", type=int)
    parser.add_argument("--coefficient-target-kind", choices=("tcp_twist", "joint_delta"), default="joint_delta")
    parser.add_argument("--coefficient-target-dim", type=int, default=14)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    dataset_roots = dict(_parse_dataset(value) for value in args.dataset)
    manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    if set(dataset_roots) != set(manifest["datasets"]):
        raise ValueError("--dataset names must exactly match manifest dataset keys")
    max_token_len = args.max_token_len or (200 if args.model_kind == "stock" else 192)
    if args.model_kind == "stock":
        config = Pi0Config(pi05=True, action_dim=32, action_horizon=50, max_token_len=max_token_len)
    else:
        config = AtomicPi05Config(
            max_token_len=max_token_len,
            fast_action_ce_loss_weight=0.0,
            coefficient_target_kind=args.coefficient_target_kind,
            coefficient_target_dim=args.coefficient_target_dim,
            enable_layerwise_atomic_flow=True,
        )
    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    tokenizer = _paligemma_tokenizer(max_token_len)

    rows = []
    manifest_hash = _sha256(args.selection_manifest)
    if args.resume and args.output.exists():
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        if prior.get("selection_manifest_sha256") != manifest_hash:
            raise ValueError("cannot resume: selection manifest hash changed")
        rows = prior.get("rows", [])
    done = {
        (str(row["dataset"]), int(row["episode"]), int(row["frame"])) for row in rows
    }
    total = sum(len(values) for values in manifest["datasets"].values())

    def write_report() -> None:
        report = {
            "model_name": args.model_name,
            "model_kind": args.model_kind,
            "checkpoint": str(args.checkpoint.resolve()),
            "dataset_roots": {name: str(path.resolve()) for name, path in dataset_roots.items()},
            "selection_manifest": str(args.selection_manifest.resolve()),
            "selection_manifest_sha256": manifest_hash,
            "seed": args.seed,
            "max_token_len": max_token_len,
            "rows_complete": len(rows),
            "rows_expected": total,
            "rows": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(args.output)

    for dataset_name, root in dataset_roots.items():
        specifications = manifest["datasets"][dataset_name]
        if all((dataset_name, int(spec["episode"]), int(spec["frame"])) in done for spec in specifications):
            continue
        dataset = build_atomic_dataset(
            (root,),
            norm_assets_dir=args.norm_assets_dir,
            norm_asset_id=args.norm_asset_id,
            action_horizon=50,
            max_token_len=max_token_len,
            include_fast=False,
            pad_subtask_horizon=True,
        )
        decoder = _output_transform(
            root,
            config,
            norm_assets_dir=args.norm_assets_dir,
            norm_asset_id=args.norm_asset_id,
        )
        raw = dataset._raw  # noqa: SLF001
        for spec in specifications:
            episode = int(spec["episode"])
            frame = int(spec["frame"])
            key = (dataset_name, episode, frame)
            if key in done:
                continue
            active_arm = str(spec["active_arm"])
            dataset_index, data_index, _, metadata = _find_row(dataset, episode, frame)
            item = dataset[dataset_index]
            batch = atomic_collate([item])
            obs_np, _ = batch_to_observation(batch)
            observation = jax.tree.map(jnp.asarray, obs_np)
            prompt_map = {str(k): str(v) for k, v in spec["prompt_variants"].items()}
            prompt_names = list(prompt_map)
            token_rows, mask_rows = zip(
                *(tokenizer.tokenize(prompt_map[name], np.asarray(item["state"])) for name in prompt_names),
                strict=True,
            )
            noise = jax.random.normal(
                jax.random.fold_in(
                    jax.random.key(args.seed),
                    (sum(ord(char) for char in dataset_name) * 1_000_000 + episode * 10_000 + frame),
                ),
                (1, 50, config.action_dim),
            )
            paired = _repeat_observation(
                observation,
                len(prompt_names),
                jnp.asarray(np.stack(token_rows)),
                jnp.asarray(np.stack(mask_rows)),
            )
            normalized = np.asarray(
                jax.device_get(
                    _sample_model(model, paired, jnp.repeat(noise, len(prompt_names), axis=0))
                )
            )
            pose_offset = 0 if active_arm == "left" else 24
            current_xyz = np.asarray(raw.tcp_pose[data_index, pose_offset : pose_offset + 3])
            endpoints = {}
            for prompt_index, prompt_name in enumerate(prompt_names):
                decoded = np.asarray(
                    decoder(
                        np.asarray(batch["state"][0]),
                        np.asarray(metadata["raw_state"]),
                        normalized[prompt_index, :, :16],
                    )["actions"]
                )
                trajectory = _endpoint_tcp_from_actions(raw, data_index, decoded, active_arm)
                endpoints[prompt_name] = np.asarray(trajectory[-1], dtype=float).tolist()
            rows.append(
                {
                    "dataset": dataset_name,
                    "episode": episode,
                    "frame": frame,
                    "frame_stage": spec.get("frame_stage"),
                    "active_arm": active_arm,
                    "native_subtask": str(metadata["subtask_prompt"]),
                    "prompt_texts": prompt_map,
                    "start_xyz_m": current_xyz.tolist(),
                    "endpoint_xyz_m": endpoints,
                }
            )
            done.add(key)
            print(
                f"{args.model_name}: {len(rows)}/{total} {dataset_name} "
                f"episode={episode} frame={frame}",
                flush=True,
            )
            if len(rows) % 10 == 0:
                write_report()
        del dataset, decoder, raw
        gc.collect()
    rows.sort(key=lambda row: (str(row["dataset"]), int(row["episode"]), int(row["frame"])))
    write_report()


if __name__ == "__main__":
    main()
