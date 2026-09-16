#!/usr/bin/env python3
"""Evaluate stock PI0.5-like or Atomic PI0.5 endpoints without plotting."""

from __future__ import annotations

import argparse
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
from evaluate_fruit_target_switch_multiframe import _native_template_prompts
from evaluate_global_episode_chunks import _endpoint_tcp_from_actions, _output_transform


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
    """Resolve a public episode/frame through the masked raw-view mapping."""
    raw = dataset._raw  # noqa: SLF001
    for dataset_index in range(len(raw)):
        data_index = raw._data_index(dataset_index)  # noqa: SLF001
        if (
            int(raw.base._episode_index[data_index]) == episode  # noqa: SLF001
            and int(raw.base._frame_index[data_index]) == frame  # noqa: SLF001
        ):
            metadata = raw.metadata(dataset_index)
            query, _ = raw.base._get_query_indices(data_index, episode)  # noqa: SLF001
            return (
                dataset_index,
                data_index,
                np.asarray(query["action"], dtype=np.int64),
                metadata,
            )
    raise RuntimeError(f"missing episode={episode} frame={frame}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=("stock", "atomic"), required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--selection-dataset-name", default="fruit_macro")
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-token-len", type=int)
    parser.add_argument("--coefficient-target-kind", choices=("tcp_twist", "joint_delta"), default="joint_delta")
    parser.add_argument("--coefficient-target-dim", type=int, default=14)
    args = parser.parse_args()

    manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    frame_specs = manifest["datasets"][args.selection_dataset_name]
    prompt_targets = list(manifest["target_names"])
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
    dataset = build_atomic_dataset(
        (args.dataset_root,),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        action_horizon=50,
        max_token_len=max_token_len,
        include_fast=False,
        pad_subtask_horizon=True,
    )
    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    tokenizer = _paligemma_tokenizer(max_token_len)
    decoder = _output_transform(
        args.dataset_root,
        config,
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
    )
    raw = dataset._raw  # noqa: SLF001

    existing_rows = []
    if args.resume and args.output.exists():
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        if prior.get("selection_manifest_sha256") != _sha256(args.selection_manifest):
            raise ValueError("cannot resume: selection manifest hash changed")
        existing_rows = prior.get("rows", [])
    done = {(int(row["episode"]), int(row["frame"])) for row in existing_rows}
    rows = list(existing_rows)

    def write_report() -> None:
        report = {
            "model_name": args.model_name,
            "model_kind": args.model_kind,
            "checkpoint": str(args.checkpoint.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "selection_manifest": str(args.selection_manifest.resolve()),
            "selection_manifest_sha256": _sha256(args.selection_manifest),
            "seed": args.seed,
            "max_token_len": max_token_len,
            "rows_complete": len(rows),
            "rows_expected": len(frame_specs),
            "rows": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(args.output)

    for spec_index, spec in enumerate(frame_specs):
        episode = int(spec["episode"])
        frame = int(spec["frame"])
        key = (episode, frame)
        if key in done:
            continue
        active_arm = str(spec["active_arm"])
        dataset_index, data_index, _, metadata = _find_row(dataset, episode, frame)
        item = dataset[dataset_index]
        batch = atomic_collate([item])
        obs_np, _ = batch_to_observation(batch)
        observation = jax.tree.map(jnp.asarray, obs_np)
        original_prompt = str(metadata["subtask_prompt"])
        prompt_map, native_target = _native_template_prompts(original_prompt, prompt_targets)
        prompt_names = list(prompt_map)
        token_rows, mask_rows = zip(
            *(
                tokenizer.tokenize(prompt_map[name], np.asarray(item["state"]))
                for name in prompt_names
            ),
            strict=True,
        )
        noise = jax.random.normal(
            jax.random.fold_in(jax.random.key(args.seed), episode * 10000 + frame),
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
        endpoints: dict[str, list[float]] = {}
        displacements: dict[str, list[float]] = {}
        for prompt_index, name in enumerate(prompt_names):
            decoded = np.asarray(
                decoder(
                    np.asarray(batch["state"][0]),
                    np.asarray(metadata["raw_state"]),
                    normalized[prompt_index, :, :16],
                )["actions"]
            )
            trajectory = _endpoint_tcp_from_actions(raw, data_index, decoded, active_arm)
            endpoint = np.asarray(trajectory[-1], dtype=float)
            endpoints[name] = endpoint.tolist()
            displacements[name] = ((endpoint - current_xyz) * 1000.0).tolist()

        rows.append(
            {
                "episode": episode,
                "frame": frame,
                "frame_stage": spec.get("frame_stage"),
                "active_arm": active_arm,
                "source_dataset": spec.get("source_dataset"),
                "native_subtask": original_prompt,
                "native_target": native_target,
                "prompt_texts": prompt_map,
                "start_xyz_m": current_xyz.tolist(),
                "endpoint_xyz_m": endpoints,
                "endpoint_displacement_mm": displacements,
            }
        )
        done.add(key)
        print(
            f"{args.model_name}: {len(rows)}/{len(frame_specs)} "
            f"episode={episode} frame={frame}",
            flush=True,
        )
        if len(rows) % 10 == 0:
            write_report()
    rows.sort(key=lambda row: (int(row["episode"]), int(row["frame"])))
    write_report()


if __name__ == "__main__":
    main()
