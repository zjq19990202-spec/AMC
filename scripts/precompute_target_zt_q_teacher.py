#!/usr/bin/env python3
"""Precompute frozen atomic-prompt zT Q1/Q3 directions for target ZM.

The output is indexed by the underlying LeRobot data row. Only rows with at
least one strict atomic arm are encoded. A mixed strict/drop row stores both
arm directions, but ZM masks cosine distillation to the strict arm and keeps
Top-5 KL on the dropped arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from torch.utils.data import DataLoader

from openpi.models import model as _model

from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import (
    _AtomicRawDataset,
    build_atomic_text_dataset,
    text_batch_to_observation,
)


def _encode_impl(model, state, prompt_tokens, prompt_mask):
    observation = _model.Observation(
        images={},
        image_masks={},
        state=state,
        tokenized_prompt=prompt_tokens,
        tokenized_prompt_mask=prompt_mask,
        token_ar_mask=None,
        token_loss_mask=None,
    )
    query_hidden, _, _, _ = model._prefix_forward(observation)  # noqa: SLF001
    active_state = model._controlled_state(observation.state)  # noqa: SLF001
    return model.queries.text_arm_directions(query_hidden, active_state)


def _shard_array(value, device_count: int, batch_size: int):
    """Pad and split one host array over the pmap device axis."""

    if batch_size <= 0:
        raise ValueError("cannot encode an empty teacher batch")
    padded_size = ((batch_size + device_count - 1) // device_count) * device_count
    array = np.asarray(value)
    if padded_size != batch_size:
        pad_width = [(0, padded_size - batch_size)] + [(0, 0)] * (array.ndim - 1)
        array = np.pad(array, pad_width, mode="edge")
    return jnp.asarray(
        array.reshape(device_count, padded_size // device_count, *array.shape[1:])
    )


def _collate(rows):
    keys = (
        "state",
        "atomic_prompt_tokens",
        "atomic_prompt_mask",
        "data_index",
        "atomic_supervision_mask",
    )
    return {key: np.stack([np.asarray(row[key]) for row in rows]) for key in keys}


def _resolve_params(checkpoint: Path) -> Path:
    params = checkpoint / "params"
    return params if params.is_dir() else checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument(
        "--atomic-composition-sidecar",
        type=Path,
        default=None,
        help=(
            "Dataset meta sidecar used by ZT/ZM to restore strict Stay rows. "
            "It must match the sidecar passed to both training stages."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of visible GPUs used for replicated data-parallel encoding.",
    )
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--max-token-len", type=int, default=200)
    parser.add_argument(
        "--coefficient-target", choices=("joint_delta", "tcp_twist"), default="joint_delta"
    )
    args = parser.parse_args()

    params_path = _resolve_params(args.checkpoint)
    if not params_path.is_dir():
        raise FileNotFoundError(f"checkpoint params not found: {params_path}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite zT teacher sidecar: {args.output}")
    if args.batch_size <= 0 or args.devices <= 0:
        raise ValueError("batch-size and devices must be positive")
    if args.max_token_len < 200:
        raise ValueError("full-32-D PI0.5 state tokenization requires max-token-len >= 200")
    if args.batch_size % args.devices:
        raise ValueError("batch-size must be divisible by devices")
    config = AtomicPi05Config(
        max_token_len=args.max_token_len,
        coefficient_target_kind=args.coefficient_target,
        coefficient_target_dim=14 if args.coefficient_target == "joint_delta" else 12,
        fast_action_ce_loss_weight=0.0,
        text_flow_loss_weight=1.0,
        coefficient_loss_weight=0.0,
    )
    raw = _AtomicRawDataset(args.dataset_root, config.action_horizon)
    total_rows = len(raw.base._states)
    dataset = build_atomic_text_dataset(
        (args.dataset_root,),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        atomic_composition_sidecar=args.atomic_composition_sidecar,
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        include_fast=False,
        reliable_atomic_only=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
        collate_fn=_collate,
    )
    # Spawn PyTorch workers before JAX initializes its multithreaded GPU
    # runtime. Forking afterward is both slow for the restored 2B model and
    # unsafe according to Python/JAX's multiprocessing contract.
    data_iter = iter(loader)

    if args.devices > jax.device_count():
        raise ValueError(
            f"requested {args.devices} devices but only {jax.device_count()} are visible"
        )
    params = _model.restore_params(params_path, dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    codebook = np.asarray(jax.device_get(model.codebook.value), dtype=np.float32)
    codebook /= np.maximum(np.linalg.norm(codebook, axis=-1, keepdims=True), 1e-8)
    devices = jax.devices()[: args.devices]
    encode = (
        nnx.jit(_encode_impl)
        if args.devices == 1
        else nnx.pmap(
            _encode_impl,
            in_axes=(None, 0, 0, 0),
            out_axes=0,
            devices=devices,
        )
    )

    temporary = args.output.with_name(f".{args.output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary sidecar exists: {temporary}")
    temporary.mkdir(parents=True)
    directions = np.lib.format.open_memmap(
        temporary / "directions.npy",
        mode="w+",
        dtype=np.float16,
        shape=(total_rows, 2, config.latent_dim),
    )
    valid = np.lib.format.open_memmap(
        temporary / "valid.npy", mode="w+", dtype=np.bool_, shape=(total_rows,)
    )
    directions[:] = 0
    valid[:] = False

    encoded = 0
    for batch in data_iter:
        observation = text_batch_to_observation(batch)
        if args.devices == 1:
            encoded_values = encode(
                model,
                jnp.asarray(observation.state),
                jnp.asarray(observation.tokenized_prompt),
                jnp.asarray(observation.tokenized_prompt_mask),
            )
        else:
            original_batch_size = int(observation.state.shape[0])
            encoded_values = encode(
                model,
                _shard_array(observation.state, args.devices, original_batch_size),
                _shard_array(
                    observation.tokenized_prompt, args.devices, original_batch_size
                ),
                _shard_array(
                    observation.tokenized_prompt_mask,
                    args.devices,
                    original_batch_size,
                ),
            )
            encoded_values = encoded_values.reshape(
                -1, encoded_values.shape[-2], encoded_values.shape[-1]
            )[:original_batch_size]
        values = np.asarray(jax.device_get(encoded_values), dtype=np.float32)
        indices = np.asarray(batch["data_index"], dtype=np.int64)
        strict = np.asarray(batch["atomic_supervision_mask"], dtype=np.bool_)
        if not np.all(np.any(strict, axis=1)):
            raise AssertionError("reliable zT teacher loader emitted a row without a strict arm")
        if np.any(valid[indices]):
            raise ValueError("duplicate LeRobot data index in zT teacher generation")
        directions[indices] = values.astype(np.float16)
        valid[indices] = True
        encoded += len(indices)

    directions.flush()
    valid.flush()
    if int(np.sum(valid)) != encoded or encoded != len(dataset):
        raise AssertionError(
            f"teacher coverage mismatch: valid={int(np.sum(valid))} "
            f"encoded={encoded} dataset={len(dataset)}"
        )
    np.save(temporary / "codebook.npy", codebook)
    manifest = {
        "version": 1,
        "prompt": "atomic",
        "checkpoint_params": str(params_path.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "norm_assets_dir": str(args.norm_assets_dir.resolve()),
        "norm_asset_id": args.norm_asset_id,
        "atomic_composition_sidecar": (
            None
            if args.atomic_composition_sidecar is None
            else str(args.atomic_composition_sidecar)
        ),
        "max_token_len": args.max_token_len,
        "coefficient_target": args.coefficient_target,
        "dataset_rows": total_rows,
        "encoded_rows": encoded,
        "devices": args.devices,
        "latent_dim": config.latent_dim,
        "directions_dtype": "float16",
        "codebook_sha256": hashlib.sha256(codebook.tobytes()).hexdigest(),
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.rename(args.output)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
