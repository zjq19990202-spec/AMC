#!/usr/bin/env python3
"""Paired zT coefficient-DiT loss: empty task text versus atomic prompt."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from openpi.models import model as _model

from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.model import dct_prefix, l2_normalize, recover_flow_endpoint
from atomic_latent_vla.pi05.training_data import (
    _paligemma_tokenizer,
    atomic_text_collate,
    build_atomic_text_dataset,
    text_batch_to_observation,
)


@nnx.jit
def _paired_losses(
    model,
    observation,
    empty_tokens,
    empty_mask,
    reverse_tokens,
    reverse_mask,
    tcp_twist_delta,
    rng,
    global_step,
):
    empty_observation = model._with_prompt(  # noqa: SLF001
        observation, empty_tokens, empty_mask
    )
    reverse_observation = model._with_prompt(  # noqa: SLF001
        observation, reverse_tokens, reverse_mask
    )
    active_state = model._controlled_state(observation.state)  # noqa: SLF001

    def directions(prompted_observation):
        query_hidden, _, _, _ = model._prefix_forward(prompted_observation)  # noqa: SLF001
        arm_latents = model.queries.text_arm_latents(query_hidden, active_state)
        return l2_normalize(model.queries.direction(arm_latents))

    atomic_direction = directions(observation)
    empty_direction = directions(empty_observation)
    reverse_direction = directions(reverse_observation)
    atomic_z = model._fuse_text_arm_latents(atomic_direction)  # noqa: SLF001
    empty_z = model._fuse_text_arm_latents(empty_direction)  # noqa: SLF001
    reverse_z = model._fuse_text_arm_latents(reverse_direction)  # noqa: SLF001

    noise_rng, time_rng = jax.random.split(rng)
    target = dct_prefix(tcp_twist_delta, model.config.coefficient_count)
    noise = jax.random.normal(noise_rng, target.shape)
    time = jax.random.uniform(time_rng, target.shape[:-2], minval=0.001, maxval=1.0)
    noisy = time[:, None, None] * noise + (1.0 - time[:, None, None]) * target

    def losses(condition):
        velocity = model._coefficient_velocity(noisy, time, condition)  # noqa: SLF001
        velocity_loss = jnp.mean(jnp.square(velocity - (noise - target)), axis=(1, 2))
        recovered = recover_flow_endpoint(noisy, time, velocity)
        wall_loss = jnp.mean(jnp.square(recovered - target), axis=(1, 2))
        step = jnp.asarray(global_step, wall_loss.dtype)
        warmup = jnp.asarray(model.config.coefficient_velocity_warmup_steps, wall_loss.dtype)
        transition = jnp.asarray(max(model.config.coefficient_wall_transition_steps, 1), wall_loss.dtype)
        wall_weight = jnp.clip((step - warmup) / transition, 0.0, 1.0)
        total = (1.0 - wall_weight) * velocity_loss + wall_weight * wall_loss
        return total, velocity_loss, wall_loss, wall_weight

    empty_total, empty_velocity, empty_wall, wall_weight = losses(empty_z)
    atomic_total, atomic_velocity, atomic_wall, _ = losses(atomic_z)
    reverse_total, reverse_velocity, reverse_wall, _ = losses(reverse_z)
    empty_atomic_cosine = jnp.sum(empty_direction * atomic_direction, axis=-1)
    atomic_reverse_cosine = jnp.sum(atomic_direction * reverse_direction, axis=-1)
    return (
        empty_total,
        atomic_total,
        reverse_total,
        empty_velocity,
        atomic_velocity,
        reverse_velocity,
        empty_wall,
        atomic_wall,
        reverse_wall,
        empty_atomic_cosine,
        atomic_reverse_cosine,
        jnp.linalg.norm(empty_z, axis=-1),
        jnp.linalg.norm(atomic_z, axis=-1),
        jnp.linalg.norm(reverse_z, axis=-1),
        wall_weight,
    )


def _select(dataset, count: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    raw = dataset._raw  # noqa: SLF001
    for index in rng.permutation(len(dataset)):
        index = int(index)
        if raw._target_sidecars is not None:  # noqa: SLF001
            data_index = int(raw.base._visible_indices[index])
            episode = int(raw.base._episode_index[data_index])
            frame = int(raw.base._frame_index[data_index])
            annotations = [
                raw._target_sidecars.atomic_horizon(episode, frame, arm=arm)  # noqa: SLF001
                for arm in ("right", "left")
            ]
            weights = np.stack(
                [
                    annotation.weights
                    if annotation is not None
                    else np.zeros(13, dtype=np.float32)
                    for annotation in annotations
                ]
            )
            prompts = [
                annotation.prompt for annotation in annotations if annotation is not None
            ]
            atomic_prompt = " ".join(prompts)
        else:
            metadata = raw.metadata(index)
            weights = np.asarray(metadata["atomic_weights"])
            atomic_prompt = str(metadata["atomic_prompt"])
        # This experiment deliberately excludes every horizon containing a
        # stay target, including one-moving/one-stationary bimanual rows.
        if np.any(weights[..., 12] > 0) or not np.any(weights[..., :12] > 0):
            continue
        if not atomic_prompt.strip():
            continue
        selected.append(index)
        if len(selected) == count:
            break
    if not selected:
        raise RuntimeError("no atomically supervised samples found")
    return selected


_OPPOSITE_WORDS = {
    "forward": "backward",
    "backward": "forward",
    "leftward": "rightward",
    "rightward": "leftward",
    "upward": "downward",
    "downward": "upward",
    "positively": "negatively",
    "negatively": "positively",
    "positive": "negative",
    "negative": "positive",
    "clockwise": "counterclockwise",
    "counterclockwise": "clockwise",
}


def reverse_atomic_prompt(text: str) -> str:
    """Reverse motion direction while preserving arm/object/task semantics."""

    pattern = re.compile(
        r"\b(counterclockwise|clockwise|positively|negatively|positive|negative|"
        r"forward|backward|leftward|rightward|upward|downward)\b",
        flags=re.IGNORECASE,
    )

    def replace(match: re.Match[str]) -> str:
        source = match.group(0)
        target = _OPPOSITE_WORDS[source.lower()]
        return target.capitalize() if source[0].isupper() else target

    reversed_text = pattern.sub(replace, text)
    if reversed_text == text:
        raise ValueError(f"atomic prompt contains no reversible motion word: {text!r}")
    return reversed_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", default="/mnt/cunchu/zjq/target")
    parser.add_argument("--norm-asset-id", default="openpi_norm_compact_accepted_v3")
    parser.add_argument(
        "--tcp-twist-norm",
        type=Path,
        help="Training-time bimanual TCP q01/q99 statistics JSON.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--noise-repeats", type=int, default=4)
    parser.add_argument("--global-step", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices-file", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    config = AtomicPi05Config(max_token_len=250, fast_action_ce_loss_weight=0.0)
    tcp_norm_path = args.tcp_twist_norm or (
        args.dataset_root / "meta" / "tcp_twist_norm_bimanual_tcp200.json"
    )
    norm_payload = json.loads(tcp_norm_path.read_text(encoding="utf-8"))
    norm_stats = norm_payload["norm_stats"]["tcp_twist_delta"]
    q01 = np.asarray(norm_stats["q01"], dtype=np.float32)
    q99 = np.asarray(norm_stats["q99"], dtype=np.float32)
    tcp_range = q99 - q01
    if tcp_range.shape != (12,) or np.any(tcp_range <= 1e-6):
        raise ValueError(f"invalid bimanual TCP quantile range: {tcp_range}")
    dataset = build_atomic_text_dataset(
        (args.dataset_root,),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        include_fast=False,
    )
    if args.indices_file is None:
        selected = _select(dataset, args.max_samples, args.seed)
    else:
        selected = [int(value) for value in json.loads(args.indices_file.read_text())]
        selected = selected[: args.max_samples]
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    selected = selected[args.shard_index :: args.num_shards]
    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    tokenizer = _paligemma_tokenizer(config.max_token_len)
    collected = [[] for _ in range(15)]

    for start in range(0, len(selected), args.batch_size):
        rows = [dataset[index] for index in selected[start : start + args.batch_size]]
        batch = atomic_text_collate(rows)
        observation = jax.tree.map(jnp.asarray, text_batch_to_observation(batch))
        blank = [tokenizer.tokenize("", state) for state in batch["state"]]
        blank_tokens = jnp.asarray(np.stack([item[0] for item in blank]))
        blank_mask = jnp.asarray(np.stack([item[1] for item in blank]))
        reverse = [
            tokenizer.tokenize(reverse_atomic_prompt(row["atomic_prompt"]), state)
            for row, state in zip(rows, batch["state"], strict=True)
        ]
        reverse_tokens = jnp.asarray(np.stack([item[0] for item in reverse]))
        reverse_mask = jnp.asarray(np.stack([item[1] for item in reverse]))
        # Exact Stage-A training contract: relative TCP zero remains zero and
        # each q99-q01 range maps to width two.  Never evaluate the DiT with
        # raw metre/radian targets.
        normalized_tcp_delta = (
            2.0
            * np.asarray(batch["tcp_twist_delta"], dtype=np.float32)
            / tcp_range[None, None, :]
        )
        repeated = []
        for repeat in range(args.noise_repeats):
            repeated.append(
                jax.device_get(
                    _paired_losses(
                        model,
                        observation,
                        blank_tokens,
                        blank_mask,
                        reverse_tokens,
                        reverse_mask,
                        jnp.asarray(normalized_tcp_delta),
                        jax.random.key(args.seed + 1000 * start + repeat),
                        jnp.asarray(args.global_step),
                    )
                )
            )
        for metric in range(14):
            collected[metric].extend(
                np.mean(np.stack([value[metric] for value in repeated]), axis=0).tolist()
            )
        collected[14].append(float(np.asarray(repeated[0][14])))
        print(f"evaluated {min(start + args.batch_size, len(selected))}/{len(selected)}", flush=True)

    values = [np.asarray(item) for item in collected]
    empty_total, atomic_total, reverse_total = values[:3]
    atomic_delta = atomic_total - empty_total
    reverse_delta = reverse_total - atomic_total
    raw = dataset._raw  # noqa: SLF001
    data_indices = [int(raw.base._visible_indices[index]) for index in selected]
    episodes = [int(raw.base._episode_index[index]) for index in data_indices]
    frames = [int(raw.base._frame_index[index]) for index in data_indices]
    report = {
        "checkpoint": str(args.checkpoint),
        "samples": len(selected),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "unique_episodes": len(set(episodes)),
        "frame_index_min_max": [min(frames), max(frames)],
        "noise_repeats": args.noise_repeats,
        "global_step": args.global_step,
        "tcp_twist_norm": str(tcp_norm_path),
        "comparison": "same state/TCP-DCT target/noise/time; empty, correct atomic and direction-reversed atomic text",
        "empty_text_zt_dit_total": float(empty_total.mean()),
        "atomic_prompt_zt_dit_total": float(atomic_total.mean()),
        "reverse_prompt_zt_dit_total": float(reverse_total.mean()),
        "atomic_minus_empty": float(atomic_delta.mean()),
        "reverse_minus_atomic": float(reverse_delta.mean()),
        "reverse_vs_atomic_relative_change": float(
            reverse_delta.mean() / max(float(atomic_total.mean()), 1e-8)
        ),
        "atomic_better_than_empty_fraction": float(np.mean(atomic_delta < 0)),
        "reverse_better_than_atomic_fraction": float(np.mean(reverse_delta < 0)),
        "empty_velocity_loss": float(values[3].mean()),
        "atomic_velocity_loss": float(values[4].mean()),
        "reverse_velocity_loss": float(values[5].mean()),
        "empty_wall_loss": float(values[6].mean()),
        "atomic_wall_loss": float(values[7].mean()),
        "reverse_wall_loss": float(values[8].mean()),
        "empty_atomic_direction_cosine_right_left": np.mean(values[9], axis=0).tolist(),
        "atomic_reverse_direction_cosine_right_left": np.mean(values[10], axis=0).tolist(),
        "empty_fused_zt_norm_mean": float(values[11].mean()),
        "atomic_fused_zt_norm_mean": float(values[12].mean()),
        "reverse_fused_zt_norm_mean": float(values[13].mean()),
        "wall_weight": float(values[14].mean()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
