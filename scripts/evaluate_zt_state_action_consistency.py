#!/usr/bin/env python3
"""Stratify zT coefficient loss by state-FK/action-FK atomic agreement.

The training contract uses state-FK atomic prompts but reconstructs future
action-FK TCP motion.  This diagnostic tests whether rows where those two
descriptions disagree are systematically harder for the coefficient DiT.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from openpi.models import model as _model

from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.model import dct_prefix, l2_normalize, recover_flow_endpoint
from atomic_latent_vla.pi05.training_data import (
    atomic_text_collate,
    build_atomic_text_dataset,
    text_batch_to_observation,
)


@nnx.jit
def _evaluate(model, observation, tcp_twist_delta, arm_mask, rng):
    active_state = model._controlled_state(observation.state)  # noqa: SLF001
    query_hidden, _, _, _ = model._prefix_forward(observation)  # noqa: SLF001
    arm_latents = model.queries.text_arm_latents(query_hidden, active_state)
    directions = l2_normalize(model.queries.direction(arm_latents))
    masked_directions = jnp.where(arm_mask[..., None], directions, 0.0)
    fused = model._fuse_text_arm_latents(masked_directions)  # noqa: SLF001

    noise_rng, time_rng = jax.random.split(rng)
    target = dct_prefix(tcp_twist_delta, model.config.coefficient_count)
    noise = jax.random.normal(noise_rng, target.shape)
    time = jax.random.uniform(time_rng, target.shape[:-2], minval=0.001, maxval=1.0)
    noisy = time[:, None, None] * noise + (1.0 - time[:, None, None]) * target
    velocity = model._coefficient_velocity(noisy, time, fused)  # noqa: SLF001
    recovered = recover_flow_endpoint(noisy, time, velocity)
    wall_error = jnp.square(recovered - target)
    velocity_error = jnp.square(velocity - (noise - target))

    def per_arm(error):
        return jnp.stack(
            [jnp.mean(error[..., :6], axis=(1, 2)), jnp.mean(error[..., 6:12], axis=(1, 2))],
            axis=1,
        )

    target_energy = jnp.stack(
        [jnp.mean(jnp.square(target[..., :6]), axis=(1, 2)),
         jnp.mean(jnp.square(target[..., 6:12]), axis=(1, 2))],
        axis=1,
    )
    trajectory_energy = jnp.stack(
        [jnp.mean(jnp.square(tcp_twist_delta[..., :6]), axis=(1, 2)),
         jnp.mean(jnp.square(tcp_twist_delta[..., 6:12]), axis=(1, 2))],
        axis=1,
    )
    return {
        "wall": per_arm(wall_error),
        "velocity": per_arm(velocity_error),
        "target_energy": target_energy,
        "trajectory_energy": trajectory_energy,
    }


def _labels_from_weights(weights: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in np.flatnonzero(weights > 0.0))


def _labels_from_segment(segment: dict) -> tuple[int, ...]:
    return tuple(sorted(int(value["label"]) for value in segment.get("gate_labels", [])))


def _kind(labels: tuple[int, ...]) -> str:
    if labels == (12,):
        return "stay"
    if len(labels) == 1:
        return "single"
    if len(labels) == 2:
        return "dual"
    return "other"


def _agreement(expected: tuple[int, ...], observed: tuple[int, ...]) -> str:
    if expected == observed:
        return "exact"
    if set(expected) & set(observed):
        return "partial"
    if not observed:
        return "action_drop"
    return "disjoint"


def _summary(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _bootstrap_ratio(a: np.ndarray, b: np.ndarray, seed: int) -> dict[str, float]:
    """Mean ratio a/b with a simple non-parametric 95% interval."""
    rng = np.random.default_rng(seed)
    ratios = np.empty(2000, dtype=np.float64)
    for index in range(len(ratios)):
        aa = rng.choice(a, size=len(a), replace=True).mean()
        bb = rng.choice(b, size=len(b), replace=True).mean()
        ratios[index] = aa / max(bb, 1e-12)
    return {
        "mean_ratio": float(a.mean() / max(b.mean(), 1e-12)),
        "ci95_low": float(np.quantile(ratios, 0.025)),
        "ci95_high": float(np.quantile(ratios, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--action-sidecar-root", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", default="/mnt/cunchu/zjq/target")
    parser.add_argument("--norm-asset-id", default="openpi_norm_compact_accepted_v3")
    parser.add_argument("--per-stratum", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--noise-repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = AtomicPi05Config(max_token_len=250, fast_action_ce_loss_weight=0.0)
    norm_path = args.dataset_root / "meta" / "tcp_twist_norm_bimanual_tcp200.json"
    norm_payload = json.loads(norm_path.read_text(encoding="utf-8"))
    norm_stats = norm_payload["norm_stats"]["tcp_twist_delta"]
    tcp_range = np.asarray(norm_stats["q99"], dtype=np.float32) - np.asarray(
        norm_stats["q01"], dtype=np.float32
    )
    dataset = build_atomic_text_dataset(
        (args.dataset_root,),
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        include_fast=False,
    )
    raw = dataset._raw  # noqa: SLF001
    raw._ensure_annotations()  # noqa: SLF001
    target_sidecars = raw._target_sidecars  # noqa: SLF001
    if target_sidecars is None:
        raise ValueError("this diagnostic requires target annotation sidecars")

    # The scan is randomized over ~2k episodes.  A bounded cache thrashes and
    # repeatedly reparses the same JSON; all sidecars together are small.
    @lru_cache(maxsize=None)
    def action_segments(episode: int, arm: str) -> dict[int, dict]:
        path = args.action_sidecar_root / arm / f"episode_{episode:06d}.json"
        if not path.is_file():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {int(row["segment_id"]): row for row in payload.get("segments", [])}

    # Equal quotas remove the large single/dual and left/right frequency confound.
    strata = {
        (arm, kind, agreement): []
        for arm in ("right", "left")
        for kind in ("single", "dual", "stay")
        for agreement in ("exact", "mismatch")
    }
    rng = np.random.default_rng(args.seed)
    visible = np.asarray(raw.base._visible_indices, dtype=np.int64)
    block_start_samples = np.flatnonzero(raw.base._frame_index[visible] % 10 == 0)
    for sample_index in rng.permutation(block_start_samples):
        data_index = int(visible[int(sample_index)])
        episode = int(raw.base._episode_index[data_index])
        frame = int(raw.base._frame_index[data_index])
        block = frame // 10
        for arm_index, arm in enumerate(("right", "left")):
            annotation = target_sidecars.atomic_horizon(episode, frame, arm=arm)
            if annotation is None:
                continue
            expected = _labels_from_weights(np.asarray(annotation.weights))
            kind = _kind(expected)
            if kind not in {"single", "dual", "stay"}:
                continue
            segment = action_segments(episode, arm).get(block, {})
            observed = _labels_from_segment(segment)
            detail = _agreement(expected, observed)
            agreement = "exact" if detail == "exact" else "mismatch"
            key = (arm, kind, agreement)
            if len(strata[key]) < args.per_stratum:
                strata[key].append((int(sample_index), detail, expected, observed, episode, frame))
        if all(len(rows) >= args.per_stratum for rows in strata.values()):
            break

    # Keep every selected row once; labels below remain arm-specific.
    selected = sorted({row[0] for rows in strata.values() for row in rows})
    selected_lookup = {sample: pos for pos, sample in enumerate(selected)}
    records: list[dict] = []
    for key, rows in strata.items():
        arm, kind, agreement = key
        for sample, detail, expected, observed, episode, frame in rows:
            records.append(
                {
                    "position": selected_lookup[sample],
                    "sample_index": sample,
                    "arm": arm,
                    "arm_index": 0 if arm == "right" else 1,
                    "kind": kind,
                    "agreement": agreement,
                    "mismatch_detail": detail,
                    "state_labels": expected,
                    "action_labels": observed,
                    "episode": episode,
                    "frame": frame,
                }
            )

    params = _model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()
    collected: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(selected), args.batch_size):
        indices = selected[start : start + args.batch_size]
        rows = [dataset[index] for index in indices]
        batch = atomic_text_collate(rows)
        observation = jax.tree.map(jnp.asarray, text_batch_to_observation(batch))
        normalized = 2.0 * np.asarray(batch["tcp_twist_delta"], dtype=np.float32) / tcp_range[None, None, :]
        arm_mask = np.asarray(batch["atomic_supervision_mask"], dtype=bool)
        repeated = []
        for repeat in range(args.noise_repeats):
            repeated.append(
                jax.device_get(
                    _evaluate(
                        model,
                        observation,
                        jnp.asarray(normalized),
                        jnp.asarray(arm_mask),
                        jax.random.key(args.seed + start * 101 + repeat),
                    )
                )
            )
        for name in repeated[0]:
            value = np.mean(np.stack([np.asarray(item[name]) for item in repeated]), axis=0)
            collected.setdefault(name, []).append(value)
        print(f"evaluated {min(start + args.batch_size, len(selected))}/{len(selected)}", flush=True)
    values = {name: np.concatenate(parts, axis=0) for name, parts in collected.items()}

    for row in records:
        pos, arm = row["position"], row["arm_index"]
        for name in values:
            row[name] = float(values[name][pos, arm])

    report: dict = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "unique_rows": len(selected),
        "arm_records": len(records),
        "noise_repeats": args.noise_repeats,
        "requested_per_stratum": args.per_stratum,
        "available_per_stratum": {"/".join(key): len(rows) for key, rows in strata.items()},
        "contract": {
            "prompt_labels": "reviewed state-FK atomic_horizon labels",
            "dct_target": "future action-FK TCP path relative to current state TCP",
            "exact": "action-FK strict gate label set exactly equals prompt label set",
            "mismatch": "partial, disjoint, or action-FK drop/idle disagreement",
        },
    }
    for metric in ("wall", "velocity", "target_energy", "trajectory_energy"):
        summaries = {}
        for arm in ("right", "left"):
            for kind in ("single", "dual", "stay"):
                for agreement in ("exact", "mismatch"):
                    data = np.asarray(
                        [row[metric] for row in records if row["arm"] == arm and row["kind"] == kind and row["agreement"] == agreement],
                        dtype=np.float64,
                    )
                    summaries[f"{arm}/{kind}/{agreement}"] = _summary(data)
        report[metric] = summaries

    comparisons = {}
    for arm in ("right", "left"):
        for kind in ("single", "dual", "stay"):
            exact = np.asarray([row["wall"] for row in records if row["arm"] == arm and row["kind"] == kind and row["agreement"] == "exact"])
            mismatch = np.asarray([row["wall"] for row in records if row["arm"] == arm and row["kind"] == kind and row["agreement"] == "mismatch"])
            if exact.size and mismatch.size:
                comparisons[f"{arm}/{kind}"] = _bootstrap_ratio(mismatch, exact, args.seed + len(comparisons))
    report["mismatch_over_exact_wall"] = comparisons
    report["mismatch_detail_counts"] = {
        detail: sum(row["mismatch_detail"] == detail for row in records)
        for detail in ("exact", "partial", "disjoint", "action_drop")
    }
    report["records"] = records
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
