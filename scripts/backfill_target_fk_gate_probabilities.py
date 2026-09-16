#!/usr/bin/env python3
"""Recompute and materialize per-block bimanual gate/KL supervision.

The source FK sidecars remain untouched.  Each output JSON preserves the
existing single/dual/drop decision and reporting distribution, adds the full
final gate distribution, and materializes the normalized Top-5 target for
every complete drop horizon.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from atomic_latent_vla.annotation.motion import MotionTrace
from atomic_latent_vla.annotation.timeline import build_fk_atomic_timeline
from atomic_latent_vla.atomic import STAY_ATOMIC_ID
from atomic_latent_vla.data.gating import (
    AtomicGateConfig,
    classify_atomic_targets,
    topk_gate_composition,
)
from atomic_latent_vla.tcp import BIMANUAL_TCP_POSE_SIDECAR


SOURCE_FPS = 30.0
SAMPLE_FPS = 3.0
HORIZON_S = 5.0 / 3.0
STATE_POSE_OFFSETS = {"left": 0, "right": 24}
SOURCE_DIRECTORIES = {"left": "left", "right": "right_recomputed_0p20m"}
GATE = AtomicGateConfig(
    single_p1=0.65,
    single_margin=0.0,
    dual_sum=0.70,
    dual_p2=0.20,
    dual_p3_max=0.15,
)


def _numeric(column, dtype) -> np.ndarray:
    return np.asarray(
        column.combine_chunks().to_numpy(zero_copy_only=False), dtype=dtype
    )


def _episode_rows(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {root / 'data'}")
    table = pq.read_table(
        files, columns=["episode_index", "timestamp", "frame_index"]
    )
    return (
        _numeric(table["episode_index"], np.int64),
        _numeric(table["timestamp"], np.float64),
        _numeric(table["frame_index"], np.int64),
    )


def _mode_and_labels(segment) -> tuple[str, tuple[int, ...]]:
    labels = tuple(int(value) for value in segment.decision.labels)
    mode = "idle" if labels == (STAY_ATOMIC_ID,) else str(segment.decision.mode)
    return mode, labels


def _stored_labels(segment: dict[str, object]) -> tuple[int, ...]:
    return tuple(
        int(value["label"] if isinstance(value, dict) else value)
        for value in segment.get("gate_labels", [])
    )


def _validate_existing_segment(
    *,
    path: Path,
    stored: dict[str, object],
    recomputed,
) -> tuple[bool, float]:
    final_mode, final_labels = _mode_and_labels(recomputed)
    gate_decision = classify_atomic_targets(
        np.asarray(recomputed.gate_probabilities, dtype=np.float64), GATE
    )
    gate_labels = tuple(int(value) for value in gate_decision.labels)
    gate_mode = (
        "idle"
        if gate_labels == (STAY_ATOMIC_ID,)
        else str(gate_decision.mode)
    )
    segment_id = int(stored["segment_id"])
    if segment_id != recomputed.segment_id:
        raise ValueError(
            f"{path}: segment id mismatch {segment_id} != {recomputed.segment_id}"
        )
    stored_decision = (str(stored.get("gate_mode")), _stored_labels(stored))
    # Historical strict labels always win. Recomputed decisions are audit-only
    # because later gate rules and float32 pose caching can move boundary rows.
    valid_decisions = {
        (gate_mode, gate_labels),
        (final_mode, final_labels),
    }
    decision_override = stored_decision not in valid_decisions
    if not np.isclose(float(stored["start_s"]), recomputed.start_s, atol=1e-6) or not np.isclose(
        float(stored["end_s"]), recomputed.end_s, atol=1e-6
    ):
        raise ValueError(f"{path}: segment {segment_id} interval changed")
    reporting = np.asarray(stored.get("atomic_probabilities", []), dtype=np.float64)
    # Historical incomplete tail blocks intentionally stored no probability.
    max_reporting_error = 0.0
    if reporting.size:
        recomputed_reporting = np.asarray(
            recomputed.atomic_probabilities, dtype=np.float64
        )
        max_reporting_error = (
            float("inf")
            if reporting.shape != (13,)
            else float(np.max(np.abs(reporting - recomputed_reporting)))
        )
    return decision_override, max_reporting_error


def _composition_gate_with_stay(
    gate: np.ndarray, active_fraction: float
) -> np.ndarray:
    """Preserve inactive 3 Hz block mass as the thirteenth stay atom."""

    active_fraction = float(np.clip(active_fraction, 0.0, 1.0))
    result = np.asarray(gate, dtype=np.float32).copy()
    result[:STAY_ATOMIC_ID] *= active_fraction
    result[STAY_ATOMIC_ID] = 1.0 - active_fraction
    total = float(result.sum())
    if total <= 0.0:
        raise ValueError("stay-aware composition gate has no probability mass")
    return result / total


def _augment_segment(
    stored: dict[str, object], recomputed, *, include_stay_mass: bool
) -> dict[str, object]:
    result = dict(stored)
    mode = str(stored["gate_mode"])
    labels = _stored_labels(stored)
    gate = np.asarray(recomputed.gate_probabilities, dtype=np.float32)
    result["gate_probabilities"] = [float(value) for value in gate]
    result["supervision_type"] = "single" if mode == "idle" else mode
    complete = "insufficient future coverage" not in str(stored.get("gate_reason", ""))
    supervise_drop = mode == "drop" and complete
    result["atomic_composition_supervision_mask"] = supervise_drop
    if supervise_drop:
        composition_gate = (
            _composition_gate_with_stay(gate, recomputed.active_fraction)
            if include_stay_mass
            else gate
        )
        if include_stay_mass:
            result["gate_probabilities_with_stay"] = [
                float(value) for value in composition_gate
            ]
            result["active_fraction"] = float(recomputed.active_fraction)
        composition, confidence = topk_gate_composition(composition_gate, top_k=5)
        result["atomic_composition_target"] = [
            float(value) for value in composition
        ]
        result["atomic_composition_confidence"] = confidence
    else:
        result["atomic_composition_target"] = []
        result["atomic_composition_confidence"] = 0.0
    if mode in {"single", "dual", "idle"} and not labels:
        raise ValueError("strict supervision requires at least one label")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--include-stay-mass",
        action="store_true",
        help=(
            "For Drop Top-5 targets, reserve 1-active_fraction probability "
            "for the stay atom while preserving historical gate decisions."
        ),
    )
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    source_root = root / "meta" / "fk_horizon_3hz"
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output root is not empty: {output_root}")
    for directory in SOURCE_DIRECTORIES.values():
        (output_root / directory).mkdir(parents=True, exist_ok=True)

    episodes, timestamps, frames = _episode_rows(root)
    poses = np.load(root / "meta" / BIMANUAL_TCP_POSE_SIDECAR, mmap_mode="r")
    if poses.shape != (len(episodes), 48):
        raise ValueError(f"invalid bimanual TCP pose shape: {poses.shape}")
    changes = np.flatnonzero(np.diff(episodes)) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, len(episodes)]
    stop = len(starts) if args.limit is None else min(len(starts), args.start + args.limit)
    if not 0 <= args.start < stop:
        raise ValueError(f"invalid episode range [{args.start}, {stop})")

    counts = {arm: Counter() for arm in STATE_POSE_OFFSETS}
    files_written = 0
    for ordinal in range(args.start, stop):
        start, end = int(starts[ordinal]), int(ends[ordinal])
        episode_index = int(episodes[start])
        frame_values = frames[start:end]
        if not np.array_equal(frame_values, np.arange(len(frame_values))):
            raise ValueError(f"episode {episode_index} frame indices are not contiguous")
        episode_timestamps = timestamps[start:end] - timestamps[start]
        for arm, pose_offset in STATE_POSE_OFFSETS.items():
            directory = SOURCE_DIRECTORIES[arm]
            source_path = source_root / directory / f"episode_{episode_index:06d}.json"
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            stored_segments = payload.get("segments", [])
            # Historical block boundaries are authoritative. This also covers
            # rare episodes whose timestamp-derived duration rounds to one
            # extra 3 Hz block compared with len/30.
            duration_s = float(payload["duration_s"])
            sampled = np.asarray(
                [float(segment["start_s"]) for segment in stored_segments],
                dtype=np.float64,
            )
            pose_rows = np.asarray(
                poses[start:end, pose_offset : pose_offset + 12], dtype=np.float64
            )
            trace = MotionTrace(
                timestamps=episode_timestamps,
                positions=pose_rows[:, :3],
                rotations=pose_rows[:, 3:].reshape(-1, 3, 3),
                source=root / "meta" / BIMANUAL_TCP_POSE_SIDECAR,
                source_type="precomputed_state_tcp200",
            )
            timeline = build_fk_atomic_timeline(
                trace,
                sampled.tolist(),
                duration_s,
                translation_scale_m_s=0.02,
                rotation_scale_rad_s=0.075,
                activity_threshold=0.15,
                minimum_regime_s=HORIZON_S,
                gate_config=GATE,
                top2_temperature=0.10,
            )
            if len(stored_segments) != len(timeline):
                raise ValueError(
                    f"{source_path}: block count changed "
                    f"{len(stored_segments)} != {len(timeline)}"
                )
            augmented = []
            for stored, recomputed in zip(stored_segments, timeline, strict=True):
                decision_override, reporting_error = _validate_existing_segment(
                    path=source_path, stored=stored, recomputed=recomputed
                )
                segment = _augment_segment(
                    stored,
                    recomputed,
                    include_stay_mass=args.include_stay_mass,
                )
                augmented.append(segment)
                counts[arm][segment["supervision_type"]] += 1
                counts[arm]["legacy_priority_override"] += int(decision_override)
                counts[arm]["reporting_error_gt_1e4"] += int(
                    reporting_error > 1e-4
                )
                counts[arm]["drop_kl"] += int(
                    segment["atomic_composition_supervision_mask"]
                )
            payload["segments"] = augmented
            payload["gate_distribution_version"] = "final_gate_top2_per_block_v1"
            payload["drop_kl_target"] = {
                "top_k": 5,
                "normalization": "within_top_k",
                "confidence": "one_minus_full_gate_entropy_over_log_13",
                "prediction_temperature": 1.0,
            }
            destination = output_root / directory / source_path.name
            destination.write_text(
                json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            files_written += 1

        if (ordinal - args.start + 1) % 100 == 0 or ordinal + 1 == stop:
            print(f"processed {ordinal - args.start + 1}/{stop - args.start}", flush=True)

    summary = {
        "dataset_root": str(root),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "episode_range": [args.start, stop],
        "files_written": files_written,
        "counts": {arm: dict(value) for arm, value in counts.items()},
        "contract": {
            "sample_fps": SAMPLE_FPS,
            "horizon_blocks": 5,
            "gate_top2_temperature": 0.10,
            "drop_top_k": 5,
            "drop_prediction_temperature": 1.0,
            "drop_include_stay_mass": args.include_stay_mass,
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
