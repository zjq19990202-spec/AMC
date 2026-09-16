#!/usr/bin/env python3
"""Build an action-path FK atomic audit without changing state-FK labels.

The production annotations use observation-state TCP motion.  This script
replays exactly the same 3 Hz, five-block, Top-2 gate on the precomputed
TCP=0.20 m *action* poses and writes an independent audit tree.  It never
modifies files under ``dataset_root/meta/fk_horizon_3hz``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from atomic_latent_vla.annotation.motion import MotionTrace
from atomic_latent_vla.annotation.timeline import build_fk_atomic_timeline
from atomic_latent_vla.atomic import ATOMIC_NAMES, STAY_ATOMIC_ID
from atomic_latent_vla.data.gating import AtomicGateConfig
from atomic_latent_vla.tcp import BIMANUAL_TCP_POSE_SIDECAR


SAMPLE_FPS = 3.0
SOURCE_FPS = 30.0
HORIZON_S = 5.0 / 3.0
TRANSLATION_SCALE_M_S = 0.02
ROTATION_SCALE_RAD_S = 0.075
ACTIVITY_THRESHOLD = 0.15
TOP2_TEMPERATURE = 0.10
GATE = AtomicGateConfig(
    single_p1=0.65,
    single_margin=0.0,
    dual_sum=0.70,
    dual_p2=0.20,
    dual_p3_max=0.15,
)
ARM_ACTION_OFFSETS = {"left": 12, "right": 36}


def _numeric(column, dtype) -> np.ndarray:
    array = column.combine_chunks()
    return np.asarray(array.to_numpy(zero_copy_only=False), dtype=dtype)


def _load_episode_rows(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {root / 'data'}")
    table = pq.read_table(files, columns=["episode_index", "timestamp", "frame_index"])
    return (
        _numeric(table["episode_index"], np.int64),
        _numeric(table["timestamp"], np.float64),
        _numeric(table["frame_index"], np.int64),
    )


def _mode_and_labels(segment) -> tuple[str, tuple[int, ...]]:
    labels = tuple(int(value) for value in segment.decision.labels)
    if labels == (STAY_ATOMIC_ID,):
        return "idle", labels
    return str(segment.decision.mode), labels


def _segment_payload(segment) -> dict[str, object]:
    mode, labels = _mode_and_labels(segment)
    return {
        "segment_id": int(segment.segment_id),
        "start_s": float(segment.start_s),
        "end_s": float(segment.end_s),
        "gate_mode": mode,
        "gate_labels": [
            {"label": label, "skill": ATOMIC_NAMES[label]} for label in labels
        ],
        "gate_weights": [float(value) for value in segment.decision.weights],
        "gate_reason": str(segment.decision.reason),
        "atomic_probabilities": [float(value) for value in segment.atomic_probabilities],
        "gate_probabilities": [float(value) for value in segment.gate_probabilities],
        "activity_score": float(segment.activity_score),
    }


def _probabilities(payload: dict[str, object]) -> np.ndarray:
    value = np.asarray(payload.get("atomic_probabilities", []), dtype=np.float64)
    if value.shape == (12,):
        value = np.pad(value, (0, 1))
    if value.shape != (13,) or not np.isfinite(value).all() or value.sum() <= 0:
        raise ValueError("invalid 13-way atomic probability vector")
    return value / value.sum()


def _labels(payload: dict[str, object]) -> tuple[int, ...]:
    result = []
    for value in payload.get("gate_labels", []):
        result.append(int(value["label"] if isinstance(value, dict) else value))
    return tuple(result)


def _top_set(probability: np.ndarray, count: int = 2) -> tuple[int, ...]:
    positive = np.flatnonzero(probability > 1e-12)
    if len(positive) <= count:
        return tuple(sorted(int(value) for value in positive))
    chosen = positive[np.argsort(probability[positive])[-count:]]
    return tuple(sorted(int(value) for value in chosen))


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    middle = 0.5 * (left + right)

    def kl(value: np.ndarray) -> float:
        mask = value > 0
        return float(np.sum(value[mask] * np.log(value[mask] / middle[mask])))

    return 0.5 * (kl(left) + kl(right))


def _fresh_comparison() -> dict[str, object]:
    return {
        "count": 0,
        "top1_matches": 0,
        "top2_matches": 0,
        "mode_matches": 0,
        "strict_both": 0,
        "strict_label_matches": 0,
        "l1": [],
        "js": [],
    }


def _update_comparison(bucket: dict[str, object], state_row: dict, action_row: dict) -> None:
    state_probability = _probabilities(state_row)
    action_probability = _probabilities(action_row)
    bucket["count"] += 1
    bucket["top1_matches"] += int(np.argmax(state_probability) == np.argmax(action_probability))
    bucket["top2_matches"] += int(_top_set(state_probability) == _top_set(action_probability))
    state_mode = str(state_row.get("gate_mode", "drop"))
    action_mode = str(action_row.get("gate_mode", "drop"))
    bucket["mode_matches"] += int(state_mode == action_mode)
    state_strict = state_mode in {"single", "dual", "idle"}
    action_strict = action_mode in {"single", "dual", "idle"}
    if state_strict and action_strict:
        bucket["strict_both"] += 1
        bucket["strict_label_matches"] += int(
            state_mode == action_mode and set(_labels(state_row)) == set(_labels(action_row))
        )
    bucket["l1"].append(float(np.abs(state_probability - action_probability).sum()))
    bucket["js"].append(_js_divergence(state_probability, action_probability))


def _finish_comparison(bucket: dict[str, object]) -> dict[str, object]:
    count = int(bucket["count"])
    strict = int(bucket["strict_both"])
    l1 = np.asarray(bucket["l1"], dtype=np.float64)
    js = np.asarray(bucket["js"], dtype=np.float64)
    return {
        "complete_horizons": count,
        "top1_agreement": float(bucket["top1_matches"] / count) if count else None,
        "top2_set_agreement": float(bucket["top2_matches"] / count) if count else None,
        "gate_mode_agreement": float(bucket["mode_matches"] / count) if count else None,
        "both_strict_horizons": strict,
        "strict_mode_and_label_agreement": (
            float(bucket["strict_label_matches"] / strict) if strict else None
        ),
        "probability_l1_mean": float(l1.mean()) if len(l1) else None,
        "probability_l1_median": float(np.median(l1)) if len(l1) else None,
        "jensen_shannon_mean_nats": float(js.mean()) if len(js) else None,
        "jensen_shannon_median_nats": float(np.median(js)) if len(js) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output is not empty: {output}; pass --overwrite")
    for arm in ARM_ACTION_OFFSETS:
        (output / arm).mkdir(parents=True, exist_ok=True)

    episode, timestamps, frames = _load_episode_rows(root)
    pose_path = root / "meta" / BIMANUAL_TCP_POSE_SIDECAR
    poses = np.load(pose_path, mmap_mode="r")
    if poses.shape != (len(episode), 48):
        raise ValueError(f"expected pose sidecar {(len(episode), 48)}, got {poses.shape}")

    changes = np.flatnonzero(np.diff(episode)) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, len(episode)]
    if args.limit_episodes is not None:
        starts = starts[: args.limit_episodes]
        ends = ends[: args.limit_episodes]

    comparisons = {arm: _fresh_comparison() for arm in ARM_ACTION_OFFSETS}
    mode_counts = {arm: Counter() for arm in ARM_ACTION_OFFSETS}
    complete_mode_counts = {arm: Counter() for arm in ARM_ACTION_OFFSETS}
    atom_counts = {arm: Counter() for arm in ARM_ACTION_OFFSETS}
    probability_sum = {arm: np.zeros(13, dtype=np.float64) for arm in ARM_ACTION_OFFSETS}
    probability_count = Counter()

    state_roots = {
        "right": root / "meta" / "fk_horizon_3hz" / "right_recomputed_0p20m",
        "left": root / "meta" / "fk_horizon_3hz" / "left",
    }
    for episode_number, (start, end) in enumerate(zip(starts, ends, strict=True)):
        episode_index = int(episode[start])
        episode_timestamps = np.asarray(timestamps[start:end], dtype=np.float64)
        episode_timestamps -= episode_timestamps[0]
        frame_values = np.asarray(frames[start:end], dtype=np.int64)
        if not np.array_equal(frame_values, np.arange(len(frame_values))):
            raise ValueError(f"episode {episode_index} frame indices are not contiguous")
        duration_s = len(frame_values) / SOURCE_FPS
        sampled = np.arange(
            0.0,
            duration_s - 0.1 + np.finfo(np.float64).eps,
            1.0 / SAMPLE_FPS,
            dtype=np.float64,
        )
        for arm, offset in ARM_ACTION_OFFSETS.items():
            rows = np.asarray(poses[start:end, offset : offset + 12], dtype=np.float64)
            trace = MotionTrace(
                timestamps=episode_timestamps,
                positions=rows[:, :3],
                rotations=rows[:, 3:].reshape(-1, 3, 3),
                source=pose_path,
                source_type="precomputed_action_tcp200",
            )
            timeline = build_fk_atomic_timeline(
                trace,
                sampled.tolist(),
                duration_s,
                translation_scale_m_s=TRANSLATION_SCALE_M_S,
                rotation_scale_rad_s=ROTATION_SCALE_RAD_S,
                activity_threshold=ACTIVITY_THRESHOLD,
                minimum_regime_s=HORIZON_S,
                gate_config=GATE,
                top2_temperature=TOP2_TEMPERATURE,
            )
            segments = [_segment_payload(segment) for segment in timeline]
            for segment in segments:
                mode_counts[arm][str(segment["gate_mode"])] += 1
                for label in _labels(segment):
                    atom_counts[arm][ATOMIC_NAMES[label]] += 1
                if "insufficient future coverage" not in str(segment["gate_reason"]):
                    complete_mode_counts[arm][str(segment["gate_mode"])] += 1
                    probability_sum[arm] += _probabilities(segment)
                    probability_count[arm] += 1

            payload = {
                "episode_index": episode_index,
                "episode_id": f"episode_{episode_index:06d}",
                "arm": arm,
                "duration_s": duration_s,
                "sample_fps": SAMPLE_FPS,
                "horizon_blocks": 5,
                "target_source": "precomputed action TCP path",
                "tcp_definition": {"offset_m": 0.20, "frame": f"{arm}_wrist_x_link"},
                "gate_config": {
                    "single_p1": GATE.single_p1,
                    "single_margin": GATE.single_margin,
                    "dual_sum": GATE.dual_sum,
                    "dual_p2": GATE.dual_p2,
                    "dual_p3_max": GATE.dual_p3_max,
                    "top2_temperature": TOP2_TEMPERATURE,
                },
                "segments": segments,
            }
            path = output / arm / f"episode_{episode_index:06d}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            state_path = state_roots[arm] / f"episode_{episode_index:06d}.json"
            state_segments = json.loads(state_path.read_text(encoding="utf-8"))["segments"]
            if len(state_segments) != len(segments):
                raise ValueError(
                    f"episode {episode_index} {arm}: state/action block count mismatch"
                )
            for state_row, action_row in zip(state_segments, segments, strict=True):
                if "insufficient future coverage" in str(action_row["gate_reason"]):
                    continue
                _update_comparison(comparisons[arm], state_row, action_row)

        if (episode_number + 1) % 100 == 0:
            print(f"processed {episode_number + 1}/{len(starts)} episodes", flush=True)

    report = {
        "dataset_root": str(root),
        "episodes": len(starts),
        "contract": {
            "source": "action-path TCP poses, not observation-state poses",
            "tcp_offset_m": 0.20,
            "sample_fps": SAMPLE_FPS,
            "future_blocks": 5,
            "horizon_s": HORIZON_S,
            "translation_scale_m_s": TRANSLATION_SCALE_M_S,
            "rotation_scale_rad_s": ROTATION_SCALE_RAD_S,
            "activity_threshold": ACTIVITY_THRESHOLD,
            "top2_temperature": TOP2_TEMPERATURE,
        },
        "action_fk": {},
        "state_vs_action": {},
    }
    for arm in ARM_ACTION_OFFSETS:
        count = max(int(probability_count[arm]), 1)
        report["action_fk"][arm] = {
            "gate_mode_counts": dict(mode_counts[arm]),
            "complete_gate_mode_counts": dict(complete_mode_counts[arm]),
            "strict_atom_occurrences": dict(atom_counts[arm]),
            "mean_complete_horizon_probability": {
                ATOMIC_NAMES[index]: float(value / count)
                for index, value in enumerate(probability_sum[arm])
            },
        }
        report["state_vs_action"][arm] = _finish_comparison(comparisons[arm])
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
