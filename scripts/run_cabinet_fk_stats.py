#!/usr/bin/env python3
"""Generate deterministic FK atomic-segmentation statistics for cabinet LeRobot data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from atomic_latent_vla.annotation.mirror import mirror_left_from_bimanual
from atomic_latent_vla.annotation.motion import CartesianPose, JointMotionTrace, PinocchioCR1FK
from atomic_latent_vla.annotation.pipeline import PipelineConfig
from atomic_latent_vla.annotation.timeline import build_fk_atomic_timeline
from atomic_latent_vla.atomic import ATOMIC_ID_TO_SKILL
from atomic_latent_vla.data.gating import AtomicGateConfig
from atomic_latent_vla.timebase import timestamps_in_seconds
from atomic_latent_vla.tcp import TCP_LOCAL_Z_OFFSET_M, TCP_OFFSET_TAG


class LastLinkOffsetFK:
    """Expose a short tool point measured from the last physical arm link."""

    def __init__(self, last_link_fk: PinocchioCR1FK, offset_m: float) -> None:
        self._last_link_fk = last_link_fk
        self._offset = np.asarray([0.0, 0.0, offset_m], dtype=np.float64)

    def pose(self, q: np.ndarray) -> CartesianPose:
        pose = self._last_link_fk.pose(q)
        return CartesianPose(
            pose.translation + pose.rotation @ self._offset,
            pose.rotation,
        )


def _load_state_rows(root: Path) -> dict[int, tuple[np.ndarray, np.ndarray, Path]]:
    """Read packed state parquet files once for the whole batch."""
    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise RuntimeError("cabinet statistics require pyarrow") from error
    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data parquet files under {root / 'data'}")
    buckets: dict[int, tuple[list[object], list[float], Path]] = {}
    for path in files:
        table = pq.read_table(path, columns=["observation.state", "timestamp", "episode_index"])
        states = table.column("observation.state").to_pylist()
        timestamps = table.column("timestamp").to_pylist()
        episodes = table.column("episode_index").to_pylist()
        for state, timestamp, episode_index in zip(states, timestamps, episodes, strict=True):
            bucket = buckets.setdefault(int(episode_index), ([], [], path))
            bucket[0].append(state)
            bucket[1].append(float(timestamp))
    result: dict[int, tuple[np.ndarray, np.ndarray, Path]] = {}
    for index, (states, timestamps, path) in buckets.items():
        state_array = np.asarray(states, dtype=np.float64)
        if state_array.ndim != 2 or state_array.shape[1] != 16:
            raise ValueError(f"episode {index} has invalid state shape {state_array.shape}")
        result[index] = (
            state_array,
            timestamps_in_seconds(np.asarray(timestamps), require_strict=True),
            path,
        )
    return result


def _segment_payload(segment: object) -> dict[str, object]:
    # Keep this independent of the annotation schema: these files are FK audit
    # statistics and are intentionally usable before a Qwen semantic pass.
    decision = segment.decision
    return {
        "segment_id": segment.segment_id,
        "start_s": segment.start_s,
        "end_s": segment.end_s,
        "duration_s": round(segment.end_s - segment.start_s, 6),
        "gate_mode": decision.mode,
        "gate_labels": [
            {"label": int(label), "skill": ATOMIC_ID_TO_SKILL[label].value}
            for label in decision.labels
        ],
        "gate_weights": list(decision.weights),
        "gate_reason": decision.reason,
        "atomic_probabilities": list(segment.atomic_probabilities),
        "gate_probabilities": list(segment.gate_probabilities),
        "atomic_ratio_blocks": [
            {
                "start_offset_s": block.start_offset_s,
                "end_offset_s": block.end_offset_s,
                "weights": list(block.weights),
                "valid": block.valid,
            }
            for block in segment.atomic_ratio_blocks
        ],
        "activity_score": segment.activity_score,
    }


def _run_one(
    episode_index: int,
    *,
    state: np.ndarray,
    timestamps_s: np.ndarray,
    source: Path,
    mirror: bool,
    fk: LastLinkOffsetFK,
    urdf_path: Path,
    config: PipelineConfig,
    gate_config: AtomicGateConfig,
) -> dict[str, object]:
    right_state = (
        mirror_left_from_bimanual(state)
        if mirror
        else np.ascontiguousarray(state[:, 8:16])
    )
    trace = JointMotionTrace(
        timestamps=timestamps_s,
        qpos=right_state[:, :7],
        source=source,
        fk=fk,
        robot_state=right_state,
    )
    duration_s = float(timestamps_s[-1] + 1.0 / 30.0)
    timestamps = np.arange(
        0.0,
        duration_s - 0.1 + np.finfo(np.float64).eps,
        1.0 / config.sample_fps,
        dtype=np.float64,
    )
    if len(timestamps) < 2:
        timestamps = np.asarray([0.0, duration_s], dtype=np.float64)
    timeline = build_fk_atomic_timeline(
        trace,
        timestamps.tolist(),
        duration_s,
        translation_scale_m_s=config.fk_translation_scale_m,
        rotation_scale_rad_s=config.fk_rotation_scale_rad,
        activity_threshold=config.fk_activity_threshold,
        minimum_regime_s=config.min_segment_duration_s,
        gate_config=gate_config,
        top2_temperature=config.fk_top2_temperature,
    )
    valid = [segment for segment in timeline if segment.decision.mode != "drop"]
    valid_s = sum(segment.end_s - segment.start_s for segment in valid)
    drop_s = sum(
        segment.end_s - segment.start_s
        for segment in timeline
        if segment.decision.mode == "drop"
    )
    return {
        "episode_index": episode_index,
        "episode_id": f"episode_{episode_index:06d}",
        "augmentation": "mirror_left_to_right" if mirror else "none",
        "task": "",
        "duration_s": duration_s,
        "sample_fps": config.sample_fps,
        "tcp_definition": {
            "reference": "last_link_origin",
            "offset_m": TCP_LOCAL_Z_OFFSET_M,
            "frame": "right_wrist_x_link",
        },
        "gate_config": {
            "single_p1": gate_config.single_p1,
            "single_margin": gate_config.single_margin,
            "dual_sum": gate_config.dual_sum,
            "dual_p2": gate_config.dual_p2,
            "dual_p3_max": gate_config.dual_p3_max,
            "top2_temperature": config.fk_top2_temperature,
        },
        "statistics": {
            "segment_count": len(timeline),
            "retained_segment_count": len(valid),
            "retained_duration_s": round(valid_s, 6),
            "retained_ratio": round(valid_s / duration_s, 6),
            "drop_duration_s": round(drop_s, 6),
            "drop_ratio": round(drop_s / duration_s, 6),
        },
        "segments": [_segment_payload(segment) for segment in timeline],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/admin123/ckrc/dscrew"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            f"/home/admin123/ckrc/cabinet/atomic_stats_{TCP_OFFSET_TAG}_dual070_p3015"
        ),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output is not empty: {output}; pass --overwrite")
    (output / "right").mkdir(parents=True, exist_ok=True)
    (output / "left_mirror").mkdir(parents=True, exist_ok=True)

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    total_episodes = int(info["total_episodes"])
    end = total_episodes if args.limit is None else min(total_episodes, args.start + args.limit)
    if not 0 <= args.start < total_episodes or end <= args.start:
        raise ValueError(f"invalid episode range: start={args.start}, end={end}")

    config = PipelineConfig(
        sample_fps=3.0,
        min_segment_duration_s=2.0,
        fk_top2_temperature=0.10,
    )
    gate_config = AtomicGateConfig(
        single_p1=0.65,
        single_margin=0.0,
        dual_sum=0.70,
        dual_p2=0.20,
        dual_p3_max=0.15,
    )
    urdf_path = Path(
        "/home/admin123/cr1_recordclient/model/CR1_UPPER/urdf/CR1ARMR.urdf"
    )
    state_rows = _load_state_rows(root)
    last_link_fk = PinocchioCR1FK(
        urdf_path=urdf_path,
        tcp_frame="right_wrist_x_link",
        mount_xyz=(-0.02, -0.2225, 0.235),
    )
    fk = LastLinkOffsetFK(last_link_fk, TCP_LOCAL_Z_OFFSET_M)

    summaries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for episode_index in range(args.start, end):
        row: dict[str, object] = {"episode_index": episode_index}
        for name, mirror in (("right", False), ("left_mirror", True)):
            try:
                payload = _run_one(
                    episode_index,
                    state=state_rows[episode_index][0],
                    timestamps_s=state_rows[episode_index][1],
                    source=state_rows[episode_index][2],
                    mirror=mirror,
                    fk=fk,
                    urdf_path=urdf_path,
                    config=config,
                    gate_config=gate_config,
                )
                path = output / name / f"episode_{episode_index:06d}.json"
                path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                row[name] = payload["statistics"]
            except Exception as error:  # keep the batch auditable and continue
                failure = {"episode_index": episode_index, "view": name, "error": str(error)}
                failures.append(failure)
                row[name] = {"error": str(error)}
        summaries.append(row)
        if (episode_index - args.start + 1) % 10 == 0 or episode_index == end - 1:
            print(f"processed {episode_index + 1 - args.start}/{end - args.start}", flush=True)

    metadata = {
        "dataset_root": str(root),
        "episode_range": [args.start, end],
        "views": ["right", "left_mirror"],
        "rules": {
            "tcp_offset_m": TCP_LOCAL_Z_OFFSET_M,
            "sample_fps": 3.0,
            "min_segment_duration_s": 2.0,
            "top2_temperature": 0.10,
            "single_p1": 0.65,
            "single_margin": 0.0,
            "dual_sum": 0.70,
            "dual_p2": 0.20,
            "dual_p3_max": 0.15,
        },
        "failures": failures,
    }
    (output / "run_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "episode_index",
            "view",
            "segment_count",
            "retained_segment_count",
            "retained_duration_s",
            "retained_ratio",
            "drop_duration_s",
            "drop_ratio",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            for view in ("right", "left_mirror"):
                values = row.get(view, {})
                writer.writerow(
                    {
                        "episode_index": row["episode_index"],
                        "view": view,
                        **{field: values.get(field, "") for field in fields[2:]},
                    }
                )
    print(f"saved {output}")
    print(f"failures={len(failures)}")


if __name__ == "__main__":
    main()
