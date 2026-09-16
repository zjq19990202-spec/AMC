#!/usr/bin/env python3
"""Run resumable native-DashScope annotations for the complete dscrew dataset.

Each LeRobot episode produces two independent annotations: the recorded right
arm and the left arm mirrored into the right-arm convention.  The first pass
does not hide failures: every failed item is appended to ``failures_pass1`` so
that it can be retried separately after the uniform pass completes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from atomic_latent_vla.annotation.client import QwenVLPlusClient
from atomic_latent_vla.annotation.lerobot import load_lerobot_episode
from atomic_latent_vla.annotation.motion import CartesianPose, JointMotionTrace
from atomic_latent_vla.annotation.pipeline import AtomicSegmentationPipeline, PipelineConfig
from atomic_latent_vla.tcp import TCP_LOCAL_Z_OFFSET_M


class LastLinkOffsetFK:
    """Use the shared tool point along the final right-wrist local z axis."""

    def __init__(self, base: object, offset_m: float = TCP_LOCAL_Z_OFFSET_M) -> None:
        self._base = base
        self._offset = np.asarray([0.0, 0.0, offset_m], dtype=np.float64)

    def pose(self, q: np.ndarray) -> CartesianPose:
        pose = self._base.pose(q)
        return CartesianPose(pose.translation + pose.rotation @ self._offset, pose.rotation)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, default=Path("/home/admin123/ckrc/dscrew"))
    result.add_argument(
        "--output",
        type=Path,
        default=Path("/home/admin123/ckrc/qwen_annotations_dscrew_native"),
    )
    result.add_argument("--start", type=int, default=0)
    result.add_argument("--end", type=int, default=None, help="exclusive episode index")
    result.add_argument("--overwrite", action="store_true")
    result.add_argument(
        "--retry-failures",
        type=Path,
        default=None,
        help="retry only items listed in a previous failures jsonl",
    )
    result.add_argument(
        "--failure-output",
        type=Path,
        default=None,
        help="jsonl path for failures from this run",
    )
    result.add_argument(
        "--pilot-output",
        type=Path,
        default=Path("/home/admin123/ckrc/qwen_annotations_dscrew_10"),
        help="reuse compatible 3 Hz pilot annotations before calling Qwen again",
    )
    result.add_argument("--no-reuse-pilot", action="store_true")
    return result


def reuse_compatible_pilot(*, pilot_output: Path, output: Path, overwrite: bool) -> int:
    """Copy only 3 Hz pilot results into the final native-run directory.

    Older annotations used a different atomic timebase. Those
    two files remain intentionally excluded so this native run replaces them
    at the requested fixed sampling rate.
    """
    if not pilot_output.is_dir() or overwrite:
        return 0

    reused: list[dict[str, object]] = []
    for view_name in ("right", "left_mirror"):
        for source in sorted((pilot_output / view_name).glob("*.json")):
            try:
                payload = json.loads(source.read_text(encoding="utf-8"))
                sampled_fps = float(payload.get("sampled_fps", 0.0))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if abs(sampled_fps - 3.0) > 1e-3:
                continue
            destination = output / view_name / source.name
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            reused.append(
                {
                    "source": str(source),
                    "destination": str(destination),
                    "sampled_fps": sampled_fps,
                }
            )
    if reused:
        (output / "reused_pilot_3hz.json").write_text(
            json.dumps(reused, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return len(reused)


def retry_items(path: Path) -> list[tuple[int, bool]]:
    result: list[tuple[int, bool]] = []
    seen: set[tuple[int, bool]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        item = (int(payload["episode_index"]), bool(payload["mirror_left_to_right"]))
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def main() -> None:
    args = parser().parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    total_episodes = int(info["total_episodes"])
    if args.retry_failures is None:
        end = total_episodes if args.end is None else min(args.end, total_episodes)
        if not 0 <= args.start < end <= total_episodes:
            raise ValueError(f"invalid range [{args.start}, {end}) for {total_episodes} episodes")
        items = [
            (episode_index, mirror)
            for episode_index in range(args.start, end)
            for mirror in (False, True)
        ]
    else:
        failure_source = args.retry_failures.expanduser().resolve()
        items = retry_items(failure_source)
        end = args.start

    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "run_config.json"
    config_path.write_text(
        json.dumps(
            {
                "backend": "dashscope_native",
                "model": "qwen3-vl-plus",
                "sample_fps": 3.0,
                "max_frames": 2000,
                "tcp_offset_m": TCP_LOCAL_Z_OFFSET_M,
                "min_segment_duration_s": 2.0,
                "single_p1": 0.65,
                "single_margin": 0.25,
                "dual_sum": 0.70,
                "dual_p2": 0.20,
                "dual_p3_max": 0.15,
                "fk_translation_scale_m": 0.02,
                "fk_rotation_scale_rad": 0.075,
                "fk_top2_temperature": 0.10,
                "write_video_file_input": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    failure_path = (
        args.failure_output.expanduser().resolve()
        if args.failure_output is not None
        else output / "failures_pass1.jsonl"
    )
    if args.overwrite and failure_path.exists():
        failure_path.unlink()

    if not args.no_reuse_pilot and args.retry_failures is None:
        reused_count = reuse_compatible_pilot(
            pilot_output=args.pilot_output.expanduser().resolve(),
            output=output,
            overwrite=args.overwrite,
        )
        if reused_count:
            print(f"REUSED_PILOT_3HZ count={reused_count}", flush=True)

    config = PipelineConfig(
        sample_fps=3.0,
        max_frames=2000,
        min_segment_duration_s=2.0,
        single_p1=0.65,
        single_margin=0.25,
        dual_sum=0.70,
        dual_p2=0.20,
        dual_p3_max=0.15,
        fk_translation_scale_m=0.02,
        fk_rotation_scale_rad=0.075,
        fk_top2_temperature=0.10,
        write_video_file_input=True,
    )
    client = QwenVLPlusClient(
        model="qwen3-vl-plus", temperature=0.10, timeout_s=240, max_retries=3
    )
    pipeline = AtomicSegmentationPipeline(client, config)
    urdf = Path("/home/admin123/cr1_recordclient/model/CR1_UPPER/urdf/CR1ARMR.urdf")

    for episode_index, mirror in items:
            episode_id = (
                f"episode_{episode_index:06d}_mirror_lr"
                if mirror
                else f"episode_{episode_index:06d}"
            )
            view_dir = output / ("left_mirror" if mirror else "right")
            destination = view_dir / f"{episode_id}.json"
            if destination.exists() and not args.overwrite:
                print(f"SKIP {episode_id}", flush=True)
                continue
            print(f"START {episode_id}", flush=True)
            try:
                episode = load_lerobot_episode(
                    root, episode_index, mirror_left_to_right=mirror
                )
                base_trace = episode.make_trace(
                    urdf_path=urdf, tcp_frame="right_wrist_x_link"
                )
                trace = JointMotionTrace(
                    timestamps=base_trace.timestamps,
                    qpos=base_trace.qpos,
                    source=base_trace.source,
                    fk=LastLinkOffsetFK(base_trace.fk),
                    robot_state=base_trace.robot_state,
                )
                annotation = pipeline.run(
                    videos=list(episode.videos),
                    view_names=["base", "right_wrist"],
                    task=episode.task,
                    episode_id=episode_id,
                    motion_trace=trace,
                    horizontal_flip=episode.horizontal_flip,
                    augmentation=episode.augmentation,
                )
                view_dir.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    json.dumps(annotation.model_dump(mode="json"), ensure_ascii=False, indent=2)
                    + "\n",
                    encoding="utf-8",
                )
                print(
                    f"SUCCESS {episode_id} segments={len(annotation.segments)} "
                    f"supervised={sum(bool(item.atomic_targets) for item in annotation.segments)}",
                    flush=True,
                )
            except Exception as error:
                failure = {
                    "episode_index": episode_index,
                    "mirror_left_to_right": mirror,
                    "episode_id": episode_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                with failure_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                print(f"FAIL {json.dumps(failure, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
