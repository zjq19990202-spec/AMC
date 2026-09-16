#!/usr/bin/env python3
"""FK-segment CR1 HDF5 recording parts and annotate their fixed intervals with Qwen.

Each ``episode_*_part*.hdf5`` file is intentionally treated as an independent
recording.  The source directory is gap-split, so joining parts with the same
episode id would fabricate a continuous robot trajectory across a recording
gap.  The resulting JSON therefore has one ``episode_id`` per input part.

The recorded D435 base view and D405-right wrist view are decoded only into a
temporary 3 Hz pair of MP4 files.  They are deleted after the DashScope call;
the final annotation points back to the immutable HDF5 source and camera keys.

Run with a secret supplied through the environment, never on the command line:

    DASHSCOPE_API_KEY=... python scripts/annotate_cr1_hdf5_parts.py \
      --root '/media/admin123/T5 EVO/record_data_gap_split'
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import re
import tempfile
from typing import Iterator, Sequence

import cv2
import h5py
import numpy as np

from atomic_latent_vla.annotation.client import QwenVLPlusClient
from atomic_latent_vla.annotation.mirror import mirror_left_arm_values
from atomic_latent_vla.annotation.motion import (
    CartesianPose,
    JointMotionTrace,
    PinocchioCR1FK,
    default_mount_xyz,
)
from atomic_latent_vla.annotation.pipeline import AtomicSegmentationPipeline, PipelineConfig
from atomic_latent_vla.annotation.schema import FinalAnnotation, FinalSegment
from atomic_latent_vla.annotation.timeline import FixedAtomicSegment, build_fk_atomic_timeline
from atomic_latent_vla.annotation.video import sample_synchronized_videos
from atomic_latent_vla.timebase import timestamps_in_seconds
from atomic_latent_vla.tcp import TCP_LOCAL_Z_OFFSET_M, TCP_OFFSET_TAG


DEFAULT_ROOT = Path("/media/admin123/T5 EVO/record_data_gap_split")
DEFAULT_OUTPUT = DEFAULT_ROOT / f"atomic_qwen_right_{TCP_OFFSET_TAG}"
DEFAULT_URDF = Path("/home/admin123/cr1_recordclient/model/CR1_UPPER/urdf/CR1ARMR.urdf")
BASE_CAMERA = "cam_d435"
RIGHT_WRIST_CAMERA = "cam_d405_right"
LEFT_WRIST_CAMERA = "cam_d405_left"
HDF_TIMESTAMPS = "/observations/timestamps"
HDF_QPOS = "/observations/qpos"
# Qwen windows are constructed from up to five *full* FK intervals, including
# drop intervals.  The normal path therefore uses exactly those boundaries;
# no artificial padding or tail context is added.
QWEN_MIN_VIDEO_DURATION_S = 0.0
# A final active interval can follow one giant (e.g. 130 s) drop interval. We
# cannot include that whole drop without violating the 25 s cap, but can show
# its trailing portion as real visual context.
QWEN_MIN_CONTEXT_WINDOW_S = 5.0


class LastLinkOffsetFK:
    """Keep the shared audited TCP convention along final local +z."""

    def __init__(self, base: object, offset_m: float = TCP_LOCAL_Z_OFFSET_M) -> None:
        self._base = base
        self._offset = np.asarray([0.0, 0.0, offset_m], dtype=np.float64)

    def pose(self, q: np.ndarray) -> CartesianPose:
        pose = self._base.pose(q)
        return CartesianPose(
            pose.translation + pose.rotation @ self._offset,
            pose.rotation,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--start", type=int, default=0, help="zero-based file offset")
    parser.add_argument("--limit", type=int, default=None, help="process at most this many files")
    parser.add_argument("--only-fk", action="store_true", help="write FK timelines without Qwen")
    parser.add_argument(
        "--no-left-mirror",
        action="store_true",
        help="annotate recorded right-arm data only; by default also create the left-to-right mirror",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model", default="qwen3-vl-plus")
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument(
        "--qwen-timeout-s",
        type=float,
        default=1200.0,
        help="native DashScope request timeout; full 3 Hz recordings can exceed 300 s",
    )
    parser.add_argument("--sample-fps", type=float, default=3.0)
    parser.add_argument("--max-frames", type=int, default=2000)
    parser.add_argument(
        "--tcp-offset-m", type=float, default=TCP_LOCAL_Z_OFFSET_M
    )
    parser.add_argument(
        "--qwen-window-max-s",
        type=float,
        default=25.0,
        help="maximum duration of one detailed 3 Hz Qwen window",
    )
    parser.add_argument("--qwen-window-max-segments", type=int, default=5)
    parser.add_argument("--global-fps", type=float, default=2.0)
    parser.add_argument("--global-window-max-s", type=float, default=120.0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    return parser


def _part_sort_key(path: Path) -> tuple[int, int]:
    match = re.fullmatch(r"episode_(\d+)_part(\d+)\.hdf5", path.name)
    if match is None:
        raise ValueError(f"unexpected recording filename: {path.name}")
    return int(match.group(1)), int(match.group(2))


def list_parts(root: Path) -> list[Path]:
    files = sorted(root.glob("episode_*_part*.hdf5"), key=_part_sort_key)
    if not files:
        raise FileNotFoundError(f"no episode_*_part*.hdf5 found under {root}")
    return files


def _dataset_or_error(handle: h5py.File, path: str) -> h5py.Dataset:
    if path not in handle:
        raise KeyError(f"{handle.filename} has no {path}")
    value = handle[path]
    if not isinstance(value, h5py.Dataset):
        raise TypeError(f"{path} is not an HDF5 dataset")
    return value


def load_trace(
    path: Path,
    urdf: Path,
    tcp_offset_m: float,
    *,
    mirror_left_to_right: bool,
) -> JointMotionTrace:
    if mirror_left_to_right:
        with h5py.File(path, "r") as handle:
            all_qpos = np.asarray(_dataset_or_error(handle, HDF_QPOS), dtype=np.float64)
            timestamps_ms = np.asarray(_dataset_or_error(handle, HDF_TIMESTAMPS), dtype=np.float64)
        if all_qpos.ndim != 2 or all_qpos.shape[1] != 16:
            raise ValueError(f"expected CR1 qpos [T,16], got {all_qpos.shape}")
        left_to_right = mirror_left_arm_values(all_qpos[:, :7])
        timestamps = timestamps_in_seconds(
            timestamps_ms,
            origin=float(timestamps_ms[0]),
            require_strict=True,
        )
        right_fk = PinocchioCR1FK(
            urdf_path=urdf,
            tcp_frame="right_wrist_x_link",
            mount_xyz=default_mount_xyz("right"),
        )
        return JointMotionTrace(
            timestamps=timestamps,
            qpos=left_to_right,
            source=path,
            fk=LastLinkOffsetFK(right_fk, tcp_offset_m),
            robot_state=mirror_left_arm_values(all_qpos[:, :8]),
        )
    base = JointMotionTrace.load_hdf5(
        path,
        arm="right",
        urdf_path=urdf,
        tcp_frame="right_wrist_x_link",
    )
    return JointMotionTrace(
        timestamps=base.timestamps,
        qpos=base.qpos,
        source=base.source,
        fk=LastLinkOffsetFK(base.fk, tcp_offset_m),
        robot_state=base.robot_state,
    )


def _source_sample_times(duration_s: float, sample_fps: float) -> np.ndarray:
    if duration_s <= 0:
        raise ValueError("recording duration must be positive")
    times = np.arange(0.0, duration_s, 1.0 / sample_fps, dtype=np.float64)
    if not len(times) or duration_s - times[-1] > 0.5 / sample_fps:
        times = np.append(times, duration_s)
    return times


def fk_timebase_from_hdf5(path: Path, *, sample_fps: float) -> tuple[float, list[float], list[float]]:
    """Return the exact 3 Hz presentation clock without decoding any JPEGs.

    The later video materializer uses this same nearest-source-frame selection
    and writes it at a constant rate.  Keeping the logic here avoids paying
    for image decoding during the all-FK pass while retaining an auditable map
    from presentation timestamps to the original recorder timestamps.
    """
    with h5py.File(path, "r") as handle:
        timestamps_ms = np.asarray(_dataset_or_error(handle, HDF_TIMESTAMPS), dtype=np.float64)
    if timestamps_ms.ndim != 1 or len(timestamps_ms) < 4:
        raise ValueError(f"{path.name} has too few timestamps")
    if np.any(np.diff(timestamps_ms) <= 0):
        raise ValueError(f"{path.name} timestamps are not strictly increasing")
    source_duration_s = float((timestamps_ms[-1] - timestamps_ms[0]) / 1000.0)
    desired = _source_sample_times(source_duration_s, sample_fps)
    relative_s = (timestamps_ms - timestamps_ms[0]) / 1000.0
    indices = np.searchsorted(relative_s, desired, side="left")
    indices = np.clip(indices, 0, len(relative_s) - 1)
    indices = indices[np.r_[True, np.diff(indices) > 0]]
    if len(indices) < 4:
        raise ValueError(f"{path.name} supplies fewer than four distinct {sample_fps:g}Hz frames")
    video_times = np.arange(len(indices), dtype=np.float64) / sample_fps
    return (
        len(indices) / sample_fps,
        [round(float(value), 6) for value in video_times],
        [round(float(value), 6) for value in relative_s[indices]],
    )


def _decode_jpeg(row: np.ndarray, encoded_length: int, *, source: Path, index: int) -> np.ndarray:
    if encoded_length <= 0 or encoded_length > len(row):
        raise ValueError(f"invalid JPEG length at {source.name}[{index}]: {encoded_length}")
    frame = cv2.imdecode(np.asarray(row[:encoded_length], dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"cannot decode JPEG at {source.name}[{index}]")
    return frame


@contextmanager
def materialize_sampled_views(
    path: Path,
    *,
    sample_fps: float,
    mirror_left_to_right: bool,
    start_s: float = 0.0,
    end_s: float | None = None,
    minimum_duration_s: float = QWEN_MIN_VIDEO_DURATION_S,
) -> Iterator[tuple[list[Path], float, list[float], list[float]]]:
    """Decode two synchronized views, adding only post-window context if needed.

    ``start_s``/``end_s`` remain the immutable FK segment window.  An optional
    positive tail is supported only for external callers; the annotation
    pipeline passes zero and uses the packed FK window exactly.
    """
    with h5py.File(path, "r") as handle:
        timestamps_ms = np.asarray(_dataset_or_error(handle, HDF_TIMESTAMPS), dtype=np.float64)
        if timestamps_ms.ndim != 1 or len(timestamps_ms) < 4:
            raise ValueError(f"{path.name} has too few timestamps")
        if np.any(np.diff(timestamps_ms) <= 0):
            raise ValueError(f"{path.name} timestamps are not strictly increasing")
        source_duration_s = float((timestamps_ms[-1] - timestamps_ms[0]) / 1000.0)
        if start_s < 0 or start_s >= source_duration_s:
            raise ValueError(f"invalid clip start {start_s:.3f}s for {path.name}")
        clip_end_s = source_duration_s if end_s is None else min(float(end_s), source_duration_s)
        if clip_end_s <= start_s:
            raise ValueError(f"invalid clip interval [{start_s:.3f}, {clip_end_s:.3f}]")
        if minimum_duration_s < 0:
            raise ValueError("minimum_duration_s must be non-negative")
        # Do not move the leading edge: the local FK timeline remains aligned
        # to t=0 of the materialized video.  Extra trailing context is safe.
        clip_end_s = min(source_duration_s, max(clip_end_s, start_s + minimum_duration_s))
        desired = start_s + _source_sample_times(clip_end_s - start_s, sample_fps)
        relative_s = (timestamps_ms - timestamps_ms[0]) / 1000.0
        frame_indices = np.searchsorted(relative_s, desired, side="left")
        frame_indices = np.clip(frame_indices, 0, len(relative_s) - 1)
        # Do not decode a duplicate frame twice merely because a timestamp jittered.
        keep = np.r_[True, np.diff(frame_indices) > 0]
        frame_indices = frame_indices[keep]
        source_frame_times = relative_s[frame_indices]
        if len(frame_indices) < 4:
            raise ValueError(f"{path.name} supplies fewer than four distinct {sample_fps:g}Hz frames")

        streams: list[tuple[h5py.Dataset, h5py.Dataset]] = []
        wrist_camera = LEFT_WRIST_CAMERA if mirror_left_to_right else RIGHT_WRIST_CAMERA
        for camera in (BASE_CAMERA, wrist_camera):
            data = _dataset_or_error(handle, f"/observations/images/{camera}")
            lengths = _dataset_or_error(handle, f"/observations/images/{camera}_len")
            if data.shape[0] != len(relative_s) or lengths.shape[0] != len(relative_s):
                raise ValueError(f"{path.name} camera {camera} is not aligned with timestamps")
            streams.append((data, lengths))

        with tempfile.TemporaryDirectory(prefix="atomic_hdf5_views_") as temporary:
            temp_root = Path(temporary)
            paths = [temp_root / "base.mp4", temp_root / "right_wrist.mp4"]
            writers: list[cv2.VideoWriter] = []
            try:
                for frame_index in frame_indices:
                    for stream_index, (data, lengths) in enumerate(streams):
                        frame = _decode_jpeg(
                            data[int(frame_index)],
                            int(lengths[int(frame_index)]),
                            source=path,
                            index=int(frame_index),
                        )
                        if mirror_left_to_right:
                            frame = cv2.flip(frame, 1)
                        if len(writers) <= stream_index:
                            height, width = frame.shape[:2]
                            writer = cv2.VideoWriter(
                                str(paths[stream_index]),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                sample_fps,
                                (width, height),
                            )
                            if not writer.isOpened():
                                raise RuntimeError(f"cannot create temporary MP4 {paths[stream_index]}")
                            writers.append(writer)
                        writers[stream_index].write(frame)
                # MP4 metadata (including the moov atom) is emitted only on
                # release.  Qwen's native client opens these paths immediately,
                # so finalize both files before yielding them to the pipeline.
                for writer in writers:
                    writer.release()
                writers.clear()
                # OpenCV writes a constant-rate MP4.  This is the exact timebase
                # that ``sample_synchronized_videos`` will later use in the
                # annotation pipeline.  Keep the original source-frame times
                # separately for audit rather than mixing the two clocks.
                video_duration_s = len(frame_indices) / sample_fps
                video_times = np.arange(len(frame_indices), dtype=np.float64) / sample_fps
                yield (
                    paths,
                    video_duration_s,
                    [round(float(value), 6) for value in video_times],
                    [round(float(value), 6) for value in source_frame_times],
                )
            finally:
                for writer in writers:
                    writer.release()


def pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        sample_fps=args.sample_fps,
        max_frames=args.max_frames,
        min_segment_duration_s=2.0,
        single_p1=0.65,
        single_margin=0.0,
        dual_sum=0.70,
        dual_p2=0.20,
        dual_p3_max=0.15,
        fk_translation_scale_m=0.02,
        fk_rotation_scale_rad=0.075,
        fk_activity_threshold=0.15,
        fk_top2_temperature=0.10,
        write_video_file_input=True,
    )


def write_fk_timeline(
    *,
    destination: Path,
    trace: JointMotionTrace,
    timestamps_s: list[float],
    duration_s: float,
    config: PipelineConfig,
    source: Path,
    tcp_offset_m: float,
    source_frame_timestamps_s: list[float],
    mirror_left_to_right: bool,
) -> None:
    timeline = build_fk_atomic_timeline(
        trace,
        timestamps_s,
        duration_s,
        translation_scale_m_s=config.fk_translation_scale_m,
        rotation_scale_rad_s=config.fk_rotation_scale_rad,
        activity_threshold=config.fk_activity_threshold,
        minimum_regime_s=config.min_segment_duration_s,
        gate_config=config.gate_config(),
        top2_temperature=config.fk_top2_temperature,
    )
    payload = {
        "source_hdf5": str(source),
        "source_cameras": {
            "base": BASE_CAMERA,
            "right_wrist": LEFT_WRIST_CAMERA if mirror_left_to_right else RIGHT_WRIST_CAMERA,
        },
        "arm": "right",
        "augmentation": "mirror_left_to_right" if mirror_left_to_right else "none",
        "qpos_columns": (
            "[0:7] from /observations/qpos, mirrored into right-arm convention"
            if mirror_left_to_right
            else "[8:15] from /observations/qpos"
        ),
        "tcp": {"frame": "right_wrist_x_link", "local_z_offset_m": tcp_offset_m},
        "duration_s": duration_s,
        "sampled_timestamps_s": timestamps_s,
        "source_frame_timestamps_s": source_frame_timestamps_s,
        "config": {
            "sample_fps": config.sample_fps,
            "min_segment_duration_s": config.min_segment_duration_s,
            "single_p1": config.single_p1,
            "single_margin": config.single_margin,
            "dual_sum": config.dual_sum,
            "dual_p2": config.dual_p2,
            "dual_p3_max": config.dual_p3_max,
            "fk_translation_scale_m_s": config.fk_translation_scale_m,
            "fk_rotation_scale_rad_s": config.fk_rotation_scale_rad,
            "fk_top2_temperature": config.fk_top2_temperature,
        },
        "segments": [
            {
                **item.prompt_payload(),
                "atomic_ratio_blocks": [
                    {
                        "start_offset_s": block.start_offset_s,
                        "end_offset_s": block.end_offset_s,
                        "weights": list(block.weights),
                        "valid": block.valid,
                    }
                    for block in item.atomic_ratio_blocks
                ],
            }
            for item in timeline
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_fixed_timeline(
    trace: JointMotionTrace,
    timestamps_s: list[float],
    duration_s: float,
    config: PipelineConfig,
) -> list[FixedAtomicSegment]:
    """Build the immutable timeline once, before any Qwen window is made."""
    return build_fk_atomic_timeline(
        trace,
        timestamps_s,
        duration_s,
        translation_scale_m_s=config.fk_translation_scale_m,
        rotation_scale_rad_s=config.fk_rotation_scale_rad,
        activity_threshold=config.fk_activity_threshold,
        minimum_regime_s=config.min_segment_duration_s,
        gate_config=config.gate_config(),
        top2_temperature=config.fk_top2_temperature,
    )


def split_timeline_for_qwen(
    timeline: Sequence[FixedAtomicSegment], *, max_duration_s: float, max_segments: int = 3
) -> list[tuple[float, float, list[FixedAtomicSegment]]]:
    """Pack all consecutive FK intervals, but request text only for active ones.

    A window's ``max_segments`` budget deliberately counts ``single``, ``dual``
    *and* ``drop`` intervals.  Drop intervals give Qwen temporal/visual context
    around an atomic motion, while their labels and prompts remain deterministic
    FK audit records and are never requested from the model.
    """
    if max_duration_s <= 0 or max_segments <= 0:
        raise ValueError("window limits must be positive")
    windows: list[tuple[float, float, list[FixedAtomicSegment]]] = []
    current: list[FixedAtomicSegment] = []
    start_s = 0.0

    def append_window(items: list[FixedAtomicSegment], first_index: int) -> None:
        active = [item for item in items if item.decision.mode in {"single", "dual"}]
        # A drop-only span is already emitted by ``finalized_drop_segment``;
        # do not spend a Qwen request on it.
        if active:
            video_start_s = items[0].start_s
            video_end_s = items[-1].end_s
            # A short isolated tail after a long drop gets the final <=25 s
            # slice of that same drop as context. The final drop label is not
            # split; this affects only the video delivered to Qwen.
            if (
                video_end_s - video_start_s < QWEN_MIN_CONTEXT_WINDOW_S
                and first_index > 0
                and timeline[first_index - 1].decision.mode == "drop"
            ):
                video_start_s = max(
                    timeline[first_index - 1].start_s,
                    video_end_s - max_duration_s,
                )
            windows.append((video_start_s, video_end_s, active))

    first_index = 0
    for index, fixed in enumerate(timeline):
        if not current:
            start_s = fixed.start_s
            first_index = index
        # A retained FK interval is much shorter than 50 s in this corpus.  A
        # giant audit-only drop may exceed it; keep that one isolated rather
        # than altering any FK boundary.
        if current and (
            fixed.end_s - start_s > max_duration_s + 1e-6
            or len(current) >= max_segments
        ):
            append_window(current, first_index)
            current = []
            start_s = fixed.start_s
            first_index = index
        current.append(fixed)
    if current:
        append_window(current, first_index)
    return windows


def localize_timeline(
    timeline: Sequence[FixedAtomicSegment], start_s: float
) -> list[FixedAtomicSegment]:
    """Keep IDs/labels fixed while expressing one video window on its local clock."""
    return [
        FixedAtomicSegment(
            segment_id=item.segment_id,
            start_s=round(item.start_s - start_s, 6),
            end_s=round(item.end_s - start_s, 6),
            atomic_probabilities=item.atomic_probabilities,
            activity_score=item.activity_score,
            decision=item.decision,
            active_fraction=item.active_fraction,
            gate_probabilities=item.gate_probabilities,
            atomic_ratio_blocks=item.atomic_ratio_blocks,
        )
        for item in timeline
    ]


def finalized_drop_segment(fixed: FixedAtomicSegment) -> FinalSegment:
    """Keep rejected FK coverage without spending a VLM request on static audit spans."""
    return FinalSegment(
        segment_id=fixed.segment_id,
        start_s=fixed.start_s,
        end_s=fixed.end_s,
        training_eligible=False,
        atomic_supervision_mask=False,
        atomic_probabilities=list(fixed.atomic_probabilities),
        gate_probabilities=(
            list(fixed.gate_probabilities) if fixed.gate_probabilities else None
        ),
        gate_mode=fixed.decision.mode,
        gate_reason=fixed.decision.reason,
        atomic_targets=[],
        atomic_ratio_blocks=[],
        strong_interaction=False,
        base_instructions=[],
        low_level_instruction="No stable right-arm atomic motion dominates this interval.",
        visual_evidence="This fixed FK interval is audit-only and excluded from atomic training.",
        label_source="none",
    )


def infer_global_task_context(
    *,
    source: Path,
    pipeline: AtomicSegmentationPipeline,
    mirror_left_to_right: bool,
    duration_s: float,
    global_fps: float,
    global_window_max_s: float,
    cache_path: Path,
) -> tuple[str, list[dict]]:
    """Use sparse video only to establish task context before detailed prompts."""
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return str(cached["task_context"]), [{"cache": str(cache_path)}]
    summaries: list[str] = []
    usages: list[dict] = []
    starts = np.arange(0.0, duration_s, global_window_max_s)
    for index, start_s in enumerate(starts):
        end_s = min(duration_s, float(start_s + global_window_max_s))
        with materialize_sampled_views(
            source, sample_fps=global_fps, mirror_left_to_right=mirror_left_to_right,
            start_s=float(start_s), end_s=end_s,
        ) as (videos, _, _, _):
            sampled = sample_synchronized_videos(
                videos, view_names=["base", "right_wrist"], sample_fps=global_fps,
                max_frames=250, write_video_file=True,
            )
            try:
                payload, usage = pipeline.client.complete_json([
                    {"role": "system", "content": "Identify only visible robot-task context. Return JSON."},
                    {"role": "user", "content": [
                        sampled.api_item(pipeline.config.max_pixels),
                        {"type": "text", "text": (
                            f"This is global context chunk {index}, time [{start_s:.1f},{end_s:.1f}] s. "
                            "Return exactly {\"summary\": \"...\"}. State objects, likely goal, and arm roles; "
                            "use tool/object/target if uncertain and do not invent task details."
                        )},
                    ]},
                ], max_tokens=800)
            finally:
                sampled.cleanup()
        summaries.append(str(payload.get("summary", "")))
        usages.append({"chunk": index, "usage": usage})
    payload, usage = pipeline.client.complete_json([
        {"role": "system", "content": "Consolidate sparse robot-video observations conservatively. Return JSON."},
        {"role": "user", "content": [{"type": "text", "text": (
            "Return exactly {\"task_context\": \"...\"}. Do not invent unseen facts. Observations: "
            + json.dumps(summaries, ensure_ascii=False)
        )}]},
    ], max_tokens=800)
    usages.append({"merge": usage})
    context = str(payload.get("task_context", "robot manipulation with uncertain objects"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"task_context": context, "usage": usages}, ensure_ascii=False), encoding="utf-8")
    return context, usages


def run_one(
    *,
    source: Path,
    output: Path,
    urdf: Path,
    config: PipelineConfig,
    tcp_offset_m: float,
    pipeline: AtomicSegmentationPipeline | None,
    overwrite: bool,
    mirror_left_to_right: bool,
    qwen_window_max_s: float,
    qwen_window_max_segments: int,
    global_fps: float,
    global_window_max_s: float,
) -> tuple[str, int, int]:
    stem = source.stem
    view = "left_mirror" if mirror_left_to_right else "right"
    episode_id = f"{stem}_mirror_lr" if mirror_left_to_right else stem
    fk_destination = output / "fk_timelines" / view / f"{episode_id}.json"
    annotation_destination = output / "annotations" / view / f"{episode_id}.json"
    if annotation_destination.exists() and not overwrite and pipeline is not None:
        return "SKIP", 0, 0
    trace = load_trace(
        source,
        urdf,
        tcp_offset_m,
        mirror_left_to_right=mirror_left_to_right,
    )
    if pipeline is None:
        duration_s, timestamps_s, source_frame_timestamps_s = fk_timebase_from_hdf5(
            source,
            sample_fps=config.sample_fps,
        )
        if overwrite or not fk_destination.exists():
            write_fk_timeline(
                destination=fk_destination,
                trace=trace,
                timestamps_s=timestamps_s,
                duration_s=duration_s,
                config=config,
                source=source,
                tcp_offset_m=tcp_offset_m,
                source_frame_timestamps_s=source_frame_timestamps_s,
                mirror_left_to_right=mirror_left_to_right,
            )
        return "FK", 0, 0
    full_duration_s, full_timestamps_s, source_frame_timestamps_s = fk_timebase_from_hdf5(
        source, sample_fps=config.sample_fps
    )
    timeline = build_fixed_timeline(trace, full_timestamps_s, full_duration_s, config)
    if overwrite or not fk_destination.exists():
        write_fk_timeline(
            destination=fk_destination,
            trace=trace,
            timestamps_s=full_timestamps_s,
            duration_s=full_duration_s,
            config=config,
            source=source,
            tcp_offset_m=tcp_offset_m,
            source_frame_timestamps_s=source_frame_timestamps_s,
            mirror_left_to_right=mirror_left_to_right,
        )

    task, global_usage = infer_global_task_context(
        source=source, pipeline=pipeline, mirror_left_to_right=mirror_left_to_right,
        duration_s=full_duration_s, global_fps=global_fps,
        global_window_max_s=global_window_max_s,
        cache_path=output / "global_context" / f"{stem}.json",
    )
    context = (
        "This is one bounded Qwen window from an independent gap-split recording part. "
        "The supplied FK intervals are immutable; do not infer continuity outside this window."
        + (
            " Both supplied views and the left-arm state were horizontally mirrored into the "
            "right-arm convention; describe this synthetic right arm only."
            if mirror_left_to_right
            else ""
        )
    )
    final_segments = [
        finalized_drop_segment(fixed)
        for fixed in timeline
        if fixed.decision.mode == "drop"
    ]
    window_descriptions: list[str] = []
    window_usage: list[dict] = []
    for window_index, (window_start_s, window_end_s, window_timeline) in enumerate(
        split_timeline_for_qwen(
            timeline,
            max_duration_s=qwen_window_max_s,
            max_segments=qwen_window_max_segments,
        )
    ):
        local_timeline = localize_timeline(window_timeline, window_start_s)
        with materialize_sampled_views(
            source,
            sample_fps=config.sample_fps,
            mirror_left_to_right=mirror_left_to_right,
            start_s=window_start_s,
            end_s=window_end_s,
        ) as (videos, _window_duration_s, _window_times_s, _window_source_times_s):
            window_result = pipeline.run(
                videos=videos,
                view_names=["base", "right_wrist"],
                task=task,
                episode_id=f"{episode_id}__window_{window_index:03d}",
                extra_context=(
                    f"Original part time range: [{window_start_s:.3f}, {window_end_s:.3f}] s. "
                    + context
                ),
                motion_trace=trace,
                augmentation="mirror_left_to_right" if mirror_left_to_right else "none",
                fixed_timeline_override=local_timeline,
            )
        final_segments.extend(
            segment.model_copy(
                update={
                    "start_s": round(segment.start_s + window_start_s, 6),
                    "end_s": round(segment.end_s + window_start_s, 6),
                }
            )
            for segment in window_result.segments
        )
        window_descriptions.append(window_result.global_description)
        window_usage.append(
            {
                "window_index": window_index,
                "start_s": window_start_s,
                "end_s": window_end_s,
                "usage": window_result.usage,
            }
        )
    # The semantic model operates on short windows; preserve its local summaries
    # rather than pretending one call watched the entire long recording.
    result = FinalAnnotation(
        episode_id=episode_id,
        task=task,
        global_description=" Window summaries: " + " | ".join(window_descriptions),
        duration_s=full_duration_s,
        sampled_fps=config.sample_fps,
        sampled_timestamps_s=full_timestamps_s,
        provider=pipeline.client.provider,
        model=pipeline.client.model,
        reviewed=False,
        axis_convention=config.axis_convention,
        source_videos=[
            f"{source}#/observations/images/{BASE_CAMERA}",
            f"{source}#/observations/images/"
            f"{LEFT_WRIST_CAMERA if mirror_left_to_right else RIGHT_WRIST_CAMERA}",
        ],
        augmentation="mirror_left_to_right" if mirror_left_to_right else "none",
        joint_trace=str(source),
        motion_source_type=trace.source_type,
        segments=sorted(final_segments, key=lambda segment: segment.segment_id),
        usage={"global_context": global_usage, "windows": window_usage},
    )
    annotation_destination.parent.mkdir(parents=True, exist_ok=True)
    annotation_destination.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    supervised = sum(bool(segment.atomic_targets) for segment in result.segments)
    return "SUCCESS", len(result.segments), supervised


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    urdf = args.urdf.expanduser().resolve()
    if not urdf.is_file():
        raise FileNotFoundError(urdf)
    parts = list_parts(root)
    if args.start < 0 or args.start >= len(parts):
        raise ValueError(f"--start must be in [0, {len(parts) - 1}]")
    indexed = list(enumerate(parts))[args.start : None if args.limit is None else args.start + args.limit]
    if args.worker_count <= 0 or not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker-index must be in [0, worker-count)")
    selected = [item for item in indexed if item[0] % args.worker_count == args.worker_index]
    config = pipeline_config(args)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(
        json.dumps(
            {
                "source_root": str(root),
                "parts_total": len(parts),
                "parts_selected": len(selected),
                "worker_count": args.worker_count,
                "worker_index": args.worker_index,
                "right_arm_only": True,
                "include_left_mirror": not args.no_left_mirror,
                "base_camera": BASE_CAMERA,
                "right_wrist_camera": RIGHT_WRIST_CAMERA,
                "urdf": str(urdf),
                "tcp_frame": "right_wrist_x_link",
                "tcp_local_z_offset_m": args.tcp_offset_m,
                "pipeline_config": config.__dict__,
                "task_source": "none; Qwen infers conservative semantics per independent part",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    pipeline = None
    if not args.only_fk:
        pipeline = AtomicSegmentationPipeline(
            QwenVLPlusClient(
                model=args.model,
                temperature=args.temperature,
                timeout_s=args.qwen_timeout_s,
                max_retries=3,
            ),
            config,
        )
    failures = output / "failures.jsonl"
    if args.overwrite:
        failures.unlink(missing_ok=True)
    mirrors = (False,) if args.no_left_mirror else (False, True)
    for index, source in selected:
        for mirror_left_to_right in mirrors:
            view = "left_mirror" if mirror_left_to_right else "right"
            print(f"START index={index} view={view} file={source.name}", flush=True)
            try:
                status, segments, supervised = run_one(
                    source=source,
                    output=output,
                    urdf=urdf,
                    config=config,
                    tcp_offset_m=args.tcp_offset_m,
                    pipeline=pipeline,
                    overwrite=args.overwrite,
                    mirror_left_to_right=mirror_left_to_right,
                    qwen_window_max_s=args.qwen_window_max_s,
                    qwen_window_max_segments=args.qwen_window_max_segments,
                    global_fps=args.global_fps,
                    global_window_max_s=args.global_window_max_s,
                )
                print(
                    f"{status} view={view} file={source.name} "
                    f"segments={segments} supervised={supervised}",
                    flush=True,
                )
            except Exception as error:
                failure = {
                    "index": index,
                    "source": str(source),
                    "view": view,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                with failures.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                print(f"FAIL {json.dumps(failure, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
