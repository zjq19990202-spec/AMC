from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
import tempfile

import cv2
import numpy as np

from .mirror import flip_frame_horizontal


@dataclass(frozen=True)
class SampledVideo:
    data_urls: list[str]
    timestamps_s: list[float]
    sampled_fps: float
    duration_s: float
    view_names: list[str]
    video_path: str | None = None

    def api_item(self, max_pixels: int = 655360) -> dict:
        video: str | list[str]
        video = self.video_path if self.video_path is not None else self.data_urls
        return {
            "type": "video",
            "video": video,
            "fps": self.sampled_fps,
            "max_pixels": max_pixels,
        }

    def cleanup(self) -> None:
        if self.video_path is None:
            return
        try:
            Path(self.video_path).unlink(missing_ok=True)
        except OSError:
            pass


def _video_metadata(path: Path) -> tuple[float, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or frame_count <= 0:
        raise ValueError(f"invalid FPS/frame count for video: {path}")
    return fps, frame_count / fps


def _read_frame(
    capture: cv2.VideoCapture, timestamp_s: float, path: Path
) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise ValueError(f"cannot read {path} at {timestamp_s:.3f}s")
    return frame


def _resize_width(frame: np.ndarray, width: int) -> np.ndarray:
    height = max(1, round(frame.shape[0] * width / frame.shape[1]))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _make_montage(
    frames: list[np.ndarray],
    view_names: list[str],
    timestamp_s: float,
    tile_width: int,
) -> np.ndarray:
    resized = [_resize_width(frame, tile_width) for frame in frames]
    max_height = max(frame.shape[0] for frame in resized)
    padded: list[np.ndarray] = []
    for frame, name in zip(resized, view_names, strict=True):
        canvas = np.zeros((max_height, tile_width, 3), dtype=np.uint8)
        canvas[: frame.shape[0]] = frame
        cv2.rectangle(canvas, (0, 0), (tile_width, 30), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            name,
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        padded.append(canvas)
    montage = np.concatenate(padded, axis=1)
    label = f"t={timestamp_s:.3f}s"
    cv2.rectangle(montage, (0, max_height - 34), (190, max_height), (0, 0, 0), -1)
    cv2.putText(
        montage,
        label,
        (8, max_height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return montage


def sample_synchronized_videos(
    video_paths: list[str | Path],
    *,
    view_names: list[str] | None = None,
    sample_fps: float = 3.0,
    max_frames: int = 400,
    tile_width: int = 448,
    jpeg_quality: int = 82,
    horizontal_flip: list[bool] | tuple[bool, ...] | None = None,
    write_video_file: bool = False,
) -> SampledVideo:
    if not video_paths:
        raise ValueError("at least one video is required")
    if not 0.1 <= sample_fps <= 10:
        raise ValueError("sample_fps must be in [0.1, 10]")
    if max_frames < 4 or max_frames > 2000:
        raise ValueError("max_frames must be in [4, 2000]")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in [1, 100]")

    paths = [Path(path).expanduser().resolve() for path in video_paths]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    names = view_names or [path.stem for path in paths]
    if len(names) != len(paths):
        raise ValueError("view_names must have the same length as video_paths")
    flip_flags = list(horizontal_flip or [False] * len(paths))
    if len(flip_flags) != len(paths):
        raise ValueError("horizontal_flip must have the same length as video_paths")

    metadata = [_video_metadata(path) for path in paths]
    duration_s = min(duration for _, duration in metadata)
    # frame_count / fps is the exclusive end timestamp. Reading exactly there
    # fails for many codecs, so sample the last decodable frame while retaining
    # the true episode duration for the annotation boundary.
    last_readable_s = min(
        max(0.0, duration - 1.0 / source_fps) for source_fps, duration in metadata
    )
    timestamps = np.arange(
        0.0,
        last_readable_s + np.finfo(np.float64).eps,
        1.0 / sample_fps,
        dtype=np.float64,
    )
    if len(timestamps) < 4:
        timestamps = np.linspace(0.0, last_readable_s, num=4, endpoint=True)
    if len(timestamps) > max_frames:
        required = len(timestamps)
        raise ValueError(
            f"strict {sample_fps:g}Hz sampling requires {required} frames for "
            f"{duration_s:.3f}s, exceeding max_frames={max_frames}; increase max_frames "
            "instead of silently changing the sampling frequency"
        )
    effective_fps = (
        float(1.0 / np.median(np.diff(timestamps)))
        if len(timestamps) > 1
        else sample_fps
    )

    captures = [cv2.VideoCapture(str(path)) for path in paths]
    data_urls: list[str] = []
    writer: cv2.VideoWriter | None = None
    video_path: str | None = None
    try:
        if write_video_file:
            handle = tempfile.NamedTemporaryFile(
                prefix="atomic_latent_vla_montage_", suffix=".mp4", delete=False
            )
            video_path = handle.name
            handle.close()
        for timestamp in timestamps:
            frames = [
                _read_frame(capture, float(timestamp), path)
                for capture, path in zip(captures, paths, strict=True)
            ]
            frames = [
                flip_frame_horizontal(frame) if should_flip else frame
                for frame, should_flip in zip(frames, flip_flags, strict=True)
            ]
            montage = _make_montage(frames, names, float(timestamp), tile_width)
            if write_video_file:
                if writer is None:
                    height, width = montage.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(video_path, fourcc, sample_fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"failed to create montage video: {video_path}")
                writer.write(montage)
                continue
            ok, encoded = cv2.imencode(
                ".jpg", montage, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            if not ok:
                raise RuntimeError(f"failed to encode montage at {timestamp:.3f}s")
            payload = base64.b64encode(encoded.tobytes()).decode("ascii")
            data_urls.append(f"data:image/jpeg;base64,{payload}")
    finally:
        if writer is not None:
            writer.release()
        for capture in captures:
            capture.release()

    return SampledVideo(
        data_urls=data_urls,
        timestamps_s=[round(float(value), 6) for value in timestamps],
        sampled_fps=effective_fps,
        duration_s=duration_s,
        view_names=list(names),
        video_path=video_path,
    )
