#!/usr/bin/env python3
"""Reuse aligned Cabinet prompts and prepare only the remaining FK windows for Qwen.

The latest deterministic FK segmentation is the source of truth.  Older Qwen
annotations are copied only when one old *training-eligible* segment covers at
least ``--reuse-overlap`` of a latest retained segment.  The other segments get
a small, base+wrist montage plus a JSONL request carrying the current FK gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np



AXIS_CONVENTION = (
    "+x points forward from the CR1 base, +y points to the robot's left, and "
    "+z points upward. Positive rotations follow the right-hand rule."
)


def _old_name(stem: str, view: str) -> str:
    return f"{stem}_mirror_lr.json" if view == "left_mirror" else f"{stem}.json"


def _eligible_overlap(latest: dict, prior: list[dict]) -> tuple[float, dict | None]:
    start, end = float(latest["start_s"]), float(latest["end_s"])
    duration = end - start
    best_fraction, best = 0.0, None
    for old in prior:
        if not old.get("training_eligible"):
            continue
        if not (old.get("low_level_instruction") or "").strip():
            continue
        overlap = max(0.0, min(end, float(old["end_s"])) - max(start, float(old["start_s"])))
        fraction = overlap / duration
        if fraction > best_fraction:
            best_fraction, best = fraction, old
    return best_fraction, best


def _local_source(path_text: str, lerobot_root: Path) -> Path:
    marker = "/videos/"
    if marker not in path_text:
        raise ValueError(f"cannot resolve LeRobot video path: {path_text}")
    return lerobot_root / "videos" / path_text.split(marker, maxsplit=1)[1]


def _video_sources(
    *,
    view: str,
    episode_stem: str,
    old_root: Path,
    mirrored_root: Path,
    lerobot_root: Path,
) -> tuple[list[Path], list[float], list[str]]:
    if view == "left_mirror":
        mirrored = mirrored_root / f"{episode_stem}_mirror_lr"
        return (
            [mirrored / "base.mp4", mirrored / "right_wrist.mp4"],
            [0.0, 0.0],
            ["base (mirrored)", "right wrist (mirrored)"],
        )
    annotation = json.loads((old_root / view / _old_name(episode_stem, view)).read_text())
    source_texts = annotation["source_videos"]
    # The first ten pilot episodes were stored as self-contained episode videos;
    # later episodes refer to offsets inside packed LeRobot video files.
    pilot_marker = "/lerobot_ep0_9/videos/"
    if all(pilot_marker in value for value in source_texts):
        paths = [
            old_root.parent / "lerobot_ep0_9" / "videos" / value.split(pilot_marker, 1)[1]
            for value in source_texts
        ]
        offsets = [0.0, 0.0]
    else:
        paths = [_local_source(value, lerobot_root) for value in source_texts]
        offsets = [float(value) for value in annotation["video_start_offsets_s"]]
    return paths, offsets, ["base", "right wrist"]


def _render_clip(
    *,
    paths: list[Path],
    offsets_s: list[float],
    names: list[str],
    start_s: float,
    end_s: float,
    output: Path,
    sample_fps: float,
    tile_width: int,
) -> list[float]:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    timestamps = np.arange(start_s, end_s - 1e-8, 1.0 / sample_fps)
    if len(timestamps) < 4:
        timestamps = np.linspace(start_s, max(start_s, end_s - 1.0 / sample_fps), 4)
    output.parent.mkdir(parents=True, exist_ok=True)
    # OpenCV cannot reliably seek into the repository's long AV1 files.  Let
    # ffmpeg software-decode both views, sample at the requested rate, and
    # produce a compact side-by-side montage for the VLM.
    duration = max(0.05, end_s - start_s)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for path, offset in zip(paths, offsets_s, strict=True):
        command.extend(["-ss", f"{start_s + offset:.6f}", "-i", str(path)])
    filter_graph = (
        f"[0:v]fps={sample_fps:g},scale={tile_width}:-2[v0];"
        f"[1:v]fps={sample_fps:g},scale={tile_width}:-2[v1];"
        "[v0][v1]hstack=inputs=2[v]"
    )
    command.extend(
        [
            "-t", f"{duration:.6f}", "-filter_complex", filter_graph,
            "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "30", "-movflags", "+faststart", str(output),
        ]
    )
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {output}: {completed.stderr[-600:]}")
    return [round(float(value), 3) for value in timestamps]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latest", type=Path, default=Path("/home/admin123/ckrc/atom_cabi_latest"))
    parser.add_argument("--old", type=Path, default=Path("/home/admin123/remote_main/remote_main/annotations"))
    parser.add_argument("--mirrored", type=Path, default=Path("/home/admin123/remote_main/remote_main/mirrored"))
    parser.add_argument("--lerobot", type=Path, default=Path("/home/admin123/ckrc/cabinet/lerobot"))
    parser.add_argument("--output", type=Path, default=Path("/home/admin123/ckrc/cabinet_latest_qwen_split"))
    parser.add_argument("--reuse-overlap", type=float, default=0.80)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--tile-width", type=int, default=336)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--render-clips", action="store_true")
    args = parser.parse_args()

    if not 0 < args.reuse_overlap <= 1:
        raise ValueError("reuse overlap must be in (0, 1]")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output already exists and is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    latest_root, old_root = args.latest.resolve(), args.old.resolve()
    mirrored_root, lerobot_root = args.mirrored.resolve(), args.lerobot.resolve()
    all_stems = sorted(path.stem for path in (latest_root / "right").glob("episode_*.json"))
    selected = [stem for stem in all_stems if int(stem.rsplit("_", 1)[1]) >= args.start]
    if args.limit is not None:
        selected = selected[: args.limit]

    manifest_path = output / "qwen_relabel_manifest.jsonl"
    summary = {"reused_segments": 0, "relabel_segments": 0, "episodes": len(selected), "by_view": {}}
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, stem in enumerate(selected, start=1):
            for view in ("right", "left_mirror"):
                latest = json.loads((latest_root / view / f"{stem}.json").read_text())
                old_path = old_root / view / _old_name(stem, view)
                old = json.loads(old_path.read_text())
                reused: list[dict] = []
                relabel: list[dict] = []
                sources = None
                for segment in latest["segments"]:
                    if segment["gate_mode"] == "drop":
                        continue
                    fraction, old_segment = _eligible_overlap(segment, old["segments"])
                    if old_segment is not None and fraction >= args.reuse_overlap:
                        reused.append(
                            {
                                **segment,
                                "low_level_instruction": old_segment["low_level_instruction"],
                                "visual_evidence": old_segment.get("visual_evidence", ""),
                                "source": {
                                    "mode": "reused_old_qwen",
                                    "old_segment_id": old_segment["segment_id"],
                                    "temporal_overlap": round(fraction, 4),
                                },
                            }
                        )
                        summary["reused_segments"] += 1
                        continue
                    if sources is None:
                        sources = _video_sources(
                            view=view,
                            episode_stem=stem,
                            old_root=old_root,
                            mirrored_root=mirrored_root,
                            lerobot_root=lerobot_root,
                        )
                    clip_rel = Path("clips") / view / f"{stem}_seg{segment['segment_id']:03d}.mp4"
                    request = {
                        "request_id": f"{view}/{stem}/segment_{segment['segment_id']:03d}",
                        "episode_id": stem + ("_mirror_lr" if view == "left_mirror" else ""),
                        "view": view,
                        "task": old.get("task", ""),
                        "global_description": old.get("global_description", ""),
                        "axis_convention": AXIS_CONVENTION,
                        "segment": segment,
                        "video_clip": str(clip_rel),
                        "video_sample_fps": args.sample_fps,
                        "instruction": (
                            "Write one concise low_level_instruction for only this video interval. "
                            "Use the visual evidence and task context. The supplied FK atomic labels "
                            "describe motion and must not be contradicted; do not invent contact."
                        ),
                    }
                    if args.render_clips:
                        paths, offsets, names = sources
                        request["sampled_timestamps_s"] = _render_clip(
                            paths=paths,
                            offsets_s=offsets,
                            names=names,
                            start_s=float(segment["start_s"]),
                            end_s=float(segment["end_s"]),
                            output=output / clip_rel,
                            sample_fps=args.sample_fps,
                            tile_width=args.tile_width,
                        )
                    manifest.write(json.dumps(request, ensure_ascii=False) + "\n")
                    relabel.append(request)
                    summary["relabel_segments"] += 1
                reused_path = output / "reused" / view / f"{stem}.json"
                reused_path.parent.mkdir(parents=True, exist_ok=True)
                reused_path.write_text(
                    json.dumps(
                        {
                            "episode_id": latest["episode_id"],
                            "augmentation": latest["augmentation"],
                            "task": old.get("task", ""),
                            "global_description": old.get("global_description", ""),
                            "axis_convention": AXIS_CONVENTION,
                            "fk_config": latest["gate_config"],
                            "segments": reused,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                summary["by_view"].setdefault(view, {"reused": 0, "relabel": 0})
                summary["by_view"][view]["reused"] += len(reused)
                summary["by_view"][view]["relabel"] += len(relabel)
            if index % 25 == 0 or index == len(selected):
                print(f"processed {index}/{len(selected)}", flush=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
