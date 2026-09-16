#!/usr/bin/env python3
"""Render side-by-side review MP4s for atomic-Qwen annotations.

Shows only global task context plus the current atomic/drop instruction and
drop reason.  Source video is not re-encoded until the final side-by-side
composition.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def ass_escape(text: str) -> str:
    return " ".join(text.replace("{", "(").replace("}", ")").split()).replace("\\", "\\\\")


def ass_time(seconds: float) -> str:
    h, rem = divmod(max(0.0, seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("episodes", nargs="+", type=int)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_map = json.loads((args.dataset / "source_episode_map.json").read_text())
    lerobot_id_by_source = {Path(source).stem: int(index) for index, source in source_map.items()}

    for episode in args.episodes:
        stem = f"episode_{episode}"  # annotation uses HDF source stem, resolved below
        candidates = sorted(args.annotations.glob(f"episode_{episode}_part*.json"))
        if not candidates:
            raise FileNotFoundError(f"no annotation for episode source id {episode}")
        annotation_path = candidates[0]
        data = json.loads(annotation_path.read_text())
        source_stem = annotation_path.stem
        video_id = lerobot_id_by_source[source_stem]
        base = args.dataset / "videos/observation.images.base_0_rgb/chunk-000" / f"file-{video_id:03d}.mp4"
        wrist = args.dataset / "videos/observation.images.right_wrist_0_rgb/chunk-000" / f"file-{video_id:03d}.mp4"
        if not base.exists() or not wrist.exists():
            raise FileNotFoundError(f"missing LeRobot video for {annotation_path.name}: {base}, {wrist}")

        ass = args.output / f"{annotation_path.stem}_review.ass"
        # ``global_description`` concatenates every window summary and is far
        # too long to audit on-video. ``task`` is the episode-level global
        # prompt/Qwen task summary intended for this purpose.
        raw_global = data.get("task", data.get("global_description", ""))
        first_sentence = raw_global.split(". ", 1)[0].strip()
        global_text = ass_escape("GLOBAL: " + (first_sentence[:280] + ("…" if len(first_sentence) > 280 else "")))
        lines = [
            "[Script Info]", "ScriptType: v4.00+", "PlayResX: 1920", "PlayResY: 1080", "",
            "[V4+ Styles]", "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
            "Style: Global,DejaVu Sans,22,&H00FFFFFF,&H000000FF,&H00101010,&H80101010,0,0,0,0,100,100,0,0,1,2,1,8,30,30,20,1",
            "Style: Segment,DejaVu Sans,26,&H00FFFFFF,&H000000FF,&H00101010,&H80101010,1,0,0,0,100,100,0,0,1,2,1,2,35,35,30,1", "",
            "[Events]", "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
            f"Dialogue: 0,0:00:00.00,{ass_time(float(data['duration_s']))},Global,,0,0,0,,{global_text}",
        ]
        for seg in data.get("segments", []):
            is_drop = seg.get("gate_mode") == "drop"
            if is_drop:
                text = f"DROP: {seg.get('low_level_instruction', '')}  Reason: {seg.get('gate_reason', '')}"
            else:
                text = f"ATOMIC: {seg.get('low_level_instruction', '')}"
            lines.append(
                f"Dialogue: 1,{ass_time(float(seg['start_s']))},{ass_time(float(seg['end_s']))},Segment,,0,0,0,,{ass_escape(text)}"
            )
        ass.write_text("\n".join(lines) + "\n")

        output = args.output / f"{annotation_path.stem}_review.mp4"
        vf = f"[0:v]scale=960:-2[left];[1:v]scale=960:-2[right];[left][right]hstack=inputs=2,subtitles='{ass.as_posix()}'"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(base), "-i", str(wrist), "-filter_complex", vf,
            "-map", "0:a?", "-c:v", "libx264", "-crf", "20", "-preset", "medium", "-movflags", "+faststart", str(output),
        ], check=True)
        print(output)


if __name__ == "__main__":
    main()
