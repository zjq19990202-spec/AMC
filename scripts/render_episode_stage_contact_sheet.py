#!/usr/bin/env python3
"""Render stage-aligned frames from a compact LeRobot episode."""

from __future__ import annotations

import argparse
import io
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CAMERAS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
)


def _frame(video: Path, index: int, size: tuple[int, int]) -> Image.Image:
    process = subprocess.run(
        (
            "ffmpeg",
            "-v",
            "error",
            "-hwaccel",
            "none",
            "-i",
            str(video),
            "-vf",
            f"select=eq(n\\,{index})",
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ),
        check=True,
        capture_output=True,
    )
    if not process.stdout:
        raise RuntimeError(f"failed to decode frame {index} from {video}")
    with Image.open(io.BytesIO(process.stdout)) as value:
        return value.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument(
        "--segment",
        action="append",
        required=True,
        help="START:END_EXCLUSIVE:LABEL (repeat for every stage)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    segments = []
    for specification in args.segment:
        start_text, end_text, label = specification.split(":", 2)
        start, end = int(start_text), int(end_text)
        if not 0 <= start < end:
            raise ValueError(specification)
        segments.append((start, end, label))

    videos = {
        camera: args.dataset_root
        / "videos"
        / camera
        / "chunk-000"
        / f"file-{args.episode}.mp4"
        for camera in CAMERAS
    }
    for video in videos.values():
        if not video.is_file():
            raise FileNotFoundError(video)

    cell_size = (320, 240)
    label_width = 370
    header_height = 116
    row_height = 282
    columns = (
        ("Base · start", CAMERAS[0], "start"),
        ("Base · midpoint", CAMERAS[0], "middle"),
        ("Base · end", CAMERAS[0], "end"),
        ("Left wrist · midpoint", CAMERAS[1], "middle"),
        ("Right wrist · midpoint", CAMERAS[2], "middle"),
    )
    canvas = Image.new(
        "RGB",
        (label_width + len(columns) * cell_size[0], header_height + len(segments) * row_height),
        "#f8fafc",
    )
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_path, 22)
    small = ImageFont.truetype(font_path, 19)
    bold = ImageFont.truetype(bold_path, 22)
    title = ImageFont.truetype(bold_path, 27)
    draw.text((18, 14), f"Episode {args.episode} · five-stage annotation", fill="#111827", font=title)
    for column_index, (heading, _, _) in enumerate(columns):
        x = label_width + column_index * cell_size[0] + 12
        draw.text((x, 70), heading, fill="#334155", font=bold)

    for row_index, (start, end, label) in enumerate(segments):
        y = header_height + row_index * row_height
        draw.rectangle((0, y, canvas.width, y + row_height), fill="#ffffff" if row_index % 2 == 0 else "#f1f5f9")
        draw.text((18, y + 22), f"Stage {row_index + 1}", fill="#0f172a", font=bold)
        draw.text((18, y + 58), f"frames {start}–{end - 1}", fill="#2563eb", font=bold)
        words = label.split()
        lines, current = [], []
        for word in words:
            proposed = " ".join(current + [word])
            if draw.textlength(proposed, font=small) > label_width - 38 and current:
                lines.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            lines.append(" ".join(current))
        draw.multiline_text((18, y + 102), "\n".join(lines), fill="#334155", font=small, spacing=7)

        indices = {"start": start, "middle": (start + end - 1) // 2, "end": end - 1}
        for column_index, (_, camera, position) in enumerate(columns):
            index = indices[position]
            image = _frame(videos[camera], index, cell_size)
            x = label_width + column_index * cell_size[0]
            canvas.paste(image, (x, y))
            draw.rectangle((x + 7, y + 7, x + 91, y + 38), fill="#111827")
            draw.text((x + 14, y + 9), f"f={index}", fill="#ffffff", font=font)
        draw.line((0, y + row_height - 1, canvas.width, y + row_height - 1), fill="#cbd5e1", width=2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=94)
    print(args.output)


if __name__ == "__main__":
    main()
