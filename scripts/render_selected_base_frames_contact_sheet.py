#!/usr/bin/env python3
"""Render selected original base-camera frames as a labeled contact sheet."""

from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument(
        "--frame-label",
        action="append",
        required=True,
        help="FRAME:LABEL, repeated in display order",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    items: list[tuple[int, str, Image.Image]] = []
    for specification in args.frame_label:
        frame_text, label = specification.split(":", 1)
        frame = int(frame_text)
        path = args.input_dir / f"episode{args.episode}_base_frame{frame}.png"
        with Image.open(path) as image:
            items.append((frame, label, image.convert("RGB")))

    if args.columns <= 0:
        raise ValueError("columns must be positive")
    columns = min(args.columns, len(items))
    rows = math.ceil(len(items) / columns)
    image_size = (480, 360)
    label_height = 100
    title_height = 76
    cell_size = (image_size[0], image_size[1] + label_height)
    canvas = Image.new(
        "RGB",
        (columns * cell_size[0], title_height + rows * cell_size[1]),
        "#f8fafc",
    )
    draw = ImageDraw.Draw(canvas)
    regular_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    regular = ImageFont.truetype(regular_path, 18)
    bold = ImageFont.truetype(bold_path, 20)
    title = ImageFont.truetype(bold_path, 27)
    draw.text(
        (18, 18),
        f"Episode {args.episode} · original base_0_rgb frames ({args.fps:g} FPS)",
        fill="#0f172a",
        font=title,
    )

    for index, (frame, label, image) in enumerate(items):
        row, column = divmod(index, columns)
        x = column * cell_size[0]
        y = title_height + row * cell_size[1]
        resized = image.resize(image_size, Image.Resampling.LANCZOS)
        canvas.paste(resized, (x, y))
        draw.rectangle((x, y + image_size[1], x + cell_size[0], y + cell_size[1]), fill="#ffffff")
        draw.text(
            (x + 12, y + image_size[1] + 9),
            f"frame {frame} · t={frame / args.fps:.1f}s",
            fill="#1d4ed8",
            font=bold,
        )
        wrapped = textwrap.wrap(label, width=50)
        draw.multiline_text(
            (x + 12, y + image_size[1] + 42),
            "\n".join(wrapped[:2]),
            fill="#334155",
            font=regular,
            spacing=3,
        )
        draw.rectangle((x, y, x + cell_size[0] - 1, y + cell_size[1] - 1), outline="#cbd5e1", width=2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=95)
    print(args.output)


if __name__ == "__main__":
    main()
