#!/usr/bin/env python3
"""Render source base-camera frames with native prompts from an eval manifest."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _decode(video: Path, frame: int, output: Path) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"select=eq(n\\,{frame})",
        "-frames:v",
        "1",
        "-y",
        str(output),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--dataset-name", default="target2058")
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--rows-per-page", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    rows = manifest["datasets"][args.dataset_name]
    if not rows:
        raise ValueError("selection manifest is empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    regular_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    regular = ImageFont.truetype(regular_path, 15)
    bold = ImageFont.truetype(bold_path, 18)
    title_font = ImageFont.truetype(bold_path, 25)
    image_size = (480, 360)
    label_height = 148
    title_height = 66
    per_page = args.columns * args.rows_per_page
    outputs = []

    with tempfile.TemporaryDirectory(prefix="atomic_eval_frames_") as temporary:
        temporary_path = Path(temporary)
        rendered: list[tuple[dict, Image.Image]] = []
        for index, row in enumerate(rows):
            episode = int(row["episode"])
            frame = int(row["frame"])
            matches = list(
                (
                    args.dataset_root
                    / "videos"
                    / "observation.images.base_0_rgb"
                ).glob(f"chunk-*/file-{episode:03d}.mp4")
            )
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"expected one base video for episode {episode}, found {matches}"
                )
            png = temporary_path / f"{index:04d}.png"
            _decode(matches[0], frame, png)
            with Image.open(png) as image:
                rendered.append((row, image.convert("RGB")))

    for page_index in range(math.ceil(len(rendered) / per_page)):
        page = rendered[page_index * per_page : (page_index + 1) * per_page]
        page_rows = math.ceil(len(page) / args.columns)
        width = args.columns * image_size[0]
        height = title_height + page_rows * (image_size[1] + label_height)
        canvas = Image.new("RGB", (width, height), "#f8fafc")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (18, 16),
            f"Reviewed source observations · page {page_index + 1}",
            fill="#0f172a",
            font=title_font,
        )
        for local_index, (row, image) in enumerate(page):
            grid_row, column = divmod(local_index, args.columns)
            x = column * image_size[0]
            y = title_height + grid_row * (image_size[1] + label_height)
            canvas.paste(image.resize(image_size, Image.Resampling.LANCZOS), (x, y))
            label_y = y + image_size[1]
            draw.rectangle(
                (x, label_y, x + image_size[0], label_y + label_height), fill="#ffffff"
            )
            header = (
                f"ep {int(row['episode'])} · frame {int(row['frame'])} · "
                f"{row.get('active_arm', '?')} arm"
            )
            draw.text((x + 10, label_y + 8), header, fill="#1d4ed8", font=bold)
            source = str(row.get("source_dataset", "unknown source"))
            draw.text((x + 10, label_y + 36), source, fill="#475569", font=regular)
            prompt = str(row.get("prompt", row.get("target", "")))
            wrapped = textwrap.wrap(prompt, width=59)
            draw.multiline_text(
                (x + 10, label_y + 62),
                "\n".join(wrapped[:3]),
                fill="#111827",
                font=regular,
                spacing=3,
            )
            draw.rectangle(
                (x, y, x + image_size[0] - 1, label_y + label_height - 1),
                outline="#cbd5e1",
                width=2,
            )
        output = args.output_dir / f"source_observations_{page_index + 1:02d}.png"
        canvas.save(output, quality=95)
        outputs.append(str(output))

    index = {
        "selection_manifest": str(args.selection_manifest),
        "dataset_root": str(args.dataset_root),
        "row_count": len(rows),
        "pages": outputs,
    }
    (args.output_dir / "contact_sheet_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(index, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
