#!/usr/bin/env python3
"""Select evenly spaced, complete observations from a policy trace."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


CAMERAS = ("cam_left_wrist", "cam_right_wrist", "cam_high")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-csv", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--anchors-per-prompt", type=int, default=16)
    parser.add_argument("--exclude-composed-prompts", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_images = args.output_dir / "images"
    output_images.mkdir()

    first_rows: dict[int, dict[str, str]] = {}
    fieldnames = None
    with args.trace_csv.open(newline="", encoding="utf-8", errors="ignore") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        for row in reader:
            request_id = int(row["request_id"])
            if int(row["action_step"]) == 0 and request_id not in first_rows:
                first_rows[request_id] = row

    groups = defaultdict(list)
    for row in first_rows.values():
        prompt = " ".join(row["prompt"].split())
        if args.exclude_composed_prompts and "; then " in prompt:
            continue
        paths = [args.images_dir / Path(row[f"{camera}_image"]).name for camera in CAMERAS]
        if all(path.is_file() for path in paths):
            groups[prompt].append(row)

    selected = []
    prompt_counts = {}
    for prompt, rows in groups.items():
        rows.sort(key=lambda row: int(row["request_id"]))
        count = min(args.anchors_per_prompt, len(rows))
        indices = np.linspace(0, len(rows) - 1, num=count, dtype=int)
        chosen = [rows[int(index)] for index in sorted(set(indices.tolist()))]
        selected.extend(chosen)
        prompt_counts[prompt] = {"available": len(rows), "selected": len(chosen)}

    selected.sort(key=lambda row: int(row["request_id"]))
    with (args.output_dir / "inference.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    for row in selected:
        for camera in CAMERAS:
            source = args.images_dir / Path(row[f"{camera}_image"]).name
            shutil.copy2(source, output_images / source.name)

    (args.output_dir / "selection.json").write_text(
        json.dumps(
            {
                "source_csv": str(args.trace_csv),
                "anchors_per_prompt": args.anchors_per_prompt,
                "exclude_composed_prompts": args.exclude_composed_prompts,
                "selected": len(selected),
                "prompt_counts": prompt_counts,
                "request_ids": [int(row["request_id"]) for row in selected],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"selected": len(selected), "prompt_counts": prompt_counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
