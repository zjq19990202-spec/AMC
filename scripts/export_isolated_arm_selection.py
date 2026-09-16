#!/usr/bin/env python3
"""Export the 250-cluster/500-frame steering selection for human review."""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from atomic_latent_vla.pi05 import AtomicPi05Config
from atomic_latent_vla.pi05.training_data import build_atomic_dataset


CAMERAS = (
    ("base", "base_0_rgb"),
    ("left wrist", "left_wrist_0_rgb"),
    ("right wrist", "right_wrist_0_rgb"),
)


def _to_pil(value: np.ndarray) -> Image.Image:
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"expected image rank 3, got {array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        if finite.size and finite.min() >= -1.1 and finite.max() <= 1.1:
            array = (array + 1.0) * 127.5 if finite.min() < 0.0 else array * 255.0
    array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).convert("RGB")


def _render(sample: dict, anchor: dict, output: Path) -> None:
    images = sample["image"]
    width, height = 320, 240
    header = 34
    text_height = 190
    canvas = Image.new("RGB", (width * len(CAMERAS), header + height + text_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (label, key) in enumerate(CAMERAS):
        image = _to_pil(images[key]).resize((width, height), Image.Resampling.BILINEAR)
        x = index * width
        canvas.paste(image, (x, header))
        draw.text((x + 8, 10), label, fill="black", font=font)
        if anchor["arm"] in label:
            draw.rectangle((x + 2, header + 2, x + width - 3, header + height - 3), outline="#d62728", width=4)
    lines = [
        f"cluster={anchor['cluster_key']} pair={anchor['pair_index']}  episode={anchor['episode']} frame={anchor['frame']}",
        f"tested_arm={anchor['arm']} kind={anchor['kind']} other_arm={anchor['other_status']}",
        "atoms=" + " + ".join(anchor["atoms"]),
        "atomic: " + anchor["atomic_prompt"],
        "reverse: " + anchor["reverse_prompt"],
        "subtask: " + anchor["subtask_prompt"],
    ]
    y = header + height + 8
    for line in lines:
        for wrapped in textwrap.wrap(line, width=145) or [""]:
            draw.text((8, y), wrapped, fill="black", font=font)
            y += 16
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    anchors = payload["anchors"]
    if len(anchors) != 500:
        raise ValueError(f"expected exactly 500 anchors, got {len(anchors)}")
    config = AtomicPi05Config(max_token_len=250, fast_action_ce_loss_weight=0.0)
    dataset = build_atomic_dataset(
        (args.dataset_root,),
        norm_assets_dir="/mnt/cunchu/zjq/target",
        norm_asset_id="openpi_norm_compact_accepted_v3",
        action_horizon=config.action_horizon,
        max_token_len=config.max_token_len,
        include_fast=False,
    )
    samples_dir = args.output_dir / "samples"
    rows = []
    for number, anchor in enumerate(anchors, 1):
        sample = dataset[int(anchor["dataset_index"])]
        atom_slug = "+".join(anchor["atoms"])
        relative = Path(anchor["cluster_key"]) / (
            f"pair{anchor['pair_index']}_ep{anchor['episode']:06d}_"
            f"f{anchor['frame']:06d}_{atom_slug}.jpg"
        )
        _render(sample, anchor, samples_dir / relative)
        rows.append(
            {
                "number": number,
                "cluster_key": anchor["cluster_key"],
                "pair_index": anchor["pair_index"],
                "episode": anchor["episode"],
                "frame": anchor["frame"],
                "arm": anchor["arm"],
                "kind": anchor["kind"],
                "other_status": anchor["other_status"],
                "atoms": "+".join(anchor["atoms"]),
                "atomic_prompt": anchor["atomic_prompt"],
                "reverse_prompt": anchor["reverse_prompt"],
                "subtask_prompt": anchor["subtask_prompt"],
                "image": str(Path("samples") / relative),
            }
        )
        if number % 25 == 0:
            print(f"exported {number}/{len(anchors)}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "README.txt").write_text(
        "250 eligible state clusters; two different atomic prompts/frames per cluster.\n"
        "Every frame has one moving single/dual arm and an opposite stay/unlabeled arm.\n"
        "The tested wrist image is outlined in red.\n",
        encoding="utf-8",
    )
    print(json.dumps({"samples": len(rows), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
