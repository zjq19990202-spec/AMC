#!/usr/bin/env python3
"""Create a full-shape CR1 left/right mirror from a completed LeRobot dataset.

The source must retain three camera streams and 16-D vectors ordered as
``[left_7, left_gripper, right_7, right_gripper]``.  The output is deliberately
unannotated at first; run ``inject_atomic_annotations_lerobot.py`` with the
left_mirror JSON directory afterwards so its text/atom labels match the mirrored
right arm rather than the original right arm.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ARM_SIGN_8 = np.asarray([1, -1, -1, 1, -1, 1, -1, 1], dtype=np.float32)


def fixed(values: np.ndarray, width: int) -> pa.Array:
    return pa.FixedSizeListArray.from_arrays(
        pa.array(np.asarray(values, dtype=np.float32).reshape(-1), type=pa.float32()), width
    )


def mirror_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 16:
        raise ValueError(f"Expected [N,16] bimanual vector, got {values.shape}")
    result = np.empty_like(values)
    result[:, :8] = values[:, 8:] * ARM_SIGN_8
    result[:, 8:] = values[:, :8] * ARM_SIGN_8
    return result


def one_parquet(root: Path) -> Path:
    paths = sorted(root.rglob("*.parquet"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected exactly one parquet below {root}, found {paths}")
    return paths[0]


def codec() -> str:
    encoders = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], stderr=subprocess.DEVNULL, text=True)
    return "h264_nvenc" if "h264_nvenc" in encoders else "libx264"


def flip_and_swap_videos(source: Path, output: Path) -> int:
    pairs = (
        ("observation.images.base_0_rgb", "observation.images.base_0_rgb"),
        ("observation.images.left_wrist_0_rgb", "observation.images.right_wrist_0_rgb"),
        ("observation.images.right_wrist_0_rgb", "observation.images.left_wrist_0_rgb"),
    )
    encoder = codec(); count = 0
    for source_key, target_key in pairs:
        root = source / "videos" / source_key
        if not root.exists():
            raise FileNotFoundError(root)
        for video in sorted(root.rglob("*.mp4")):
            destination = output / "videos" / target_key / video.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(video), "-vf", "hflip", "-an", "-c:v", encoder]
            cmd += ["-preset", "p4", "-cq", "23"] if encoder == "h264_nvenc" else ["-preset", "medium", "-crf", "20"]
            subprocess.run(cmd + ["-pix_fmt", "yuv420p", str(destination)], check=True)
            count += 1
    return count


def main(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    source_info = json.loads((source / "meta" / "info.json").read_text())
    required = {"observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb", "observation.images.right_wrist_0_rgb"}
    missing = required - set(source_info["features"])
    if missing: raise ValueError(f"Source lacks required original cameras: {sorted(missing)}")
    output.mkdir(parents=True)
    shutil.copytree(source / "meta", output / "meta")
    shutil.copytree(source / "data", output / "data")
    if (source / "source_episode_map.json").exists():
        shutil.copy2(source / "source_episode_map.json", output / "source_episode_map.json")
    data_path = one_parquet(output / "data")
    table = pq.read_table(data_path)
    cols = {name: table.column(name) for name in table.column_names if name not in {"observation.state", "action"}}
    cols["observation.state"] = fixed(mirror_vector(np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)), 16)
    cols["action"] = fixed(mirror_vector(np.asarray(table.column("action").to_pylist(), dtype=np.float32)), 16)
    replacement = data_path.with_suffix(".mirroring")
    pq.write_table(pa.table(cols), replacement, compression="zstd")
    replacement.replace(data_path)
    episodes_path = one_parquet(output / "meta" / "episodes")
    rows = pq.read_table(episodes_path).to_pylist()
    for row in rows:
        for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
            left = f"videos/observation.images.left_wrist_0_rgb/{suffix}"
            right = f"videos/observation.images.right_wrist_0_rgb/{suffix}"
            if left in row and right in row: row[left], row[right] = row[right], row[left]
    replacement = episodes_path.with_suffix(".mirroring")
    pq.write_table(pa.Table.from_pylist(rows), replacement, compression="zstd")
    replacement.replace(episodes_path)
    videos = flip_and_swap_videos(source, output)
    (output / "mirror_conversion_summary.json").write_text(json.dumps({
        "source": str(source), "output": str(output), "vectors": "[L7,Lg,R7,Rg] -> [S*R7,Rg,S*L7,Lg]",
        "joint_sign": ARM_SIGN_8.tolist(), "videos": videos, "cameras": 3, "state_action_dim": 16,
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"Mirrored {videos} videos into {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--source", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); main(args.source, args.output)
