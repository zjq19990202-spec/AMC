#!/usr/bin/env python3
"""Call the validated HDF->LeRobot converter without dropping control-switch frames.

The Qwen annotations use source 30 Hz seconds; retaining every source index is
therefore required before prompt columns can be injected exactly by timestamp.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

UPSTREAM = Path("/home/admin123/ckrc/checkrecord-dataset-ops-bundle/scripts/hdf52lerobot.py")


def load_upstream():
    spec = importlib.util.spec_from_file_location("hdf52lerobot_upstream", UPSTREAM)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {UPSTREAM}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules on Python 3.12.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Do not discard any 30Hz source frames: timestamps then remain source_t/FPS.
    module.build_valid_indices = lambda ctrl: np.arange(ctrl.size, dtype=np.int64)
    # Local LeRobot v3 has no ``vcodec`` argument on create(), while the
    # checked-in converter supports newer releases. Video encoding is still
    # controlled by its patched ffmpeg helper, so only remove the unsupported
    # constructor kwarg.
    original_create = module.LeRobotDataset.create

    def create_compat(**kwargs):
        kwargs.pop("vcodec", None)
        return original_create(**kwargs)

    module.LeRobotDataset.create = staticmethod(create_compat)
    return module


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--task", default="")
    p.add_argument("--vcodec", default="h264_nvenc")
    p.add_argument("--max-files", type=int, default=None, help="smoke-test limit only")
    p.add_argument("--start-file", type=int, default=0, help="inclusive sorted source-file offset")
    p.add_argument("--stop-file", type=int, default=None, help="exclusive sorted source-file offset")
    a = p.parse_args()
    m = load_upstream()
    if a.max_files is not None or a.start_file or a.stop_file is not None:
        original_find = m.find_source_episode_paths
        def selected_files(data_dir):
            paths = original_find(data_dir)[a.start_file:a.stop_file]
            return paths[:a.max_files] if a.max_files is not None else paths
        m.find_source_episode_paths = selected_files
    root = m.main(data_dirs=[a.data_dir], output_dir=a.output_dir, repo_id=a.repo_id,
           task_description=a.task, camera_keys=["cam_d435", "cam_d405_left", "cam_d405_right"],
           vcodec=a.vcodec, crf=23, one_video_per_episode=True,
           image_writer_threads=0, image_writer_processes=0)
    # Conversion preserves sorted source order and every source frame. Store the
    # exact filename mapping so the prompt-injection pass never guesses episode IDs.
    paths = m.find_source_episode_paths(a.data_dir)
    Path(root, "source_episode_map.json").write_text(
        json.dumps({str(i): path.name for i, path in enumerate(paths)}, ensure_ascii=False, indent=2) + "\n"
    )
