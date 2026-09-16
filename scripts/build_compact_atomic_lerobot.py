#!/usr/bin/env python3
"""Build a compact, prompt-aligned LeRobot v3 view from an existing v3 dataset.

This intentionally keeps *all* source frames.  It creates one normal ``right``
view at a time; left-mirror video is a separate conversion because it must be
horizontally decoded/flipped/re-encoded rather than falsely reusing right video.

Per frame columns written directly into data parquet:
  annotation.segment_id  int64   (-1 means an uncovered/drop interval)
  annotation.type        int8    (0 drop, 1 atomic)
  annotation.prompt      string
  annotation.atoms       float32[12]
The episode-level global Qwen task is represented by ``task_index`` and the
standard LeRobot ``meta/tasks.parquet`` table.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from atomic_latent_vla.data.atomic_ratios import local_atomic_weight_rows


N_ATOMS = 12
RIGHT_SLICE = slice(8, 15)
LEFT_SLICE = slice(0, 7)
LEFT_TO_RIGHT_SIGNS = np.asarray([1, -1, -1, 1, -1, 1, -1], dtype=np.float32)


def episode_number(name: str) -> int:
    # episode_000123.json -> 123; intentionally ignores mirror suffixes.
    stem = Path(name).stem
    if not stem.startswith("episode_"):
        raise ValueError(f"Unexpected annotation filename: {name}")
    return int(stem.split("_")[1])


def load_annotations(directory: Path) -> dict[int, dict]:
    result: dict[int, dict] = {}
    for path in sorted(directory.glob("episode_*.json")):
        ann = json.loads(path.read_text())
        ep = episode_number(path.name)
        if ep in result:
            raise ValueError(f"Duplicate episode {ep}: {path}")
        result[ep] = ann
    if not result:
        raise FileNotFoundError(f"No annotation JSON under {directory}")
    return result


def atom_targets(segment: dict) -> list[tuple[int, float]]:
    """Read both historical annotation schemas into label/weight pairs."""
    if segment.get("atomic_targets"):
        out = []
        for target in segment["atomic_targets"]:
            out.append(
                (
                    int(target["label"]),
                    float(target.get("confidence", target.get("weight", 1.0))),
                )
            )
        return out
    labels = segment.get("gate_labels", [])
    weights = segment.get("gate_weights", [])
    return [(int(item["label"]), float(weights[i]) if i < len(weights) else 1.0)
            for i, item in enumerate(labels)]


def segment_rows(annotation: dict) -> list[dict]:
    rows = []
    for source_id, segment in enumerate(annotation.get("segments", [])):
        start = float(segment["start_s"])
        end = float(segment["end_s"])
        if end <= start:
            continue
        targets = atom_targets(segment)
        eligible = bool(segment.get("training_eligible", bool(targets))) and bool(targets)
        # cabinet's deterministic gate is authoritative when it is present.
        if "gate_mode" in segment:
            eligible = segment["gate_mode"] in {"single", "dual"} and bool(targets)
        atoms = np.zeros(N_ATOMS, dtype=np.float32)
        if eligible:
            for label, weight in targets:
                if not 0 <= label < N_ATOMS:
                    raise ValueError(f"Invalid atom label {label}")
                atoms[label] = weight
        prompt = str(segment.get("low_level_instruction", "")).strip()
        rows.append({"id": int(segment.get("segment_id", source_id)), "start": start,
                     "end": end, "kind": int(eligible), "atoms": atoms, "prompt": prompt,
                     "atomic_ratio_blocks": segment.get("atomic_ratio_blocks", [])})
    return sorted(rows, key=lambda row: (row["start"], row["end"], row["id"]))


def per_frame_annotation(timestamps: np.ndarray, segments: list[dict]):
    """Assign exactly one segment using the last matching segment on overlaps."""
    n = len(timestamps)
    ids = np.full(n, -1, dtype=np.int64)
    kinds = np.zeros(n, dtype=np.int8)
    prompts = ["No stable atomic motion is annotated for this frame."] * n
    atoms = np.zeros((n, N_ATOMS), dtype=np.float32)
    for segment in segments:
        hit = (timestamps >= segment["start"] - 1e-6) & (timestamps < segment["end"] - 1e-6)
        ids[hit] = segment["id"]
        kinds[hit] = segment["kind"]
        atoms[hit] = local_atomic_weight_rows(
            timestamps[hit],
            {
                "start_s": segment["start"],
                "atomic_ratio_blocks": segment["atomic_ratio_blocks"],
            },
            segment["atoms"],
        )
        for i in np.nonzero(hit)[0]:
            prompts[int(i)] = segment["prompt"] or prompts[int(i)]
    return ids, kinds, prompts, atoms


def fixed_list(values: np.ndarray, width: int, value_type=pa.float32()) -> pa.Array:
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=value_type), width)


def copy_tree_without(source: Path, dest: Path, excluded: set[str]) -> None:
    if not source.exists():
        return
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns(*excluded))


def rewrite_episode_meta(source_meta: Path, output_meta: Path, global_tasks: dict[int, str], arm: str) -> None:
    """Preserve video offsets while replacing each episode task and dropping left metadata."""
    source = next((source_meta / "episodes").rglob("*.parquet"))
    table = pq.ParquetFile(source).read()
    rows = table.to_pylist()
    clean = []
    for row in rows:
        ep = int(row["episode_index"])
        if ep not in global_tasks:
            continue
        if arm == "left_mirror":
            # The hflipped left-wrist movie is written at the conventional right key.
            for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
                old = f"videos/observation.images.left_wrist_0_rgb/{suffix}"
                new = f"videos/observation.images.right_wrist_0_rgb/{suffix}"
                row[new] = row[old]
        row = {k: v for k, v in row.items() if "left_wrist" not in k}
        row["tasks"] = [global_tasks[ep]]
        clean.append(row)
    target = output_meta / "episodes" / "chunk-000" / "file-000.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(clean), target, compression="zstd")


def update_info(output: Path, global_tasks: dict[int, str], total_frames: int) -> None:
    info_path = output / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    features = info["features"]
    features.pop("observation.images.left_wrist_0_rgb", None)
    for key, width, dtype, names in (
        ("observation.state", 7, "float32", [f"joint_{i}" for i in range(7)]),
        ("action", 7, "float32", [f"joint_{i}" for i in range(7)]),
        ("annotation.segment_id", 1, "int64", None),
        ("annotation.type", 1, "int8", None),
        ("annotation.prompt", 1, "string", None),
        ("annotation.atoms", N_ATOMS, "float32", [f"atom_{i}" for i in range(N_ATOMS)]),
    ):
        features[key] = {"dtype": dtype, "shape": [width], "names": names}
    info["total_episodes"] = len(global_tasks)
    info["total_frames"] = int(total_frames)
    info["total_tasks"] = len(set(global_tasks.values()))
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")


def mirror_video_tree(source: Path, output: Path) -> None:
    """Create hflipped base + left-wrist movies as base + right-wrist movies.

    Keeping video key names conventional is important: downstream code sees only
    base/right regardless of whether an episode was physically captured left or
    right. NVENC is used when locally available; libx264 remains the fallback.
    """
    codec = "h264_nvenc" if "h264_nvenc" in subprocess.check_output(
        ["ffmpeg", "-hide_banner", "-encoders"], stderr=subprocess.DEVNULL, text=True
    ) else "libx264"
    mappings = (("observation.images.base_0_rgb", "observation.images.base_0_rgb"),
                ("observation.images.left_wrist_0_rgb", "observation.images.right_wrist_0_rgb"))
    for src_key, dst_key in mappings:
        for src in sorted((source / "videos" / src_key).rglob("*.mp4")):
            relative = src.relative_to(source / "videos" / src_key)
            dst = output / "videos" / dst_key / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            command = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
                       "-vf", "hflip", "-an", "-c:v", codec]
            if codec == "h264_nvenc":
                command += ["-preset", "p4", "-cq", "23"]
            else:
                command += ["-preset", "medium", "-crf", "20"]
            command += ["-pix_fmt", "yuv420p", str(dst)]
            subprocess.run(command, check=True)


def build_existing_v3(source: Path, annotations: Path, output: Path, max_episodes: int | None, arm: str) -> None:
    ann = load_annotations(annotations)
    if max_episodes is not None:
        ann = dict(sorted(ann.items())[:max_episodes])
    # A non-empty output is never silently overwritten.
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    if arm == "right":
        # videos are byte-identical for normal right view; never copy the unused left view.
        copy_tree_without(source / "videos", output / "videos", {"observation.images.left_wrist_0_rgb"})
    else:
        mirror_video_tree(source, output)
    shutil.copytree(source / "meta", output / "meta")
    shutil.rmtree(output / "data", ignore_errors=True)
    (output / "data" / "chunk-000").mkdir(parents=True)

    source_data = next((source / "data").rglob("*.parquet"))
    data = pq.ParquetFile(source_data).read().to_pydict()
    episode_ids = np.asarray(data["episode_index"], dtype=np.int64)
    timestamps = np.asarray(data["timestamp"], dtype=np.float64)
    keep = np.isin(episode_ids, np.asarray(list(ann), dtype=np.int64))
    # Keep source order, while remapping global task strings to standard task_index.
    task_text = {ep: str(a.get("task") or a.get("global_description") or "") for ep, a in ann.items()}
    task_to_id: OrderedDict[str, int] = OrderedDict()
    episode_task_id = {}
    for ep, text in task_text.items():
        episode_task_id[ep] = task_to_id.setdefault(text, len(task_to_id))

    ids = np.full(len(episode_ids), -1, dtype=np.int64)
    kinds = np.zeros(len(episode_ids), dtype=np.int8)
    prompts = ["No stable atomic motion is annotated for this frame."] * len(episode_ids)
    atoms = np.zeros((len(episode_ids), N_ATOMS), dtype=np.float32)
    for ep, annotation in ann.items():
        mask = episode_ids == ep
        e_ids, e_kinds, e_prompts, e_atoms = per_frame_annotation(timestamps[mask], segment_rows(annotation))
        ids[mask], kinds[mask], atoms[mask] = e_ids, e_kinds, e_atoms
        positions = np.flatnonzero(mask)
        for pos, text in zip(positions, e_prompts):
            prompts[int(pos)] = text

    joint_slice = RIGHT_SLICE if arm == "right" else LEFT_SLICE
    state = np.asarray(data["observation.state"], dtype=np.float32)[keep, joint_slice]
    action = np.asarray(data["action"], dtype=np.float32)[keep, joint_slice]
    if arm == "left_mirror":
        state *= LEFT_TO_RIGHT_SIGNS
        action *= LEFT_TO_RIGHT_SIGNS
    cols: dict[str, pa.Array] = {}
    for name, values in data.items():
        if name in {"observation.state", "action", "task_index"}:
            continue
        cols[name] = pa.array(np.asarray(values)[keep])
    cols["observation.state"] = fixed_list(state, 7)
    cols["action"] = fixed_list(action, 7)
    cols["task_index"] = pa.array(np.asarray([episode_task_id[int(ep)] for ep in episode_ids[keep]], dtype=np.int64))
    cols["annotation.segment_id"] = pa.array(ids[keep])
    cols["annotation.type"] = pa.array(kinds[keep])
    cols["annotation.prompt"] = pa.array(np.asarray(prompts, dtype=object)[keep], type=pa.string())
    cols["annotation.atoms"] = fixed_list(atoms[keep], N_ATOMS)
    table = pa.table(cols)
    pq.write_table(table, output / "data" / "chunk-000" / "file-000.parquet", compression="zstd")

    tasks_table = pa.table({"task_index": pa.array(list(task_to_id.values()), type=pa.int64()),
                            "task": pa.array(list(task_to_id.keys()), type=pa.string())})
    pq.write_table(tasks_table, output / "meta" / "tasks.parquet", compression="zstd")
    # Existing global stats have 16-D state/action. Remove them rather than lie.
    (output / "meta" / "stats.json").unlink(missing_ok=True)
    rewrite_episode_meta(source / "meta", output / "meta", task_text, arm)
    update_info(output, task_text, int(keep.sum()))
    summary = {"source": str(source), "annotations": str(annotations), "arm": arm,
               "episodes": len(ann), "frames": int(keep.sum()), "atomic_frames": int(kinds[keep].sum()),
               "fields": ["task", "annotation.segment_id", "annotation.type", "annotation.prompt", "annotation.atoms"]}
    (output / "conversion_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Existing LeRobot v3 source root")
    parser.add_argument("--annotations", type=Path, required=True, help="One view's JSON directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("right", "left_mirror"), default="right")
    parser.add_argument("--max-episodes", type=int, default=None, help="Smoke test only")
    args = parser.parse_args()
    build_existing_v3(args.source, args.annotations, args.output, args.max_episodes, args.arm)


if __name__ == "__main__":
    main()
