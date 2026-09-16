#!/usr/bin/env python3
"""Build a compact target dataset with short and fully-idle episodes removed.

The source dataset is never modified.  Remaining episodes, frame rows, videos,
the active prompt/FK sidecars, TCP poses, and the frozen zT teacher are remapped
to contiguous episode/frame indices in a fresh destination.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ACTIVE_JSONL = (
    "global_episode_prompts.jsonl",
    "episode_subtasks.jsonl",
    "atomic_horizon_prompts_3hz.jsonl",
)
ACTIVE_JSON = (
    "atomic_horizon_prompts_3hz_manifest.json",
    "tcp_pose_bimanual_base_tcp200.json",
)
VIDEO_KEYS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _episode_file(directory: Path, episode_index: int) -> Path:
    return directory / f"episode_{episode_index:06d}.json"


def _complete_gate_modes(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        str(segment.get("gate_mode", ""))
        for segment in payload.get("segments", [])
        if "insufficient future coverage" not in str(segment.get("gate_reason", ""))
    ]


def _choose_episodes(
    source: Path,
    *,
    min_seconds: float,
    gate_sidecar: str,
) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    episodes = _read_jsonl(source / "meta" / "episodes.jsonl")
    fps = float(json.loads((source / "meta" / "info.json").read_text())["fps"])
    short = {
        int(row["episode_index"])
        for row in episodes
        if float(row["length"]) / fps < min_seconds
    }
    gate_root = source / "meta" / gate_sidecar
    stationary: set[int] = set()
    for row in episodes:
        episode_index = int(row["episode_index"])
        modes: list[str] = []
        for directory in ("left", "right_recomputed_0p20m"):
            path = _episode_file(gate_root / directory, episode_index)
            if not path.is_file():
                raise FileNotFoundError(path)
            modes.extend(_complete_gate_modes(path))
        if modes and all(mode == "idle" for mode in modes):
            stationary.add(episode_index)
    return episodes, short, stationary


def _remap_episode_row(row: dict[str, Any], mapping: dict[int, int]) -> dict[str, Any]:
    old = int(row["episode_index"])
    result = dict(row)
    result["episode_index"] = mapping[old]
    if "episode_id" in result:
        result["episode_id"] = f"episode_{mapping[old]:06d}"
    result.setdefault("filtered_from_episode_index", old)
    return result


def _copy_videos(source: Path, destination: Path, mapping: dict[int, int]) -> None:
    for key in VIDEO_KEYS:
        source_dir = source / "videos" / key / "chunk-000"
        destination_dir = destination / "videos" / key / "chunk-000"
        destination_dir.mkdir(parents=True, exist_ok=True)
        for old, new in mapping.items():
            source_path = source_dir / f"file-{old:03d}.mp4"
            destination_path = destination_dir / f"file-{new:03d}.mp4"
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            os.link(source_path, destination_path)


def _rewrite_frame_parquet(
    source: Path,
    destination: Path,
    mapping: dict[int, int],
    task_mapping: dict[int, int],
) -> np.ndarray:
    paths = sorted((source / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(source / "data")
    table = pa.concat_tables([pq.read_table(path) for path in paths], promote_options="default")
    old_episode = np.asarray(table["episode_index"].combine_chunks())
    keep = np.isin(old_episode, np.asarray(list(mapping), dtype=np.int64))
    filtered = table.filter(pa.array(keep))
    old_episode_kept = old_episode[keep]
    new_episode = np.fromiter(
        (mapping[int(value)] for value in old_episode_kept),
        dtype=np.int64,
        count=len(old_episode_kept),
    )
    old_tasks = np.asarray(filtered["task_index"].combine_chunks())
    new_tasks = np.fromiter(
        (task_mapping[int(value)] for value in old_tasks),
        dtype=np.int64,
        count=len(old_tasks),
    )
    for name, values in (
        ("episode_index", new_episode),
        ("task_index", new_tasks),
        ("index", np.arange(len(filtered), dtype=np.int64)),
    ):
        column_index = filtered.schema.get_field_index(name)
        filtered = filtered.set_column(
            column_index,
            name,
            pa.array(values, type=filtered.schema.field(name).type),
        )
    output = destination / "data" / "chunk-000" / "file-000.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(filtered, output, compression="zstd")
    return keep


def _rewrite_tasks(
    source: Path,
    destination: Path,
    used_task_ids: list[int],
) -> dict[int, int]:
    table = pq.read_table(source / "meta" / "tasks.parquet")
    text_key = next(name for name in table.column_names if name != "task_index")
    rows = {int(row["task_index"]): row[text_key] for row in table.to_pylist()}
    missing = set(used_task_ids) - set(rows)
    if missing:
        raise ValueError(f"tasks.parquet is missing task ids: {sorted(missing)}")
    mapping = {old: new for new, old in enumerate(sorted(set(used_task_ids)))}
    output_table = pa.table(
        {
            "task_index": pa.array(range(len(mapping)), type=pa.int64()),
            text_key: pa.array([rows[old] for old in sorted(mapping)], type=pa.string()),
        }
    )
    pq.write_table(output_table, destination / "meta" / "tasks.parquet", compression="zstd")
    return mapping


def _rewrite_active_jsonl(source: Path, destination: Path, mapping: dict[int, int]) -> None:
    for name in ACTIVE_JSONL:
        rows = _read_jsonl(source / "meta" / name)
        output = [
            _remap_episode_row(row, mapping)
            for row in rows
            if int(row["episode_index"]) in mapping
        ]
        _write_jsonl(destination / "meta" / name, output)
    for name in ACTIVE_JSON:
        source_path = source / "meta" / name
        if not source_path.is_file() or name == "tcp_pose_bimanual_base_tcp200.json":
            continue
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        payload["filtered_from_dataset_root"] = str(source)
        payload["dataset_root"] = str(destination)
        (destination / "meta" / name).write_text(json.dumps(payload, indent=2) + "\n")


def _rewrite_gate_sidecar(
    source: Path,
    destination: Path,
    mapping: dict[int, int],
    gate_sidecar: str,
) -> None:
    source_root = source / "meta" / gate_sidecar
    output_root = destination / "meta" / gate_sidecar
    counts: dict[str, Counter[str]] = {}
    for directory in ("left", "right_recomputed_0p20m"):
        output_dir = output_root / directory
        output_dir.mkdir(parents=True, exist_ok=True)
        counter: Counter[str] = Counter()
        for old, new in mapping.items():
            source_path = _episode_file(source_root / directory, old)
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            payload["episode_index"] = new
            if "episode_id" in payload:
                payload["episode_id"] = f"episode_{new:06d}"
            payload.setdefault("filtered_from_episode_index", old)
            for segment in payload.get("segments", []):
                counter[str(segment.get("gate_mode", "unknown"))] += 1
            _episode_file(output_dir, new).write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
        counts[directory] = counter
    old_summary_path = source_root / "summary.json"
    old_summary = json.loads(old_summary_path.read_text()) if old_summary_path.is_file() else {}
    summary = {
        "dataset_root": str(destination),
        "source_dataset_root": str(source),
        "source_sidecar": str(source_root),
        "episode_range": [0, len(mapping)],
        "files_written": 2 * len(mapping),
        "gate_mode_counts": {key: dict(value) for key, value in counts.items()},
        "contract": old_summary.get("contract", {}),
    }
    output_root.joinpath("summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def _filter_npy(source_path: Path, output_path: Path, keep: np.ndarray) -> None:
    source = np.load(source_path, mmap_mode="r")
    if len(source) != len(keep):
        raise ValueError(f"{source_path}: {len(source)} rows != {len(keep)} parquet rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=source.dtype,
        shape=(int(keep.sum()), *source.shape[1:]),
    )
    cursor = 0
    for start in range(0, len(source), 65_536):
        stop = min(start + 65_536, len(source))
        block = np.asarray(source[start:stop])[keep[start:stop]]
        output[cursor : cursor + len(block)] = block
        cursor += len(block)
    output.flush()
    del output
    if cursor != int(keep.sum()):
        raise RuntimeError(f"{output_path}: wrote {cursor} rows")


def _rewrite_dense_sidecars(
    source: Path,
    destination: Path,
    keep: np.ndarray,
) -> None:
    source_meta = source / "meta"
    output_meta = destination / "meta"
    tcp_name = "tcp_pose_bimanual_base_tcp200.npy"
    _filter_npy(source_meta / tcp_name, output_meta / tcp_name, keep)
    tcp_metadata = json.loads((source_meta / "tcp_pose_bimanual_base_tcp200.json").read_text())
    tcp_metadata["rows"] = int(keep.sum())
    tcp_metadata["filtered_from_dataset_root"] = str(source)
    (output_meta / "tcp_pose_bimanual_base_tcp200.json").write_text(
        json.dumps(tcp_metadata, indent=2) + "\n"
    )

    teacher_name = "zt_q_teacher_target_v3_layerwise_shared_flow_zt27k_atomic_v1"
    source_teacher = source_meta / teacher_name
    output_teacher = output_meta / teacher_name
    _filter_npy(source_teacher / "directions.npy", output_teacher / "directions.npy", keep)
    _filter_npy(source_teacher / "valid.npy", output_teacher / "valid.npy", keep)
    shutil.copy2(source_teacher / "codebook.npy", output_teacher / "codebook.npy")
    manifest = json.loads((source_teacher / "manifest.json").read_text())
    valid = np.load(output_teacher / "valid.npy", mmap_mode="r")
    manifest["dataset_root"] = str(destination)
    manifest["dataset_rows"] = int(keep.sum())
    manifest["encoded_rows"] = int(np.count_nonzero(valid))
    manifest["filtered_from_dataset_root"] = str(source)
    (output_teacher / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _rewrite_info(
    source: Path,
    destination: Path,
    *,
    episodes: int,
    frames: int,
    tasks: int,
) -> None:
    info = json.loads((source / "meta" / "info.json").read_text())
    info["total_episodes"] = episodes
    info["total_frames"] = frames
    info["total_tasks"] = tasks
    info["dataset_revision"] = "v3_reviewed_no_short_no_idle_2026-08-19"
    info["filter_policy"] = "duration >= 5 seconds and not fully idle on both arms"
    (destination / "meta" / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--min-seconds", type=float, default=5.0)
    parser.add_argument("--gate-sidecar", default="fk_horizon_3hz_gate_top5_stay_v2")
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)
    if args.min_seconds <= 0:
        raise ValueError("--min-seconds must be positive")

    episodes, short, stationary = _choose_episodes(
        source,
        min_seconds=args.min_seconds,
        gate_sidecar=args.gate_sidecar,
    )
    removed = short | stationary
    kept = [row for row in episodes if int(row["episode_index"]) not in removed]
    mapping = {int(row["episode_index"]): new for new, row in enumerate(kept)}
    if not mapping or not removed:
        raise ValueError(f"unexpected selection: kept={len(mapping)} removed={len(removed)}")

    destination.mkdir(parents=True)
    (destination / "meta").mkdir()
    used_tasks = [int(row["task_index"]) for row in kept]
    task_mapping = _rewrite_tasks(source, destination, used_tasks)
    keep_rows = _rewrite_frame_parquet(source, destination, mapping, task_mapping)
    _copy_videos(source, destination, mapping)

    episode_rows = []
    for row in kept:
        result = _remap_episode_row(row, mapping)
        result["task_index"] = task_mapping[int(row["task_index"])]
        episode_rows.append(result)
    _write_jsonl(destination / "meta" / "episodes.jsonl", episode_rows)
    _rewrite_active_jsonl(source, destination, mapping)
    _rewrite_gate_sidecar(source, destination, mapping, args.gate_sidecar)
    _rewrite_dense_sidecars(source, destination, keep_rows)
    _rewrite_info(
        source,
        destination,
        episodes=len(kept),
        frames=int(keep_rows.sum()),
        tasks=len(task_mapping),
    )

    manifest = {
        "schema": "atomic_latent_vla.filtered_target.v1",
        "source_dataset_root": str(source),
        "destination_dataset_root": str(destination),
        "minimum_duration_seconds": args.min_seconds,
        "fully_stationary_definition": (
            "every complete 50-frame FK gate block on both arms has gate_mode=idle; "
            "incomplete tail blocks are ignored"
        ),
        "removed_short_episode_indices": sorted(short),
        "removed_stationary_episode_indices": sorted(stationary),
        "removed_union_episode_indices": sorted(removed),
        "source_episodes": len(episodes),
        "destination_episodes": len(kept),
        "source_frames": int(sum(int(row["length"]) for row in episodes)),
        "destination_frames": int(keep_rows.sum()),
        "episode_index_mapping": {str(old): new for old, new in mapping.items()},
        "video_storage": "hard links to immutable source video bytes, with remapped filenames",
        "active_gate_sidecar": args.gate_sidecar,
        "teacher_norm_asset_id": "openpi_norm_compact_accepted_v3",
    }
    (destination / "FILTER_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "episode_index_mapping"}, indent=2))


if __name__ == "__main__":
    main()
