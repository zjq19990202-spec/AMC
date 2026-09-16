#!/usr/bin/env python3
"""Split a native LeRobot-v3 dataset by exact episode-level global prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def tables(paths) -> pa.Table:
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError("no parquet files")
    return pa.concat_tables([pq.read_table(path) for path in paths], promote_options="default")


def replace(table: pa.Table, name: str, values) -> pa.Table:
    index = table.schema.get_field_index(name)
    return table.set_column(index, name, pa.array(values, type=table.schema.field(name).type))


def filter_npy(source: Path, destination: Path, keep: np.ndarray) -> None:
    array = np.load(source, mmap_mode="r")
    if len(array) != len(keep):
        raise ValueError(f"{source}: {len(array)} rows != {len(keep)} frame rows")
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        destination, mode="w+", dtype=array.dtype, shape=(int(keep.sum()), *array.shape[1:])
    )
    cursor = 0
    for start in range(0, len(array), 65536):
        stop = min(start + 65536, len(array))
        block = np.asarray(array[start:stop])[keep[start:stop]]
        output[cursor : cursor + len(block)] = block
        cursor += len(block)
    output.flush()


def remap_episode_json(row: dict, mapping: dict[int, int]) -> dict:
    result = dict(row)
    old = int(result["episode_index"])
    result["episode_index"] = mapping[old]
    if "episode_id" in result:
        result["episode_id"] = f"episode_{mapping[old]:06d}"
    result.setdefault("split_from_episode_index", old)
    return result


def build_split(source: Path, destination: Path, selected: list[int], prompt: str) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    mapping = {old: new for new, old in enumerate(sorted(selected))}
    destination.mkdir(parents=True)
    (destination / "meta").mkdir()

    data = tables((source / "data").glob("chunk-*/*.parquet"))
    old_episode = np.asarray(data["episode_index"].combine_chunks(), dtype=np.int64)
    keep = np.isin(old_episode, np.asarray(selected, dtype=np.int64))
    data = data.filter(pa.array(keep))
    data = replace(data, "episode_index", [mapping[int(x)] for x in old_episode[keep]])
    data = replace(data, "task_index", np.zeros(len(data), dtype=np.int64))
    data = replace(data, "index", np.arange(len(data), dtype=np.int64))
    data_path = destination / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True)
    pq.write_table(data, data_path, compression="zstd")

    episode_table = tables((source / "meta" / "episodes").glob("chunk-*/*.parquet"))
    episode_rows = {int(row["episode_index"]): row for row in episode_table.to_pylist()}
    output_rows = []
    cursor = 0
    video_keys = [
        key.removeprefix("videos/").removesuffix("/file_index")
        for key in episode_table.column_names
        if key.startswith("videos/") and key.endswith("/file_index")
    ]
    for old, new in mapping.items():
        row = dict(episode_rows[old])
        length = int(row["length"])
        row["episode_index"] = new
        row["tasks"] = [prompt]
        row["dataset_from_index"] = cursor
        row["dataset_to_index"] = cursor + length
        row["data/chunk_index"] = 0
        row["data/file_index"] = 0
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0
        for key in video_keys:
            row[f"videos/{key}/chunk_index"] = 0
            row[f"videos/{key}/file_index"] = new
            src = source / "videos" / key / "chunk-000" / f"file-{old:03d}.mp4"
            dst = destination / "videos" / key / "chunk-000" / f"file-{new:03d}.mp4"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        output_rows.append(row)
        cursor += length
    episodes_path = destination / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(output_rows, schema=episode_table.schema), episodes_path, compression="zstd")

    task_table = pq.read_table(source / "meta" / "tasks.parquet")
    text_key = next(key for key in task_table.column_names if key != "task_index")
    pq.write_table(pa.table({"task_index": pa.array([0], type=pa.int64()), text_key: [prompt]}), destination / "meta" / "tasks.parquet")

    for name in ("global_episode_prompts.jsonl", "episode_subtasks.jsonl", "atomic_horizon_prompts_3hz.jsonl"):
        rows = [remap_episode_json(row, mapping) for row in read_jsonl(source / "meta" / name) if int(row["episode_index"]) in mapping]
        write_jsonl(destination / "meta" / name, rows)

    gate_name = "fk_horizon_3hz_gate_top5_stay_v2"
    gate_source = source / "meta" / gate_name
    gate_destination = destination / "meta" / gate_name
    for arm_dir in ("left", "right"):
        for old, new in mapping.items():
            src = gate_source / arm_dir / f"episode_{old:06d}.json"
            payload = json.loads(src.read_text(encoding="utf-8"))
            payload["episode_index"] = new
            payload.setdefault("split_from_episode_index", old)
            dst = gate_destination / arm_dir / f"episode_{new:06d}.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    for name in ("summary.json",):
        src = gate_source / name
        if src.is_file():
            payload = json.loads(src.read_text())
            payload["dataset_root"] = str(destination)
            payload["split_from_dataset_root"] = str(source)
            (gate_destination / name).write_text(json.dumps(payload, indent=2) + "\n")

    tcp_name = "tcp_pose_bimanual_base_tcp200.npy"
    filter_npy(source / "meta" / tcp_name, destination / "meta" / tcp_name, keep)
    tcp_json = json.loads((source / "meta" / "tcp_pose_bimanual_base_tcp200.json").read_text())
    tcp_json["rows"] = int(keep.sum())
    tcp_json["dataset_root"] = str(destination)
    tcp_json["split_from_dataset_root"] = str(source)
    (destination / "meta" / "tcp_pose_bimanual_base_tcp200.json").write_text(json.dumps(tcp_json, indent=2) + "\n")

    info = json.loads((source / "meta" / "info.json").read_text())
    info.update(total_episodes=len(mapping), total_frames=int(keep.sum()), total_tasks=1, splits={"train": f"0:{len(mapping)}"})
    info["split_from_dataset_root"] = str(source)
    info["split_global_prompt"] = prompt
    (destination / "meta" / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    shutil.copy2(source / "meta" / "stats.json", destination / "meta" / "stats.json")

    manifest = {
        "schema": "atomic_latent_vla.prompt_split.v1",
        "source_dataset_root": str(source),
        "destination_dataset_root": str(destination),
        "selection_global_prompt": prompt,
        "source_episode_indices": sorted(selected),
        "episode_index_mapping": {str(k): v for k, v in mapping.items()},
        "episodes": len(mapping),
        "frames": int(keep.sum()),
        "video_storage": "physical copies",
    }
    (destination / "SPLIT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {k: v for k, v in manifest.items() if k not in {"source_episode_indices", "episode_index_mapping"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    prompts = {"plug": "insert the plug into the socket.", "vase": "wipe the vase."}
    global_rows = read_jsonl(args.source / "meta" / "global_episode_prompts.jsonl")
    by_prompt = {prompt: [] for prompt in prompts.values()}
    for row in global_rows:
        prompt = str(row["global_prompt"])
        if prompt not in by_prompt:
            raise ValueError(f"unexpected global prompt: {prompt!r}")
        by_prompt[prompt].append(int(row["episode_index"]))
    if sum(map(len, by_prompt.values())) != len(global_rows):
        raise RuntimeError("episode selection is not exhaustive")
    results = [build_split(args.source, args.output_root / name, by_prompt[prompt], prompt) for name, prompt in prompts.items()]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
